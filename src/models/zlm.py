import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from omegaconf import DictConfig

from transformers.activations import ACT2FN

from utils.torch_utils import (
    safe_copy_state,
    unsqueeze_to_batch, expand_to_batch,
    unsqueeze_to_channel, expand_to_channel,
    shift,
    scale_gradient,
    gaussian_init,
)

from models.llama import LlamaForCausalLM
from models.custom_llama import LlamaMLP, LlamaRMSNorm, LlamaDecoderLayer, CustomLlamaModel
from models import load_checkpoint_state
from utils.torch_modules import ContinuousEmbedding
import utils.constants as constants
from utils.logging_utils import print_sharding_info


class DiffusionScheduler(nn.Module):

    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__()

        self.min_timestep = config.minimum_diffusion_timestep
        self.num_timesteps = config.num_diffusion_timesteps

        timesteps = torch.linspace(
            self.min_timestep,
            1.0,
            self.num_timesteps,
        )
        self.register_buffer(
            "timesteps", timesteps, persistent=True
        )

        a = torch.sqrt(1.0 - timesteps)
        self.register_buffer(
            "a", a, persistent=True
        )

        b = torch.sqrt(1 - a.pow(2))
        self.register_buffer(
            "b", b, persistent=True
        )


    def snr(
        self,
        timestep: torch.LongTensor,
    ) -> torch.FloatTensor:
        # https://arxiv.org/pdf/2107.00630 equation 2
        return self.a[timestep].pow(2) / self.b[timestep].pow(2)


    def alpha_t_s(
        self,
        timestep: torch.LongTensor,
    ) -> torch.FloatTensor:
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)
        # https://arxiv.org/pdf/2107.00630 equation 21
        return self.a[timestep] / self.a[timestep - 1]

    
    def var_t_s(
        self,
        timestep: torch.LongTensor,
    ) -> torch.FloatTensor:
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)
        # https://arxiv.org/pdf/2107.00630 equation 22
        return self.b[timestep].pow(2) - (
            self.alpha_t_s(timestep).pow(2) *
            self.b[timestep - 1].pow(2)
        )

    
    def kl(
        self,
        x_0: torch.FloatTensor,
        timestep: torch.LongTensor,
        pred_x_0: torch.FloatTensor,
        dim: int=None,
        keepdim: bool=False,
    ) -> torch.FloatTensor:
        """
        https://arxiv.org/pdf/2107.00630 equation 48

        KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t)) = 
        0.5 * (SNR(t-1) - SNR(t)) * || pred_x_0 - x_0 ||^2
        """
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)
        timestep = unsqueeze_to_channel(timestep, x_0)

        snr_t = self.snr(timestep)
        snr_t_minus_1 = self.snr(timestep - 1)

        kl = 0.5 * (snr_t_minus_1 - snr_t) * F.mse_loss(
            pred_x_0, x_0, reduction="none"
        )

        if dim is not None:
            return kl.sum(dim=dim, keepdim=keepdim)
        return kl


    def add_noise(
        self,
        x_0: torch.FloatTensor,
        timestep: torch.LongTensor,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # https://arxiv.org/pdf/2107.00630 equation 1
        return (
            unsqueeze_to_channel(self.a[timestep], x_0) * x_0 +
            unsqueeze_to_channel(self.b[timestep], noise) * noise
        )


    def step(
        self,
        x_t: torch.FloatTensor,
        timestep: torch.LongTensor,
        pred_x_0: torch.FloatTensor,
        noise: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)

        if noise is None:
            noise = torch.randn_like(x_t)

        timestep = unsqueeze_to_channel(timestep, x_t)
        s = timestep - 1
        
        # https://arxiv.org/pdf/2107.00630 equation 26
        mu = (
            (self.alpha_t_s(timestep) * self.b[s].pow(2) / self.b[timestep].pow(2)) * x_t +
            (self.a[s] * self.var_t_s(timestep) / self.b[timestep].pow(2)) * pred_x_0
        )

        # https://arxiv.org/pdf/2107.00630 equation 25
        var = self.var_t_s(timestep) * self.b[s].pow(2) / self.b[timestep].pow(2)

        return mu + torch.sqrt(var) * noise

    
    def ddim_step(
        self,
        x_t: torch.FloatTensor,
        timestep: torch.LongTensor,
        pred_x_0: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)

        timestep = unsqueeze_to_channel(timestep, x_t)
        
        implied_noise = (
            (x_t - unsqueeze_to_channel(self.a[timestep], pred_x_0) * pred_x_0) /
            unsqueeze_to_channel(self.b[timestep], x_t)
        )

        return self.add_noise(
            pred_x_0,
            timestep - 1,
            implied_noise
        )


class DiffusionHead(nn.Module):

    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__()

        # modules
        self.t_embed = ContinuousEmbedding(
            num_frequencies=config.num_t_embed_frequencies,
            embedding_dim=config.t_mlp_size,
            input_min=config.minimum_diffusion_timestep,
            input_max=1.0,
            bias=True,
        )
        self.t_act = ACT2FN[config.hidden_act]
        self.t_mlp_proj = nn.Linear(config.t_mlp_size, config.hidden_size, bias=False)

        self.x_in_proj = nn.Linear(config.latent_size, config.hidden_size, bias=False)

        self.hidden_states_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_states_in_proj = (
            nn.Linear(config.hidden_size, config.hidden_size, bias=False) if config.diffusion_in_proj else nn.Identity()
        )

        self.layer_norms = nn.ModuleList(
            [LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(config.num_diffusion_head_layers)]
        )
        self.layers = nn.ModuleList([
            LlamaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.diffusion_mlp_size,
                hidden_act=config.hidden_act,
            )
            for _ in range(config.num_diffusion_head_layers)
        ])

        self.out_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(config.hidden_size, config.latent_size, bias=False)

    
    def forward(
        self,
        x_t: torch.FloatTensor,
        timestep: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        scheduler: DiffusionScheduler,
    ) -> torch.FloatTensor:

        t = scheduler.timesteps[timestep]

        # process the hidden inputs
        hidden_states = (
            self.t_mlp_proj(self.t_act(self.t_embed(t))) +
            self.hidden_states_in_proj(self.hidden_states_norm(hidden_states)) +
            self.x_in_proj(x_t)
        )

        for layer_norm, layer in zip(self.layer_norms, self.layers):
            hidden_states = hidden_states + layer(layer_norm(hidden_states))

        hidden_states = self.out_norm(hidden_states)
        pred = self.out_proj(hidden_states)

        return pred


class EncoderModelLayer(LlamaDecoderLayer):
    offload_name = "encoder_model_input"
class EncoderModel(CustomLlamaModel):
    layer_type = EncoderModelLayer

class DecoderModelLayer(LlamaDecoderLayer):
    offload_name = "decoder_model_input"
class DecoderModel(CustomLlamaModel):
    layer_type = DecoderModelLayer


class ZLMModel(nn.Module):
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # save config
        self.hidden_size = config.hidden_size
        self.input_length = config.input_length
        self.output_length = config.output_length
        self.z_length = config.z_length
        self.latent_size = config.latent_size

        # craete the transformer backbones
        self.encoder_model = EncoderModel(config)
        self.decoder_model = DecoderModel(config)

        # create the LM head
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Load the pretrained Llama model
        if config.pretrained_llama is not None:
            
            llama = LlamaForCausalLM(config)
            load_checkpoint_state(
                llama,
                config.pretrained_llama,
                step=0,
                strict=False,
            )

            safe_copy_state(llama.model, self.encoder_model, strict=False)
            safe_copy_state(llama.model, self.decoder_model, strict=False)

            safe_copy_state(llama.lm_head, self.lm_head)
        
            if "TinyLlama" in config.pretrained_llama:
                # For some reason TinyLlama seems to have a broken start token embedding (?)
                self.encoder_model.embed_tokens.weight.data[1] = self.encoder_model.embed_tokens.weight.data[2].clone().detach()
                self.decoder_model.embed_tokens.weight.data[1] = self.decoder_model.embed_tokens.weight.data[2].clone().detach()

                # also fix the unk token
                self.encoder_model.embed_tokens.weight.data[0] = self.encoder_model.embed_tokens.weight.data[2].clone().detach()
                self.decoder_model.embed_tokens.weight.data[0] = self.decoder_model.embed_tokens.weight.data[2].clone().detach()

        # handle pretrained norms
        self.encoder_model.norm.weight.data.fill_(1.0)
        self.decoder_model.skip_norm = True

        # remove the embeddings from the transformers
        self.embed_tokens = self.encoder_model.embed_tokens
        self.encoder_model.embed_tokens = None
        self.decoder_model.embed_tokens = None

        # calculate embedding stats TODO: what if llama_pretrained is None?
        embed_std = self.embed_tokens.weight.data.std(0).detach()
        embed_mean = self.embed_tokens.weight.data.mean(0).detach()
        def embed_prep(x): 
            return embed_mean[None] + x * embed_std[None]

        # create encoder special tokens
        self.encoder_sep_token = nn.Parameter(
            embed_prep(torch.randn(1, self.hidden_size))
        )
        self.encoder_z_tokens = nn.Parameter(
            embed_prep(torch.randn(self.z_length, self.hidden_size))
        )

        # create decoder special tokens
        self.decoder_z_tokens = nn.Parameter(
            embed_prep(torch.randn(1 + self.z_length, self.hidden_size))
        )
        self.decoder_start_output_token = nn.Parameter(
            embed_prep(torch.randn(1, self.hidden_size))
        )

        # create the encoder embeddings
        self.encoder_input_embeddings = nn.Parameter(
            torch.zeros(self.hidden_size)
        )
        self.encoder_output_embeddings = nn.Parameter(
            torch.zeros(self.hidden_size)
        )

        # create the decoder embeddings
        self.decoder_input_embeddings = nn.Parameter(
            torch.zeros(self.hidden_size)
        )
        self.decoder_output_embeddings = nn.Parameter(
            torch.zeros(self.hidden_size)
        )

        # create the input linears
        self.encoder_noise_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.decoder_z_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)

        # scale input layers by embedding stats
        self.encoder_noise_proj_in.weight.data *= embed_std[:, None]
        # self.encoder_noise_proj_in.weight.data.zero_()
        self.decoder_z_proj_in.weight.data *= embed_std[:, None]

        # create the output linear
        self.encoder_mu_proj_out = nn.Linear(self.hidden_size, self.latent_size, bias=False)

        # create the diffusion components
        self.diffusion_head = DiffusionHead(config)
        self.scheduler = DiffusionScheduler(config)

        # unconditional diffusion modules
        self.uncond_diffusion_head = DiffusionHead(config)
        self.uncond_tokens = nn.Parameter(
            torch.randn(self.z_length, self.hidden_size)
        )

        if config.pretrained_llama is None:
            self.apply(gaussian_init)

        else:
            self.diffusion_head.apply(gaussian_init)
            self.uncond_diffusion_head.apply(gaussian_init)
            gaussian_init(self.encoder_noise_proj_in)
            gaussian_init(self.decoder_z_proj_in)
            gaussian_init(self.encoder_mu_proj_out)

    
    def sample_noise(
        self, 
        input_ids: torch.LongTensor,
        noise_scale: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        input_tokens = self.embed_tokens(input_ids)
        
        # generate the noise
        noise = torch.randn(
            *input_ids.shape[:-1],
            self.z_length,
            self.latent_size,
            device=input_tokens.device,
            dtype=input_tokens.dtype,
        )

        if noise_scale is not None:
            noise = noise * noise_scale

        return noise


    def encode(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        noise: torch.FloatTensor=None,
        input_mask: torch.BoolTensor=None,
        output_mask: torch.BoolTensor=None,
        noise_scale: torch.FloatTensor=None,
    ):

        if noise is None:
            noise = self.sample_noise(
                input_ids,
                noise_scale=noise_scale,
            ) 

        input_tokens = self.embed_tokens(input_ids) + unsqueeze_to_batch(
            self.encoder_input_embeddings, input_ids
        )
        output_tokens = self.embed_tokens(output_ids) + unsqueeze_to_batch(
            self.encoder_output_embeddings, output_ids
        )

        z_tokens = (
            unsqueeze_to_batch(self.encoder_z_tokens, noise) +
            shift(
                self.encoder_noise_proj_in(noise),
                n=1, dim=-2, direction="right", narrow=True
            )
        )

        tokens = torch.cat(
            [
                input_tokens,
                expand_to_batch(self.encoder_sep_token, input_tokens),
                output_tokens,
                z_tokens
            ],
            dim=-2
        )
        print_sharding_info(tokens, name="encoder input tokens")

        mask = None
        if input_mask is not None or output_mask is not None:
            assert input_mask is not None and output_mask is not None

            mask = torch.cat(
                [
                    input_mask,
                    torch.ones_like(input_ids[:, :1], dtype=torch.bool),
                    output_mask,
                    torch.ones(input_ids.shape[0], self.z_length, dtype=torch.bool, device=input_ids.device),
                ],
                dim=-1
            )

        hidden_states = self.encoder_model(
            inputs_embeds=tokens,
            elementwise_pad_mask=mask,
        )
        print_sharding_info(hidden_states, name="encoder output hidden states")

        mu = self.encoder_mu_proj_out(
            hidden_states[..., -self.z_length:, :]
        )
        print_sharding_info(mu, name="mu before rms norm")

        # TODO: leave, remove, or scale?
        mu = F.rms_norm(mu, [mu.shape[-1]], eps=self.config.rms_norm_eps)

        z = self.scheduler.add_noise(
            mu, torch.zeros(1, dtype=torch.long, device=mu.device), noise
        )

        return z, mu


    def decode(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        z: torch.FloatTensor,
        logit_grad_scale: float = None,
        input_mask: torch.BoolTensor=None,
        output_mask: torch.BoolTensor=None,
    ):
        z = F.rms_norm(z, [z.shape[-1]], eps=self.config.rms_norm_eps)

        input_tokens = self.embed_tokens(input_ids) + unsqueeze_to_batch(
            self.decoder_input_embeddings, input_ids
        )
        output_tokens = self.embed_tokens(output_ids) + unsqueeze_to_batch(
            self.decoder_output_embeddings, output_ids
        )

        z_tokens = (
            unsqueeze_to_batch(self.decoder_z_tokens, z) +
            shift(
                self.decoder_z_proj_in(z),
                n=1, dim=-2, direction="right", narrow=False
            )
        )
        start_output_token = expand_to_batch(
            self.decoder_start_output_token, output_tokens
        )

        tokens = torch.cat(
            [input_tokens, z_tokens, start_output_token, output_tokens], dim=-2
        )

        mask = None
        if input_mask is not None or output_mask is not None:
            assert input_mask is not None and output_mask is not None

            mask = torch.cat(
                [
                    input_mask,
                    torch.ones(input_ids.shape[0], self.z_length + 2, dtype=torch.bool, device=input_ids.device),
                    output_mask,
                ],
                dim=-1
            )

        hidden_states = self.decoder_model(
            inputs_embeds=tokens,
            elementwise_pad_mask=mask,
        )

        logit_states = hidden_states[:, -(self.output_length+1):-1, :]
        if logit_grad_scale is not None:
            logit_states = scale_gradient(
                logit_states, logit_grad_scale
            )
        logits = self.lm_head(self.decoder_model.norm(logit_states))

        z_states = hidden_states[:, self.input_length:self.input_length + self.z_length]

        return logits, z_states


    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        noise: torch.FloatTensor=None,
        encoded_z: torch.FloatTensor=None,
    ):
        from transformers.cache_utils import DynamicCache
        from tqdm import tqdm

        # handle the noise
        if noise is None:
            noise = self.sample_noise(input_ids)

        # initialize the cache
        cache = DynamicCache()

        # pass the input tokens through the decoder
        input_tokens = self.embed_tokens(input_ids)
        input_tokens += unsqueeze_to_batch(
            self.decoder_input_embeddings, input_tokens
        )
        self.decoder_model(
            inputs_embeds=input_tokens,
            use_cache=True,
            past_key_values=cache,
        )

        # sample each z
        all_z = []
        prev_normed_z = torch.zeros_like(noise[:, 0, :]) # [B, latent_size]
        t_iter = torch.arange(1, self.scheduler.num_timesteps).to(noise.device).flip(0)
        for i in tqdm(range(self.z_length), desc="sampling z"):
            
            # pass the previous z token through the decoder
            z_token = (
                unsqueeze_to_batch(self.decoder_z_tokens[i], prev_normed_z) +
                self.decoder_z_proj_in(prev_normed_z)
            ) # [B, hidden_size]
            z_states = self.decoder_model(
                inputs_embeds=z_token[:, None, :],
                use_cache=True,
                past_key_values=cache,
            )[:, -1, :] # [B, hidden_size]

            # diffusion loop to sample the next z
            z_t = noise[:, i, :] # [B, latent_size]
            for t in t_iter:

                pred_z_0 = self.diffusion_head(
                    z_t,
                    t,
                    z_states,
                    self.scheduler,
                ) # [B, latent_size]
                z_t = self.scheduler.ddim_step(
                    z_t,
                    t,
                    pred_z_0,
                ) # [B, latent_size]

            if encoded_z is not None:
                z_t = encoded_z[:, i, :]

            # handle the sampled z
            all_z.append(z_t)
            prev_normed_z = F.rms_norm(z_t, [pred_z_0.shape[-1]], eps=self.config.rms_norm_eps)

        all_z = torch.stack(all_z, dim=1) # [B, z_length, latent_size]

        # sample the output tokens
        output_ids = []
        prev_logit_token = (
            unsqueeze_to_batch(self.decoder_z_tokens[-1], prev_normed_z) +
            self.decoder_z_proj_in(prev_normed_z)
        ) # [B, hidden_size]
        for i in tqdm(range(self.output_length), desc="sampling output"):

            logit_states = self.decoder_model(
                inputs_embeds=prev_logit_token[:, None, :],
                use_cache=True,
                past_key_values=cache,
            )[:, -1, :] # [B, hidden_size]

            logits = self.lm_head(self.decoder_model.norm(logit_states))
            next_token = torch.argmax(logits, dim=-1) # [B]

            output_ids.append(next_token)
            prev_logit_token = (
                unsqueeze_to_batch(self.decoder_output_embeddings, prev_logit_token) +
                self.embed_tokens(next_token)
            ) # [B, hidden_size]

        output_ids = torch.stack(output_ids, dim=-1)

        return output_ids, all_z
