import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
import math
from omegaconf import DictConfig

from utils.torch_utils import (
    safe_copy_state,
    unsqueeze_to_batch, expand_to_batch,
    unsqueeze_to_channel, expand_to_channel,
    shift,
    scale_gradient,
    gaussian_init,
    slerp,
)

from models.llama import LlamaForCausalLM, LlamaRMSNorm
from models.custom_llama import CustomLlamaModel, CustomLlamaDecoderLayer
from models import load_checkpoint_state
from utils.torch_modules import ARLinear, UnbiasedEMA
from utils.cauchy import sample_sp_cauchy_noise, sample_sp_cauchy_noise_like, sample_sp_cauchy
import utils.constants as constants


class ARHead(nn.Module):

    def __init__(
        self,
        config: DictConfig,
        ar_steps: int = None,
        input_fn = None,
        output_fn = None,
    ):
        super().__init__()
        self.input_fn = input_fn
        self.output_fn = output_fn

        self.ar_steps = ar_steps
        if self.ar_steps is None:
            self.ar_steps = config.z_ar_steps

        self.states_gate_proj = nn.Linear(
            config.hidden_size, config.head_intermediate_size, bias=False
        )
        self.states_up_proj = nn.Linear(
            config.hidden_size, config.head_intermediate_size, bias=False
        )

        self.z_gate_proj = ARLinear(
            config.latent_size,
            config.head_intermediate_size,
            self.ar_steps,
            self_attend=False,
            bias=False
        )
        self.z_up_proj = ARLinear(
            config.latent_size,
            config.head_intermediate_size,
            self.ar_steps,
            self_attend=False,
            bias=False
        )

        self.down_proj = ARLinear(
            config.head_intermediate_size,
            config.latent_size,
            self.ar_steps,
            self_attend=True,
            bias=False
        )
        
        self.cross_proj = nn.Linear(
            config.hidden_size, config.latent_size, bias=False
        )

        self.act = ACT2FN[config.hidden_act]

    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        z: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.input_fn is not None:
            z = self.input_fn(z)

        g = (
            self.states_gate_proj(hidden_states) +
            self.z_gate_proj(z)
        )
        u = (
            self.states_up_proj(hidden_states) +
            self.z_up_proj(z)
        )

        h = self.act(g) * u

        out = (
            self.down_proj(h) +
            self.cross_proj(hidden_states)
        )

        if self.output_fn is not None:
            out = self.output_fn(out)
        return out


class EncoderModelLayer(CustomLlamaDecoderLayer):
    offload_name = "encoder_model_input"
class EncoderModel(CustomLlamaModel):
    layer_type = EncoderModelLayer

class DecoderModelLayer(CustomLlamaDecoderLayer):
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
        self.z_ar_steps = config.z_ar_steps
        self.sphere_size = self.latent_size // self.z_ar_steps

        initial_concentration = torch.tensor([config.init_concentration])
        self.log_concentration = nn.Parameter(
            torch.logit(initial_concentration) / math.sqrt(self.hidden_size)
        )

        # create the transformer backbones
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
                strict=True,
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
        self.decoder_model.do_norm = False
        self.decoder_z_states_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=False) # no affine to avoid vanishing gradients?

        # remove the embeddings from the transformers
        self.embed_tokens = self.encoder_model.embed_tokens
        self.encoder_model.embed_tokens = None
        self.decoder_model.embed_tokens = None

        # calculate embedding distribution
        if config.pretrained_llama is not None:
            embed_mean = self.embed_tokens.weight.data.mean(0).detach()
            embed_cov = torch.cov(self.embed_tokens.weight.data.T).detach()        
        else:
            embed_mean = torch.zeros(self.hidden_size)
            embed_cov = torch.eye(self.hidden_size)
        embed_dist = torch.distributions.MultivariateNormal(
            loc=embed_mean,
            covariance_matrix=(embed_cov + config.rms_norm_eps * torch.eye(self.hidden_size, device=embed_cov.device))
        )

        # create encoder special tokens
        self.encoder_sep_token = nn.Parameter(
            embed_dist.sample((1,))
        )
        self.encoder_z_tokens = nn.Parameter(
            embed_dist.sample((self.z_length,))
        )

        # create decoder special tokens
        self.decoder_z_tokens = nn.Parameter(
            embed_dist.sample((1 + self.z_length,))
        )
        self.decoder_start_output_token = nn.Parameter(
            embed_dist.sample((1,))
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

        # create the input functions
        self.encoder_noise_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        
        self.decoder_z_proj_in = nn.Linear(self.latent_size, self.hidden_size, bias=False)

        # create the output functions
        self.encoder_mu_proj_out = nn.Linear(
            self.hidden_size, self.latent_size, bias=False
        )

        self.decoder_head = ARHead(
            config, input_fn=self._l2_to_rms, output_fn=self._group_l2_norm
        )
        
        # for training
        self.lm_loss_ema = UnbiasedEMA([1], config.lm_loss_ema_beta, eps=config.rms_norm_eps)

        # optional extras
        self.is_probe = config.get("is_probe", False)
        if self.is_probe:
            self.progress_embeddings = nn.Parameter(
                torch.zeros(self.hidden_size)
            )

        if config.pretrained_llama is None:
            self.apply(gaussian_init)

        else:
            gaussian_init(self.encoder_noise_proj_in)
            gaussian_init(self.decoder_z_proj_in)

            gaussian_init(self.encoder_mu_proj_out)

            self.decoder_head.apply(gaussian_init)

        # ignore noise on encoder input at init
        self.encoder_noise_proj_in.weight.data.zero_()

        # init decoder_z_proj_in using the top |z| of the embedding covariance
        eigvals, eigvecs = torch.linalg.eigh(embed_cov)
        self.decoder_z_proj_in.weight.data.copy_(
            eigvecs[:, -self.latent_size:] * torch.sqrt(eigvals[None, -self.latent_size:])
        )


    def _sphere_shape(self, x: torch.FloatTensor) -> torch.FloatTensor:
        assert x.shape[-1] == self.latent_size, f"Expected last dimension to be {self.latent_size}, but got {x.shape[-1]}"
        return x.reshape(*x.shape[:-1], self.z_ar_steps, self.sphere_size)

    def _vec_shape(self, x: torch.FloatTensor) -> torch.FloatTensor:
        assert x.shape[-2] == self.z_ar_steps and x.shape[-1] == self.sphere_size, f"Expected last two dimensions to be ({self.z_ar_steps}, {self.sphere_size}), but got ({x.shape[-2]}, {x.shape[-1]})"
        return x.reshape(*x.shape[:-2], self.latent_size)


    def get_concentration(self):
        return torch.sigmoid(
            self.log_concentration * math.sqrt(self.hidden_size)
        )


    def _group_l2_norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self._sphere_shape(x)
        x = F.normalize(x.float(), dim=-1).to(x.dtype)
        return self._vec_shape(x)
    

    def _rms_to_l2(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x / math.sqrt(self.sphere_size)

    def _l2_to_rms(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x * math.sqrt(self.sphere_size)

    
    def sample_noise(
        self, 
        input_ids: torch.LongTensor,
    ) -> torch.FloatTensor:

        input_tokens = self.embed_tokens(torch.zeros_like(input_ids))
        
        noise = sample_sp_cauchy_noise(
            [input_ids.shape[0], self.z_length, self.z_ar_steps, self.sphere_size],
            device=input_tokens.device, dtype=input_tokens.dtype
        )

        return self._vec_shape(noise)

    
    def add_noise(
        self,
        mu: torch.FloatTensor,
        noise: torch.FloatTensor | None = None,
        noise_scale: float | None = None,
    ) -> torch.FloatTensor:

        mu = self._sphere_shape(mu)

        if noise is None:
            noise = sample_sp_cauchy_noise_like(mu)
        else:
            noise = self._sphere_shape(noise)

        z = sample_sp_cauchy(
            mu, self.get_concentration(), noise
        )
  
        if noise_scale is not None:
            z = slerp(mu, z, noise_scale)

        return self._vec_shape(z)


    def encode(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        noise: torch.FloatTensor=None,
        input_mask: torch.BoolTensor=None,
        output_mask: torch.BoolTensor=None,
        noise_scale: torch.FloatTensor = None,
        return_extra: bool = False,
    ):

        if noise is None:
            noise = self.sample_noise(
                input_ids,
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
                self.encoder_noise_proj_in(self._l2_to_rms(noise)),
                n=1, dim=-2, direction="right", narrow=True
            )
        )

        tokens = torch.cat(
            [
                input_tokens,
                expand_to_batch(self.encoder_sep_token, input_tokens),
                output_tokens,
                z_tokens,
            ],
            dim=-2
        )

        mask = None
        if input_mask is not None or output_mask is not None:
            assert input_mask is not None and output_mask is not None

            mask = torch.cat(
                [
                    input_mask,
                    torch.ones_like(input_ids[:, :1], dtype=torch.bool), # sep token
                    output_mask,
                    torch.ones(input_ids.shape[0], self.z_length, dtype=torch.bool, device=input_ids.device),
                ],
                dim=-1
            )

        hidden_states = self.encoder_model(
            inputs_embeds=tokens,
            elementwise_pad_mask=mask,
        )
        z_states = hidden_states[:, -self.z_length:, :]
        
        mu = self.encoder_mu_proj_out(z_states)
        mu = self._group_l2_norm(mu)

        z = self.add_noise(mu, noise, noise_scale)

        if return_extra:
            return z, mu, z_states

        return z, mu


    def decode(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        z: torch.FloatTensor,
        input_mask: torch.BoolTensor=None,
        output_mask: torch.BoolTensor=None,
        logit_grad_scale: float = None,
        progress: torch.IntTensor=None,
    ):

        input_tokens = self.embed_tokens(input_ids) + unsqueeze_to_batch(
            self.decoder_input_embeddings, input_ids
        )
        output_tokens = self.embed_tokens(output_ids) + unsqueeze_to_batch(
            self.decoder_output_embeddings, output_ids
        )

        z_projed = self.decoder_z_proj_in(
            self._l2_to_rms(z)
        )

        if progress is not None:
            assert self.is_probe, "Progress can only be used when is_probe is True"
            ar = torch.arange(self.z_length, device=progress.device, dtype=progress.dtype)
            progress_mask = progress[:, None] > ar[None]
            z_projed = torch.where(
                progress_mask[:, :, None],
                z_projed,
                expand_to_batch(self.progress_embeddings, z_projed),
            )
            
        z_tokens = (
            unsqueeze_to_batch(self.decoder_z_tokens, z_projed) +
            shift(
                z_projed,
                n=1, dim=-2, direction="right", narrow=False
            )
        )
        
        start_output_token = expand_to_batch(
            self.decoder_start_output_token, output_tokens
        )

        tokens = torch.cat(
            [
                input_tokens,
                z_tokens,
                start_output_token,
                output_tokens,
            ],
            dim=-2
        )

        mask = None
        if input_mask is not None or output_mask is not None:
            assert input_mask is not None and output_mask is not None

            mask = torch.cat(
                [
                    input_mask,
                    torch.ones(input_ids.shape[0], self.z_length + 1, dtype=torch.bool, device=input_ids.device), # z states
                    torch.ones(input_ids.shape[0], 1, dtype=torch.bool, device=input_ids.device), # start output token
                    output_mask,
                ],
                dim=-1
            )

        hidden_states = self.decoder_model(
            inputs_embeds=tokens,
            elementwise_pad_mask=mask,
        )

        input_length = input_ids.shape[-1]
        output_length = output_ids.shape[-1]

        logit_states = hidden_states[:, -(output_length+1):-1, :]
        if logit_grad_scale is not None:
            logit_states = scale_gradient(
                logit_states, logit_grad_scale
            )
        logits = self.lm_head(self.decoder_model.norm(logit_states)).float()

        z_states = hidden_states[:, input_length:input_length + self.z_length]
        z_states = self.decoder_z_states_norm(z_states)

        return logits, z_states


    @torch.no_grad()
    def set_ar_cache(self, value):
        for m in self.modules():
            if isinstance(m, ARLinear):
                m.set_cache(value)


    @torch.no_grad()
    def ar_rollout(
        self,
        z_states: torch.FloatTensor,
        noise: torch.FloatTensor,
        noise_temperature: float = 1.0,
    ):

        g_states = self.decoder_head.states_gate_proj(z_states)
        u_states = self.decoder_head.states_up_proj(z_states)
        cross = self.decoder_head.cross_proj(z_states)

        # naive iteration to perform AR head sampling
        z = torch.zeros_like(noise)
        for t in range(self.z_ar_steps):

            z_in = self.decoder_head.input_fn(z)
            g = g_states + self.decoder_head.z_gate_proj(z_in)
            u = u_states + self.decoder_head.z_up_proj(z_in)

            h = self.decoder_head.act(g) * u
            mu = self.decoder_head.output_fn(cross + self.decoder_head.down_proj(h))

            z = self.add_noise(mu, noise, noise_scale=noise_temperature)

        return z


    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        input_mask: torch.BoolTensor=None,
        noise: torch.FloatTensor=None,
        encoded_z: torch.FloatTensor=None,
        temperature: float | str = "greedy",
        num_output_tokens: int = None,
        verbose: bool=False,
        **rollout_kwargs,
    ):
        # TODO: update to cauchy

        from transformers.cache_utils import DynamicCache
        from tqdm import tqdm

        # create compiled functions
        if not getattr(self, "_compile_inited", False):
            self._compile_inited = True

            self._decoder_fn = self.decoder_model.forward
            self._ar_fn = self.ar_rollout

            if not constants.XLA_AVAILABLE:
                self._decoder_fn = torch.compile(self._decoder_fn, fullgraph=True, dynamic=True)
                self._ar_fn = torch.compile(self._ar_fn, fullgraph=True, dynamic=True, mode="reduce-overhead")

        # handle the noise
        if noise is None:
            noise = self.sample_noise(input_ids)

        # initialize the cache
        cache = DynamicCache()

        # pass the input tokens through the decoder to store kv in cache
        input_tokens = self.embed_tokens(
            torch.where(input_mask, input_ids, torch.zeros_like(input_ids))
            if input_mask is not None else input_ids
        )
        input_tokens += unsqueeze_to_batch(
            self.decoder_input_embeddings, input_tokens
        )
        self._decoder_fn(
            inputs_embeds=input_tokens,
            elementwise_pad_mask=input_mask,
            past_key_values=cache,
        )
        
        # initialize the position ids
        position_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        if input_mask is not None:
            position_ids += input_mask.sum(dim=-1)
        else:
            position_ids += input_ids.shape[1]

        # enable the AR cache
        self.set_ar_cache(True)

        # sample each z
        all_z = []
        prev_z = torch.zeros_like(noise[:, 0, :]) # [B, latent_size]
        for i in tqdm(range(self.z_length + 1), desc="sampling z", disable=(not verbose)):
            
            # pass the previous z token through the decoder
            z_token = (
                unsqueeze_to_batch(self.decoder_z_tokens[i], prev_z) +
                self.decoder_z_proj_in(self._l2_to_rms(prev_z))
            ) # [B, hidden_size]

            z_states = self._decoder_fn(
                inputs_embeds=z_token[:, None, :],
                past_key_values=cache,
                position_ids=position_ids[:, None],
            )[:, -1, :] # [B, hidden_size]
            z_states = self.decoder_z_states_norm(z_states)
            
            # update the position ids
            position_ids += 1

            # we did an extra pass to put the last z in the cache
            if i >= self.z_length:
                break

            # sample the next z
            if encoded_z is None:
                prev_z = self._ar_fn(
                    z_states,
                    noise[:, i, :],
                    **rollout_kwargs,
                ).clone()

            else:
                prev_z = encoded_z[:, i, :] # [B, latent_size]

            # handle the sampled z
            all_z.append(prev_z)

        # save the z
        all_z = torch.stack(all_z, dim=1) # [B, z_length, latent_size]

        # disable the AR cache
        self.set_ar_cache(False)

        # sample the output tokens
        if num_output_tokens is None:
            num_output_tokens = self.output_length
        if num_output_tokens <= 0:
            return None, all_z

        # sample each output token
        output_ids = []
        prev_logit_token = expand_to_batch(
            self.decoder_start_output_token, prev_z[:, None, :]
        )[:, 0, :] # [B, hidden_size]
        for i in tqdm(range(num_output_tokens), desc="sampling output", disable=(not verbose)):

            # pass the previous output token through the decoder
            logit_states = self._decoder_fn(
                inputs_embeds=prev_logit_token[:, None, :],
                past_key_values=cache,
                position_ids=position_ids[:, None],
            )[:, -1, :] # [B, hidden_size]

            # update the position ids
            position_ids += 1

            logits = self.lm_head(self.decoder_model.norm(logit_states)).float()
            
            # sample the next token
            if isinstance(temperature, str):
                assert temperature == "greedy", "Only 'greedy' temperature string is supported"
                next_token = torch.argmax(logits, dim=-1) # [B]
            
            else:
                probs = F.softmax(logits / temperature, dim=-1) # [B, vocab_size]
                next_token = torch.multinomial(probs, num_samples=1)[:, 0] # [B]

            # save the token
            output_ids.append(next_token)
            prev_logit_token = (
                unsqueeze_to_batch(self.decoder_output_embeddings, prev_logit_token) +
                self.embed_tokens(next_token)
            ) # [B, hidden_size]

        # stack output ids [B, output_length]
        output_ids = torch.stack(output_ids, dim=-1)

        return output_ids, all_z


    def get_logits(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        noise: torch.FloatTensor=None,
        z: torch.FloatTensor=None,
        verbose: bool=False,
        **rollout_kwargs,
    ):
        
        input_mask = (input_ids != self.config.pad_token_id)
        output_mask = (output_ids != self.config.pad_token_id)

        input_for_model = torch.where(
            input_mask,
            input_ids,
            torch.zeros_like(input_ids)
        )
        output_for_model = torch.where(
            output_mask,
            output_ids,
            torch.zeros_like(output_ids)
        )

        if z is None:
            _, z = self.sample(
                input_for_model,
                input_mask=input_mask,
                noise=noise,
                num_output_tokens=0,
                verbose=verbose,
                **rollout_kwargs,
            )
        
        logits, _ = self.decode(
            input_for_model,
            output_for_model,
            z,
            input_mask=input_mask,
            output_mask=output_mask,
        )

        return logits
    
