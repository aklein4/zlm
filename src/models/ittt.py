import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
if constants.XLA_AVAILABLE:
    pass

import math
from omegaconf import DictConfig
from tqdm import tqdm

from models.llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import newton_schulz, cuda_newton_schulz
from utils.loss_utils import lm_loss_fn



class ItttFunction(torch.autograd.Function):

    @staticmethod
    # @torch.custom_fwd(device_type='xla', cast_inputs=torch.get_autocast_dtype('xla'))
    def forward(
        ctx,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        mod: "ItttLinear",
        momentum: torch.FloatTensor,
        state: torch.FloatTensor,
        s_pre_norm: torch.FloatTensor,
        s_norm: torch.FloatTensor,
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x, momentum, state, s_pre_norm, s_norm)
        ctx.mod = mod
        return z.clone()


    @staticmethod
    # @custom_bwd(device_type='xla')
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        og_grad = grad.clone()

        x, momentum, state, s_pre_norm, s_norm = ctx.saved_tensors
        mod: ItttWeight = ctx.mod

        x: torch.FloatTensor = x.float()
        g = grad.float()

        x = F.rms_norm(x, [x.shape[-1]], eps=mod.eps) # [b, s, i]
        g = F.normalize(g, dim=-2, eps=mod.eps) * math.sqrt(x.shape[-2])  # [b, s, r]

        x = x.to(mod.momentum_dtype)
        g = g.to(mod.momentum_dtype)

        # [b, r, i]
        update = (
            g.transpose(-2, -1) @ x
        ) / math.sqrt(x.shape[-2])

        # account for frobenious norm in update
        s_pre_norm = s_pre_norm.to(mod.momentum_dtype)

        norm_correction = s_pre_norm * torch.einsum("boi,boi->b", update, s_pre_norm)[:, None, None]

        update = update.float()
        norm_correction = norm_correction.float()
        s_norm = s_norm.float()

        # absolute scale doesn't matter because ns normalizes
        update = (
            update / (s_norm + mod.eps) -
            norm_correction / (s_norm.pow(3) + mod.eps)
        ).to(mod.momentum_dtype)

        new_momentum = torch.lerp(
            momentum,
            update,
            1 - mod.momentum_beta
        ).detach()

        ns_fn = newton_schulz
        if not constants.XLA_AVAILABLE:
            ns_fn = cuda_newton_schulz()

        state_delta = -ns_fn(
            new_momentum,
            eps=mod.eps
        ).detach().to(mod.state_dtype)
    
        return None, og_grad, None, new_momentum, state_delta, None, None


class ItttWeight(nn.Module):

    def __init__(
        self,
        config: DictConfig,
        in_features: int,
        out_features: int,
    ):
        super().__init__()

        # save config
        self.in_features = in_features
        self.out_features = out_features

        # puts post-muon updates on scale 1/sqrt(in_features), same as initial base_state
        self.base_lr = (
            math.sqrt(max(self.in_features, self.out_features))
            / math.sqrt(self.in_features)
        )
        self.momentum_beta = config.momentum_beta

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = getattr(torch, config.momentum_dtype)
        self.state_dtype = getattr(torch, config.state_dtype)
        
        # params
        self.log_lr = nn.Parameter(
            torch.zeros(self.out_features, self.in_features)
        )
        self.base_state = nn.Parameter(
            torch.randn(self.out_features, self.in_features)
        )

        # ephemeral state
        self.state: nn.Buffer
        self.momentum: nn.Buffer

        # weight initialization
        self.base_state.data.normal_(
            std=1/math.sqrt(self.in_features)
        )
    

    def get_lr(self):
        return (
            self.base_lr *
            torch.exp(self.log_lr * self.scalar_scaler)
        )


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:

        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        s_pre_norm = (
            self.get_lr()[None] * self.state.detach()
            + self.base_state
        )
        s_norm = torch.norm(s_pre_norm, dim=[-2, -1], keepdim=True)

        # dividing by norm puts on scale 1/sqrt(in_features * out_features), multiplying by sqrt(out_features) puts on scale 1/sqrt(in_features)
        s = math.sqrt(self.out_features) * s_pre_norm / (s_norm + self.eps)

        z = torch.einsum("boi,bsi->bso", s, x)
        z = ItttFunction.apply(x, z, self, self.momentum, self.state, s_pre_norm, s_norm)

        return z


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):

        state = torch.zeros(
            bs, self.out_features, self.in_features,
            device=device, dtype=self.state_dtype,
        )
        momentum = torch.zeros_like(
            state, dtype=self.momentum_dtype
        )

        state = maybe_shard_with_gradients(state)
        momentum = maybe_shard_with_gradients(momentum)
    
        self.register_buffer("state", state, persistent=False)
        self.register_buffer("momentum", momentum, persistent=False)
        
        self.state.requires_grad_(True)
        self.state.grad = torch.zeros_like(self.state)
        self.state.grad = maybe_shard_with_gradients(self.state.grad)

        self.momentum.requires_grad_(True)
        self.momentum.grad = torch.zeros_like(self.momentum)
        self.momentum.grad = maybe_shard_with_gradients(self.momentum.grad)


    @torch.no_grad()
    def empty_state(self):

        self.state.zero_()
        self.state.grad.zero_()

        self.momentum.zero_()
        self.momentum.grad.zero_()

    
    @torch.no_grad()
    def update_state(self):
        
        self.state.add_(self.state.grad)
        self.state.grad.zero_()
        
        self.momentum.copy_(self.momentum.grad)
        self.momentum.grad.zero_()

        
class ItttLoRA(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        config: DictConfig,
        rank: int | None = None,
    ):
        super().__init__()

        # save config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank if rank is not None else config.rank

        # base linear
        self.linear = linear

        # ittt weights
        self.ittt_down = ItttWeight(config, self.in_features, self.rank)
        
        self.up = nn.Linear(self.rank, self.out_features, bias=False)
        self.up.weight.data.normal_(
            std=1/math.sqrt(self.rank)
        )

        # for baselines
        self.disable_ittt = config.get("disable_ittt", False)


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.disable_ittt:
            return self.linear(x)

        z = self.ittt_down(x)
        z = self.up(z)

        return self.linear(x) + z


class ItttModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp.down_proj = ItttLoRA(
                layer.mlp.down_proj,
                config
            )


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for m in self.modules():
            if isinstance(m, ItttWeight):
                m.init_state(bs, device)


    @torch.no_grad()
    def empty_state(self):
        for m in self.modules():
            if isinstance(m, ItttWeight):
                m.empty_state()
                

    @torch.no_grad()
    def update_state(self):
        for m in self.modules():
            if isinstance(m, ItttWeight):
                m.update_state()
                    

    def compute_logits(
        self,
        input_ids: torch.LongTensor,
        verbose: bool = False,
    ):
        
        chunks = torch.split(input_ids, self.config.chunk_size, dim=-1)

        ac_kwargs = {
            "device_type": str(input_ids.device),
            "dtype": torch.bfloat16,
        }

        self.init_state(input_ids.shape[0], input_ids.device)

        all_logits = []

        # first chunk
        with torch.autocast(**ac_kwargs):

            logits = self(
                chunks[0],
                logits_to_keep=slice(0, -1)
            )[0]
            all_logits.append(logits.detach().cpu())

            loss = lm_loss_fn(
                logits, chunks[0],
                shift_logits=False,
                ignore_index=self.config.pad_token_id,
            )

        loss.backward()

        # remaining chunks
        for i in tqdm(range(1, len(chunks)), desc="Processing Chunks", leave=False, disable=(not verbose)):
            
            first_chunk = chunks[i-1]
            second_chunk = chunks[i]
            all_chunk = torch.cat([first_chunk, second_chunk], dim=-1)

            self.update_state()

            with torch.autocast(**ac_kwargs):

                logits = self(
                    all_chunk,
                    logits_to_keep=slice(first_chunk.shape[-1]-1, -1)
                )[0]
                all_logits.append(logits.detach().cpu())

                loss = lm_loss_fn(
                    logits,
                    all_chunk[:, first_chunk.shape[-1]-1:].to(logits.device),
                    shift_logits=False,
                    ignore_index=self.config.pad_token_id,
                )

            loss.backward()

        self.zero_grad(True)
        self.empty_state()
            
        return torch.cat(all_logits, dim=1).detach()
