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
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x, momentum, state)
        ctx.mod = mod
        return z.clone()


    @staticmethod
    # @custom_bwd(device_type='xla')
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        og_grad = grad.clone()

        x, momentum, state = ctx.saved_tensors
        mod: ItttLinear = ctx.mod
        g = grad

        if mod.normalize_vectors:
            
            x: torch.FloatTensor = x.float()
            g = g.float()

            x = F.rms_norm(x, [x.shape[-1]], eps=mod.eps) # [b, s, i]
            g = F.normalize(g, dim=-2, eps=mod.eps) * math.sqrt(x.shape[-2])  # [b, s, r]

        if mod.center_x:

            x = x - x.mean(dim=-2, keepdim=True)

        x = x.to(mod.momentum_dtype)
        g = g.to(mod.momentum_dtype)

        # [b, r, i]
        update = (
            g.transpose(-2, -1) @ x
        ) / math.sqrt(x.shape[-2]) # approx 1 std

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
    
        return None, og_grad, None, new_momentum, state_delta

        
class ItttLinear(nn.Module):

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

        self.base_lr = config.base_lr
        self.momentum_beta = config.momentum_beta

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = getattr(torch, config.momentum_dtype)
        self.state_dtype = getattr(torch, config.state_dtype)

        self.normalize_vectors = config.get("normalize_vectors", True)
        self.center_x = config.get("center_x", False)

        # save linear
        self.linear = linear
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(self.rank, self.in_features)
        )
        self.base_state_proj = nn.Linear(
            self.in_features, self.rank, bias=False
        )
        self.out_proj = nn.Linear(
            self.rank, self.out_features, bias=False
        )

        # ephemeral state
        self.state: nn.Buffer
        self.momentum: nn.Buffer

        # weight initialization
        self.base_state_proj.weight.data.zero_()
        self.out_proj.weight.data.normal_(
            std=config.initializer_range
        )

        # self.svd_init()

        # for baselines
        self.disable_ittt = config.get("disable_ittt", False)

    
    # @torch.no_grad()
    # def svd_init(self):

    #     u, s, v = torch.linalg.svd(self.weight, full_matrices=False)

    #     self.out_proj.copy_(
    #         u[:, :self.rank] *
    #         s[None, :self.rank]
    #     )
    

    def get_lr(self):
        return (
            self.base_lr *
            torch.exp(self.log_lr * self.scalar_scaler)
        )


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.disable_ittt:
            return self.linear(x)

        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        lr = self.get_lr()
        s = lr[None] * self.state.detach()

        z = torch.einsum("boi,bsi->bso", s, x)
        z = ItttFunction.apply(x, z, self, self.momentum, self.state)

        z = z + self.base_state_proj(x)

        y_lora = self.out_proj(z)
        y_base = self.linear(x)

        y = y_base + y_lora

        return y


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        if self.disable_ittt:
            return

        state = torch.zeros(
            bs, self.rank, self.in_features,
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
        if self.disable_ittt:
            return

        self.state.zero_()
        self.state.grad.zero_()

        self.momentum.zero_()
        self.momentum.grad.zero_()

    
    @torch.no_grad()
    def update_state(self):
        if self.disable_ittt:
            return
        
        self.state.add_(self.state.grad)
        self.state.grad.zero_()
        
        self.momentum.copy_(self.momentum.grad)
        self.momentum.grad.zero_()


class ItttModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        attn_ittt = config.get("attn_ittt", False)

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp.down_proj = ItttLinear(
                layer.mlp.down_proj,
                config
            )

            if attn_ittt:
                layer.self_attn.q_proj = ItttLinear(
                    layer.self_attn.q_proj,
                    config
                )
                layer.self_attn.o_proj = ItttLinear(
                    layer.self_attn.o_proj,
                    config
                )


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.init_state(bs, device)


    @torch.no_grad()
    def empty_state(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.empty_state()
                

    @torch.no_grad()
    def update_state(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
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
                    all_chunk[:, first_chunk.shape[-1]:],
                    shift_logits=False,
                    shift_labels=False,
                    ignore_index=self.config.pad_token_id,
                )

            loss.backward()

        self.zero_grad(True)
        self.empty_state()
            
        return torch.cat(all_logits, dim=1).detach()
