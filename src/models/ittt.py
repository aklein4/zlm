import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
if constants.XLA_AVAILABLE:
    pass

import math
from omegaconf import DictConfig

from models.llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import newton_schulz



class ItttFunction(torch.autograd.Function):

    @staticmethod
    # @torch.custom_fwd(device_type='xla', cast_inputs=torch.get_autocast_dtype('xla'))
    def forward(
        ctx,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        mod: "ItttLinear"
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x)
        ctx.mod = mod
        return z.clone()


    @staticmethod
    # @custom_bwd(device_type='xla')
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        og_grad = grad.clone()

        x, = ctx.saved_tensors
        mod: ItttLinear = ctx.mod

        x: torch.FloatTensor = x.float()
        g = grad.float()

        x = F.rms_norm(x, [x.shape[-1]], eps=mod.eps) # [b, s, i]
        g = F.normalize(g, dim=-2, eps=mod.eps) * math.sqrt(x.shape[-2])  # [b, s, r]

        x = x.to(mod.momentum_dtype)
        g = g.to(mod.momentum_dtype)

        # [b, r, i]
        update = (
            g.transpose(-2, -1) @ x
        ) / math.sqrt(x.shape[-2]) # approx 1 std

        assert mod.momentum is not None
        mod.momentum.lerp_(
            mod.momentum,
            update,
            1 - mod.momentum_beta
        )

        assert mod.delta is not None
        mod.delta.copy_(
            -newton_schulz(
                mod.momentum,
                eps=mod.eps
            )
        )

        return None, og_grad, None

        
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

        # save linear
        self.linear = linear
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )
        self.base_state_proj = nn.Linear(
            self.in_features, self.rank, bias=False
        )
        self.out_proj = nn.Linear(
            self.rank, self.out_features, bias=False
        )

        # ephemeral state
        self.state = None
        self.delta = None
        self.momentum = None

        # weight initialization
        self.base_state_proj.weight.data.zero_()
        self.out_proj.weight.data.normal_(
            std=config.initializer_range
        )

        # self.svd_init()

    
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
        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"
        assert self.state is not None

        lr = self.get_lr()
        s = lr[None] * self.state

        z = torch.einsum("boi,bsi->bso", s, x)
        z = ItttFunction.apply(x, z, self)

        z = z + self.base_state_proj(x)

        y_lora = self.out_proj(z)
        y_base = self.linear(x)

        y = y_base + y_lora

        return y


    @torch.no_grad()
    def init_state(self, input_ids: torch.LongTensor):

        self.state = torch.zeros(
            input_ids.shape[0], self.rank, self.in_features,
            device=input_ids.device, dtype=self.state_dtype,
        )
        self.delta = torch.zeros_like(
            self.state, dtype=self.momentum_dtype
        )
        self.momentum = torch.zeros_like(
            self.state, dtype=self.momentum_dtype
        )

        self.state = maybe_shard_with_gradients(self.state)
        self.delta = maybe_shard_with_gradients(self.delta)
        self.momentum = maybe_shard_with_gradients(self.momentum)
        

    @torch.no_grad()
    def empty_state(self):
        self.state = None
        self.momentum = None

    
    @torch.no_grad()
    def update_state(self):
        assert self.delta is not None
        assert self.state is not None

        self.state.add_(self.delta.to(self.state_dtype))


class ItttModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp.down_proj = ItttLinear(
                layer.mlp.down_proj,
                config
            )


    @torch.no_grad()
    def init_state(self, input_ids: torch.LongTensor):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.init_state(input_ids)


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
                    