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



class FoItttFunction(torch.autograd.Function):

    @staticmethod
    # @torch.custom_fwd(device_type='xla', cast_inputs=torch.get_autocast_dtype('xla'))
    def forward(
        ctx,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        mod: "ItttLinear",
        state: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        final_grad_buffer: torch.FloatTensor,
        lr: torch.FloatTensor,
        out_proj_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        ctx.save_for_backward(x, state, grad_buffer, final_grad_buffer, lr, out_proj_weight)
        ctx.mod = mod

        return y.clone()


    @staticmethod
    # @custom_bwd(device_type='xla')
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        if ctx.mod.second_pass:
            return second_backward(ctx, grad)

        return first_backward(ctx, grad)        


def first_backward(ctx, grad, x_kwarg=None):
    og_grad = grad.clone()

    x, state, grad_buffer, final_grad_buffer, lr, out_proj_weight = ctx.saved_tensors
    if x_kwarg is not None:
        x = x_kwarg
    mod: FoItttLinear = ctx.mod

    # get the inner gradient blo,bor->blr
    g = (
        grad.to(torch.bfloat16)
        @ out_proj_weight[None].to(torch.bfloat16)
    )

    # calculate the raw gradients for the grad buffer
    raw_G = (
        g.transpose(-2, -1)
        @ x.to(torch.bfloat16)
    )

    # calculate the update for the state
    update = -newton_schulz(
        raw_G, eps=mod.eps,
    ).to(state.dtype)

    return None, og_grad, None, update, raw_G.to(grad_buffer.dtype), None, None, None


def second_backward(ctx, grad):
    og_grad = grad.clone()

    x, state, grad_buffer, final_grad_buffer, lr, out_proj_weight = ctx.saved_tensors
    mod: FoItttLinear = ctx.mod

    x_dtype = x.dtype

    # split into lm and first-order components
    x = x[:x.shape[0]//2]
    x = maybe_shard_with_gradients(x.clone())

    grad_lm = grad[:grad.shape[0]//2]
    grad_lm = maybe_shard_with_gradients(grad_lm.clone())

    # do a regular backwards with the lm components
    _, __, ___, update_lm, raw_G_lm, ____, _____, ______ = first_backward(ctx, g_lm, x_kwarg=x)

    # calculate the future first-order gradients
    G_so_far = grad_buffer + raw_G_lm
    G_future = final_grad_buffer - G_so_far

    with torch.set_grad_enabled(True):

        x_leaf = x.detach().clone().requires_grad_(True).to(torch.bfloat16)
        out_proj_weight_leaf = out_proj_weight.detach().requires_grad_(True).to(torch.bfloat16)
        
        grad_lm = grad_lm.detach().requires_grad_(False).to(torch.bfloat16) 
        G_future = G_future.detach().requires_grad_(False).to(torch.bfloat16)
        lr = lr.detach().requires_grad_(False).to(torch.bfloat16)

        # get the inner gradient blo,bor->blr
        g_lm = (
            grad_lm
            @ out_proj_weight_leaf[None]
        )

        # calculate the raw gradients for the grad buffer
        raw_G = (
            g_lm.transpose(-2, -1)
            @ x_leaf
        )

        # calculate the update for the state
        update = -newton_schulz(
            raw_G, eps=mod.eps,
        )
        delta = update * lr[None]

        x_grad_fo, out_proj_weight_grad_fo = torch.autograd.grad(
            delta,
            [x_leaf, out_proj_weight_leaf],
            grad_outputs=G_future,
        )

    # lm, fo
    x_grad = torch.cat(
        [torch.zeros_like(x_grad_fo), x_grad_fo],
        dim=0
    ).to(x_dtype)
    x_grad = maybe_shard_with_gradients(x_grad).detach()

    out_proj_weight_grad = out_proj_weight_grad_fo.to(out_proj_weight.dtype)

    return x_grad, og_grad, None, update_lm, raw_G_lm, None, None, out_proj_weight_grad

        
class FoItttLinear(nn.Module):

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

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

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
        self.grad_buffer: nn.Buffer
        self.final_grad_buffer: nn.Buffer

        # weight initialization
        self.base_state_proj.weight.data.zero_()
        self.out_proj.weight.data.normal_(
            std=config.initializer_range
        )

        # first_pass: run the arch like normal
        # second_pass: run the arch with a duplicated forward and estimate the first-order gradients of the state
        self.second_pass = False
    

    def get_lr(self) -> torch.FloatTensor:
        return (
            self.base_lr *
            math.sqrt(max(self.in_features, self.rank) / self.in_features) *
            torch.exp(self.log_lr * self.scalar_scaler)
        )


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        lr = self.get_lr()
        s = lr[None] * self.state.detach()

        if self.second_pass:
            s = s.repeat(2, 1, 1)
            s = maybe_shard_with_gradients(s)

        z = torch.einsum("boi,bsi->bso", s, x)
        z = z + self.base_state_proj(x)
        
        y_lora = self.out_proj(z)

        y_lora = FoItttFunction.apply(
            x, y_lora, self,
            self.state, self.grad_buffer, self.final_grad_buffer,
            lr, self.out_proj.weight
        )
        
        y_base = self.linear(x)
        y = y_base + y_lora

        return y


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):

        state = torch.zeros(
            bs, self.rank, self.in_features,
            device=device, dtype=torch.float32
        )
        grad_buffer = torch.zeros_like(state)
        final_grad_buffer = torch.zeros_like(state)

        state = maybe_shard_with_gradients(state)
        grad_buffer = maybe_shard_with_gradients(grad_buffer)
        final_grad_buffer = maybe_shard_with_gradients(final_grad_buffer)
    
        self.register_buffer("state", state, persistent=False)
        self.register_buffer("grad_buffer", grad_buffer, persistent=False)
        self.register_buffer("final_grad_buffer", final_grad_buffer, persistent=False)

        self.state.requires_grad_(True)
        self.state.grad = torch.zeros_like(self.state)
        self.state.grad = maybe_shard_with_gradients(self.state.grad)

        self.grad_buffer.requires_grad_(True)
        self.grad_buffer.grad = torch.zeros_like(self.grad_buffer)
        self.grad_buffer.grad = maybe_shard_with_gradients(self.grad_buffer.grad)

        self.final_grad_buffer.requires_grad_(False)


    @torch.no_grad()
    def finalize_gradients(self):
        
        self.state.zero_()
        self.state.grad.zero_()

        self.final_grad_buffer.copy_(self.grad_buffer)
        
        self.grad_buffer.zero_()
        self.grad_buffer.grad.zero_()


    @torch.no_grad()
    def empty_state(self):

        self.state.zero_()
        self.state.grad.zero_()

        self.grad_buffer.zero_()
        self.grad_buffer.grad.zero_()

        self.final_grad_buffer.zero_()

    
    @torch.no_grad()
    def update_state(self):
        
        self.state.add_(self.state.grad)
        self.state.grad.zero_()

        self.grad_buffer.add_(self.grad_buffer.grad)
        self.grad_buffer.grad.zero_()

    
    @torch.no_grad()
    def relative_grad_error(self):

        est = self.grad_buffer + self.grad_buffer.grad
        target = self.final_grad_buffer

        err = (est - target).norm()
        denom = target.norm() + self.eps

        return err / denom
        

class FoItttModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp.down_proj = FoItttLinear(
                layer.mlp.down_proj,
                config
            )


    @torch.no_grad()
    def set_second_pass(self, value):
        for m in self.modules():
            if isinstance(m, FoItttLinear):
                m.second_pass = value


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for m in self.modules():
            if isinstance(m, FoItttLinear):
                m.init_state(bs, device)


    @torch.no_grad()
    def finalize_gradients(self):
        for m in self.modules():
            if isinstance(m, FoItttLinear):
                m.finalize_gradients()


    @torch.no_grad()
    def empty_state(self):
        for m in self.modules():
            if isinstance(m, FoItttLinear):
                m.empty_state()
                

    @torch.no_grad()
    def update_state(self):
        for m in self.modules():
            if isinstance(m, FoItttLinear):
                m.update_state()
                    
    
    @torch.no_grad()
    def relative_grad_error(self):
        errors = []
        for m in self.modules():
            if isinstance(m, FoItttLinear):
                errors.append(m.relative_grad_error())
        return torch.stack(errors).mean()
    