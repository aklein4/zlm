import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
if constants.XLA_AVAILABLE:
    from torch_xla.distributed.spmd.xla_sharding import XLAPatchedLinear

import math
from omegaconf import DictConfig
from tqdm import tqdm

from models.llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import newton_schulz, cuda_newton_schulz
from utils.loss_utils import lm_loss_fn



class IMLFunction(torch.autograd.Function):

    @staticmethod
    # @torch.custom_fwd(device_type='xla', cast_inputs=torch.get_autocast_dtype('xla'))
    def forward(
        ctx,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        loss_buffer: nn.Buffer,
        log_loss_buffer: nn.Buffer,
        eps: float,
        loss_scale: float,
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x, loss_buffer, log_loss_buffer)
        ctx.eps = eps
        ctx.loss_scale = loss_scale
        return y.clone()


    @staticmethod
    # @custom_bwd(device_type='xla')
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:
        og_grad = grad.clone()

        x, loss_buffer, log_loss_buffer = ctx.saved_tensors
        eps = ctx.eps
        loss_scale = ctx.loss_scale

        x_dtype = x.dtype

        # lm will come first
        x = x[:x.shape[0]//2].to(torch.bfloat16)
        x = maybe_shard_with_gradients(x)

        g = grad[:grad.shape[0]//2].to(torch.bfloat16)
        g = maybe_shard_with_gradients(g)

        with torch.set_grad_enabled(True):

            x = x.detach().clone().requires_grad_(True)
            g = g.detach().clone().requires_grad_(False)

            # calculate the update
            update = g.transpose(-2, -1) @ x

            # adam-like preconditioning
            update = update / (
                update.pow(2).mean(0, keepdim=True).sqrt() + eps
            )

            direction = F.normalize(update, dim=[-2, -1], eps=eps)
            direction_sum = direction.sum(dim=0, keepdim=True)
            direction_sum = maybe_shard_with_gradients(direction_sum, spec=[None, ['data', 'fsdp'], None])

            l_raw = torch.einsum(
                "io,io->",
                direction_sum, direction_sum
            ) / direction.shape[0]
            l = ((l_raw - 1) / (direction.shape[0] - 1)).mean()

            l = l * math.sqrt(direction.shape[-2] * direction.shape[-1])
            log_l = torch.log2(l + 1.0)

            l_for_backwards = l * loss_scale
            x_grad = torch.autograd.grad(
                l_for_backwards, x
            )[0]

        # iml comes second
        x_grad = torch.cat(
            [torch.zeros_like(x_grad), x_grad],
            dim=0
        ).to(x_dtype)
        x_grad = maybe_shard_with_gradients(x_grad)

        loss_buffer_grad = l.detach().to(loss_buffer.dtype).reshape_as(loss_buffer)
        log_loss_buffer_grad = log_l.detach().to(log_loss_buffer.dtype).reshape_as(log_loss_buffer)

        return x_grad, og_grad, loss_buffer_grad, log_loss_buffer_grad, None, None


class IMLLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        config: DictConfig,
    ):
        super().__init__()

        # save config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.eps = config.rms_norm_eps
        self.loss_scale = config.iml_loss_scale

        # parameters
        self.weight = linear.weight
        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)

        # buffers
        self.register_buffer(
            "loss_buffer", torch.zeros(1), persistent=True
        )
        self.loss_buffer: nn.Buffer
        self.register_buffer(
            "log_loss_buffer", torch.zeros(1), persistent=True
        )
        self.log_loss_buffer: nn.Buffer


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if constants.XLA_AVAILABLE:
            y = XLAPatchedLinear.apply(x, self.weight, self.bias)
        else:
            y = F.linear(x, self.weight, self.bias)

        y = IMLFunction.apply(x, y, self.loss_buffer, self.log_loss_buffer, self.eps, self.loss_scale)

        return y

    
    @torch.no_grad()
    def get_previous_loss(self):
        out = self.loss_buffer.grad.clone()
        self.loss_buffer.grad.zero_()
        return out

    @torch.no_grad()
    def get_previous_log_loss(self):
        out = self.log_loss_buffer.grad.clone()
        self.log_loss_buffer.grad.zero_()
        return out


class IMLModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        for m in self.modules():

            if isinstance(m, (nn.Linear, IMLLinear)):
                continue

            for name, m_ in m.named_modules():
                if isinstance(m_, nn.Linear):
                    m.set_submodule(
                        name,
                        IMLLinear(m_, config),
                        strict=True
                    )


    @torch.no_grad()
    def get_previous_loss(self):

        losses = []
        for m in self.modules():

            if isinstance(m, IMLLinear):
                losses.append(m.get_previous_loss())

        return torch.stack(losses).mean()


    @torch.no_grad()
    def get_previous_log_loss(self):

        log_losses = []
        for m in self.modules():

            if isinstance(m, IMLLinear):
                log_losses.append(m.get_previous_log_loss())

        return torch.stack(log_losses).mean()
    