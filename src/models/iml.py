import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
if constants.XLA_AVAILABLE:
    from torch_xla.distributed.spmd.xla_sharding import XLAPatchedLinear

import math
from omegaconf import DictConfig

from models.llama import LlamaForCausalLM
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import newton_schulz



class IMLFunction(torch.autograd.Function):

    @staticmethod
    # @torch.custom_fwd(device_type='xla', cast_inputs=torch.get_autocast_dtype('xla'))
    def forward(
        ctx,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        loss_buffer: nn.Buffer,
        log_loss_buffer: nn.Buffer,
        x_bias_buffer: nn.Buffer,
        g_bias_buffer: nn.Buffer,
        eps: float,
        loss_scale: float,
        optimizer: torch.optim.Optimizer,
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x, loss_buffer, log_loss_buffer, x_bias_buffer, g_bias_buffer)
        ctx.eps = eps
        ctx.loss_scale = loss_scale
        ctx.optimizer = optimizer
        return y.clone()


    @staticmethod
    # @custom_bwd(device_type='xla')
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:
        og_grad = grad.clone()

        x, loss_buffer, log_loss_buffer, x_bias_buffer, g_bias_buffer = ctx.saved_tensors
        eps = ctx.eps
        loss_scale = ctx.loss_scale
        optimizer = ctx.optimizer

        x_dtype = x.dtype

        with torch.set_grad_enabled(True):

            x = x.detach().clone().requires_grad_(True)
            g = grad.detach().clone().requires_grad_(False)

            # first portion will will come will come first
            x_train = x[:x.shape[0]//2].to(torch.bfloat16)
            x_train = maybe_shard_with_gradients(x_train)

            x_val = x[x.shape[0]//2:].to(torch.bfloat16)
            x_val = maybe_shard_with_gradients(x_val)
            
            g_train = g[:g.shape[0]//2].to(torch.bfloat16)
            g_train = maybe_shard_with_gradients(g_train)

            g_val = g[g.shape[0]//2:].to(torch.bfloat16)
            g_val = maybe_shard_with_gradients(g_val)

            lr = optimizer.param_groups[0]['lr']
            rms_scale = optimizer.param_groups[0]['rms_scale']

            x_bias = (x_train.float().mean(0).abs() / x_train.float().std(0).clamp(min=eps)).mean()
            g_bias = (g_train.float().mean(0).abs() / g_train.float().std(0).clamp(min=eps)).mean()

            # calculate the update
            G_train = torch.einsum(
                'bol,bli->oi',
                g_train.transpose(-2, -1),
                x_train
            )

            val_scale = x.shape[0] / x_val.shape[0]
            G_val = val_scale * torch.einsum(
                'bol,bli->oi',
                g_val.transpose(-2, -1),
                x_val
            )

            # muon-like preconditioning
            update = newton_schulz(G_train)

            # elements on ~1 -> adam-like is ~0.2, negative like adam
            step_size = lr * rms_scale * math.sqrt(
                max(update.shape[0], update.shape[1])
            )

            iml_loss = torch.sum(G_val * (-step_size * update))
            loss_for_backward = loss_scale * iml_loss

            x_grad = torch.autograd.grad(
                loss_for_backward, x
            )[0]

        # iml comes second
        x_grad = x_grad.detach().to(x_dtype)
        x_grad = maybe_shard_with_gradients(x_grad)

        loss_buffer_grad = iml_loss.detach().to(loss_buffer.dtype).reshape_as(loss_buffer)
        log_loss_buffer_grad = iml_loss.detach().to(log_loss_buffer.dtype).reshape_as(log_loss_buffer)

        x_bias_grad = x_bias.detach().to(x_bias_buffer.dtype).reshape_as(x_bias_buffer)
        g_bias_grad = g_bias.detach().to(g_bias_buffer.dtype).reshape_as(g_bias_buffer)

        return x_grad, og_grad, loss_buffer_grad, log_loss_buffer_grad, x_bias_grad, g_bias_grad, None, None, None


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
        self.eps = config.iml_eps
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

        self.register_buffer(
            "x_bias_buffer", torch.zeros(1), persistent=True
        )
        self.x_bias_buffer: nn.Buffer
        self.register_buffer(
            "g_bias_buffer", torch.zeros(1), persistent=True
        )
        self.g_bias_buffer: nn.Buffer

        self.optimizer = None


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if constants.XLA_AVAILABLE:
            y = XLAPatchedLinear.apply(x, self.weight, self.bias)
        else:
            y = F.linear(x, self.weight, self.bias)

        y = IMLFunction.apply(
            x, y,
            self.loss_buffer, self.log_loss_buffer,
            self.x_bias_buffer, self.g_bias_buffer,
            self.eps, self.loss_scale, self.optimizer
        )

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
    

    @torch.no_grad()
    def get_previous_biases(self):
        x_bias = self.x_bias_buffer.grad.clone()
        g_bias = self.g_bias_buffer.grad.clone()

        self.x_bias_buffer.grad.zero_()
        self.g_bias_buffer.grad.zero_()

        return x_bias, g_bias


class IMLModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        num_iml = 0
        for m in self.model.layers.modules():

            if isinstance(m, (nn.Linear, IMLLinear)):
                continue

            for name, m_ in m.named_modules():
                if isinstance(m_, nn.Linear):
                    m.set_submodule(
                        name,
                        IMLLinear(m_, config),
                        strict=True
                    )
                    num_iml += 1

        # for m in self.modules():
            
        #     if isinstance(m, IMLLinear):
        #         m.loss_scale /= num_iml


    @torch.no_grad()
    def get_previous_loss(self):

        losses = []
        for m in self.modules():

            if isinstance(m, IMLLinear):
                losses.append(m.get_previous_loss())

        return torch.stack(losses).sum()


    @torch.no_grad()
    def get_previous_log_loss(self):

        log_losses = []
        for m in self.modules():

            if isinstance(m, IMLLinear):
                log_losses.append(m.get_previous_log_loss())

        return torch.stack(log_losses).mean()
    

    @torch.no_grad()
    def get_previous_biases(self):

        x_biases = []
        g_biases = []
        for m in self.modules():

            if isinstance(m, IMLLinear):
                x_bias, g_bias = m.get_previous_biases()
                x_biases.append(x_bias)
                g_biases.append(g_bias)

        return torch.stack(x_biases).mean(), torch.stack(g_biases).mean()
    