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

        # first portion will will come will come first
        x_train = x[:x.shape[0]//2].float()
        x_train = maybe_shard_with_gradients(x_train)

        x_val = x[x.shape[0]//2:].float()
        x_val = maybe_shard_with_gradients(x_val)
        
        g_train = grad[:grad.shape[0]//2].float()
        g_train = maybe_shard_with_gradients(g_train)

        g_val = grad[grad.shape[0]//2:].float()
        g_val = maybe_shard_with_gradients(g_val)

        with torch.set_grad_enabled(True):

            B = x_train.shape[0]
            lr = optimizer.param_groups[0]['lr']

            x_train = x_train.detach().clone().requires_grad_(True)
            g_train = g_train.detach().clone().requires_grad_(False)

            x_val = x_val.detach().clone().requires_grad_(False)
            g_val = g_val.detach().clone().requires_grad_(False)

            x_bias = (x_train.mean(0).abs() / x_train.std(0).clamp(min=eps)).mean()
            g_bias = (g_train.mean(0).abs() / g_train.std(0).clamp(min=eps)).mean()

            # calculate the update
            G_train = (
                g_train.transpose(-2, -1).to(torch.bfloat16)
                @ x_train.to(torch.bfloat16)
            )

            val_scale = x.shape[0] / x_val.shape[0] # this should be enabled for forward compatability
            G_val = val_scale * torch.einsum(
                'bol,bli->oi',
                g_val.transpose(-2, -1).to(torch.bfloat16),
                x_val.to(torch.bfloat16)
            )

            # adam-like preconditioning
            s = G_train.float().pow(2).mean(dim=0, keepdim=True).sqrt()
            P = 1 / torch.clamp(s, min=eps)
            PG = (P * G_train).to(torch.bfloat16)

            # elements on ~1 -> adam-like is ~0.2, negative like adam
            update = -0.2 * torch.sum(loss_scale * lr * PG, dim=0) / math.sqrt(B)

            # taylor approximation of the change in validation loss if we added update to the weights
            # which we want to make negative, and will be made so just like the main loss)
            iml_loss = torch.sum(G_val * update)

            x_grad = torch.autograd.grad(
                iml_loss, x_train
            )[0]

        # iml comes second
        x_grad = torch.cat(
            [x_grad, torch.zeros_like(x_val)],
            dim=0
        ).to(x_dtype)
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
    