import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.torch_utils import attach_gradient


class ScaledEmbedding(nn.Embedding):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale = self.embedding_dim ** 0.5


    def ones_init(self):
        self.weight.data.fill_(1.0 / self.scale)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input) * self.scale


class ContinuousEmbedding(nn.Module):

    def __init__(
        self,
        num_frequencies: int,
        embedding_dim: int,
        input_min: float=0.0,
        input_max: float=1.0,
        bias: bool=False,
    ):
        super().__init__()
        
        self.num_frequencies = num_frequencies
        self.embedding_dim = embedding_dim

        self.input_min = input_min
        self.input_max = input_max

        frequencies = np.pi * torch.arange(
            num_frequencies,
            dtype=torch.float32
        )
        self.register_buffer('frequencies', frequencies, persistent=True)

        self.proj = nn.Linear(
            2 * num_frequencies,
            embedding_dim,
            bias=bias,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_min) / (self.input_max - self.input_min)

        theta = x.unsqueeze(-1) * self.frequencies

        emb_sin = torch.sin(theta)
        emb_cos = torch.cos(theta)

        emb = torch.cat([emb_sin, emb_cos], dim=-1)

        return self.proj(emb)


class UnbiasedEMA(nn.Module):

    def __init__(
        self,
        shape: torch.Size,
        beta: float,
        eps: float=1e-8,
    ):
        super().__init__()

        self.shape = tuple(shape)
        self.beta = beta
        self.eps = eps

        self.register_buffer(
            'num_updates', torch.zeros(1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            'weight', torch.zeros(shape), persistent=True
        )


    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        assert x.shape == self.shape, f"Input shape {x.shape} does not match EMA shape {self.shape}"
       
        self.num_updates += 1
        self.weight.mul_(self.beta).add_(
            x.detach().to(self.weight.dtype), alpha=1 - self.beta
        )


    @torch.no_grad()
    def retrieve(self) -> torch.Tensor:
        bias_correction = 1 - self.beta ** self.num_updates.to(self.weight.dtype)
        return self.weight / (bias_correction + self.eps)


class CustomBatchNorm(nn.Module):

    def __init__(
        self,
        shape: torch.Size,
        beta: float,
        eps: float=1e-5,
        attach_gradients: bool=False,
    ):
        super().__init__()

        self.shape = tuple(shape)
        self.beta = beta
        self.eps = eps
        self.attach_gradients = attach_gradients

        self.mean_tracker = UnbiasedEMA(
            shape, beta, eps
        )
        self.var_tracker = UnbiasedEMA(
            shape, beta, eps
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == self.shape, f"Input shape {x.shape} does not match BatchNorm shape {self.shape}"

        og_dtype = x.dtype
        x = x.to(torch.float32)

        x_mean = x.mean(dim=0)
        x_var = x.var(dim=0, unbiased=False)

        if self.training:
            self.mean_tracker.update(x_mean)
            self.var_tracker.update(x_var)

        mean = self.mean_tracker.retrieve()
        var = self.var_tracker.retrieve()

        if self.attach_gradients:
            mean.requires_grad_(True)
            var.requires_grad_(True)

            mean = attach_gradient(mean, x_mean)
            var = attach_gradient(var, x_var)

        y = (x - mean[None]) / torch.sqrt(var + self.eps)[None]

        return y.to(og_dtype)
    