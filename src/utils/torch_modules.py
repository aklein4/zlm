import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.torch_utils import attach_gradient
from utils.sharding_utils import maybe_shard_with_gradients


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

        x = x.detach().to(self.weight.dtype)

        all_finite = torch.isfinite(x).all()

        self.num_updates += all_finite.to(self.num_updates.dtype)

        # only update the ema if all values are finite
        new_weight = (
            self.beta * self.weight +
            (1 - self.beta) * x
        )
        self.weight.copy_(
            torch.where(all_finite, new_weight, self.weight)
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

        y = (x - mean[None]) * torch.rsqrt(var + self.eps)[None]

        return y.to(og_dtype)


class MeanNorm(nn.Module):

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

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == self.shape, f"Input shape {x.shape} does not match BatchNorm shape {self.shape}"

        og_dtype = x.dtype
        x = x.to(torch.float32)

        x_mean = x.mean(dim=0)

        if self.training:
            self.mean_tracker.update(x_mean)
        mean = self.mean_tracker.retrieve()

        if self.attach_gradients:
            mean.requires_grad_(True)

            mean = attach_gradient(mean, x_mean)

        y = x - mean[None]

        return y.to(og_dtype)


class InitialBatchNorm(nn.Module):

    def __init__(
        self,
        shape: torch.Size,
        eps: float=1e-5,
    ):
        super().__init__()

        self.shape = tuple(shape)
        self.eps = eps

        self.register_buffer(
            "initialized", torch.zeros(1, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "shift", torch.zeros(shape), persistent=True
        )
        self.register_buffer(
            "scale", torch.ones(shape), persistent=True
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == self.shape, f"Input shape {x.shape} does not match BatchNorm shape {self.shape}"

        og_dtype = x.dtype
        x = x.to(torch.float32)

        if self.training:

            all_finite = torch.isfinite(x).all()
            do_update = (~self.initialized) & all_finite

            self.initialized |= do_update
            self.shift.copy_(
                torch.where(
                    do_update,
                    -x.mean(dim=0).detach(),
                    self.shift
                )
            )
            self.scale.copy_(
                torch.where(
                    do_update,
                    torch.rsqrt(x.var(dim=0, unbiased=False).detach() + self.eps),
                    self.scale
                )
            )

        y = (x + self.shift[None]) * self.scale[None]

        return y.to(og_dtype)
    

class SpectralBatchNorm(nn.Module):

    def __init__(
        self,
        shape: torch.Size,
        beta: float,
        eps: float=1e-5,
    ):
        super().__init__()
    
        self.shape = tuple(shape)
        assert len(self.shape) == 2, "SpectralBatchNorm only supports 2D inputs"
        self.eps = eps

        self.cov_tracker = UnbiasedEMA(
            self.shape + (shape[-1],),
            beta=beta, eps=eps
        )
        self.mean_tracker = UnbiasedEMA(
            shape,
            beta=beta, eps=eps
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral normalize x along the last dimension. Assume that the second
        axis of x in the sequence dimension, each of which should be handled independently.
        
        Args:
            x: (batch_size, sequence_length, hidden_size).
        """
        assert x.shape[1:] == self.shape, f"Input shape {x.shape} does not match BatchNorm shape {self.shape}"

        x = x.transpose(0, 1) # [S, B, H]
        x = maybe_shard_with_gradients(x)

        og_dtype = x.dtype
        x = x.to(torch.float32)

        x_mean = x.mean(dim=1)  # [S, H]
        x_cov = torch.einsum(
            'sbi,sbj->sij',
            (x - x_mean[:, None]),
            (x - x_mean[:, None]),
        ) / x.shape[1] # [S, H, H]

        if self.training:
            self.mean_tracker.update(x_mean)
            self.cov_tracker.update(x_cov)
        else:
            x_mean = self.mean_tracker.retrieve()
            x_cov = self.cov_tracker.retrieve()

        eig_vals, eig_vecs = torch.linalg.eigh(
            x_cov + self.eps * torch.eye(self.shape[-1], device=x.device, dtype=x_cov.dtype)[None]
        )

        eig_vals = torch.clamp(eig_vals, min=self.eps) # [S, H]
        inv_sqrt_cov = (
            eig_vecs @
            torch.diag_embed(eig_vals.rsqrt()) @
            eig_vecs.transpose(-1, -2)  
        ) # [S, H, H]

        y = torch.einsum(
            'shl,sbh->sbl',
            inv_sqrt_cov,
            (x - x_mean[:, None]),
        ) # [S, B, H]

        y = y.transpose(0, 1) # [B, S, H]
        y = maybe_shard_with_gradients(y)

        return y.to(og_dtype)
    