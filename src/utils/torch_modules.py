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

    
    @torch.no_grad()
    def zero_out(self) -> None:
        self.num_updates.zero_()
        self.weight.zero_()


class CustomBatchNorm(nn.Module):

    def __init__(
        self,
        shape: torch.Size,
        beta: float,
        eps: float=1e-5,
    ):
        super().__init__()

        self.shape = tuple(shape)
        self.beta = beta
        self.eps = eps

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
        else:
            x_mean = self.mean_tracker.retrieve()
            x_var = self.var_tracker.retrieve()

        y = (x - x_mean[None]) * torch.rsqrt(x_var + self.eps)[None]

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

        min_val = torch.min(eig_vals.detach())
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

        return y.to(og_dtype), min_val


class OnceSpectralBatchNorm(SpectralBatchNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.count = 0

    
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

        if self.training and self.count < 2:
            self.count += 1

            self.mean_tracker.zero_out()
            self.cov_tracker.zero_out()

            self.mean_tracker.update(x_mean)
            self.cov_tracker.update(x_cov)

        x_mean = self.mean_tracker.retrieve()
        x_cov = self.cov_tracker.retrieve()

        eig_vals, eig_vecs = torch.linalg.eigh(
            x_cov + self.eps * torch.eye(self.shape[-1], device=x.device, dtype=x_cov.dtype)[None]
        )

        min_val = torch.min(eig_vals.detach())
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

        return y.to(og_dtype), min_val


class InitialSpectralNorm(nn.Module):

    def __init__(
        self,
        shape: torch.Size,
        eps: float=1e-5,
    ):
        super().__init__()
    
        self.shape = tuple(shape)
        assert len(self.shape) == 2, "SpectralBatchNorm only supports 2D inputs"
        self.eps = eps

        self.register_buffer(
            "initialized", torch.zeros(1, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "shift", torch.zeros(self.shape), persistent=True
        )
        self.register_buffer(
            "mat", torch.zeros(self.shape + (self.shape[-1],)), persistent=True
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

        if self.training:

            all_finite = torch.isfinite(x).all()
            do_update = (~self.initialized) & all_finite

            x_mean = x.mean(dim=1)  # [S, H]
            x_cov = torch.einsum(
                'sbi,sbj->sij',
                (x - x_mean[:, None]),
                (x - x_mean[:, None]),
            ) / x.shape[1] # [S, H, H]

            eig_vals, eig_vecs = torch.linalg.eigh(
                x_cov + self.eps * torch.eye(self.shape[-1], device=x.device, dtype=x_cov.dtype)[None]
            )

            eig_vals = torch.clamp(eig_vals, min=self.eps) # [S, H]         
            inv_sqrt_cov = (
                eig_vecs @
                torch.diag_embed(eig_vals.rsqrt()) @
                eig_vecs.transpose(-1, -2)  
            ) # [S, H, H]

            self.initialized |= do_update
            self.shift.copy_(
                torch.where(
                    do_update,
                    -x_mean.detach(),
                    self.shift
                )
            )
            self.mat.copy_(
                torch.where(
                    do_update,
                    inv_sqrt_cov.detach(),
                    self.mat
                )
            )

        y = torch.einsum(
            'shl,sbh->sbl',
            self.mat,
            (x + self.shift[:, None])
        ) # [S, B, H]

        y = y.transpose(0, 1) # [B, S, H]
        y = maybe_shard_with_gradients(y)

        return y.to(og_dtype)


class SeqPool(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        sequence_length: int,
        normalize: bool=False,
        eps: float=1e-8,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.normalize = normalize

        self.sequence_logits = nn.Parameter(
            torch.randn(sequence_length, hidden_size)
        )

        self.in_proj = nn.Linear(
            input_size, hidden_size,
            bias=False
        )
        self.out_proj = nn.Linear(
            hidden_size, output_size,
            bias=False
        )

        if self.normalize:
            self.norm = nn.RMSNorm(
                [output_size], eps=eps, elementwise_affine=False
            )
        else:
            self.register_parameter('norm', None)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        h = self.in_proj(hidden_states)
        weights = torch.softmax(self.sequence_logits, dim=-2)

        h = (h * weights).sum(dim=-2)

        output = self.out_proj(h)
        if self.normalize:
            output = self.norm(output)
        
        return output


class AdaPool(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pool_dim: int=-2,
        normalize: bool=False,
        eps: float=1e-8,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pool_dim = pool_dim
        self.normalize = normalize

        self.k_proj = nn.Linear(
            input_size, hidden_size,
            bias=False
        )
        self.v_proj = nn.Linear(
            input_size, hidden_size,
            bias=False
        )
        self.out_proj = nn.Linear(
            hidden_size, output_size,
            bias=False
        )

        if self.normalize:
            self.norm = nn.RMSNorm(
                [output_size], eps=eps, elementwise_affine=False
            )
        else:
            self.register_parameter('norm', None)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        v_states: torch.Tensor|None = None,
    ) -> torch.Tensor:

        if v_states is None:
            v_states = hidden_states

        keys = self.k_proj(hidden_states)
        values = self.v_proj(v_states)

        weights = torch.softmax(keys, dim=self.pool_dim)
        h = (values * weights).sum(dim=self.pool_dim)

        output = self.out_proj(h)
        if self.normalize:
            output = self.norm(output)
        
        return output
    