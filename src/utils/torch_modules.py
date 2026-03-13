import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import utils.constants as constants
if constants.XLA_AVAILABLE:
    from torch_xla.distributed.spmd.xla_sharding import XLAPatchedLinear

from utils.torch_utils import attach_gradient, unsqueeze_to_batch
from utils.sharding_utils import maybe_shard_with_gradients


class GroupRMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_groups: int,
        eps: float=1e-6,
        elementwise_affine: bool=True,
    ):
        super().__init__()

        assert hidden_size % num_groups == 0, f"hidden_size {hidden_size} must be divisible by num_groups {num_groups}"

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = hidden_size // num_groups
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("weight", None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.hidden_size, f"Input hidden size {x.shape[-1]} does not match expected hidden size {self.hidden_size}"

        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        input_shape = x.shape
        x = x.reshape(*x.shape[:-1], self.num_groups, self.group_size)

        x = F.rms_norm(
            x, [self.group_size], eps=self.eps
        )

        x = x.reshape(input_shape)
        if self.elementwise_affine:
            x = x * (1.0 + unsqueeze_to_batch(self.weight, x))
        
        return x.to(input_dtype)
    

class ARLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_steps: int,
        self_attend: bool,
        bias: bool=True,
    ):
        """ An autoregressive linear layer.
        
        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            num_steps (int): The number of autoregressive steps.
            self_attend (bool): Whether to allow the current step to see itself.
            bias (bool): Whether to include a bias term.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_steps = num_steps
        self.self_attend = self_attend

        assert self.in_features % self.num_steps == 0, f"in_features {in_features} must be divisible by num_steps {num_steps}"
        assert self.out_features % self.num_steps == 0, f"out_features {out_features} must be divisible by num_steps {num_steps}"

        self.weight = nn.Parameter(
            torch.randn(self.out_features, self.in_features)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features)
            )
        else:
            self.register_parameter('bias', None)

        n = self.num_steps
        mask = torch.ones(n, n)
        mask = torch.tril(mask, diagonal=0 if self.self_attend else -1)
        mask = mask.repeat_interleave(self.out_features // n, dim=0).repeat_interleave(self.in_features // n, dim=1)
        self.register_buffer('mask', mask, persistent=True)

        self.weight.data.copy_(
            (
                self.mask *
                torch.randn_like(self.weight) /
                (torch.sqrt(self.mask.sum(dim=-1, keepdim=True)) + 1e-6)
            ).detach()
        )

        self.cache = None


    def set_cache(self, on):
        if on:
            self.cache = self.weight * self.mask.to(self.weight.dtype)
        else:
            self.cache = None

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache is not None:
            masked_weight = self.cache
        else:
            masked_weight = self.weight * self.mask.to(self.weight.dtype)

        if constants.XLA_AVAILABLE:
            return XLAPatchedLinear.apply(x, masked_weight, self.bias)
        else:
            return F.linear(x, masked_weight, self.bias)
        

class ContinuousEmbedding(nn.Module):

    def __init__(
        self,
        num_frequencies: int,
        embedding_dim: int | None = None,
        input_min: float=0.0,
        input_max: float=1.0,
        bias: bool=False,
    ):
        super().__init__()
        
        self.num_frequencies = num_frequencies
        self.embedding_dim = embedding_dim

        self.input_min = input_min
        self.input_max = input_max

        frequencies = np.pi * (1 + torch.arange(
            num_frequencies,
            dtype=torch.float32
        ))
        self.register_buffer('frequencies', frequencies, persistent=True)

        if embedding_dim is not None:
            self.proj = nn.Linear(
                2 * num_frequencies,
                embedding_dim,
                bias=bias,
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_min) / (self.input_max - self.input_min)

        theta = x.unsqueeze(-1) * self.frequencies.to(x.dtype)

        emb_sin = torch.sin(theta)
        emb_cos = torch.cos(theta)

        emb = torch.cat([emb_sin, emb_cos], dim=-1)

        if self.embedding_dim is not None:
            return self.proj(emb)

        return emb


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

        self.weight.lerp_(
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
            (1 - self.beta) * all_finite.to(self.weight.dtype)
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
        eps: float=1e-5,
    ):
        super().__init__()
        self.norm = InternalSpectralBatchNorm(shape, eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            y, min_val = self.norm(x)
        return y, min_val


class InternalSpectralBatchNorm(nn.Module):

    def __init__(
        self,
        shape: torch.Size,
        eps: float=1e-5,
    ):
        super().__init__()
    
        self.shape = tuple(shape)
        assert len(self.shape) == 2, "SpectralBatchNorm only supports 2D inputs"
        self.eps = eps

    
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

        # if self.training:
        #     self.mean_tracker.update(x_mean)
        #     self.cov_tracker.update(x_cov)
        # else:
        #     x_mean = self.mean_tracker.retrieve()
        #     x_cov = self.cov_tracker.retrieve()

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

        y = y.to(og_dtype)

        return y, min_val


class OnceSpectralBatchNorm(InternalSpectralBatchNorm):

    def __init__(self, *args, **kwargs):
        inited = kwargs.pop("inited", False)
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "mean", torch.zeros(self.shape, dtype=torch.float32), persistent=True
        )
        self.mean: nn.Buffer

        self.register_buffer(
            "inv_sqrt_cov", torch.zeros(self.shape + (self.shape[-1],), dtype=torch.float32), persistent=True
        )
        self.inv_sqrt_cov: nn.Buffer

        self.inited = inited

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral normalize x along the last dimension. Assume that the second
        axis of x in the sequence dimension, each of which should be handled independently.
        
        Args:
            x: (batch_size, sequence_length, hidden_size).
        """
        assert x.shape[1:] == self.shape, f"Input shape {x.shape} does not match BatchNorm shape {self.shape}"

        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
        
            x = x.transpose(0, 1) # [S, B, H]
            x = maybe_shard_with_gradients(x)

            og_dtype = x.dtype
            x = x.to(torch.float32)

            if self.training and not self.inited:
                
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

                self.mean.copy_(x_mean.detach())
                self.inv_sqrt_cov.copy_(inv_sqrt_cov.detach())

                self.inited = True

            y = torch.einsum(
                'shl,sbh->sbl',
                self.inv_sqrt_cov,
                (x - self.mean[:, None]),
            ) # [S, B, H]

            y = y.transpose(0, 1) # [B, S, H]
            y = maybe_shard_with_gradients(y)

            y = y.to(og_dtype)

        return y, 0.0


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
    