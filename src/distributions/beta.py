"""
Beta distribution sampler using only Python-level PyTorch operations.

Strategy:
    Beta(a, b) = X / (X + Y)  where X ~ Gamma(a, 1), Y ~ Gamma(b, 1)

Gamma samples are drawn via Marsaglia & Tsang's transformation method (2000).
For shape parameter alpha < 1, we use Ahrens-Dieter's boost:
    Gamma(alpha, 1) = Gamma(alpha + 1, 1) * U^(1/alpha),  U ~ Uniform(0, 1)
"""

import torch
from torch import Tensor


def _standard_gamma(alpha: Tensor, eps: float = 1e-20) -> Tensor:
    """
    Sample from Gamma(alpha, 1) for arbitrary alpha > 0.

    Uses Marsaglia & Tsang (2000) for alpha >= 1, with the
    Ahrens-Dieter boost for alpha < 1.

    Args:
        alpha: Shape parameter (any shape tensor, all entries > 0).
        eps:   Small constant to clamp intermediate values for numerical safety.

    Returns:
        Tensor of independent Gamma(alpha, 1) samples, same shape as alpha.
    """
    # --- Ahrens-Dieter boost setup for alpha < 1 ---
    # Work with alpha_work >= 1 everywhere; we'll correct at the end.
    needs_boost = alpha < 1.0
    alpha_work = torch.where(needs_boost, alpha + 1.0, alpha)

    # --- Marsaglia & Tsang's method for alpha_work >= 1 ---
    d = alpha_work - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d).clamp(min=eps)

    # We'll iterate until every element has a valid sample.
    # `done` tracks which elements are finished.
    samples = torch.zeros_like(alpha)
    done = torch.zeros_like(alpha, dtype=torch.bool)

    max_iter = 10  # safety bound; typically converges in a few iterations
    for _ in range(max_iter):

        # Proposal: generate normal and uniform variates
        z = torch.randn_like(alpha)
        u = torch.rand_like(alpha).clamp(min=eps)

        v = (1.0 + c * z) ** 3  # v = (1 + c*z)^3

        # Rejection conditions:
        #   1) v must be positive  (i.e. 1 + c*z > 0)
        #   2) log(u) < 0.5*z^2 + d - d*v + d*log(v)
        valid_v = v > 0.0
        log_accept = 0.5 * z * z + d - d * v + d * torch.log(v.clamp(min=eps))
        accepted = valid_v & (torch.log(u) < log_accept) & ~done

        samples = torch.where(accepted, d * v, samples)
        done = done | accepted

    # --- Apply Ahrens-Dieter boost for original alpha < 1 ---
    # Gamma(alpha, 1) = Gamma(alpha+1, 1) * U^(1/alpha)
    u_boost = torch.rand_like(alpha).clamp(min=eps)
    boost_factor = u_boost.pow(1.0 / alpha.clamp(min=eps))
    samples = torch.where(needs_boost, samples * boost_factor, samples)

    return samples.clamp(min=eps)


def sample_beta(
    concentration1: Tensor,
    concentration0: Tensor,
) -> Tensor:
    """
    Draw independent samples from Beta(concentration1, concentration0).

    Args:
        concentration1: Alpha parameter (a > 0). Tensor of any shape.
        concentration0: Beta  parameter (b > 0). Must broadcast with concentration1.

    Returns:
        Tensor of Beta-distributed samples in (0, 1), shape determined by
        broadcasting concentration1 and concentration0.
    """
    with torch.autocast(device_type=concentration0.device.type, enabled=False):
        concentration1 = concentration1.float()
        concentration0 = concentration0.float()

        eps = 1e-7

        # Broadcast parameters to a common shape
        concentration1, concentration0 = torch.broadcast_tensors(
            concentration1, concentration0
        )

        x = _standard_gamma(concentration1)
        y = _standard_gamma(concentration0)

        # Beta sample = x / (x + y), clamped away from exact 0 and 1
        return (x / (x + y).clamp(min=eps)).clamp(min=eps, max=1.0 - eps)
