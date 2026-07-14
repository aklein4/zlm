from __future__ import annotations

import torch
import torch.nn.functional as F

import math
from typing import Sequence


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(
        x.float(), dim=-1
    ).to(x.dtype)


def _concentration_tensor(
    concentration: torch.Tensor | float,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Convert a concentration and align its dimensions with a vector tensor."""

    if isinstance(concentration, float):
        concentration = torch.full_like(reference[..., :1], concentration)
    else:
        concentration = concentration.unsqueeze(-1)
    
    if concentration.ndim > reference.ndim:
        raise ValueError("concentration has more dimensions than direction.")
    while concentration.ndim < reference.ndim:
        concentration = concentration.unsqueeze(0)

    return concentration


def _parameter_vector(
    direction: torch.Tensor,
    concentration: torch.Tensor | float,
) -> torch.Tensor:
    direction = _normalize(direction)
    concentration = _concentration_tensor(concentration, direction)
    return direction * concentration


def sample_sp_cauchy_noise(
    shape: Sequence[int] | torch.Size,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate uniform noise on a unit hypersphere.

    ``shape`` a shape to use
    template.  The final dimension is interpreted as the ambient dimension.
    The returned noise can be passed to :func:`sample_spherical_cauchy` to
    reproduce a sample exactly.
    """

    if len(shape) == 0 or shape[-1] < 2:
        raise ValueError("noise shape must end in an ambient dimension >= 2")

    noise = torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    
    return _normalize(noise)


def sample_sp_cauchy_noise_like(
    template: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate uniform noise on a unit hypersphere with the same shape/device/dtype as a template tensor."""
    if device is None:
        device = template.device
    if dtype is None:
        dtype = template.dtype
    return sample_sp_cauchy_noise(
        template.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def sample_sp_cauchy(
    direction: torch.Tensor,
    concentration: torch.Tensor | float,
    noise: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Draw a differentiable, rejection-free spherical Cauchy sample.

    Args:
        direction: Tensor ending in the ambient dimension. It is
            normalized internally.
        concentration: Value in ``[0, 1)``. Batch dimensions are allowed and
            are aligned with the leading dimensions of ``direction``.
        noise: Optional uniform unit-sphere noise. Non-unit vectors are
            normalized, so fixed Gaussian noise may also be supplied. It may
            contain extra leading sample dimensions broadcastable with
            ``direction``.

    Returns:
        A unit vector with broadcasted batch/sample dimensions.
    """
    with torch.autocast(str(direction.device.type), enabled=False):
        return_dtype = direction.dtype

        if noise is None:
            noise = sample_sp_cauchy_noise_like(direction)

        direction = direction.float()
        noise = noise.float()
        
        parameter = _parameter_vector(direction, concentration)
        noise = _normalize(noise)
        
        parameter, noise = torch.broadcast_tensors(parameter, noise)

        parameter_norm_sq = parameter.square().sum(dim=-1, keepdim=True)
        denominator = (noise + parameter).square().sum(dim=-1, keepdim=True)
        denominator = denominator.clamp_min(eps)

        sample = (
            (1.0 - parameter_norm_sq) * (noise + parameter) / denominator
            + parameter
        )
        
        # The Mobius transform is analytically unit norm. Renormalizing removes
        # the small radial drift introduced by finite-precision arithmetic.
        sample = _normalize(sample)
        
        return sample.to(return_dtype)


def spherical_cauchy_log_prob(
    value: torch.Tensor,
    direction: torch.Tensor,
    concentration: torch.Tensor | float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Evaluate the exact log density with respect to surface-area measure.
    """
    if value.shape[-1] != direction.shape[-1]:
        raise ValueError(f"The last dimension of value and direction must match, got {value.shape[-1]} and {direction.shape[-1]}")

    with torch.autocast(str(direction.device.type), enabled=False):
        return_dtype = value.dtype

        value = value.float()
        direction = direction.float()

        value = _normalize(value)
        parameter = _parameter_vector(direction, concentration)
        
        value, parameter = torch.broadcast_tensors(value, parameter)

        d = value.shape[-1]
        intrinsic_d = d - 1

        sphere_log_area = (
            math.log(2.0)
            + 0.5 * d * math.log(math.pi)
            - math.lgamma(0.5 * d)
        )
        
        parameter_norm_sq = parameter.square().sum(dim=-1)
        denominator = (value - parameter).square().sum(dim=-1)

        log_prob = -sphere_log_area + intrinsic_d * (
            torch.log1p(-parameter_norm_sq.clamp_max(1-torch.finfo(torch.float32).eps))
            - torch.log(denominator.clamp_min(eps))
        )

        return log_prob.to(return_dtype)


@torch.no_grad()
def _gauss_jacobi_rule(
    intrinsic_dimension: int,
    num_points: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quadrature for a coordinate of a uniform point on ``S^n``.

    A coordinate has density proportional to
    ``(1 - x**2)**((n - 2) / 2)``.  Golub--Welsch gives nodes and normalized
    weights without introducing a SciPy dependency.
    """

    if intrinsic_dimension < 2:
        raise ValueError("Gauss-Jacobi rule is only needed for S^n with n >= 2")
    if num_points < 2:
        raise ValueError("num_quadrature_points must be at least 2")

    alpha = 0.5 * (intrinsic_dimension - 2)
    k = torch.arange(1, num_points, dtype=torch.float32, device=device)
    off_diagonal = torch.sqrt(
        k * (k + 2.0 * alpha) / ((2.0 * k + 2.0 * alpha).square() - 1.0)
    )
    jacobi_matrix = torch.diag(off_diagonal, diagonal=1)
    jacobi_matrix += torch.diag(off_diagonal, diagonal=-1)
    nodes, eigenvectors = torch.linalg.eigh(jacobi_matrix)
    weights = eigenvectors[0].square()

    # Enforce the exact symmetry of the rule. This prevents odd quadrature
    # errors from dominating the very small KL near the uniform distribution.
    nodes = 0.5 * (nodes - nodes.flip(0))
    weights = 0.5 * (weights + weights.flip(0))
    weights /= weights.sum()

    return nodes, weights


def _kl_to_uniform(
    relative_radius_sq: torch.Tensor,
    d: int,
    num_quadrature_points: int,
) -> torch.Tensor:
    """KL(SC(a) || uniform) as a one-dimensional analytic expectation."""

    intrinsic_d = d - 1
    radius_sq = relative_radius_sq.clamp(min=0.0, max=1.0 - torch.finfo(torch.float32).eps)

    # On S^1 the spherical Cauchy is the wrapped Cauchy and the expression is
    # elementary. This is also more accurate than endpoint-singular quadrature.
    if intrinsic_d == 1:
        return -torch.log1p(-radius_sq)

    nodes, weights = _gauss_jacobi_rule(
        intrinsic_d,
        num_quadrature_points,
        device=radius_sq.device,
    )

    # Pair the symmetric nodes analytically:
    # log(1+r^2+2rx) + log(1+r^2-2rx)
    #   = log((1+r^2)^2 - 4r^2 x^2).
    # Besides halving the work, this expresses KL directly in r^2 and avoids
    # the undefined gradient of sqrt(r^2) when the two distributions coincide.
    half = num_quadrature_points // 2
    positive_nodes = nodes[-half:]
    positive_weights = weights[-half:]
    radius_sq_expanded = radius_sq.unsqueeze(-1)
    paired_log_denominator = torch.log1p(
        radius_sq_expanded
        * (
            2.0
            + radius_sq_expanded
            - 4.0 * positive_nodes.square()
        )
    )
    expected_log_denominator = torch.sum(
        positive_weights * paired_log_denominator,
        dim=-1,
    )
    if num_quadrature_points % 2:
        expected_log_denominator = expected_log_denominator + (
            weights[half] * torch.log1p(radius_sq)
        )

    kl = intrinsic_d * (
        -torch.log1p(-radius_sq) + expected_log_denominator
    )
    
    # Roundoff can produce tiny negatives when the two distributions are
    # almost identical; the exact divergence is non-negative.
    return kl.clamp_min(0.0)


def sp_cauchy_kl(
    posterior_direction: torch.Tensor,
    posterior_concentration: torch.Tensor | float,
    prior_direction: torch.Tensor,
    prior_concentration: torch.Tensor | float | None = None,
    num_quadrature_points: int = 64,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute ``KL(posterior || prior)`` for spherical Cauchy distributions.

    The family is closed under Mobius transformations. Applying the inverse
    prior transform reduces an arbitrary pair to a spherical Cauchy versus the
    uniform distribution. The remaining analytic one-dimensional expectation
    is evaluated with deterministic Gauss-Jacobi quadrature.

    Concentrations may have batch dimensions; the returned tensor contains the
    broadcasted batch/sample dimensions, without the final vector dimension.
    """
    if prior_concentration is None:
        prior_concentration = posterior_concentration

    with torch.autocast(str(posterior_direction.device.type), enabled=False):
        return_dtype = posterior_direction.dtype

        posterior_direction = posterior_direction.float()
        prior_direction = prior_direction.float()

        posterior_parameter = _parameter_vector(
            posterior_direction,
            posterior_concentration,
        )
        prior_parameter = _parameter_vector(
            prior_direction,
            prior_concentration,
        )
        posterior_parameter, prior_parameter = torch.broadcast_tensors(
            posterior_parameter,
            prior_parameter,
        )

        d = posterior_parameter.shape[-1]

        difference_sq = (posterior_parameter - prior_parameter).square().sum(dim=-1)
        posterior_radius_sq = posterior_parameter.square().sum(dim=-1)
        prior_radius_sq = prior_parameter.square().sum(dim=-1)

        # The norm of the relative Mobius parameter. The second denominator form
        # is both symmetric in the pair and stable when the parameters are close.
        denominator = difference_sq + (
            (1.0 - posterior_radius_sq) * (1.0 - prior_radius_sq)
        )
        relative_radius_sq = difference_sq / denominator.clamp_min(eps)

        kl = _kl_to_uniform(
            relative_radius_sq,
            d,
            num_quadrature_points,
        )

        return kl.to(return_dtype)
