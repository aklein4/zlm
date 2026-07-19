"""Plot spherical-Cauchy KL to uniform for an 8-D direction vector.

The distribution is on S^(d-1), where d is the ambient dimension of the
direction vector.  Here d=8, so the distribution is on S^7.

The KL is the one in Theorem 1 of arXiv:2506.21278v2.  We evaluate an
equivalent one-dimensional expectation after the paper's Mobius
reparameterization.  This avoids cancellation between the logarithm and the
endpoint-singular integral in the theorem when rho is close to zero or one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import roots_jacobi


AMBIENT_DIMENSION = 8
NUM_QUADRATURE_POINTS = 256
OUTPUT_PATH = Path(__file__).with_name("sp_cauchy_kl_to_uniform_d8.png")


def kl_to_uniform(
    rho: np.ndarray,
    ambient_dimension: int = AMBIENT_DIMENSION,
    num_quadrature_points: int = NUM_QUADRATURE_POINTS,
) -> np.ndarray:
    """Return KL(spCauchy_d(mu, rho) || Uniform(S^(d-1))) in nats.

    By rotational symmetry, the result does not depend on the direction mu.
    If X is uniform on S^(d-1), its first coordinate has density proportional
    to (1-x^2)^((d-3)/2).  The paper's Mobius map therefore gives

      KL = (d-1) E[log((1 + rho^2 + 2 rho X_1)/(1-rho^2))].

    Symmetric Gauss-Jacobi nodes are paired to improve accuracy near rho=0.
    """

    rho = np.asarray(rho, dtype=np.float64)
    if np.any((rho < 0.0) | (rho >= 1.0)):
        raise ValueError("rho must satisfy 0 <= rho < 1")
    if ambient_dimension < 2:
        raise ValueError("ambient_dimension must be at least 2")
    if num_quadrature_points % 2:
        raise ValueError("use an even number of quadrature points")

    sphere_dimension = ambient_dimension - 1
    alpha = (ambient_dimension - 3) / 2
    nodes, weights = roots_jacobi(
        num_quadrature_points, alpha, alpha
    )
    weights /= weights.sum()

    positive = nodes > 0
    x_squared = nodes[positive] ** 2
    positive_weights = weights[positive]
    rho_squared = rho[:, None] ** 2

    # Pair x and -x before taking the logarithm:
    # log(1+r^2+2rx) + log(1+r^2-2rx)
    # = log(1 + r^2(2+r^2-4x^2)).
    expected_log_denominator = np.sum(
        positive_weights
        * np.log1p(
            rho_squared
            * (2.0 + rho_squared - 4.0 * x_squared)
        ),
        axis=1,
    )
    return sphere_dimension * (
        -np.log1p(-(rho**2)) + expected_log_denominator
    )


def main() -> None:
    # Dense linear coverage plus logarithmic coverage of the singular limit.
    rho = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.99, 800),
                1.0 - np.logspace(-2.0, -6.0, 300),
            ]
        )
    )
    kl = kl_to_uniform(rho)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(rho, kl, color="#1769aa", linewidth=2.2)
    ax.scatter([0.0], [0.0], color="#1769aa", s=28, zorder=3)
    ax.annotate(
        r"$\mathrm{KL}\to+\infty$ as $\rho\to1^{-}$",
        xy=(rho[-1], kl[-1]),
        xytext=(0.60, 0.82),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=11,
    )
    ax.set(
        title=(
            "Spherical Cauchy KL to uniform\n"
            r"8-D direction vector ($S^7$), measured in nats"
        ),
        xlabel=r"Concentration $\rho$",
        ylabel=r"$\mathrm{KL}(\mathrm{spCauchy}_8\,\|\,\mathrm{Uniform}(S^7))$",
        xlim=(0.0, 1.0),
        ylim=(0.0, None),
    )
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    x = np.linspace(0.0, 1.0, 1001)
    y = 8 * (x/0.73)**2.3
    ax.plot(x, y, color="#aa4444", linewidth=1.5, linestyle="--")
    ax.set_ylim(top=10.0)

    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved {OUTPUT_PATH}")
    for value in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999999):
        result = kl_to_uniform(np.array([value]))[0]
        print(f"rho={value:.6f}: KL={result:.9f} nats")


if __name__ == "__main__":
    main()
