"""KL divergence between two spherical-Cauchy distributions vs. cosine similarity.

This reproduces the inverse-Mobius-transform KL from ``src/utils/cauchy.py``
(:func:`sp_cauchy_kl`) using only the standard library ``math`` module -- no
torch, no numpy, no scipy.

Method (Theorem 1 of arXiv:2506.21278v2, in the paper's Mobius parameterization):

The spherical-Cauchy family is closed under Mobius transformations.  Applying
the inverse of the prior's transform maps the pair ``(posterior, prior)`` onto
``(spCauchy(relative), uniform)``, so an arbitrary KL reduces to a KL against
the uniform distribution at a single "relative radius" ``r``::

    p_i = mu_i * u_i                       (Mobius parameter, |u_i| = 1)
    difference_sq        = |p_post - p_prior|^2
    denominator          = difference_sq + (1 - |p_post|^2)(1 - |p_prior|^2)
    relative_radius_sq   = difference_sq / denominator

    KL = (d-1) * ( -log(1 - r^2) + E_X[ log(1 + r^2 - 2 r X_1) ] )

where ``X`` is uniform on ``S^(d-1)`` and its coordinate ``X_1`` has density
proportional to ``(1 - x^2)^((d-3)/2)``.  The 1-D expectation is evaluated with
Gauss-Jacobi quadrature (Golub-Welsch), matching cauchy.py's ``_gauss_jacobi_rule``.

Here both distributions share concentration 0.735 in d=8 (the sphere S^7), so the
KL depends only on the cosine similarity ``c`` between their direction vectors:

    difference_sq = mu^2 * |u - v|^2 = 2 mu^2 (1 - c).
"""

from __future__ import annotations

import math
from pathlib import Path


AMBIENT_DIMENSION = 8
CONCENTRATION = 0.735
NUM_QUADRATURE_POINTS = 64
OUTPUT_PATH = Path(__file__).with_name("sp_cauchy_kl_vs_cosine_d8.png")


def _symmetric_tridiagonal_eig(
    diagonal: list[float],
    off_diagonal: list[float],
) -> tuple[list[float], list[list[float]]]:
    """Eigen-decompose a symmetric tridiagonal matrix (QL with implicit shifts).

    ``off_diagonal[i]`` is the sub/super-diagonal entry between rows ``i`` and
    ``i+1`` (length ``n-1``).  Returns eigenvalues and the eigenvector matrix
    ``z`` (columns are eigenvectors), a pure-Python port of Numerical Recipes'
    ``tqli`` restricted to what Golub-Welsch needs.
    """

    n = len(diagonal)
    d = list(diagonal)
    # e[i] is the off-diagonal element linking row i to row i+1; e[n-1] = 0.
    e = list(off_diagonal) + [0.0]
    z = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for ell in range(n):
        it = 0
        while True:
            # Find a small sub-diagonal element to split the matrix.
            m = ell
            while m < n - 1:
                dd = abs(d[m]) + abs(d[m + 1])
                if abs(e[m]) + dd == dd:
                    break
                m += 1
            if m == ell:
                break

            it += 1
            if it > 50:
                raise RuntimeError("tqli: too many iterations")

            g = (d[ell + 1] - d[ell]) / (2.0 * e[ell])
            r = math.hypot(g, 1.0)
            g = d[m] - d[ell] + e[ell] / (g + math.copysign(r, g))
            s = 1.0
            c = 1.0
            p = 0.0
            broke = False
            for i in range(m - 1, ell - 1, -1):
                f = s * e[i]
                b = c * e[i]
                r = math.hypot(f, g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m] = 0.0
                    broke = True
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                for k in range(n):
                    f = z[k][i + 1]
                    z[k][i + 1] = s * z[k][i] + c * f
                    z[k][i] = c * z[k][i] - s * f
            if broke:
                continue
            d[ell] -= p
            e[ell] = g
            e[m] = 0.0

    return d, z


def gauss_jacobi_rule(
    intrinsic_dimension: int,
    num_points: int,
) -> tuple[list[float], list[float]]:
    """Nodes and normalized weights for the coordinate density on ``S^n``.

    A coordinate of a uniform point on ``S^n`` has density proportional to
    ``(1 - x^2)^((n - 2) / 2)`` -- Jacobi weight with ``alpha = beta = (n-2)/2``.
    Golub-Welsch: the Jacobi matrix is symmetric tridiagonal with zero diagonal
    (symmetric weight) and known off-diagonals; weights come from the squared
    first component of each eigenvector.  Mirrors ``_gauss_jacobi_rule``.
    """

    alpha = 0.5 * (intrinsic_dimension - 2)
    off_diagonal = [
        math.sqrt(k * (k + 2.0 * alpha) / ((2.0 * k + 2.0 * alpha) ** 2 - 1.0))
        for k in range(1, num_points)
    ]
    diagonal = [0.0] * num_points

    nodes, eigenvectors = _symmetric_tridiagonal_eig(diagonal, off_diagonal)
    weights = [eigenvectors[0][j] ** 2 for j in range(num_points)]

    # Sort by node so the symmetric pairing below is exact.
    order = sorted(range(num_points), key=lambda j: nodes[j])
    nodes = [nodes[j] for j in order]
    weights = [weights[j] for j in order]

    # Enforce the exact symmetry of the rule (as in cauchy.py).
    nodes = [0.5 * (nodes[j] - nodes[-1 - j]) for j in range(num_points)]
    weights = [0.5 * (weights[j] + weights[-1 - j]) for j in range(num_points)]
    total = sum(weights)
    weights = [w / total for w in weights]

    return nodes, weights


def kl_to_uniform(
    relative_radius_sq: float,
    ambient_dimension: int = AMBIENT_DIMENSION,
    num_quadrature_points: int = NUM_QUADRATURE_POINTS,
) -> float:
    """KL(spCauchy(relative) || uniform) as a 1-D analytic expectation."""

    intrinsic_d = ambient_dimension - 1
    r2 = min(max(relative_radius_sq, 0.0), 1.0 - 1e-7)

    # On S^1 the spherical Cauchy is the wrapped Cauchy: elementary and exact.
    if intrinsic_d == 1:
        return -math.log1p(-r2)

    nodes, weights = gauss_jacobi_rule(intrinsic_d, num_quadrature_points)

    # Pair symmetric nodes before the log:
    #   log(1 + r^2 + 2 r x) + log(1 + r^2 - 2 r x) = log(1 + r^2(2 + r^2 - 4x^2)).
    half = num_quadrature_points // 2
    expected_log_denominator = 0.0
    for x, w in zip(nodes[-half:], weights[-half:]):
        expected_log_denominator += w * math.log1p(r2 * (2.0 + r2 - 4.0 * x * x))
    if num_quadrature_points % 2:
        expected_log_denominator += weights[half] * math.log1p(r2)

    kl = intrinsic_d * (-math.log1p(-r2) + expected_log_denominator)
    return max(kl, 0.0)


def sp_cauchy_kl_from_cosine(
    cosine_similarity: float,
    concentration: float = CONCENTRATION,
    ambient_dimension: int = AMBIENT_DIMENSION,
    num_quadrature_points: int = NUM_QUADRATURE_POINTS,
) -> float:
    """KL between two spCauchy distributions sharing a concentration.

    Depends only on the cosine similarity between the two direction vectors.
    """

    mu2 = concentration * concentration
    difference_sq = 2.0 * mu2 * (1.0 - cosine_similarity)
    # Both radii equal mu^2 since the concentrations match.
    denominator = difference_sq + (1.0 - mu2) * (1.0 - mu2)
    relative_radius_sq = difference_sq / max(denominator, 1e-12)
    return kl_to_uniform(
        relative_radius_sq, ambient_dimension, num_quadrature_points
    )


def gaussian_kl_from_cosine(
    cosine_similarity: float,
    ambient_dimension: int = AMBIENT_DIMENSION,
) -> float:
    """KL between two identity-covariance Gaussians on the sqrt(2)-RMS sphere.

    The means lie on the sphere of RMS radius sqrt(2), i.e. |mu|^2 = 2 d.  For
    equal covariance KL(N(mu1, I) || N(mu2, I)) = 0.5 |mu1 - mu2|^2, and with
    equal-norm means at cosine c,

        |mu1 - mu2|^2 = 2 |mu|^2 (1 - c) = 4 d (1 - c),

    so KL = 2 d (1 - c).  (Symmetric in the two distributions.)
    """

    return 2.0 * ambient_dimension * (1.0 - cosine_similarity)


def main() -> None:
    print(
        f"KL(spCauchy || spCauchy) in {AMBIENT_DIMENSION}-D (S^{AMBIENT_DIMENSION - 1}), "
        f"concentration={CONCENTRATION}, in nats\n"
    )

    # Dense grid for the curve.
    num_curve = 801
    cosine_curve = [-1.0 + 2.0 * i / (num_curve - 1) for i in range(num_curve)]
    kl_curve = [sp_cauchy_kl_from_cosine(c) for c in cosine_curve]
    gauss_curve = [gaussian_kl_from_cosine(c) for c in cosine_curve]

    # Sparse grid for the printed table.
    print(f"{'cosine_similarity':>18}   {'spCauchy KL':>14}   {'Gaussian KL':>14}")
    print(f"{'-' * 18}   {'-' * 14}   {'-' * 14}")
    for i in range(201):
        cosine = -1.0 + 2.0 * i / 200
        print(
            f"{cosine:>18.4f}   "
            f"{sp_cauchy_kl_from_cosine(cosine):>14.9f}   "
            f"{gaussian_kl_from_cosine(cosine):>14.9f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(
        cosine_curve,
        kl_curve,
        color="#1769aa",
        linewidth=2.2,
        label=rf"spCauchy, concentration $={CONCENTRATION}$ ($S^{{{AMBIENT_DIMENSION - 1}}}$)",
    )
    ax.plot(
        cosine_curve,
        gauss_curve,
        color="#c0392b",
        linewidth=2.2,
        linestyle="--",
        label=r"Gaussian $\mathcal{N}(\mu, I)$, $|\mu|^2 = 2d$ ($\sqrt{2}$-RMS sphere)",
    )
    ax.scatter([1.0], [0.0], color="#444444", s=28, zorder=3)
    ax.set(
        title=(
            "KL divergence vs. cosine similarity between directions\n"
            rf"$d={AMBIENT_DIMENSION}$, measured in nats"
        ),
        xlabel="Cosine similarity between direction / mean vectors",
        ylabel="KL divergence (nats)",
        xlim=(-1.0, 1.0),
        ylim=(0.0, None),
    )
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
