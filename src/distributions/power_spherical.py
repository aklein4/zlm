import torch
import torch.nn as nn

from torch.distributions import Beta

import math


def l2_norm(
    x: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        return x.float() / torch.clamp(x.float().norm(dim=-1, keepdim=True), min=eps)


def unsqueeze_to_batch(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    while x.dim() < target.dim():
        x = x[None]
    return x


class PowerSphericalDistribution(nn.Module):

    def __init__(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        eps: float = 1e-7
    ):
        super().__init__()
        self.eps: float = eps

        # [..., m]
        self.mu: torch.Tensor = l2_norm(mu, eps)
        # [...,]
        self.kappa: torch.Tensor = torch.clamp(kappa, min=0.0)

        self.m: int = self.mu.shape[-1]
        self.d: int = self.m - 1
        beta_const: float = 0.5 * self.d

        # 0-d
        self.d_tensor: torch.Tensor = torch.full_like(self.kappa.mean(), self.d)

        # [...,]
        self.alpha: torch.Tensor = self.kappa + beta_const
        # [...,]
        self.beta: torch.Tensor = torch.full_like(self.kappa, beta_const)

        # [...,]
        self.log_normalizer: torch.Tensor = self._get_log_normalizer()
        
        # [...,]
        self._entropy = self._get_entropy()

        # [...,]
        self._kl_to_uniform = self._get_kl_to_uniform()

        self.device_type = self.mu.device.type


    def _get_log_normalizer(self) -> torch.Tensor:
        # log N_X(κ,d) = -[ (α+β)log 2 + β log π + lgamma(α) - lgamma(α+β) ]
        with torch.autocast(device_type=self.device_type, enabled=False):
            alpha = self.alpha.float()
            beta = self.beta.float()
            return (
                -(alpha + beta) * math.log(2.0)
                - beta * math.log(math.pi)
                - torch.lgamma(alpha)
                + torch.lgamma(alpha + beta)
            )


    def _get_entropy(self) -> torch.Tensor:
        # H = -[ log N_X + κ ( log 2 + ψ(α) - ψ(α+β) ) ]
        with torch.autocast(device_type=self.device_type, enabled=False):
            return -(
                self.log_normalizer.float()
                + self.kappa.float()
                * (
                    math.log(2.0)
                    + (torch.digamma(self.alpha.float()) - torch.digamma(self.alpha.float() + self.beta.float()))
                )
            )
    
    def entropy(self) -> torch.Tensor:
        return self._entropy


    def _get_kl_to_uniform(self) -> torch.Tensor:
        # KL(q || U(S^{d})) = -H(q) + log |S^{d}|
        with torch.autocast(device_type=self.device_type, enabled=False):
            d = self.d_tensor.float()
            log_area = (
                math.log(2.0)
                + 0.5 * (d + 1.0) * math.log(math.pi)
                - torch.lgamma(0.5 * (d + 1.0))
            )
            return -self.entropy() + log_area

    def kl_to_uniform(self) -> torch.Tensor:
        return self._kl_to_uniform


    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = l2_norm(x.float(), self.eps)
            
            mu = unsqueeze_to_batch(self.mu.float(), x)
            kappa = unsqueeze_to_batch(self.kappa.float(), x[..., 0])
            normalizer = unsqueeze_to_batch(self.log_normalizer.float(), x[..., 0])

            dot = (mu * x).sum(dim=-1)
            return normalizer + kappa * torch.log1p(dot)


    def sample_noise(self, sample_shape=None):
        with torch.autocast(device_type=self.device_type, enabled=False):
            alpha = self.alpha.float()
            beta = self.beta.float()
            mu = self.mu.float()

            if sample_shape is not None:
                alpha = alpha.expand(*sample_shape, *alpha.shape)
                beta = beta.expand(*sample_shape, *beta.shape)
                mu = mu.expand(*sample_shape, *mu.shape)

            Z = Beta(alpha, beta).rsample()

            v = torch.randn(
                *mu.shape[:-1],
                self.m - 1,
                device=self.mu.device,
                dtype=self.mu.dtype,
            )
            v = l2_norm(v, self.eps)

            return v, Z


    def rsample(self, v=None, Z=None, sample_shape=None) -> torch.Tensor:
        if sample_shape is not None:
            assert v is None and Z is None, "Cannot specify sample_shape when providing v or Z"
        if v is not None or Z is not None:
            assert sample_shape is None, "Cannot specify v or Z when providing sample_shape"
        assert (v is None) == (Z is None), "Must provide both v and Z or neither"

        with torch.autocast(device_type=self.device_type, enabled=False):

            if v is None:
                v, Z = self.sample_noise(sample_shape)
            v = v.float()
            Z = Z.float()

            t = (2.0 * Z - 1.0).unsqueeze(-1)  # [..., 1]

            # 2) v ~ U(S^{m-2})
            # [..., m-1]
            v = l2_norm(v, self.eps)

            # [..., m]
            y = torch.cat(
                [t, torch.sqrt(torch.clamp(1 - t**2, min=0.0)) * v], dim=-1
            )

            # [..., m]
            mu = unsqueeze_to_batch(self.mu.float(), y)

            # [..., m]
            e1 = torch.zeros_like(mu)
            e1[..., 0] = 1.0

            # [..., m]
            u = l2_norm(e1 - mu, self.eps)

            # [..., m]
            z = y - 2.0 * (y * u).sum(dim=-1, keepdim=True) * u

            parallel = (mu - e1).abs().sum(dim=-1, keepdim=True) < self.eps
            z = torch.where(parallel, y, z)

            return z


class Golden4dPSDistribution(PowerSphericalDistribution):

    # for 4 dimensions, the KL to uniform is 4 nats
    GOLDEN_KAPPA = 63.7869

    def __init__(
        self,
        mu: torch.Tensor,
        eps: float = 1e-7
    ):
        assert mu.shape[-1] == 4, "Golden4dPSDistribution only supports 4-dimensional inputs"

        kappa = torch.full_like(mu[..., 0], self.GOLDEN_KAPPA)

        super().__init__(mu, kappa, eps)


class Golden4DKLCalculator(nn.Module):

    # KL curve fit for dot >= 0.0
    lower_coeffs = [
        5.24411214e-01, -2.96590039e-01, 1.21247993e-01, 1.26576863e+00,
        2.77426800e+00, 7.62221589e+00, 3.17313031e+01, 1.38956018e-03
    ]

    # KL curve fit for dot < 0.0
    upper_coeffs = [
        18.95245979, -34.8507762, -52.2592189, 36.06428744,
        194.84618774, 86.81489707, -547.88970628, 344.09985189,
    ]


    def __init__(self):
        super().__init__()

        lower_coeffs = torch.tensor(self.lower_coeffs, dtype=torch.float32)
        upper_coeffs = torch.tensor(self.upper_coeffs, dtype=torch.float32)
        exponents = torch.arange(len(self.lower_coeffs), dtype=torch.float32).flip(0)

        self.register_buffer("lower_coeffs", lower_coeffs, persistent=True)
        self.register_buffer("upper_coeffs", upper_coeffs, persistent=True)
        self.register_buffer("exponents", exponents, persistent=True)


    def _polyval(self, coeffs, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float().unsqueeze(-1)
            exponents = unsqueeze_to_batch(self.exponents, x)
            coeffs = unsqueeze_to_batch(coeffs, x)

            return (coeffs * x**exponents).sum(dim=-1)

    
    def forward(
        self,
        dist: Golden4dPSDistribution,
        other: Golden4dPSDistribution,
    ) -> torch.Tensor:
        with torch.autocast(device_type=dist.device_type, enabled=False):

            dot = (dist.mu.float() * other.mu.float()).sum(dim=-1)
            x = 1 - dot

            upper = self._polyval(self.upper_coeffs, x)
            lower = self._polyval(self.lower_coeffs, x)

            return torch.where(x > 1, upper, lower)


@torch.no_grad()
def get_kl_curve():
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    import math
    import pickle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_mu = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    base_kappa = torch.tensor([Golden4dPSDistribution.GOLDEN_KAPPA], device=device)
    base_dist = PowerSphericalDistribution(base_mu, base_kappa)

    d = torch.logspace(-4, math.log10(2.0-1e-4), 10001, base=10.0, device=device)
    dots = 1 - d
    thetas = torch.acos(dots)

    mu = torch.stack([torch.zeros_like(thetas), torch.cos(thetas), torch.sin(thetas), torch.zeros_like(thetas)], dim=-1)
    kappa = base_kappa.repeat(len(thetas))
    dist = PowerSphericalDistribution(mu, kappa)

    z = base_dist.rsample(sample_shape=(100000,))
    logp = dist.log_prob(z).mean(0)

    kl = -base_dist.entropy() - logp

    data = {
        "kl": kl.cpu().numpy(),
        "dot": dots.cpu().numpy(),
    }
    with open("power_spherical_kl.pkl", "wb") as f:
        pickle.dump(data, f)

    plt.plot(dots.cpu().numpy(), kl.cpu().numpy())
    plt.xlabel("q.mu * p.mu")
    plt.ylabel("KL(q || p)")
    plt.title("Emperical KL divergence between PowerSpherical distributions")
    plt.grid()
    plt.savefig("power_spherical_kl.png")
    plt.clf()


def approximate_kl(dot):

    import numpy as np

    lower_coeffs = [
        5.24411214e-01, -2.96590039e-01, 1.21247993e-01, 1.26576863e+00,
        2.77426800e+00, 7.62221589e+00, 3.17313031e+01, 1.38956018e-03
    ]

    upper_coeffs = [
        18.95245979, -34.8507762, -52.2592189, 36.06428744,
        194.84618774, 86.81489707, -547.88970628, 344.09985189,
    ]

    x = 1 - dot
    upper = np.polyval(upper_coeffs, x)
    lower = np.polyval(lower_coeffs, x)

    return np.where(x > 1, upper, lower)


def fit_kl_curve():

    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    with open("power_spherical_kl.pkl", "rb") as f:
        data = pickle.load(f)

    x = (1 - data["dot"])
    y = data["kl"]

    num_keep =  (x < 1).sum()
    # x = x[num_keep:]
    # y = y[num_keep:]

    coeffs = np.polyfit(
        x,
        y,
        deg=7
    )
    # y_fit = (
    #     np.polyval(
    #         coeffs,
    #         x
    #     )
    # )
    y_fit = approximate_kl(1 - x)

    error = ((y - y_fit)**2).mean()
    print("MSE:", error)

    error = (y - y_fit)**2 / y**2
    print("Relative MSE:", error.mean())

    error = (np.abs(y - y_fit)).mean()
    print("MAE:", error)

    error = (np.abs(y - y_fit) / y).mean()
    print("Relative MAE:", error)

    plt.plot(x, y, label="Empirical KL")
    plt.plot(x, y_fit, label="Fitted Curve")

    plt.legend()
    plt.grid()

    plt.xlabel("1 - q.mu * p.mu")
    plt.ylabel("KL(q || p)")

    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("power_spherical_kl_fit.png")
    plt.clf()


def test_kl_curve():
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    base_mu = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    base_dist = Golden4dPSDistribution(base_mu)

    d = torch.logspace(-4, math.log10(2.0-1e-4), 10001, base=10.0)
    dots = 1 - d
    thetas = torch.acos(dots)

    mu = torch.stack([torch.zeros_like(thetas), torch.cos(thetas), torch.sin(thetas), torch.zeros_like(thetas)], dim=-1)
    dist = Golden4dPSDistribution(mu)

    with open("power_spherical_kl.pkl", "rb") as f:
        data = pickle.load(f)

    x = (1 - data["dot"])
    y = data["kl"]
    y_approx = dist.kl_to_other(base_dist).cpu().numpy()

    error = ((y - y_approx)**2).mean()
    print("MSE:", error)

    error = (y - y_approx)**2 / y**2
    print("Relative MSE:", error.mean())

    error = (np.abs(y - y_approx)).mean()
    print("MAE:", error)

    error = (np.abs(y - y_approx) / y).mean()
    print("Relative MAE:", error)

    plt.plot(x, y, label="Empirical KL")
    plt.plot(x, y_approx, label="Approximate KL")

    plt.legend()
    plt.grid()

    plt.xlabel("1 - q.mu * p.mu")
    plt.ylabel("KL(q || p)")

    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("power_spherical_kl_test.png")
    plt.clf()


if __name__ == "__main__":
    
    # get_kl_curve()

    # fit_kl_curve()

    test_kl_curve()
    