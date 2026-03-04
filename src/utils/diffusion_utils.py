import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

import utils.constants as constants
from utils.torch_utils import unsqueeze_to_channel


class DiffusionScheduler(nn.Module):

    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__()

        self.min_timestep = config.minimum_diffusion_timestep
        self.num_timesteps = config.num_diffusion_timesteps

        timesteps = torch.linspace(
            self.min_timestep,
            1.0,
            self.num_timesteps,
        )
        self.register_buffer(
            "timesteps", timesteps, persistent=True
        )

        a = torch.sqrt(1.0 - timesteps)
        self.register_buffer(
            "a", a, persistent=True
        )

        b = torch.sqrt(1 - a.pow(2))
        self.register_buffer(
            "b", b, persistent=True
        )


    def snr(
        self,
        timestep: torch.LongTensor,
    ) -> torch.FloatTensor:
        # https://arxiv.org/pdf/2107.00630 equation 2
        return self.a[timestep].pow(2) / self.b[timestep].pow(2)


    def alpha_t_s(
        self,
        timestep: torch.LongTensor,
    ) -> torch.FloatTensor:
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)
        # https://arxiv.org/pdf/2107.00630 equation 21
        return self.a[timestep] / self.a[timestep - 1]

    
    def var_t_s(
        self,
        timestep: torch.LongTensor,
    ) -> torch.FloatTensor:
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)
        # https://arxiv.org/pdf/2107.00630 equation 22
        return self.b[timestep].pow(2) - (
            self.alpha_t_s(timestep).pow(2) *
            self.b[timestep - 1].pow(2)
        )

    
    def kl(
        self,
        x_0: torch.FloatTensor,
        timestep: torch.LongTensor,
        pred_x_0: torch.FloatTensor,
        dim: int=None,
        keepdim: bool=False,
    ) -> torch.FloatTensor:
        """
        https://arxiv.org/pdf/2107.00630 equation 48

        KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t)) = 
        0.5 * (SNR(t-1) - SNR(t)) * || pred_x_0 - x_0 ||^2
        """
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)
        timestep = unsqueeze_to_channel(timestep, x_0)

        snr_t = self.snr(timestep)
        snr_t_minus_1 = self.snr(timestep - 1)

        kl = 0.5 * (snr_t_minus_1 - snr_t) * F.mse_loss(
            pred_x_0, x_0, reduction="none"
        )

        if dim is not None:
            return kl.sum(dim=dim, keepdim=keepdim)
        return kl


    def add_noise(
        self,
        x_0: torch.FloatTensor,
        timestep: torch.LongTensor,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # https://arxiv.org/pdf/2107.00630 equation 1
        return (
            unsqueeze_to_channel(self.a[timestep], x_0) * x_0 +
            unsqueeze_to_channel(self.b[timestep], noise) * noise
        )


    def step(
        self,
        x_t: torch.FloatTensor,
        timestep: torch.LongTensor,
        pred_x_0: torch.FloatTensor,
        noise: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)

        if noise is None:
            noise = torch.randn_like(x_t)

        timestep = unsqueeze_to_channel(timestep, x_t)
        s = timestep - 1
        
        # https://arxiv.org/pdf/2107.00630 equation 26
        mu = (
            (self.alpha_t_s(timestep) * self.b[s].pow(2) / self.b[timestep].pow(2)) * x_t +
            (self.a[s] * self.var_t_s(timestep) / self.b[timestep].pow(2)) * pred_x_0
        )

        # https://arxiv.org/pdf/2107.00630 equation 25
        var = self.var_t_s(timestep) * self.b[s].pow(2) / self.b[timestep].pow(2)

        return mu + torch.sqrt(var) * noise

    
    def ddim_step(
        self,
        x_t: torch.FloatTensor,
        timestep: torch.LongTensor,
        pred_x_0: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not constants.XLA_AVAILABLE:
            assert torch.all(timestep > 0)

        timestep = unsqueeze_to_channel(timestep, x_t)
        
        implied_noise = (
            (x_t - unsqueeze_to_channel(self.a[timestep], pred_x_0) * pred_x_0) /
            unsqueeze_to_channel(self.b[timestep], x_t)
        )

        return self.add_noise(
            pred_x_0,
            timestep - 1,
            implied_noise
        )
    