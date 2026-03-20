from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer

import math
from utils.torch_utils import newton_schulz, safe_finite


class Amphibian(Optimizer):
    """
    Implements scaled Muon optimizer.

    See: https://arxiv.org/abs/2502.16982

    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        num_iterations (:obj:`int`, `optional`, defaults to 5):
            Number of iterations to perform in the Newton-Schulz method for computing the whitened matrix.
        rms_scale (:obj:`float`, `optional`, defaults to 0.2):
            The scale to apply to the RMS of the muon update.
        nesterov (:obj:`bool`, `optional`, defaults to `True`):
            Whether to use Nesterov-style momentum in the muon update.
        fix_nan (:obj:`bool`, `optional`, defaults to `True`):
            Whether to fix NaN values in gradients and updates by replacing them with zeros.
        state_dtype (:obj:`str`, `optional`, defaults to "float32"):
            The data type to use for the optimizer state (e.g., "float32", "bfloat16", "float16").
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        num_iterations: int = 5,
        rms_scale: float = 0.2,
        nesterov: bool = True,
        state_dtype: str = "float32",
        inner_weight: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            num_iterations=num_iterations, rms_scale=rms_scale, nesterov=nesterov,
            state_dtype=getattr(torch, state_dtype),
            inner_weight=inner_weight,
        )

        super().__init__(params, defaults)


    @torch.no_grad()
    def handle_g1(self):
        for group in self.param_groups:
            for p in group["params"]:
                
                if p.grad is None:
                    continue
                
                self.state[p]["g1"] = p.grad.clone().to(group["state_dtype"])

    
    @torch.no_grad()
    def handle_g2(self):
        for group in self.param_groups:
            for p in group["params"]:
                
                if p.grad is None:
                    continue

                state = self.state[p]
                
                state["g2"] = p.grad.clone().to(group["state_dtype"])
                state["base_p"] = p.clone()

                beta1, beta2 = group["betas"]

                if "step" in state.keys():

                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = 0.1 * (1.0 - beta1) * group["lr"] * math.sqrt(bias_correction2)

                    update = (
                        p.grad /
                        torch.clamp(state["exp_avg_sq"].to(p.dtype), min=group["eps"]**2).sqrt()
                    )

                    p.add_(-step_size * update)


    def adamw_update(self, group, p, grad):

        state = self.state[p]
        state_dtype = group["state_dtype"]

        # State initialization
        if "step" not in state.keys():
            state["step"] = 0
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(grad, dtype=state_dtype)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(grad, dtype=state_dtype)
        else:
            state["exp_avg"] = state["exp_avg"].to(grad.device, dtype=state_dtype)
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad.device, dtype=state_dtype)

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]

        state["step"] += 1

        # Decay the first and second moment running average coefficient
        # In-place operations to update the averages at the same time
        exp_avg.lerp_(
            grad.to(exp_avg.dtype),
            1.0 - beta1
        )
        exp_avg_sq.lerp_(
            grad.to(exp_avg_sq.dtype).pow(2),
            1.0 - beta2
        )

        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

        update = (
            exp_avg.to(p.dtype) /
            torch.clamp(exp_avg_sq.to(p.dtype), min=group["eps"]**2).sqrt()
        )

        if group["weight_decay"] > 0.0:
            p.add_(-group["lr"] * group["weight_decay"] * p)
        p.add_(-step_size * update)

    
    def muon_update(self, group, p, grad):

        state = self.state[p]
        state_dtype = group["state_dtype"]

        # State initialization
        if "exp_avg" not in state.keys():
            state["exp_avg"] = torch.zeros_like(grad, dtype=state_dtype)
        else:
            state["exp_avg"] = state["exp_avg"].to(grad.device, dtype=state_dtype)

        exp_avg = state["exp_avg"]
        beta1, _ = group["betas"]

        # Decay the first and moment running average coefficient
        # In-place operations to update the averages at the same time
        exp_avg.lerp_(
            grad.to(exp_avg.dtype),
            1.0 - beta1
        )

        update = exp_avg
        if group["nesterov"]:
            update = torch.lerp(grad.to(exp_avg.dtype), exp_avg, beta1)

        update = newton_schulz(
            update.to(torch.bfloat16), steps=group["num_iterations"], eps=group["eps"]
        ).to(p.dtype)

        step_size = group["lr"] * group["rms_scale"] * math.sqrt(
            max(grad.shape[0], grad.shape[1])
        )

        if group["weight_decay"] > 0.0:
            p.add_(-group["lr"] * group["weight_decay"] * p)
        p.add_(-step_size * update)


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        g1_mag = 0.0
        g2_mag = 0.0
        g1_H_mag = 0.0
        avg_grad_mag = 0.0
        avg_grad_inner_mag = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                
                # get the grad
                if p.grad is None:
                    continue

                state = self.state[p]

                # restore the original parameters
                p.copy_(state["base_p"])

                # compute the gradient components
                g1_H = p.grad.to(group["state_dtype"])

                avg_grad = state["g1"] + state["g2"]
                avg_grad_inner = g1_H - state["g1"]

                grad = avg_grad + group["inner_weight"] * avg_grad_inner

                g1_mag = g1_mag + state["g1"].pow(2).sum()
                g2_mag = g2_mag + state["g2"].pow(2).sum()
                g1_H_mag = g1_H_mag + g1_H.pow(2).sum()
                avg_grad_mag = avg_grad_mag + avg_grad.pow(2).sum()
                avg_grad_inner_mag = avg_grad_inner_mag + avg_grad_inner.pow(2).sum()

                if True or (
                    grad.dim() != 2 or
                    min(grad.shape) <= 1 or
                    (hasattr(p, "no_muon") and p.no_muon)
                ):
                    self.adamw_update(group, p, grad)
                else:
                    self.muon_update(group, p, grad)

        return {
            "g1_norm": torch.sqrt(g1_mag),
            "g2_norm": torch.sqrt(g2_mag),
            "g1_H_norm": torch.sqrt(g1_H_mag),
            "avg_grad_norm": torch.sqrt(avg_grad_mag),
            "avg_grad_inner_norm": torch.sqrt(avg_grad_inner_mag),
        }
    