import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        if isinstance(ctx.scale, dict):
            return grad_output * ctx.scale['value'], None
        return grad_output * ctx.scale, None

def scale_gradient(x, scale):
    return _ScaleGradient.apply(x, scale)


def unsqueeze_to_batch(x, target):
    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(x, target):

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.repeat(
        *(
            [target.shape[i] for i in range(num_unsqueeze)] +
            [1] * (x.dim() - num_unsqueeze)
        )
    )

    return x


class FakeModule(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(
        self,
        fn,
        *args,
        **kwargs
    ):
        return fn(*args, **kwargs)
