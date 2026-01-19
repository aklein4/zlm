import torch
import torch_xla.core.xla_model as xm

xla_device = xm.xla_device()

x = torch.zeros(4, device=xla_device)

x[1] = float('nan')
x[2] = float('inf')
x[3] = float('-inf')

y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

print("Original tensor:", x)
print("After nan_to_num:", y)
