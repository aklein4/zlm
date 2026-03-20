import torch
import torch.nn.functional as F

import math

B = 100
O = 25
I = 16


def main():

    G = torch.randn(B, O, I)

    # adam-like preconditioning
    # denominator is sqrt(E[mean(update)^2]) with v/N accounting for the expectation
    # m = G.mean(dim=0, keepdim=True)
    # v = G.var(dim=0, keepdim=True)
    # s = (m.pow(2) + v/B).sqrt()
    s = G.pow(2).mean(dim=0, keepdim=True).sqrt()
    P = 1 / torch.clamp(s, min=1e-6)

    # elements on ~1
    updates = P * G

    # L2 normalized each update
    directions = F.normalize(updates, dim=[-2, -1])
    
    # E_{i!=j}[direction_i * update_j]
    l_raw = torch.einsum(
        "boi,boi->",
        updates.sum(0, keepdim=True) - updates, # B - 1
        directions # B
    ) / (B * (B - 1))
    print(l_raw)

    sims = (updates[None] * directions[:, None]).sum(dim=[-2, -1])
    sims = sims * (1 - torch.eye(B, device=G.device))
    print(sims.sum() / (B * (B - 1)))

if __name__ == "__main__":
    main()
