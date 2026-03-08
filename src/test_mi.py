import torch

import matplotlib.pyplot as plt
import numpy as np


D = 128
N = 512


@torch.no_grad()
def main():

    mu = torch.randn(N, D)

    dists = torch.cdist(mu, mu, p=2)

    logp = (
        -dists.pow(2) / 2
        # - np.log(np.sqrt(2 * np.pi))
    )
    masked_logp = logp - torch.eye(logp.shape[-1], device=logp.device, dtype=logp.dtype)[None] * 1e9
    
    mi = -torch.logsumexp(masked_logp, dim=-1).mean()

    print(mi.item())


if __name__ == "__main__":
    main()