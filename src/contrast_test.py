import torch

import numpy as np
import matplotlib.pyplot as plt


def mutual_information(mu):

    dists = torch.cdist(mu, mu, p=2)

    scores = -dists.pow(2) / (2 * mu.shape[-1])
    masked_scores = scores - torch.eye(scores.shape[-1], device=scores.device, dtype=scores.dtype) * 1e9

    mi = -torch.logsumexp(masked_scores - np.log(scores.shape[-1] - 1), dim=-1).mean()

    return mi


def main():
    
    mu1 = torch.randn(512, 64)
    mu2 = torch.randn(512, 64)

    mi = mutual_information(mu1)

    print("Mutual Information:", mi.item())


if __name__ == "__main__":
    main()