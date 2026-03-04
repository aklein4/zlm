import torch

import numpy as np
import matplotlib.pyplot as plt

from transformers import LlamaForCausalLM


URL = 'HuggingFaceTB/SmolLM2-1.7B'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def main():
    
    model = LlamaForCausalLM.from_pretrained(URL).to(DEVICE)

    w = model.model.embed_tokens.weight.data
    w -= w.mean(0, keepdim=True)

    cov = (w.T @ w) / w.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)

    std = w.std(0)
    sample = torch.randn_like(w) * std[None]
    # dist = torch.distributions.MultivariateNormal(
    #     loc=torch.zeros(w.shape[1], device=DEVICE),
    #     covariance_matrix=cov,
    # )
    # sample = dist.sample((w.shape[0],))
    
    sampled_var = (sample @ eigvecs).var(0)

    plt.plot(eigvals.flip(0).cpu().numpy(), label='True')
    plt.plot(sampled_var.flip(0).cpu().numpy(), label='Sampled')
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.savefig('embed_eigvals.png')


if __name__ == '__main__':
    main()
