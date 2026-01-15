import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ContinuousEmbedding(nn.Module):

    def __init__(
        self,
        num_frequencies: int,
        embedding_dim: int,
        input_min: float=0.0,
        input_max: float=1.0
    ):
        super().__init__()
        
        self.num_frequencies = num_frequencies
        self.embedding_dim = embedding_dim

        self.input_min = input_min
        self.input_max = input_max

        frequencies = np.pi * torch.arange(
            num_frequencies,
            dtype=torch.float32
        )
        self.register_buffer('frequencies', frequencies, persistent=True)

        self.proj = nn.Linear(
            2 * num_frequencies,
            embedding_dim,
            bias=False
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_min) / (self.input_max - self.input_min)

        theta = x.unsqueeze(-1) * self.frequencies

        emb_sin = torch.sin(theta)
        emb_cos = torch.cos(theta)

        emb = torch.cat([emb_sin, emb_cos], dim=-1)

        return self.proj(emb)
    