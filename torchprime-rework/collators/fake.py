import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class FakeCollator:

    def __init__(
        self,
        sequence_length: int,
        vocab_size: int,
    ):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

    
    def __call__(
        self,
        batch,
    ):
        bs = len(batch)

        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(bs, self.sequence_length),
            dtype=torch.long,
        )
