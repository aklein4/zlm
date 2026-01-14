import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SeqToSeqCollator:

    def __init__(
        self,
        input_length: int,
        output_length: int,
        pad_token_id: int,
        vocab_size: int,
    ):
        self.input_length = input_length
        self.output_length = output_length

        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

    
    def __call__(
        self,
        batch,
    ):
        return {
            "input_ids": handle_ids(
                batch,
                "input_ids",
                self.input_length,
                self.pad_token_id,
                self.vocab_size
            ),
            "output_ids": handle_ids(
                batch,
                "output_ids",
                self.output_length,
                self.pad_token_id,
                self.vocab_size
            )
        }


def handle_ids(batch, key, sequence_length, pad_token_id, vocab_size):

    input_ids = []
    for x in batch:

        input_ids.append(
            torch.tensor(x[key]).long().flatten()
        )

    # pad to length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id,   
    )
    input_ids = input_ids[:, :sequence_length]
    
    # pad to sequence length
    pad = torch.full(
        (input_ids.shape[0], sequence_length - input_ids.shape[1]),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device
    )
    input_ids = torch.cat(
        [
            input_ids,
            pad
        ],
        dim=1
    )

    input_ids = torch.clip(input_ids, 0, vocab_size - 1)

    return input_ids
