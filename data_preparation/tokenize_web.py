
import torch
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt
import json

import datasets
from transformers import AutoTokenizer, GPT2TokenizerFast


TOKENIZER = './tokenizer'

INPUT_LENGTH = 256
OUTPUT_LENGTH = 512
TOTAL_LENGTH = INPUT_LENGTH + OUTPUT_LENGTH


def tokenize_batch(batch, tokenizer: GPT2TokenizerFast=None):

    ids = tokenizer(
        batch["input"],
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=TOTAL_LENGTH
    ).input_ids

    return {
        "ids": ids.astype(np.uint16),
    }


def main():

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    data = datasets.load_dataset(
        "EleutherAI/proof-pile-2",
        "algebraic-stack",
        split="train",
        streaming=True,
    )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=1000,
        collate_fn=partial(tokenize_batch, tokenizer=tokenizer),
        shuffle=False,
        drop_last=False,
    )

    for batch in loader:
        print(batch)
        break


if __name__ == "__main__":
    main()
