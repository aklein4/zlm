import torch

import numpy as np
from functools import partial

import datasets
from transformers import LlamaTokenizerFast
import matplotlib.pyplot as plt

from constants import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH


TOKENIZER = './tokenizer'


INPUT_REPO = "aklein4/chat-formatted"
OUTPUT_REPO = "aklein4/chat-tokenized"


def tokenize_example(example, tokenizer: LlamaTokenizerFast=None):

    input_tokens = tokenizer(
        example["input"],
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH+10
    ).input_ids
    output_tokens = tokenizer(
        example["output"],
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_OUTPUT_LENGTH+10
    ).input_ids

    # input_text = tokenizer.decode(input_tokens, skip_special_tokens=False)
    # output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

    assert np.max(input_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    assert np.max(output_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    input_tokens = input_tokens.astype(np.uint16)
    output_tokens = output_tokens.astype(np.uint16)

    input_tokens = [
        t[t != tokenizer.pad_token_id][:-1] for t in input_tokens # Exclude pad and EOS
    ]
    output_tokens = [
        t[t != tokenizer.pad_token_id][1:] for t in output_tokens # Exclude pad and BOS
    ]

    return {
        "input_ids": input_tokens,
        "output_ids": output_tokens,
        "num_input_tokens": [len(t) for t in input_tokens],
        "num_output_tokens": [len(t) for t in output_tokens],
    }


def main():

    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(TOKENIZER)

    dataset = datasets.load_dataset(INPUT_REPO, split="train")

    dataset = dataset.map(
        partial(tokenize_example, tokenizer=tokenizer),
        remove_columns=["input", "output"],
        batched=True,
        batch_size=1000,
    )
    dataset = dataset.filter(
        lambda x: x["num_input_tokens"] <= MAX_INPUT_LENGTH and x["num_output_tokens"] <= MAX_OUTPUT_LENGTH
    )

    dataset.push_to_hub(OUTPUT_REPO, split="train")

    input_lengths = np.array(dataset["num_input_tokens"])
    output_lengths = np.array(dataset["num_output_tokens"])
    total_lengths = np.array([i + o for i, o in zip(input_lengths, output_lengths)])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].hist(total_lengths, bins=100)
    ax[0].set_title("Total Token Lengths")
    ax[0].grid(True)

    ax[1].hist(input_lengths, bins=100)
    ax[1].set_title("Input Token Lengths")
    ax[1].grid(True)

    ax[2].hist(output_lengths, bins=100)
    ax[2].set_title("Output Token Lengths")
    ax[2].grid(True)

    plt.tight_layout()
    plt.savefig("chat_length_distributions.png")


if __name__ == "__main__":

    # dataset = datasets.load_dataset(OUTPUT_REPO, split="train")

    # input_lengths = np.array(dataset["num_input_tokens"])
    # output_lengths = np.array(dataset["num_output_tokens"])
    # total_lengths = np.array([i + o for i, o in zip(input_lengths, output_lengths)])

    # fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # ax[0].hist(total_lengths, bins=100)
    # ax[0].set_title("Total Token Lengths")
    # ax[0].grid(True)

    # ax[1].hist(input_lengths, bins=100)
    # ax[1].set_title("Input Token Lengths")
    # ax[1].grid(True)

    # ax[2].hist(output_lengths, bins=100)
    # ax[2].set_title("Output Token Lengths")
    # ax[2].grid(True)

    # plt.tight_layout()
    # plt.savefig("chat_length_distributions.png")

    main()
