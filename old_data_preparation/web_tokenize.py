import torch

import numpy as np
from functools import partial
import argparse

import datasets
from transformers import LlamaTokenizerFast

from constants import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH


TOKENIZER = './tokenizer'


def tokenize_example(example, tokenizer: LlamaTokenizerFast=None, source="", column=""):

    tokens = tokenizer(
        example[column],
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH
    ).input_ids

    assert np.max(tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    tokens = tokens.astype(np.uint16)

    input_tokens = []
    output_tokens = []
    for t in tokens:
        t = t[t != tokenizer.pad_token_id][:MAX_INPUT_LENGTH+MAX_OUTPUT_LENGTH]  # Exclude padding tokens

        if len(t) >= MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH - 50:
            input_tokens.append(t[:MAX_INPUT_LENGTH])
            output_tokens.append(t[MAX_INPUT_LENGTH:])

        elif len(t) < 50:
            input_tokens.append(None)
            output_tokens.append(None)
        
        else:
            split = np.random.randint(
                max(len(t) - MAX_OUTPUT_LENGTH, 20),
                min(len(t) - 20, MAX_INPUT_LENGTH) + 1
            )
            
            t_in = t[:split]
            t_out = t[split:]
            if len(t_in) > MAX_INPUT_LENGTH or len(t_out) > MAX_OUTPUT_LENGTH:
                t_in = None
                t_out = None
            
            input_tokens.append(t_in)
            output_tokens.append(t_out)

    assert len(input_tokens) == len(output_tokens), "Input and output token lists must be of the same length."

    return {
        "source": [source] * len(input_tokens),
        "input_ids": input_tokens,
        "output_ids": output_tokens,
        "num_input_tokens": [(len(t) if t is not None else 0) for t in input_tokens],
        "num_output_tokens": [(len(t) if t is not None else 0) for t in output_tokens],
    }


def main(args):

    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(TOKENIZER)

    if args.subset != "":
        dataset = datasets.load_dataset(args.repo, args.subset, split="train")
        source = f"{args.repo}/{args.subset}"
    else:
        dataset = datasets.load_dataset(args.repo, split="train")
        source = args.repo

    dataset = dataset.map(
        partial(tokenize_example, tokenizer=tokenizer, source=source, column=args.column),
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=1000,
    )
    dataset = dataset.filter(
        lambda x: (
            x["input_ids"] is not None and
            len(x["input_ids"]) <= MAX_INPUT_LENGTH and
            len(x["output_ids"]) <= MAX_OUTPUT_LENGTH
        )
    )
    dataset = dataset.shuffle(seed=42)

    if args.subset != "":
        out_repo = f"aklein4/{args.repo.replace("/", "-")}-{args.subset}-tokenized"
    else:
        out_repo = f"aklein4/{args.repo.replace("/", "-")}-tokenized"
    dataset.push_to_hub(out_repo, split="train")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tokenize chat dataset")
    parser.add_argument("--repo", type=str, required=True, help="Input dataset repository")
    parser.add_argument("--subset", type=str, default="", help="Subset of the dataset to use")
    parser.add_argument("--column", type=str, default="text", help="Column to tokenize")
    args = parser.parse_args()

    main(args)
