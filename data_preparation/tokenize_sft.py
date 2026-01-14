
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt
import json

import datasets
from transformers import AutoTokenizer, GPT2TokenizerFast


STAT_DIR = "data_statistics"

LOG_FILE = "tokenization_log.txt"

MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 1024

TOKENIZER = './tokenizer'

INPUT_REPO = "aklein4/raw-compilation"
OUTPUT_REPO = "aklein4/compilation-SmolLM2"

SKIP_SUBSETS = [
    "nvidia--Llama-Nemotron-Post-Training-Dataset--SFT",
]

DEBUG = False


def tokenize_example(example, tokenizer: GPT2TokenizerFast=None):

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

    # assert np.max(input_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    # assert np.max(output_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    input_tokens = input_tokens.astype(np.uint16)
    output_tokens = output_tokens.astype(np.uint16)

    input_mask = input_tokens != tokenizer.pad_token_id
    input_tokens = [
        t[input_mask[i]] for i, t in enumerate(input_tokens)
    ]
    output_mask = output_tokens != tokenizer.pad_token_id
    output_tokens = [
        t[output_mask[i]] for i, t in enumerate(output_tokens)
    ]

    output_tokens = [
        (np.append(t, tokenizer.eos_token_id) if t[-1] != tokenizer.eos_token_id else t) for t in output_tokens
    ]

    out = {
        "input_ids": input_tokens,
        "output_ids": output_tokens,
        "num_input_tokens": [len(t) for t in input_tokens],
        "num_output_tokens": [len(t) for t in output_tokens],
    }
    out["num_total_tokens"] = [
        i + o for i, o in zip(out["num_input_tokens"], out["num_output_tokens"])
    ]

    return out


def tokenize_subset(
    url: str,
    subset: str,
    tokenizer: GPT2TokenizerFast,
):
    
    data = datasets.load_dataset(url, subset, split="train")
    
    data = data.map(
        partial(tokenize_example, tokenizer=tokenizer),
        batched=True,
        batch_size=1000,
        load_from_cache_file=False,
    )
    data = data.filter(
        lambda x: x["num_input_tokens"] <= MAX_INPUT_LENGTH and x["num_output_tokens"] <= MAX_OUTPUT_LENGTH,
        load_from_cache_file=False,
    )

    return data


def save_subset(
    data: datasets.Dataset,
    subset: str,
):
    
    path = os.path.join(STAT_DIR, subset)
    os.makedirs(path, exist_ok=True)

    input_lengths = np.array(data["num_input_tokens"])
    output_lengths = np.array(data["num_output_tokens"])
    total_lengths = np.array(data["num_total_tokens"])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].hist(total_lengths, bins=100)
    ax[0].set_title("Total Token Lengths")
    ax[0].grid(True)
    ax[0].set_xlim(-1, MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH + 10)

    ax[1].hist(input_lengths, bins=100)
    ax[1].set_title("Input Token Lengths")
    ax[1].grid(True)
    ax[1].set_xlim(-1, MAX_INPUT_LENGTH+10)

    ax[2].hist(output_lengths, bins=100)
    ax[2].set_title("Output Token Lengths")
    ax[2].grid(True)
    ax[2].set_xlim(-1, MAX_OUTPUT_LENGTH+10)

    plt.suptitle(f"Token Length Distributions for {subset}")
    plt.tight_layout()

    plt.savefig(os.path.join(path, "length_distributions.png"))

    stats = {
        "num_examples": len(data),
        "total_tokens": np.sum(total_lengths).item(),
        "input_tokens": np.sum(input_lengths).item(),
        "output_tokens": np.sum(output_lengths).item(),
        "mean_total_length": round(np.mean(total_lengths).item(), 2),
        "mean_input_length": round(np.mean(input_lengths).item(), 2),
        "mean_output_length": round(np.mean(output_lengths).item(), 2),
    }
    with open(os.path.join(path, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    data.push_to_hub(
        OUTPUT_REPO,
        config_name=subset,
        private=False,
        split="train",
    )

    return len(data), np.sum(total_lengths).item()


def main():

    os.makedirs(STAT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    with open(LOG_FILE, "w") as f:
        f.write("")

    subsets = datasets.get_dataset_config_names(INPUT_REPO)
    for s in SKIP_SUBSETS:
        if s in subsets:
            subsets.remove(s)
        else:
            raise ValueError(f"Removal subset {s} not found in subset list.")

    total_examples = 0
    total_tokens = 0
    for i, subset in enumerate(subsets):

        print("")
        print(f"[{i+1}/{len(subsets)}] Tokenizing subset: {subset}")
        print("")

        try:
            data = tokenize_subset(
                INPUT_REPO,
                subset,
                tokenizer
            )
            ex, tok = save_subset(
                data,
                subset
            )

            total_examples += ex
            total_tokens += tok

        except Exception as e:
            if isinstance(e, KeyboardInterrupt) or DEBUG:
                raise e

            with open(LOG_FILE, "a") as f:
                f.write(f"\n[{i+1}/{len(subsets)}] {subset}: FAIL")
            continue

        with open(LOG_FILE, "a") as f:
            f.write(f"\n[{i+1}/{len(subsets)}] {subset}: SUCCESS ({ex:_} examples, {tok:_} tokens)")

    with open(LOG_FILE, "a") as f:
        f.write(f"\n\nTotal examples: {total_examples:_}\n")
        f.write(f"Total tokens: {total_tokens:_}\n")


if __name__ == "__main__":
    main()
