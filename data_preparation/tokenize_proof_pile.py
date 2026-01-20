
import numpy as np
from functools import partial

import datasets
from transformers import GPT2TokenizerFast


INPUT_LENGTH = 256
OUTPUT_LENGTH = 512
TOTAL_LENGTH = INPUT_LENGTH + OUTPUT_LENGTH

TOKENIZER = './tokenizer'

INPUT_REPO = "aklein4/proof-pile-2-fixed"
SUBSET = "algebraic-stack"

OUTPUT_REPO = "aklein4/compilation-SmolLM2"
OUTPUT_SUBSET = "proof-pile"


def tokenize_example(example, tokenizer: GPT2TokenizerFast=None):

    text = []
    formats = []
    for t, meta in zip(example["text"], example["meta"]):

        if "lang" in meta and meta["lang"] is not None:
            lang = meta["lang"].lower()
            t = f"```{lang}\n{t}\n```"
            text.append(t)
            formats.append("code_block")

        else:
            text.append(t)
            formats.append("raw_code")

    input_tokens = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=TOTAL_LENGTH
    ).input_ids.astype(np.uint16)

    keep = input_tokens[:, -1] != tokenizer.pad_token_id

    return {
        "input_ids": [x for x in input_tokens],
        "keep": [x for x in keep],
        "format": formats,
    }


def split_example(example):

    input_ids = [
        x[:INPUT_LENGTH] for x in example["input_ids"]
    ]
    output_ids = [
        x[INPUT_LENGTH:] for x in example["input_ids"]
    ]

    num_input_tokens = [
        len(x) for x in input_ids
    ]
    num_output_tokens = [
        len(x) for x in output_ids
    ]
    num_total_tokens = [
        ni + no for ni, no in zip(num_input_tokens, num_output_tokens)
    ]

    return {
        "source": f"{INPUT_REPO}/{SUBSET}",
        "kind": "web_code",
        "format": example["format"],
        "input_ids": input_ids,
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "num_total_tokens": num_total_tokens,
    }


def main():

    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER)

    data = datasets.load_dataset(
        INPUT_REPO,
        SUBSET,
        split="train",
        streaming=False,
    )

    data = data.map(
        partial(tokenize_example, tokenizer=tokenizer),
        remove_columns=data.column_names,
        batched=True,
        batch_size=1024,
    )
    data = data.filter(
        lambda example: example["keep"],
    )

    data = data.map(
        split_example,
        remove_columns=data.column_names,
        batched=True,
        batch_size=1024,
    )

    data = data.shuffle(seed=42)

    data.push_to_hub(
        OUTPUT_REPO,
        OUTPUT_SUBSET,
        private=False,
        split="train",
    )

    print(f"\nTokenized dataset has {len(data)} examples.\n")


if __name__ == "__main__":
    main()
