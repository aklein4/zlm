
import numpy as np
from functools import partial

import datasets
from transformers import AutoTokenizer, GPT2TokenizerFast


TOKENIZER = './tokenizer'

DATASETS = [
    ("HuggingFaceFW/fineweb", "sample-10BT"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    ("HuggingFaceTB/finemath", "finemath-4plus")
]

INPUT_LENGTH = 256
OUTPUT_LENGTH = 512
TOTAL_LENGTH = INPUT_LENGTH + OUTPUT_LENGTH

OUTPUT_REPO = "aklein4/compilation-SmolLM2"


def tokenize_batch(batch, tokenizer: GPT2TokenizerFast=None):

    input_tokens = tokenizer(
        batch["text"],
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
    }


def split_example(example, source: str=None):

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
        "source": [source for _ in input_ids],
        "kind": ["web_text" for _ in input_ids],
        "format": ["raw_text" for _ in input_ids],
        "input_ids": input_ids,
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "num_total_tokens": num_total_tokens,
    }


def tokenize_dataset(
    data_path,
    tokenizer
):

    data = datasets.load_dataset(
        *data_path,
        split="train",
        streaming=False,
    )

    data = data.map(
        partial(tokenize_batch, tokenizer=tokenizer),
        remove_columns=data.column_names,
        batched=True,
        batch_size=1024,
    )
    data = data.filter(
        lambda example: example["keep"],
    )

    data = data.map(
        partial(split_example, source=f"{data_path[0]}/{data_path[1]}"),
        remove_columns=data.column_names,
        batched=True,
        batch_size=1024,
    )

    return data


def main():

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    for data_path in DATASETS:

        data = tokenize_dataset(
            data_path,
            tokenizer,
        )
        
        data = data.shuffle(seed=42)

        data.push_to_hub(
            OUTPUT_REPO,
            config_name=f"{data_path[0]/data_path[1]}".replace("/", "--"),
            private=False,
        )


if __name__ == "__main__":
    main()