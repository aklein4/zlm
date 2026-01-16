
import numpy as np
from functools import partial

import datasets

from transformers import AutoTokenizer


TOKENIZER_URL = "TinyLlama/TinyLlama_v1.1"

DATASETS = [
    ("HuggingFaceFW/fineweb", "sample-10BT"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
]

BS = 1024

LENGTHS = [
    1024,
    2048
]

SAVE_URL = "aklein4/fineweb-w-edu-tinyllama"


def tokenize_example(
    example,
    tokenizer: AutoTokenizer=None,
    length: int = None,
):

    input_tokens = tokenizer(
        example["text"],
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=length,
    ).input_ids.astype(np.uint16)

    keep = input_tokens[:, -1] != tokenizer.pad_token_id

    out = {
        "input_ids": [x for x in input_tokens],
        "keep": [x for x in keep],
    }

    return out


def tokenize_dataset(
    data_url: tuple[str, str],
    tokenizer: AutoTokenizer,
    length: int,
):
    
    data = datasets.load_dataset(*data_url, split="train", streaming=False)

    data = data.map(
        partial(tokenize_example, tokenizer=tokenizer, length=length),
        remove_columns=data.column_names,
        batched=True,
        batch_size=BS,
    )
    data = data.filter(
        lambda example: example["keep"],
    )
    data = data.remove_columns("keep")

    return data


def main():

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_URL)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenizer.push_to_hub(
        SAVE_URL,
        private=False,
        repo_type="dataset",
    )

    for length in LENGTHS:

        datas = []
        for data_url in DATASETS:

            d = tokenize_dataset(data_url, tokenizer, length)
            datas.append(d)

        full_data = datasets.concatenate_datasets(datas)
        full_data = full_data.shuffle(seed=42)

        full_data.push_to_hub(
            SAVE_URL, 
            config_name=f"length_{length}",
            private=False,
        )

        del full_data


if __name__ == "__main__":
    main()