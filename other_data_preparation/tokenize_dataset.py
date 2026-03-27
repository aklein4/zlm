
import numpy as np
from functools import partial
import time

import datasets

from transformers import AutoTokenizer


TOKENIZER_URL = "meta-llama/Llama-2-7b-hf" # "TinyLlama/TinyLlama_v1.1"

DATASET = "allenai/dolma3_longmino_mix-100B-1125"

BS = 128

SAVE_URL = "aklein4/fineweb-edu-SmolLM2"


def tokenize_example(
    example,
    tokenizer: AutoTokenizer=None,
    length: int = None,
):

    input_tokens = tokenizer(
        example["text"],
        add_special_tokens=True,
        padding=False,
        truncation=False,
    ).input_ids

    print(input_tokens)
    exit()

    keep = input_tokens[:, -1] != tokenizer.pad_token_id

    out = {
        "input_ids": [x for x in input_tokens],
        "keep": [x for x in keep],
    }

    return out


def main():

    # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_URL)
    
    done = False
    while not done:
        try:
            data = datasets.load_dataset(
                DATASET, split="train", streaming=False
            )
            done = True
        except: 
            print("Failed to load dataset, retrying in 6 minutes...")
            time.sleep(360)
            print("Retrying now...") 
 
    print("Dataset loaded successfully.") 
    return

    data = data.map(
        partial(tokenize_example, tokenizer=tokenizer, length=None),
        remove_columns=data.column_names,
        batched=True,
        batch_size=BS,
    )
    data = data.filter(
        lambda example: example["keep"],
    )
    data = data.remove_columns("keep")

    return data


if __name__ == "__main__":
    main()
    