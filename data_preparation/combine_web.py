
import numpy as np

import datasets


INPUT_REPO = "aklein4/compilation-SmolLM2"
OUTPUT_REPO = "aklein4/seq2seq-mixed-pretraining-SmolLM2"

WEB_SUBSETS = [
    "EleutherAI--proof-pile-2--algebraic-stack",
    "HuggingFaceFW--fineweb--sample-10BT",
    "HuggingFaceFW--fineweb-edu--sample-10BT",
    "HuggingFaceTB--finemath--finemath-4plus",
    "HuggingFaceTB--smollm-corpus--cosmopedia-v2",
]


def main():
    
    # dss = []
    # for subset in WEB_SUBSETS:

    #     ds = datasets.load_dataset(
    #         INPUT_REPO,
    #         subset,
    #         split="train",
    #         streaming=False,
    #     )
    #     dss.append(ds)

    # web_ds = datasets.concatenate_datasets(dss)
    # web_ds = web_ds.shuffle(seed=42)

    # web_ds.push_to_hub(
    #     OUTPUT_REPO,
    #     "web",
    #     private=False,
    #     split="train",
    # )

    web_ds = datasets.load_dataset(
        OUTPUT_REPO,
        "web",
        split="train",
        streaming=False,
    )

    sft_ds = datasets.load_dataset(
        OUTPUT_REPO,
        "sft_no_text",
        split="train",
        streaming=False,
    )

    web_ds = web_ds.cast_column("input_ids", sft_ds.features["input_ids"])
    web_ds = web_ds.cast_column("output_ids", sft_ds.features["output_ids"])

    ds = datasets.concatenate_datasets([web_ds, sft_ds])
    ds = ds.shuffle(seed=42)

    # 90% train, 5% test, 5% valid
    train_testvalid = ds.train_test_split(test_size=0.1)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    split_ds = datasets.DatasetDict(
        {
            'train': train_testvalid['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        }
    )

    split_ds.push_to_hub(
        OUTPUT_REPO,
        "all",
        private=False,
    )


if __name__ == "__main__": 
    main()