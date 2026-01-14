
import datasets


DATASET_NAMES = [
    "mcqa-tokenized",
    "OpenScience-tokenized",
    "math-tokenized",
    "chat-tokenized",
    "HuggingFaceTB-finemath-finemath-4plus-tokenized",
    "HuggingFaceFW-fineweb-edu-sample-10BT-tokenized",
    "HuggingFaceFW-fineweb-sample-10BT-tokenized"
]


def main():
    
    ds_list = [datasets.load_dataset(f"aklein4/{name}", split="train") for name in DATASET_NAMES]
    combined_dataset = datasets.concatenate_datasets(ds_list)

    combined_dataset = combined_dataset.shuffle(seed=42)

    combined_dataset.push_to_hub("aklein4/mixed-pretraining-tokenized", split="train")


if __name__ == "__main__":
    main()