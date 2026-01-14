
import datasets

MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 512

INPUT_REPO = "aklein4/compilation-SmolLM2"
OUTPUT_REPO = "aklein4/seq2seq-mixed-pretraining-SmolLM2"


def main():
    
    dss = []
    for subset in datasets.get_dataset_config_names(INPUT_REPO):

        ds = datasets.load_dataset(
            INPUT_REPO,
            subset,
            split="train",
        )
        ds = ds.filter(
            lambda example: example["num_input_tokens"] <= MAX_INPUT_LENGTH and
                            example["num_output_tokens"] <= MAX_OUTPUT_LENGTH
        )
        dss.append(ds)

    combined_ds = datasets.concatenate_datasets(dss)
    combined_ds = combined_ds.shuffle(seed=42)

    combined_ds.push_to_hub(
        OUTPUT_REPO,
        "sft_text",
        private=False,
        split="train",
    )

    combined_ds = combined_ds.remove_columns(["input", "output"])
    combined_ds.push_to_hub(
        OUTPUT_REPO,
        "sft_no_text",
        private=False,
        split="train",
    )

    print(f"\nCombined dataset has {len(combined_ds)} examples.\n")


if __name__ == "__main__":
    main()