

import datasets
import os
from transformers import LlamaTokenizerFast


OUT_DIR = "examples"
NUM_EXAMPLES = 10

TOKENIZER = './tokenizer'

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
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = LlamaTokenizerFast.from_pretrained(TOKENIZER)

    for dataset_name in DATASET_NAMES:
        
        dataset = datasets.load_dataset(f"aklein4/{dataset_name}", split="train", streaming=True)
        with open(os.path.join(OUT_DIR, f"{dataset_name}.txt"), "w") as f:

            for i, example in enumerate(dataset):
                if i >= NUM_EXAMPLES:
                    break

                input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=False)
                output_text = tokenizer.decode(example["output_ids"], skip_special_tokens=False)

                input_text = f"[[[{input_text}]]]"
                output_text = f"[[[{output_text}]]]"

                f.write(f"\n ===== Example {i} ===== \n")
                f.write("\n - Input - \n")
                f.write(f"\n{input_text}\n")
                f.write("\n - Output - \n")
                f.write(f"\n{output_text}\n")

        print(f"Saved examples from {dataset_name} to {OUT_DIR}!")


if __name__ == "__main__":
    main()