
import datasets
from tqdm import tqdm
import os

from transformers import AutoTokenizer


URL = "aklein4/compilation-SmolLM2"
OUTPUT_DIR = "data_statistics"

TOKENIZER_URL = "./tokenizer"

NUM_EXAMPLES = 10


def main():
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_URL)

    subsets = list(datasets.get_dataset_config_names(URL))
    for subset in tqdm(subsets):

        subset_dir = os.path.join(OUTPUT_DIR, subset)
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "examples.txt"), "w", encoding="utf-8") as f:

            dataset = datasets.load_dataset(URL, subset, split="train", streaming=True)

            for i, example in enumerate(dataset):
                f.write(f"\n\n ========== Example {i} ==========")

                f.write("\n\n --- Input --- \n\n")
                f.write("[[["+tokenizer.decode(example["input_ids"], skip_special_tokens=False)+"]]]")

                f.write("\n\n --- Output --- \n\n")
                f.write("[[["+tokenizer.decode(example["output_ids"], skip_special_tokens=False)+"]]]")

                if i + 1 >= NUM_EXAMPLES:
                    break


if __name__ == "__main__":
    main()
    