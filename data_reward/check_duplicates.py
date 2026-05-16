
from tqdm import tqdm

import datasets


def main():
    
    data = datasets.load_dataset("JinaLeejnl/AlignX", split="train", streaming=True)

    count = 0
    prompt_set = set()
    user_set = set()
    with tqdm(data) as pbar:
        for example in pbar:

            count += 1
            prompt_set.add(example["prompt"].strip())
            user_set.add(tuple(example["Preference Direction"]))

            pbar.set_postfix(count=count, unique_prompts=len(prompt_set), unique_user_preferences=len(user_set))

    print(f"Total examples: {count:_}")
    print(f"Unique prompts: {len(prompt_set):_}")
    print(f"Unique user preferences: {len(user_set):_}")


if __name__ == "__main__":
    main()
