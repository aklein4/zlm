
import os
import zstandard as zstd
import json
import pandas as pd
from tqdm import tqdm

import datasets
import huggingface_hub as hf


DATA_URL = "EleutherAI/proof-pile-2"
SUBSETS = [
    # "algebraic-stack",
    "arxiv",
    "open-web-math"
]
SPLITS = [
    "train",
    "validation",
    "test"
]

LOCAL_DIR = "./local_data/proof-pile-2"

OUT_URL = 'aklein4/proof-pile-2-fixed'


def download_data(
    url: str,
    subset: str,
    split: str,
):
    hf.snapshot_download(
        repo_id=url,
        repo_type="dataset",
        allow_patterns=[f"{subset}/{split}/*"],
        local_dir=LOCAL_DIR,
    )

    return os.path.join(LOCAL_DIR, subset, split)


def format_data(
    url: str,
    subset: str ,
    split: str,
):
    
    # download the data
    folder = download_data(url, subset, split)

    # get all files in the local dir
    data_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".zst")
    ]

    # read all of the .jsonl.zst files
    datas = []
    for file_path in tqdm(data_files):
        
        examples = []

        with zstd.open(open(file_path, "rb"), "rt", encoding="utf-8") as f: 
            for x in f.readlines():
                examples.append(json.loads(x))

        df = pd.DataFrame(examples)
        datas.append(datasets.Dataset.from_pandas(df))

    dataset = datasets.concatenate_datasets(datas)    
    dataset = dataset.shuffle(seed=42)

    dataset.push_to_hub(
        OUT_URL,
        config_name=subset,
        split=split,
        private=False
    )


def main():

    for subset in SUBSETS:
        for split in SPLITS:

            format_data(
                DATA_URL,
                subset,
                split,
            )


if __name__ == "__main__":
    main()