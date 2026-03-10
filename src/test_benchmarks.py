import torch

import os

from transformers import AutoTokenizer

from models import load_checkpoint
from models.zlm import ZLMModel
import utils.constants as constants
from evaluation.benchmarks import arc_e, gsm8k


TOKENIZER_PATH = os.path.join(constants.LOCAL_DATA_PATH, "tokenizer")

INPUT_LENGTH = 256
OUTPUT_LENGTH = 512

URL = "aklein4/ZEBRA_muon-1p7b-once"
STEP = 18000


@torch.no_grad()
def main():
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print("TOKENIZER LOADED")

    # bench = arc_e(tokenizer, INPUT_LENGTH, OUTPUT_LENGTH)
    bench = gsm8k(tokenizer, INPUT_LENGTH, OUTPUT_LENGTH)
    loader = torch.utils.data.DataLoader(
        bench,
        batch_size=16,
        shuffle=False,
        collate_fn=bench.collate_fn,
    )
    print("DATA LOADED")

    # model: ZLMModel = load_checkpoint(
    #     URL, STEP,
    #     attention_kernel="gpu_flash_attention",
    # ).to(constants.DEVICE)
    # model.eval()
    # print("MODEL LOADED")

    for batch in loader:

        print(" === INPUT === ")
        print(tokenizer.decode(batch["input_ids"][0].cpu(), skip_special_tokens=False))

        print(" === EXPECTED OUTPUT === ")
        print(tokenizer.decode(batch["output_ids"][0].cpu(), skip_special_tokens=False))

        return

        # with torch.autocast("cuda", torch.bfloat16):
        #     logits = model.get_logits(
        #         batch["input_ids"],
        #         batch["output_ids"],
        #         verbose=True,
        #     )

        # grade = bench.grade(batch, logits)

        # print(grade)


if __name__ == "__main__":
    main()