import torch

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import datasets

from models import load_checkpoint
from models.llama import LlamaForCausalLM
from models.ittt import ItttModel
from collators.simple import SimpleCollator
import utils.constants as constants
from utils.loss_utils import lm_loss_fn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LM_URL = 'aklein4/iTTT-TPU_attn-baseline-1b'
LM_STEP = 2000

ITTT_URL = 'aklein4/iTTT-TPU_alpha-1b'
ITTT_STEP = 500

DATA_URL = "aklein4/longattn-SmolLM2"

NUM_EXAMPLES = 512
BS = 4


def main():
    
    print("Loading data...")
    data = datasets.load_dataset(DATA_URL, split='train', streaming=True)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BS,
        collate_fn=SimpleCollator(),
    )

    print("Loading model...")
    lm_model: LlamaForCausalLM = load_checkpoint(
        LM_URL,
        LM_STEP,
        attention_kernel="gpu_flash_attention",
    ).to(DEVICE)
    # lm_model.lm_head.weight = lm_model.lm_head.weight.cpu()

    print("Loading ItttModel...")
    ittt_model: ItttModel = load_checkpoint(
        ITTT_URL,
        ITTT_STEP,
        attention_kernel="gpu_flash_attention",
    ).to(DEVICE)

    print("Running models...")
    lm_losses = []
    ittt_losses = []
    for i, batch in tqdm(enumerate(loader), total=NUM_EXAMPLES // BS):

        input_ids = batch["input_ids"].to(DEVICE)

        logits = ittt_model.compute_logits(
            input_ids, verbose=True,
        )   
        loss = lm_loss_fn(
            logits, input_ids,
            shift_logits=False,
            ignore_index=ittt_model.config.pad_token_id,
            reduction='none',
        )
        ittt_losses.append(loss.cpu())
        del logits

        with torch.no_grad():
            with torch.autocast(str(constants.DEVICE), torch.bfloat16):

                logits = lm_model.forward(
                    input_ids,
                    shift_states=True,
                )[0]

            loss = lm_loss_fn(
                logits, input_ids,
                shift_logits=False,
                ignore_index=lm_model.config.pad_token_id,
                reduction='none',
            )

        lm_losses.append(loss.cpu())
        del logits

        if len(ittt_losses) >= NUM_EXAMPLES // BS:
            break
    
    lm_losses = torch.cat(lm_losses, dim=0)
    torch.save(
        lm_losses,
        os.path.join(constants.LOCAL_DATA_PATH, "lm_losses_for_comparison.pt")
    )
    
    ittt_losses = torch.cat(ittt_losses, dim=0)
    torch.save(
        ittt_losses,
        os.path.join(constants.LOCAL_DATA_PATH, "ittt_losses_for_comparison.pt")
    )


def nan_mean(x):
    mask = np.isfinite(x)
    s = mask.sum(0)
    x = np.where(mask, x, np.zeros_like(x))
    return x.sum(0) / s


def analyze_results():

    lm_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "lm_losses_for_comparison.pt")).float().numpy()
    ittt_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "ittt_losses_for_comparison.pt")).float().numpy()

    df = pd.DataFrame({
        "lm_loss": nan_mean(lm_losses),
        "ittt_loss": nan_mean(ittt_losses),
    })

    for col in df.columns:

        x = np.arange(len(df[col]))
        y_running = df[col].rolling(window=500)
        
        plt.plot(x, y_running.mean(), label=col)
    
    plt.legend()
    plt.grid()
    plt.savefig("loss_comparison.png")


if __name__ == "__main__":
    main()
    analyze_results()
