import torch

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import datasets

from models.ittt import ItttModel
from models import load_checkpoint
from collators.tokenize import TokenizeCollator
import utils.constants as constants
from utils.loss_utils import lm_loss_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CHECKPOINT_URL = 'aklein4/iTTT-TPU_alpha-1b'
CHECKPOINT_STEP = 500

DATA_URL = "Geralt-Targaryen/books3"
TOKENIZER_URL = os.path.join(constants.LOCAL_DATA_PATH, "tokenizer")

NUM_EXAMPLES = 128
BS = 4

SEQUENCE_LENGTH = 1024 * 128


def main():
    
    print("Loading data...")
    data = datasets.load_dataset(DATA_URL, split='train', streaming=True)
    collator = TokenizeCollator(TOKENIZER_URL, SEQUENCE_LENGTH)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BS,
        collate_fn=collator,
    )

    print("Loading model...")
    model: ItttModel = load_checkpoint(
        CHECKPOINT_URL,
        CHECKPOINT_STEP,
        attention_kernel="gpu_flash_attention",
    ).to(DEVICE)

    # lr = model.model.layers[0].mlp.down_proj.log_lr * model.model.layers[16].mlp.down_proj.scalar_scaler
    # plt.matshow(lr.detach().cpu().numpy())
    # plt.colorbar()
    # plt.savefig("ittt_lr.png")
    
    # plt.clf()
    # plt.hist(lr.detach().cpu().numpy().flatten(), bins=100)
    # plt.grid()
    # plt.savefig("ittt_lr_hist.png")
    # exit()

    print("Running models...")
    losses = []
    for i, batch in tqdm(enumerate(loader), total=NUM_EXAMPLES // BS, desc="Processing Batches"):

        input_ids = batch["input_ids"].to(DEVICE)

        logits = model.compute_logits(
            input_ids,
            verbose=True,
        )
        loss = lm_loss_fn(
            logits, input_ids.cpu(),
            shift_logits=False,
            ignore_index=model.config.pad_token_id,
            reduction='none',
        )
        losses.append(loss)
        
        if len(losses) >= NUM_EXAMPLES // BS:
            break
        
    losses = torch.cat(losses, dim=0)
    torch.save(
        losses,
        os.path.join(constants.LOCAL_DATA_PATH, "ittt_losses.pt")
    )


def nan_mean(x):
    mask = np.isfinite(x)
    s = mask.sum(0)
    x = np.where(mask, x, np.zeros_like(x))
    return x.sum(0) / s


def analyze_results():

    losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "ittt_losses.pt")).float().numpy()
    
    df = pd.DataFrame({
        "loss": nan_mean(losses),
    })

    x = np.arange(len(df["loss"]))
    y_running = df["loss"].rolling(window=2000)
        
    plt.plot(x, y_running.mean())
    
    # plt.legend()
    plt.grid()
    plt.savefig("ittt_extrapolation_loss.png")


if __name__ == "__main__":
    # main()
    analyze_results()
