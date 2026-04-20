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
from collators.tokenize import TokenizeCollator
import utils.constants as constants
from utils.loss_utils import lm_loss_fn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LM_URL = 'aklein4/iTTT-TPU_attn-baseline-theta-1b'
LM_STEP = 2000

ITTT_URL = 'aklein4/iTTT-TPU_mlp-1b'
ITTT_STEP = 500

DATA_URL = "Geralt-Targaryen/books3"
TOKENIZER_URL = os.path.join(constants.LOCAL_DATA_PATH, "tokenizer")

NUM_EXAMPLES = 64
BS = 2

SEQUENCE_LENGTH = 1024 * 128

THIS_PREFIX = "mlp_"
PREFIX = "extrapolate_"

DO_LM = True


def main():
    
    print("Loading data...")
    data = datasets.load_dataset(DATA_URL, split='train', streaming=True)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BS,
        collate_fn=TokenizeCollator(TOKENIZER_URL, SEQUENCE_LENGTH),
    )

    if DO_LM:
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
            logits, input_ids.to(logits.device),
            shift_logits=False,
            ignore_index=ittt_model.config.pad_token_id,
            reduction='none',
        )
        ittt_losses.append(loss.cpu())
        del logits

        if DO_LM:
            with torch.no_grad():
                with torch.autocast(str(constants.DEVICE), torch.bfloat16):

                    states = lm_model.forward(
                        input_ids,
                        return_states=True,
                        logits_to_keep=slice(0, 1),
                    )[-1]
                    states = lm_model.model.norm(states)

                    states = states[:, :-1].split(1024, dim=1)
                    labels = input_ids[:, 1:].split(1024, dim=1)

                    loss = []
                    for s, l in tqdm(zip(states, labels), total=len(states), desc="Processing LM Chunks", leave=False):

                        logits = lm_model.lm_head(s).float()

                        curr_loss = lm_loss_fn(
                            logits, l,
                            shift_logits=False,
                            shift_labels=False,
                            ignore_index=lm_model.config.pad_token_id,
                            reduction='none',
                        )
                        loss.append(curr_loss)
                    
                    loss = torch.cat(loss, dim=1)

            lm_losses.append(loss.cpu())
            del logits

        if len(ittt_losses) >= NUM_EXAMPLES // BS:
            break
    
    if DO_LM:
        lm_losses = torch.cat(lm_losses, dim=0)
        torch.save(
            lm_losses,
            os.path.join(constants.LOCAL_DATA_PATH, "theta_"+PREFIX+"lm_losses_for_comparison.pt")
        )
    
    ittt_losses = torch.cat(ittt_losses, dim=0)
    torch.save(
        ittt_losses,
        os.path.join(constants.LOCAL_DATA_PATH, "mlp_"+PREFIX+"ittt_losses_for_comparison.pt")
    )


def nan_mean(x):
    mask = np.isfinite(x)
    s = mask.sum(0)
    x = np.where(mask, x, np.zeros_like(x))
    return x.sum(0) / (s + 1e-7)


def analyze_results():

    lm_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "theta_"+PREFIX+"lm_losses_for_comparison.pt")).float().numpy()
    # ittt_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, PREFIX+"ittt_losses_for_comparison.pt")).float().numpy()
    # norm_ittt_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "norm_"+PREFIX+"ittt_losses_for_comparison.pt")).float().numpy()
    fancy_ittt_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "fancy_"+PREFIX+"ittt_losses_for_comparison.pt")).float().numpy()
    # mlp_ittt_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "mlp_"+PREFIX+"ittt_losses_for_comparison.pt")).float().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    df = pd.DataFrame({
        "lm_loss": nan_mean(lm_losses)[:128*1024],
        "ittt_loss": nan_mean(fancy_ittt_losses)[:128*1024],
        # "fancy_ittt_loss": nan_mean(fancy_ittt_losses),
        # "mlp_ittt_loss": nan_mean(mlp_ittt_losses),
    })

    print("\n === Average Losses === ")
    for col in df.columns:
        print(f"    {col}: {df[col].mean():.2f}")
    print("")

    for col in df.columns:

        x = np.arange(len(df[col]))
        y_running = df[col].rolling(window=5000, min_periods=1000)
        
        ax[0].plot(x, y_running.mean(), label=col)
    
    ax[0].legend()
    ax[0].grid()

    ax[0].set_title("Loss by Token Position")
    ax[0].set_xlabel("Token Position")
    ax[0].set_ylabel("Loss (log perplexity)")
    # ax[0].set_ylim(3.0, 5.5)

    # plt.savefig(PREFIX+"loss_comparison.png")
    # plt.clf()
    
    diff = fancy_ittt_losses - lm_losses
    # fancy_diff = fancy_ittt_losses - lm_losses
    # mlp_diff = mlp_ittt_losses - lm_losses

    df = pd.DataFrame({
        "ittt_diff": nan_mean(diff)[:128*1024],
        # "fancy_ittt_diff": nan_mean(fancy_diff),
        # "mlp_ittt_diff": nan_mean(mlp_diff),
    })

    for col in df.columns:
        
        x = np.arange(len(df[col]))
        y_running = df[col].rolling(window=5000)
        
        ax[1].plot(x, y_running.mean(), label=col)
    
    # plt.legend()
    plt.grid()

    ax[1].set_title("Loss Delta by Token Position")
    ax[1].set_xlabel("Token Position")
    ax[1].set_ylabel("Loss difference with full attn.")

    plt.suptitle("Comparison at 128K Context Length")
    plt.tight_layout()
    plt.savefig("128K_comparison.png", dpi=300)


if __name__ == "__main__":
    # main()
    analyze_results()
