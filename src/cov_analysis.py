import torch

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import hydra
import omegaconf

import datasets

from utils.import_utils import import_model
import utils.constants as constants
from collators.seq_to_seq import SeqToSeqCollator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_URL = ("aklein4/seq2seq-mixed-pretraining-SmolLM2", "all")

MU_PATH = os.path.join(
    constants.LOCAL_DATA_PATH, "mu_values.pt"
)

BS = 128
NUM_STEPS = (1024 * 4) // BS

COV_BS = 512


def local_dir(path):
    return os.path.join(constants.LOCAL_DATA_PATH, path)


@torch.no_grad()
def get_data(config: omegaconf.DictConfig):

    config.model.attention_kernel = "gpu_flash_attention"
    model = import_model(config.model.type)(config.model).to(DEVICE)
    model.eval()

    pad_token_id = model.config.pad_token_id

    data = datasets.load_dataset(
        *DATA_URL, split="validation", streaming=True,
    )
    collator = SeqToSeqCollator(
        input_length=model.config.input_length,
        output_length=model.config.output_length,
        pad_token_id=pad_token_id,
    )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BS,
        shuffle=False,
        drop_last=True,
        collate_fn=collator,
    )

    all_mu = []
    pbar = tqdm(total=NUM_STEPS)
    for i, batch in enumerate(loader):
        if i >= NUM_STEPS:
            break

        input_ids = batch["input_ids"].to(DEVICE)
        output_ids = batch["output_ids"].to(DEVICE)

        # prepare inputs
        input_mask = (input_ids != pad_token_id)
        output_mask = (output_ids != pad_token_id)

        input_for_model = torch.where(
            input_mask,
            input_ids,
            torch.zeros_like(input_ids)
        )
        output_for_model = torch.where(
            output_mask,
            output_ids,
            torch.zeros_like(output_ids)
        )

        # encode and decode
        _, mu = model.encode(
            input_for_model, output_for_model,
            input_mask=input_mask, output_mask=output_mask,
        )
        all_mu.append(mu.cpu())

        pbar.update(1)

    pbar.close()

    all_mu = torch.cat(all_mu, dim=0)
    torch.save(all_mu, MU_PATH)


@torch.no_grad()
def analyze():

    mu = torch.load(MU_PATH).to(DEVICE)
    mu = mu[:, 100]

    x, other = mu[:128], mu[128:]

    ys = []
    for start in tqdm(range(0, other.shape[0], COV_BS)):
        end = start + COV_BS
        if end > other.shape[0]:
            break
        batch = other[start:end]

        m = batch.mean(0)
        cov = torch.cov(batch.T)
    
        eig_vals, eig_vecs = torch.linalg.eigh(
            cov + 1e-5 * torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        )
        eig_vals = torch.clamp(eig_vals, min=1e-5)        
        
        inv_sqrt_cov = (
            eig_vecs @
            torch.diag_embed(eig_vals.rsqrt()) @
            eig_vecs.transpose(-1, -2)  
        )

        y = (x - m[None]) @ inv_sqrt_cov
        ys.append(y)
    
    y = torch.stack(ys, 0)

    kl = (y.var(0).sum(-1) / 2).sort(descending=True)[0].cpu().numpy()

    plt.plot(kl)
    plt.grid()
    plt.savefig("mu_whitened_variance.png")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    get_data(config)
    analyze()


if __name__ == "__main__":
    main()
