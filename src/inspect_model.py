import torch

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import PIL.Image as Image
import yaml
import omegaconf

import datasets

from utils.import_utils import import_model
import utils.constants as constants
from collators.seq_to_seq import SeqToSeqCollator
from utils.attention_utils import AtttentionProbe as AP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = "zlm-smollm2-360m"

DATA_URL = ("aklein4/seq2seq-mixed-pretraining-SmolLM2", "all")

MU_PATH = os.path.join(
    constants.LOCAL_DATA_PATH, "mu_values_ada.pt"
)

BS = 128
NUM_STEPS = (1024 * 4) // BS

ATTN_IDX = 24


def local_dir(path):
    return os.path.join(constants.LOCAL_DATA_PATH, path)


@torch.no_grad()
def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
 
    with open(os.path.join(constants.BASE_PATH, f"configs/model/{CONFIG}.yaml"), "r") as f:
        config_dict = yaml.safe_load(f)
    config = omegaconf.OmegaConf.create(config_dict)
    config.attention_kernel = None

    config.z_proj_in_init_scale = 100.0

    model = import_model(config.type)(config).to(DEVICE)
    model.train()
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

    print("Collecting mu values...")
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

        AP.call_fn(model, "enable", idx=ATTN_IDX)

        # encode and decode
        _, mu = model.encode(
            input_for_model, output_for_model,
            input_mask=input_mask, output_mask=output_mask,
        )
        all_mu.append(mu.cpu())

        _ = model.decode(
            input_for_model, output_for_model, _,
            input_mask=input_mask, output_mask=output_mask,
        )
        attn = AP.call_fn(model.decoder_model, "get")
        
        os.makedirs(local_dir("attention_visualizations"), exist_ok=True)

        for h in tqdm(range(attn.shape[1])):
            attn_h = attn[:, h, :, :]  # [batch, seq, seq]
            
            full_mask = torch.cat(
                [
                    input_mask,
                    torch.ones(attn_h.shape[0], attn_h.shape[-1]-input_mask.shape[-1]-output_mask.shape[-1], device=DEVICE, dtype=torch.bool),
                    output_mask,
                ],
                dim=-1
            )
            attn_mask = full_mask[:, None, :] & full_mask[:, :, None]
            
            attn_h = torch.where(
                attn_mask,
                attn_h,
                torch.full_like(attn_h, 0.0)
            )

            # attn_h = attn_h.sum(0) / (attn_mask.float().sum(0) + 1e-7)
            attn_h = attn_h / (attn_h.max(dim=-1, keepdim=True).values + 1e-7)
            
            # attn_h = attn_h.max(0).values
            attn_h = attn_h[10]

            # attn_h = attn_h / attn_h.max(dim=-1, keepdim=True).values

            img = Image.fromarray(
                (attn_h.cpu().numpy() * 255).astype(np.uint8)
            )
            img.save(
                local_dir(
                    os.path.join(
                        "attention_visualizations",
                        f"layer={ATTN_IDX}_head={h}.png"
                    ),
                )
            )

            # attn_h = (
            #     attn_h /
            #     attn_h.nan_to_num(float("-inf"), float("-inf"), float("-inf")).max(dim=-1, keepdim=True).values
            # )
            # attn_h = torch.log10(attn_h)

            # plt.matshow(
            #     attn_h.cpu().numpy(),
            #     vmin=0.0, vmax=attn_h.max().item(),
            # )
            # plt.colorbar()
            # plt.savefig(
            #     local_dir(
            #         os.path.join(
            #             "attention_visualizations",
            #             f"layer={ATTN_IDX}_head={h}.png"
            #         ),
            #     ),
            #     dpi=300,
            # )
            # plt.clf()

        exit(0)

        pbar.update(1)

    pbar.close()

    all_mu = torch.cat(all_mu, dim=0)
    torch.save(all_mu, MU_PATH)


@torch.no_grad()
def huh():

    mu = torch.load(MU_PATH)
    x = mu[:, 100] # [batch, dim]

    cov = torch.cov(x.T) # [dim, dim]
    max_abs = cov.abs().max().item()

    plt.matshow(
        cov.cpu().numpy(),
        cmap="bwr",
        vmin=-max_abs, vmax=max_abs
    )
    plt.colorbar()
    plt.savefig(local_dir("mu_cov.png"))
    plt.clf()

    val, vec = torch.linalg.eigh(cov)
    plt.plot(val.flip(0).cpu().numpy())
    plt.grid()
    plt.savefig(local_dir("mu_cov_eigenvalues.png"))
    plt.clf()

    mean = x.mean(0).cpu().numpy()
    mean = np.sort(mean)[::-1]
    plt.plot(mean)
    plt.grid()
    plt.savefig(local_dir("mu_mean.png"))


if __name__ == "__main__":
    main()
