import torch

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import datasets
from transformers import AutoTokenizer

from models import load_checkpoint
from models.zlm import ZLMModel
from collators.seq_to_seq import SeqToSeqCollator
import utils.constants as constants


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# URL = "aklein4/ZEBRA_ar-1p7b-kernel"
# STEP = 6000
URL = "aklein4/ZEBRA_ar-1p7b-kernel-strong"
STEP = 9000

SAVE_PATH = os.path.join(
    constants.LOCAL_DATA_PATH,
    "zlm_probe_data",
    URL.replace("/", "--"),
    f"step_{STEP:012d}.pt",
)

DATASET = {
    "path": "aklein4/seq2seq-mixed-pretraining-SmolLM2",
    "split": "train",
    "name": "all",
    "streaming": True,
}

BS = 64
NUM_STEPS = 64

SEED = 42


LOCAL_PATH = lambda x: os.path.join(constants.LOCAL_DATA_PATH, x)


@torch.no_grad()
def create_data():

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model: ZLMModel = load_checkpoint(
        URL, STEP,
        attention_kernel="gpu_flash_attention",
    ).to(DEVICE)
    model.eval()

    data = datasets.load_dataset(**DATASET)
    collator = SeqToSeqCollator(
        input_length=model.input_length,
        output_length=model.output_length,
        pad_token_id=model.config.pad_token_id,
    )
    
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BS,
        collate_fn=collator,
        shuffle=False,
    )

    noises = []
    mus = []
    zs = []

    other_noises = []
    other_mus = []
    other_zs = []

    for batch in tqdm(loader, total=NUM_STEPS, desc="Encoding Data"):
        if len(noises) >= NUM_STEPS:
            break

        input_ids = batch["input_ids"].to(DEVICE)
        output_ids = batch["output_ids"].to(DEVICE)

        input_mask = (input_ids != model.config.pad_token_id)
        output_mask = (output_ids != model.config.pad_token_id)

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

        with torch.autocast("cuda", dtype=torch.bfloat16):

            noise = model.sample_noise(input_for_model)
            other_noise = model.sample_noise(input_for_model)

            z, mu, z_states = model.encode(
                input_for_model, output_for_model,
                input_mask=input_mask, output_mask=output_mask,
                noise=noise,
                return_extra=True,
            )

            other_mu = model.encoder_head(
                z_states,
                other_noise,
            )
            other_mu = model.z_out_norm(other_mu)
            other_z = model.add_noise(other_mu, other_noise)

        noises.append(noise)
        mus.append(mu)
        zs.append(z)

        other_noises.append(other_noise)
        other_mus.append(other_mu)
        other_zs.append(other_z)

    d = {
        "noise": noises,
        "mu": mus,
        "z": zs,
        "other_noise": other_noises,
        "other_mu": other_mus,
        "other_z": other_zs,
    }
    for k in d:
        d[k] = torch.cat(d[k], dim=0).cpu().float()

    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(d, SAVE_PATH)


def kde(x, tmin, tmax, num_bins):

    # silverman's rule
    bw = 1.06 * x.std() * (len(x) ** (-1 / 5))

    dist = torch.distributions.Normal(
        x, bw
    )

    t = torch.linspace(tmin, tmax, num_bins, device=x.device, dtype=x.dtype)
    log_probs = dist.log_prob(t[:, None])

    probs = torch.exp(log_probs).mean(-1)

    return t, probs


def SIGReg(x):
    # x: [B, D]

    dev = dict(device=x.device, dtype=x.dtype)

    t = torch.linspace(0.0, 4.0, 16, **dev)
    
    # theoretical CF for N(0, 1) and Gauss. window
    exp_f = torch.exp(-0.5 * t**2)

    # empirical CF
    x_t = x[:, :, None] * t[None, None, :] # [B, D, T]

    real = torch.cos(x_t).mean(0) # [D, T]
    imag = torch.sin(x_t).mean(0)

    # weighted L2 distance
    real_err = (real - exp_f[None])
    imag_err = (imag - torch.zeros_like(exp_f)[None])
    err = real_err**2 + imag_err**2

    # weight by theoretical CF
    # multiply by 2 since we only integrate over positive frequencies
    w_err = err * 2.0 * exp_f[None]

    out = torch.trapz(w_err, t , dim=1) * x.shape[0]

    return out


@torch.no_grad()
def plot_slices():

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tmin = -6.0
    tmax = 6.0
    num_bins = 1001

    num_columns = 5
    num_rows = 4

    num_checks = 1000
    num_slices = num_columns * num_rows

    # reference normal
    t = torch.linspace(tmin, tmax, num_bins)
    ref_dist = torch.distributions.Normal(0, 1) # since z = mu + noise
    ref_probs = torch.exp(ref_dist.log_prob(t))

    data = torch.load(SAVE_PATH)
    z = data["z"][:512, -10].to(DEVICE) / np.sqrt(2)

    w = torch.randn(z.shape[-1], num_checks, device=z.device, dtype=z.dtype)
    w = w / w.norm(dim=0, keepdim=True)

    x = z @ w
    sig_regs = SIGReg(x)
    print(f"\n SIGReg: {sig_regs.mean().item():.2f}\n")

    highest = torch.topk(sig_regs, num_slices, sorted=True).indices
    x = x[:, highest]
    sig_regs = sig_regs[highest]

    slices = []
    for i in tqdm(range(num_slices), desc="Computing KDEs"):
        slices.append(
            kde(x[:, i], tmin, tmax, num_bins)[1]
        )

    fig, ax = plt.subplots(num_rows, num_columns, figsize=(num_columns * 3, num_rows * 3.25))
    ax = ax.flatten()

    for i in range(num_slices):

        ax[i].plot(t, ref_probs, "k--")
        ax[i].plot(t, slices[i].cpu(), color="red")

        ax[i].grid()
        ax[i].set_title(
            f"{sig_regs[i].item():.2f}",
            fontsize=20,
        )
    
    plt.tight_layout()
    plt.savefig(LOCAL_PATH("zlm_probe_slices.png"), dpi=300)


if __name__ == "__main__":

    # create_data()

    plot_slices()
