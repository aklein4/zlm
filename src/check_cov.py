import torch

import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import datasets

from models import load_checkpoint
import utils.constants as constants
from collators.seq_to_seq import SeqToSeqCollator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = "aklein4/ZLM-v2_zlm-med-bn-Sym"
STEP = 5000

DATA_URL = ("aklein4/seq2seq-mixed-pretraining-SmolLM2", "all")

MU_PATH = os.path.join(
    constants.LOCAL_DATA_PATH, "mu_values.pt"
)

BS = 128
NUM_STEPS = (1024 * 8) // BS


@torch.no_grad()
def get_data():

    model = load_checkpoint(
        MODEL_URL, STEP,
        attention_kernel="gpu_flash_attention",
    ).to(DEVICE)
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
        pbar.update(1)
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
        all_mu.append(mu.cpu() / model.config.mu_alpha)

    pbar.close()

    all_mu = torch.cat(all_mu, dim=0)
    torch.save(all_mu, MU_PATH)


def main():

    mu = torch.load(MU_PATH)
    # x = mu[:, 50] # [batch, dim]
    x = torch.cat(
        [mu[:, 0], mu[:, 1]], dim=-1
    )

    cov = torch.cov(x.T) # [dim, dim]
    max_abs = cov.abs().max().item()

    plt.matshow(
        cov.cpu().numpy(),
        cmap="bwr",
        vmin=-max_abs, vmax=max_abs
    )
    plt.colorbar()
    plt.savefig("mu_cov.png")
    plt.clf()

    pc = torch.linalg.eigvalsh(cov).flip(0).cpu().numpy()
    plt.plot(pc[:10])
    plt.yscale("log")
    plt.savefig("mu_cov_eigenvalues.png")

    L = torch.linalg.cholesky(cov)
    whitened = torch.linalg.solve_triangular(
        L, x.T, upper=False
    ).T  # [batch, dim]

    cov_whitened = torch.cov(whitened.T)
    max_abs_whitened = cov_whitened.abs().max().item()

    plt.matshow(
        cov_whitened.cpu().numpy(),
        cmap="bwr",
        vmin=-max_abs_whitened, vmax=max_abs_whitened
    )
    plt.colorbar()
    plt.savefig("mu_cov_whitened.png")


if __name__ == "__main__":
    # get_data()
    main()
