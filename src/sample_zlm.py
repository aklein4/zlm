import torch

import os
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from models import load_checkpoint
from utils.chat_utils import format_chat, remove_pad
import utils.constants as constants


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

URL = "aklein4/ZLM-v2_zlm-med-ada"
STEP = 10000

TOKENIZER_PATH = os.path.join(constants.LOCAL_DATA_PATH, "tokenizer")

MESSAGES = [
    {
        "role": "user",
        # "content": "What are some ideas for a good short story about a city not on a planet, but rather a generation ship, or on the moon of a gas giant, or somewhere else unusual?"
        "content": "Bob had a farm with animals. He had 12 cows and twice as many sheep. He decided to buy 3 pigs for every sheep he had. How many animals were on the farm after the transaction?",
        # "content": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?"
    },
    {
        "role": "assistant",
        "content": "Bob had 12 cows.\nHe had twice as many sheep as cows, so he had 12 * 2 = 24 sheep.\nHe decided to buy 3 pigs for every sheep he had, so he bought 24 * 3 = 72 pigs.\nIn total, after the transaction, Bob had 12 cows + 24 sheep + 72 pigs = 108 animals on the farm.\n#### 108\nThe answer is: 108"
    }
]

TEMPERATURE = 1.0

SEED = 42

@torch.no_grad()
def main():

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = load_checkpoint(
        URL, STEP,
        attention_kernel="gpu_flash_attention",
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
    )

    input_text, output_text = format_chat(MESSAGES)
    input_ids = tokenizer(
        [input_text],
        return_tensors="pt",
    ).input_ids.to(DEVICE)
    real_output_ids = tokenizer(
        [output_text],
        return_tensors="pt",
    ).input_ids.to(DEVICE)

    z = None
    if False:
        z, mu = model.encode(
            input_ids.repeat(2, 1), real_output_ids.repeat(2, 1),
        )
        
        kl = 0.5 * (
            ((mu[0] - mu[1]) * model.scheduler.a[0]).pow(2) / model.scheduler.b[0].pow(2)
        ).sum(-1)

        plt.plot(kl.cpu().numpy())
        plt.ylim(0, 1)
        plt.savefig("mu_kl.png")
        return 
    
    output_ids, z = model.sample(
        input_ids.repeat(2, 1),
        encoded_z=z,
        temperature=TEMPERATURE,
    )

    print("")
    print(" === INPUT === ")
    print(tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=False))
    print("")
    print(" === EXPECTED OUTPUT === ")
    print(remove_pad(tokenizer.decode(real_output_ids[0].cpu(), skip_special_tokens=False)))
    print("")
    print(" === OUTPUT 1 === ")
    print(remove_pad(tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=False)))
    print("")
    print(" === OUTPUT 2 === ")
    print(remove_pad(tokenizer.decode(output_ids[1].cpu(), skip_special_tokens=False)))
    print("")


if __name__ == "__main__":
    main()
