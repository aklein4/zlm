import torch

import os
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from models import load_checkpoint
from models.zlm import ZLMModel
from utils.chat_utils import format_chat, remove_pad, format_cot, format_no_cot, mcqa_question
import utils.constants as constants


URL = "aklein4/ZEBRA-v2_360m-alpha-lowReg"
STEP = 21000

TOKENIZER_PATH = os.path.join(constants.LOCAL_DATA_PATH, "tokenizer")

# MESSAGES = format_chat(
#     [
#         {
#             "role": "user",
#             "content": "Write a 1 paragraph summary of a creative sci-fi setting.",
#         },
#         {
#             "role": "assistant",
#             "content": "On a tidally locked exoplanet, the habitable zone is a narrow ring of perpetual twilight between a scorched day side and a frozen night side. Explorers travel along this ring in self-repairing crawler cities, powered by atmospheric turbines and geothermal taps, while swarms of programmable “lichen” convert toxic minerals into soil, oxygen, and building material. Beneath the surface, quantum-linked sensor roots map the planet’s shifting magnetic storms and awaken dormant climate machines left by an extinct civilization, whose weather-control network may be the only technology capable of preventing the twilight belt from collapsing into fire or ice.",
#         }
#     ],
# )

# MESSAGES = format_chat(
#     [
#         {
#             "role": "user",
#             "content": "Write a 1 paragraph introduction to a DND campaign where the 4 player characters meet in a tavern. You should describe each of the characters in the narrative.",
#         },
#         {
#             "role": "assistant",
#             "content": "In a bustling tavern at the heart of a lively town, four adventurers find themselves seated at the same worn wooden table. A towering half-orc barbarian with a scarred face and a massive greataxe slung across his back grunts a greeting. Beside him, a nimble elven rogue with a mischievous glint in her eyes twirls a dagger between her fingers. Across from them, a human wizard with a long, flowing robe and a staff adorned with glowing runes adjusts his spectacles, eyeing the others curiously. Finally, a cheerful halfling bard with a lute strapped to his back and a wide, infectious smile raises a mug of ale in a friendly toast. As the tavern's warm light flickers over their faces, the four adventurers exchange stories, laughter, and the promise of shared quests to come."
#         }
#     ],
# )

MESSAGES = format_chat(
    [
        {
            "role": "user",
            "content": "Describe 4 player characters that might be part of a party in a DND compaign.",
        },
        {
            "role": "assistant",
            "content": "1. A towering half-orc barbarian with a scarred face and a massive greataxe slung across his back.\n\n2. A nimble elven rogue with a mischievous glint in her eyes and a dagger twirling between her fingers.\n\n3. A human wizard with a long, flowing robe and a staff adorned with glowing runes, adjusting his spectacles.\n\n4. A cheerful halfling bard with a lute strapped to his back and a wide, infectious smile."
        }
    ],
)

# MESSAGES = format_no_cot(
#     "Bob had a farm with animals. He had 12 cows and twice as many sheep. He decided to buy 3 pigs for every sheep he had. How many animals were on the farm after the transaction?",
#     "Bob had 12 cows.\nHe had twice as many sheep as cows, so he had 12 * 2 = 24 sheep.\nHe decided to buy 3 pigs for every sheep he had, so he bought 24 * 3 = 72 pigs.\nIn total, after the transaction, Bob had 12 cows + 24 sheep + 72 pigs = 108 animals on the farm.\n#### 108\nThe answer is: 108",
#     108
# )

TEMPERATURE = "greedy"

ROLLOUT_KWARGS = {
    "noise_temperature": 1.0,
    "guidance_scale": None,
}

SEED = 42


@torch.no_grad()
def main():

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model: ZLMModel = load_checkpoint(
        URL, STEP,
        attention_kernel="gpu_flash_attention",
    ).to(constants.DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
    )

    input_text, output_text = MESSAGES
    input_ids = tokenizer(
        [input_text],
        return_tensors="pt",
    ).input_ids.to(constants.DEVICE)
    real_output_ids = tokenizer(
        [output_text],
        return_tensors="pt",
    ).input_ids.to(constants.DEVICE)

    with torch.autocast("cuda", torch.bfloat16, enabled=torch.cuda.is_available()):
        
        output_ids, z = model.sample(
            input_ids.repeat(2, 1),
            temperature=TEMPERATURE,
            verbose=True,
            **ROLLOUT_KWARGS,
        )

    torch.save(
        {
            "input_ids": input_ids.cpu()[:1],
            "output_ids": output_ids.cpu()[:1],
            "z": z[:1].cpu(),
        },
        "zlm_sample.pt"
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
