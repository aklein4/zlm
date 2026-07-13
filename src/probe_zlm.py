import torch

import json
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from transformers import AutoTokenizer

from models import load_checkpoint
from models.zlm import ZLMModel
from utils.chat_utils import format_chat, remove_pad, format_cot, format_no_cot, mcqa_question
from utils.loss_utils import lm_loss_fn
import utils.constants as constants


URL = "aklein4/ZEBRA-v2_360m-alpha-probe"
STEP = 26000

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

SAVED_SAMPLE = "zlm_sample.pt"

BATCH_SIZE = 64

SEED = 42

HTML_PATH = os.path.join(constants.LOCAL_DATA_PATH, "logp_animation.html")
HTML_FRAME_INTERVAL_MS = 50
HTML_BACKGROUND_COLOR = "#08111f"
HTML_CARD_COLOR = "#111d2e"
HTML_TEXT_COLOR = "#f1f5f9"
HTML_MUTED_COLOR = "#94a3b8"
HTML_ACCENT_COLOR = "#38bdf8"
HTML_TOKEN_COLOR = "#f8fafc"


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

    if SAVED_SAMPLE is not None and os.path.exists(SAVED_SAMPLE):
        sample = torch.load(SAVED_SAMPLE)
        input_ids = sample["input_ids"].to(constants.DEVICE)
        output_ids = sample["output_ids"].to(constants.DEVICE)
        z = sample["z"].to(constants.DEVICE)

    else:
        input_text, output_text = MESSAGES
        input_ids = tokenizer(
            [input_text],
            return_tensors="pt",
        ).input_ids.to(constants.DEVICE)
        output_ids = tokenizer(
            [output_text],
            return_tensors="pt",
        ).input_ids.to(constants.DEVICE)

        with torch.autocast("cuda", torch.bfloat16, enabled=torch.cuda.is_available()):
            z, mu = model.encode(
                input_ids, output_ids
            )

    logps = []
    with torch.autocast("cuda", torch.bfloat16, enabled=torch.cuda.is_available()):

        for i in tqdm(range(0, model.z_length+1, BATCH_SIZE)):
           
            progress = torch.arange(
                i, min(i + BATCH_SIZE, model.z_length+1)
            ).to(constants.DEVICE)
            bs = progress.shape[0]

            inp = input_ids.repeat(bs, 1)
            outp = output_ids.repeat(bs, 1)
            zp = z.repeat(bs, 1, 1)

            logits, z_states = model.decode(
                inp, outp, zp,
                progress=progress
            )

            logp = -lm_loss_fn(
                logits, outp,
                shift_logits=False, shift_labels=False,
                reduction="none"
            )
            
            # use the entropy
            # logp = (logits * logits.exp()).sum(dim=-1)
            # logits = torch.nn.functional.log_softmax(logits, dim=-1)

            logps.append(logp)
    
    logps = torch.cat(logps, dim=0).float()
    # logps = torch.minimum(logps, logps[-1:])

    mn = logps.amin(dim=0, keepdim=True)
    mx = logps.amax(dim=0, keepdim=True)

    v = ((logps - mn) / (mx - mn + 1e-10))

    create_html_animation(tokenizer, input_ids, output_ids, v.cpu())

    return

    v = v[::5]

    positions = torch.arange(v.shape[-1])
    fig, ax = plt.subplots()
    scatter = ax.scatter(positions, v[0])
    ax.set(
        xlabel="Position",
        ylabel="v",
        xlim=(-0.5, v.shape[-1] - 0.5),
        ylim=(-0.05, 1.05),
    )

    def update(frame):
        scatter.set_offsets(torch.stack((positions, v[frame]), dim=-1).numpy())
        ax.set_title(f"Iteration {frame}")
        return scatter,

    animation = FuncAnimation(
        fig, update, frames=v.shape[0], interval=100, blit=True
    )
    output_path = os.path.join(os.path.dirname(__file__), "logp_animation.gif")
    animation.save(output_path, writer=PillowWriter(fps=10))
    plt.close(fig)


def create_html_animation(tokenizer, input_ids, output_ids, v):
    """Create an HTML animation of output-token opacity over iterations."""

    input_text = tokenizer.decode(
        input_ids[0].detach().cpu().tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    token_text = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in output_ids[0].detach().cpu().tolist()
    ]
    animation_data = json.dumps(
        {
            "input": input_text,
            "tokens": token_text,
            "values": v.detach().float().cpu().tolist(),
        }
    ).replace("<", "\\u003c")

    replacements = {
        "__ANIMATION_DATA__": animation_data,
        "__FRAME_INTERVAL__": str(HTML_FRAME_INTERVAL_MS),
        "__BACKGROUND__": HTML_BACKGROUND_COLOR,
        "__CARD__": HTML_CARD_COLOR,
        "__TEXT__": HTML_TEXT_COLOR,
        "__MUTED__": HTML_MUTED_COLOR,
        "__ACCENT__": HTML_ACCENT_COLOR,
        "__TOKEN__": HTML_TOKEN_COLOR,
    }
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Token confidence over decoding progress</title>
  <style>
    :root {
      color-scheme: dark;
      --background: __BACKGROUND__; --card: __CARD__; --text: __TEXT__;
      --muted: __MUTED__; --accent: __ACCENT__; --token: __TOKEN__;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--background); color: var(--text);
           font: 16px/1.6 ui-sans-serif, system-ui, sans-serif; }
    main { width: min(960px, calc(100% - 32px)); margin: 48px auto; }
    header { margin-bottom: 24px; }
    h1 { margin: 0 0 6px; font-size: clamp(26px, 4vw, 40px); letter-spacing: -0.03em; }
    .subtitle, .label { color: var(--muted); }
    .card { padding: 24px; margin-top: 16px; border: 1px solid rgba(148,163,184,.18);
            border-radius: 16px; background: var(--card); box-shadow: 0 20px 50px rgba(0,0,0,.25); }
    .label { margin-bottom: 8px; font-size: 12px; font-weight: 700; letter-spacing: .12em;
             text-transform: uppercase; }
    #prompt { max-height: 130px; overflow: auto; white-space: pre-wrap; font-size: 14px; }
    #output { min-height: 180px; white-space: pre-wrap; color: var(--token);
              font: 20px/1.8 ui-serif, Georgia, serif; }
    .token { transition: opacity 80ms linear; }
    .controls { display: grid; grid-template-columns: auto 1fr auto; gap: 14px;
                align-items: center; margin-bottom: 22px; }
    button { padding: 8px 16px; border: 0; border-radius: 999px; background: var(--accent);
             color: var(--background); font-weight: 800; cursor: pointer; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    #status { min-width: 132px; color: var(--muted); font-variant-numeric: tabular-nums; text-align: right; }
    .legend { display: flex; align-items: center; gap: 10px; margin-top: 18px;
              color: var(--muted); font-size: 13px; }
    .gradient { width: 150px; height: 8px; border-radius: 999px;
                background: linear-gradient(90deg, transparent, var(--token)); }
    @media (max-width: 600px) { .controls { grid-template-columns: auto 1fr; }
      #status { grid-column: 1 / -1; text-align: left; } .card { padding: 18px; } }
  </style>
</head>
<body>
<main>
  <header><h1>Token confidence</h1><div class="subtitle">Output visibility across decoding progress</div></header>
  <section class="card"><div class="label">Input</div><div id="prompt"></div></section>
  <section class="card">
    <div class="controls">
      <button id="toggle" type="button">Pause</button>
      <input id="timeline" type="range" min="0" value="0" step="1" aria-label="Animation frame">
      <div id="status"></div>
    </div>
    <div class="label">Output</div><div id="output"></div>
    <div class="legend"><span>v = 0</span><span class="gradient"></span><span>v = 1</span></div>
  </section>
</main>
<script>
  const data = __ANIMATION_DATA__;
  const prompt = document.getElementById("prompt");
  const output = document.getElementById("output");
  const status = document.getElementById("status");
  const timeline = document.getElementById("timeline");
  const toggle = document.getElementById("toggle");
  prompt.textContent = data.input;
  timeline.max = data.values.length - 1;
  const spans = data.tokens.map(token => {
    const span = document.createElement("span");
    span.className = "token"; span.textContent = token; output.appendChild(span); return span;
  });
  let frame = 0;
  let playing = true;
  function draw(nextFrame) {
    frame = nextFrame;
    spans.forEach((span, position) => { span.style.opacity = data.values[frame][position]; });
    timeline.value = frame;
    status.textContent = `Iteration ${frame} / ${data.values.length - 1}`;
  }
  timeline.addEventListener("input", event => draw(Number(event.target.value)));
  toggle.addEventListener("click", () => {
    playing = !playing; toggle.textContent = playing ? "Pause" : "Play";
  });
  draw(0);
  setInterval(() => { if (playing) draw((frame + 1) % data.values.length); }, __FRAME_INTERVAL__);
</script>
</body>
</html>
"""
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    return HTML_PATH


if __name__ == "__main__":
    main()
