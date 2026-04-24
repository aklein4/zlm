
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

import utils.constants as constants


SAVE_PATH = os.path.join(constants.LOCAL_DATA_PATH, "benchmark_chart.png")

BASE_PATH = os.path.join(constants.LOCAL_DATA_PATH, "evaluation_results")

# CHECKPOINTS = {
#     "aklein4--ZEBRA_muon-1p7b-once/000000024000": "Low-MI-09K",
#     "aklein4--ZEBRA_muon-1p7b-mi/000000018000": "Diffusion-18K",
#     "aklein4--ZEBRA_muon-1p7b-cont/000000009000": "Diffusion-09K",
#     "../guidance_evaluation_results": "Guided",
#     "../guidance_evaluation_results_2": "Guided-2",
#     "aklein4--ZEBRA_ar-1p7b-kernel-strong/000000009000": "AR-09K",
#     "aklein4--ZEBRA_ar-1p7b-sigreg/000000007000": "SIGReg-07K",
#     "../no_noise_scaled_eval_results/aklein4--ZEBRA_ar-1p7b-kernel-strong/000000009000": "AR-NN-09K",
#     # "aklein4--ZEBRA_baseline-1p7b/000000050000": "Baseline-50K",
#     # "aklein4--ZEBRA_baseline-1p7b/000000020000": "Baseline-20K",
#     # "aklein4--ZEBRA_baseline-1p7b/000000015000": "Baseline-15K",
#     "aklein4--ZEBRA_baseline-1p7b/000000010000": "Baseline-10K",
#     # "aklein4--ZEBRA_baseline-1p7b/000000005000": "Baseline-5K",
# }
CHECKPOINTS = {
    "../guidance_evaluation_results_2": "Diffusion Guided",
    "../no_noise_scaled_eval_results/aklein4--ZEBRA_ar-1p7b-kernel-strong/000000009000": "Autoregressive",
    "aklein4--ZEBRA_baseline-1p7b/000000010000": "Baseline LLM",
}

REASONING_BENCHMARKS = [
    # "ARC-E-Train",
    # "ARC-C-Train",
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "MMLU-Pro",
    "GPQA",
    "PIQA",
    "SciQ",
    "AR-LSAT",
    "StrategyQA",
]

MATH_BENCHMARKS = [
    "GSM8K",
    "MATH-500",
    "SVAMP",
    "Minerva-Math",
    "AMC-23",
]


def main():
    
    all_benchmarks = REASONING_BENCHMARKS + MATH_BENCHMARKS

    checkpoint_results = {}
    for checkpoint, name in CHECKPOINTS.items():

        ckpt_results = {}

        ckpt_path = os.path.join(BASE_PATH, checkpoint)
        for b in all_benchmarks:

            f_path = os.path.join(ckpt_path, f"{b}.json")

            try:
                if os.path.exists(f_path):
                    with open(f_path, "r") as f:
                        ckpt_results[b] = round(100 * json.load(f)["accuracy"], 1)
                else:
                    raise FileNotFoundError(f"File not found: {f_path}")
            except:
                ckpt_results[b] = None

        checkpoint_results[name] = ckpt_results

    benchmarks_mean = {}
    benchmarks_std = {}
    for b in all_benchmarks:
        vals = [v[b] for v in checkpoint_results.values() if v[b] is not None]

        benchmarks_mean[b] = np.mean(vals)
        benchmarks_std[b] = np.std(vals)

    for results in checkpoint_results.values():
        
        reasoning_results = [
            v for k, v in results.items() if k in REASONING_BENCHMARKS
        ]
        math_results = [
            v for k, v in results.items() if k in MATH_BENCHMARKS
        ]

        reasoning_scores = [
            (v - benchmarks_mean[k]) / benchmarks_std[k] if v is not None else None
            for k, v in results.items() if k in REASONING_BENCHMARKS
        ]
        math_scores = [
            (v - benchmarks_mean[k]) / (benchmarks_std[k]+1e-5) if v is not None else None
            for k, v in results.items() if k in MATH_BENCHMARKS
        ]

        if None in reasoning_results:
            results["Reasoning Avg"] = None
            results["Reasoning Score"] = None
        else:
            results["Reasoning Avg"] = round(np.mean(reasoning_results), 1)
            results["Reasoning Score"] = round(np.mean(reasoning_scores), 2)
            
        if None in math_results:
            results["Math Avg"] = None
            results["Math Score"] = None
        else:
            results["Math Avg"] = round(np.mean(math_results), 1)
            results["Math Score"] = round(np.mean(math_scores), 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15.5, 2.5+len(CHECKPOINTS)*0.5))
    fig.subplots_adjust(hspace=0.4)

    checkpoint_names = list(checkpoint_results.keys())

    for ax, benchmarks, title in [
        (ax1, REASONING_BENCHMARKS + ["Reasoning Avg", "Reasoning Score"], "Reasoning"),
        (ax2, MATH_BENCHMARKS + ["Math Avg", "Math Score"], "Math"),
    ]:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

        # Build cell text and find max per column
        cell_text = []
        raw_values = []
        for name in checkpoint_names:
            row = []
            row_raw = []
            for b in benchmarks:
                val = checkpoint_results[name].get(b)
                row_raw.append(val)
                if val is None:
                    row.append("--")
                else:
                    row.append(f"{val:.1f}")
            cell_text.append(row)
            raw_values.append(row_raw)

        # Find the max for each benchmark column
        max_per_col = []
        for col_idx in range(len(benchmarks)):
            col_vals = [raw_values[row_idx][col_idx] for row_idx in range(len(checkpoint_names))]
            valid = [v for v in col_vals if v is not None]
            max_per_col.append(max(valid) if valid else None)

        # Bold the highest scores
        for row_idx in range(len(checkpoint_names)):
            for col_idx in range(len(benchmarks)):
                val = raw_values[row_idx][col_idx]
                if val is not None and max_per_col[col_idx] is not None and val >= max_per_col[col_idx]:
                    cell_text[row_idx][col_idx] = r"$\bf{" + f"{val:.1f}" + r"}$"

        table = ax.table(
            cellText=cell_text,
            rowLabels=checkpoint_names,
            colLabels=benchmarks,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f"Chart saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
