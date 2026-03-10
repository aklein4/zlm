import torch

import os
from tqdm import tqdm
import json

from transformers.utils.logging import disable_progress_bar, enable_progress_bar, is_progress_bar_enabled

from evaluation.benchmarks import BENCHMARK_DICT
import utils.constants as constants


def run_benchmarks(
    model,
    tokenizer,
    max_input_length: int,
    max_output_length: int,
    batch_size: int,
    benchmarks: list[str] | None = None,
    max_examples: int | None = None,
    autocast: bool = False, 
    save_path: str | None = None,
    meta_data: dict = {},
):
    pb_was_enabled = is_progress_bar_enabled()
    disable_progress_bar()
    
    if benchmarks is None:
        benchmarks = list(BENCHMARK_DICT.keys())
    print(f"\nEvaluating on {len(benchmarks)} benchmarks:")
    for i in range(len(benchmarks)):
        print(f"  {i+1}. {benchmarks[i]}")

    if save_path is None:
        save_path = os.path.join(constants.LOCAL_DATA_PATH, "evaluation_results")
    os.makedirs(save_path, exist_ok=True)
    print(f"\nEvaluation results will be saved to: {save_path}")

    for i, benchmark_name in enumerate(benchmarks):
        if benchmark_name not in BENCHMARK_DICT:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
        
        print(f"\nStarting benchmark {i+1}/{len(benchmarks)}: {benchmark_name}")
        bench = BENCHMARK_DICT[benchmark_name](
            tokenizer,
            max_input_length,
            max_output_length,
            max_examples,
        )
        print(f"Number of examples: {len(bench)}")
        loader = torch.utils.data.DataLoader(
            bench,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=bench.collate_fn,
            drop_last=False,
        )

        seen = 0
        correct = 0
        with tqdm(loader, desc=f"{benchmark_name} ({i+1}/{len(benchmarks)})") as pbar:

            pbar.set_postfix({"seen": 0, "correct": 0, "acc": "N/A"})

            for batch in loader:

                with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
                    logits = model.get_logits(
                        batch["input_ids"],
                        batch["output_ids"],
                        # verbose=True,
                    )

                grade = bench.grade(batch, logits)

                seen += batch["input_ids"].shape[0]
                correct += grade.sum().item()

                pbar.update(1)
                pbar.set_postfix({
                    "seen": f"{seen:_}",
                    "correct": f"{correct:_}",
                    "acc": f"{100*correct/seen:.1f}",
                })
            
        results = {
            "seen": seen,
            "correct": correct,
            "accuracy": correct / seen,
            "meta": {
                "model": model.__class__.__name__,
                "tokenizer": tokenizer.__class__.__name__,
                "max_input_length": max_input_length,
                "max_output_length": max_output_length,
                "batch_size": batch_size,
                "autocast": autocast,
                "max_examples": max_examples,
            } | meta_data
        }
        with open(os.path.join(save_path, f"{benchmark_name}.json"), "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nFinished benchmark {i+1}/{len(benchmarks)}: {benchmark_name}")
        print(f"  seen: {seen:_}")
        print(f"  correct: {correct:_}")
        print(f"  accuracy: {100*correct/seen:.1f}%")

    print(f"\nFinished all benchmarks. Results saved to: {save_path}\n")

    if pb_was_enabled:
        enable_progress_bar()