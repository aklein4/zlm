import torch

import os
from tqdm import tqdm
import json

from transformers.utils.logging import disable_progress_bar, enable_progress_bar, is_progress_bar_enabled

from evaluation.benchmarks import BENCHMARK_DICT, DEFAULT_BENCHMARK_DICT
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
    model_kwargs: dict = {},
    benchmark_kwargs: dict = {},
    save_path: str | None = None,
    seed: int = 42,
    meta_data: dict = {},
):
    assert constants.DEVICE.type == "cuda", "Evaluation currently only supports CUDA devices."

    pb_was_enabled = is_progress_bar_enabled()
    disable_progress_bar()
    
    if benchmarks is None:
        benchmarks = list(DEFAULT_BENCHMARK_DICT.keys())
    print(f"\nEvaluating on {len(benchmarks)} benchmarks:")
    for i in range(len(benchmarks)):
        print(f"  {i+1}. {benchmarks[i]}")

    if save_path is None:
        save_path = "evaluation_results"
    save_path = os.path.join(constants.LOCAL_DATA_PATH, save_path)
    print(f"\nEvaluation results will be saved to: {save_path}")  

    all_results = {}
    for i, benchmark_name in enumerate(benchmarks): 
        if benchmark_name not in BENCHMARK_DICT:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print(f"\nStarting benchmark {i+1}/{len(benchmarks)}: {benchmark_name}")
        benchmark_cls = BENCHMARK_DICT[benchmark_name]
        if getattr(benchmark_cls, "requires_batch_size_one", False) and batch_size != 1:
            raise ValueError(f"{benchmark_name} requires batch_size=1, got {batch_size}.")
        
        curr_bench_kwargs = benchmark_kwargs.get(benchmark_name, {})
        if not isinstance(curr_bench_kwargs, dict):
            raise ValueError(f"benchmark_kwargs for {benchmark_name} must be a dict.")

        bench = benchmark_cls(
            tokenizer,
            max_input_length,
            max_output_length,
            max_examples,
            **curr_bench_kwargs,
        )
        print(f"  Number of examples: {len(bench)}")
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
                        **model_kwargs,
                    )

                grade = bench.grade(batch, logits)
                if isinstance(grade, torch.Tensor):
                    grade = grade.sum().item()

                seen += batch["input_ids"].shape[0]
                correct += grade

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
                "model_kwargs": model_kwargs,
                "benchmark_kwargs": curr_bench_kwargs,
                "seed": seed,
            } | meta_data
        }

        all_results[benchmark_name] = {
            "seen": seen,
            "correct": correct,
            "accuracy": correct / seen,
        }

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"{benchmark_name}.json"), "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nFinished benchmark {i+1}/{len(benchmarks)}: {benchmark_name}")
        print(f"  seen: {seen:_}")
        print(f"  correct: {correct:_}")
        print(f"  accuracy: {100*correct/seen:.1f}%")

    # summarize all results
    with open(os.path.join(save_path, "summary.json"), "w") as f:
        summary = {
            "meta": {
                "model": model.__class__.__name__,
                "tokenizer": tokenizer.__class__.__name__,
                "max_input_length": max_input_length,
                "max_output_length": max_output_length,
                "batch_size": batch_size,
                "autocast": autocast,
                "max_examples": max_examples,
                "model_kwargs": model_kwargs,
                "benchmark_kwargs": benchmark_kwargs,
                "seed": seed,
            } | meta_data,
            "results": all_results,
        }
        json.dump(summary, f, indent=4)

    print(f"\nFinished all benchmarks. Results saved to: {save_path}\n")

    if pb_was_enabled:
        enable_progress_bar()
