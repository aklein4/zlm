import torch

import argparse
import os
import json

from transformers import AutoTokenizer

from models import load_checkpoint
from evaluation import run_benchmarks, BENCHMARK_DICT
import utils.constants as constants


@torch.no_grad()
def main(args):
    assert constants.DEVICE.type == "cuda", "Evaluation currently only supports CUDA devices."

    print(f"\nLoading model {args.checkpoint_url} at step {args.checkpoint_step}...")
    model = load_checkpoint(
        args.checkpoint_url, args.checkpoint_step,
        strict=(not args.checkpoint_not_strict),
        attention_kernel="gpu_flash_attention",
    ).to(constants.DEVICE)
    model.eval()

    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
    )

    save_folder = "evaluation_results"
    if args.save_folder is not None:
        save_folder = args.save_folder
    save_path = os.path.join(
        constants.LOCAL_DATA_PATH,
        save_folder,
        args.checkpoint_url.replace("/", "--"),
        f"{args.checkpoint_step:012d}",
    )
    
    print("\nStarting evaluation...")
    run_benchmarks(
        model,
        tokenizer,
        args.max_input_length,
        args.max_output_length,
        args.batch_size,
        benchmarks=args.benchmarks,
        max_examples=args.max_examples,
        autocast=(not args.no_autocast),
        model_kwargs=args.model_kwargs,
        benchmark_kwargs=args.benchmark_kwargs,
        save_path=save_path,
        seed=args.seed,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the model on various benchmarks.")
    parser.add_argument(
        "--checkpoint-url",
        type=str,
        help="The URL of the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help="The training step of the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--checkpoint-not-strict",
        action="store_true",
        help="Whether to NOT use strict loading when loading the model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=os.path.join(constants.LOCAL_DATA_PATH, "tokenizer"),
        help="The path to the tokenizer to use for evaluation.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=256,
        help="The maximum input length for the benchmarks.",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=512,
        help="The maximum output length for the benchmarks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="The batch size to use for evaluation.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="The maximum number of examples to evaluate on for each benchmark. If not specified, evaluates on the entire benchmark.",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Whether to use NOT autocast for evaluation.",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default=None,
        help="The path to save the evaluation results. If not specified, saves to 'evaluation_results/'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed to use for evaluation.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        default="{}",
        help="A JSON string representing a dictionary of additional keyword arguments to pass to the model during evaluation.",
    )
    parser.add_argument(
        "--benchmark-kwargs",
        type=json.loads,
        default="{}",
        help="A JSON string representing a dictionary of additional keyword arguments to pass to the benchmarks during evaluation.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help=f"List of benchmarks to evaluate on. Defaults to all benchmarks. Available: " + ", ".join(BENCHMARK_DICT.keys()),
    )

    args = parser.parse_args()

    main(args)
