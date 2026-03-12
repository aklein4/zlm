import torch

import argparse
import os

from transformers import AutoTokenizer

from models import load_checkpoint
from evaluation import run_benchmarks, BENCHMARK_DICT
import utils.constants as constants


@torch.no_grad()
def main(args):
    assert constants.DEVICE.type == "cuda", "Evaluation currently only supports CUDA devices."

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"\nLoading model {args.checkpoint_url} at step {args.checkpoint_step}...")
    model = load_checkpoint(
        args.checkpoint_url, args.checkpoint_step,
        attention_kernel="gpu_flash_attention",
    ).to(constants.DEVICE)
    model.eval()

    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
    )

    if args.save_path is None:
        save_path = os.path.join(
            constants.LOCAL_DATA_PATH,
            "evaluation_results",
            args.checkpoint_url.replace("/", "--"),
            f"{args.checkpoint_step:012d}",
        )
    else:
        save_path = args.save_path
    
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
        save_path=save_path,
        meta_data={
            "seed": args.seed,
        }
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the model on various benchmarks.")
    parser.add_argument(
        "--checkpoint_url",
        type=str,
        help="The URL of the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        help="The training step of the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="The path to the tokenizer to use for evaluation.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="The maximum input length for the benchmarks.",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=512,
        help="The maximum output length for the benchmarks.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The batch size to use for evaluation.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="The maximum number of examples to evaluate on for each benchmark. If not specified, evaluates on the entire benchmark.",
    )
    parser.add_argument(
        "--no_autocast",
        action="store_true",
        help="Whether to use NOT autocast for evaluation.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The path to save the evaluation results. If not specified, saves to a default location.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed to use for evaluation.",
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
