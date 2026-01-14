import torch

import numpy as np
from functools import partial

import datasets
from transformers import LlamaTokenizerFast
import matplotlib.pyplot as plt

from constants import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH


TOKENIZER = './tokenizer'


ANSWER_INSTRUCTIONS = [
    "For the following question, provide only your final answer. Then explain your reasoning.",
    "Provide your final answer to the question, followed by an explanation of your reasoning.",
    "Give your final answer to the question, and then explain how you arrived at the answer.",
    "For the question below, provide your final answer and then explain how you got the answer.",
    "Provide your final answer to the question, and then explain your reasoning.",
    "For the question below, give your final answer and then explain your reasoning.",
    "For the question below, provide your answer followed by an explanation of your reasoning.",
    "First provide your final answer to the question, then explain your reasoning.",
    "For the question below, first provide your final answer, then justify your answer.",
    "Provide your answer to the question below, and then explain how you arrived at that answer.",
]

COT_INSTRUCTIONS = [
    "For the following question, provide a step-by-step explanation of your reasoning, with your final answer at the end.",
    "Explain your reasoning step-by-step for the following question, and provide your final answer at the end.",
    "Work through the following question step-by-step, and place your final answer at the end of your explanation.",
    "For the question below, provide a detailed step-by-step explanation of your reasoning with your final answer at the end.",
    "For this question, place your answer at the end of a step-by-step explanation of your reasoning.",
    "First work through the question step-by-step, then provide your final answer.",
    "Provide a step-by-step explanation of your reasoning for the question below, with your final answer at the end.",
    "For the question below, thoroughly explain your reasoning before giving you answer.",
    "Explain your reasoning step-by-step for the question below, then provide your final answer.",
    "This question requires a step-by-step explanation of your reasoning, ending with the final answer.",
]


INPUT_REPO = "aklein4/math-formatted"
OUTPUT_REPO = "aklein4/math-tokenized"


def format_example(example, kind=""):

    if kind == "answer":
        instruction = np.random.choice(ANSWER_INSTRUCTIONS)
        question = example["question"]
        answer = example["answer"]
        explanation = example["explanation"]

        if np.random.rand() < 0.5:
            inp = f"Instructions:\n{instruction}\nQuestion:\n{question}\nAnswer:\n"
        else:
            inp = f"Question:\n{question}\nAnswer:\n"

        out_suffix = f"\nExplanation:\n{explanation}"

        return inp, answer, out_suffix

    elif kind == "cot":
        instruction = np.random.choice(COT_INSTRUCTIONS)
        question = example["question"]
        explanation = example["explanation"]

        if np.random.rand() < 0.5:
            inp = f"Instructions:\n{instruction}\nQuestion:\n{question}\nAnswer:\n"
        else:
            inp = f"Question:\n{question}\nAnswer:\n"

        out_suffix = explanation

        return inp, out_suffix

    else:
        raise ValueError(f"Unknown example kind: {kind}")


def answer_map(example, tokenizer: LlamaTokenizerFast=None):
    num_examples = len(example["source"])
    
    inputs = []
    answers = []
    outputs = []
    for i in range(num_examples):
        inp, answer, out_suffix = format_example(
            {k: example[k][i] for k in example.keys()},
            kind="answer"
        )
        inputs.append(inp)
        answers.append(answer)
        outputs.append(out_suffix)

    input_tokens = tokenizer(
        inputs,
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH+10
    ).input_ids

    answer_tokens = tokenizer(
        answers,
        add_special_tokens=False,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_OUTPUT_LENGTH+10
    ).input_ids
    output_tokens = tokenizer(
        outputs,
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_OUTPUT_LENGTH+10
    ).input_ids

    assert np.max(input_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    assert np.max(answer_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    assert np.max(output_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    input_tokens = input_tokens.astype(np.uint16)
    answer_tokens = answer_tokens.astype(np.uint16)
    output_tokens = output_tokens.astype(np.uint16)

    input_tokens = [
        t[t != tokenizer.pad_token_id][:-1] for t in input_tokens  # Exclude pad and EOS
    ]
    answer_tokens = [
        t[t != tokenizer.pad_token_id] for t in answer_tokens # Exclude pad
    ]
    output_tokens = [
        t[t != tokenizer.pad_token_id][1:] for t in output_tokens  # Exclude pad and BOS
    ]

    output_tokens = [
        np.concatenate([a, o], axis=0) for a, o in zip(answer_tokens, output_tokens)
    ]

    return {
        "input_ids": input_tokens,
        "output_ids": output_tokens,
        "num_input_tokens": [len(t) for t in input_tokens],
        "num_output_tokens": [len(t) for t in output_tokens],
    }


def cot_map(example, tokenizer: LlamaTokenizerFast=None):
    num_examples = len(example["source"])

    inputs = []
    outputs = []
    for i in range(num_examples):
        inp, out_suffix = format_example(
            {k: example[k][i] for k in example.keys()},
            kind="cot"
        )
        inputs.append(inp)
        outputs.append(out_suffix)

    input_tokens = tokenizer(
        inputs,
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH+10
    ).input_ids

    output_tokens = tokenizer(
        outputs,
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_OUTPUT_LENGTH+10
    ).input_ids

    assert np.max(input_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    assert np.max(output_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    input_tokens = input_tokens.astype(np.uint16)
    output_tokens = output_tokens.astype(np.uint16)

    input_tokens = [
        t[t != tokenizer.pad_token_id][:-1] for t in input_tokens  # Exclude pad and EOS
    ]
    output_tokens = [
        t[t != tokenizer.pad_token_id][1:] for t in output_tokens  # Exclude pad and BOS
    ]

    return {
        "input_ids": input_tokens,
        "output_ids": output_tokens,
        "num_input_tokens": [len(t) for t in input_tokens],
        "num_output_tokens": [len(t) for t in output_tokens],
    }


def main():

    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(TOKENIZER)

    dataset = datasets.load_dataset(INPUT_REPO, split="train")

    answer_dataset = dataset.filter(
        lambda x: x["answer"] is not None
    )
    cot_dataset = dataset.filter(
        lambda x: x["answer"] is None
    )

    print(f"Number of answer examples: {len(answer_dataset)}")
    print(f"Number of COT examples: {len(cot_dataset)}")

    answer_dataset = answer_dataset.map(
        partial(answer_map, tokenizer=tokenizer),
        remove_columns=["question", "answer", "explanation"],
        batched=True,
        batch_size=1000,
    )
    cot_dataset = cot_dataset.map(
        partial(cot_map, tokenizer=tokenizer),
        remove_columns=["question", "answer", "explanation"],
        batched=True,
        batch_size=1000,
    )

    combined = datasets.concatenate_datasets(
        [
            answer_dataset,
            cot_dataset,
        ]
    )
    combined = combined.filter(
        lambda x: x["num_input_tokens"] <= MAX_INPUT_LENGTH and x["num_output_tokens"] <= MAX_OUTPUT_LENGTH
    )
    combined = combined.shuffle(seed=42)

    combined.push_to_hub(OUTPUT_REPO, split="train")

    input_lengths = np.array(combined["num_input_tokens"])
    output_lengths = np.array(combined["num_output_tokens"])
    total_lengths = np.array([i + o for i, o in zip(input_lengths, output_lengths)])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].hist(total_lengths, bins=100)
    ax[0].set_title("Total Token Lengths")
    ax[0].grid(True)

    ax[1].hist(input_lengths, bins=100)
    ax[1].set_title("Input Token Lengths")
    ax[1].grid(True)

    ax[2].hist(output_lengths, bins=100)
    ax[2].set_title("Output Token Lengths")
    ax[2].grid(True)

    plt.tight_layout()
    plt.savefig("math_length_distributions.png")


if __name__ == "__main__":

    # combined = datasets.load_dataset(OUTPUT_REPO, split="train")

    # input_lengths = np.array(combined["num_input_tokens"])
    # output_lengths = np.array(combined["num_output_tokens"])
    # total_lengths = np.array([i + o for i, o in zip(input_lengths, output_lengths)])

    # fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # ax[0].hist(total_lengths, bins=100)
    # ax[0].set_title("Total Token Lengths")
    # ax[0].grid(True)

    # ax[1].hist(input_lengths, bins=100)
    # ax[1].set_title("Input Token Lengths")
    # ax[1].grid(True)

    # ax[2].hist(output_lengths, bins=100)
    # ax[2].set_title("Output Token Lengths")
    # ax[2].grid(True)

    # plt.tight_layout()
    # plt.savefig("math_length_distributions.png")

    main()
