import torch

import numpy as np
from functools import partial

import datasets
from transformers import LlamaTokenizerFast
import matplotlib.pyplot as plt

from constants import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH


TOKENIZER = './tokenizer'

INSTRUCTIONS = [
    "For the following multiple choice question, provide the letter of the correct answer followed by an explanation of your reasoning.",
    "Give the letter of the correct answer to this question, followed by an explanation of why the answer is correct.",
    "For the question below, provide the letter of the correct answer and an explanation of why it is correct.",
    "For this question, give the letter of the correct answer and then explain your reasoning.",
    "Select the letter of the correct answer to the question below, and explain why it is correct.",
    "Choose the letter of the correct answer to the question, and provide an explanation of why it is correct.",
    "Identify the letter of the correct answer to the question, and explain why it is the correct choice.",
    "Determine the letter of the correct answer to the question, and provide an explanation of your choice.",
    "Select the letter of the correct answer to the question, and explain your reasoning.",
    "The following question has multiple choice answers. Provide the letter of the correct answer and an explanation of why it is correct.",
    "This question has multiple choice answers. Give the letter of the correct answer and an explanation of why it is correct.",
    "This is a multiple choice question. Provide the letter of the correct answer and explain your reasoning.",
    "For this multiple choice question, give your answer as a letter, then justify your answer.",
    "Justify your answer to the following multiple choice question after providing the your answer as a letter.",
    "Provide the letter of the correct answer to the question, and explain your reasoning in detail.",
    "For the question below, first provide the letter of the correct answer. Then, explain why it is the correct choice.",
    "For the question below, first provide the letter of the correct answer, then explain your reasoning.",
    "This is a multiple choice question. Provide the letter of the correct answer followed by an explanation.",
    "The question below is multiple choice. Choose the correct answer and provide it as a letter, followed by an explanation.",
    "You must provide the letter of the correct answer to the question, followed by an explanation of your reasoning."
]

INPUT_REPO = "aklein4/mcqa-formatted"
OUTPUT_REPO = "aklein4/mcqa-tokenized"


def format_example(example):
    instruction = np.random.choice(INSTRUCTIONS)
    question = example["question"]
    answer = example["answer"]
    explanation = example["explanation"]

    if np.random.rand() < 0.5:
        inp = f"Instructions:\n{instruction}\n{question}\nAnswer:\n"
    else:
        inp = f"{question}\nAnswer:\n"
    
    out_suffix = f"\nExplanation:\n{explanation}"

    return inp, out_suffix


def tokenize_example(example, tokenizer: LlamaTokenizerFast=None):
    inp, out_suffix = format_example(example)

    input_tokens = tokenizer.encode(
        inp,
        add_special_tokens=True,
        return_tensors="np",
        truncation=True,
        max_length=MAX_INPUT_LENGTH+10
    )[0, :-1]  # Exclude the last token (EOS)

    answer_token = tokenizer.encode(
        example["answer"],
        add_special_tokens=False,
        return_tensors="np",
        truncation=True,
        max_length=MAX_OUTPUT_LENGTH+10
    )[0]
    output_tokens = tokenizer.encode(
        out_suffix,
        add_special_tokens=True,
        return_tensors="np",
        truncation=True,
        max_length=MAX_OUTPUT_LENGTH+10
    )[0, 1:]  # Exclude the first token (BOS)
    output_tokens = np.concatenate([answer_token, output_tokens])

    # input_text = tokenizer.decode(input_tokens, skip_special_tokens=False)
    # output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

    assert np.max(input_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    assert np.max(output_tokens) < 2**16, "Token IDs exceed the maximum value for uint16."
    input_tokens = input_tokens.astype(np.uint16)
    output_tokens = output_tokens.astype(np.uint16)

    return {
        "input_ids": input_tokens,
        "output_ids": output_tokens,
        "num_input_tokens": len(input_tokens),
        "num_output_tokens": len(output_tokens),
    }


def main():

    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(TOKENIZER)

    dataset = datasets.load_dataset(INPUT_REPO, split="train")

    dataset = dataset.map(
        partial(tokenize_example, tokenizer=tokenizer),
        remove_columns=["question", "answer", "explanation"],
    )
    dataset = dataset.filter(
        lambda x: x["num_input_tokens"] <= MAX_INPUT_LENGTH and x["num_output_tokens"] <= MAX_OUTPUT_LENGTH
    )

    dataset.push_to_hub(OUTPUT_REPO, split="train")

    input_lengths = np.array(dataset["num_input_tokens"])
    output_lengths = np.array(dataset["num_output_tokens"])
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
    plt.savefig("mcqa_length_distributions.png")


if __name__ == "__main__":
    main()
