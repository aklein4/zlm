
import datasets


def natural_map(example):
    return {
        "source": 'facebook/natural_reasoning',
        "question": example["question"].strip(),
        "answer": None,
        "explanation": example["responses"][0]["response"].strip(),
    }


def stack_map(example):
    return {
        "source": "math-ai/StackMathQA",
        "question": example["Q"].strip(),
        "answer": None,
        "explanation": example["A"].strip(),
    }


def meta_map(example):

    split = example["response"].split("The answer is:")
    if len(split) < 2:
        answer = None
    else:
        answer = split[-1].strip()

    return {
        "source": "meta-math/MetaMathQA",
        "question": example["query"].strip(),
        "answer": answer,
        "explanation": example["response"].strip(),
    }


def plus_map(example):

    split = example["output"].split("The answer is")
    if len(split) < 2:
        answer = None
    else:
        answer = split[-1].strip()[:-1]  # Remove the period at the end

    return {
        "source": "TIGER-Lab/MATH-plus",
        "question": example["instruction"].strip(),
        "answer": answer,
        "explanation": example["output"].strip(),
    }


def open_map(example):
    return {
        "source": "nvidia/OpenMathInstruct-2",
        "question": example["problem"].strip(),
        "answer": example["expected_answer"].strip(),
        "explanation": example["generated_solution"].strip(),
    }


def main():
    
    natural = datasets.load_dataset("facebook/natural_reasoning", split="train")
    stack = datasets.load_dataset("math-ai/StackMathQA", "stackmathqafull-1q1a", split="train")
    meta = datasets.load_dataset("meta-math/MetaMathQA", split="train")
    plus = datasets.load_dataset("TIGER-Lab/MATH-plus", split="train")
    openmath = datasets.load_dataset("nvidia/OpenMathInstruct-2", split="train_2M")

    natural = natural.map(
        natural_map,
        remove_columns=natural.column_names,
    )
    stack = stack.map(
        stack_map,
        remove_columns=stack.column_names,
    )
    meta = meta.map(
        meta_map,
        remove_columns=meta.column_names,
    )
    plus = plus.map(
        plus_map,
        remove_columns=plus.column_names,
    )
    openmath = openmath.map(
        open_map,
        remove_columns=openmath.column_names,
    )

    combined = datasets.concatenate_datasets(
        [
            natural,
            stack,
            meta,
            plus,
            openmath,
        ]
    )
    shuffled = combined.shuffle(seed=42)

    shuffled.push_to_hub(
        "math-formatted",
    )


if __name__ == "__main__":
    main()