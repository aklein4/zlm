
import datasets
import random
from functools import partial


MMLU_REPO = "cais/mmlu"
SCIQ_REPO = "allenai/sciq"
ARC_REPO = "allenai/ai2_arc"

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def letter2index(letter):
    return LETTERS.index(letter)

def index2letter(index):
    try:
        return LETTERS[index]
    except:
        raise ValueError(f"Index {index} is out of range for letter mapping.")


def format_question(question, options):
    out = f"Question:\n{question}\nChoices:"
    
    for i in range(len(options)):
        out += f"\n{LETTERS[i]}. {options[i]}"

    return out


def mmlu_map(example):
    example = example["train"]

    options = [c.strip() for c in example["choices"]]
    correct_letter = index2letter(example["answer"])

    return {
        "source": "mmlu",
        "question": format_question(example["question"].strip(), options),
        "answer": correct_letter,
    }


def sciq_map(example):
    
    options = [example["distractor1"], example["distractor2"], example["distractor3"], example["correct_answer"]]
    options = [option.strip() for option in options]

    random.shuffle(options)

    correct_index = options.index(example["correct_answer"].strip())
    correct_letter = index2letter(correct_index)

    return {
        "source": "sciq",
        "question": format_question(example["question"].strip(), options),
        "answer": correct_letter,
    }


def arc_map(example, source="arc"):

    options = example["choices"]["text"]
    options = [option.strip() for option in options]

    answer = example["answerKey"].strip()
    if not answer.isalpha():
        answer = index2letter(int(answer))

    return {
        "source": source,
        "question": format_question(example["question"].strip(), options),
        "answer": example["answerKey"].strip(),
    }



def main():
    
    mmlu = datasets.load_dataset(MMLU_REPO, "auxiliary_train", split="train")
    sciq = datasets.load_dataset(SCIQ_REPO, split="train")
    arc_e = datasets.load_dataset(ARC_REPO, "ARC-Easy", split="train")
    arc_c = datasets.load_dataset(ARC_REPO, "ARC-Challenge", split="train")

    mmlu = mmlu.map(mmlu_map, remove_columns=mmlu.column_names)
    sciq = sciq.map(sciq_map, remove_columns=sciq.column_names)
    arc_e = arc_e.map(partial(arc_map, source="arc_easy"), remove_columns=arc_e.column_names)
    arc_c = arc_c.map(partial(arc_map, source="arc_challenge"), remove_columns=arc_c.column_names)

    combined = datasets.concatenate_datasets([mmlu, sciq, arc_e, arc_c])
    shuffled = combined.shuffle(seed=42)

    shuffled.push_to_hub(
        "mcqa_compilation"
    )


if __name__ == "__main__":
    main()