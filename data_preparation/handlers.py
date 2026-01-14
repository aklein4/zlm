
import sys
import inspect

import datasets
import random

from base_handler import BaseHandler


""" ===== Helpers ===== """


def get_splits(url, subset, remove=[]):
    all_splits = list(datasets.get_dataset_split_names(url, subset))
    for s in remove:
        assert s in all_splits, f"Expected split {s} not found in dataset."
        all_splits.remove(s)
    return all_splits


def convert_role(role):
    if role in ["system", "user", "assistant"]:
        return role
    if role == "human":
        return "user"
    if role == "gpt":
        return "assistant"
    return role


def remove_think(content):
    if "</think>" in content:
        return content.split("</think>")[-1].strip()
    return content


def format_chat(data):
    try:
        roles = [convert_role(msg["role"]) for msg in data]
    except:
        roles = [convert_role(msg["from"]) for msg in data]
    try:
        content = [remove_think(msg["content"]).strip() for msg in data]
    except:
        content = [remove_think(msg["value"]).strip() for msg in data]

    if roles[:2] == ["user", "assistant"]:
        return f"<|im_start|>user\n{content[0]}<|im_end|>\n<|im_start|>assistant\n", content[1]+"<|im_end|>"

    if roles[:3] == ["system", "user", "assistant"]:
        if content[0] == "":
            return f"<|im_start|>user\n{content[1]}<|im_end|>\n<|im_start|>assistant\n", content[2]+"<|im_end|>"

        return f"<|im_start|>system\n{content[0]}<|im_end|>\n<|im_start|>user\n{content[1]}<|im_end|>\n<|im_start|>assistant\n", content[2]+"<|im_end|>"

    return None, None


def format_chat_paired(x, y, system=None):
    if x is None or y is None:
        return None, None

    messages = [
        {"role": "user", "content": x},
        {"role": "assistant", "content": y},
    ]

    if system is not None:
        messages = [{"role": "system", "content": system}] + messages

    return format_chat(messages)


def format_no_cot(x, y, answer):
    y = remove_think(y)
    return format_chat(
        [
            {"role": "system", "content": "Place the final answer to the following question inside of a \\boxed{} command. This must appear at the start of your response before any other text."},
            {"role": "user", "content": x},
            {"role": "assistant", "content": "\\boxed{"+str(answer)+"}"+f"\n{y}"},
        ]
    )


def format_cot(x, y, answer):
    y = remove_think(y)
    return format_chat(
        [
            {"role": "system", "content": "Place the final answer to the following question inside of a \\boxed{} command. This must come at the end of your response, and no other text should come after it."},
            {"role": "user", "content": x},
            {"role": "assistant", "content": f"{y}\nFinal answer: \\boxed"+"{"+str(answer)+"}"},
        ]
    )


def none_filter(example, keys):
    for key in keys:
        if example[key] is None:
            return False
    return True


def length_filter(
    example,
    max_input_characters,
    max_output_characters,
    min_input_characters=0,
    min_output_characters=0,
):
    if not none_filter(example, ["input", "output"]):
        return False
    return (
        len(example["input"]) <= max_input_characters and
        len(example["output"]) <= max_output_characters and
        len(example["input"]) >= min_input_characters and
        len(example["output"]) >= min_output_characters
    )


""" ===== Post-Training ===== """


class NemotronPTHandler(BaseHandler):

    url = "nvidia/Nemotron-Post-Training-Dataset-v2"
    subset = None
    split = ["stem", "chat", "math", "code"]

    kind = "post_training"
    default_format = "chat"

    def map(self, example):
        return format_chat(example["messages"])

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class LlamaNemotronPTHandler(BaseHandler):

    url = "nvidia/Llama-Nemotron-Post-Training-Dataset"
    subset = "SFT"
    split = get_splits(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "SFT"
    )

    kind = "post_training"
    default_format = "chat"

    def map(self, example):
        return format_chat_paired(
            example["input"][0]["content"],
            example["output"]
        )

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


""" ===== Chat ===== """


class SmolTalkHandler(BaseHandler):

    url = "HuggingFaceTB/smoltalk"
    subset = "all"
    split = "train"

    kind = "chat"
    default_format = "chat"

    def map(self, example):
        return format_chat(example["messages"])

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class SmolTalk2Handler(BaseHandler):

    url = "HuggingFaceTB/smoltalk2"
    subset = "SFT"
    split = get_splits(
        "HuggingFaceTB/smoltalk2",
        "SFT",
        [
            "LongAlign_64k_Qwen3_32B_yarn_131k_think",
            "aya_dataset_Qwen3_32B_think",
            "smoltalk_multilingual8_Qwen3_32B_think",
            "LongAlign_64k_context_lang_annotated_lang_6_no_think",
            "smoltalk_multilingual_8languages_lang_5_no_think",   
        ],
    )

    kind = "chat"
    default_format = "chat"

    def map(self, example):
        return format_chat(example["messages"])

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class HermesHandler(BaseHandler):

    url = "teknium/OpenHermes-2.5"
    subset = None
    split = get_splits(
        "teknium/OpenHermes-2.5",
        None
    )

    kind = "chat"
    default_format = "chat"

    def map(self, example):
        return format_chat(example["conversations"])

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


# class SynthQuestionsHandler(BaseHandler):

#     url = "IgnoraZ/SynthQuestions"
#     subset = None
#     split = ["synthquestions", "realquestions"]

#     kind = "chat"
#     default_format = "chat"

#     def map(self, example):
#         return format_chat(example["message"])

#     def filter(self, example):
#         return length_filter(
#             example,
#             self.max_input_characters,
#             self.max_output_characters,
#         )


class WebInstructHandler(BaseHandler):

    url = "TIGER-Lab/WebInstructSub"
    subset = None
    split = "train"

    kind = "chat"
    default_format = "chat"

    def map(self, example):
        return format_chat_paired(
            example["question"],
            example["answer"],
        )

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class NemotronIFHandler(BaseHandler):

    url = "nvidia/Nemotron-Instruction-Following-Chat-v1"
    subset = None
    split = "chat_if"

    kind = "chat"
    default_format = "chat"

    def map(self, example):
        return format_chat(example["messages"])

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


""" ===== Reasoning ===== """


class NaturalReasoningHandler(BaseHandler):

    url = "facebook/natural_reasoning"
    subset = None
    split = "train"

    kind = "reasoning"
    default_format = "chat"

    def map(self, example):
        return format_chat_paired(
            example["question"],
            example["responses"][0]["response"]
        )

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class AceReasonHandler(BaseHandler):

    url = "nvidia/AceReason-1.1-SFT"
    subset = None
    split = "train"

    kind = "reasoning"
    default_format = "chat"

    def map(self, example):
        return format_chat_paired(
            example["input"],
            example["output"]
        )

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class AceMathHandler(BaseHandler):

    url = "nvidia/AceMath-Instruct-Training-Data"
    subset = None
    split = get_splits(
        "nvidia/AceMath-Instruct-Training-Data",
        None,
    )

    kind = "reasoning"
    default_format = "chat"

    verification_mode = datasets.VerificationMode.NO_CHECKS

    def map(self, example):
        return format_chat(
            example["messages"] + [{"role": "assistant", "content": example["answer"]}]
        )

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


""" ===== MCQA ===== """


def extract_openscience_answer(x):
    answer = x.split("\\boxed{")[-1].split("}")[0].strip().upper()

    if len(answer) != 1 or not answer.isalpha():
        return None

    after = x.split("\\boxed{")[-1]
    if len(after) > 10:
        return None

    return answer


def random_mcqa_format(x, y, answer):
    choice = random.choice(["cot", "no_cot", "none"])

    if choice == "cot":
        return format_cot(x, y, answer) + ("mcqa_cot",)
    elif choice == "no_cot":
        return format_no_cot(x, y, answer) + ("mcqa_no_cot",)
    else:
        return format_chat_paired(x, y)


class OpenScienceHandler(BaseHandler):

    url = "nvidia/OpenScience"
    subset = [
        "OS-Q2.5-32B-10",
        "OS-Q2.5-32B-4",
        "OS-Q2.5-72B-10",
        "OS-Q3-235B-4"
    ]
    split = "train"

    kind = "mcqa"
    default_format = "chat"

    def map(self, example):

        x = example["input"]
        y = remove_think(example["output"])

        answer = extract_openscience_answer(y)
        if answer is None:
            return format_chat_paired(x, y)

        return random_mcqa_format(x, y, answer)         

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )
    

class OpenScienceReasoningHandler(BaseHandler):

    url = "nvidia/OpenScienceReasoning-2"
    subset = None
    split = "train"

    kind = "mcqa"
    default_format = "chat"

    def map(self, example):
        return random_mcqa_format(
            example["input"],
            example["output"],
            example["expected_answer"],
        )
                
    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class SciqHandler(BaseHandler):

    url = "allenai/sciq"
    subset = None
    split = "train"

    kind = "mcqa"
    default_format = "chat"

    def map(self, example):
        
        options = [
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
            example["correct_answer"]
        ]
        random.shuffle(options)

        correct_ind = options.index(example["correct_answer"])
        answer = ["A", "B", "C", "D"][correct_ind]

        x = example["question"]+"\n"
        for i, opt in enumerate(options):
            x += f"{['A', 'B', 'C', 'D'][i]}: {opt}\n"

        return random_mcqa_format(
            x,
            example["support"],
            answer,
        )

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


""" ===== Math ===== """


def random_math_format(x, y, answer):
    choice = random.choice(["cot", "no_cot", "none"])

    if choice == "cot":
        return format_cot(x, y, answer) + ("math_cot",)
    elif choice == "no_cot":
        return format_no_cot(x, y, answer) + ("math_no_cot",)
    else:
        return format_chat_paired(x, y)


class StackMathHandler(BaseHandler):

    url = "math-ai/StackMathQA"
    subset = "stackmathqa800k"
    split = "train"

    kind = "math"
    default_format = "chat"

    def map(self, example):
        return format_chat_paired(
            example["Q"],
            example["A"]
        )    

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class MetaMathHandler(BaseHandler):

    url = "meta-math/MetaMathQA"
    subset = None
    split = "train"

    kind = "math"
    default_format = "chat"

    def map(self, example):

        if "The answer is:" not in example["response"]:
            return format_chat_paired(
                example["query"],
                example["response"]
            )
        
        answer = example["response"].split("The answer is:")[-1].strip()
        
        return random_math_format(
            example["query"],
            example["response"],
            answer,
        )        

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class MathPlusHandler(BaseHandler):

    url = "TIGER-Lab/MATH-plus"
    subset = None
    split = "train"

    kind = "math"
    default_format = "chat"

    def map(self, example):

        if "The answer is" not in example["output"]:
            return format_chat_paired(
                example["instruction"],
                example["output"]
            )
        
        answer = example["output"].split("The answer is")[-1].strip()
        if len(answer) == 0:
            return format_chat_paired(
                example["instruction"],
                example["output"]
            )
        
        if answer[-1] == ".":
            answer = answer[:-1]

        return random_math_format(
            example["instruction"],
            example["output"],
            answer,
        )        

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class OpenMathInstruct1Handler(BaseHandler):

    url = "nvidia/OpenMathInstruct-1"
    subset = None
    split = "train"

    kind = "math"
    default_format = "chat"

    def map(self, example):
        if not example["is_correct"]:
            return None, None

        return random_math_format(
            example["question"],
            example["generated_solution"],
            example["expected_answer"]
        )        

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class OpenMathInstruct2Handler(BaseHandler):

    url = "nvidia/OpenMathInstruct-2"
    subset = None
    split = "train_2M" # use a smaller subset because idk how good the quality is

    kind = "math"
    default_format = "chat"

    def map(self, example):
        return random_math_format(
            example["problem"],
            example["generated_solution"],
            example["expected_answer"]
        )        

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class PrismMathHandler(BaseHandler):

    url = "nvidia/Nemotron-PrismMath"
    subset = None
    split = "train"

    kind = "math"
    default_format = "chat"

    def map(self, example):
        return format_chat_paired(
            example["problem"],
            example["solution"]
        )        

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class ODAMathHandler(BaseHandler):

    url = "OpenDataArena/ODA-Math-460k"
    subset = None
    split = "train"

    kind = "math"
    default_format = "chat"

    def map(self, example):
        return random_math_format(
            example["question"],
            example["response"],
            example["expected_answer"],
        )        

    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


""" ===== Code ===== """


class OpenCodeReasoningHandler(BaseHandler):

    url = "nvidia/OpenCodeReasoning"
    subset = "split_0"
    split = "split_0"

    kind = "code"
    default_format = "chat"

    def map(self, example):
        sol = example["solution"].strip()
        return format_chat_paired(
            example["input"],
            f"```python\n{sol}\n```"
        )
                
    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


LEAN_PROMPTS = [
    "Use Lean to formally prove the following statement.",
    "Use Lean to formally prove the following statement.",
    "Construct a formal proof in Lean for the following mathematical claim.",
    "Develop a rigorous proof in Lean for the stated mathematical proposition.",
    "Write the proof of the following mathematical statement in Lean.",
    "Demonstrate the proof of the given mathematical assertion using Lean.",
    "Formulate a detailed proof in Lean for the following mathematical claim.",
    "Using Lean, provide a formal proof for the stated mathematical proposition.",
    "",
    "",
]


class NemotronMathProofsHandler(BaseHandler):

    url = "nvidia/Nemotron-Math-Proofs-v1"
    subset = None
    split = "lean"

    kind = "code"
    default_format = "chat"

    def map(self, example):

        system = random.choice(LEAN_PROMPTS)

        head = example["lean_header"].strip()+"\n\n" if example["lean_header"] is not None else ""
        statement = example["formal_statement"].strip() if example["formal_statement"] is not None else None

        return format_chat_paired(
            example["problem"],
            f"```lean\n{head}{statement}\n```" if statement is not None else None,
            system=system,
        )
                
    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class TinyCodesHandler(BaseHandler):

    url = "nampdn-ai/tiny-codes"
    subset = None
    split = "train"

    kind = "code"
    default_format = "chat"

    def map(self, example):
        if "```" not in example["response"]:
            return None, None

        code = "```" + example["response"].split("```")[1].strip() + "\n```"

        return format_chat_paired(
            example["prompt"],
            code
        )
                
    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


class GoedelHandler(BaseHandler):

    url = "Goedel-LM/Goedel-Pset-v1"
    subset = None
    split = "train"

    kind = "code"
    default_format = "chat"

    def map(self, example):

        statement = example["formal_statement"].strip()
        system = random.choice(LEAN_PROMPTS)

        return format_chat_paired(
            example["informal_statement"],
            f"```lean\n{statement}\n```",
            system=system,
        )
                
    def filter(self, example):
        return length_filter(
            example,
            self.max_input_characters,
            self.max_output_characters,
        )


HANDLERS = [x[1] for x in inspect.getmembers(sys.modules[__name__]) if inspect.isclass(x[1]) and issubclass(x[1], BaseHandler) and x[1] is not BaseHandler][::-1]
