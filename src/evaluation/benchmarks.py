import torch

import numpy as np
import string
import sys
import inspect

import datasets
from transformers import PreTrainedTokenizer

import utils.constants as constants
import utils.chat_utils as chat

BS = 1000


class BaseBenchmark:

    name: str = None

    url: str = None
    subset: str = None
    split: str = None


    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
        max_examples: int | None = None,
    ):
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_examples = max_examples

        self.dataset = self.get_dataset()

    
    def get_dataset(self):

        data = datasets.load_dataset(
            self.url,
            name=self.subset,
            split=self.split,
            streaming=False,
        )

        data = data.map(
            self.data_map_fn,
            batched=True,
            batch_size=BS,
            load_from_cache_file=False,
        )
        data = data.filter(
            self.data_filter_fn,
            batched=True,
            batch_size=BS,
            load_from_cache_file=False,
        )
        data = data.remove_columns("keep")

        self.dataset = data
        data = data.map(
            self.truncate_map_fn,
            batched=True,
            batch_size=BS,
            load_from_cache_file=False,
        )
        self.dataset = None

        if self.max_examples is not None:
            data = data.select(range(min(self.max_examples, len(data))))

        return data


    def tokenize(self, text, in_or_out):

        if in_or_out == "output":
            max_length = self.max_output_length + 1
        elif in_or_out == "input":
            max_length = self.max_input_length + 1
        else:
            raise ValueError(f"in_or_out must be either 'input' or 'output', got {in_or_out}")

        ids = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).input_ids

        keep = ids[:, -1] == self.tokenizer.pad_token_id

        return ids[:, :-1], keep


    def data_map_fn(self, batch):
        """
        Return a dict containing "input_ids", "output_ids", "keep", and other keys.
        """
        raise NotImplementedError("data_map_fn must be implemented by subclasses of BaseBenchmark")


    def data_filter_fn(self, batch):
        return batch["keep"]


    def truncate_map_fn(self, batch):
        l_in = self.largest_input()
        l_out = self.largest_output()

        batch["input_ids"] = [
            ids[:l_in] for ids in batch["input_ids"]
        ]
        batch["output_ids"] = [
            ids[:l_out] for ids in batch["output_ids"]
        ]

        return batch


    def largest_input(self):
        lengths = [
            np.sum(np.array(ids) != self.tokenizer.pad_token_id) for ids in self.dataset["input_ids"]
        ]
        return np.max(lengths)
    
    def largest_output(self):
        lengths = [
            np.sum(np.array(ids) != self.tokenizer.pad_token_id) for ids in self.dataset["output_ids"]
        ]
        return np.max(lengths)
    

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        self._curr_index = 0
        return self

    def __next__(self):
        if self._curr_index >= len(self):
            raise StopIteration

        example = self[self._curr_index]
        self._curr_index += 1

        return example


    def __getitem__(self, idx):
        return {k: self.dataset[k][idx] for k in self.dataset.column_names}


    def collate_fn(self, batch):

        d = {}
        for k in batch[0].keys():
            ex = batch[0][k]

            if isinstance(ex, (int, float, list)):
                
                if isinstance(ex, list):
                    if isinstance(ex[0], (int, float)):
                        d[k] = torch.tensor([example[k] for example in batch], device=constants.DEVICE)
                        continue
                else:
                    try:
                        d[k] = torch.tensor([example[k] for example in batch], device=constants.DEVICE)
                        continue
                    except:
                        pass

            d[k] = [example[k] for example in batch]

        return d


    def grade(self, batch, model_logits):
        """
        Determine whether the model is correct using its output_id logits.
        """
        raise NotImplementedError("grade must be implemented by subclasses of BaseBenchmark")


class MCQABenchmark(BaseBenchmark):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
        max_examples: int | None = None,
    ):
        
        self.prefix_length = len(
            tokenizer(chat.NO_COT_PREFIX, add_special_tokens=False).input_ids
        )
        self.letter_ids = tokenizer(
            list(string.ascii_uppercase),
            add_special_tokens=False,
        ).input_ids
        self.torch_letter_ids = torch.tensor(self.letter_ids, device=constants.DEVICE, dtype=torch.long)[:, 0]

        super().__init__(
            tokenizer,
            max_input_length,
            max_output_length,
            max_examples,
        )


    def extract_example(self, example):
        """
        Should return prompt, choices, and answer (letter)
        """
        raise NotImplementedError("extract_example must be implemented by subclasses of MCQABenchmark")


    def mcqa_format(self, prompt, choices, answer):
        answer = answer.strip().upper()

        input_text = chat.format_no_cot(
            chat.mcqa_question(prompt, choices), "_", "_"
        )[0]
        output_text = chat.no_cot_format(answer)

        keep = answer in string.ascii_uppercase[:len(choices)]
        answer_index = string.ascii_uppercase.index(answer) if keep else None

        return {
            "input_text": input_text,
            "output_text": output_text,
            "num_choices": len(choices),
            "answer_letter": answer,
            "keep": keep,
            "answer_index": answer_index,
        }


    def data_map_fn(self, batch):

        b = []
        for i in range(len(list(batch.values())[0])):
            example = {k: v[i] for k, v in batch.items()}
            b.append(example)
        batch = b

        l = []
        for example in batch:

            prompt, choices, answer = self.extract_example(example)
            formatted = self.mcqa_format(prompt, choices, answer)
            l.append(formatted)

        d = {}
        for k in l[0].keys():
            d[k] = [x[k] for x in l]

        input_ids, keep_input = self.tokenize(d["input_text"], "input")
        output_ids, keep_output = self.tokenize(d["output_text"], "output")

        keep = (
            keep_input & keep_output & np.array(d["keep"], dtype=bool)
        )

        d.update({
                "input_ids": input_ids,
                "output_ids": output_ids,
                "keep": keep,
        })

        return d

    
    def grade(self, batch, logits):
        logits = logits[:, self.prefix_length]

        logits = torch.index_select(logits, -1, self.torch_letter_ids)

        mask = (
            torch.arange(logits.shape[-1], device=logits.device).long()[None] >=
            batch["num_choices"][:, None]
        )
        logits = logits.masked_fill(mask, float("-inf"))

        pred_indices = torch.argmax(logits, dim=-1)

        return pred_indices == batch["answer_index"]



class MathBenchmark(BaseBenchmark):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
        max_examples: int | None = None,
    ):
        
        self.prefix_length = len(
            tokenizer(chat.NO_COT_PREFIX, add_special_tokens=False).input_ids
        )

        super().__init__(
            tokenizer,
            max_input_length,
            max_output_length,
            max_examples
        )


    def extract_example(self, example):
        """
        Should return prompt, choices, and answer (letter)
        """
        raise NotImplementedError("extract_example must be implemented by subclasses of MCQABenchmark")


    def math_format(self, prompt, answer):
        answer = answer.strip()

        input_text = chat.format_no_cot(
            prompt, "_", "_"
        )[0]
        output_text = chat.no_cot_format(answer)

        keep = True
        for c in answer:
            if c not in string.digits+".e":
                keep = False
                break

        return {
            "input_text": input_text,
            "output_text": output_text,
            "answer": answer,
            "keep": keep,
        }


    def data_map_fn(self, batch):

        b = []
        for i in range(len(list(batch.values())[0])):
            example = {k: v[i] for k, v in batch.items()}
            b.append(example)
        batch = b

        l = []
        for example in batch:

            prompt, answer = self.extract_example(example)
            formatted = self.math_format(prompt, answer)
            l.append(formatted)

        d = {}
        for k in l[0].keys():
            d[k] = [x[k] for x in l]

        input_ids, keep_input = self.tokenize(d["input_text"], "input")
        output_ids, keep_output = self.tokenize(d["output_text"], "output")

        keep = (
            keep_input & keep_output & np.array(d["keep"], dtype=bool)
        )

        d.update({
                "input_ids": input_ids,
                "output_ids": output_ids,
                "keep": keep,
        })

        return d

    
    def grade(self, batch, logits):

        target_ids = batch["output_ids"][:, self.prefix_length:]
        logits = logits[:, self.prefix_length:]

        pred_indices = torch.argmax(logits, dim=-1)

        correct = (pred_indices == target_ids) | (target_ids == self.tokenizer.pad_token_id)
        correct = correct.all(dim=-1)

        return correct


class arc_e(MCQABenchmark):

    name = "ARC-Easy"

    url = "allenai/ai2_arc"
    subset = "ARC-Easy"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["choices"]["text"],
            example["answerKey"],
        )


class arc_c(arc_e):

    name = "ARC-Challenge"
    subset = "ARC-Challenge"


class sciq(MCQABenchmark):
    
    name = "SciQ"

    url = "allenai/sciq"
    split = "test"

    def extract_example(self, example):
        prompt = example["question"]

        choices = [
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
            example["correct_answer"]
        ]

        answer = 'D'

        return prompt, choices, answer


class piqa(MCQABenchmark):

    name = "PIQA"

    url = "lighteval/piqa"
    split = "test"

    def extract_example(self, example):
        prompt = example["goal"]

        choices = [
            example["sol1"],
            example["sol2"],
        ]

        answer = 'A' if example["label"] == 0 else 'B'

        return prompt, choices, answer


class mmlu(MCQABenchmark):

    name = "MMLU"

    url = "lighteval/mmlu"
    subset = "all"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["choices"],
            string.ascii_uppercase[example["answer"]],
        )


class mmlu_pro(MCQABenchmark):

    name = "MMLU-Pro"

    url = "TIGER-Lab/MMLU-Pro"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["options"],
            example["answer"],
        )


class gpqa(MCQABenchmark):

    name = "GPQA"

    url = "Idavidrein/gpqa"
    subset = "gpqa_main"
    split = "train"

    def extract_example(self, example):
        return (
            example["Question"],
            [
                example["Correct Answer"],
                example["Incorrect Answer 1"],
                example["Incorrect Answer 2"],
                example["Incorrect Answer 3"],
            ],
            'A',
        )


class strategy_qa(MCQABenchmark):

    name = "StrategyQA"

    url = "tasksource/strategy-qa" 
    split = "train"

    def extract_example(self, example):

        prompt = ".".join(example["facts"]) + " " + example["question"]

        return (
            prompt,
            ["Yes", "No"],
            'A' if example["answer"] else 'B',
        )


class ar_lsat(MCQABenchmark):

    name = "AR-LSAT"

    url = 'olegbask/AR-LSAT'
    split = 'test'


    def extract_example(self, example):
        
        prompt = example["context"] + " " + example["question"]

        return (
            prompt,
            example["answers"],
            string.ascii_uppercase[example["label"]],
        )


class gsm8k(MathBenchmark):

    name = "GSM8K"

    url = "openai/gsm8k"
    subset = "main"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example['answer'].split("####")[-1].strip(),
        )


class math500(MathBenchmark):

    name = "MATH-500"

    url = "HuggingFaceH4/MATH-500"
    split = "test"

    def extract_example(self, example):
        return (
            example["problem"],
            example["answer"].strip(),
        )


class minervamath(MathBenchmark):

    name = "Minerva-Math"

    url = "math-ai/minervamath"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["answer"].strip(),
        )


class amc23(MathBenchmark):

    name = "AMC-23"

    url = "math-ai/amc23"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["answer"].strip(),
        )


class aime2025(MathBenchmark):

    name = "AIME-2025"

    url = "MathArena/aime_2025"
    split = "train"

    def extract_example(self, example):
        return (
            example["problem"],
            str(example["answer"]),
        )


class aime2026(aime2025):

    name = "AIME-2026"

    url = "MathArena/aime_2026"


class svamp(MathBenchmark):

    name = "SVAMP"

    url = "MU-NLPC/Calc-svamp"
    subset = "default"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["result"].replace("_", "").strip(),
        )


BENCHMARK_DICT = {
    cls[1].name: cls[1]
    for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if issubclass(cls[1], BaseBenchmark) and cls[1].name is not None
}
