
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


def remove_think(content):
    if "</think>" in content:
        return content.split("</think>")[-1].strip()
    return content


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


def random_from_pair(
    example,
    prompt_key="prompt",
    chosen_key="chosen",
    rejected_key="rejected",
):
    inp = example[prompt_key]

    if random.random() > 0.5:
        out = example[chosen_key]
        r = 1
    else:
        out = example[rejected_key]
        r = -1

    return {
        "input": inp.strip(),
        "output": out.strip(),
        "reward": r
    }


""" ===== Chat ===== """

class ChatHandler(BaseHandler):
    domain = "chat"


class SHPHandler(ChatHandler):

    url = "stanfordnlp/SHP-2"
    subset = None
    split = "train"

    reward_type = "preference"

    def map(self, example):
        
        inp = example["history"]
        out = example["human_ref_A"]
        r = 2 * int(example["label"]) - 1

        source = self.name() + f"/{example['domain']}"

        return {
            "source": source,
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class TuluPreferenceHandler(ChatHandler):

    url = "allenai/tulu-2.5-preference-data"
    subset = None
    split = get_splits("allenai/tulu-2.5-preference-data", None)

    reward_type = "preference"

    def map(self, example):
        if len(example["chosen"]) != 2 or len(example["rejected"]) != 2:
            return None

        if random.random() > 0.5:
            c = example["chosen"]
            r = 1
        else:
            c = example["rejected"]
            r = -1

        inp = c[0]['content']
        out = c[1]['content']

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class DolciThink32BHandler(ChatHandler):

    url = "allenai/Dolci-Think-DPO-32B"
    subset = None
    split = "train"

    reward_type = "preference"

    def map(self, example):
        if len(example["chosen"]) != 2 or len(example["rejected"]) != 2:
            return None

        if random.random() > 0.5:
            c = example["chosen"]
            r = 1
        else:
            c = example["rejected"]
            r = -1

        inp = remove_think(c[0]['content'])
        out = remove_think(c[1]['content'])

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class DolciThink7BHandler(DolciThink32BHandler):

    url = "allenai/Dolci-Think-DPO-7B"


class DolciInstructHandler(DolciThink32BHandler):

    url = "allenai/Dolci-Instruct-DPO"


class OlmoPreferenceHandler(DolciThink32BHandler):

    url = "allenai/olmo-2-1124-13b-preference-mix"


class TuluLlama405BPreferenceHandler(DolciThink32BHandler):

    url = "allenai/llama-3.1-tulu-3-405b-preference-mixture"


class MagpieLlamaDPOHandler(DolciThink32BHandler):

    url = "Magpie-Align/Magpie-Llama-3.1-Pro-DPO-100K-v0.1"


class MagpieDPOHandler(DolciThink32BHandler):

    url = "Magpie-Align/Magpie-DPO-100K-SML"


class MagpieDPOProHandler(DolciThink32BHandler):

    url = "Magpie-Align/Magpie-Pro-DPO-100K-v0.1"


class MagpieDPOAirHandler(DolciThink32BHandler):

    url = "Magpie-Align/Magpie-Air-DPO-100K-v0.1"


class MultiCollectionORPOHandler(ChatHandler):

    url = "kaist-ai/Multifaceted-Collection-ORPO"
    subset = None
    split = "train"

    reward_type = "preference"

    def map(self, example):
        out = DolciThink32BHandler.map(self, example)

        if out is not None:
            out["cluster_key"] = example["system"].strip()

        return out


class MultiCollectionDPOHandler(MultiCollectionORPOHandler):

    url = "kaist-ai/Multifaceted-Collection-DPO"


class FeedbackCollectionHandler(ChatHandler):

    url = "prometheus-eval/Feedback-Collection"
    subset = None
    split = "train"

    reward_type = "rubric"

    def map(self, example):
        inp = example["orig_instruction"]
        out = example["orig_response"]

        r = int(example["orig_score"])
        if r >= 4:
            r = 1
        elif r <= 2:
            r = -1
        else:
            r = 1 if random.random() > 0.5 else -1

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class HelpSteer3PreferenceHandler(ChatHandler):

    url = "nvidia/HelpSteer3"
    subset = "preference"
    split = "train"

    reward_type = "preference"
    
    def map(self, example):
        if len(example["context"]) > 1:
            return None

        inp = example["context"][0]['content']
        out = example["response1"]

        r = example["overall_preference"]
        if r < 0:
            r = 1
        elif r == 0:
            r = 1 if random.random() > 0.5 else -1
        else:
            r = -1

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class UltraFeedbackBinarizedHandler(ChatHandler):

    url = "HuggingFaceH4/ultrafeedback_binarized"
    subset = None
    split = "train_prefs"

    reward_type = "preference"

    def map(self, example):
        if len(example["chosen"]) > 2 or len(example["rejected"]) > 2:
            return None

        if random.random() > 0.5:
            c = example["chosen"]
            r = 1
        else:
            c = example["rejected"]
            r = -1

        inp = c[0]['content']
        out = c[1]['content']

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class UltraInteractHandler(ChatHandler):

    url = "openbmb/UltraInteract_pair"
    subset = None
    split = "train"

    reward_type = "preference"
    
    def map(self, example):
        if len(example["trajectory"]) != 1:
            return None

        inp = example["trajectory"][0]['value']

        if random.random() > 0.5:
            out = example["chosen"]
            r = 1
        else:
            out = example["rejected"]
            r = -1

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class SkyworkRewardPreferenceHandler(ChatHandler):

    url = "Skywork/Skywork-Reward-Preference-80K-v0.2"
    subset = None
    split = "train"

    reward_type = "preference"
    
    def map(self, example):
        
        if random.random() > 0.5:
            c = example["chosen"]
            r = 1
        else:
            c = example["rejected"]
            r = -1

        if len(c) != 2:
            return None

        inp = c[0]['content']
        out = c[1]['content']

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


""" ===== Math ===== """

class MathHandler(BaseHandler):
    domain = "math"


class DeepseekORMHandler(MathHandler):

    url = "RLHFlow/Deepseek-ORM-Data"
    subset = None
    split = "train"

    reward_type = "verified"

    def map(self, example):
        if len(example['conversations']) != 2:
            return None

        x = example['conversations'][0]['content']
        y = example['conversations'][1]['content']

        if x.count('?') != 1:
            return None
        
        inp, out = x.split('?')
        inp = inp.strip() + '?'

        if y == '+':
            r = 1
        elif y == '-':
            r = -1
        else:
            return None
        
        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class MistralORMHandler(MathHandler):

    url = "RLHFlow/Mistral-ORM-Data"
    subset = None
    split = "train"

    reward_type = "verified"

    def map(self, example):
        if len(example['conversations']) != 2:
            return None

        x = example['conversations'][0]['content']
        y = example['conversations'][1]['content']

        if x.count('Step 1:') != 1:
            return None
        
        inp, out = x.split('Step 1:')
        out = 'Step 1: ' + out.strip()

        if y == '+':
            r = 1
        elif y == '-':
            r = -1
        else:
            return None
        
        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


class MetaMathDPOHandler(MathHandler):

    url = "abacusai/MetaMath_DPO_FewShot"
    subset = None
    split = "train"

    reward_type = "preference"
    
    def map(self, example):
        return random_from_pair(example)


class AceMathRMHandler(MathHandler):

    url = "nvidia/AceMath-RM-Training-Data"
    subset = None
    split = "train"

    reward_type = "verified"

    def map(self, example):
        if len(example["message"]) != 3:
            return None

        inp = example["message"][1]['content']
        out = example["message"][2]['content']
        r = 2 * int(example["label"]) - 1

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }


""" ===== Code ===== """

class CodeHandler(BaseHandler):
    domain = "code"


class AceCodePairHandler(CodeHandler):

    url = "TIGER-Lab/AceCodePair-300K"
    subset = "default"
    split = "train"

    reward_type = "preference"

    def map(self, example):
        return random_from_pair(example, prompt_key="instruction")


""" ===== Safety ===== """

class SafetyHandler(BaseHandler):
    domain = "safety"


class HHrlhfHandler(SafetyHandler):

    url = "Anthropic/hh-rlhf"
    subset = None
    split = "train"

    reward_type = "preference"
    
    def map(self, example):
        s = example["chosen"]
        r = 1
        if random.random() < 0.5:
            s = example["rejected"]
            r = -1

        if s.count("Human: ") != 1 or s.count("Assistant: ") != 1:
            return None

        inp, out = s.split("Assistant: ")

        if inp.count("Human: ") != 1:
            return None
        inp = inp.split("Human: ")[-1]

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }
    

class PKUSafeHandler(SafetyHandler):

    url = "PKU-Alignment/PKU-SafeRLHF"
    subset = "default"
    split = "train"

    reward_type = "rubric"
    
    def map(self, example):
        inp = example["prompt"]
        out = example["response_0"]
        r = 2 * int(bool(example["is_response_0_safe"])) - 1

        return {
            "input": inp.strip(),
            "output": out.strip(),
            "reward": r
        }
        

HANDLERS = [x[1] for x in inspect.getmembers(sys.modules[__name__]) if inspect.isclass(x[1]) and issubclass(x[1], BaseHandler) and x[1] is not BaseHandler][::-1]
for h in [ChatHandler, MathHandler, CodeHandler, SafetyHandler]:
    if h in HANDLERS:
        HANDLERS.remove(h)
