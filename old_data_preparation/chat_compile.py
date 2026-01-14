
import datasets
import json
from tqdm import tqdm


def format_data(data):

    if data[0]["role"] == "system":
        input_message = f"Instructions:\n{data[0]['content'].strip()}\nQuestion:\n{data[1]['content'].strip()}\nAnswer:\n"
        
        try:
            output_message = data[2]["content"].strip()
        except:
            output_message = None

    else:
        input_message = f"Question:\n{data[0]['content'].strip()}\nAnswer:\n"
        
        try:
            output_message = data[1]["content"].strip()
        except:
            output_message = None

    return input_message, output_message


def compilation_map(example):
    data = example["data"]

    inp, out = format_data(data)

    return {
        "source": example["source"],
        "input": inp,
        "output": out,
    }


def synth_map(example, subset=""):
    data = example["message"]

    inp, out = format_data(data)

    return {
        "source": f'IgnoraZ/SynthQuestions/{subset}',
        "input": inp,
        "output": out,
    }


def webinstruct_map(example):
    return {
        "source": "TIGER-Lab/WebInstructSub",
        "input": f"Question:\n{example["question"].strip()}\nAnswer:\n",
        "output": example["answer"].strip(),
    }


def main():
    
    compilation = datasets.load_dataset("aklein4/chat-compilation", split="train")

    with open("../local_data/realquestions.jsonl", "r", encoding="utf-8") as f:
        l_real = [json.loads(line) for line in tqdm(f.readlines(), desc="Loading realquestions")]
    realquestions = datasets.Dataset.from_list(l_real)
    
    with open("../local_data/synthquestions_1m.moderated.jsonl", "r", encoding="utf-8") as f:
        l_synth = [json.loads(line) for line in tqdm(f.readlines(), desc="Loading synthquestions")]
    synthquestions = datasets.Dataset.from_list(l_synth)

    webinstruct = datasets.load_dataset("TIGER-Lab/WebInstructSub", split="train")

    compilation = compilation.filter(
        lambda x: x["source"] not in ["facebook/natural_reasoning", "lmsys/lmsys-chat-1m"]
    )

    compilation = compilation.map(
        compilation_map,
        remove_columns=compilation.column_names,
    )
    realquestions = realquestions.map(
        synth_map,
        remove_columns=realquestions.column_names,
        fn_kwargs={"subset": "realquestions"},
    )
    synthquestions = synthquestions.map(
        synth_map,
        remove_columns=synthquestions.column_names,
        fn_kwargs={"subset": "synthquestions"},
    )
    webinstruct = webinstruct.map(
        webinstruct_map,
        remove_columns=webinstruct.column_names,
    )

    compilation = compilation.filter(
        lambda x: x["output"] is not None
    )
    realquestions = realquestions.filter(
        lambda x: x["output"] is not None
    )
    synthquestions = synthquestions.filter(
        lambda x: x["output"] is not None
    )
    webinstruct = webinstruct.filter(
        lambda x: x["output"] is not None
    )

    print("\n")
    print(f"Loaded {len(compilation)} compilation examples!")
    print(f"Loaded {len(realquestions)} realquestions examples!")
    print(f"Loaded {len(synthquestions)} synthquestions examples!")
    print(f"Loaded {len(webinstruct)} webinstruct examples!")
    print("\n")

    combined = datasets.concatenate_datasets(
        [
            compilation,
            realquestions,
            synthquestions,
            webinstruct
        ]
    )
    shuffled = combined.shuffle(seed=42)

    shuffled.push_to_hub(
        "chat-formatted",
    )


if __name__ == "__main__":
    main()