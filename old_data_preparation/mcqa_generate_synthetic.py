import sys
import datasets
from functools import partial
import multiprocessing as mp

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


DATA_REPO = "aklein4/mcqa_compilation"


def format_request(question, answer):

    out = "For the following question and its answer, give a one paragraph explaination of why the given answer is correct. The given answer is definitely the correct one, and you should not disagree. If this is a reasoning or math question, explain how the final answer should be determined. If this is a knowledge, trivia, or reading comprehension question, explain why the given answer is correct and why the others are incorrect. Your full response should be 1 paragraph!\n"
    out += question
    out += f"\nCorrect Answer: {answer}"

    return out


def synthetic_map(example, client=None):
    question = example["question"]
    answer = example["answer"]

    request = format_request(question, answer)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=request,
            config=GenerateContentConfig(
                thinking_config=ThinkingConfig(
                    thinking_budget=0,
                )
            )
        ).text
    except KeyboardInterrupt:
        raise KeyboardInterrupt("Process interrupted by user.")
    except:
        response = None

    return {
        "explanation": response,
    }


def process_subset(api_key: str, num_processes, process_id: int):

    # Load the dataset
    dataset = datasets.load_dataset(DATA_REPO, split="train")
    subset = dataset.select(range(process_id, len(dataset), num_processes))
    
    client = genai.Client(api_key=api_key)
    
    processed_subset = subset.map(
        partial(synthetic_map, client=client),
    )
    processed_subset = processed_subset.filter(lambda x: x["explanation"] is not None)
    
    return processed_subset.to_dict()


def main():

    # ds = datasets.load_dataset('aklein4/mcqa-synthetic-explanations-lite', split='train')
    # n = 0
    # for example in ds:
    #     print(example)
    #     n += 1
    #     if n > 10:
    #         break

    if len(sys.argv) < 3:
        print("Usage: python generate_synthetic_multi.py <api_key> <num_processes>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    num_processes = int(sys.argv[2])
    
    print(f"Starting multiprocessing with {num_processes} processes")
    
    # Start processes
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_subset, [(api_key, num_processes, i) for i in range(num_processes)])

    print("All processes completed!")
    print("Combining all parts into a single dataset...")
    results = [datasets.Dataset.from_dict(r) for r in results]
    final_dataset = datasets.concatenate_datasets(results)

    final_dataset = final_dataset.map(
        lambda x: {"source": x["source"]+"-synthetic-explanations"},
    )

    print("Pushing combined dataset to Hugging Face Hub...")
    final_dataset.push_to_hub("mcqa-formatted")
    print("Combined dataset pushed to mcqa_synthetic_explanations")


if __name__ == "__main__":
    main()
