
import random

from handlers import HANDLERS


LOG_FILE = "compilation_log.txt"

MAX_INPUT_CHARACTERS = 1000 * 10
MAX_OUTPUT_CHARACTERS = 1000 * 10

NAMES_TO_DO = None

DEBUG = False


def main():
    
    random.seed(42)

    with open(LOG_FILE, "w") as f:
        f.write("")

    handler_list = HANDLERS
    if NAMES_TO_DO is not None:

        names = [h().name() for h in handler_list]
        for name in NAMES_TO_DO:
            if name not in names:
                raise ValueError(f"Dataset name {name} not found in handlers.")

        handler_list = [
            h for h in handler_list if h().name() in NAMES_TO_DO
        ]

    total_examples = 0
    for i, h_type in enumerate(handler_list):

        h = h_type(
            max_input_characters=MAX_INPUT_CHARACTERS,
            max_output_characters=MAX_OUTPUT_CHARACTERS,
        )

        print("")
        print(f"[{i+1}/{len(handler_list)}] Processing dataset: {h.name()}")
        print("")

        try:

            ds = h.load_dataset()
            
            ds = ds.map(h.full_map, remove_columns=ds.column_names, load_from_cache_file=False)
            ds = ds.filter(h.filter, load_from_cache_file=False)
        
            ds.push_to_hub(
                "aklein4/raw-compilation",
                config_name=h.name().replace("/", "--"),
                private=False,
                split="train",
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt) or DEBUG:
                raise e

            with open(LOG_FILE, "a") as f:
                f.write(f"\n[{i+1}/{len(handler_list)}] {h.name()}: FAIL")
            continue

        with open(LOG_FILE, "a") as f:
            f.write(f"\n[{i+1}/{len(handler_list)}] {h.name()}: SUCCESS ({len(ds):_} examples)")
        total_examples += len(ds)
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\nTotal examples: {total_examples:_}\n")


if __name__ == "__main__":
    main()