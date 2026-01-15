
from models import load_checkpoint, load_checkpoint_state


URL = "aklein4/SmolLM2-360M-TPU"
STEP = 0


def main():
    
    load_checkpoint_state(
        None,
        URL,
        STEP,
        remove_folder=False,
        strict=True,
    )


if __name__ == "__main__":
    main()
