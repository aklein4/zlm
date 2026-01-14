
from models import load_checkpoint

from torchprime.torch_xla_models.model.model_utils import log_parameter_breakdown

URL = "aklein4/ZRM_llama-baseline-scaled"
STEP = 40000


def main():
    
    model = load_checkpoint(URL, STEP)
    
    log_parameter_breakdown(model)


if __name__ == "__main__":
    main()
