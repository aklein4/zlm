
import os

try:
    import torch_xla.runtime as xr
    XLA_AVAILABLE = True
except ImportError:
    print("Warning: torch_xla not found", flush=True)
    XLA_AVAILABLE = False

# get the base path of src
BASE_PATH = os.path.dirname( # src
    os.path.dirname( # utils
        __file__ # utils.constants
    )
)

# id of the current device
PROCESS_INDEX = lambda: xr.process_index()

# whether this is the main process on its device
PROCESS_IS_MAIN = lambda: xr.process_index() == 0

PROCESS_COUNT = lambda: xr.process_count()

# local data path
LOCAL_DATA_PATH = os.path.join(BASE_PATH, "local_data")

# paths to config files
CONFIG_PATH = os.path.join(BASE_PATH, "configs")

# modules for classes
MODEL_MODULE = "models"
TRAINER_MODULE = "trainers"
COLLATOR_MODULE = "collators"

# huggingface login id
HF_ID = "aklein4"

# token for huggingface
HF_TOKEN = os.getenv("HF_TOKEN")
