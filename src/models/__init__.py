""" Models """

import torch

import omegaconf
import os
import huggingface_hub as hf
import shutil

from utils.import_utils import import_model
from utils import constants


def load_checkpoint(
    url: str,
    step: int,
    attention_kernel: str = "other", # uses non-kernel attention by default
    remove_folder: bool = False,
):
    
    subfolder = f"{step:012d}"

    # download the folder
    save_path = hf.snapshot_download(
        repo_id=url,
        allow_patterns=[subfolder + "/*"],
        local_dir=constants.CHECKPOINTS_PATH,
    )
    save_path = os.path.join(save_path, subfolder)

    # load the model
    config_path = os.path.join(save_path, "config.json")
    config = omegaconf.OmegaConf.load(config_path)
    config.attention_kernel = attention_kernel

    model = import_model(config.type)(config)

    # load the weights
    state_path = os.path.join(save_path, "model.pt")
    state_dict = torch.load(state_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=True)

    if remove_folder:
        shutil.rmtree(save_path, ignore_errors=True)

    return model


def load_checkpoint_state(
    model: torch.nn.Module,
    url: str,
    step: int,
    remove_folder: bool = False,
):
    
    subfolder = f"{step:012d}"

    # download the folder
    save_path = hf.snapshot_download(
        repo_id=url,
        allow_patterns=[subfolder + "/*"],
        local_dir=constants.CHECKPOINTS_PATH,
    )
    save_path = os.path.join(save_path, subfolder)

    # load the weights
    state_path = os.path.join(save_path, "model.pt")
    state_dict = torch.load(state_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=True)

    if remove_folder:
        shutil.rmtree(save_path, ignore_errors=True)

    return model
