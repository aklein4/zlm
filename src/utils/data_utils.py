import torch

import io
import numpy as np


def load_any_array(
    x: list | np.ndarray | torch.Tensor
) -> torch.Tensor:
    """
    Load an array from a list, numpy array, or torch tensor and convert it to a torch tensor if necessary.

    Args:
        x (list | np.ndarray | torch.Tensor): The input array to load.
    Returns:
        torch.Tensor: The loaded array as a torch tensor.
    """

    if isinstance(x, list):
        return torch.tensor(x)
    
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    
    elif isinstance(x, torch.Tensor):
        return x
    
    raise ValueError(f"Unsupported type {type(x)} for load_any_array")


def load_byte_array(
    data: bytes
) -> torch.LongTensor:
    """ Convert the data from a byte stream to a tensor.
        - see npy_loads() in https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py
    """

    # depending on datasets versions, sometimes functions that call this will already be in a list?
    try:
        stream = io.BytesIO(data)
        out = np.lib.format.read_array(stream)
    except:
        out = np.array(data)

    return out


def get_hf_files(
    url: str,
    branch: str = "main",
    splits: list = ["train", "val", "test"]
) -> dict:
    """ Get datafile urls for the given dataset name.
     - see example at https://huggingface.co/docs/hub/en/datasets-webdataset 
     
    Args:
        url (str): dataset url on huggingface hub
        branch (str, optional): branch to load from. Defaults to "main".
        splits (list, optional): list of splits to get urls for. Defaults to ["train", "val", "test"].

    Returns:
        Dict[str, str]: dict of splits and their urls
    """
    data_files = {}
    for split in splits:

        data_files[split] = f"https://huggingface.co/datasets/{url}/resolve/{branch}/{split}/*"
    
    return data_files
