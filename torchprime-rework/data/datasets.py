
import datasets
import os

from utils import constants


class FakeIterableDataset(datasets.IterableDataset):

    def __init__(self):
        pass


    def __len__(self):
        return 1_000_000_000
    

    def __iter__(self):
        yield {"text": "This is a fake dataset for testing purposes."}


def get_dataset(name: str, **kwargs) -> datasets.Dataset:
    """
    Get a dataset by name.
    
    Args:
        name (str): The name of the dataset to retrieve.
    
    Returns:
        datasets.Dataset: The requested dataset.
    """
    if name == "fake":
        return FakeIterableDataset()

    ds = datasets.load_dataset(name, **kwargs, token=constants.HF_TOKEN)

    if "streaming" in kwargs.keys() and kwargs["streaming"]:
        ds = ds.shard(num_shards=constants.PROCESS_COUNT(), index=constants.PROCESS_INDEX())

    return ds

