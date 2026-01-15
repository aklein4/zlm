"""Train script for LLMs using PyTorch/XLA with some torchax for lowering."""

import os
os.environ['PJRT_DEVICE'] = 'TPU'

import logging
import sys

import datasets
import hydra
import omegaconf
import torch
import torch_xla
import torch_xla.runtime as xr
import transformers

import torch_xla.core.xla_model as xm

from torchprime.torch_xla_models.model import model_utils
from torchprime.torch_xla_models.utils.config_utils import config_vaidator

from data.datasets import get_dataset
from utils import constants
from utils.import_utils import import_class
from models import load_checkpoint_state

transformers.utils.check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    # Validate the config to avoid misuse and feature combination
    # Adding any new feature should update the config validator to
    # ensure different features can be combined together
    config_vaidator(config)

    # Print the config for debugging
    if constants.PROCESS_IS_MAIN():
        print("\n ===== Configuration ===== \n", flush=True)
        print(omegaconf.OmegaConf.to_yaml(config), flush=True)
        print("\n ========================= \n", flush=True)

    # set up logging
    logger.setLevel(logging.INFO)
    if constants.PROCESS_IS_MAIN():
        verbosity = logging.INFO 
    else:
        logging.disable(logging.CRITICAL)
        verbosity = logging.CRITICAL
    datasets.utils.logging.set_verbosity(verbosity)
    transformers.utils.logging.set_verbosity(verbosity)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # set training seeds
    torch.manual_seed(config.seed)
    transformers.set_seed(config.seed)
    torch_xla.manual_seed(config.seed)

    # Set the model dtype to bfloat16, and set the default device to the XLA device.
    # This will capture the model constructor into a graph so that we can add
    # sharding annotations to the weights later, and run the constructor on the XLA device.
    # assert config.torch_dtype == "bfloat16", "Currently only bfloat16 is supported"
    torch.set_default_dtype(torch.float32)
    model_dtype = getattr(torch, config.torch_dtype)
    with model_utils.set_default_dtype(model_dtype), torch_xla.device():
        model_cls = import_class(config.model.model_class, constants.MODEL_MODULE)
        model = model_cls(config.model)

    # load the pretrained model if specified
    if config.model.pretrained_model is not None:
        model = load_checkpoint_state(
            model,
            config.model.pretrained_model,
            config.model.pretrained_step,
            remove_folder=True
        )
    xm.rendezvous("load_model")

    # print model information
    model_utils.log_parameter_breakdown(model, logger)
    logger.info(f"Model initialized: {config.model.model_class}")

    # Create the dataset
    data = get_dataset(config.data.dataset.url, config.data.dataset.kwargs)
    logger.info(f"Dataset loaded: {config.data.dataset.name}")

    # initialize the trainer
    trainer_cls = import_class(config.trainer.trainer_type, constants.TRAINER_MODULE)
    trainer = trainer_cls(
        model=model,
        config=config,
        train_dataset=data,
    )
    logger.info(f"Trainer initialized: {config.trainer.trainer_type}")

    # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
    with torch_xla._internal.jax_workarounds.jax_env_context():
        trainer.train_loop()

    return 0


if __name__ == "__main__":

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    sys.exit(main())
