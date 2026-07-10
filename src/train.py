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
import torch_xla.backends as xla_backends
import transformers
from transformers.utils.versions import require_version

import torch_xla.core.xla_model as xm

from torchprime.torch_xla_models.model import model_utils
from torchprime.torch_xla_models.utils.config_utils import config_validator

from data.datasets import get_dataset
from utils import constants
from utils.import_utils import import_model, import_trainer
from models import load_checkpoint_state


# Model/cache contracts are tested against the pinned TPU version and newer v4
# releases. Major Transformers upgrades require an explicit compatibility pass.
require_version("transformers>=4.52.1,<5")
logger = logging.getLogger(__name__)


# Enable SPMD mode for better performance on large models
xr.use_spmd()
assert xr.is_spmd() is True


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    # Validate the config to avoid misuse and feature combination
    # Adding any new feature should update the config validator to
    # ensure different features can be combined together
    config_validator(config)

    # Print the config for debugging
    if constants.PROCESS_IS_MAIN():
        print("\n ===== Configuration ===== \n", flush=True)
        print(omegaconf.OmegaConf.to_yaml(config), flush=True)
        print(" ========================= \n", flush=True)

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

    # Set the dtype
    torch.set_default_dtype(getattr(torch, config.torch_dtype))
    xla_backends.set_mat_mul_precision(config.matmul_precision)

    # print dtype environment info
    logger.info(f"Default torch dtype: {torch.get_default_dtype()}")
    logger.info(f"Matmul precision: {xla_backends.get_mat_mul_precision()}")
    logger.info(f"XLA_USE_BF16: {os.environ.get('XLA_USE_BF16', None)}")
    logger.info(f"XLA_DOWNCAST_BF16: {os.environ.get('XLA_DOWNCAST_BF16', None)}")

    # set the default device to the XLA device.
    # This will capture the model constructor into a graph so that we can add
    # sharding annotations to the weights later, and run the constructor on the XLA device.
    with model_utils.set_default_dtype(getattr(torch, config.model.torch_dtype)):
        model = import_model(config.model.type)(config.model)

    initialize_from = config.checkpoint.initialize_from
    initialize_step = config.checkpoint.initialize_step
    initialize_revision = config.checkpoint.initialize_revision
    initialize_strict = config.checkpoint.initialize_strict

    # Backward-compatible spelling. Unlike the old behavior, initialization
    # never advances the training step; exact continuation uses resume_from.
    if initialize_from is None and config.model.pretrained_url is not None:
        initialize_from = config.model.pretrained_url
        initialize_step = config.model.pretrained_step
        initialize_strict = config.model.pretrained_strict

    if initialize_from is not None and config.checkpoint.resume_from is not None:
        raise ValueError("Configure either checkpoint initialization or resume, not both.")

    if initialize_from is not None:
        model = load_checkpoint_state(
            model,
            initialize_from,
            initialize_step,
            strict=initialize_strict,
            revision=initialize_revision,
        )

    # TODO: initialize the model weights directly on the XLA device
    model = model.to(torch_xla.device())

    xm.rendezvous("Model Initialization")
    logger.info(f"Model initialized: {config.model.type}")
    model_utils.log_parameter_breakdown(model, logger)

    # Create the dataset
    data = get_dataset(config.data.dataset.url, config.data.dataset.kwargs)
    logger.info(f"Dataset loaded: {config.data.dataset.url}")

    # initialize the trainer
    trainer = import_trainer(config.trainer.type)(
        model=model,
        config=config,
        train_dataset=data,
    )
    logger.info(f"Trainer initialized: {config.trainer.type}")

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
