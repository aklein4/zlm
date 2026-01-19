
import pathwaysutils  # pylint: disable=unused-import

import tensorflow as tf

import jax
import os

import datasets

URL = "aklein4/fineweb-w-edu-tinyllama"


def initialize():
  """Initialization of hyperparameters and utilities"""
  
  # system initialization
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  # sharding
  jax.config.update("jax_use_shardy_partitioner", True)
  # update explicit sharding-supported config
#   if config.shard_mode == ShardMode.EXPLICIT:
#     jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)

  # os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""

  return


def main(
    url,
    ds_kwargs,
    global_mesh,
    process_indices_train,
):

    initialize()
    
    train_ds = datasets.load_dataset(
        URL, split="train", streaming=True, 
    )

