from torchprime.utils.parallelism_utils import cp_enabled, lb_cp_enabled


def config_vaidator(config: dict):
  """
  This validator checks whether the user provided config is valid
  in advance, thus avoiding unnecessary unclear failure or misuses,
  improving usability.
  """
  if (
    "load_balance_cp" in config.model
    and config.model.load_balance_cp
    and not cp_enabled
  ):
    raise RuntimeError(
      "Load balanced context parallelism can only be used when cp is enabled"
    )

  if lb_cp_enabled(config) and config.attention_kernel != "splash_attention":
    raise RuntimeError(
      "Load balanced context parallelism is only supported with splash attention kernel"
    )

  if cp_enabled(config):
    if "context" not in config.model:
      raise RuntimeError("Specify context parallelism size in config.model as well")
    elif config.model.context != config.ici_mesh.context:
      raise RuntimeError(
        "ici context size should equal to model context parallelism size"
      )

  if config.trainer.checkpoint_interval <= 0:
    raise ValueError("trainer.checkpoint_interval must be positive")

  if (
    config.checkpoint.export_interval is not None
    and config.checkpoint.export_interval <= 0
  ):
    raise ValueError("checkpoint.export_interval must be positive or null")

  if config.checkpoint.keep_last is not None and config.checkpoint.keep_last <= 0:
    raise ValueError("checkpoint.keep_last must be positive or null")

  if not isinstance(config.checkpoint.initialize_strict, bool):
    raise TypeError("checkpoint.initialize_strict must be true or false")

  if (
    config.checkpoint.initialize_from is not None
    and config.checkpoint.resume_from is not None
  ):
    raise ValueError("Checkpoint initialization and exact resume are mutually exclusive")

  if config.model.pretrained_url is not None and not isinstance(
    config.model.pretrained_strict, bool
  ):
    raise TypeError("model.pretrained_strict must be true or false")


# Correct spelling for new callers; retain the old name for compatibility.
config_validator = config_vaidator
