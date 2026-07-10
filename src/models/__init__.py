"""Model registration and portable checkpoint loading."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

import huggingface_hub as hf
import omegaconf
import torch

from utils import constants
from utils.checkpointing import canonicalize_state_dict
from utils.import_utils import import_model


logger = logging.getLogger(__name__)


def register_transformers_auto_classes():
    """Register local model classes with Transformers AutoClasses."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    from models.configuration import CustomLlamaConfig, TPULlamaConfig, ZLMConfig
    from models.llama import LlamaForCausalLM
    from models.custom_llama import CustomLlamaForCausalLM
    from models.zlm import ZLMModel

    AutoConfig.register(TPULlamaConfig.model_type, TPULlamaConfig, exist_ok=True)
    AutoConfig.register(CustomLlamaConfig.model_type, CustomLlamaConfig, exist_ok=True)
    AutoConfig.register(ZLMConfig.model_type, ZLMConfig, exist_ok=True)
    AutoModelForCausalLM.register(
        TPULlamaConfig, LlamaForCausalLM, exist_ok=True
    )
    AutoModelForCausalLM.register(
        CustomLlamaConfig, CustomLlamaForCausalLM, exist_ok=True
    )
    AutoModel.register(ZLMConfig, ZLMModel, exist_ok=True)

    return {
        "causal_lm": (LlamaForCausalLM, CustomLlamaForCausalLM),
        "conditional_generation": ZLMModel,
    }


def _resolve_checkpoint_dir(
    url: str,
    step: int | None,
    ignore_cache: bool,
    revision: str,
) -> tuple[Path, bool]:
    """Resolve either a real local directory or a Hub checkpoint directory."""
    source = Path(url).expanduser()
    if source.exists():
        checkpoint_dir = source
        if step is not None and (source / f"{step:012d}").is_dir():
            checkpoint_dir = source / f"{step:012d}"
        if (checkpoint_dir / "transformers").is_dir():
            checkpoint_dir = checkpoint_dir / "transformers"
        return checkpoint_dir, False

    name = url.replace("/", "--")
    local_path = Path(constants.CHECKPOINTS_PATH) / name
    subfolder = f"{step:012d}" if step is not None else None
    checkpoint_dir = local_path / subfolder if subfolder is not None else local_path

    if ignore_cache:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    if not checkpoint_dir.exists():
        allow_patterns = [f"{subfolder}/*"] if subfolder is not None else None
        hf.snapshot_download(
            repo_id=url,
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=local_path,
        )

    if (checkpoint_dir / "transformers").is_dir():
        checkpoint_dir = checkpoint_dir / "transformers"

    return checkpoint_dir, True


def _load_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load safetensors when available, with legacy model.pt fallback."""
    from torchprime.torch_xla_models.model.model_utils import (
        load_safetensors_to_state_dict,
    )

    if (
        (checkpoint_dir / "model.safetensors").exists()
        or (checkpoint_dir / "model.safetensors.index.json").exists()
    ):
        state_dict = load_safetensors_to_state_dict(str(checkpoint_dir))
    else:
        state_path = checkpoint_dir / "model.pt"
        if not state_path.exists():
            raise FileNotFoundError(
                f"No safetensors or legacy model.pt found in {checkpoint_dir}."
            )
        state_dict = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )

    return canonicalize_state_dict(state_dict)


def load_checkpoint(
    url: str,
    step: int | None = None,
    attention_kernel: str = "other", # uses non-kernel attention by default
    strict: bool = True,
    ignore_cache: bool = False,
    remove_folder: bool = False,
    model_type: str | None = None,
    skip_state_dict: bool = False,
    config_name: str | None = None,
    revision: str = "main",
) -> torch.nn.Module:
    """Load a complete model from a local directory or Hugging Face Hub."""
    if not isinstance(strict, bool):
        raise TypeError("strict must be an explicit boolean.")

    checkpoint_dir, downloaded = _resolve_checkpoint_dir(
        url, step, ignore_cache, revision
    )

    if config_name is not None:
        config_path = (
            Path(constants.BASE_PATH) / "configs" / "model" / f"{config_name}.yaml"
        )
    else:
        config_path = checkpoint_dir / "config.json"
    config = omegaconf.OmegaConf.load(config_path)
    config.attention_kernel = attention_kernel

    if model_type is None:
        model_type = config.get("type", None)

    if model_type is not None:
        model = import_model(model_type)(config)
    else:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        register_transformers_auto_classes()
        transformers_config = AutoConfig.from_pretrained(checkpoint_dir)
        transformers_config.attention_kernel = attention_kernel
        if transformers_config.model_type == "zlm":
            model = AutoModel.from_config(transformers_config)
        else:
            model = AutoModelForCausalLM.from_config(transformers_config)

    if not skip_state_dict:
        load_result = model.load_state_dict(
            _load_state_dict(checkpoint_dir),
            strict=strict,
        )
        if not strict and (load_result.missing_keys or load_result.unexpected_keys):
            logger.warning(
                "Non-strict checkpoint load: missing=%s unexpected=%s",
                load_result.missing_keys,
                load_result.unexpected_keys,
            )

    if remove_folder and downloaded:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    return model


def load_checkpoint_state(
    model: torch.nn.Module,
    url: str,
    step: int | None = None,
    strict: bool = True,
    ignore_cache: bool = False,
    remove_folder: bool = False,
    revision: str = "main",
):
    """Load portable weights into an existing model."""
    if not isinstance(strict, bool):
        raise TypeError("strict must be an explicit boolean.")

    checkpoint_dir, downloaded = _resolve_checkpoint_dir(
        url, step, ignore_cache, revision
    )
    load_result = model.load_state_dict(
        _load_state_dict(checkpoint_dir),
        strict=strict,
    )
    if not strict and (load_result.missing_keys or load_result.unexpected_keys):
        logger.warning(
            "Non-strict checkpoint load: missing=%s unexpected=%s",
            load_result.missing_keys,
            load_result.unexpected_keys,
        )

    if remove_folder and downloaded:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    return model
