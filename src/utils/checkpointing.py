"""Training checkpoint and portable model export utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import random
import shutil
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp


logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
SUCCESS_FILE = "_SUCCESS"
MANIFEST_FILE = "checkpoint.json"


def canonical_parameter_name(name: str) -> str:
    """Remove module-wrapper path components without changing substrings."""
    return ".".join(part for part in name.split(".") if part != "_orig_mod")


def canonicalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a state dict with stable, portable parameter names."""
    output = {}
    for name, value in state_dict.items():
        canonical_name = canonical_parameter_name(name)
        if canonical_name in output:
            raise ValueError(
                f"State-dict key collision after canonicalization: {canonical_name}"
            )
        output[canonical_name] = value
    return output


@dataclass
class CheckpointManifest:
    """Small, versioned description of a complete training checkpoint."""

    global_step: int
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    checkpoint_type: str = "training"
    examples_seen: int = 0
    atoms_seen: int = 0
    git_commit: str | None = None
    git_dirty: bool = False
    torch_version: str = torch.__version__
    world_size: int = 1


def capture_rng_state() -> dict[str, Any]:
    """Capture host RNG state; XLA state is added when XLA is available."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    try:
        import torch_xla.core.xla_model as xm

        state["xla"] = xm.get_rng_state()
    except (ImportError, AttributeError):
        pass

    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

    if "xla" in state:
        try:
            import torch_xla.core.xla_model as xm

            xm.set_rng_state(state["xla"])
        except ImportError:
            logger.warning("Checkpoint has XLA RNG state, but torch-xla is unavailable.")


def initialize_checkpoint_process_group() -> None:
    """Initialize the CPU process group required by XLA DCP planners."""
    if dist.is_initialized():
        return

    try:
        import torch_xla.distributed.xla_backend  # noqa: F401

        dist.init_process_group("gloo", init_method="xla://")
    except ImportError:
        # Single-process CPU/GPU tests do not require a process group.
        return


def prime_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Materialize lazy optimizer state before an in-place DCP restore."""
    try:
        import torch_xla.experimental.distributed_checkpoint as xc

        xla_prime_optimizer = getattr(xc, "prime_optimizer", None)
        if xla_prime_optimizer is not None:
            xla_prime_optimizer(optimizer)
            return
    except ImportError:
        pass

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    try:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        xm.wait_device_ops()
    except ImportError:
        pass

def _save_planner():
    try:
        import torch_xla.experimental.distributed_checkpoint as xc

        return xc.SPMDSavePlanner()
    except ImportError:
        return None


def _load_planner():
    try:
        import torch_xla.experimental.distributed_checkpoint as xc

        return xc.SPMDLoadPlanner()
    except ImportError:
        return None


def distributed_save(state: dict[str, Any], checkpoint_dir: Path) -> None:
    """Save tensors directly from their current CPU/GPU/XLA shards."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initialize_checkpoint_process_group()

    kwargs = {
        "state_dict": state,
        "storage_writer": dist_cp.FileSystemWriter(
            str(checkpoint_dir),
            thread_count=max(2, min(8, os.cpu_count() or 2)),
        ),
    }
    planner = _save_planner()
    if planner is not None:
        kwargs["planner"] = planner
    dist_cp.save(**kwargs)


def distributed_load(state: dict[str, Any], checkpoint_dir: Path) -> None:
    """Load only the shards required by the destination state dictionary."""
    initialize_checkpoint_process_group()

    kwargs = {
        "state_dict": state,
        "storage_reader": dist_cp.FileSystemReader(str(checkpoint_dir)),
    }
    planner = _load_planner()
    if planner is not None:
        kwargs["planner"] = planner
    dist_cp.load(**kwargs)


def materialize_model_state(
    model,
    checkpoint_dir: str | os.PathLike,
) -> dict[str, torch.Tensor]:
    """Materialize a DCP model state on CPU for portable export."""
    cpu_state = {
        "model": {
            name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            for name, tensor in model.state_dict().items()
        }
    }
    distributed_load(cpu_state, Path(checkpoint_dir))
    return cpu_state["model"]


def save_portable_model(
    model,
    save_dir: str | os.PathLike,
    state_dict: dict[str, torch.Tensor] | None = None,
    max_shard_size: str = "5GB",
) -> None:
    """Export a Transformers-compatible sharded safetensors directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if state_dict is None:
        state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }
    state_dict = canonicalize_state_dict(state_dict)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(
            save_dir,
            state_dict=state_dict,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        generation_config_path = save_dir / "generation_config.json"
        if not generation_config_path.exists():
            from transformers import GenerationConfig

            GenerationConfig.from_model_config(model.config).save_pretrained(save_dir)
        readme_path = save_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(
                "---\nlibrary_name: transformers\n---\n\n"
                f"# {model.__class__.__name__}\n\n"
                "Portable safetensors export produced by the ZLM training system.\n"
            )
        return

    from safetensors.torch import save_file

    save_file(state_dict, save_dir / "model.safetensors")


class TrainingCheckpointManager:
    """Coordinate atomic DCP saves and exact training restoration."""

    def __init__(
        self,
        root_dir: str | os.PathLike,
        is_main_process: Callable[[], bool] = lambda: True,
        process_index: Callable[[], int] = lambda: 0,
        process_count: Callable[[], int] = lambda: 1,
        barrier: Callable[[str], None] | None = None,
        keep_last: int | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.is_main_process = is_main_process
        self.process_index = process_index
        self.process_count = process_count
        self.barrier = barrier or (lambda name: None)
        self.keep_last = keep_last

    def checkpoint_dir(self, step: int) -> Path:
        return self.root_dir / f"{step:012d}"

    def all_steps(self) -> list[int]:
        if not self.root_dir.exists():
            return []
        return sorted(
            int(path.name)
            for path in self.root_dir.iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and (path / SUCCESS_FILE).exists()
        )

    def latest_step(self) -> int | None:
        steps = self.all_steps()
        return steps[-1] if steps else None

    def save(
        self,
        step: int,
        distributed_state: dict[str, Any],
        trainer_state: dict[str, Any],
        manifest: CheckpointManifest,
    ) -> Path:
        """Save a checkpoint and publish it only after every rank succeeds."""
        if step != manifest.global_step:
            raise ValueError(
                f"Checkpoint step {step} does not match manifest step "
                f"{manifest.global_step}."
            )

        final_dir = self.checkpoint_dir(step)
        staging_dir = self.root_dir / f".{step:012d}.staging"

        if self.is_main_process():
            self.root_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True)
        self.barrier(f"checkpoint_stage_{step}")

        distributed_save(distributed_state, staging_dir / "distributed")

        rank_state_path = staging_dir / f"rank_{self.process_index():05d}.pt"
        torch.save(
            {"rng": capture_rng_state()},
            rank_state_path,
        )
        self.barrier(f"checkpoint_state_{step}")

        if self.is_main_process():
            torch.save(trainer_state, staging_dir / "trainer.pt")
            (staging_dir / MANIFEST_FILE).write_text(
                json.dumps(asdict(manifest), indent=2)
            )
            (staging_dir / SUCCESS_FILE).write_text("complete\n")
            shutil.rmtree(final_dir, ignore_errors=True)
            staging_dir.rename(final_dir)
        self.barrier(f"checkpoint_publish_{step}")

        self._remove_old_checkpoints()
        return final_dir

    def restore(
        self,
        step: int,
        distributed_state: dict[str, Any],
    ) -> tuple[CheckpointManifest, dict[str, Any]]:
        """Restore distributed tensors plus rank and trainer state."""
        checkpoint_dir = self.checkpoint_dir(step)
        if not (checkpoint_dir / SUCCESS_FILE).exists():
            raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint_dir}")

        manifest_data = json.loads((checkpoint_dir / MANIFEST_FILE).read_text())
        manifest = CheckpointManifest(**manifest_data)
        if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema {manifest.schema_version}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION}."
            )
        if manifest.global_step != step:
            raise ValueError(
                f"Checkpoint directory step {step} contains step "
                f"{manifest.global_step}."
            )
        if manifest.world_size != self.process_count():
            raise ValueError(
                f"Checkpoint world size {manifest.world_size} does not match "
                f"the current world size {self.process_count()}; per-rank RNG and "
                "dataset state cannot be restored exactly."
            )

        distributed_load(distributed_state, checkpoint_dir / "distributed")

        rank_state_path = checkpoint_dir / f"rank_{self.process_index():05d}.pt"
        rank_state = torch.load(rank_state_path, map_location="cpu", weights_only=False)
        restore_rng_state(rank_state["rng"])

        trainer_state = torch.load(
            checkpoint_dir / "trainer.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.barrier(f"checkpoint_restore_{step}")
        return manifest, trainer_state

    def _remove_old_checkpoints(self) -> None:
        if not self.is_main_process() or self.keep_last is None:
            return
        steps = self.all_steps()
        for step in steps[:-self.keep_last]:
            shutil.rmtree(self.checkpoint_dir(step), ignore_errors=True)
