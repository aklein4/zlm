"""Base trainer module for TPU-based model training using PyTorch/XLA.

This script provides a `Trainer` class that sets up model sharding, activation checkpointing,
optimization, and the training loop with XLA-specific configurations. It is designed to work with
distributed TPU training.
"""

import logging
import math
import os
from pathlib import Path
from timeit import default_timer as timer
import time

import torch
import torch.nn.utils as nn_utils
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr

from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear
from transformers import (
    get_scheduler,
)

from torchprime.torch_xla_models.model_rewriting.assume_pure import (
    mark_pure_modules,
)
from torchprime.torch_xla_models.model_rewriting.auto_trace import auto_trace
from torchprime.torch_xla_models.model_rewriting.sharding_initialization import (
    setup_sharding_and_mesh,
)
from torchprime.utils.parallelism_utils import lb_cp_enabled, reorder_sequence

import wandb
import huggingface_hub as hf

from utils.import_utils import import_optimizer, import_collator
from utils import constants
from utils.remat_utils import advanced_remat
from utils.git_utils import get_current_commit_hash, is_worktree_dirty
from utils.checkpointing import (
    CheckpointManifest,
    SUCCESS_FILE,
    TrainingCheckpointManager,
    materialize_model_state,
    prime_optimizer,
    save_portable_model,
)


logger = logging.getLogger(__name__)


class BaseTrainer:
    """Trainer class for TPU-accelerated model training using PyTorch/XLA.

    This class encapsulates model preparation, optimizer configuration, data loading,
    and the training loop. It is designed to handle distributed training across TPU cores,
    enabling features like SPMD sharding, and activation checkpointing.

    Args:
        model: The model to train.
        config: Configuration object containing training hyperparameters and setup.
        train_dataset: Dataset used for training.
    """

    minibatch: bool

    def __init__(
        self,
        model: torch.nn.Module,
        config: DictConfig,
        train_dataset: Dataset | IterableDataset,
    ):
        self.config = config
        self.device = xm.xla_device()
        
        self.global_batch_size = self.config.trainer.global_batch_size
        self.train_dataset = train_dataset
        self.global_step = 0
        self.epoch = 0
        self.atoms_seen = 0

        self.model = self.prepare_model(model, config)

        self.optimizers, self.lr_schedulers = self.prepare_optimization(self.model, config)

        checkpoint_path = self.config.checkpoint.path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                constants.LOCAL_DATA_PATH,
                "training_checkpoints",
                f"{self.config.project}_{self.config.name}",
            )
        if self.config.checkpoint.resume_from is not None:
            checkpoint_path = self.config.checkpoint.resume_from
            resume_path = Path(checkpoint_path)
            if (resume_path / SUCCESS_FILE).exists() and resume_path.name.isdigit():
                inferred_step = int(resume_path.name)
                configured_step = self.config.checkpoint.resume_step
                if configured_step is not None and configured_step != inferred_step:
                    raise ValueError(
                        f"resume_step={configured_step} does not match {resume_path}."
                    )
                self.config.checkpoint.resume_step = inferred_step
                checkpoint_path = resume_path.parent
        self.checkpoint_manager = TrainingCheckpointManager(
            checkpoint_path,
            is_main_process=constants.PROCESS_IS_MAIN,
            process_index=constants.PROCESS_INDEX,
            process_count=constants.PROCESS_COUNT,
            barrier=xm.rendezvous,
            keep_last=self.config.checkpoint.keep_last,
        )

        # set up saving
        if not self.config.debug and constants.PROCESS_IS_MAIN():
            os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)

            if self.config.checkpoint.upload_to_hub:
                if constants.HF_ID is None:
                    raise ValueError("Set HF_ID before enabling checkpoint Hub uploads.")
                self.repo_name = f"{constants.HF_ID}/{self.config.project}_{self.config.name}"
                hf.create_repo(
                    self.repo_name,
                    private=self.config.checkpoint.hub_private,
                    exist_ok=True,
                )

            # create the wandb project
            notes = f"GIT HASH: {get_current_commit_hash()}"
            if self.config.notes is not None:
                notes += f"\n\n{self.config.notes}"
            wandb.init(
                project=self.config.project,
                name=self.config.name,
                notes=notes,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        self.post_init()

        if self.config.checkpoint.resume_from is not None:
            self.restore_checkpoint(self.config.checkpoint.resume_step)

        # Execute all initialization work queued so far before starting training.
        torch_xla.sync()


    def post_init(self):
        return


    def prepare_model(
        self, model, config: DictConfig
    ):
        """ Prepares the model for training by setting up sharding and rematerialization. """
        
        # Recursively replace `nn.Linear` layers with einsum operations in the model.
        # Without this patch, an `nn.Linear` module will flatten non-contracting dimensions
        # (e.g. batch and sequence), thus destroying the sharding constraints on those dimensions.
        model = apply_xla_patch_to_nn_linear(model)

        # Add `xp.Trace` to linear layers in the module tree (just for profiling?).
        # model = auto_trace(model)

        # print model parameters that to not have sharding spec
        config_names = set(self.config.model.sharding.keys())
        # param_names = set()
        # for name, p in model.named_parameters():
        #     if p is not None:
        #         param_names.add(re.sub("\.\d+\.", ".*.", name))
        # all_found = True
        # for name in param_names:
        #     if name not in config_names:
        #         logger.warning(f"Parameter {name} does not have sharding spec!")
        #         all_found = False
        # if all_found:
        #     logger.info("All model parameters have sharding spec.")

        # Setup SPMD mesh and shard the model.
        model, self.input_sharding_spec, self.minibatch, shard_info = setup_sharding_and_mesh(
            model, config
        )
        logger.info("Sharding info:")
        logger.info(f"    Seen params:      {len(shard_info['seen_params'])}")
        logger.info(f"    Implied params:   {len(shard_info['implied_params'])}")
        logger.info(f"    Seen modules:     {len(shard_info['seen_modules'])}")
        logger.info(f"    Unused names:     {len((config_names - shard_info['seen_params']) - shard_info['seen_modules'])}")
        logger.info(f"    Unsharded params: {len(shard_info['unsharded_params'])}")

        model = mark_pure_modules(model, config)

        model = advanced_remat(model, config)

        return model


    def get_trainable_parameters(self, model: torch.nn.Module):
        if hasattr(model, "get_trainable_parameters"):
            return model.get_trainable_parameters()
        return model.parameters()


    def prepare_optimization(
        self,
        model: torch.nn.Module,
        config: DictConfig
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """ Sets up the optimizer and learning rate scheduler. """

        params = self.get_trainable_parameters(model)

        if "multiple_optimizers" in config.trainer:

            assert isinstance(params, dict)
            assert len(params) == len(config.trainer.multiple_optimizers)

            optimizers = {}
            lr_schedulers = {}

            for key, c in config.trainer.multiple_optimizers.items():
                if len(list(params[key])) == 0:
                    logger.warning(f"No parameters found for optimizer {key}!")
                    continue
                
                optimizer_config = c.optimizer
                lr_scheduler_config = c.lr_scheduler

                optimizers[key] = import_optimizer(optimizer_config.type)(
                    params=params[key],
                    **optimizer_config.kwargs,
                )

                lr_schedulers[key] = get_scheduler(
                    name=lr_scheduler_config.type,
                    optimizer=optimizers[key],
                    num_warmup_steps=lr_scheduler_config.num_warmup_steps,
                    num_training_steps=(
                        lr_scheduler_config.num_training_steps if "num_training_steps" in lr_scheduler_config else None
                    ),
                    scheduler_specific_kwargs=lr_scheduler_config.kwargs,
                )

            return optimizers, lr_schedulers

        assert not isinstance(params, dict)

        optimizer = import_optimizer(config.trainer.optimizer.type)(
            params=params,
            **config.trainer.optimizer.kwargs,
        )

        lr_scheduler = get_scheduler(
            name=config.trainer.lr_scheduler.type,
            optimizer=optimizer,
            num_warmup_steps=config.trainer.lr_scheduler.num_warmup_steps,
            num_training_steps=(
                config.trainer.lr_scheduler.num_training_steps if "num_training_steps" in config.trainer.lr_scheduler else None
            ),
            scheduler_specific_kwargs=config.trainer.lr_scheduler.kwargs,
        )

        return {"main": optimizer}, {"main": lr_scheduler}
    

    def _get_train_dataloader(self) -> pl.MpDeviceLoader:

        num_replicas = xr.process_count()
        logger.info("Num replicas: %d", num_replicas)

        if self.minibatch:
            # Each process loads the per-host batch size.
            batch_size = self.global_batch_size // num_replicas
        else:
            # Each process will load the global batch, then discard the unneeded parts.
            batch_size = self.global_batch_size

        # handle the collator
        collator = import_collator(self.config.data.collator.type)(
            **self.config.data.collator.kwargs
        )
        dataloader = DataLoader(
            self.train_dataset,
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        loader = pl.MpDeviceLoader(
            dataloader, self.device, input_sharding=self.input_sharding_spec
        )
        
        return loader
    

    def save_checkpoint(
        self,
        step: int,
    ):
        """Save exact training state, then optionally export portable weights."""
        if self.config.debug:
            logger.info("Skipping checkpoint %d in debug mode", step)
            return
        if step != self.global_step:
            raise ValueError(
                f"Cannot save step {step}; the trainer has completed "
                f"{self.global_step} updates."
            )

        logger.info("[SAVING] Starting distributed checkpoint...")
        xm.mark_step()
        xm.wait_device_ops()
        xm.rendezvous(f"checkpoint_start_{step}")

        distributed_state = {"model": self.model.state_dict()}
        distributed_state.update({
            f"optimizer_{name}": optimizer.state_dict()
            for name, optimizer in self.optimizers.items()
        })
        manifest = CheckpointManifest(
            global_step=step,
            examples_seen=step * self.global_batch_size,
            atoms_seen=self.atoms_seen,
            git_commit=get_current_commit_hash(),
            git_dirty=is_worktree_dirty(),
            world_size=constants.PROCESS_COUNT(),
        )
        checkpoint_dir = self.checkpoint_manager.save(
            step,
            distributed_state,
            self.trainer_state_dict(),
            manifest,
        )

        export_error_path = checkpoint_dir / "_EXPORT_ERROR"
        if constants.PROCESS_IS_MAIN():
            try:
                export_interval = self.config.checkpoint.export_interval
                should_export = (
                    self.config.checkpoint.export_safetensors
                    and export_interval is not None
                    and step % export_interval == 0
                )
                if should_export:
                    export_dir = checkpoint_dir / "transformers"
                    state = materialize_model_state(
                        self.model,
                        checkpoint_dir / "distributed",
                    )
                    save_portable_model(
                        self.model,
                        export_dir,
                        state_dict=state,
                        max_shard_size=self.config.checkpoint.max_shard_size,
                    )
                    OmegaConf.save(self.config, export_dir / "train_config.yaml")
                    self._save_tokenizer(export_dir)

                    if self.config.checkpoint.upload_to_hub:
                        logger.info("Uploading checkpoint to %s", self.repo_name)
                        hf.HfApi().upload_folder(
                            repo_id=self.repo_name,
                            folder_path=export_dir,
                            path_in_repo=f"{step:012d}",
                            repo_type="model",
                        )
                export_error_path.unlink(missing_ok=True)
            except Exception as error:
                export_error_path.write_text(f"{type(error).__name__}: {error}\n")
        xm.rendezvous(f"checkpoint_exported_{step}")

        if export_error_path.exists():
            raise RuntimeError(
                f"Checkpoint {step} was saved, but portable export failed: "
                f"{export_error_path.read_text().strip()}"
            )

        logger.info("[SAVING] Finished distributed checkpoint.")


    def _save_tokenizer(self, save_dir: Path) -> None:
        tokenizer_url = self.config.checkpoint.tokenizer_url
        if tokenizer_url is None and "tokenizer_url" in self.config.data.collator.kwargs:
            tokenizer_url = self.config.data.collator.kwargs.tokenizer_url
        if tokenizer_url is None:
            return

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.save_pretrained(save_dir)


    def trainer_state_dict(self) -> dict:
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "atoms_seen": self.atoms_seen,
            "config": OmegaConf.to_container(self.config, resolve=True),
            "lr_schedulers": {
                name: scheduler.state_dict()
                for name, scheduler in self.lr_schedulers.items()
            },
            "extra": self.extra_trainer_state_dict(),
        }
        if hasattr(self.train_dataset, "state_dict"):
            state["dataset"] = self.train_dataset.state_dict()
        return self._state_to_cpu(state)


    def extra_trainer_state_dict(self) -> dict:
        return {}


    def load_extra_trainer_state_dict(self, state: dict) -> None:
        if state:
            logger.warning("Ignoring unsupported extra trainer state: %s", state.keys())


    def _state_to_cpu(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach().cpu()
        if isinstance(state, dict):
            return {key: self._state_to_cpu(value) for key, value in state.items()}
        if isinstance(state, list):
            return [self._state_to_cpu(value) for value in state]
        if isinstance(state, tuple):
            return tuple(self._state_to_cpu(value) for value in state)
        return state


    def restore_checkpoint(self, step: int | None = None) -> int:
        if step is None:
            step = self.checkpoint_manager.latest_step()
        if step is None:
            raise FileNotFoundError(
                f"No complete checkpoint found in {self.checkpoint_manager.root_dir}."
            )

        if step > 0:
            for optimizer in self.optimizers.values():
                prime_optimizer(optimizer)

        distributed_state = {"model": self.model.state_dict()}
        distributed_state.update({
            f"optimizer_{name}": optimizer.state_dict()
            for name, optimizer in self.optimizers.items()
        })
        manifest, trainer_state = self.checkpoint_manager.restore(
            step,
            distributed_state,
        )

        saved_model_config = trainer_state.get("config", {}).get("model")
        current_model_config = OmegaConf.to_container(self.config.model, resolve=True)
        if (
            saved_model_config is not None
            and saved_model_config != current_model_config
            and not self.config.checkpoint.allow_model_config_mismatch
        ):
            raise ValueError(
                "The checkpoint model configuration differs from the current "
                "configuration. Set checkpoint.allow_model_config_mismatch=true "
                "only for an intentional migration."
            )

        self.model.load_state_dict(distributed_state["model"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(distributed_state[f"optimizer_{name}"])
        for name, scheduler in self.lr_schedulers.items():
            scheduler.load_state_dict(trainer_state["lr_schedulers"][name])

        self.global_step = manifest.global_step
        self.epoch = trainer_state["epoch"]
        self.atoms_seen = trainer_state["atoms_seen"]
        self.load_extra_trainer_state_dict(trainer_state.get("extra", {}))

        if "dataset" in trainer_state:
            if not hasattr(self.train_dataset, "load_state_dict"):
                raise TypeError(
                    "The checkpoint has dataset state, but the current dataset cannot restore it."
                )
            self.train_dataset.load_state_dict(trainer_state["dataset"])
        elif self.config.checkpoint.require_data_state:
            raise ValueError(
                "Exact resume was requested, but the dataset has no checkpointable state."
            )
        else:
            logger.warning(
                "Dataset state is unavailable; model and optimizer resume is exact, "
                "but input data will restart from the current dataset position."
            )

        logger.info("Restored training checkpoint at global step %d", self.global_step)
        return self.global_step
    

    def train_loop(self) -> None:

        # prepare model for training
        for p in self.model.parameters():
            p.requires_grad_(False)
        params = self.get_trainable_parameters(self.model)
        if isinstance(params, dict):
            for ps in params.values():
                for p in ps:
                    p.requires_grad_(True)
        else:
            for p in params:
                p.requires_grad_(True)
        
        self.model.train()
        self.model.zero_grad()

        # prepare data loader
        max_step = self.config.trainer.max_steps
        train_loader = self._get_train_dataloader()
        train_iterator = iter(train_loader)

        # print training information
        logger.info("Starting training")
        logger.info("    Max step: %d", max_step)
        logger.info("    Global batch size: %d", self.global_batch_size)
        logger.info(f"    Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"    Model dtype: {list(self.model.parameters())[0].dtype}")

        # global_step is the number of completed optimizer updates. Logs,
        # checkpoint names, and loop termination all use this same convention.
        error_count = 0
        training_start_time = timer()

        # run the training loop
        # TODO: enable multi-epoch training
        while self.global_step < max_step:
            try:
                batch = next(train_iterator)
            except StopIteration as error:
                logger.error("DataLoader exhausted at global step %d", self.global_step)
                if self.config.checkpoint.save_on_error:
                    self.save_checkpoint(self.global_step)
                raise RuntimeError("DataLoader exhausted before max_steps") from error
            except Exception:
                logger.exception(
                    "Unexpected error when fetching data at global step %d",
                    self.global_step,
                )
                error_count += 1
                if error_count > self.config.checkpoint.max_data_errors:
                    if self.config.checkpoint.save_on_error:
                        self.save_checkpoint(self.global_step)
                    raise RuntimeError("Too many errors when fetching data")
                time.sleep(self.config.checkpoint.data_error_backoff_seconds)
                continue
            error_count = 0

            # can be reached by forward
            self.step = self.global_step

            # when context parallel and load balance context parallel is enabled,
            # we will reorder the sequence here for each batch
            if lb_cp_enabled(self.config):
                batch = {
                    key: reorder_sequence(
                        tensor=value,
                        cp_size=self.config.ici_mesh.context,
                        seq_dim=1,
                        to_contiguous=False,
                    )
                    for key, value in batch.items()
                }

            # perform the training step
            trace_start_time = timer()
            loss, aux, grad_norm = self.train_step(batch)
            trace_end_time = timer()
            self.global_step += 1
            completed_step = self.global_step

            # post-step closure for logging
            def step_closure(
                start_time, epoch, step, loss, grad_norm, aux, trace_start_time, trace_end_time
            ):
                training_time_elapsed = (
                    timer() - start_time  
                ) / 3600 # in hours

                # keep track of something like number of tokens trained on
                if "atom_count" in aux.keys():
                    if isinstance(aux["atom_count"], torch.Tensor):
                        self.atoms_seen += aux["atom_count"].detach().item()
                    else:
                        self.atoms_seen += aux["atom_count"]

                loss = loss.detach().item()
                grad_norm = grad_norm.detach().item()

                logger.info(
                    "Hours elapsed: %.3f, epoch: %d, step: %d, loss: %.3f, grad_norm: %.3f, trace time: %.0f ms",
                    training_time_elapsed,
                    epoch,
                    step,
                    loss,
                    grad_norm,
                    (trace_end_time - trace_start_time) * 1000,
                )

                to_wandb = {}
                for k, v in aux.items():
                    if isinstance(v, torch.Tensor):
                        to_wandb[k] = v.detach().item()
                    else:
                        to_wandb[k] = v

                to_wandb["loss"] = loss
                to_wandb["grad_norm"] = grad_norm
                to_wandb["trace_time_ms"] = (trace_end_time - trace_start_time) * 1000
                to_wandb["epoch"] = epoch

                to_wandb["examples_seen"] = step * self.global_batch_size
                if "atom_count" in aux.keys():
                    to_wandb["atoms_seen"] = self.atoms_seen

                to_wandb["loss_nan"] = 1 - int(math.isfinite(loss))

                to_wandb["training_time_elapsed_hr"] = training_time_elapsed
                to_wandb["avg_time_per_step_s"] = training_time_elapsed * 3600 / step
                to_wandb["avg_steps_per_hr"] = step / training_time_elapsed

                if not self.config.debug and constants.PROCESS_IS_MAIN():
                    wandb.log(to_wandb, step=step)
            
            # execute
            xm.add_step_closure(
                step_closure,
                args=(
                    training_start_time,
                    self.epoch,
                    completed_step,
                    loss.detach().clone(),
                    grad_norm.detach().clone(),
                    {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in aux.items()},
                    trace_start_time,
                    trace_end_time,
                ),
                run_async=True,
            )
            xm.mark_step()

            # save checkpoint
            if self.global_step % self.config.trainer.checkpoint_interval == 0:
                self.save_checkpoint(self.global_step)

        xm.wait_device_ops()
        if (
            self.config.checkpoint.save_on_completion
            and self.global_step % self.config.trainer.checkpoint_interval != 0
        ):
            self.save_checkpoint(self.global_step)
        logger.info("Finished training run")


    @torch_xla.compile(full_graph=True)
    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict, torch.Tensor]:
        
        with torch.autocast('xla', dtype=torch.bfloat16, enabled=self.config.trainer.use_autocast):
            loss, aux = self.forward(**batch)

        loss.backward()
        
        grad_norm = self.clip_gradients()
        
        aux.update(self.optimization_step())

        self.model.zero_grad(set_to_none=False)

        return loss, aux, grad_norm


    def optimization_step(self):

        aux = {}
        key_name = lambda key, x: f"{key}_{x}" if len(self.optimizers) > 1 else x

        for key, optimizer in self.optimizers.items():

            opt_aux = optimizer.step()
            if opt_aux is not None:
                aux.update(
                    {key_name(key, k): v for k, v in opt_aux.items()}
                )

        for key, lr_scheduler in self.lr_schedulers.items():
            
            lr = lr_scheduler.get_last_lr()[0]
            aux.update({key_name(key, "lr"): lr})
            lr_scheduler.step()

        return aux


    def forward(self, **batch) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError(
            "The forward method should be implemented in the derived class."
        )


    def clip_gradients(self):
        """Clip gradients by the specified max norm and/or max absolute value."""
        max_grad_norm = self.config.trainer.max_grad_norm
        
        parameters = self.get_trainable_parameters(self.model)
        if isinstance(parameters, dict):
            p = []
            for v in parameters.values():
                p += list(v)
            parameters = p
        else:
            parameters = list(parameters)

        if max_grad_norm is None or max_grad_norm <= 0:
            grad_norm = nn_utils.get_total_norm(parameters, norm_type=2)
        else:
            grad_norm = nn_utils.clip_grad_norm_(
                parameters, max_norm=max_grad_norm, norm_type=2
            )
        max_grad_value = self.config.trainer.max_grad_value
        if max_grad_value is not None and max_grad_value > 0:
            nn_utils.clip_grad_value_(parameters, clip_value=max_grad_value)
        return grad_norm
