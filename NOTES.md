
# Changes versus reference

### removed
 - experimental
 - hf_models
 - launcher
 - metrics
 - models (with resnet and transact)
 - tests
 - torch_xla_models/tests
 - torch_xla_vision_models


### changed

train.py
 - remove profiling
 - modify logging
 - change how dataset/dataloading works
 - change how model class is indexed (functionally similar)

torchprime/torch_xla_models/model/model_utils.py
 - only import xla modules if on TPU

torchprime/torch_xla_models/model_rewriting/rematerialization_utils.py
 - pass remat_config instead of base config containing it

torchprime/torch_xla_models/attention.py
 - only import xla modules if on TPU
 - add is_causal option to attention module
 - must be on XLA for splash_attention and flash_attention
 - check XLA available before mesh stuff
 - modify block sizes in flash_attention
 - pad to block sizes

torchprime/torch_xla_models/utils/profiling.py
 - task.max_steps -> trainer.max_steps

trainers/base_trainer.py (torch_xla_models/trainer/base_trainer.py)
 - init
   - 2 space tabs -> 4 space tabs
   - remove profiling stuff
   - remove tensorboard stuff
   - call add_activation_checkpointing_and_scan on specified submodules instead of only base model
   - config.task -> config.trainer
   - add wandb and custom checkpointing init
 - _create_optimizer
   - use custom AdamW optimizer
 - del
   - remove function (only does tensorboard stuff)
 - _initialize_tensorboard_writer
   - remove function (only does tensorboard stuff)
 - _get_train_dataloader
   - remove sampler (we do not shuffle)
 - _log_to_tensorboard
   - remove function (only does tensorboard stuff)
 - save_checkpoint
   - add custom checkpoint saving
 - train_loop
   - enable grad for all parameters
   - use explicit max_step and steps_per_epoch
   - skip pretrained_step
   - handle aux from train_step
   - perform step_closure every step
   - add wandb lossing to step closure
   - remove profiling stuff
   - mark_step before saving checkpoint
  - finalize_training
   - remove function
  - train_step
   - call self.forward to get loss and aux
  - clip_gradients
   - config.task -> config.trainer

models/xla.py (torchprime/torch_xla_models/model/base_causal_lm.py)
 - indent when saving config
 - change checkpoint saving logic (unused?)

models/llama.py (torchprime/torch_xla_models/model/llama/model.py)
 - based on nn.Module instead of BaseXLAModel or BaseCausalLM
 - only import XLA if available
 - remove some trace_me decorators
 - add is_causal argument to LlamaAttention
 - .size() -> shape
 - add elementwise_attention_bias funcionality
 - only offload if XLA is available
 - modify _init_weights
 - allow shifting states before calculating logits
 - ignore pad tokens in loss calculation

### added

optimizers/adamw.py
 - add gradient checkpointing and update clipping

data/datasets.py
 - load dataset with correct sharding when streaming

collators/
 - custom collating functions for datasets

# Startup

1. Set up logging (not important)

2. Validate Config
 - just checks context parallelism

3. Print config

4. More logging setup

5. Set random seed

6. Set the default dtype to float32
 - TODO: why?

7. Initialize model
 - wraps initialization (and therefore model weights) in torch_dtype -> model_dtype 
 - wraps initialization in torch_xla.device()
   - TODO: what does this do and why?
 - rendezvous devices
   - TODO: how does this compare to other sync methods?
 - log model info

8. Initialize trainer
 - apply_xla_patch_to_nn_linear
   - to avoid breaking sharding
 - auto_trace
   - just for profiling/debugging?
 - setup_sharding_and_mesh
   - get a mesh based on the config
   - set global mesh (just sets global variable)
   - get the sharding spec (later used for dataloader)
   - apply sharding to the model
     - shard parameters as found in config (ignore others)
     - apply activation sharding to modules as found in config by turning into ShardedModule (ignore others)
 - mark pure modules (that don't modify state in forward pass) for improved compilation and tracing
 - add_activation_checkpointing_and_scan
   - TODO: this logic is convoluted
   - TODO: which modules should have activation sharding?
   - TODO: how can this work with multiple transformers in a model?
 - add_optimization_barriers
   - modifies forward pass to add optimization barrier to every module of type listed in remat_config.optimization_barrier_layers

# Questions
 - What is the deal with dtypes?
 - What is the difference between the device syncing methods (xm.rendevous...)?