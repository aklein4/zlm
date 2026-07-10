# Checkpoints

Training recovery and portable model export use different formats.

Native training checkpoints use `torch.distributed.checkpoint`. On TPU,
`SPMDSavePlanner` and `SPMDLoadPlanner` operate directly on XLA shards instead
of gathering a full copy of the model on every host. A complete checkpoint is
published only after its manifest, per-rank RNG state, trainer state, and
`_SUCCESS` marker have been written.

`global_step` always means the number of completed optimizer updates. A
checkpoint named `000000024000` contains the state after 24,000 updates, and
the next update is logged as step 24,001.

Configure exact continuation with:

```yaml
checkpoint:
  resume_from: /shared/checkpoints/my-run
  resume_step: 24000  # null selects the latest complete checkpoint
```

Exact resume restores model weights, optimizer state, scheduler state, trainer
state, and per-rank RNG state. Dataset state is restored when the dataset
implements `state_dict()` and `load_state_dict()`. Set `require_data_state: true`
to reject a resume when the input pipeline cannot restore its cursor.

Portable initialization is separate:

```yaml
checkpoint:
  initialize_from: owner/model
  initialize_step: 24000
  initialize_revision: main  # pin a commit for reproducible initialization
  initialize_strict: true
```

Initialization loads model weights only and starts training at global step zero.
Portable checkpoints contain sharded safetensors, a Transformers `config.json`,
an optional tokenizer, and the resolved training configuration.
`checkpoint.export_interval` controls how often the more expensive consolidated
export runs independently of the native DCP recovery interval.

The DCP path must be visible to every participating host. A shared filesystem
or mounted object-storage filesystem is required for multi-host training.
