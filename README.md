
# ZLM

Train parallel and test serial for scalable latent reasoning. Currently a work in progress.

Built on the [aklein4/easy-torch-tpu](https://github.com/aklein4/easy-torch-tpu) training framework.

See [docs/checkpoints.md](docs/checkpoints.md) for exact TPU training resume and
portable safetensors export, and [docs/transformers.md](docs/transformers.md) for
the Transformers model interface.

CPU/GPU compatibility tests do not import torch-xla and can be run with:

```bash
~/iTTT/src/.venv/bin/python -m unittest discover -s tests -v
```


### Acknowledgements

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).
