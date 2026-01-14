#!/bin/bash

: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command to set the environment variables in the current shell.

Arguments:
    $1: Huggingface token
    $2: wandb token

Example:
    . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>
'

# # upgrade and update pip
pip install --upgrade pip
pip install --upgrade setuptools

# install torch
pip install torch==2.9.0.dev20250709+cpu --index-url https://download.pytorch.org/whl/nightly/cpu

# install torch_xla for TPU VM
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev20250709-cp310-cp310-linux_x86_64.whl' -f https://storage.googleapis.com/libtpu-wheels/index.html

# update path(?)
export PATH="/home/$USER/.local/bin:$PATH"

# install extras
pip install setuptools==67.7.2
pip install transformers==4.52.1 datasets==4.0.0 hydra-core==1.3.0 optax==0.2.4 wandb

# login to huggingface
# huggingface-cli login --token $1 --add-to-git-credential

# login to wandb
python -m wandb login $1
