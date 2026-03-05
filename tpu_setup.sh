#!/bin/bash

: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command to set the environment variables in the current shell.

Arguments:
    $1: Hugging Face ID (ex. aklein4)
    $2: Hugging Face token
    $3: wandb token

Example:
    . tpu_setup.sh <HF_ID> <HF_TOKEN> <WANDB_TOKEN>
'

# boilerplate
sudo apt-get update -y

# install python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-pip

# update path(?)
export PATH="/home/$USER/.local/bin:$PATH"

# # upgrade and update pip
python3.11 -m pip --upgrade pip
python3.11 -m pip --upgrade setuptools

# install torch
python3.11 -m pip torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python3.11 -m pip torch_xla[tpu]==2.9.0
python3.11 -m pip --pre torch_xla[pallas]==2.9.0 --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html

# install extras
python3.11 -m pip setuptools==67.7.2
python3.11 -m pip -r tpu_requirements.txt

# login to huggingface
hf auth login --token $2

# login to wandb
python -m wandb login $3

# create a .env file to store the Hugging Face ID
echo "HF_ID=$1" > .env
