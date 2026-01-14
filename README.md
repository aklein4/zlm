
# ZLM

A work in progress.

## TPU VM Setup Instructions

1. Create VM with version: `tpu-ubuntu2204-base`

2. Run command: `git clone https://github.com/aklein4/latent-reasoning.git`

3. Run command: `cd ~/zlm && . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>`

## Running in the background

`nohup python ~/zlm/src/train.py > train.log 2>&1 &`
