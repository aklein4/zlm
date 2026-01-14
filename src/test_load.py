
from torch.distributed.checkpoint import load, FileSystemReader

import huggingface_hub as hf
import os
import omegaconf

import torch_xla.experimental.distributed_checkpoint as xc

from models.llama import LlamaForCausalLM

def main():
    
    path = hf.snapshot_download(
        repo_id="aklein4/prime-testing_test",
        allow_patterns=["*000000000009*"],
        repo_type="model",
        local_dir="../local_data"
    )
    path = os.path.join(path, "000000000009")
    print(f"Downloaded to {path}")

    config = omegaconf.OmegaConf.load(os.path.join(path, "config.json"))
    model = LlamaForCausalLM(config)

    model_sd = model.state_dict()
    reader = FileSystemReader(path)
    load(
        model_sd,
        storage_reader=reader,
        no_dist=True,
        planner=xc.SPMDLoadPlanner(),
    )

    model.load_state_dict(model_sd)
    print("Model loaded successfully!")


if __name__ == "__main__":
    main()