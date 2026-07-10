import argparse
import tempfile

import huggingface_hub as hf
from transformers import AutoModelForCausalLM


def main(args):
    
    model = AutoModelForCausalLM.from_pretrained(args.in_url, trust_remote_code=True)

    hf.create_repo(
        args.out_url,
        private=False,
        exist_ok=True,
        repo_type="model",
    )

    api = hf.HfApi()
    with tempfile.TemporaryDirectory() as save_dir:
        model.save_pretrained(
            save_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        api.upload_folder(
            folder_path=save_dir,
            path_in_repo=f"{0:012d}",
            repo_id=args.out_url,
            repo_type="model",
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert HuggingFace model to checkpoint format compatible with easy-torch-tpu.")
    parser.add_argument("--in_url", type=str, required=True, help="HuggingFace model URL to convert.")
    parser.add_argument("--out_url", type=str, required=True, help="HuggingFace model URL to save converted model to.")
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="Maximum safetensors shard size.")
    
    args = parser.parse_args()

    main(args)
