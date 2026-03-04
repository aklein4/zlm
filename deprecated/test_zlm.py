import torch

import hydra
import omegaconf

from utils.import_utils import import_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    config.model.attention_kernel = None

    model = import_model(config.model.type)(config.model).to(DEVICE)
    print(f"\nModel initialized: {config.model.type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):_}\n")


    input_ids = torch.randint(
        0, config.model.vocab_size-10, (3, model.input_length)
    ).to(DEVICE)
    output_ids = torch.randint(
        0, config.model.vocab_size-10, (3, model.output_length)
    ).to(DEVICE)
    
    input_mask = torch.ones_like(input_ids, dtype=torch.bool)
    output_mask = torch.ones_like(output_ids, dtype=torch.bool)

    z, mu = model.encode(
        input_ids, output_ids,
        input_mask=input_mask,
        output_mask=output_mask
    )
    print(f"z shape: {z.shape}, mu shape: {mu.shape}\n")

    logits, z_states = model.decode(
        input_ids, output_ids, z,
        input_mask=input_mask,
        output_mask=output_mask
    )
    print(f"logits shape: {logits.shape}, z_states shape: {z_states.shape}\n")    


if __name__ == "__main__":
    main()