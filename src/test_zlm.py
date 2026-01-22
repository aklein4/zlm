import torch

import hydra
import omegaconf

from utils.import_utils import import_model


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    config.model.attention_kernel = None

    model = import_model(config.model.type)(config.model)
    print(f"Model initialized: {config.model.type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):_}")

    print(model.diffusion_head.out_norm.embed.weight)

    x = torch.ones(config.model.hidden_size)
    print(model.diffusion_head.out_norm(x, torch.tensor([0]).long()))


if __name__ == "__main__":
    main()