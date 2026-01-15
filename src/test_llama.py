

import torch_xla
import torch_xla.core.xla_model as xm

import hydra
import omegaconf

from utils.import_utils import import_model


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    with torch_xla.device():
        model = import_model(config.model.model_class)(config.model)

    xm.rendezvous("load_model")

    print(f"Model initialized: {config.model.model_class}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):_}")


if __name__ == "__main__":
    main()