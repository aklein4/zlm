
import hydra
import omegaconf

from utils.import_utils import import_model


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    model = import_model(config.model.model_class)(config.model)

    print(f"Model initialized: {config.model.model_class}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):_}")


if __name__ == "__main__":
    main()