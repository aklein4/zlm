import torch

import hydra
import omegaconf

from utils.import_utils import import_model


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    config.model.attention_kernel = None

    model = import_model(config.model.model_class)(config.model)
    print(f"Model initialized: {config.model.model_class}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):_}")

    input_ids = torch.randint(
        0, config.model.vocab_size, (3, 16)
    )
    pad_mask = (input_ids != config.model.pad_token_id)
    logits, _ = model(
        input_ids=input_ids,
        shift_states=True,
        elementwise_pad_mask=pad_mask
    )

    input_ids = torch.cat(
        [
            input_ids[:, :4],
            torch.full_like(input_ids[:, :3], config.model.pad_token_id),
            input_ids[:, 4:8],
            torch.full_like(input_ids[:, :2], config.model.pad_token_id),
            input_ids[:, 8:],
        ],
        dim=1
    )
    pad_mask = (input_ids != config.model.pad_token_id)
    input_ids = torch.where(
        pad_mask,
        input_ids,
        torch.zeros_like(input_ids)
    )

    pad_logits, _ = model(
        input_ids=input_ids,
        shift_states=True,
        elementwise_pad_mask=pad_mask
    )

    diff = torch.abs(logits[:, -1] - pad_logits[:, -1]).max()
    print(f"Max difference after padding: {diff.item()}")


if __name__ == "__main__":
    main()