import torch
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class LMTrainer(BaseTrainer):

    def forward(self, input_ids, output_ids):
        pad_token_id = self.model.config.pad_token_id

        all_ids = torch.cat([input_ids, output_ids], dim=1)
        
        elementwise_pad_mask = (all_ids != pad_token_id)
        ids_for_model = torch.where(
            elementwise_pad_mask,
            all_ids,
            torch.zeros_like(all_ids)
        )

        logits, _ = self.model(
            input_ids=ids_for_model,
            shift_states=True,
            elementwise_pad_mask=elementwise_pad_mask
        )

        lm_loss = F.cross_entropy(
            logits.view(-1),
            all_ids[:, 1:].reshape(-1),
            ignore_index=pad_token_id,
        )

        return lm_loss, {
            "lm_loss": lm_loss,
        }
    