import torch

from trainers.base_trainer import BaseTrainer
from utils import loss as loss_utils

class LLMTrainer(BaseTrainer):

    def forward(self, input_ids):
        pad_token_id = self.model.config.pad_token_id

        logits, _ = self.model(
            input_ids=input_ids,
            shift_states=True
        )

        losses = loss_utils.fast_lm_loss(
            logits=logits,
            labels=input_ids,
            ignore_index=pad_token_id,
            shift_logits=False,
            shift_labels=True
        )

        loss = losses['loss']
        aux = {
            'lm_loss': loss,
            'acc': losses['acc'],
            'pcorr': losses['pcorr'],

            # check for NaNs
            'nan_loss': (~torch.isfinite(loss)).any().float(),
        }

        return loss, aux
    