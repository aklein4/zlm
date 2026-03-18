import torch

from models.iml import IMLModel, IMLLinear
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn
from utils.sharding_utils import shard_with_gradients


class IMLTrainer(BaseTrainer):

    model: IMLModel


    def post_init(self):
        self.model.model.embed_tokens.weight.no_muon = True
        try:
            self.model.lm_head.weight.no_muon = True
        except:
            # ShardedModule
            self.model.lm_head._orig_mod.weight.no_muon = True

        for m in self.model.modules():
            if isinstance(m, IMLLinear):

                m.loss_buffer.requires_grad_(True)
                m.loss_buffer.grad = torch.zeros_like(m.loss_buffer)

                m.log_loss_buffer.requires_grad_(True)
                m.log_loss_buffer.grad = torch.zeros_like(m.log_loss_buffer)


    def forward(self, input_ids):

        doubled_inputs = torch.cat([input_ids, input_ids], dim=0)
        doubled_inputs = shard_with_gradients(doubled_inputs)

        logits, _ = self.model(
            input_ids=doubled_inputs,
            shift_states=True,
            sequences_to_keep=slice(0, input_ids.shape[0])
        )

        lm_loss = lm_loss_fn(
            logits,
            input_ids,
            ignore_index=self.model.config.pad_token_id,
            shift_logits=False,
            shift_labels=True,
        )
        lm_acc = lm_acc_fn(
            logits,
            input_ids,
            ignore_index=self.model.config.pad_token_id,
            shift_logits=False,
            shift_labels=True,
        )

        loss = lm_loss

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "atom_count": (input_ids != self.model.config.pad_token_id).long().sum(), # number of tokens that the model sees this step
        }
    

    def post_backward(self, **batch):
        return {
            "iml_loss": self.model.get_previous_loss(),
            "iml_log_loss": self.model.get_previous_log_loss(),
        }
    