import torch
import torch.nn.functional as F

import torch_xla

from models.llama import LlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn


class IMLTrainer(BaseTrainer):
    """ Implicit Meta-Learning Trainer """

    model: LlamaForCausalLM


    @torch_xla.compile(full_graph=True)
    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict, torch.Tensor]:

        loss, _ = self.forward(batch["input_ids"])

        loss.backward()
        
        grad_norm = self.clip_gradients()
        self.optimizer.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        self.lr_scheduler.step()
        self.model.zero_grad()

        with torch.no_grad():
            _, train_aux = self.forward(batch["input_ids"])
            _, meta_aux = self.forward(batch["output_ids"])
            _, other_aux = self.forward(batch["other_ids"])

        aux = {}
        aux.update({f"train_{k}": v for k, v in train_aux.items()})
        aux.update({f"meta_{k}": v for k, v in meta_aux.items()})
        aux.update({f"other_{k}": v for k, v in other_aux.items()})

        return loss, aux, grad_norm, lr


    def forward(self, input_ids):

        logits, _ = self.model(
            input_ids=input_ids,
            shift_states=True,
        )

        lm_loss = lm_loss_fn(
            logits,
            input_ids,
            shift_logits=False,
            shift_labels=True,
        )
        lm_acc = lm_acc_fn(
            logits,
            input_ids,
            shift_logits=False,
            shift_labels=True,
        )

        loss = lm_loss

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
        }
    