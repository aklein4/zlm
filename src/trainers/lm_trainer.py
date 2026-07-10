import torch

from models.llama import LlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn


class LMTrainer(BaseTrainer):

    model: LlamaForCausalLM


    def post_init(self):
        self.model.model.embed_tokens.weight.no_muon = True
        try:
            self.model.lm_head.weight.no_muon = True
        except:
            # ShardedModule
            self.model.lm_head._orig_mod.weight.no_muon = True


    def forward(self, input_ids, attention_mask=None, labels=None):

        # this hack gets around some models not having a pad token id in their embeddings
        pad_token_id = self.model.config.pad_token_id
        if attention_mask is None:
            attention_mask = input_ids != pad_token_id
        inputs_for_model = torch.where(
            attention_mask,
            input_ids,
            torch.zeros_like(input_ids)
        )

        logits = self.model(
            input_ids=inputs_for_model,
            attention_mask=attention_mask,
            shift_states=True,
            return_dict=True,
        ).logits

        if labels is None:
            labels = input_ids
        labels = torch.where(
            attention_mask,
            labels,
            torch.full_like(labels, -100),
        )

        lm_loss = lm_loss_fn(
            logits,
            labels,
            ignore_index=-100,
            shift_logits=False,
            shift_labels=True,
        )
        lm_acc = lm_acc_fn(
            logits,
            labels,
            ignore_index=-100,
            shift_logits=False,
            shift_labels=True,
        )

        loss = lm_loss

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "atom_count": attention_mask.long().sum(), # number of tokens that the model sees this step
        }
