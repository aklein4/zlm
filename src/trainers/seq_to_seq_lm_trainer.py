import torch

from models.custom_llama import CustomLlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn


class SeqToSeqLMTrainer(BaseTrainer):

    model: CustomLlamaForCausalLM


    def post_init(self):
        self.model.model.embed_tokens.weight.no_muon = True
        try:
            self.model.lm_head.weight.no_muon = True
        except:
            # ShardedModule
            self.model.lm_head._orig_mod.weight.no_muon = True


    def forward(
        self,
        input_ids,
        output_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        pad_token_id = self.model.config.pad_token_id

        if attention_mask is None:
            attention_mask = input_ids != pad_token_id
        if decoder_attention_mask is None:
            decoder_attention_mask = output_ids != pad_token_id

        all_ids = torch.cat(
            [
                input_ids,
                torch.full_like(output_ids[:, :1], self.model.config.bos_token_id),
                output_ids
            ],
            dim=1
        )
        elementwise_pad_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(output_ids[:, :1], dtype=torch.bool),
                decoder_attention_mask,
            ],
            dim=1
        )
        
        ids_for_model = torch.where(
            elementwise_pad_mask,
            all_ids,
            torch.zeros_like(all_ids)
        )

        logits = self.model(
            input_ids=ids_for_model,
            shift_states=slice(-(output_ids.shape[-1]+1), -1),
            elementwise_pad_mask=elementwise_pad_mask,
            return_dict=True,
        ).logits

        if labels is None:
            labels = output_ids
        labels = torch.where(
            decoder_attention_mask,
            labels,
            torch.full_like(labels, -100),
        )

        lm_loss = lm_loss_fn(
            logits,
            labels,
            ignore_index=-100,
            shift_logits=False,
            shift_labels=False,
        )
        lm_acc = lm_acc_fn(
            logits,
            labels,
            ignore_index=-100,
            shift_logits=False,
            shift_labels=False,
        )

        loss = lm_loss

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "elbo": lm_loss,
            "atom_count": decoder_attention_mask.long().sum(),
            "logit_nan": (~torch.isfinite(logits)).any().long(),
        }
