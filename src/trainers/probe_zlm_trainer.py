import torch
import torch.nn.functional as F

import numpy as np

from trainers.base_trainer import BaseTrainer
from models.zlm import ZLMModel
from utils.loss_utils import lm_loss_fn, lm_acc_fn 
from utils.torch_modules import ARLinear


class ProbeZLMTrainer(BaseTrainer):
    
    model: ZLMModel


    def post_init(self):        
        assert self.model.is_probe, "ProbeZLMTrainer can only be used with a probe model"

        # disable muon for parameters that shouldn't use it
        # (io embeddings are 1D)
        self.model.embed_tokens._orig_mod.weight.no_muon = True
        self.model.lm_head._orig_mod.weight.no_muon = True
        
        self.model.encoder_sep_token.no_muon = True
        self.model.encoder_z_tokens.no_muon = True

        self.model.decoder_z_tokens.no_muon = True
        self.model.decoder_start_output_token.no_muon = True

        for m in self.model.modules():
            if isinstance(m, ARLinear):
                m.weight.no_muon = True


    def get_trainable_parameters(self, model: ZLMModel):
        params = []
        for name, p in model.named_parameters():
            
            if name.count("embed_tokens"):
                continue

            params.append(p)

        return params


    def forward(self, input_ids, output_ids):
        pad_token_id = self.model.config.pad_token_id

        # prepare inputs
        input_mask = (input_ids != pad_token_id)
        output_mask = (output_ids != pad_token_id)

        # model doesn't actually have the pad token embedding
        input_for_model = torch.where(
            input_mask,
            input_ids,
            torch.zeros_like(input_ids)
        )
        output_for_model = torch.where(
            output_mask,
            output_ids,
            torch.zeros_like(output_ids)
        )

        # encode and decode
        with torch.no_grad():
            z, mu = self.model.encode(
                input_for_model, output_for_model,
                input_mask=input_mask, output_mask=output_mask,
            )

        progress = torch.randint_like(
            input_ids[:, 0], low=0, high=self.model.z_length+1
        )
        logits, z_states = self.model.decode(
            input_for_model, output_for_model, z,
            input_mask=input_mask,
            output_mask=output_mask,
            progress=progress,
        )

        # get the lm loss metrics
        lm_loss = lm_loss_fn(
            logits,
            output_ids,
            ignore_index=pad_token_id,
            shift_labels=False,
            shift_logits=False,
        )
        lm_acc = lm_acc_fn(
            logits,
            output_ids,
            ignore_index=pad_token_id,
            shift_labels=False,
            shift_logits=False,
        )

        loss = lm_loss

        aux = {

            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            
            "atom_count": (output_ids != pad_token_id).long().sum(),
        }

        return loss, aux
