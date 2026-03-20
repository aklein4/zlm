import torch

import torch_xla

from models.llama import LlamaForCausalLM
from optimizers.amphibian import Amphibian
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn
from utils.sharding_utils import shard_with_gradients


class AmphibianTrainer(BaseTrainer):

    model: LlamaForCausalLM


    def post_init(self):
        self.model.model.embed_tokens.weight.no_muon = True
        try:
            self.model.lm_head.weight.no_muon = True
        except:
            # ShardedModule
            self.model.lm_head._orig_mod.weight.no_muon = True


    @torch_xla.compile(full_graph=True)
    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict, torch.Tensor]:
        
        optimizer: Amphibian = self.optimizers["main"]
        lr_scheduler = self.lr_schedulers["main"]

        input_ids = batch["input_ids"]
        ids_1, ids_2 = input_ids.chunk(2, dim=0)
        ids_1 = shard_with_gradients(ids_1)
        ids_2 = shard_with_gradients(ids_2)

        # first pass for g1
        with torch.autocast('xla', dtype=torch.bfloat16, enabled=self.config.trainer.use_autocast):
            loss_1, aux_1 = self.forward(ids_1)

        loss_1.backward()

        optimizer.handle_g1()
        self.model.zero_grad(set_to_none=False)

        # first pass for g2
        with torch.autocast('xla', dtype=torch.bfloat16, enabled=self.config.trainer.use_autocast):
            loss_2, aux_2 = self.forward(ids_2)
        
        loss_2.backward()

        optimizer.handle_g2()
        self.model.zero_grad(set_to_none=False)

        # second pass for g1
        with torch.autocast('xla', dtype=torch.bfloat16, enabled=self.config.trainer.use_autocast):
            loss_1_H, aux_1_H = self.forward(ids_1)
        
        loss_1_H.backward()
        
        optimizer.step()
        self.model.zero_grad(set_to_none=False)

        aux = {}
        for k, v in aux_1.items():
            aux[f"{k}_1"] = v
        for k, v in aux_2.items():
            aux[f"{k}_2"] = v
        for k, v in aux_1_H.items():
            aux[f"{k}_1_H"] = v
        aux["lm_loss"] = (aux["lm_loss_1"] + aux["lm_loss_2"]) / 2
        aux["lm_acc"] = (aux["lm_acc_1"] + aux["lm_acc_2"]) / 2
        aux["atom_count"] = aux["atom_count_1"] + aux["atom_count_2"]

        lr = lr_scheduler.get_last_lr()[0]
        aux["lr"] = lr
        lr_scheduler.step()
        
        return aux["lm_loss"], aux, 0.0


    def forward(self, input_ids):

        logits, _ = self.model(
            input_ids=input_ids,
            shift_states=True,
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
    