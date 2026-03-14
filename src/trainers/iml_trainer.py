import torch

import torch_xla

from models.llama import LlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn


class IMLTrainer(BaseTrainer):

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
        input_ids = batch["input_ids"]

        input_ids_a, input_ids_b = input_ids.chunk(2, dim=-1)
        coin = torch.rand(input_ids.shape[0], device=input_ids.device) < 0.5

        first_ids = torch.where(
            coin[:, None],
            input_ids_a,
            input_ids_b
        )
        second_ids = torch.where(
            coin[:, None],
            input_ids_b,
            input_ids_a
        )

        with torch.autocast('xla', dtype=torch.bfloat16, enabled=self.config.trainer.use_autocast):
            loss, aux = self.forward(first_ids)

        loss.backward()
        
        grad_norm = self.clip_gradients()

        key_name = lambda key, x: f"{key}_{x}" if len(self.optimizers) > 1 else x

        for key, optimizer in self.optimizers.items():

            opt_aux = optimizer.step()
            if opt_aux is not None:
                aux.update(
                    {key_name(key, k): v for k, v in opt_aux.items()}
                )

        for key, lr_scheduler in self.lr_schedulers.items():
            
            lr = lr_scheduler.get_last_lr()[0]
            aux.update({key_name(key, "lr"): lr})
            lr_scheduler.step()
        
        self.model.zero_grad(set_to_none=False)

        with torch.no_grad():
            with torch.autocast('xla', dtype=torch.bfloat16, enabled=self.config.trainer.use_autocast):
                
                _, first_aux = self.forward(first_ids)
                _, second_aux = self.forward(second_ids)

        aux.update({f"memorized_{k}": v for k, v in first_aux.items()})
        aux.update({f"extrapolated_{k}": v for k, v in second_aux.items()})

        return loss, aux, grad_norm


    def forward(self, input_ids):

        # this hack gets around some models not having a pad token id in their embeddings
        pad_token_id = self.model.config.pad_token_id
        inputs_for_model = torch.where(
            input_ids != pad_token_id,
            input_ids,
            torch.zeros_like(input_ids)
        )

        logits, _ = self.model(
            input_ids=inputs_for_model,
            shift_states=True,
        )

        lm_loss = lm_loss_fn(
            logits,
            input_ids,
            ignore_index=pad_token_id,
            shift_logits=False,
            shift_labels=True,
        )
        lm_acc = lm_acc_fn(
            logits,
            input_ids,
            ignore_index=pad_token_id,
            shift_logits=False,
            shift_labels=True,
        )

        loss = lm_loss

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "atom_count": (input_ids != pad_token_id).long().sum(), # number of tokens that the model sees this step
        }
    