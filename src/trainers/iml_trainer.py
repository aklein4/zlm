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

                m.x_bias_buffer.requires_grad_(True)
                m.x_bias_buffer.grad = torch.zeros_like(m.x_bias_buffer)

                m.g_bias_buffer.requires_grad_(True)
                m.g_bias_buffer.grad = torch.zeros_like(m.g_bias_buffer)

                m.optimizer = self.optimizers['main']


    def forward(self, input_ids):

        first_, second_ = input_ids.chunk(2, dim=-1)

        coin = torch.rand(input_ids.shape[0], device=input_ids.device) < 0.5
        first = torch.where(coin, first_, second_)
        second = torch.where(coin, second_, first_)

        sorted_inputs = torch.cat([first, second], dim=0)

        logits, _ = self.model(
            input_ids=sorted_inputs,
            shift_states=True,
        )

        lm_loss = lm_loss_fn(
            logits,
            sorted_inputs,
            ignore_index=self.model.config.pad_token_id,
            shift_logits=False,
            shift_labels=True,
        )
        lm_acc = lm_acc_fn(
            logits,
            sorted_inputs,
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
        x_bias, g_bias = self.model.get_previous_biases()
        return {
            "iml_loss": self.model.get_previous_loss(),
            "iml_log_loss": self.model.get_previous_log_loss(),
            "x_bias": x_bias,
            "g_bias": g_bias,
        }
    