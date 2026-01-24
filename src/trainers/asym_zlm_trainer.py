import torch
import torch.nn.functional as F

from torch_xla.experimental.scan import scan

import numpy as np

from trainers.base_trainer import BaseTrainer
from models.zlm import ZLMModel
from utils.scheduling_utils import linear_warmup
from utils.torch_utils import scale_gradient, safe_repeat
from utils.loss_utils import lm_loss_fn, lm_acc_fn
from utils.sharding_utils import shard_no_gradients, shard_with_gradients
from utils.logging_utils import visualize_sharding_info


class AsymZLMTrainer(BaseTrainer):
    
    model: ZLMModel

    hooked: torch.BoolTensor
    hook_step: torch.LongTensor


    def post_init(self):        
        
        self.hooked = torch.tensor(
            [self.config.trainer.init_hook],
            dtype=torch.bool, device=self.device
        ).reshape(1)

        self.hook_step = torch.zeros(
            1, dtype=torch.long, device=self.device
        )

        if self.config.trainer.init_hook:
            if self.config.trainer.init_hook_step is not None:
                self.hook_step.fill_(
                    self.config.trainer.init_hook_step
                )
            else:
                self.hook_step.fill_(
                    self.config.trainer.hook_warmup_steps
                )
    

    def get_kl_weights(self, kl):
        if kl.dim() > 1:
            kl = kl.mean(0)

        w = kl / kl.mean()
        w = torch.relu(w - self.config.trainer.kl_weight_relu_shift)

        w = w * (kl.sum() / ((w * kl).sum() + self.model.config.rms_norm_eps))

        return w.detach()


    def get_effective_parties(self, x):
        p = x / (x.sum() + self.model.config.rms_norm_eps)

        n = 1 / (p.pow(2).sum() + self.model.config.rms_norm_eps)

        return n / x.numel()



    def forward(self, input_ids, output_ids):
        pad_token_id = self.model.config.pad_token_id
        bs = input_ids.shape[0]

        # prepare gradient scales for the encoder
        enc_kl_grad_scale = self.config.trainer.beta * linear_warmup(
            self.hook_step.float(),
            self.config.trainer.hook_warmup_steps
        )
        enc_kl_head_grad_weights = {}
        enc_lm_grad_scale = {}

        # prepare inputs
        input_mask = (input_ids != pad_token_id)
        output_mask = (output_ids != pad_token_id)

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

        # encode
        z, mu = self.model.encode(
            input_for_model, output_for_model,
            input_mask=input_mask, output_mask=output_mask,
        )

        # we pass z through the decoder TWICE in this version
        # [:bs] for lm loss, [bs:] for kl loss
        input_for_model_2 = shard_no_gradients(
            safe_repeat(input_for_model, 2, dim=0)
        )
        output_for_model_2 = shard_no_gradients(
            safe_repeat(output_for_model, 2, dim=0)
        )
        input_mask_2 = shard_no_gradients(
            safe_repeat(input_mask, 2, dim=0)
        )
        output_mask_2 = shard_no_gradients(
            safe_repeat(output_mask, 2, dim=0)
        )
        z_2 = shard_with_gradients(
            torch.cat(
                [
                    scale_gradient(z, enc_lm_grad_scale),
                    scale_gradient(z, enc_kl_grad_scale)
                ],
                dim=0
            )
        )
        visualize_sharding_info(z_2)
        logits, z_states = self.model.decode(
            input_for_model_2, output_for_model_2, z_2,
            input_mask=input_mask_2,
            output_mask=output_mask_2,
        )
        logits = shard_with_gradients(
            logits[:bs]
        )
        z_states = shard_with_gradients(
            z_states[bs:]
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

        # update the lm grad scale with hindsight
        enc_lm_grad_scale["value"] = self.config.trainer.min_lm_loss_scale + (1 - self.config.trainer.min_lm_loss_scale) * linear_warmup(
            lm_loss - self.config.trainer.lower_loss_threshold,
            self.config.trainer.upper_loss_threshold - self.config.trainer.lower_loss_threshold,
        ).reshape(1)

        # update hooking status
        self.hooked = self.hooked | (lm_loss < self.config.trainer.upper_loss_threshold).reshape(1)
        self.hook_step += self.hooked.long()

        # get the kls by diffusion sampling
        kls = 0.0
        uncond_kls = 0.0
        for i in range(self.config.trainer.num_diffusion_samples):

            # sample z_t
            t = torch.randint(
                low=1,
                high=self.model.config.num_diffusion_timesteps,
                size=z.shape[:-1],
                device=input_ids.device,
                dtype=torch.long,
            )
            noise = torch.randn_like(z)

            z_t = self.model.scheduler.add_noise(
                mu, t, noise
            )

            # conditional kl
            pred_z_0 = self.model.diffusion_head(
                scale_gradient(z_t, enc_kl_head_grad_weights),
                t,
                z_states,
            )
            kl = self.model.scheduler.kl(
                scale_gradient(mu, enc_kl_head_grad_weights),
                t,
                pred_z_0,
                dim=-1
            )
            kls = kls + kl

            # unconditional kl
            if i < self.config.trainer.num_uncond_diffusion_samples:
                uncond_pred_z_0 = self.model.uncond_diffusion_head(
                    z_t.detach(),
                    t,
                    self.model.uncond_tokens[None, :, :],
                )
                uncond_kl = self.model.scheduler.kl(
                    mu.detach(),
                    t,
                    uncond_pred_z_0,
                    dim=-1
                )
                uncond_kls = uncond_kls + uncond_kl

        # mean over samples and sum over batch to get [Z,]
        kl = kls.sum(0) * (self.model.config.num_diffusion_timesteps - 1) / self.config.trainer.num_diffusion_samples
        uncond_kl = uncond_kls.sum(0) * (self.model.config.num_diffusion_timesteps - 1) / self.config.trainer.num_uncond_diffusion_samples

        # update the head kl grad weights with hindsight
        head_kl_weights = enc_kl_grad_scale * self.get_kl_weights(kl)
        enc_kl_head_grad_weights["value"] = head_kl_weights[None, :, None]

        # calculate kls per token
        denom = (output_ids != pad_token_id).float().sum() + self.model.config.rms_norm_eps

        kl_per_token = kl.sum() / denom
        effective_parties = self.get_effective_parties(kl)
        elbo = lm_loss + kl_per_token

        uncond_kl_per_token = uncond_kl.sum() / denom
        uncond_effective_parties = self.get_effective_parties(uncond_kl)

        baseline_kl = 0.5 * (mu * self.model.scheduler.a[0]).pow(2) / self.model.scheduler.b[0].pow(2)
        baseline_kl_per_token = baseline_kl.sum() / denom
        baseline_effective_parties = self.get_effective_parties(baseline_kl.sum(-1).sum(0))

        mean_kl = 0.5 * ((mu - mu.mean(0, keepdim=True)) * self.model.scheduler.a[0]).pow(2) / self.model.scheduler.b[0].pow(2)
        mean_kl_per_token = mean_kl.sum() / denom
        mean_effective_parties = self.get_effective_parties(mean_kl.sum(-1).sum(0))

        loss = (
            lm_loss +
            kl_per_token +
            uncond_kl_per_token
        )

        aux = {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "kl_grad_scale": enc_kl_grad_scale,
            "lm_loss_scale": enc_lm_grad_scale["value"],
            "kl_per_token": kl_per_token,
            "effective_parties": effective_parties,
            "uncond_kl_per_token": uncond_kl_per_token,
            "uncond_effective_parties": uncond_effective_parties,
            "baseline_kl_per_token": baseline_kl_per_token,
            "baseline_effective_parties": baseline_effective_parties,
            "mean_kl_per_token": mean_kl_per_token,
            "mean_effective_parties": mean_effective_parties,
            "elbo": elbo,
            "hooked": self.hooked,
            "hook_step": self.hook_step,
            "atom_count": (output_ids != pad_token_id).long().sum(),
        }

        return loss, aux
    

    # def scan_kl(self):

    #     def scan_fn(carry, t_curr):
    #         t_curr = t_curr.long()

    #         noise = torch.randn_like(z)

    #         z_t = self.model.scheduler.add_noise(
    #             mu, t_curr, noise
    #         )

    #         pred_z_0 = self.model.diffusion_head(
    #             scale_gradient(z_t, kl_grad_weights),
    #             t_curr,
    #             scale_gradient(z_states, full_grad_scale),
    #             self.model.scheduler,
    #         )
    #         kl = self.model.scheduler.kl(
    #             scale_gradient(mu, kl_grad_weights),
    #             t_curr,
    #             pred_z_0,
    #             dim=-1
    #         )

    #         uncond_pred_z_0 = self.model.uncond_diffusion_head(
    #             z_t.detach(),
    #             t_curr,
    #             self.model.uncond_tokens[None, :, :],
    #             self.model.scheduler,
    #         )
    #         uncond_kl = self.model.scheduler.kl(
    #             mu.detach(),
    #             t,
    #             uncond_pred_z_0,
    #             dim=-1
    #         )

    #         return carry, torch.stack([kl, uncond_kl], dim=-1)

    #     t = torch.randint(
    #         low=1,
    #         high=self.model.config.num_diffusion_timesteps,
    #         size=(self.config.trainer.num_diffusion_samples, *z.shape[:-1]),
    #         device=input_ids.device,
    #         dtype=torch.long,
    #     )

    #     _, kl_uncond_kl = scan(
    #         scan_fn,
    #         t.clone().float(),
    #         t.float(),
    #     )