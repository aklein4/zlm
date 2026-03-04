import torch
import torch.nn.functional as F

import numpy as np

from trainers.base_trainer import BaseTrainer
from models.zlm import ZLMModel
from utils.scheduling_utils import linear_warmup
from utils.torch_utils import scale_gradient
from utils.loss_utils import lm_loss_fn, lm_acc_fn 
from utils.sharding_utils import shard_with_gradients


class ZLMTrainer(BaseTrainer):
    
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
                self.hook_step.fill_(self.config.trainer.init_hook_step)
            else:
                self.hook_step.fill_(
                    self.config.trainer.hook_warmup_steps +
                    self.config.trainer.hook_wait_steps
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


    def get_spectral_parties(self, x):

        x = x.transpose(0, 1) # [S, B, H]
        x = shard_with_gradients(x)

        x = x - x.mean(dim=1, keepdim=True)
        cov = torch.einsum(
            'sbi,sbj->sij',
            x, x
        ) / x.shape[1] # [S, H, H]

        v = torch.linalg.eigvalsh(
            cov + self.model.mu_out_norm.eps * torch.eye(x.shape[-1], device=x.device, dtype=cov.dtype)[None]
        ) # [S, H]

        p = v / (v.sum(-1, keepdim=True) + self.model.config.rms_norm_eps)
        n = 1 / (p.pow(2).sum(-1) + self.model.config.rms_norm_eps)
        v = n / x.shape[-1]

        return v.mean().detach()


    def forward(self, input_ids, output_ids):
        pad_token_id = self.model.config.pad_token_id

        # get the hook progress
        hook_progress = linear_warmup(
            self.hook_step.float(),
            self.config.trainer.hook_warmup_steps
        )
        wait_hook_progress = linear_warmup(
            self.hook_step.float() - self.config.trainer.hook_wait_steps,
            self.config.trainer.hook_warmup_steps
        )
        # double_wait_hook_progress = linear_warmup(
        #     self.hook_step.float() - (2 * self.config.trainer.hook_wait_steps),
        #     self.config.trainer.hook_warmup_steps
        # )

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

        # encode and decode
        noise_scale = hook_progress
        z, mu, min_eig_val = self.model.encode(
            input_for_model, output_for_model,
            input_mask=input_mask, output_mask=output_mask,
            noise_scale=noise_scale,
            return_extra=True,
        )

        logit_grad_scale = {}
        logits, z_states = self.model.decode(
            input_for_model, output_for_model, z,
            logit_grad_scale=logit_grad_scale,
            input_mask=input_mask,
            output_mask=output_mask,
            return_extra=True,
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

        # calculate logit grad scale
        self.model.lm_loss_ema.update(lm_loss.detach().reshape(1))
        lm_loss_scale = self.config.trainer.min_lm_loss_scale + (1 - self.config.trainer.min_lm_loss_scale) * linear_warmup(
            self.model.lm_loss_ema.retrieve() - self.config.trainer.lower_loss_threshold,
            self.config.trainer.upper_loss_threshold - self.config.trainer.lower_loss_threshold,
        )
        logit_grad_scale["value"] = lm_loss_scale

        # update hooking status
        self.hooked = self.hooked | (lm_loss < self.config.trainer.upper_loss_threshold).reshape(1)
        self.hook_step += self.hooked.long()

        # gradient scales
        mu_kl_grad_scale = wait_hook_progress
        z_states_kl_grad_scale = wait_hook_progress
        weighted_mu_kl_grad_scale = {}

        with torch.autocast("xla", enabled=False):
            
            # scaled gradients
            mu_for_kl = scale_gradient(mu, weighted_mu_kl_grad_scale)[None].repeat(
                self.config.trainer.num_diffusion_samples, 1, 1, 1
            ).float()
            z_states_for_kl = scale_gradient(z_states, z_states_kl_grad_scale)[None].float()

            # diffusion sampling
            t = torch.randint(
                low=1,
                high=self.model.config.num_diffusion_timesteps,
                size=mu_for_kl.shape[:-1],
                device=input_ids.device,
                dtype=torch.long,
            )
            noise = torch.randn_like(mu_for_kl)

            z_t = self.model.scheduler.add_noise(
                mu_for_kl, t, noise
            )
            pred_z_0 = self.model.diffusion_head(
                z_t, t, z_states_for_kl,
            )
            kls = self.model.scheduler.kl(
                mu_for_kl, t, pred_z_0, dim=-1
            ).mean(0)

            n_uncond = self.config.trainer.num_uncond_diffusion_samples
            uncond_pred_z_0 = self.model.uncond_diffusion_head(
                z_t[:n_uncond].detach(), t[:n_uncond], self.model.uncond_tokens[None, None, :, :],
            )
            uncond_kls = self.model.scheduler.kl(
                mu_for_kl[:n_uncond].detach(), t[:n_uncond], uncond_pred_z_0, dim=-1
            ).mean(0)

            # sum over batch to get [Z,]
            kl = kls.sum(0) * (self.model.config.num_diffusion_timesteps - 1)
            uncond_kl = uncond_kls.sum(0) * (self.model.config.num_diffusion_timesteps - 1)

            denom = (output_ids != pad_token_id).float().sum() + self.model.config.rms_norm_eps

            # set the weights for the kl grad scaling
            weighted_mu_kl_grad_scale["value"] = (
                mu_kl_grad_scale *
                self.get_kl_weights(kl)[None, None, :, None]
            )

        # calculate kls per token
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
            self.config.trainer.beta * kl_per_token +
            self.config.trainer.beta * uncond_kl_per_token
        )

        spectral_parties = 1.0 # self.get_spectral_parties(mu.detach())

        aux = {
            "elbo": elbo,

            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "lm_loss_scale": lm_loss_scale,

            "kl_grad_scale": mu_kl_grad_scale,
            "full_grad_scale": z_states_kl_grad_scale,

            "kl_per_token": kl_per_token,
            "effective_parties": effective_parties,
            "uncond_kl_per_token": uncond_kl_per_token,
            "uncond_effective_parties": uncond_effective_parties,
            "baseline_kl_per_token": baseline_kl_per_token,
            "baseline_effective_parties": baseline_effective_parties,
            "mean_kl_per_token": mean_kl_per_token,
            "mean_effective_parties": mean_effective_parties,
            
            "hooked": self.hooked,
            "hook_step": self.hook_step,
            "hook_progress": hook_progress,
            "wait_hook_progress": wait_hook_progress,
            "noise_scale": noise_scale,

            "min_eig_val": min_eig_val,
            "spectral_parties": spectral_parties,
            
            "atom_count": (output_ids != pad_token_id).long().sum(),
        }

        return loss, aux
