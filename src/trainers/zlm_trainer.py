import torch
import torch.nn.functional as F

import numpy as np

from trainers.base_trainer import BaseTrainer
from models.zlm import ZLMModel
from utils.scheduling_utils import linear_warmup, cosine_warmup
from utils.torch_utils import scale_gradient, unsqueeze_to_batch
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
                self.hook_step.fill_(self.config.trainer.hook_warmup_steps)

        # disable muon for parameters that shouldn't use it
        # (io embeddings are 1D)
        self.model.embed_tokens._orig_mod.weight.no_muon = True
        self.model.lm_head._orig_mod.weight.no_muon = True
        
        self.model.encoder_sep_token.no_muon = True
        self.model.encoder_z_tokens.no_muon = True

        self.model.decoder_z_tokens.no_muon = True
        self.model.decoder_start_output_token.no_muon = True

        self.model.uncond_tokens.no_muon = True


    def get_effective_parties(self, x):
        p = x / (x.sum() + self.model.config.rms_norm_eps)

        n = 1 / (p.pow(2).sum() + self.model.config.rms_norm_eps)

        return n / x.numel()


    def get_spectral_info(self, x):
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):

            x = x.transpose(0, 1) # [S, B, H]
            x = shard_with_gradients(x)

            x = x - x.mean(dim=1, keepdim=True)
            cov = torch.einsum(
                'sbi,sbj->sij',
                x, x
            ).float() / x.shape[1] # [S, H, H]

            v = torch.linalg.eigvalsh(
                cov + self.model.config.rms_norm_eps * torch.eye(x.shape[-1], device=x.device, dtype=cov.dtype)[None]
            ) # [S, H]

        spectral_reg = (v.pow(2)/2 - v.log() - 1/2).mean()
        spectral_parties = self.get_effective_parties(v)

        return spectral_reg.detach(), spectral_parties.detach()


    def kl_loss(
        self,
        mu: torch.FloatTensor,
        pred_mu: torch.FloatTensor,
    ):

        mu_kl_scale = {}
        scaled_mu = scale_gradient(mu, mu_kl_scale)
    
        kl = ((scaled_mu - pred_mu).pow(2) / 2).sum((0, -1)) # [S,]

        weights = kl
        weights = weights * ( # normalize so that mean(kl*weights) = mean(kl)
            kl.mean() / ((weights * kl).mean() + self.model.config.rms_norm_eps)
        )

        mu_kl_scale["value"] = weights[None, :, None]

        return kl.sum(), weights


    def forward(self, input_ids, output_ids):
        pad_token_id = self.model.config.pad_token_id

        # get the hook progress
        hook_progress = cosine_warmup(
            self.hook_step.float(),
            self.config.trainer.hook_warmup_steps
        )
        wait_hook_progress = cosine_warmup(
            self.hook_step.float() - self.config.trainer.hook_warmup_steps,
            self.config.trainer.hook_warmup_steps
        )
        double_wait_hook_progress = cosine_warmup(
            self.hook_step.float() - 2*self.config.trainer.hook_warmup_steps,
            self.config.trainer.hook_warmup_steps
        )

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
        noise_scale = hook_progress
        noise = self.model.sample_noise(input_for_model)
        z, mu = self.model.encode(
            input_for_model, output_for_model,
            input_mask=input_mask, output_mask=output_mask,
            noise=noise,
            noise_scale=noise_scale,
        )

        logit_grad_scale = {}
        logits, z_states = self.model.decode(
            input_for_model, output_for_model, z,
            logit_grad_scale=logit_grad_scale,
            input_mask=input_mask,
            output_mask=output_mask,
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
        mu_kl_grad_scale = double_wait_hook_progress
        mu_for_kl = scale_gradient(mu, mu_kl_grad_scale)
        z_for_kl = self.model.add_noise(mu_for_kl, noise)

        z_states_kl_grad_scale = wait_hook_progress
        z_states_for_kl = scale_gradient(z_states, z_states_kl_grad_scale)

        # get decoder predictions
        pred_mu = self.model.decoder_head(
            z_states_for_kl, z_for_kl
        )
        uncond_pred_mu = self.model.uncond_decoder_head(
            self.model.uncond_tokens[None], z_for_kl.detach()
        )

        # get kl
        kl, weights = self.kl_loss(
            mu_for_kl, pred_mu
        )
        uncond_kl, uncond_weights = self.kl_loss(
            mu_for_kl.detach(), uncond_pred_mu
        )
        mean_kl, mean_weights = self.kl_loss(
            mu_for_kl.detach(), mu_for_kl.detach().mean(0, keepdim=True)
        )

        denom = (output_ids != pad_token_id).float().sum() + self.model.config.rms_norm_eps
        latent_denom = mu.shape[0] * mu.shape[1]

        # calculate kls per token
        kl_per_token = kl / denom
        kl_per_latent = kl / latent_denom
        kl_parties = self.get_effective_parties(weights)
        elbo = lm_loss + kl_per_token

        uncond_kl_per_token = uncond_kl / denom
        uncond_kl_per_latent = uncond_kl / latent_denom
        uncond_kl_parties = self.get_effective_parties(uncond_weights)

        mean_kl_per_token = mean_kl / denom
        mean_kl_per_latent = mean_kl / latent_denom
        mean_kl_parties = self.get_effective_parties(mean_weights)

        # get the regularization loss
        regularize_scale = hook_progress
        spectral_reg, spectral_parties = self.get_spectral_info(mu)

        loss = (
            lm_loss +
            self.config.trainer.beta * kl_per_token +
            self.config.trainer.beta * uncond_kl_per_token +
            self.config.trainer.regularize_weight * regularize_scale * spectral_reg
        )

        aux = {
            "elbo": elbo,

            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "lm_loss_scale": lm_loss_scale,

            "mu_kl_grad_scale": mu_kl_grad_scale,
            "states_kl_grad_scale": z_states_kl_grad_scale,

            "kl_per_token": kl_per_token,
            "kl_per_latent": kl_per_latent,
            "kl_full_parties": kl_parties,

            "uncond_kl_per_token": uncond_kl_per_token,
            "uncond_kl_per_latent": uncond_kl_per_latent,
            "uncond_kl_parties": uncond_kl_parties,

            "mean_kl_per_token": mean_kl_per_token,
            "mean_kl_per_latent": mean_kl_per_latent,
            "mean_kl_parties": mean_kl_parties,
            
            "regularize_scale": regularize_scale,
            "regularize_loss": spectral_reg,
            "spectral_parties": spectral_parties,

            "hooked": self.hooked,
            "hook_step": self.hook_step,
            "hook_progress": hook_progress,
            "wait_hook_progress": wait_hook_progress,
            "double_wait_hook_progress": double_wait_hook_progress,

            "noise_scale": noise_scale,
            
            "atom_count": (output_ids != pad_token_id).long().sum(),
        }

        return loss, aux
