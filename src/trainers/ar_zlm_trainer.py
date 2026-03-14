import torch
import torch.nn.functional as F
from torch.func import functional_call

import numpy as np

from trainers.base_trainer import BaseTrainer
from models.ar_zlm import ARZLMModel
from utils.scheduling_utils import linear_warmup, cosine_warmup
from utils.torch_utils import scale_gradient, unsqueeze_to_batch
from utils.loss_utils import lm_loss_fn, lm_acc_fn 
from utils.sharding_utils import shard_with_gradients


class ARZLMTrainer(BaseTrainer):
    
    model: ARZLMModel

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

        # disable muon for parameters that shouldn't use it
        self.model.embed_tokens._orig_mod.weight.no_muon = True
        self.model.lm_head._orig_mod.weight.no_muon = True
        
        self.model.uncond_tokens.no_muon = True

        self.model.encoder_sep_token.no_muon = True
        self.model.encoder_z_tokens.no_muon = True

        self.model.decoder_z_tokens.no_muon = True
        self.model.decoder_start_output_token.no_muon = True
        

    def get_effective_parties(self, x):
        p = x / (x.sum() + self.model.config.rms_norm_eps)

        n = 1 / (p.pow(2).sum() + self.model.config.rms_norm_eps)

        return n / x.numel()


    def kl_loss(
        self,
        mu: torch.FloatTensor,
        pred_mu: torch.FloatTensor,
        disable_weights=False,
    ):

        mu_kl_scale = {}
        scaled_mu = scale_gradient(mu, mu_kl_scale)
    
        kl = ((scaled_mu - pred_mu).pow(2) / 2).sum(0) # [S, Z]

        weights = kl
        sequence_weights = kl.mean(-1, keepdim=True)
        channel_weights = kl.mean(-2, keepdim=True)

        weights = weights * kl.mean() / ((weights * kl).mean() + self.model.config.rms_norm_eps)
        weights = weights.detach()
        if disable_weights:
            weights = torch.ones_like(weights)

        mu_kl_scale["value"] = weights[None]

        return kl.sum(), weights, sequence_weights, channel_weights


    def SIGReg(self, x):

        dev = dict(device=x.device, dtype=x.dtype)

        # [S, B, D]
        x = x.transpose(0, 1)
        x = shard_with_gradients(x)

        # [S, D, N]
        w = torch.randn(
            x.shape[0], x.shape[-1], self.config.trainer.num_slices,
            **dev
        )
        w = shard_with_gradients(w)

        w = w / (w.norm(dim=1, keepdim=True) + self.model.config.rms_norm_eps)
        
        # [S, B, N]
        z = x @ w

        # [T]
        t = torch.linspace(0.0, 4.0, 16, **dev)
        
        # theoretical CF for N(0, 1) and Gauss. window
        exp_f = torch.exp(-0.5 * t**2)

        # empirical CF [S, B, N, T]
        z_t = z[..., None] * unsqueeze_to_batch(t, z[..., None])

        # [S, N, T]
        real = torch.cos(z_t).mean(1)
        imag = torch.sin(z_t).mean(1)

        # weighted L2 distance [S, N, T]
        real_err = (real - unsqueeze_to_batch(exp_f, real))
        imag_err = (imag - 0.0)
        err = real_err**2 + imag_err**2

        # weight by theoretical CF
        # multiply by 2 since we only integrate over positive frequencies
        w_err = err * 2.0 * unsqueeze_to_batch(exp_f, err)

        # [S, N]
        out = torch.trapz(w_err, t , dim=-1) * x.shape[-2]

        return out.mean()


    def forward(self, input_ids, output_ids):
        pad_token_id = self.model.config.pad_token_id

        # get the hook progress
        hook_progress = cosine_warmup(
            self.hook_step.float(),
            self.config.trainer.hook_warmup_steps
        )
        wait_hook_progress = cosine_warmup(
            self.hook_step.float() - self.config.trainer.hook_wait_steps,
            self.config.trainer.hook_warmup_steps
        )

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
        mu_kl_grad_scale = wait_hook_progress
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
        kl, full_weights, sequence_weights, channel_weights = self.kl_loss(
            mu_for_kl, pred_mu
        )
        uncond_kl, uncond_full_weights, uncond_sequence_weights, uncond_channel_weights = self.kl_loss(
            mu_for_kl.detach(), uncond_pred_mu
        )
        mean_kl, mean_full_weights, mean_sequence_weights, mean_channel_weights = self.kl_loss(
            mu_for_kl.detach(), mu_for_kl.detach().mean(0, keepdim=True)
        )

        denom = (output_ids != pad_token_id).float().sum() + self.model.config.rms_norm_eps
        l_denom = mu.shape[0] * mu.shape[1]

        # calculate kls per token
        kl_per_token = kl / denom
        kl_per_latent = kl / l_denom
        full_parties = self.get_effective_parties(full_weights)
        effective_parties = self.get_effective_parties(sequence_weights)
        channel_parties = self.get_effective_parties(channel_weights)
        elbo = lm_loss + kl_per_token

        uncond_kl_per_token = uncond_kl / denom
        uncond_kl_per_latent = uncond_kl / l_denom
        uncond_full_parties = self.get_effective_parties(uncond_full_weights)
        uncond_effective_parties = self.get_effective_parties(uncond_sequence_weights)
        uncond_channel_parties = self.get_effective_parties(uncond_channel_weights)

        mean_kl_per_token = mean_kl / denom
        mean_kl_per_latent = mean_kl / l_denom
        mean_full_parties = self.get_effective_parties(mean_full_weights)
        mean_effective_parties = self.get_effective_parties(mean_sequence_weights)
        mean_channel_parties = self.get_effective_parties(mean_channel_weights)

        # get the regularization loss
        regularize_scale = hook_progress

        self.model.uncond_kl_ema.update(uncond_kl_per_latent.detach().reshape(1))
        reg_loss_gate = 1 - linear_warmup(
            self.model.uncond_kl_ema.retrieve() - self.config.trainer.lower_reg_threshold,
            self.config.trainer.upper_reg_threshold - self.config.trainer.lower_reg_threshold,
        )
        reg_loss_gate = self.config.trainer.min_reg_loss_gate + (1 - self.config.trainer.min_reg_loss_gate) * reg_loss_gate

        regularize_loss = self.SIGReg(
            z / torch.sqrt(self.model.mu_scale**2 + noise_scale.pow(2))
        )

        loss = (
            lm_loss +
            self.config.trainer.beta * kl_per_token +
            self.config.trainer.beta * uncond_kl_per_token +
            self.config.trainer.regularize_weight * regularize_scale * reg_loss_gate * regularize_loss
        )

        aux = {
            "elbo": elbo,

            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "lm_loss_scale": lm_loss_scale,

            "kl_grad_scale": mu_kl_grad_scale,
            "full_grad_scale": z_states_kl_grad_scale,

            "kl_per_token": kl_per_token,
            "kl_per_latent": kl_per_latent,
            "full_parties": full_parties,
            "effective_parties": effective_parties,
            "channel_parties": channel_parties,

            "uncond_kl_per_token": uncond_kl_per_token,
            "uncond_kl_per_latent": uncond_kl_per_latent,
            "uncond_full_parties": uncond_full_parties,
            "uncond_effective_parties": uncond_effective_parties,
            "uncond_channel_parties": uncond_channel_parties,

            "mean_kl_per_token": mean_kl_per_token,
            "mean_kl_per_latent": mean_kl_per_latent,
            "mean_full_parties": mean_full_parties,
            "mean_effective_parties": mean_effective_parties,
            "mean_channel_parties": mean_channel_parties,
            
            "regularize_scale": regularize_scale,
            "reg_loss_gate": reg_loss_gate,
            "regularize_loss": regularize_loss,

            "hooked": self.hooked,
            "hook_step": self.hook_step,
            "hook_progress": hook_progress,
            "wait_hook_progress": wait_hook_progress,
            "noise_scale": noise_scale,
            
            "atom_count": (output_ids != pad_token_id).long().sum(),
        }

        return loss, aux
