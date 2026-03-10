import torch
import torch.nn.functional as F
from torch.func import functional_call

import numpy as np

from trainers.base_trainer import BaseTrainer
from models.zlm import ZLMModel
from utils.scheduling_utils import linear_warmup, cosine_warmup
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

        sequence_weights = kl.mean(-1, keepdim=True)
        channel_weights = kl.mean(-2, keepdim=True)
        weights = sequence_weights * channel_weights

        weights = weights * kl.mean() / ((weights * kl).mean() + self.model.config.rms_norm_eps)
        weights = weights.detach()
        if disable_weights:
            weights = torch.ones_like(weights)

        mu_kl_scale["value"] = weights[None]

        return kl.sum(), sequence_weights, channel_weights


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

        contrast_scale = wait_hook_progress
        mu_for_contrast = mu.flip(0)
        z_for_contrast = z.flip(0)
        contrast_params = {
            name: p.detach() for name, p in self.model.decoder_head.named_parameters()
        }
        contrast_buffers = {
            name: b.detach() for name, b in self.model.decoder_head.named_buffers()
        }
        contrast_pred_mu = functional_call(
            self.model.decoder_head,
            (contrast_params, contrast_buffers),
            (z_states.detach(), z_for_contrast),
        )

        # get kl
        kl, sequence_weights, channel_weights = self.kl_loss(
            mu_for_kl, pred_mu
        )
        uncond_kl, uncond_sequence_weights, uncond_channel_weights = self.kl_loss(
            mu_for_kl.detach(), uncond_pred_mu
        )
        mean_kl, mean_sequence_weights, mean_channel_weights = self.kl_loss(
            mu_for_kl.detach(), mu_for_kl.detach().mean(0, keepdim=True)
        )
        contrast_kl, contrast_sequence_weights, contrast_channel_weights = self.kl_loss(
            mu_for_contrast, contrast_pred_mu, disable_weights=True
        )

        denom = (output_ids != pad_token_id).float().sum() + self.model.config.rms_norm_eps

        # calculate kls per token
        kl_per_token = kl / denom
        effective_parties = self.get_effective_parties(sequence_weights)
        channel_parties = self.get_effective_parties(channel_weights)
        elbo = lm_loss + kl_per_token

        uncond_kl_per_token = uncond_kl / denom
        uncond_effective_parties = self.get_effective_parties(uncond_sequence_weights)
        uncond_channel_parties = self.get_effective_parties(uncond_channel_weights)

        mean_kl_per_token = mean_kl / denom
        mean_effective_parties = self.get_effective_parties(mean_sequence_weights)
        mean_channel_parties = self.get_effective_parties(mean_channel_weights)
        
        contrast_kl_per_token = contrast_kl / denom
        contrast_effective_parties = self.get_effective_parties(contrast_sequence_weights)
        contrast_channel_parties = self.get_effective_parties(contrast_channel_weights)

        loss = (
            lm_loss +
            self.config.trainer.beta * kl_per_token +
            self.config.trainer.beta * uncond_kl_per_token +
            (-self.config.trainer.contrast_weight) * contrast_scale * contrast_kl_per_token
        )

        aux = {
            "elbo": elbo,

            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "lm_loss_scale": lm_loss_scale,

            "kl_grad_scale": mu_kl_grad_scale,
            "full_grad_scale": z_states_kl_grad_scale,

            "kl_per_token": kl_per_token,
            "effective_parties": effective_parties,
            "channel_parties": channel_parties,

            "uncond_kl_per_token": uncond_kl_per_token,
            "uncond_effective_parties": uncond_effective_parties,
            "uncond_channel_parties": uncond_channel_parties,

            "mean_kl_per_token": mean_kl_per_token,
            "mean_effective_parties": mean_effective_parties,
            "mean_channel_parties": mean_channel_parties,
            
            "contrast_scale": contrast_scale,
            "contrast_kl_per_token": contrast_kl_per_token,
            "contrast_effective_parties": contrast_effective_parties,
            "contrast_channel_parties": contrast_channel_parties,

            "hooked": self.hooked,
            "hook_step": self.hook_step,
            "hook_progress": hook_progress,
            "wait_hook_progress": wait_hook_progress,
            "noise_scale": noise_scale,
            
            "atom_count": (output_ids != pad_token_id).long().sum(),
        }

        return loss, aux
