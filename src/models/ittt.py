import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
if constants.XLA_AVAILABLE:
    pass

import math
from omegaconf import DictConfig
from tqdm import tqdm

from models.llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, LlamaMLP
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import newton_schulz, cuda_newton_schulz
from utils.loss_utils import lm_loss_fn



class ItttFunction(torch.autograd.Function):

    @staticmethod
    # @torch.custom_fwd(device_type='xla', cast_inputs=torch.get_autocast_dtype('xla'))
    def forward(
        ctx,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        mod: "ItttLinear",
        momentum: torch.FloatTensor,
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x)
        ctx.mod = mod
        return z.clone()


    @staticmethod
    # @custom_bwd(device_type='xla')
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        x, = ctx.saved_tensors
        mod: ItttLinear = ctx.mod

        # [b, r, i]
        update = (
            grad.to(mod.momentum_dtype).transpose(-2, -1) @
            x.to(mod.momentum_dtype)
        )
    
        return None, grad, None, update

        
class ItttLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        config: DictConfig,
        rank: int | None = None,
    ):
        super().__init__()

        # save config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank if rank is not None else config.rank

        self.base_lr = config.base_lr
        self.momentum_beta = config.momentum_beta

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = getattr(torch, config.momentum_dtype)
        self.state_dtype = getattr(torch, config.state_dtype)

        # save linear
        self.linear = linear
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(self.rank, self.in_features)
        )
        self.base_state_proj = nn.Linear(
            self.in_features, self.rank, bias=False
        )
        self.out_proj = nn.Linear(
            self.rank, self.out_features, bias=False
        )

        # ephemeral state
        self.state: nn.Buffer
        self.momentum: nn.Buffer

        # weight initialization
        self.base_state_proj.weight.data.zero_()
        self.out_proj.weight.data.normal_(
            std=config.initializer_range
        )

        # self.svd_init()

        # for baselines
        self.disable_ittt = config.get("disable_ittt", False)

    
    # @torch.no_grad()
    # def svd_init(self):

    #     u, s, v = torch.linalg.svd(self.weight, full_matrices=False)

    #     self.out_proj.copy_(
    #         u[:, :self.rank] *
    #         s[None, :self.rank]
    #     )
    

    def get_lr(self):
        return (
            self.base_lr *
            math.sqrt(max(self.in_features, self.rank)) * math.sqrt(1/self.in_features) *
            torch.exp(self.log_lr * self.scalar_scaler)
        )


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.disable_ittt:
            return self.linear(x)

        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        lr = self.get_lr()
        s = lr[None] * self.state.detach()

        z = torch.einsum("boi,bsi->bso", s, x)
        z = ItttFunction.apply(x, z, self, self.momentum)

        z = z + self.base_state_proj(x)

        y_lora = self.out_proj(z)
        y_base = self.linear(x)

        y = y_base + y_lora

        return y


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        if self.disable_ittt:
            return

        state = torch.zeros(
            bs, self.rank, self.in_features,
            device=device, dtype=self.state_dtype,
        )
        momentum = torch.zeros_like(
            state, dtype=self.momentum_dtype
        )

        state = maybe_shard_with_gradients(state)
        momentum = maybe_shard_with_gradients(momentum)
    
        self.register_buffer("state", state, persistent=False)
        self.register_buffer("momentum", momentum, persistent=False)
        
        self.state.requires_grad_(False)

        self.momentum.requires_grad_(True)
        self.momentum.grad = torch.zeros_like(self.momentum)
        self.momentum.grad = maybe_shard_with_gradients(self.momentum.grad)


    @torch.no_grad()
    def empty_state(self):
        if self.disable_ittt:
            return

        self.state.zero_()

        self.momentum.zero_()
        self.momentum.grad.zero_()

    
    @torch.no_grad()
    def update_state(self):
        raise NotImplementedError("update_state should be called on the model, not the layer")
        if self.disable_ittt:
            return
        
        update = self.momentum.grad

        new_momentum = torch.lerp(
            self.momentum,
            update,
            1 - self.momentum_beta
        )

        whitened = newton_schulz(
            new_momentum, eps=self.eps
        )
        state_delta = -whitened.to(self.state_dtype)
        
        self.state.add_(state_delta.detach())
        
        self.momentum.copy_(new_momentum.detach())
        self.momentum.grad.zero_()


class ItttModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp.down_proj = ItttLinear(
                layer.mlp.down_proj,
                config
            )

        self.nepa_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.nepa_head = LlamaMLP(config)


    def forward(self, *args, **kwargs):
        return_states = kwargs.pop("return_states", False)

        logits, loss, hidden_states = super().forward(*args, **kwargs, return_states=True)

        nepa_target = F.rms_norm(hidden_states, [hidden_states.shape[-1]], eps=self.config.rms_norm_eps)
        nepa_target = nepa_target.sum(dim=1, keepdim=True) - nepa_target.cumsum(dim=1)
        nepa_target = F.rms_norm(nepa_target, [nepa_target.shape[-1]], eps=self.config.rms_norm_eps).detach()

        nepa_pred = self.nepa_head(self.nepa_norm(hidden_states))

        if return_states:
            return logits, loss, hidden_states, nepa_pred, nepa_target
    
        return logits, loss, nepa_pred, nepa_target


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.init_state(bs, device)


    @torch.no_grad()
    def empty_state(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.empty_state()
                

    @torch.no_grad()
    def update_state(self):
        
        try:
            ref: ItttLinear = self.model.layers[0].mlp.down_proj
        except:
            ref: ItttLinear = self.model.layers[0]._orig_mod.mlp.down_proj
        if ref.disable_ittt:
            return

        updates = []
        momentums = []
        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            try:
                m: ItttLinear = layer.mlp.down_proj
            except:
                m: ItttLinear = layer._orig_mod.mlp.down_proj

            updates.append(m.momentum.grad)
            momentums.append(m.momentum)
        
        updates = torch.stack(updates, dim=1)
        momentums = torch.stack(momentums, dim=1)

        updates = maybe_shard_with_gradients(updates)
        momentums = maybe_shard_with_gradients(momentums)

        new_momentums = torch.lerp(
            momentums,
            updates,
            1 - ref.momentum_beta
        )

        whitened = newton_schulz(
            momentums, eps=ref.eps
        )
        new_whitened = newton_schulz(
            new_momentums, eps=ref.eps
        )

        state_deltas = -(
            (new_whitened - whitened * ref.momentum_beta) /
            (1 - ref.momentum_beta)
        ).to(ref.state_dtype)

        for i, layer in enumerate(self.model.layers):
            layer: LlamaDecoderLayer

            try:
                m: ItttLinear = layer.mlp.down_proj
            except:
                m: ItttLinear = layer._orig_mod.mlp.down_proj

            m.state.add_(state_deltas[:, i].detach())
            m.momentum.copy_(new_momentums[:, i].detach())
            m.momentum.grad.zero_()


    def compute_logits(
        self,
        input_ids: torch.LongTensor,
        verbose: bool = False,
    ):
        
        chunks = torch.split(input_ids, self.config.chunk_size, dim=-1)

        ac_kwargs = {
            "device_type": str(input_ids.device),
            "dtype": torch.bfloat16,
        }

        self.init_state(input_ids.shape[0], input_ids.device)

        all_logits = []

        # first chunk
        with torch.autocast(**ac_kwargs):

            logits = self(
                chunks[0],
                logits_to_keep=slice(0, -1)
            )[0]
            all_logits.append(logits.detach().cpu())

            loss = lm_loss_fn(
                logits, chunks[0],
                shift_logits=False,
                ignore_index=self.config.pad_token_id,
            )

        loss.backward()

        # remaining chunks
        for i in tqdm(range(1, len(chunks)), desc="Processing Chunks", leave=False, disable=(not verbose)):
            
            first_chunk = chunks[i-1]
            second_chunk = chunks[i]
            all_chunk = torch.cat([first_chunk, second_chunk], dim=-1)

            self.update_state()

            with torch.autocast(**ac_kwargs):

                logits = self(
                    all_chunk,
                    logits_to_keep=slice(first_chunk.shape[-1]-1, -1)
                )[0]
                all_logits.append(logits.detach().cpu())

                loss = lm_loss_fn(
                    logits,
                    all_chunk[:, first_chunk.shape[-1]:],
                    shift_logits=False,
                    shift_labels=False,
                    ignore_index=self.config.pad_token_id,
                )

            loss.backward()

        self.zero_grad(True)
        self.empty_state()
            
        return torch.cat(all_logits, dim=1).detach()
