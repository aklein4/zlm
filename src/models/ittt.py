import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
if constants.XLA_AVAILABLE:
    pass

import math
from omegaconf import DictConfig
from tqdm import tqdm

from transformers.activations import ACT2FN

from models.llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import newton_schulz, cuda_newton_schulz
from utils.loss_utils import lm_loss_fn



class ItttFunction(torch.autograd.Function):

    @staticmethod
    # @torch.custom_fwd(device_type='xla', cast_inputs=torch.get_autocast_dtype('xla'))
    def forward(
        ctx,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        mod: "ItttLinear",
        momentum: torch.FloatTensor,
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x)
        ctx.mod = mod
        return y.clone()


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
        in_features: int,
        out_features: int,
        config: DictConfig,
    ):
        super().__init__()

        # save config
        self.in_features = in_features
        self.out_features = out_features

        self.base_lr = config.base_lr
        self.momentum_beta = config.momentum_beta

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = getattr(torch, config.momentum_dtype)
        self.state_dtype = getattr(torch, config.state_dtype)
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(self.out_features, self.in_features)
        )
        self.base_proj = nn.Linear(
            self.in_features, self.out_features, bias=False
        )

        # ephemeral state
        self.state: nn.Buffer
        self.momentum: nn.Buffer

        # weight initialization
        self.base_proj.weight.data.normal_(
            std=1/math.sqrt(self.in_features)
        )
    

    def get_lr(self):
        return (
            self.base_lr *
            math.sqrt(max(self.in_features, self.out_features)) * math.sqrt(1/self.in_features) *
            torch.exp(self.log_lr * self.scalar_scaler)
        )


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:

        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        s = self.get_lr()[None] * self.state.detach()

        y = torch.einsum("boi,bsi->bso", s, x)
        y = ItttFunction.apply(x, y, self, self.momentum)

        y = y + self.base_proj(x)

        return y


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):

        state = torch.zeros(
            bs, self.out_features, self.in_features,
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

        self.state.zero_()

        self.momentum.zero_()
        self.momentum.grad.zero_()

    
    @torch.no_grad()
    def update_state(self):
        
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


class ItttMLP(nn.Module):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.ittt_size = config.ittt_size
        
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.ittt_gate_proj = ItttLinear(self.hidden_size, self.ittt_size, config)
        self.ittt_up_proj = ItttLinear(self.hidden_size, self.ittt_size, config)
        self.ittt_down_proj = ItttLinear(self.ittt_size, self.hidden_size, config)


    def forward(self, x):
    
        y = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )
        y_itt = self.ittt_down_proj(
            self.act_fn(self.ittt_gate_proj(x)) * self.ittt_up_proj(x)
        )

        return y + y_itt


class ItttModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp = ItttMLP(config)


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
    def update_state_module(self, name: str):
        
        try:
            ref: ItttLinear = self.model.layers[0].get_submodule(name)
        except:
            ref: ItttLinear = self.model.layers[0]._orig_mod.get_submodule(name)

        updates = []
        momentums = []
        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            try:
                m: ItttLinear = layer.get_submodule(name)
            except:
                m: ItttLinear = layer._orig_mod.get_submodule(name)

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
            new_momentums, eps=ref.eps
        )

        state_deltas = -whitened.to(ref.state_dtype)

        for i, layer in enumerate(self.model.layers):
            layer: LlamaDecoderLayer

            try:
                m: ItttLinear = layer.get_submodule(name)
            except:
                m: ItttLinear = layer._orig_mod.get_submodule(name)

            m.state.add_(state_deltas[:, i].detach())
            m.momentum.copy_(new_momentums[:, i].detach())
            m.momentum.grad.zero_()


    @torch.no_grad()
    def update_state(self):
        for name in ["mlp.ittt_gate_proj", "mlp.ittt_up_proj", "mlp.ittt_down_proj"]:
            self.update_state_module(name)


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
