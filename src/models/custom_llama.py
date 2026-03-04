# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

import torch
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers.cache_utils import Cache

from torchprime.layers.sequential import HomogeneousSequential
from torchprime.rope.rope import RopeScaling, llama3_rope_frequencies
from torchprime.torch_xla_models.attention import AttentionModule

from utils import constants
if constants.XLA_AVAILABLE:
    from torchprime.torch_xla_models import offloading

from models.llama import (
    LlamaRMSNorm, 
    LlamaRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    LlamaMLP,
    repeat_kv,
)

from utils.torch_utils import gaussian_init
from utils.attention_utils import AtttentionProbe
from utils.loss_utils import lm_loss_fn


logger = logging.get_logger(__name__)


class CustomLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DictConfig, layer_idx: int | None = None, is_causal: bool = True):
        super().__init__()
        self.config = config
        self.attention_block = AttentionModule(config, is_causal=is_causal)
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = is_causal

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )

        self.probe = AtttentionProbe(layer_idx)


    # @xp.trace_me("LlamaAttention")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        elementwise_pad_mask=None,
        past_key_values: Cache | None = None,
    ) -> torch.FloatTensor:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if (past_key_values is not None) and (not constants.XLA_AVAILABLE):
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # apply elementwise attention bias 
        if elementwise_pad_mask is not None:

            query_pad, key_pad = elementwise_pad_mask
            query_scale, query_offset = query_pad
            key_scale, key_offset = key_pad

            query_states = (
                query_states * query_scale[:, None].to(query_states.dtype)
                + query_offset[:, None].to(query_states.dtype)
            )
            key_states = (
                key_states * key_scale[:, None].to(key_states.dtype)
                + key_offset[:, None].to(key_states.dtype)
            )

        attn_output = self.attention_block(
            query_states,
            key_states,
            value_states,
            attention_mask,
            attention_probe=self.probe,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class CustomLlamaDecoderLayer(nn.Module):
    
    offload_name: str = "decoder_input"
    is_causal: bool = True

    
    def __init__(self, config: DictConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CustomLlamaAttention(config=config, layer_idx=layer_idx, is_causal=self.is_causal)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    # @xp.trace_me("LlamaDecoderLayer")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,    # necessary, but kept here for BC
        elementwise_pad_mask=None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """
        Args:
                hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                attention_mask (`torch.FloatTensor`, *optional*):
                        attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                        query_sequence_length, key_sequence_length)` if default attention is used.
        """
        # This gives the `hidden_states` tensor a name so that we can layer specify
        # to offload this tensor to host RAM to save memory. This is not a standard
        # torch API because there is no such feature in PyTorch. Instead, the name
        # becomes node metadata during FX graph capture.
        if constants.XLA_AVAILABLE:
            hidden_states = offloading.offload_name(hidden_states, self.offload_name)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            elementwise_pad_mask=elementwise_pad_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CustomLlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: DictConfig
    """

    layer_type = CustomLlamaDecoderLayer
    do_norm: bool = True

    def __init__(self, config: DictConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # `HomogeneousSequential` is similar to `nn.Sequential` but can be compiled with
        # `scan` described in https://pytorch.org/xla/release/r2.6/features/scan.html.
        self.layers = HomogeneousSequential(
            *[
                self.layer_type(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        rope_scaling = config.get("rope_scaling", None)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        if rope_scaling is not None:
            rope_scaling = RopeScaling(**rope_scaling)
        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim=head_dim, rope_theta=config.rope_theta, scaling=rope_scaling
        )

        self.init_elementwise_pad_mask(config)

    
    def init_elementwise_pad_mask(self, config):
        head_dim = config.hidden_size // config.num_attention_heads

        first_ind = (head_dim // 2) - 1
        sec_ind = -1

        query_scales = torch.ones(2, head_dim)
        query_scales[:, first_ind] = 0.0
        query_scales[:, sec_ind] = 0.0
        self.register_buffer("query_scales", query_scales, persistent=True)

        query_offsets = torch.zeros(2, head_dim)
        query_offsets[:, first_ind] = 0.5
        query_offsets[:, sec_ind] = 0.5
        self.register_buffer("query_offsets", query_offsets, persistent=True)

        key_scales = query_scales.clone()
        self.register_buffer("key_scales", key_scales, persistent=True)

        key_offsets = torch.zeros(2, head_dim)
        key_offsets[0, first_ind] = config.pad_attention_bias_value
        key_offsets[0, sec_ind] = config.pad_attention_bias_value
        self.register_buffer("key_offsets", key_offsets, persistent=True)


    def get_elementwise_pad_mask(self, elementwise_pad_mask: torch.Tensor):

        elementwise_pad_mask = elementwise_pad_mask.long()
        return (
            (
                F.embedding(elementwise_pad_mask, self.query_scales),
                F.embedding(elementwise_pad_mask, self.query_offsets),
            ),
            (
                F.embedding(elementwise_pad_mask, self.key_scales),
                F.embedding(elementwise_pad_mask, self.key_offsets)
            ),
        )

    
    def get_default_position_ids(
        self,
        seq_length: int,
        device: torch.device,
        elementwise_pad_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.LongTensor:
        if past_key_values is not None and elementwise_pad_mask is not None:
            raise NotImplementedError(
                "Passing both `past_key_values` and `elementwise_pad_mask` is not supported."
            )

        if elementwise_pad_mask is not None:
            mask = elementwise_pad_mask.long()
            return torch.cumsum(mask, dim=1) - 1

        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        
        if past_key_values is not None:
            position_ids = position_ids + past_key_values.get_seq_length(0)

        return position_ids


    # @xp.trace_me("LlamaModel")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None, # only used in non-kernel attention
        position_ids: torch.LongTensor | None = None,
        elementwise_pad_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        assert (input_ids is not None) ^ (inputs_embeds is not None), (
            "You have to specify either input_ids or inputs_embeds, but not both."
        )
        
        # convert input ids to embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # get shapes
        seq_length = inputs_embeds.shape[1]

        # get position ids
        # TODO(https://github.com/pytorch/xla/issues/8783): Pass position_ids as `long()` when `scan` can take non-differentiable inputs.
        if position_ids is None:
            position_ids = self.get_default_position_ids(
                seq_length, inputs_embeds.device,
                elementwise_pad_mask,
                past_key_values
            ).float()

        # Create a causal attention mask
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float("-inf"), device=inputs_embeds.device),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimension
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask[:, None, None, :]

        # currently cannot be None because scan needs differentiable inputs
        if constants.XLA_AVAILABLE:
            if past_key_values is None:
                past_key_values = position_ids.clone() # this is fine as a dummy value

        # convert the boolean pad mask to scale and offset masks
        if elementwise_pad_mask is None:
            elementwise_pad_mask = torch.ones_like(position_ids, dtype=torch.bool)
        elementwise_pad_mask = self.get_elementwise_pad_mask(elementwise_pad_mask)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # decoder layers
        hidden_states = inputs_embeds
        hidden_states = self.layers(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            elementwise_pad_mask=elementwise_pad_mask,
            past_key_values=past_key_values,
        )

        if self.do_norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class CustomLlamaForCausalLM(nn.Module):

    transformer_type = CustomLlamaModel


    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = self.transformer_type(config)
        self.model.do_norm = False

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    
    def _init_weights(self, module: nn.Module):
        """Initialize weights for Linear and Embedding layers.

        This method initializes the weights of Linear and Embedding layers
        using a normal distribution with mean 0 and standard deviation specified
        by `self.config.initializer_range`. Biases are initialized to zero.

        Args:
            module: The module whose weights need to be initialized.
        """
        if self.config.gaussian_init:
            return gaussian_init(module)

        std = self.config.initializer_range
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


    # @xp.trace_me("LlamaForCausalLM")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None, # only used in non-kernel attention
        shift_states: bool = False,
        elementwise_pad_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        return_states: bool = False,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            elementwise_pad_mask=elementwise_pad_mask,
            past_key_values=past_key_values,
        )

        lm_states = self.model.norm(hidden_states)
        if shift_states:
            # Shift the hidden states to the right for causal language modeling
            lm_states = lm_states[..., :-1, :].contiguous()

        logits = self.lm_head(lm_states)
        logits = logits.to(torch.float32)

        # logits = torch.nn.functional.log_softmax(logits, dim=-1)
        
        loss = None
        if labels is not None:
        
            loss = lm_loss_fn(
                logits,
                labels=labels,
                ignore_index=self.config.pad_token_id,
                shift_logits=(not shift_states),
            )

        if return_states:
            return logits, loss, hidden_states
        
        return logits, loss
    