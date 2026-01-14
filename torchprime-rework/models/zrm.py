import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import BaseXLAModel
from models.llama import LlamaModel
from utils.torch_utils import (
    scale_gradient,
    expand_to_batch,
    unsqueeze_to_batch
)


class LoRaModulator(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        splits,
    ):
        super().__init__()

        self.base_linear = base_linear
        self.rank = rank
        self.splits = splits

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.num_splits = len(splits)
        self.total_rank = self.rank * self.num_splits

        self.lora_down = nn.Linear(
            self.in_features, self.total_rank, bias=False
        )
        self.lora_up = nn.Linear(
            self.total_rank, self.out_features, bias=False
        )

        # create the mask
        split_mask = torch.zeros(sum(splits), self.total_rank)

        row_start = 0
        col_start = 0
        for split_size in splits:

            split_mask[
                row_start:(row_start + split_size),
                col_start:(col_start + self.rank)
            ] = 1.0

            row_start += split_size
            col_start += self.rank

        self.register_buffer('split_mask', split_mask, persistent=False)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        inner = (
            self.lora_down(x) * 
            unsqueeze_to_batch(self.split_mask, x)
        )
        outer = self.lora_up(inner)

        # return the result
        return self.base_linear(x) + outer


class ZRModel(BaseXLAModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # transformer config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.lora_rank = config.lora_rank

        # length config
        self.input_length = config.input_length
        self.output_length = config.output_length
        self.z_length = config.z_length
        
        # transformers
        self.encoder = LlamaModel(config)
        self.generator = LlamaModel(config)
        self.decoder = LlamaModel(config)
        
        # add LoRa modulator to qkv and gate_up
        transformer_splits = [
            (
                self.encoder,
                [self.input_length, self.output_length, self.z_length]
            ),
            (
                self.generator,
                [self.input_length, self.z_length]
            ),
            (
                self.decoder,
                [self.input_length, self.z_length, self.output_length]
            )
        ]
        for transformer, splits in transformer_splits:
            for layer in transformer.layers:
                layer.self_attn.qkv_proj = LoRaModulator(
                    layer.self_attn.qkv_proj,
                    self.lora_rank,
                    splits
                )
                layer.mlp.gate_up_proj = LoRaModulator(
                    layer.mlp.gate_up_proj,
                    self.lora_rank,
                    splits
                )
        
        # LM components
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # input embeddings
        self.encoder_input_emb = nn.Parameter(
                torch.randn(1, self.hidden_size) * self.config.initializer_range
        )
        self.encoder_sep_emb = nn.Parameter(
                torch.randn(1, self.hidden_size) * self.config.initializer_range
        )
        self.encoder_output_emb = nn.Parameter(
                torch.randn(1, self.hidden_size) * self.config.initializer_range
        )
        self.encoder_z_tokens = nn.Parameter(
                torch.randn(self.z_length, self.hidden_size) * self.config.initializer_range
        )

        self.generator_input_emb = nn.Parameter(
                torch.randn(1, self.hidden_size) * self.config.initializer_range
        )
        self.generator_z_tokens = nn.Parameter(
                torch.randn(self.z_length, self.hidden_size) * self.config.initializer_range
        )

        self.decoder_input_emb = nn.Parameter(
                torch.randn(1, self.hidden_size) * self.config.initializer_range
        )
        self.decoder_z_tokens = nn.Parameter(
                torch.randn(self.z_length, self.hidden_size) * self.config.initializer_range
        )
        self.decoder_start_output_token = nn.Parameter(
                torch.randn(1, self.hidden_size) * self.config.initializer_range
        )
        self.decoder_output_emb = nn.Parameter(
                torch.randn(1, self.hidden_size) * self.config.initializer_range
        )

        # z/noise io components
        self.encoder_noise_proj_in = nn.Linear(
                self.z_size, self.hidden_size, bias=False
        )
        self.encoder_mu_proj_out = nn.Linear(
                self.hidden_size, self.z_size, bias=False
        )

        self.generator_z_proj_in = nn.Linear(
                self.z_size, self.hidden_size, bias=False
        )
        self.generator_mu_proj_out = nn.Linear(
                self.hidden_size, self.z_size, bias=False
        )

        self.decoder_z_proj_in = nn.Linear(
                self.z_size, self.hidden_size, bias=False
        )

        # scaling components
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

        # Initialize weights and apply final processing
        self.apply(self._init_weights)


    def forward(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        z_grad_scale: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        assert input_ids.shape[-1] == self.input_length
        assert output_ids.shape[-1] == self.output_length

        # get the real alpha value
        alpha = F.softplus(self.log_alpha) / np.log(2.0)

        # get reusable components
        input_tokens = self.embed_tokens(input_ids)
        output_tokens = self.embed_tokens(output_ids)

        input_mask = (input_ids != self.config.pad_token_id).float()
        output_mask = (output_ids != self.config.pad_token_id).float()

        input_bias = (input_ids == self.config.pad_token_id).float() * self.config.pad_bias
        output_bias = (output_ids == self.config.pad_token_id).float() * self.config.pad_bias

        # get the noise
        noise = torch.randn(
            input_tokens.shape[0], self.z_length, self.z_size,
            device=input_tokens.device,
            dtype=input_tokens.dtype,
        )

        # run the encoder
        encoder_mu_raw = self.encode(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_mask=input_mask,
            output_mask=output_mask,
            input_bias=input_bias,
            output_bias=output_bias,
            noise=noise,
        )
        encoder_mu_raw = F.rms_norm(
            encoder_mu,
            self.z_size,
            eps=self.config.rms_norm_eps
        )
        encoder_mu = encoder_mu_raw * alpha

        # run the generator
        generator_mu_raw = self.generate(
            input_tokens=input_tokens,
            input_mask=input_mask,
            input_bias=input_bias,
            z=(encoder_mu + noise)
        )
        generator_mu = generator_mu_raw * alpha

        # run the decoder
        decoder_z = scale_gradient(
            encoder_mu,
            z_grad_scale,
        ) + noise
        lm_logits = self.decode(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_mask=input_mask,
            output_mask=output_mask,
            input_bias=input_bias,
            output_bias=output_bias,
            z=decoder_z,
        )

        return {
            "lm_logits": lm_logits,
            "encoder_mu": encoder_mu,
            "generator_mu": generator_mu,
            "encoder_mu_raw": encoder_mu_raw,
            "generator_mu_raw": generator_mu_raw,
            "alpha": alpha,
        }
    

    def encode(
        self,
        input_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        output_mask: torch.Tensor,
        input_bias: torch.FloatTensor,
        output_bias: torch.FloatTensor,
        noise: torch.FloatTensor,
    ):
        
        # construct the encoder input
        input_states = (
            expand_to_batch(self.encoder_input_emb, input_tokens) +
            input_tokens
        )

        output_states = (
            expand_to_batch(self.encoder_output_emb, output_tokens) +
            torch.cat(
                [
                    output_tokens[:, :1] + expand_to_batch(self.encoder_sep_emb, output_tokens[:, :1]),
                    output_tokens[:, 1:],
                ],
                dim=-2
            )
        )

        z_states = torch.cat(
            [
                expand_to_batch(self.encoder_z_tokens[:, :1], input_tokens),
                (
                    expand_to_batch(self.encoder_z_tokens[:, 1:], input_tokens) +
                    self.encoder_noise_proj_in(noise[:, :-1])
                )
            ],
            dim=-2
        )

        encoder_states = torch.cat(
            [
                input_states,
                output_states,
                z_states,
            ],
            dim=-2
        )

        # create the position ids
        position_mask = torch.cat(
            [
                input_mask,
                output_mask,
                torch.ones_like(z_states[..., 0]),
            ],
            dim=-1
        )
        position_ids = position_mask.cumsum(dim=-1) - 1

        # create the bias
        attention_bias = torch.cat(
            [
                input_bias,
                output_bias,
                torch.zeros_like(z_states[..., 0]),
            ],
            dim=-1
        )

        # run the encoder
        encoder_states = self.encoder(
            input_embeds=encoder_states,
            position_ids=position_ids,
            elementwise_attention_bias=attention_bias
        )
        
        # get the mu values
        mu = self.encoder_mu_proj_out(
            encoder_states[:, -self.z_length:]
        )

        return mu


    def generate(
        self,
        input_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        input_bias: torch.FloatTensor,
        z: torch.FloatTensor,
    ):
        
        # construct the generator input
        input_states = (
            expand_to_batch(self.generator_input_emb, input_tokens) +
            input_tokens
        )

        z_states = torch.cat(
            [
                expand_to_batch(self.generator_z_tokens[:, :1], input_tokens),
                (
                    expand_to_batch(self.generator_z_tokens[:, 1:], input_tokens) +
                    self.generator_z_proj_in(z[:, :-1])
                )
            ],
            dim=-2
        )

        generator_states = torch.cat(
            [
                input_states,
                z_states,
            ],
            dim=-2
        )

        # create the position ids
        position_mask = torch.cat(
            [
                input_mask,
                torch.ones_like(z_states[..., 0]),
            ],
            dim=-1
        )
        position_ids = position_mask.cumsum(dim=-1) - 1

        # create the bias
        attention_bias = torch.cat(
            [
                input_bias,
                torch.zeros_like(z_states[..., 0]),
            ],
            dim=-1
        )

        # run the encoder
        generator_states = self.encoder(
            input_embeds=generator_states,
            position_ids=position_ids,
            elementwise_attention_bias=attention_bias
        )
        
        # get the mu values
        mu = self.generator_mu_proj_out(
            generator_states[:, -self.z_length:]
        )

        return mu

    
    def decode(
        self,
        input_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        output_mask: torch.Tensor,
        input_bias: torch.FloatTensor,
        output_bias: torch.FloatTensor,
        z: torch.FloatTensor,
    ):
        
        # construct the decoder input
        input_states = (
            expand_to_batch(self.decoder_input_emb, input_tokens) +
            input_tokens
        )

        z_states = (
            expand_to_batch(self.decoder_z_tokens, input_tokens) +
            self.decoder_z_proj_in(z)
        )

        output_states = (
            expand_to_batch(self.decoder_output_emb, output_tokens) +
            torch.cat(
                [
                    expand_to_batch(self.decoder_start_output_token, output_tokens[:, :-1]),
                    output_tokens[:, :-1],
                ],
                dim=-2
            )
        )

        decoder_states = torch.cat(
            [
                input_states,
                z_states,
                output_states,
            ],
            dim=-2
        )

        # create the position ids
        position_mask = torch.cat(
            [
                input_mask,
                torch.ones_like(z_states[..., 0]),
                torch.cat(
                    [
                        torch.ones_like(output_mask[..., :1]),
                        output_mask[..., :-1],
                    ],
                    dim=-1
                )
            ],
            dim=-1
        )
        position_ids = position_mask.cumsum(dim=-1) - 1

        # create the bias
        attention_bias = torch.cat(
            [
                input_bias,
                torch.zeros_like(z_states[..., 0]),
                torch.cat(
                    [
                        torch.zeros_like(output_bias[..., :1]),
                        output_bias[..., :-1],
                    ],
                    dim=-1
                )
            ],
            dim=-1
        )

        # run the encoder
        decoder_states = self.encoder(
            input_embeds=decoder_states,
            position_ids=position_ids,
            elementwise_attention_bias=attention_bias
        )
        
        # get the lm head logits
        lm_logits = self.lm_head(decoder_states[:, -self.output_length:])
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1)

        return lm_logits
    