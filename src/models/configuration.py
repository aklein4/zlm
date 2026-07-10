"""Transformers-compatible model configuration classes."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf
from transformers import PretrainedConfig


class TPULlamaConfig(PretrainedConfig):
    """Configuration for the XLA-optimized Llama implementation."""

    model_type = "tpu_llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_kernel: str | None = None,
        pad_attention_bias_value: float = -100.0,
        use_cache: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_kernel = attention_kernel
        self.pad_attention_bias_value = pad_attention_bias_value

        super().__init__(
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get(self, key: str, default=None):
        """Provide the small mapping API used by the existing model cores."""
        return getattr(self, key, default)


class CustomLlamaConfig(TPULlamaConfig):
    """Configuration for the elementwise-mask and cache-aware Llama model."""

    model_type = "custom_tpu_llama"

    def __init__(self, use_cache: bool = True, **kwargs):
        super().__init__(use_cache=use_cache, **kwargs)


class ZLMConfig(CustomLlamaConfig):
    """Configuration for the ZLM conditional generation model."""

    model_type = "zlm"

    def __init__(
        self,
        input_length: int = 256,
        output_length: int = 512,
        z_length: int = 384,
        latent_size: int = 64,
        z_ar_steps: int = 16,
        head_intermediate_size: int = 2560,
        use_z_norm_in: bool = True,
        lm_loss_ema_beta: float = 0.75,
        pretrained_llama: str | None = None,
        **kwargs,
    ):
        self.input_length = input_length
        self.output_length = output_length
        self.z_length = z_length
        self.latent_size = latent_size
        self.z_ar_steps = z_ar_steps
        self.head_intermediate_size = head_intermediate_size
        self.use_z_norm_in = use_z_norm_in
        self.lm_loss_ema_beta = lm_loss_ema_beta
        self.pretrained_llama = pretrained_llama

        super().__init__(**kwargs)


def coerce_config(config, config_class: type[PretrainedConfig]):
    """Convert a Hydra model config into its public Transformers config."""
    if isinstance(config, config_class):
        return config
    if isinstance(config, PretrainedConfig):
        return config_class(**config.to_dict())
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return config_class(**config)
    raise TypeError(
        f"Expected a mapping or PretrainedConfig, got {type(config).__name__}."
    )
