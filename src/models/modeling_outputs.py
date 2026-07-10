"""Public model outputs used by the Transformers compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput


@dataclass
class ZLMCausalLMOutput(ModelOutput):
    """Causal language-model output with optional ZLM latent states."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    latent_states: torch.FloatTensor | None = None
    latent_means: torch.FloatTensor | None = None
