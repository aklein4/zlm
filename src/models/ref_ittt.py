import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers.modeling_utils import PreTrainedModel

from models.ittt.configuration_ittt import ItttConfig
from models.reference_llama.modelling_llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.torch_utils import simple_rms_norm
from utils.training_utils import lm_loss
from tqdm import tqdm


@torch.compile(fullgraph=True, mode="reduce-overhead")
def newtonschulz(
    x: torch.FloatTensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.FloatTensor:
    # https://kellerjordan.github.io/posts/muon/
    
    assert x.ndim == 3, "x must be a 2D and batch"
    bs = x.shape[0]

    a, b, c = (3.4445, -4.7750, 2.0315)

    y = x / (
        x.reshape(bs, -1).norm(dim=-1)[:, None, None] + eps
    )

    if x.shape[-2] > x.shape[-1]:
        y = y.transpose(-2, -1)

    for _ in range(steps):
        m = y @ y.transpose(-2, -1)
        n = b * m + c * m @ m
        y = a * y + n @ y

    if x.shape[-2] > x.shape[-1]:
        y = y.transpose(-2, -1)

    return y


class ItttFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, z, mod):
        ctx.save_for_backward(x)
        ctx.mod = mod
        return z.clone()
    

    @staticmethod
    def backward(ctx, grad):
        og_grad = grad.clone()

        x, = ctx.saved_tensors
        mod = ctx.mod

        x = x.float()
        g = grad.float()

        x = simple_rms_norm(x, eps=mod.eps) # [b, s, i]
        g = F.normalize(g, dim=-2, eps=mod.eps) * math.sqrt(x.shape[-2])  # [b, s, r]

        x = x.to(mod.momentum_dtype)
        g = g.to(mod.momentum_dtype)

        # [b, r, i]
        mod.update = (
            g.transpose(-2, -1) @ x
        ) / math.sqrt(x.shape[-2]) # approx 1 std

        if mod.momentum is None:
            mod.momentum = torch.zeros_like(mod.update)
        
        mod.momentum = torch.lerp(
            mod.momentum,
            mod.update,
            1 - mod.momentum_beta
        )

        return None, og_grad, None

        
class ItttLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        base_lr: float,
        momentum_beta: float,
        eps: float = 1e-7,
        momentum_dtype=torch.bfloat16,
        state_dtype=torch.float32,
    ):
        super().__init__()

        # save config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank

        self.base_lr = base_lr
        self.momentum_beta = momentum_beta

        self.eps = eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = momentum_dtype
        self.state_dtype = state_dtype

        # save linear
        self.weight = linear.weight
        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )
        self.out_proj = nn.Parameter(
            torch.randn(self.out_features, rank) / math.sqrt(self.rank)
        )

        # ephemeral state
        self.state = None
        self.momentum = None
        self.update = None

        self.svd_init()

    
    @torch.no_grad()
    def svd_init(self):

        u, s, v = torch.linalg.svd(self.weight, full_matrices=False)

        self.out_proj.copy_(
            u[:, :self.rank] *
            s[None, :self.rank]
        )
    

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        if self.state is not None:

            lr = (
                self.base_lr *
                torch.exp(self.log_lr * self.scalar_scaler)
            )
            s = lr[None] * self.state

        else:
            s = torch.zeros_like(self.log_lr)[None]

        z = torch.einsum("boi,bji->bjo", s, x)
        z = ItttFunction.apply(x, z, self)

        y_lora = F.linear(z, self.out_proj)
        y_base = F.linear(x, self.weight, self.bias)

        y = y_base + y_lora

        return y

    
    @torch.no_grad()
    def reset_state(self):
        self.state = None
        self.momentum = None

    
    @torch.no_grad()
    def update_state(self):
        if self.momentum is None:
            print("WARNING: Momentum is None, skipping update.")
            return
        
        # nesterov-style momentum update
        delta = torch.lerp(
            self.update,
            self.momentum,
            self.momentum_beta
        )

        # we don't worry about adam-like biased momentum because newton-schulz normalizes anyway
        delta = -newtonschulz(
            delta,
            eps=self.eps
        ).to(self.state_dtype)

        if self.state is None:
            self.state = delta
        else:
            self.state += delta

        self.update = None


class ItttModel(PreTrainedModel):

    config: ItttConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True


    def __init__(self, config):
        super().__init__(config)

        self.llama = LlamaForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float32,
            attn_implementation=config._attn_implementation
        )

        self.start_layer = config.start_layer
        self.rank = config.rank
        self.eps = self.llama.config.rms_norm_eps

        for layer in self.llama.model.layers[self.start_layer:]:
            layer: LlamaDecoderLayer

            layer.self_attn.q_proj = ItttLinear(
                layer.self_attn.q_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                eps=self.eps
            )
            layer.self_attn.k_proj = ItttLinear(
                layer.self_attn.k_proj,
                rank=min(config.rank, layer.self_attn.k_proj.in_features, layer.self_attn.k_proj.out_features),
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                eps=self.eps
            )
            layer.self_attn.v_proj = ItttLinear(
                layer.self_attn.v_proj,
                rank=min(config.rank, layer.self_attn.v_proj.in_features, layer.self_attn.v_proj.out_features),
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                eps=self.eps
            )
            layer.self_attn.o_proj = ItttLinear(
                layer.self_attn.o_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                eps=self.eps
            )
            layer.mlp.down_proj = ItttLinear(
                layer.mlp.down_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                eps=self.eps
            )

        self.post_init()

    
    def _init_weights(self, module):
        # We don't want to re-initialize the weights, so we override this method to do nothing.
        return


    @torch.no_grad()
    def reset_state(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.reset_state()
                

    @torch.no_grad()
    def update_state(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.update_state()
                

    def forward(self, *args, **kwargs):
        return self.llama(*args, **kwargs)
    

    def compute_logits(
        self,
        input_ids: torch.LongTensor,
        chunk_size: int,
        labels = None,
        ignore_index: int = -100,
        verbose: bool = False,
        do_update: bool = True,
    ):
        
        if labels is None:
            labels = input_ids
        inputs_for_model = torch.where(
            input_ids == ignore_index,
            torch.zeros_like(input_ids),
            input_ids,
        )

        input_chunks = torch.split(inputs_for_model, chunk_size, dim=-1)
        label_chunks = torch.split(labels, chunk_size, dim=-1)

        ac_kwargs = {
            "device_type": str(input_ids.device),
            "dtype": torch.bfloat16,
        }

        self.reset_state()

        all_logits = []

        # first chunk
        with torch.autocast(**ac_kwargs):

            logits = self(
                input_chunks[0],
                logits_to_keep=slice(0, -1)
            ).logits
            all_logits.append(logits.detach().to(torch.bfloat16))

            loss = lm_loss(
                label_chunks[0].to(logits.device), logits,
                shift_logits=False,
                ignore_index=ignore_index,
            )

        loss.backward()

        # remaining chunks
        for i in tqdm(range(1, len(input_chunks)), desc="Processing Chunks", leave=False, disable=(not verbose)):
            
            first_chunk = input_chunks[i-1]
            second_chunk = input_chunks[i]
            all_chunk = torch.cat([first_chunk, second_chunk], dim=-1)

            first_labels = label_chunks[i-1]
            second_labels = label_chunks[i]
            all_labels = torch.cat([first_labels, second_labels], dim=-1)

            if do_update:
                self.update_state()

            with torch.autocast(**ac_kwargs):

                logits = self(
                    all_chunk,
                    logits_to_keep=slice(first_chunk.shape[-1]-1, -1)
                ).logits
                all_logits.append(logits.detach().to(torch.bfloat16))

                loss = lm_loss(
                    all_labels[:, first_labels.shape[-1]-1:].to(logits.device),
                    logits,
                    shift_logits=False,
                    ignore_index=ignore_index,
                )

            loss.backward()

        self.zero_grad(True)
        self.reset_state()
            
        return torch.cat(all_logits, dim=1).detach()
