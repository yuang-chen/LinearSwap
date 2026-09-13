"""Hugging Face ``transformers`` integration: config, cache and ``PreTrainedModel`` wrapper.

    import linswap                                   # registers "linswap" with the Auto classes
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("hf/Qwen3.5-0.8B-KDA", dtype=torch.bfloat16).cuda()
    out = model.generate(**tok("Hello", return_tensors="pt").to("cuda"), max_new_tokens=32)

    # or convert in memory
    from linswap.hf import LinearSwapForCausalLM
    model = LinearSwapForCausalLM.from_swap("kda", base_model_dir="models/Qwen3.5-0.8B")
    model.save_pretrained("hf/Qwen3.5-0.8B-KDA")

Limitations: batches must be unpadded or right-padded (loss / logits); generation needs
equal-length prompts; greedy / sampling decoding only — the model is stateful (recurrent
state + KV cache), so beam search is unsupported.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .backbones import load_backbone_config
from .load_weights import DEFAULT_BASE_MODEL_DIR, build_model
from .model import LinearSwapBackbone, SwapCache
from .registry import get_kernel

_BACKBONE_KEYS = {  # config attribute -> backbone cfg key
    "vocab_size": "vocab_size", "max_position_embeddings": "context_length", "hidden_size": "emb_dim",
    "num_attention_heads": "n_heads", "num_hidden_layers": "n_layers", "intermediate_size": "hidden_dim",
    "head_dim": "head_dim", "num_key_value_heads": "n_kv_groups", "rope_theta": "rope_base",
    "partial_rotary_factor": "partial_rotary_factor", "rms_norm_eps": "rms_norm_eps",
    "linear_conv_kernel_dim": "linear_conv_kernel_dim", "linear_key_head_dim": "linear_key_head_dim",
    "linear_value_head_dim": "linear_value_head_dim", "linear_num_key_heads": "linear_num_key_heads",
    "linear_num_value_heads": "linear_num_value_heads", "layer_types": "layer_types",
}


class LinearSwapConfig(PretrainedConfig):
    """Backbone architecture (Qwen3-Next / Qwen3.5 hybrid layout) plus the linear-attention ``kernel``."""

    model_type = "linswap"

    def __init__(self, kernel="gdn", vocab_size=248_320, hidden_size=1024, num_hidden_layers=24,
                 num_attention_heads=8, num_key_value_heads=2, head_dim=256, intermediate_size=3584,
                 rms_norm_eps=1e-6, rope_theta=10_000_000.0, partial_rotary_factor=0.25,
                 max_position_embeddings=262_144, linear_conv_kernel_dim=4, linear_key_head_dim=128,
                 linear_value_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=16,
                 layer_types=None, base_model=None, sft_mode=None, tie_word_embeddings=True, **kwargs):
        self.kernel = kernel
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.max_position_embeddings = max_position_embeddings
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.layer_types = layer_types or (["linear_attention"] * 3 + ["full_attention"]) * (num_hidden_layers // 4)
        self.base_model = base_model      # informational: the pretrained backbone the swap started from
        self.sft_mode = sft_mode          # informational: gate_only / full / distill / None
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def to_backbone_cfg(self, dtype=torch.bfloat16) -> dict:
        cfg = {v: getattr(self, k) for k, v in _BACKBONE_KEYS.items()}
        cfg.update({"qk_norm": True, "dtype": dtype, "tie_word_embeddings": self.tie_word_embeddings})
        return cfg

    @classmethod
    def from_backbone_cfg(cls, cfg: dict, kernel: str, **kwargs) -> "LinearSwapConfig":
        attrs = {k: cfg[v] for k, v in _BACKBONE_KEYS.items()}
        return cls(kernel=kernel, tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)), **attrs, **kwargs)


class LinearSwapCache(Cache):
    """``transformers`` cache facade over the model's own ``SwapCache`` (per-layer KV + FLA recurrent states)."""

    is_compileable = False

    def __init__(self, n_layers: int):
        super().__init__(layers=[])
        self.swap = SwapCache(n_layers)
        self.seq_len = 0

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.seq_len

    def get_max_length(self, layer_idx: int | None = None):
        return None

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        q = cache_position.shape[0]
        return q + self.seq_len, self.seq_len

    def has_previous_state(self, layer_idx=None, state_idx=None) -> bool:
        return self.seq_len > 0

    def reset(self):
        self.swap.reset()
        self.seq_len = 0

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("LinearSwap models are stateful; beam search is not supported")

    def __len__(self):
        return len(self.swap._kv)


class LinearSwapPreTrainedModel(PreTrainedModel):
    config_class = LinearSwapConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _is_stateful = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        # Weights always come from a pretrained backbone or a checkpoint; nothing to initialise.
        return

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False  # the forward pass creates its own LinearSwapCache


class LinearSwapForCausalLM(LinearSwapPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LinearSwapConfig):
        super().__init__(config)
        dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        # Cast everything (including the fp32 A_log / dt_bias the FLA layers create) to one dtype,
        # exactly like linswap.build_model does; mixed dtypes make the Triton kernels misbehave.
        cfg = config.to_backbone_cfg(dtype)
        self.model = LinearSwapBackbone(cfg, get_kernel(config.kernel)).to(dtype)
        self.lm_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=dtype)
        self.post_init()

    # ---- embeddings / tying
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new):
        self.lm_head = new

    # ---- gradient checkpointing (delegates to the backbone's flag)
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        self.model.gradient_checkpointing = enable

    # ---- forward
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=None, labels=None,
                logits_to_keep=0, inputs_embeds=None, return_dict=None, **kwargs):
        if inputs_embeds is not None:
            raise ValueError("LinearSwapForCausalLM takes input_ids, not inputs_embeds")
        if attention_mask is not None and not bool(attention_mask.all()):
            # Right padding is exact for a causal model (padded positions never influence real ones);
            # left padding would need a mask inside attention / a recurrent-state reset.
            if not bool((attention_mask.cummin(dim=1).values == attention_mask).all()):
                raise ValueError("left padding is not supported: pad on the right (tokenizer.padding_side = 'right')")
            if past_key_values is not None or use_cache:
                raise ValueError("padded batches are supported for loss / logits only, not for cached generation")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = LinearSwapCache(len(self.model.layers))
        cache = past_key_values.swap if past_key_values is not None else None
        if past_key_values is not None:
            self.model.current_pos = past_key_values.seq_len
        hidden = self.model(input_ids, cache=cache, use_cache=cache is not None)
        if past_key_values is not None:
            past_key_values.seq_len += input_ids.shape[1]
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden[:, keep, :].to(self.lm_head.weight.dtype))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values if use_cache else None)

    # ---- conversion helpers
    @classmethod
    def from_swap(cls, kernel: str | None = None, base_model_dir=DEFAULT_BASE_MODEL_DIR, ckpt_dir=None,
                  device="cpu", dtype=torch.bfloat16) -> "LinearSwapForCausalLM":
        """Wrap a swapped model (base swap or checkpoint) built by ``linswap.build_model``."""
        swapped = build_model(kernel, base_model_dir=base_model_dir, ckpt_dir=ckpt_dir, device=device, dtype=dtype)
        sft_mode = None
        if ckpt_dir is not None and (Path(ckpt_dir) / "config.json").exists():
            import json
            sft_mode = json.load(open(Path(ckpt_dir) / "config.json")).get("sft_mode")
        config = LinearSwapConfig.from_backbone_cfg(swapped.cfg, swapped.kernel.name,
                                                    base_model=str(Path(base_model_dir).name), sft_mode=sft_mode)
        config.dtype = dtype
        model = cls(config)
        model.model = swapped.model
        model.lm_head = swapped.lm_head
        model.tie_weights()
        return model


def register_auto_classes():
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("linswap", LinearSwapConfig, exist_ok=True)
    AutoModelForCausalLM.register(LinearSwapConfig, LinearSwapForCausalLM, exist_ok=True)


def export(kernel, out_dir, base_model_dir=DEFAULT_BASE_MODEL_DIR, ckpt_dir=None, dtype=torch.bfloat16,
           save_tokenizer=True) -> Path:
    """Write a swapped model as an HF checkpoint (config + safetensors + tokenizer + model card)."""
    from transformers import AutoTokenizer

    model = LinearSwapForCausalLM.from_swap(kernel, base_model_dir=base_model_dir, ckpt_dir=ckpt_dir, dtype=dtype)
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    if save_tokenizer:
        AutoTokenizer.from_pretrained(base_model_dir).save_pretrained(out_dir)
    spec = get_kernel(model.config.kernel)
    (out_dir / "README.md").write_text(f"""---
library_name: transformers
base_model: {model.config.base_model}
license: apache-2.0
tags: [linswap, linear-attention, {model.config.kernel}]
---
# {out_dir.name}

`{model.config.base_model}` with its Gated-DeltaNet linear-attention layers swapped for **{model.config.kernel}**
({spec.description}); post-training: {model.config.sft_mode or 'none (base swap)'}.
Built with [LinearSwap](https://github.com/yuang-chen/LinearSwap).

```python
import linswap  # registers the architecture
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{out_dir.name}")
model = AutoModelForCausalLM.from_pretrained("{out_dir.name}", dtype="bfloat16").cuda()
print(tok.decode(model.generate(**tok("The capital of France is", return_tensors="pt").to("cuda"), max_new_tokens=16)[0]))
```
""")
    return out_dir
