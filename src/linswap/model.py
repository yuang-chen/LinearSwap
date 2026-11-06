"""The swapped model: ``transformers``' own Qwen3.5 vision-language model with nothing but each decoder
layer's ``linear_attn`` replaced by a kernel's layer.

    from linswap import build_model
    model = build_model("native")                           # the unmodified backbone
    model = build_model("gdn")                              # the teacher: Gated DeltaNet on FLA's kernels
    model = build_model("kda")                              # a swap at step 0, initialised from the backbone
    model = build_model("outputs/distill/gla")              # a distilled checkpoint
    model.generate(**processor.apply_chat_template(messages_with_images, ..., return_tensors="pt").to("cuda"))

Decoder, full attention, RoPE (multimodal positions included), vision tower, merger, processor contract
and ``generate`` stay upstream code and upstream weights, so a swapped model takes text, images and video
exactly as the backbone does and every standard evaluation harness can drive it.

The backbone is any checkpoint of that model class: the Qwen3.5 family and Qwen3.8-27B (``base_model_dir``).

A checkpoint is the *difference* to the backbone: ``swap.safetensors`` holds the swapped layers (and any
other tensor that was trained), ``linswap.json`` names the kernel and the backbone.  Everything else is read from the
backbone, which distillation never changes.

Limits: the kernel's recurrent / convolution state lives in an FLA cache attached to the upstream cache
object, which upstream's ``reorder_cache`` / ``crop`` do not see, so decoding is greedy or sampling (no beam
search).  Padding follows upstream (hidden states of pad positions are zeroed before the mixer), which is
exact for left padding with every kernel that writes nothing for a zero input — not ``swa``, which is run
one prompt at a time.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch

from .registry import get_kernel, list_kernels

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL_DIR = REPO_ROOT / "models" / "Qwen3.5-0.8B"
NATIVE = "native"                      # the unswapped upstream model
FLA_CACHE_ATTR = "_linswap_fla_cache"


# ------------------------------------------------------------------------------ the swap
def _mixer_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """``Qwen3_5DecoderLayer``'s call of its linear mixer -> the FLA layer interface the kernels implement.

    ``cache_params`` is the upstream cache of the whole model (or None); the kernel's own state lives in an
    FLA cache attached to it, so it is created, carried and dropped with that request's cache."""
    mask = {}
    if attention_mask is not None and not attention_mask.all():      # 2D padding mask of the current chunk
        if getattr(self, "takes_padding_mask", False):               # the layer packs the valid tokens itself
            mask = {"attention_mask": attention_mask}
        else:                                                        # as upstream's own layer: zero the pad positions
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    fla_cache = None
    if cache_params is not None:
        fla_cache = getattr(cache_params, FLA_CACHE_ATTR, None)
        if fla_cache is None:
            from fla.models.utils import Cache as FLACache

            fla_cache = FLACache()
            setattr(cache_params, FLA_CACHE_ATTR, fla_cache)
    out = self._linswap_forward(hidden_states, past_key_values=fla_cache, use_cache=fla_cache is not None, **mask)[0]
    return out.to(hidden_states.dtype)        # float32 master weights (distill) must not widen the residual stream


def kernel_config(text_config) -> dict:
    """The backbone dimensions a kernel's ``build`` reads, from the upstream text config."""
    rope = getattr(text_config, "rope_parameters", None) or {}
    return {"emb_dim": text_config.hidden_size, "rms_norm_eps": text_config.rms_norm_eps,
            "linear_conv_kernel_dim": text_config.linear_conv_kernel_dim,
            "linear_key_head_dim": text_config.linear_key_head_dim,
            "linear_value_head_dim": text_config.linear_value_head_dim,
            "linear_num_key_heads": text_config.linear_num_key_heads,
            "linear_num_value_heads": text_config.linear_num_value_heads,
            "rope_base": float(rope.get("rope_theta", 10_000_000.0)), "dtype": torch.bfloat16}


def text_model(model):
    """The upstream ``Qwen3_5TextModel`` inside a conditional-generation model."""
    return model.model.language_model


def mixers(model) -> dict:
    """``{layer index: linear mixer}`` of the swappable layers."""
    return {i: layer.linear_attn for i, layer in enumerate(text_model(model).layers) if hasattr(layer, "linear_attn")}


def swap_mixers(model, kernel: str):
    """Replace every ``linear_attn`` in place by ``kernel``'s layer, initialised from the Gated-DeltaNet
    layer it replaces (function preserving when the kernel's ``exact_init`` says so)."""
    spec = get_kernel(kernel)
    text = text_model(model)
    cfg = kernel_config(text.config)
    for idx, old in mixers(model).items():
        ref = old.out_proj.weight
        source = {f"m.layers.{idx}.linear_attn.{k}": v for k, v in old.state_dict().items()}
        mixer = spec.init_from_gdn(spec.build(cfg, idx), source, idx, "m").to(device=ref.device, dtype=ref.dtype)
        mixer._linswap_forward = mixer.forward
        mixer.forward = types.MethodType(_mixer_forward, mixer)      # parameters and their names are untouched
        text.layers[idx].linear_attn = mixer
    model.linswap_kernel = spec.name
    return model


def fast_patch_embed(model):
    """Compute the vision tower's patch embedding as a matrix product instead of ``torch.conv3d``.

    The convolution's kernel and stride equal its input (one output per patch), so it *is*
    ``weight.flatten(1) @ patch + bias``; in bf16 ``conv3d`` takes ~15 ms per visual token (16 s for a
    1024 x 1024 image) against 0.05 s for the product, and the two agree to bf16 rounding.  Applied to every
    model alike, the unswapped one included."""
    def forward(self, hidden_states):
        proj = self.proj
        patches = hidden_states.reshape(-1, proj.weight[0].numel()).float()
        bias = None if proj.bias is None else proj.bias.float()
        return torch.nn.functional.linear(patches, proj.weight.flatten(1).float(), bias).to(proj.weight.dtype)

    for module in model.modules():
        if type(module).__name__.endswith("VisionPatchEmbed") and isinstance(getattr(module, "proj", None), torch.nn.Conv3d):
            module.forward = types.MethodType(forward, module)
    return model


# -------------------------------------------------------------------------- checkpoints
def save_checkpoint(model, out_dir, trained=None, **meta) -> Path:
    """Write the swapped layers — and the tensors named in ``trained``, if more was trained — in bf16, the
    precision the model runs in, with the kernel name and ``meta`` in ``linswap.json``."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(trained or ())
    state = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in model.state_dict().items()
             if ".linear_attn." in k or k in keep}
    save_file(state, str(out_dir / "swap.safetensors"))
    with open(out_dir / "linswap.json", "w") as f:
        json.dump({"kernel": model.linswap_kernel, "base_model": Path(model.name_or_path).name, "tensors": len(state),
                   **meta}, f, indent=1)
    return out_dir


def _legacy_state(ckpt_dir) -> dict:
    """``model.pt`` of a checkpoint written before the swap moved into the upstream model: the whole
    language backbone in Qwen's layout, without the ``language_model`` level."""
    state = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=True)
    return {("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}


def checkpoint_kernel(ckpt_dir) -> str:
    ckpt_dir = Path(ckpt_dir)
    for name, key in (("linswap.json", "kernel"), ("config.json", "linear_kernel")):
        if (ckpt_dir / name).exists():
            with open(ckpt_dir / name) as f:
                kernel = json.load(f).get(key)
            if kernel:
                return kernel
    raise SystemExit(f"{ckpt_dir}: not a linswap checkpoint (no linswap.json naming a kernel)")


def load_checkpoint(model, ckpt_dir):
    """Load a checkpoint into a model whose mixers are already swapped.  Every tensor of the checkpoint must
    land, and every tensor of the swapped layers must come from it."""
    ckpt_dir = Path(ckpt_dir)
    if (ckpt_dir / "swap.safetensors").exists():
        from safetensors.torch import load_file

        state = load_file(str(ckpt_dir / "swap.safetensors"))
    else:
        state = _legacy_state(ckpt_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    stray = [k for k in missing if ".linear_attn." in k]
    if unexpected or stray:
        raise RuntimeError(f"{ckpt_dir} does not fit kernel {model.linswap_kernel!r}: {len(unexpected)} tensors have no "
                           f"place, {len(stray)} swapped tensors are missing, e.g. {(unexpected + stray)[:3]}")
    return model


# ------------------------------------------------------------------------------ building
def resolve(spec: str):
    """``native`` | kernel | checkpoint dir | ``label=<one of those>``  ->  (label, kernel, checkpoint dir | None)."""
    label, _, target = str(spec).rpartition("=")
    if target == NATIVE or target in list_kernels():
        return label or target, target, None
    ckpt = Path(target).resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"{target!r} is neither 'native', a kernel ({', '.join(list_kernels())}) nor a checkpoint directory")
    kernel = checkpoint_kernel(ckpt)
    return label or (f"{kernel}-distilled" if ckpt.name == kernel else f"{kernel}@{ckpt.name}"), kernel, ckpt


def build_model(spec: str = NATIVE, base_model_dir=DEFAULT_BASE_MODEL_DIR, device="cuda", dtype=torch.bfloat16):
    """The upstream Qwen3.5 vision-language model for ``spec`` (see ``resolve``), in eval mode."""
    from transformers import AutoModelForImageTextToText

    _, kernel, ckpt = resolve(spec)
    if ckpt is not None and (ckpt / "linswap.json").exists():
        base = json.loads((ckpt / "linswap.json").read_text()).get("base_model")
        if base and base != Path(base_model_dir).name:
            raise SystemExit(f"{ckpt} holds layers of {base}, not of {Path(base_model_dir).name}: pass that backbone "
                             f"as --base_model_dir")
    model = fast_patch_embed(AutoModelForImageTextToText.from_pretrained(base_model_dir, dtype=dtype))
    model.linswap_kernel = NATIVE
    if kernel != NATIVE:
        swap_mixers(model, kernel)
        if ckpt is not None:
            load_checkpoint(model, ckpt)
    return model.to(device).eval()
