import torch

"""Reference architecture dict for Qwen/Qwen3.5-0.8B (what ``backbones.load_backbone_config`` returns for it).
Kept for tests and as documentation of the config keys; models are normally configured from the HF config."""

QWEN3_5_0_8B_CONFIG = {
    "vocab_size": 248_320,
    "context_length": 262_144,
    "emb_dim": 1_024,
    "n_heads": 8,
    "n_layers": 24,
    "hidden_dim": 3_584,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "dtype": torch.bfloat16,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
}
