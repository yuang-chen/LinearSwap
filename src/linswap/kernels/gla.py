"""Gated Linear Attention (GLA, arXiv:2312.06635) via FLA's ``GatedLinearAttention`` — the
"stock FLA layer, zero custom code" example.

GLA is linear attention with a data-dependent per-key-channel decay and no delta rule:
``S_t = Diag(exp(gk_t)) S_{t-1} + k_t v_t^T``, ``gk_t = logsigmoid(gk_proj(x_t)) / 16``.
Relative to GDN it lacks the erase term and it does not L2-normalise q/k, so the init is
inexact: the shared weights (q/k/v, convolutions, the output gate — GLA's ``g_proj`` +
``g_norm_swish_gate`` is the backbone's gate under another name) are copied and the decay
is *approximately* mapped.  Distillation is required.

Decay mapping.  GLA's gate is a rank-``gate_low_rank_dim`` MLP ``Linear(D->r) -> Linear(r->H*K)``
and the backbone has ``H <= r`` heads, so the first linear takes GDN's ``a_proj`` unchanged and
the second linear needs one weight and bias per head, repeated over its ``K`` key channels
(``fit_gate``): ``logsigmoid(w_h a + b_h) / 16 ~ -exp(A_log_h) softplus(a + dt_bias_h)``.  The two
gate functions have different shapes so the match is a least-squares fit over the range in
which the pretrained gate operates, not an identity; on real activations the retention factor is
matched to 0.014 on average, with a few heads off by up to 0.15 in their tails.  What the tiling
buys is that every channel of a head starts with the *same* decay: the state's growth under
repeated writes is then a per-head scalar which the per-head RMSNorm cancels (as in Mamba-2).
With FLA's random per-channel gate init, channels of one head have different horizons, the slow
ones never saturate, and the layer's output keeps changing with the repeat count — which is what
breaks RULER's repeated-noise needle (``niah_single_1``).
FLA's random gate init gave 87.5 / 64.8 / 43.2 / 1.6 (RULER task average at 4K-128K) and 94.3
relative on short context on the same recipe; the mapped init gives 99.1 / 85.4 / 76.0 / 68.8 and 99.2.
"""

import math

import torch
import torch.nn.functional as F
from fla.layers import GatedLinearAttention

from .common import tile_vec
from .fla_layer import register_fla_kernel

GATE_LOGIT_NORMALIZER = 16          # FLA's default; the layer is built with it below


def fit_gate(A_log: torch.Tensor, dt_bias: torch.Tensor, steps: int = 800, s_range: tuple[float, float] = (-9.0, 6.0),
             s_weight: tuple[float, float] | None = (-4.0, 4.0)) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head ``(w, b)`` with ``logsigmoid(w a + b) / 16 ~ -exp(A_log) softplus(a + dt_bias)``.

    Fitted in retention space (``alpha = exp(log-decay)``) on a grid of the softplus argument
    ``s = a + dt_bias`` over ``s_range`` (the pretrained Qwen3.5 gates sit around ``s = -5``, sd 2,
    with a tail past 0 that must be covered), optionally weighted by a Gaussian ``s_weight =
    (centre, sd)``.  The start point is the closed form of the ``s << 0`` regime, where GDN's
    log-decay is ``-exp(A_log) e^s`` and GLA's is ``-e^{-u} / 16``: ``w = -1``,
    ``b = -dt_bias - A_log - log 16``.  Runs on CPU in float32; a few seconds for all layers."""
    A = A_log.detach().float().cpu()
    db = dt_bias.detach().float().cpu()
    s = torch.linspace(*s_range, 601)[:, None]                       # [G, 1]
    weight = torch.ones_like(s) if s_weight is None else torch.exp(-0.5 * ((s - s_weight[0]) / s_weight[1]) ** 2)
    a = s - db[None]                                                 # [G, H]
    target = torch.exp(-A.exp()[None] * F.softplus(s))               # GDN retention  [G, H]
    w = torch.full_like(A, -1.0).requires_grad_()
    b = (-db - A - math.log(GATE_LOGIT_NORMALIZER)).clone().requires_grad_()
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=steps, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred = torch.exp(F.logsigmoid(w * a + b) / GATE_LOGIT_NORMALIZER)
        loss = (weight * (pred - target) ** 2).sum(0).mean()          # heads are independent: sum per head
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def init_extra(layer, src):
    w1, w2 = layer.gk_proj[0], layer.gk_proj[-1]
    H, K = src["a"].shape[0], layer.head_k_dim
    r = w1.weight.shape[0]
    if r < H:
        raise ValueError(f"gla: gate_low_rank_dim {r} < {H} heads; GDN's decay projection does not fit")
    if w2.weight.shape[0] != H * K:
        raise ValueError(f"gla: gk_proj output {w2.weight.shape[0]} != {H} heads * {K} channels")
    w, b = fit_gate(src["A_log"], src["dt_bias"])
    head = torch.arange(H * K) // K
    with torch.no_grad():
        w1.weight[:H].copy_(src["a"].to(w1.weight.dtype))   # rows >= H (r > H): FLA init, zero output weight
        sel = torch.zeros(H * K, r, dtype=w2.weight.dtype)
        sel[torch.arange(H * K), head] = w[head].to(w2.weight.dtype)
        w2.weight.copy_(sel)
        w2.bias.copy_(tile_vec(b, K).to(w2.bias.dtype))


def layer_kwargs(cfg):
    hidden = cfg["emb_dim"]
    if cfg["linear_num_key_heads"] != cfg["linear_num_value_heads"]:
        raise ValueError("GLA has a single num_heads; key/value head counts must match")
    return dict(
        expand_k=cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] / hidden,
        expand_v=cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"] / hidden,
        num_heads=cfg["linear_num_key_heads"],
        gate_logit_normalizer=GATE_LOGIT_NORMALIZER,
        gate_low_rank_dim=max(16, cfg["linear_num_key_heads"]),
        use_short_conv=True, use_output_gate=True, gate_fn="swish", fuse_norm=True,
    )


register_fla_kernel(
    "gla", GatedLinearAttention,
    description="Gated Linear Attention (FLA GatedLinearAttention); shared weights copied, GDN decay mapped "
                "per head into the low-rank gate (approximate) — NOT function preserving.",
    layer_kwargs=layer_kwargs, output_gate="native", norm_attr="g_norm_swish_gate",
    init_extra=init_extra, new_param_names=("gk_proj",), exact_init=False,
    notes="Inexact swap (no erase, approximate decay, no q/k normalisation): distil before SFT.",
)
