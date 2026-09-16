"""Gated DeltaNet with Bregman soft-thresholding of the state (``gdn_breg``).

The user's in-development ``gated_breg_delta_rule`` package (repo root, not part of the wheel) is GDN
with the proximal step of an L1 penalty applied to the state after every complete 64-token chunk:
``S <- sign(S) * max(|S| - lam, 0)``.  Its parameter set is exactly FLA's ``GatedDeltaNet``, so the
swap is a weight copy; at ``lam = 0`` it is the ``gdn`` control and must verify at the noise floor.

``lam`` is read from the environment variable ``LINSWAP_BREG_LAM`` (default ``0``) when the layer is
built, so ``LINSWAP_BREG_LAM=0.01 linswap verify --kernel gdn_breg`` compares the thresholded operator
against the backbone; ``infer_chunk`` follows ``LINSWAP_BREG_CHUNK`` (default 64).  The kernel is only
registered when the package imports."""

import os

from .common import copy_
from .fla_layer import register_fla_kernel

try:
    from gated_breg_delta_rule.layers.gated_breg_net import GatedBregNet
except Exception:  # package absent or its Triton kernels fail to import
    GatedBregNet = None


def init_extra(layer, src):
    copy_(layer.a_proj.weight, src["a"], "a_proj")
    copy_(layer.b_proj.weight, src["b"], "b_proj")
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, src["dt_bias"], "dt_bias")


def breg_kwargs(cfg):
    return dict(lam=float(os.environ.get("LINSWAP_BREG_LAM", "0")),
                infer_chunk=int(os.environ.get("LINSWAP_BREG_CHUNK", "64")),
                dynamic_lam=False, lam_per_head=True)


if GatedBregNet is not None:
    register_fla_kernel(
        "gdn_breg", GatedBregNet,
        description="Gated DeltaNet + Bregman soft-threshold on the state (gated_breg_delta_rule); exact copy of GDN, "
                    "lam from LINSWAP_BREG_LAM (0 = GDN).",
        output_gate="native",
        layer_kwargs=breg_kwargs,
        init_extra=init_extra,
        new_param_names=("a_proj", "b_proj", "A_log", "dt_bias"),
        exact_init=True,
        notes="lam > 0 is a (small) inexact perturbation of GDN; register verifies at lam = 0.",
    )
