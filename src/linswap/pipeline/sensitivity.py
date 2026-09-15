"""Per-layer swap sensitivity and greedy mixed-kernel construction.

    linswap sensitivity --kernel mamba2 --bld_ckpt outputs/lit/mamba2/distill/checkpoint-762 --greedy

Transformer→hybrid conversion papers choose *which* attention layers to keep by measuring each
layer's marginal effect after a brief distillation (KL-guided layer selection, arXiv 2512.20569)
or by greedy validation-guided replacement after blockwise local distillation of every layer
(Distill-then-Replace, arXiv 2601.11667).  The kernel-swap analogue: every linear-attention
layer of the backbone is replaced by the target kernel *one at a time* (forward only) and scored
by KL(teacher ‖ student) on generic text plus multi-query associative recall; then layers are
swapped greedily in order of least KL damage, re-scoring the remaining candidates each round.
The result is a sensitivity map and a curve "k layers swapped → KL / MQAR", from which mixed
models ``<default>;<kernel>@<layers>`` can be built for the full evaluation.

The candidate layers come from a full-model checkpoint of the target kernel (``--bld_ckpt``,
e.g. the checkpoint after the ``layer`` stage of ``linswap distill``, i.e. blockwise local
distillation), or from the tiled / copied init when no checkpoint is given.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer

from ..load_weights import DEFAULT_BASE_MODEL_DIR, REPO_ROOT, build_model
from ..mqar import mqar_accuracy
from ..registry import get_kernel, kernel_map_name
from ..textdata import DEFAULT_TEXT_DIR, ensure_text_data
from .distill import fixed_packed_val, linear_blocks


def add_args(ap):
    ap.add_argument("--kernel", required=True, help="target kernel for the swapped layers")
    ap.add_argument("--baseline", default="gdn", help="kernel of the unswapped layers / the teacher")
    ap.add_argument("--bld_ckpt", default=None, help="full-model checkpoint of --kernel providing the per-layer candidates")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--text_data", default="fineweb-edu")
    ap.add_argument("--text_dir", default=str(DEFAULT_TEXT_DIR))
    ap.add_argument("--val_seqs", type=int, default=16, help="packed validation sequences for the KL")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--mqar_pairs", default="256,1024")
    ap.add_argument("--mqar_samples", type=int, default=5)
    ap.add_argument("--greedy", action="store_true", help="also run the greedy replacement curve")
    ap.add_argument("--output_dir", default=None, help="default outputs/sensitivity/<kernel>")
    ap.add_argument("--ce_chunk_size", type=int, default=4096)
    return ap


@torch.no_grad()
def kl_to_teacher(student, teacher_hidden, teacher_head, loader, chunk):
    """Mean KL(teacher ‖ student) over all positions of the cached teacher hidden states."""
    device = next(student.parameters()).device
    total, n = 0.0, 0
    for batch, t_h in zip(loader, teacher_hidden):
        ids = batch["input_ids"].to(device)
        s_h = student(ids, return_hidden=True).reshape(-1, t_h.shape[-1])
        t_flat = t_h.to(device).reshape(-1, t_h.shape[-1])
        for start in range(0, s_h.shape[0], chunk):
            t_logp = F.log_softmax(F.linear(t_flat[start:start + chunk], teacher_head.weight).float(), -1)
            s_logp = F.log_softmax(F.linear(s_h[start:start + chunk], student.lm_head.weight).float(), -1)
            total += (t_logp.exp() * (t_logp - s_logp)).sum(-1).sum().item()
        n += s_h.shape[0]
    return total / n


def main(args) -> Path:
    device = torch.device("cuda")
    out_dir = (Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "sensitivity" / args.kernel).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base_model_dir)
    pairs = [int(p) for p in args.mqar_pairs.split(",") if p]

    text_dir = ensure_text_data(args.base_model_dir, args.text_data, 1, args.text_dir)
    loader = fixed_packed_val(load_from_disk(str(text_dir / "validation")), args.max_length, args.val_seqs)

    teacher = build_model(args.baseline, base_model_dir=args.base_model_dir, device=device).eval()
    with torch.no_grad():
        teacher_hidden = [teacher(b["input_ids"].to(device), return_hidden=True).cpu() for b in loader]
    donor = build_model(args.kernel, base_model_dir=args.base_model_dir, device=device, ckpt_dir=args.bld_ckpt).eval()
    work = build_model(args.baseline, base_model_dir=args.base_model_dir, device=device).eval()
    idx = [i for i, _ in linear_blocks(work)]
    orig = {i: work.layers[i].linear_attn for i in idx}
    cand = {i: donor.layers[i].linear_attn for i in idx}

    def with_swapped(layers):
        for i in idx:
            work.layers[i].linear_attn = cand[i] if i in layers else orig[i]

    def score(layers, mqar=False):
        with_swapped(layers)
        r = {"kl": kl_to_teacher(work, teacher_hidden, teacher.lm_head, loader, args.ce_chunk_size)}
        if mqar:
            for p in pairs:
                r[f"mqar@{p}"] = mqar_accuracy(work, tok, p, n_samples=args.mqar_samples)["acc"]
        return r

    t0 = time.time()
    base = score(set(), mqar=True)
    full = score(set(idx), mqar=True)
    print(f"[sensitivity] {args.baseline}: {base}   all→{args.kernel}: {full}   ({len(idx)} linear layers)", flush=True)
    single = {}
    for i in idx:
        single[i] = score({i}, mqar=True)
        print(f"  swap layer {i:2d}: KL {single[i]['kl']:.4f}  " +
              "  ".join(f"mqar@{p} {single[i][f'mqar@{p}']:.3f}" for p in pairs), flush=True)
    result = {"kernel": args.kernel, "baseline": args.baseline, "bld_ckpt": args.bld_ckpt, "layers": idx,
              "baseline_scores": base, "all_swapped_scores": full, "single": {str(i): s for i, s in single.items()}}

    if args.greedy:
        chosen, curve = [], []
        remaining = list(idx)
        while remaining:
            best = min(remaining, key=lambda i: score(set(chosen) | {i})["kl"])
            chosen.append(best)
            remaining.remove(best)
            r = score(set(chosen), mqar=True)
            spec = kernel_map_name(get_kernel(args.baseline), {i: get_kernel(args.kernel) for i in chosen})
            curve.append({"k": len(chosen), "added": best, "layers": list(chosen), "spec": spec, **r})
            print(f"  greedy k={len(chosen):2d} +layer {best:2d}: KL {r['kl']:.4f}  " +
                  "  ".join(f"mqar@{p} {r[f'mqar@{p}']:.3f}" for p in pairs) + f"   [{spec}]", flush=True)
        result["greedy"] = curve

    with open(out_dir / "sensitivity.json", "w") as f:
        json.dump(result, f, indent=1)
    md = [f"# Swap sensitivity: {args.baseline} → {args.kernel} (candidates: {args.bld_ckpt or 'init'})", "",
          f"baseline: {base}", f"all swapped: {full}", "", "| layer | KL | " + " | ".join(f"mqar@{p}" for p in pairs) + " |",
          "|---|---|" + "---|" * len(pairs)]
    for i in idx:
        md.append(f"| {i} | {single[i]['kl']:.4f} | " + " | ".join(f"{single[i][f'mqar@{p}']:.3f}" for p in pairs) + " |")
    if args.greedy:
        md += ["", "| k | added | KL | " + " | ".join(f"mqar@{p}" for p in pairs) + " | spec |", "|---|---|---|" + "---|" * (len(pairs) + 1)]
        for c in result["greedy"]:
            md.append(f"| {c['k']} | {c['added']} | {c['kl']:.4f} | " + " | ".join(f"{c[f'mqar@{p}']:.3f}" for p in pairs) + f" | `{c['spec']}` |")
    (out_dir / "sensitivity.md").write_text("\n".join(md) + "\n")
    print(f"[sensitivity] done in {(time.time()-t0)/60:.1f} min -> {out_dir}")
    return out_dir
