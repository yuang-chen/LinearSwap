"""Stage 1 — verify: is the swapped model a function-preserving replacement of the HF backbone?

    linswap verify --kernel kda [--baseline gdn] [--checks layer,logits,layerwise,cache,generation]
                                          [--lengths 8,64,512,4096]
    linswap verify --ckpt outputs/kda/sft_full/checkpoint-50 --checks cache,generation

Checks (all use the pretrained Qwen3.5-0.8B weights, function-preserving init):
    layer       pretrained linear layers 0/1/2 in isolation vs transformers'
                Qwen3_5GatedDeltaNet (recurrent path T=64 and chunk path T=1024)
    logits      full model logits vs HF at several sequence lengths
    layerwise   per-block hidden-state diff vs HF (locates a mismatch)
    cache       the swapped model's cached decode vs its own no-cache decode
    generation  greedy text vs HF

``--ckpt DIR`` verifies an SFT checkpoint instead of the base swap (the kernel is
read from its config.json; the ``layer`` check does not apply, and logits will
legitimately differ from HF after fine-tuning — ``cache`` is the useful one).

``--baseline gdn`` runs the same checks for a second kernel so the numbers can
be read against pure kernel/bf16 noise (the ``gdn`` kernel is an exact weight
copy of the original architecture on FLA Triton kernels).

Acceptance.  With a baseline, the base swap of a kernel registered ``exact_init=True`` is *gated*: every
measurement must sit at the baseline's own noise level, measured in the same run on the same inputs
(same dtype, backend and lengths, so no absolute threshold is carried between setups) —

    layer        rel-L2 error   <= ``--tolerance`` x the baseline's, per layer and length
    logits       KL vs HF       <= ``--tolerance`` x the baseline's, per length
    cache / generation          agreement no more than ``--agreement_slack`` below the baseline's

``main`` returns the measurements with ``status`` ``pass`` / ``fail`` / ``not_gated`` (inexact kernels,
checkpoints and runs without a baseline are reported, not judged), writes them to ``--json``, and exits
non-zero on ``fail`` — which is what stops ``linswap run`` before it trains a broken swap.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..backbones import load_backbone_config
from ..load_weights import DEFAULT_BASE_MODEL_DIR, build_model, load_hf_state_dict, read_checkpoint_kernel
from ..registry import get_kernel


def fmt(x):
    return f"{x:.4g}"


# ----------------------------------------------------------------------------- checks
def check_layer(kernel, weights, device, base_model_dir, layers=(0, 1, 2), lengths=(64, 1024)):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    hf_cfg = AutoConfig.from_pretrained(base_model_dir)
    hf_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    spec = get_kernel(kernel)
    cfg = load_backbone_config(base_model_dir)
    rows = []
    for l in layers:
        ref = Qwen3_5GatedDeltaNet(hf_cfg, l).to(device=device, dtype=torch.bfloat16)
        prefix = f"model.language_model.layers.{l}.linear_attn."
        ref.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
        ours = spec.build(cfg, l)
        spec.init_from_gdn(ours, weights, l, "model.language_model")
        ours = ours.to(device=device, dtype=torch.bfloat16).eval()
        for T in lengths:
            torch.manual_seed(l * 1000 + T)
            x = torch.randn(1, T, cfg["emb_dim"], device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                y_ref = ref(x)
                y_ref = (y_ref[0] if isinstance(y_ref, tuple) else y_ref).float()
                y, _, _ = ours(x)
            d = (y.float() - y_ref).abs()
            rows.append({"layer": l, "T": T, "max": d.max().item(), "mean": d.mean().item(),
                         "rel": (d.norm() / y_ref.norm()).item()})
    print(f"  [{kernel}] layer check (max|diff|, mean|diff|, rel-L2) vs HF torch GDN layer")
    for r in rows:
        print(f"    layer {r['layer']:2d} T={r['T']:5d}: max {fmt(r['max']):>8} mean {fmt(r['mean']):>8} "
              f"rel {fmt(r['rel']):>8}")
    return rows


def make_ids(tokenizer, length, device):
    text = ("The quick brown fox jumps over the lazy dog. In 1815 the Congress of Vienna redrew "
            "the map of Europe; meanwhile, steam engines began to transform British industry. ")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    reps = length // len(ids) + 1
    return ids.repeat(reps)[:length].unsqueeze(0).to(device)


def check_logits(model, hf, tokenizer, lengths, device, kernel):
    print(f"  [{kernel}] logits vs HF")
    rows = []
    for L in lengths:
        ids = make_ids(tokenizer, L, device)
        with torch.no_grad():
            ours = model(ids).float()
            theirs = hf(ids, use_cache=False).logits.float()
        d = (ours - theirs).abs()
        top1 = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
        kl = F.kl_div(F.log_softmax(ours, -1), F.log_softmax(theirs, -1), log_target=True, reduction="none").sum(-1).mean().item()
        rows.append({"T": L, "max": d.max().item(), "mean": d.mean().item(), "top1": top1, "kl": kl})
        print(f"    T={L:6d}: max {fmt(d.max().item()):>7} mean {fmt(d.mean().item()):>7} top1 {top1:.4f} KL {fmt(kl)}")
        del ours, theirs, d
        torch.cuda.empty_cache()
    return rows


def check_layerwise(model, hf, tokenizer, device, kernel, length=256):
    ids = make_ids(tokenizer, length, device)
    ours_tr, hf_tr = {}, {}

    def hook(store, name):
        def _h(mod, inp, out):
            store[name] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return _h

    hs = []
    for i, blk in enumerate(model.layers):
        hs.append(blk.register_forward_hook(hook(ours_tr, i)))
    for i, blk in enumerate(hf.model.layers):
        hs.append(blk.register_forward_hook(hook(hf_tr, i)))
    with torch.no_grad():
        model(ids)
        hf(ids, use_cache=False)
    for h in hs:
        h.remove()
    print(f"  [{kernel}] per-block output diff vs HF (T={length}); L=linear, A=full attention")
    line = []
    for i in range(len(model.layers)):
        d = (ours_tr[i] - hf_tr[i]).abs()
        tag = "L" if model.layers[i].layer_type == "linear_attention" else "A"
        line.append(f"{i:02d}{tag}:{d.max().item():.3f}/{d.mean().item():.4f}")
        if len(line) == 4:
            print("    " + "  ".join(line)); line = []
    if line:
        print("    " + "  ".join(line))


def check_cache(model, tokenizer, device, kernel, cases=((5, 20), (120, 20), (4096, 16))):
    print(f"  [{kernel}] cached decode vs no-cache decode")
    rows = []
    for plen, new in cases:
        ids = make_ids(tokenizer, plen, device)
        t = time.time()
        a = model.generate(ids, max_new_tokens=new, use_cache=False)
        b = model.generate(ids, max_new_tokens=new, use_cache=True)
        agree = (a[0, -new:] == b[0, -new:]).float().mean().item()
        print(f"    prompt {plen:5d} + {new} new: agreement {agree:.3f}  ({time.time()-t:.1f}s)  "
              f"cached='{tokenizer.decode(b[0, plen:])[:60]!s}'")
        rows.append({"prompt": plen, "new": new, "agreement": agree})
    return rows


def check_generation(model, hf, tokenizer, device, kernel, prompt="The capital of France is", new=20):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    ours = model.generate(ids, max_new_tokens=new, use_cache=False)[0, ids.shape[1]:]
    with torch.no_grad():
        theirs = hf.generate(ids, max_new_tokens=new, do_sample=False)[0, ids.shape[1]:]
    n = min(len(ours), len(theirs))
    agree = (ours[:n] == theirs[:n]).float().mean().item()
    print(f"  [{kernel}] greedy generation vs HF: token agreement {agree:.3f}")
    print(f"    ours: {tokenizer.decode(ours)!r}")
    print(f"    HF  : {tokenizer.decode(theirs)!r}")
    return [{"prompt": prompt, "new": new, "agreement": agree}]


# ------------------------------------------------------------------------------ acceptance
GATES = {  # check -> (row key, measured field, kind)
    "layer": (("layer", "T"), "rel", "error"),
    "logits": (("T",), "kl", "error"),
    "cache": (("prompt", "new"), "agreement", "agreement"),
    "generation": (("prompt", "new"), "agreement", "agreement"),
}


def judge(measured, baseline, tolerance=3.0, agreement_slack=0.25, error_floor=1e-7):
    """Failures of ``measured`` against ``baseline`` (both ``{check: [row, ...]}``), as readable strings.
    An error passes up to ``tolerance`` x the baseline's (plus ``error_floor``, for a baseline that is
    exactly 0); an agreement may fall ``agreement_slack`` below the baseline's."""
    failures = []
    for check, (key, field, kind) in GATES.items():
        ref = {tuple(r[k] for k in key): r[field] for r in baseline.get(check, [])}
        for row in measured.get(check, []):
            ident = tuple(row[k] for k in key)
            if ident not in ref:
                continue
            got, base = row[field], ref[ident]
            where = f"{check} " + " ".join(f"{k}={v!r}" for k, v in zip(key, ident))
            if got != got:       # NaN
                failures.append(f"{where}: {field} is NaN")
            elif kind == "error" and got > tolerance * base + error_floor:
                failures.append(f"{where}: {field} {fmt(got)} > {tolerance:g} x baseline {fmt(base)}")
            elif kind == "agreement" and got < base - agreement_slack:
                failures.append(f"{where}: {field} {got:.3f} < baseline {base:.3f} - {agreement_slack:g}")
    return failures


# ------------------------------------------------------------------------------ main
def add_args(ap):
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--ckpt", default=None, help="SFT checkpoint dir (kernel read from its config.json)")
    ap.add_argument("--baseline", default=None, help="second kernel to run for comparison, e.g. gdn")
    ap.add_argument("--checks", default="layer,logits,layerwise,cache,generation")
    ap.add_argument("--lengths", default="8,64,512,4096")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="an exact swap passes while its errors stay within this factor of the baseline's")
    ap.add_argument("--agreement_slack", type=float, default=0.25,
                    help="how far cache / generation agreement may fall below the baseline's")
    ap.add_argument("--json", default=None, help="write the measurements and the status to this file")


def main(args):
    if args.ckpt:
        args.kernel = args.kernel or read_checkpoint_kernel(args.ckpt)
    if args.kernel is None:
        raise SystemExit("verify: --kernel or --ckpt is required")

    checks = set(args.checks.split(","))
    if args.ckpt:
        checks.discard("layer")
    lengths = [int(x) for x in args.lengths.split(",")]
    device = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Purpose: check that the swapped model is a function-preserving replacement of the HF backbone "
          f"(kernel={args.kernel}, baseline={args.baseline}, ckpt={args.ckpt}).")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    weights = load_hf_state_dict(args.base_model_dir)
    hf = None
    if checks & {"logits", "layerwise", "generation"}:
        hf = AutoModelForCausalLM.from_pretrained(args.base_model_dir, dtype=torch.bfloat16).to(device).eval()

    measurements = {}
    for kernel in dict.fromkeys(k for k in (args.kernel, args.baseline) if k):
        spec = get_kernel(kernel)
        got = measurements[kernel] = {}
        print(f"\n=== kernel: {kernel} — {spec.description}")
        if not spec.exact_init:
            print(f"  NOTE: init is not function preserving ({spec.notes}); deviations below are expected.")
        if "layer" in checks:
            got["layer"] = check_layer(kernel, weights, device, args.base_model_dir)
        model = None
        if checks - {"layer"}:
            ckpt = args.ckpt if kernel == args.kernel else None
            model = build_model(kernel, base_model_dir=args.base_model_dir, hf_weights=weights, device=device,
                                ckpt_dir=ckpt).eval()
            n_new = sum(p.numel() for _, p in model.new_parameters())
            print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
                  f"kernel gate/new params: {n_new/1e6:.2f}M ({spec.new_param_names})")
        if "logits" in checks:
            got["logits"] = check_logits(model, hf, tokenizer, lengths, device, kernel)
        if "layerwise" in checks:
            check_layerwise(model, hf, tokenizer, device, kernel)
        if "cache" in checks:
            got["cache"] = check_cache(model, tokenizer, device, kernel)
        if "generation" in checks:
            got["generation"] = check_generation(model, hf, tokenizer, device, kernel)
        del model
        torch.cuda.empty_cache()

    exact = get_kernel(args.kernel).exact_init
    result = {"kernel": args.kernel, "baseline": args.baseline, "ckpt": args.ckpt, "exact_init": exact,
              "tolerance": args.tolerance, "agreement_slack": args.agreement_slack, "lengths": lengths,
              "measurements": measurements, "failures": []}
    if args.ckpt:
        result["status"], why = "not_gated", "a trained checkpoint legitimately differs from the backbone"
    elif not exact:
        result["status"], why = "not_gated", "the kernel's init is not function preserving"
    elif not args.baseline or args.baseline == args.kernel:
        result["status"], why = "not_gated", "no --baseline to take the noise level from"
    else:
        result["failures"] = judge(measurements[args.kernel], measurements[args.baseline], args.tolerance,
                                   args.agreement_slack)
        result["status"] = "fail" if result["failures"] else "pass"
        why = (f"every check within {args.tolerance:g} x the {args.baseline} noise level" if not result["failures"]
               else "; ".join(result["failures"]))
    print(f"\n[verify] {args.kernel}: {result['status'].upper()} ({why})")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    if result["status"] == "fail":
        raise SystemExit(f"verify: {args.kernel} is registered exact_init=True but does not reproduce the backbone")
    return result
