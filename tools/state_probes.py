#!/usr/bin/env python
"""State probes for the unmodified GDN backbone: SVD truncation curves (probe 1) and horizon vs effective rank
of the memory state (probe 2).  Spec: single documents >= 16K tokens, first 16,384 tokens of each.

    python tools/state_probes.py docs                       # pick the long documents -> outputs/state_probes/docs.json
    python tools/state_probes.py probe1 --worker 0 --workers 8   # one GPU's share of the truncation sweep
    python tools/state_probes.py probe2                     # gates, keys, P_t and S_t at 1K / 4K / 16K
    python tools/state_probes.py report                     # CSVs, plots, numbers for the write-up

Conventions (printed by ``probe1``/``probe2`` at start):
  * state per linear layer: ``[B, H, d_v, d_k] = [B, 16, 128, 128]`` fp32 -- FLA's ``state_v_first=True``,
    so the readout is ``o_t = S_t q_t`` and the key axis is the LAST one;
  * decay ``g_t = -exp(A_log) * softplus(a_t + dt_bias)`` (log space, per head), ``alpha_t = exp(g_t)``;
  * write gate ``beta_t = sigmoid(b_t)`` in (0, 1) (``allow_neg_eigval=False``);
  * q, k are L2-normalised inside the kernel (FLA ``l2norm_fwd``, eps 1e-6); probe 2 records that k;
  * layers are numbered by their global index in the 24-layer stack (linear layers 0,1,2,4,...,22).
"""
import argparse, json, math, os, zlib
from pathlib import Path

import numpy as np
import torch

OUT = Path("outputs/state_probes")
T = 16384
SEG = 256
R_KEEP = [128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2, 1]
R_DROP = [1, 2, 5, 10, 20]
SHORTLIST = {  # (layer, head): group
    (18, 0): "retrieval", (20, 10): "retrieval", (17, 11): "retrieval", (21, 4): "retrieval",
    (16, 2): "retrieval", (20, 13): "retrieval",
    (0, 1): "L0 anomaly", (0, 2): "L0 anomaly", (0, 8): "L0 anomaly", (0, 12): "L0 anomaly",
    (0, 0): "tiny control", (1, 1): "tiny control",
}
BUCKETS = [("0-1K", 0, 1024), ("1K-4K", 1024, 4096), ("4K-8K", 4096, 8192), ("8K-16K", 8192, T)]
EPS32 = 1.19e-7


# ---------------------------------------------------------------------------------------------- common
def erank(sv):
    """Roy-Vetterli effective rank over the last axis of non-negative singular values / eigenvalues."""
    sv = np.clip(np.asarray(sv, dtype=np.float64), 0, None)
    p = sv / np.maximum(sv.sum(-1, keepdims=True), 1e-300)
    return np.exp(-(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(-1))


def numrank(sv):
    sv = np.asarray(sv, dtype=np.float64)
    return (sv > sv.max(-1, keepdims=True) * 128 * EPS32).sum(-1)


def load_ids(n):
    from datasets import load_from_disk
    docs = json.load(open(OUT / "docs.json"))
    ds = load_from_disk(docs["dataset"])
    ids = [ds[i]["input_ids"][:T] for i in docs["index"][:n]]
    return torch.tensor(ids)


def build(ckpt=None):
    """The unmodified backbone (weight copy), or -- with ``ckpt`` -- a native checkpoint such as the distilled control."""
    from linswap import build_model
    model = build_model("gdn", ckpt_dir=ckpt).eval()
    lin = [i for i, l in enumerate(model.model.layers) if l.layer_type != "full_attention"]
    return model, lin


def print_conventions(model, lin):
    la = model.model.layers[lin[0]].linear_attn
    print(f"[conventions] linear layers (global ids): {lin}")
    print(f"[conventions] heads={la.num_heads} v_heads={la.num_v_heads} d_k={la.head_k_dim} d_v={la.head_v_dim}; "
          f"state [B, H, d_v, d_k] fp32 (FLA state_v_first=True): readout o = S q, key axis = last")
    print("[conventions] g = -exp(A_log)*softplus(a + dt_bias) (log alpha, per head); beta = sigmoid(b) in (0,1) "
          f"(allow_neg_eigval={la.allow_neg_eigval}); q,k L2-normalised in kernel (l2norm_fwd, eps 1e-6)")
    print("[conventions] erank = exp(-sum p log p), p = s/sum(s) (Roy-Vetterli); numerical rank = #{s > s_max*128*1.19e-7}")


def token_nll(model, hidden, targets):
    """Per-token NLL (fp32) for hidden [B, t, D] predicting targets [B, t]."""
    out = []
    for s in range(0, hidden.shape[1], 64):
        logits = model.lm_head(hidden[:, s:s + 64]).float()
        out.append(torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets[:, s:s + 64], reduction="none"))
    return torch.cat(out, 1)


@torch.no_grad()
def segmented_nll(model, lin, ids, intervene=None):
    """NLL of tokens 1..T-1 (index t-1 = predicting token t), prefilling in SEG-token segments through the cache;
    ``intervene(cache)`` edits the carried linear states between segments."""
    from linswap.model import SwapCache
    model.reset_cache_state()
    cache = SwapCache(len(model.model.layers))
    nll = []
    for s in range(0, T, SEG):
        h = model(ids[:, s:s + SEG], cache=cache, use_cache=True, return_hidden=True)
        tgt = ids[:, s + 1:s + SEG + 1]
        nll.append(token_nll(model, h[:, :tgt.shape[1]], tgt))
        if intervene is not None and s + SEG < T:
            intervene(cache)
    return torch.cat(nll, 1).cpu()


@torch.no_grad()
def unsegmented_nll(model, ids):
    model.reset_cache_state()
    h = model(ids, return_hidden=True)
    return token_nll(model, h[:, :-1], ids[:, 1:]).cpu()


def retrievable_mask(ids):
    """[B, T-1]: predicting token t+1 from context whose last bigram (x_{t-1}, x_t) already occurred at
    (x_{j-1}, x_j) with t - j > 512 -- an induction/copy opportunity from far context."""
    B = ids.shape[0]
    m = np.zeros((B, T - 1), dtype=bool)
    x = ids.numpy()
    for b in range(B):
        first = {}
        for t in range(1, T - 1):
            key = (x[b, t - 1], x[b, t])
            j = first.setdefault(key, t)
            m[b, t] = t - j > 512
    return torch.from_numpy(m)


def summarise(nll, mask_retr):
    """Per-sequence mean NLL per bucket: {bucket: (per-seq means [B], n_tokens)}; the bucket is the position of
    the PREDICTED token (index t-1 predicts token t)."""
    pos = torch.arange(1, T)
    out = {"all": (nll.mean(1).numpy(), nll.numel())}
    for name, lo, hi in BUCKETS:
        sel = (pos >= lo) & (pos < hi)
        out[name] = (nll[:, sel].mean(1).numpy(), int(sel.sum()) * nll.shape[0])
    per = [(nll[b][mask_retr[b]].mean().item() if mask_retr[b].any() else float("nan")) for b in range(nll.shape[0])]
    out["retrievable"] = (np.array(per), int(mask_retr.sum()))
    return out


# ---------------------------------------------------------------------------------------------- probe 1
def make_intervention(lin, scope, layer, head, variant, r, threads):
    layers = lin if scope == "global" else [layer]
    heads = None if head < 0 else [head]

    def intervene(cache):
        for i in layers:
            st = cache.linear_cache[i]["recurrent_state"]               # [B, H, V, K] fp32
            sub = st if heads is None else st[:, heads]
            # LAPACK SVD in float64: an fp32 round trip perturbs S by ~1e-6, which over 63 boundaries x 18 layers
            # flips enough bf16 roundings downstream to move single tokens by ~0.4 nats; in float64 keep_top(128)
            # casts back to the original fp32 state.
            x = sub.detach().cpu().double()
            U, s, Vh = torch.linalg.svd(x, full_matrices=False)
            keep = (U[..., :r] * s[..., None, :r]) @ Vh[..., :r, :]
            new = keep if variant == "keep_top" else x - keep
            new = new.to(st.device, st.dtype)
            if heads is None:
                st.copy_(new)
            else:
                st[:, heads] = new
    return intervene


def probe1_jobs(lin):
    jobs = [("global", -1, -1, "keep_top", r) for r in R_KEEP]
    jobs += [("layer", L, -1, "keep_top", r) for L in lin for r in R_KEEP]
    for (L, h) in SHORTLIST:
        jobs += [("head", L, h, "keep_top", r) for r in R_KEEP]
        jobs += [("head", L, h, "drop_top", r) for r in R_DROP]
    return jobs


def cmd_probe1(a):
    torch.set_num_threads(max(1, os.cpu_count() // a.workers))
    model, lin = build()
    if a.worker == 0:
        print_conventions(model, lin)
    ids = load_ids(a.n_seq).cuda()
    mask = retrievable_mask(ids.cpu())
    base = segmented_nll(model, lin, ids)
    np.save(OUT / f"p1_baseline_w{a.worker}.npy", base.numpy())
    rows_path = OUT / f"p1_rows_w{a.worker}.jsonl"
    done = set()
    if rows_path.exists():
        done = {tuple(json.loads(l)["job"]) for l in open(rows_path)}
    if a.worker == 0 and not (OUT / "p1_sanity.json").exists():
        unseg = unsegmented_nll(model, ids)
        json.dump({"segmented_mean_nll": base.mean().item(), "unsegmented_mean_nll": unseg.mean().item(),
                   "abs_diff": abs(base.mean().item() - unseg.mean().item()),
                   "per_seq_abs_diff_max": (base.mean(1) - unseg.mean(1)).abs().max().item()},
                  open(OUT / "p1_sanity.json", "w"), indent=1)
        print("[probe1] segmented vs unsegmented:", open(OUT / "p1_sanity.json").read(), flush=True)
    bsum = summarise(base, mask)
    jobs = probe1_jobs(lin)
    # global jobs are ~10x the cost of the rest: deal them first, round-robin
    mine = [j for k, j in enumerate(jobs) if k % a.workers == a.worker]
    with open(rows_path, "a") as f:
        for job in mine:
            if tuple(job) in done:
                continue
            nll = segmented_nll(model, lin, ids, make_intervention(lin, *job, threads=None))
            s = summarise(nll, mask)
            for bucket, (per, n) in s.items():
                d = per - bsum[bucket][0]
                ok = ~np.isnan(d)
                f.write(json.dumps({"job": list(job), "bucket": bucket, "n_tokens": n,
                                    "nll": float(np.nanmean(per)), "nll_baseline": float(np.nanmean(bsum[bucket][0])),
                                    "delta": float(d[ok].mean()), "delta_std": float(d[ok].std(ddof=1)) if ok.sum() > 1 else 0.0,
                                    "delta_per_seq": d.tolist()}) + "\n")
            f.flush()
            print(f"[probe1 w{a.worker}] {job}: delta(all) = {s['all'][0].mean() - bsum['all'][0].mean():+.5f}", flush=True)


# ---------------------------------------------------------------------------------------------- probe 2
class Recorder:
    """Wraps FLA's chunk_gated_delta_rule: records log alpha, beta and the L2-normalised k that enter the
    recurrence, and carries the decayed key covariance P_t = alpha_t P_{t-1} + beta_t k_t k_t^T exactly."""

    def __init__(self):
        self.layer = None
        self.g, self.beta, self.P, self.last_k = {}, {}, {}, {}

    def wrap(self, fn):
        from fla.modules.l2norm import l2norm_fwd

        def wrapped(**kw):
            kn = l2norm_fwd(kw["k"])[0].float()                                  # [B, t, H, K]
            g = -kw["A_log"].float().exp() * torch.nn.functional.softplus(kw["g"].float() + kw["dt_bias"].float())
            beta = torch.sigmoid(kw["beta"].float())                            # [B, t, H]
            L = self.layer
            self.g.setdefault(L, []).append(g.cpu())
            self.beta.setdefault(L, []).append(beta.cpu())
            C = g.double().cumsum(1)                                            # inclusive cumulative log decay
            w = (beta.double() * (C[:, -1:] - C).exp()).float()                 # prod_{s>t} alpha_s * beta_t
            B, t, H, K = kn.shape
            P = self.P.get(L)
            if P is None:
                P = torch.zeros(B, H, K, K, device=kn.device, dtype=torch.float64)
            P = P * C[:, -1].exp()[:, :, None, None]
            P = P + torch.einsum("bthk,bthj->bhkj", kn * w[..., None], kn).double()
            self.P[L] = P
            self.last_k[L] = kn[:, -4096:].transpose(1, 2).contiguous()         # [B, H, n, K]
            return fn(**kw)
        return wrapped


def cmd_probe2(a):
    import fla.layers.gated_deltanet as gdn_mod
    from linswap.model import SwapCache
    model, lin = build()
    print_conventions(model, lin)
    rec = Recorder()
    gdn_mod.chunk_gated_delta_rule = rec.wrap(gdn_mod.chunk_gated_delta_rule)
    for i in lin:
        model.model.layers[i].linear_attn.register_forward_pre_hook(lambda m, args, i=i: setattr(rec, "layer", i))
    ids = load_ids(a.n_seq).cuda()
    marks = [1024, 4096, T]
    res = {"layers": lin, "marks": marks, "S_sv": {}, "P_ev": {}, "overlap": {}}
    model.reset_cache_state()
    cache = SwapCache(len(model.model.layers))
    pos = 0
    with torch.no_grad():
        for m in marks:
            model(ids[:, pos:m], cache=cache, use_cache=True, last_logits_only=True)
            pos = m
            for i in lin:
                S = cache.linear_cache[i]["recurrent_state"].double()          # [B, H, V, K]
                _, sv, Vh = torch.linalg.svd(S.cpu(), full_matrices=False)
                ev, E = torch.linalg.eigh(rec.P[i].cpu())                      # ascending
                ev, E = ev.flip(-1), E.flip(-1)
                res["S_sv"][f"{i}@{m}"] = sv.numpy()
                res["P_ev"][f"{i}@{m}"] = ev.numpy()
                if m == T:                                                     # key-side subspace overlap
                    Vk = Vh.transpose(-1, -2)                                   # [B, H, K, K], columns = key directions
                    for k in (8, 32):
                        ov = (E[..., :k].transpose(-1, -2) @ Vk[..., :k]).pow(2).sum((-1, -2)) / k
                        res["overlap"][f"{i}@{k}"] = ov.numpy()
            print(f"[probe2] {m} tokens", flush=True)
    g = {i: torch.cat(rec.g[i], 1).numpy() for i in lin}                        # [B, T, H]
    beta = {i: torch.cat(rec.beta[i], 1).numpy() for i in lin}
    lastk = {i: rec.last_k[i].cpu().numpy() for i in lin}                      # [B, H, 4096, K]
    np.savez_compressed(OUT / "p2_raw.npz", **{f"g{i}": g[i] for i in lin}, **{f"beta{i}": beta[i] for i in lin},
                        **{f"lastk{i}": lastk[i] for i in lin},
                        **{f"S{k}": v for k, v in res["S_sv"].items()}, **{f"P{k}": v for k, v in res["P_ev"].items()},
                        **{f"ov{k}": v for k, v in res["overlap"].items()})
    print("[probe2] wrote", OUT / "p2_raw.npz")


# ---------------------------------------------------------------------------------------------- probe 1r
# Probe 1 with retrieval as the metric: the same keep_top(r) on the carried state at every SEG-token boundary of the
# prompt, then greedy decoding from the intervened cache, scored with RULER's string_match_all.  r = 128 is the
# untouched segmented run (the baseline).  Prompts come from the validation.jsonl files RULER already prepared.
R_RETR = [128, 32, 12, 4, 1]
RETR_JOBS = [  # (task, length, n_samples)
    ("niah_multikey_1", 16384, 200), ("niah_multikey_1", 65536, 100), ("niah_multikey_1", 131072, 100),
    ("cwe", 16384, 50), ("cwe", 65536, 50), ("fwe", 16384, 50), ("fwe", 65536, 50),
    ("vt", 16384, 50), ("vt", 65536, 50),
]
GEN_TOKENS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}     # RULER's tokens_to_generate per family


def task_family(task):
    return "niah" if task.startswith("niah") else ("qa" if task.startswith("qa") else task)


def find_ruler_data(task, L):
    import glob
    cands = [c for c in glob.glob(f"outputs/eval/*/ruler/*/{L}/data/{task}/validation.jsonl") if "_superseded" not in c]
    if not cands:
        raise FileNotFoundError(f"no prepared RULER data for {task} @ {L}")
    return max(sorted(cands), key=lambda c: sum(1 for _ in open(c)))


def string_match_all(pred, refs):
    return sum(1.0 for r in refs if r.lower() in pred.lower()) / len(refs)


def _cholqr(Y):
    """Orthonormal basis of the columns of Y [N, V, q] (two passes of Cholesky-QR in float64; batched QR fallback)."""
    for _ in range(2):
        A = Y.transpose(-1, -2) @ Y
        q = A.shape[-1]
        eye = torch.eye(q, dtype=A.dtype, device=A.device)
        A = A + eye * (A.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)[..., None] * 1e-14 + 1e-300)
        R, info = torch.linalg.cholesky_ex(A)
        if int(info.max()) != 0:
            Y, _ = torch.linalg.qr(Y)
            continue
        Y = torch.linalg.solve_triangular(R, Y.transpose(-1, -2), upper=False).transpose(-1, -2)
    return Y


def keep_top_subspace(S, r, iters=30, gen=None):
    """Rank-r truncation of every matrix in S [N, V, K] (float64) keeping its top-r left singular directions.
    Subspace iteration on S S^T with q = r + 8 (capped so the q x q Rayleigh-Ritz eigh stays on the GPU's batched
    Jacobi path, n <= 32), then U_r U_r^T S.  Matches the exact SVD truncation to ~1e-5 in retained energy at a
    small fraction of its cost; used instead of torch.linalg.svd, which is ~1.7 s per boundary for 288 matrices."""
    N, V, K = S.shape
    q = r + 8 if r + 8 <= 32 else r
    q = min(q, V)
    G = S @ S.transpose(-1, -2)
    Q = _cholqr(torch.randn(N, V, q, dtype=S.dtype, device=S.device, generator=gen))
    for _ in range(iters):
        Q = _cholqr(G @ Q)
    M = Q.transpose(-1, -2) @ G @ Q
    try:
        _, W = torch.linalg.eigh(M if q <= 32 else M.cpu())                    # ascending
    except torch._C._LinAlgError:   # cuSOLVER's batched Jacobi can fail on a (near-)zero state; LAPACK does not
        _, W = torch.linalg.eigh(M.cpu())
    Ur = Q @ W[..., -r:].to(S.device)
    return Ur @ (Ur.transpose(-1, -2) @ S)


def make_retrieval_intervention(lin, r, gen):
    if r >= 128:
        return None

    def intervene(cache):
        sts = [cache.linear_cache[i]["recurrent_state"] for i in lin]           # each [B, H, V, K]
        S = torch.cat([s.reshape(-1, s.shape[-2], s.shape[-1]) for s in sts]).double()
        Kp = keep_top_subspace(S, r, gen=gen)
        off = 0
        for st in sts:
            n = st.shape[0] * st.shape[1]
            st.copy_(Kp[off:off + n].reshape(st.shape).to(st.dtype))
            off += n
    return intervene


@torch.no_grad()
def segmented_generate(model, ids, max_new, stop_ids, intervene=None, seg=SEG):
    """Prefill ``ids`` [1, Tp] in ``seg``-token segments through the cache, applying ``intervene(cache)`` after every
    segment but the last, then decode greedily from that cache.  Returns the generated token ids."""
    from linswap.model import SwapCache
    model.reset_cache_state()
    cache = SwapCache(len(model.model.layers))
    Tp = ids.shape[1]
    logits = None
    for s in range(0, Tp, seg):
        logits = model(ids[:, s:s + seg], cache=cache, use_cache=True, last_logits_only=True)
        if intervene is not None and s + seg < Tp:
            intervene(cache)
    out = []
    for _ in range(max_new):
        nxt = logits.reshape(ids.shape[0], -1).argmax(-1, keepdim=True)
        t = int(nxt)
        if t in stop_ids:
            break
        out.append(t)
        logits = model(nxt, cache=cache, use_cache=True, last_logits_only=True)
    return out


def load_done_p1r(prefix="p1r"):
    """(task, L, r, index) of every sample any worker has already scored -- resume is global, so a relaunch with a
    different worker count just redistributes what is left (launch all workers together so they see the same files)."""
    import glob
    done = set()
    for f in glob.glob(str(OUT / f"{prefix}_rows_w*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            done.add((r["task"], r["L"], r["r"], r["index"]))
    return done


def probe1r_jobs(a, done):
    rs = [int(x) for x in a.r.split(",")]
    specs = RETR_JOBS if not a.jobs else [(t, int(L), int(n)) for t, L, n in (j.split(":") for j in a.jobs.split(","))]
    left = {}
    for (t, L, n) in specs:
        for r in rs:
            left[(t, L, n, r)] = n - sum(1 for d in done if d[:3] == (t, L, r))
    jobs = [j for j, k in left.items() if k > 0]
    # measured: ~4 s / sample at 16K, ~16 s at 64K, ~37 s at 128K (batch 1, 256-token segments)
    cost = lambda j: left[j] * {16384: 4, 65536: 16, 131072: 37}.get(j[1], j[1] / 4096)
    jobs.sort(key=lambda j: -cost(j))                                          # longest-first, then LPT over workers
    load = [0.0] * a.workers
    owner = []
    for j in jobs:
        w = min(range(a.workers), key=lambda i: load[i])
        load[w] += cost(j)
        owner.append(w)
    return [j for j, w in zip(jobs, owner) if w == a.worker]


def cmd_probe1r(a):
    import time
    from transformers import AutoTokenizer
    torch.set_num_threads(max(1, os.cpu_count() // a.workers))
    model, lin = build(a.ckpt or None)
    prefix = "p1r" + (f"_{a.tag}" if a.tag else "")
    if a.worker == 0:
        print_conventions(model, lin)
        print(f"[probe1r] model = {a.ckpt or 'unmodified backbone'}; rows prefix {prefix}; segment {a.seg}, r in {a.r}, "
              f"truncation = subspace iteration (see keep_top_subspace)")
    tok = AutoTokenizer.from_pretrained("models/Qwen3.5-0.8B")
    stop_ids = {tok.eos_token_id, tok.pad_token_id}
    gen = torch.Generator(device="cuda").manual_seed(0)
    rows_path = OUT / f"{prefix}_rows_w{a.worker}.jsonl"
    done = load_done_p1r(prefix)
    data_cache = {}
    with open(rows_path, "a") as f:
        for task, L, n, r in probe1r_jobs(a, done):
            if (task, L) not in data_cache:
                path = find_ruler_data(task, L)
                data_cache[(task, L)] = [json.loads(l) for l in open(path)][:n]
                print(f"[probe1r w{a.worker}] {task}@{L}: {len(data_cache[(task, L)])} samples from {path}", flush=True)
            samples = data_cache[(task, L)]
            intervene = make_retrieval_intervention(lin, r, gen)
            scores, t0 = [], time.time()
            for smp in samples:
                if (task, L, r, smp["index"]) in done:
                    continue
                ids = tok(smp["input"] + (smp.get("answer_prefix") or ""), return_tensors="pt")["input_ids"].cuda()
                t1 = time.time()
                out = segmented_generate(model, ids, GEN_TOKENS[task_family(task)], stop_ids, intervene, seg=a.seg)
                pred = tok.decode(out, skip_special_tokens=True)
                sc = string_match_all(pred, smp["outputs"])
                scores.append(sc)
                f.write(json.dumps({"task": task, "L": L, "r": r, "index": smp["index"], "score": sc, "pred": pred,
                                    "refs": smp["outputs"], "n_prompt_tokens": int(ids.shape[1]),
                                    "seg": a.seg, "seconds": round(time.time() - t1, 2)}) + "\n")
                f.flush()
            if scores:
                print(f"[probe1r w{a.worker}] {task}@{L} r={r}: {100 * np.mean(scores):.1f} over {len(scores)} "
                      f"samples, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------- docs
def cmd_docs(a):
    from datasets import load_from_disk
    import pyarrow.compute as pc
    from transformers import AutoTokenizer
    ds = load_from_disk(a.dataset)
    L = np.concatenate([pc.list_value_length(c).to_numpy() for c in ds.data.column("input_ids").chunks])
    idx = np.where(L >= T + 1)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    tok = AutoTokenizer.from_pretrained("models/Qwen3.5-0.8B")
    chosen, info = [], []
    for i in idx:                     # skip degenerate pages: over-compressible text or repeated boilerplate
        ids = ds[int(i)]["input_ids"][:T]
        txt = tok.decode(ids)
        cr = len(zlib.compress(txt.encode())) / len(txt.encode())
        uniq = len(set(zip(ids, ids[1:]))) / len(ids)
        if cr > 0.30 and uniq > 0.45:
            chosen.append(int(i))
            info.append({"index": int(i), "tokens": int(L[i]), "zlib_ratio": round(cr, 3), "bigram_uniq": round(uniq, 3),
                         "head": txt[:80]})
        if len(chosen) == a.n:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": a.dataset, "index": chosen, "info": info}, open(OUT / "docs.json", "w"), indent=1)
    print(f"[docs] {len(chosen)} single documents >= {T + 1} tokens from {a.dataset}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("docs"); d.add_argument("--dataset", default="data/text/dclm-10shard/train"); d.add_argument("--n", type=int, default=40)
    p1 = sub.add_parser("probe1"); p1.add_argument("--worker", type=int, default=0); p1.add_argument("--workers", type=int, default=1)
    p1.add_argument("--n_seq", type=int, default=8)
    p2 = sub.add_parser("probe2"); p2.add_argument("--n_seq", type=int, default=8)
    pr = sub.add_parser("probe1r"); pr.add_argument("--worker", type=int, default=0); pr.add_argument("--workers", type=int, default=1)
    pr.add_argument("--r", default=",".join(map(str, R_RETR)), help="keep_top ranks, comma separated (128 = baseline)")
    pr.add_argument("--jobs", default="", help="override RETR_JOBS: task:length:n_samples,... (default: the built-in list)")
    pr.add_argument("--seg", type=int, default=SEG)
    pr.add_argument("--ckpt", default="", help="native checkpoint dir (model.pt) to probe instead of the unmodified backbone")
    pr.add_argument("--tag", default="", help="suffix for the rows files, e.g. 'control' -> p1r_control_rows_w*.jsonl")
    sub.add_parser("report")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "report":
        import state_probes_report
        state_probes_report.main()
    else:
        {"docs": cmd_docs, "probe1": cmd_probe1, "probe2": cmd_probe2, "probe1r": cmd_probe1r}[a.cmd](a)
