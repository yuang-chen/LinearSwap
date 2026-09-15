"""Chunked losses vs a dense autograd reference (CPU, no model, ~1 s).

``chunked_cross_entropy_with_backward`` and ``chunked_kl_with_backward`` compute the loss in
chunks and run the backward themselves, so no framework checks them.  The original SFT loss
scaled the LM-head gradient by the token *sum* and the hidden-state gradient by the token
*mean*; the LM head is tied to the embedding, so the tied matrix took a step several orders of
magnitude too large and the pre-clip gradient norm sat near 1000 (docs/framework.md).  These
checks reproduce that setting -- tied head, masked prompt, right padding, uneven chunks -- and
compare loss, hidden-path gradient and tied-matrix gradient against dense autograd.

Parameters are float64 while both implementations upcast logits with ``.float()``, so agreement
is bounded by float32 rounding (~1e-7), far below any scaling mistake.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linswap.pipeline.distill import chunked_kl_with_backward  # noqa: E402
from linswap.sft_utils import chunked_cross_entropy_eval, chunked_cross_entropy_with_backward  # noqa: E402

B, T, D, V = 2, 9, 5, 13
CHUNK = 4          # does not divide B * (T - 1) = 16
RTOL = 1e-5        # float32 logits; the bug this guards against was a factor of ~1e5


def make_model(seed=0):
    """Two-layer stand-in with the LM head tied to the embedding, as in Qwen3.5-0.8B."""
    torch.manual_seed(seed)
    emb = nn.Embedding(V, D)
    proj = nn.Linear(D, D, bias=False)
    lm_head = nn.Linear(D, V, bias=False)
    lm_head.weight = emb.weight
    return emb, proj, lm_head


def fixture():
    torch.manual_seed(1)
    ids = torch.randint(0, V, (B, T))
    labels = ids.clone()
    labels[0, :3] = -100      # prompt positions
    labels[1, -4:] = -100     # right padding
    return ids, labels


def close(a, b, what, results):
    d = (a - b).norm() / max(b.norm().item(), 1e-30)
    ok = d < RTOL
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: rel diff {d:.2e}")
    results.append(ok)


def check_cross_entropy(results):
    print("cross-entropy vs dense autograd (tied head, masked labels, uneven chunks)")
    ids, labels = fixture()

    emb, proj, head = make_model()
    hidden = proj(emb(ids))
    logits = head(hidden)
    ref_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
    ref_loss.backward()
    ref_tied, ref_proj = emb.weight.grad.clone(), proj.weight.grad.clone()

    emb, proj, head = make_model()
    hidden = proj(emb(ids))
    got_loss = chunked_cross_entropy_with_backward(hidden, labels, head, chunk_size=CHUNK)
    print(f"  {'ok  ' if abs(got_loss - ref_loss.item()) < RTOL else 'FAIL'} "
          f"loss: dense {ref_loss.item():.9f} chunked {got_loss:.9f}")
    results.append(abs(got_loss - ref_loss.item()) < RTOL)
    close(emb.weight.grad, ref_tied, "tied embedding / LM head gradient", results)
    close(proj.weight.grad, ref_proj, "hidden-path gradient", results)

    # chunk size must not change the answer
    for cs in (1, 3, 16, 64):
        emb, proj, head = make_model()
        loss = chunked_cross_entropy_with_backward(proj(emb(ids)), labels, head, chunk_size=cs)
        close(emb.weight.grad, ref_tied, f"tied gradient, chunk_size={cs}", results)
        results.append(abs(loss - ref_loss.item()) < RTOL)

    # loss_scale scales the gradients, not the reported loss
    emb, proj, head = make_model()
    scaled = chunked_cross_entropy_with_backward(proj(emb(ids)), labels, head, chunk_size=CHUNK, loss_scale=0.5)
    print(f"  {'ok  ' if abs(scaled - ref_loss.item()) < RTOL else 'FAIL'} loss_scale leaves the reported loss alone")
    results.append(abs(scaled - ref_loss.item()) < RTOL)
    close(emb.weight.grad, 0.5 * ref_tied, "loss_scale=0.5 halves the tied gradient", results)

    # eval path
    emb, proj, head = make_model()
    with torch.no_grad():
        hidden = proj(emb(ids))
        mean = chunked_cross_entropy_eval(hidden, labels, head, chunk_size=CHUNK)
        total, count = chunked_cross_entropy_eval(hidden, labels, head, chunk_size=CHUNK, return_sum=True)
    n_sup = int((labels[:, 1:] != -100).sum())
    ok = abs(mean - ref_loss.item()) < RTOL and count == n_sup and abs(total / count - ref_loss.item()) < RTOL
    print(f"  {'ok  ' if ok else 'FAIL'} eval path: mean {mean:.9f}, sum/count {total:.4f}/{count} "
          f"(supervised tokens {n_sup})")
    results.append(ok)

    # a microbatch with nothing supervised must be a no-op, not a NaN
    emb, proj, head = make_model()
    dead = chunked_cross_entropy_with_backward(proj(emb(ids)), torch.full_like(labels, -100), head, chunk_size=CHUNK)
    ok = dead == 0.0 and emb.weight.grad.abs().max() == 0.0
    print(f"  {'ok  ' if ok else 'FAIL'} all-masked microbatch: loss {dead}, zero gradient")
    results.append(ok)


def check_kl(results):
    print("distillation KL vs dense autograd (temperature 2.0, CE term on)")
    ids, labels = fixture()
    temperature, ce_weight = 2.0, 0.5
    t_emb, t_proj, t_head = make_model(seed=7)          # teacher: frozen
    for p in list(t_emb.parameters()) + list(t_proj.parameters()):
        p.requires_grad_(False)
    with torch.no_grad():
        t_hidden = t_proj(t_emb(ids))
        t_logp = F.log_softmax(t_head(t_hidden).float() / temperature, dim=-1)

    emb, proj, head = make_model()
    s_hidden = proj(emb(ids))
    s_logits = head(s_hidden).float()
    s_logp = F.log_softmax(s_logits / temperature, dim=-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).reshape(-1)
    ref_kl = kl.mean()
    flat_labels = labels.reshape(-1)
    n_lab = int((flat_labels != -100).sum())
    ref_ce = F.cross_entropy(s_logits.reshape(-1, V), flat_labels, ignore_index=-100, reduction="sum") / n_lab
    (ref_kl * temperature ** 2 + ce_weight * ref_ce).backward()
    ref_tied, ref_proj = emb.weight.grad.clone(), proj.weight.grad.clone()

    emb, proj, head = make_model()
    got_kl, got_ce = chunked_kl_with_backward(proj(emb(ids)), t_hidden, head, t_head, chunk_size=CHUNK,
                                              temperature=temperature, labels=labels, ce_weight=ce_weight)
    ok = abs(got_kl - ref_kl.item()) < RTOL and abs(got_ce - ref_ce.item()) < RTOL
    print(f"  {'ok  ' if ok else 'FAIL'} loss: dense KL {ref_kl.item():.9f} / CE {ref_ce.item():.9f}, "
          f"chunked KL {got_kl:.9f} / CE {got_ce:.9f}")
    results.append(ok)
    close(emb.weight.grad, ref_tied, "tied embedding / LM head gradient", results)
    close(proj.weight.grad, ref_proj, "hidden-path gradient", results)


def main():
    torch.set_default_dtype(torch.float64)
    results = []
    check_cross_entropy(results)
    check_kl(results)
    ok = all(results)
    print(f"{'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)} checks)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
