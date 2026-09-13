"""Batch > 1 correctness (one GPU, ~1 min), in float32 so kernel rounding cannot hide a padding bug:
right-padded chunked loss == per-example dense loss, batched greedy generation == per-row generation,
HF forward on a right-padded batch == per-row logits."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import linswap  # noqa: E402,F401
from linswap import DEFAULT_BASE_MODEL_DIR, build_model  # noqa: E402
from linswap.hf import LinearSwapForCausalLM  # noqa: E402
from linswap.sft_utils import chunked_cross_entropy_eval, chunked_cross_entropy_with_backward, collate_fn  # noqa: E402


def main():
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL_DIR)
    texts = ["The quick brown fox jumps over the lazy dog and keeps running through the forest.",
             "In 1815 the Congress of Vienna redrew the map of Europe."]
    exs = []
    for t in texts:
        ids = tok(t).input_ids
        exs.append({"input_ids": ids, "labels": ids})
    exs.sort(key=lambda e: -len(e["input_ids"]))  # row 0 longest, row 1 right-padded
    batch = {k: v.to(device) for k, v in collate_fn(exs).items()}
    assert batch["input_ids"][1, -1].item() == 0 and batch["labels"][1, -1].item() == -100, "right padding expected"
    model = build_model("kda", device=device, dtype=torch.float32)
    ok = True

    # 1. loss: batched chunked (with backward) vs per-example dense
    model.train()
    hidden = model(batch["input_ids"], return_hidden=True)
    loss_b = chunked_cross_entropy_with_backward(hidden, batch["labels"], model.lm_head, chunk_size=7)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    tot, cnt = 0.0, 0
    for ex in exs:
        ids = torch.tensor([ex["input_ids"]], device=device)
        logits = model(ids).float()
        l = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")
        tot += l.item(); cnt += ids.shape[1] - 1
        (l / (sum(len(e["input_ids"]) - 1 for e in exs))).backward()
    ref = tot / cnt
    num = sum(((grads[n] - p.grad) ** 2).sum() for n, p in model.named_parameters() if p.grad is not None)
    den = sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)
    gdiff = (num / den).sqrt().item()
    print(f"batched loss {loss_b:.6f} vs per-example {ref:.6f} | rel grad-norm diff {gdiff:.2e}")
    ok &= abs(loss_b - ref) < 1e-3 and gdiff < 1e-2   # fp32 GEMM shape nondeterminism is ~1e-4; a padding bug would be O(1)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        e = chunked_cross_entropy_eval(model(batch["input_ids"], return_hidden=True), batch["labels"], model.lm_head)
    print(f"eval loss batched {e:.6f}")
    ok &= abs(e - ref) < 1e-3

    # 2. generation: batch of equal-length prompts vs per-row
    with torch.no_grad():
        p = torch.tensor([exs[0]["input_ids"][:12], exs[1]["input_ids"][:12]], device=device)
        gb = model.generate(p, max_new_tokens=8, eos_token_id=tok.eos_token_id)
        g0 = model.generate(p[:1], max_new_tokens=8, eos_token_id=tok.eos_token_id)
        g1 = model.generate(p[1:], max_new_tokens=8, eos_token_id=tok.eos_token_id)
    same = torch.equal(gb[0], g0[0]) and torch.equal(gb[1], g1[0])
    print(f"batched generate == per-row: {same} | max_new_tokens=0 returns input: {model.generate(p, 0).shape == p.shape}")
    ok &= same

    # 3. HF forward on a right-padded batch vs per-row
    hf = LinearSwapForCausalLM.from_swap("kda", device=device, dtype=torch.float32).eval()
    am = (batch["labels"] != -100).long()
    with torch.no_grad():
        out = hf(batch["input_ids"], attention_mask=am, use_cache=False).logits
        r0 = hf(batch["input_ids"][:1], use_cache=False).logits
        n1 = int(am[1].sum()); r1 = hf(batch["input_ids"][1:, :n1], use_cache=False).logits
    d_hf = max((out[0] - r0[0]).abs().max().item(), (out[1, :n1] - r1[0]).abs().max().item())
    same_hf = d_hf < 1e-2
    print(f"HF right-padded batch vs per-row logits: max diff {d_hf:.2e} (fp32)")
    try:
        hf(batch["input_ids"], attention_mask=am.flip(1), use_cache=False); left_ok = False
    except ValueError:
        left_ok = True
    print(f"left padding rejected: {left_ok}")
    ok &= same_hf and left_ok
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
