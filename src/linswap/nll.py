"""Token-weighted raw-text negative log-likelihood on held-out corpora, binned by position.

    nll = raw_text_nll(model, tokenizer, docs="pg19", n_docs=20, max_length=131072)
    -> {"nll": 2.31, "tokens": 2_500_000, "bins": {"0-4k": 2.5, "4k-16k": 2.3, "16k-64k": 2.2, "64k-128k": 2.2}}

Corpora (downloaded with `datasets` on first use):
  pg19      deepmind/pg19 test split — long books, one document per book, truncated to max_length
  wikitext  wikitext-103-raw-v1 test split — the concatenated test text split into max_length segments
Unlike the SFT validation loss (assistant tokens of the fine-tuning mixture, macro-averaged),
this is corpus-level next-token NLL on text the models were never fine-tuned on.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import HF_CACHE_DIR

BINS = [(0, 4096, "0-4k"), (4096, 16384, "4k-16k"), (16384, 65536, "16k-64k"), (65536, 1 << 30, "64k-128k")]


def load_docs(name: str, n_docs: int, tokenizer, max_length: int, seed: int = 0) -> list[list[int]]:
    from datasets import load_dataset

    if name == "pg19":
        # deepmind/pg19 is a script dataset (unsupported by datasets>=4); use its parquet conversion, else a mirror.
        try:
            ds = load_dataset("deepmind/pg19", revision="refs/convert/parquet", split="test", streaming=True)
        except Exception:
            ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        texts = [ex["text"] for _, ex in zip(range(n_docs), ds)]
        return [tokenizer(t, add_special_tokens=False).input_ids[:max_length] for t in texts]
    if name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", cache_dir=str(HF_CACHE_DIR))
        ids = tokenizer("".join(ds["text"]), add_special_tokens=False).input_ids
        segs = [ids[i:i + max_length] for i in range(0, len(ids), max_length)]
        return [s for s in segs if len(s) > 1024][:n_docs]
    raise ValueError(f"unknown corpus {name!r} (pg19 | wikitext)")


@torch.no_grad()
def raw_text_nll(model, tokenizer, docs: str | list, n_docs: int = 20, max_length: int = 131072,
                 chunk_size: int = 2048) -> dict:
    device = next(model.parameters()).device
    seqs = load_docs(docs, n_docs, tokenizer, max_length) if isinstance(docs, str) else docs
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    bin_tot = {b[2]: torch.zeros((), device=device, dtype=torch.float64) for b in BINS}
    bin_cnt = {b[2]: 0 for b in BINS}
    was_training = model.training
    model.eval()
    for ids in seqs:
        ids = torch.tensor([ids], device=device)
        hidden = model(ids, return_hidden=True)[0, :-1]
        labels = ids[0, 1:]
        for start in range(0, hidden.shape[0], chunk_size):
            end = min(start + chunk_size, hidden.shape[0])
            logits = F.linear(hidden[start:end], model.lm_head.weight).float()
            nll = F.cross_entropy(logits, labels[start:end], reduction="none")
            total += nll.sum().double()
            count += nll.numel()
            pos = torch.arange(start + 1, end + 1, device=device)  # position of the predicted token
            for lo, hi, name in BINS:
                m = (pos >= lo) & (pos < hi)
                if m.any():
                    bin_tot[name] += nll[m].sum().double()
                    bin_cnt[name] += int(m.sum())
    if was_training:
        model.train()
    return {"nll": (total / max(count, 1)).item(), "tokens": count,
            "bins": {k: (bin_tot[k] / bin_cnt[k]).item() for k in bin_tot if bin_cnt[k] > 0},
            "docs": len(seqs)}
