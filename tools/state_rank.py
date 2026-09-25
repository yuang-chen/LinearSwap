#!/usr/bin/env python
"""Singular values of every linear-attention memory state S (key_dim x value_dim per head) along a prefill.

Prefills B sequences of packed DCLM validation text in chunks and, after each chunk, stores the singular
values of each head's recurrent state.  `tools/state_rank_report.py` turns the dumps into ranks.

    python tools/state_rank.py --name gdn                                             # the unmodified backbone
    python tools/state_rank.py --name gdn2 --ckpt outputs/gdn2/distill/checkpoint-16338
"""
import argparse, json
from pathlib import Path

import torch
from datasets import load_from_disk

from linswap import build_model
from linswap.model import SwapCache

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--ckpt", default=None, help="distilled checkpoint; omit for the unmodified backbone")
p.add_argument("--out", default="outputs/state_rank")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--marks", default="64,128,256,1024,4096,16384", help="prefix lengths at which to read the state")
p.add_argument("--data", default="data/text/dclm-10shard/validation")
a = p.parse_args()
marks = [int(m) for m in a.marks.split(",")]

docs = load_from_disk(a.data)["input_ids"]
flat = torch.tensor([t for doc in docs for t in doc])
ids = flat[: a.batch * marks[-1]].view(a.batch, marks[-1]).cuda()

model = build_model("gdn" if a.ckpt is None else None, ckpt_dir=a.ckpt).eval()
lin = [i for i, l in enumerate(model.model.layers) if l.layer_type != "full_attention"]
cache = SwapCache(len(model.model.layers))
res = {"name": a.name, "ckpt": a.ckpt, "layers": lin, "marks": marks, "sv": {}}
pos = 0
with torch.no_grad():
    for m in marks:
        model(ids[:, pos:m], cache=cache, use_cache=True, last_logits_only=True)
        pos = m
        for i in lin:
            state = cache.linear_cache[i]["recurrent_state"]            # [B, H, K, V]; FLA indexes by global layer id
            res["sv"][f"{i}@{m}"] = torch.linalg.svdvals(state.float()).cpu().tolist()
        print(f"[state_rank] {a.name}: {m} tokens", flush=True)
Path(a.out).mkdir(parents=True, exist_ok=True)
json.dump(res, open(Path(a.out) / f"{a.name}.json", "w"))
