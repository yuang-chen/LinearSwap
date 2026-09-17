#!/usr/bin/env python
"""Prefill and decode throughput of swapped models: one GPU, bf16, cached greedy decode.

    python tools/throughput.py --models gdn rwkv7-radlads=outputs/radlads/rwkv7/distill/checkpoint-N --lengths 8192,32768
"""
import argparse, sys, time
import torch
sys.path.insert(0, "src")
from linswap import build_model
from linswap.pipeline.evaluate import resolve_model

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--lengths", default="8192,32768")
ap.add_argument("--new_tokens", type=int, default=256)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--base_model_dir", default="models/Qwen3.5-0.8B")
args = ap.parse_args()
rows = []
for spec in args.models:
    disp, kernel, base, ckpt = resolve_model(spec, args.base_model_dir)
    m = build_model(kernel, base_model_dir=base, ckpt_dir=ckpt, device="cuda").eval()
    for L in (int(x) for x in args.lengths.split(",")):
        ids = torch.randint(100, 1000, (args.batch, L), device="cuda")
        with torch.no_grad():
            m.generate(ids[:, :256], max_new_tokens=4, use_cache=True)          # warm-up / compile
            torch.cuda.synchronize(); t0 = time.time()
            m.generate(ids, max_new_tokens=1, use_cache=True)                    # prefill only
            torch.cuda.synchronize(); t_pre = time.time() - t0
            torch.cuda.synchronize(); t0 = time.time()
            m.generate(ids, max_new_tokens=args.new_tokens, use_cache=True)     # prefill + decode
            torch.cuda.synchronize(); t_all = time.time() - t0
        dec_ms = 1000 * (t_all - t_pre) / (args.new_tokens - 1)
        rows.append((disp, L, args.batch * L / t_pre, dec_ms, torch.cuda.max_memory_allocated() / 2**30))
        print(f"{disp:24s} L={L:6d}  prefill {args.batch*L/t_pre/1e3:8.1f}K tok/s   decode {dec_ms:6.1f} ms/token   peak {rows[-1][4]:.1f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats()
    del m; torch.cuda.empty_cache()
print("\n| model | length | prefill tok/s | decode ms/token | peak GiB |\n|---|---|---|---|---|")
for d, L, pre, dec, mem in rows:
    print(f"| {d} | {L} | {pre/1e3:.1f}K | {dec:.1f} | {mem:.1f} |")
