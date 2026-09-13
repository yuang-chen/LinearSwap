"""Stage 2 — posttrain: SFT of a swapped model (data preparation included).

    python linswap.py posttrain --kernel kda                       # gate_only then full, standard recipe
    python linswap.py posttrain --kernel kda --modes full --full_steps 200 --full_lr 1e-4

Recipe (identical for every kernel): bf16, gradient checkpointing, chunked
cross-entropy, micro-batch 1 × ``--grad_accum_steps``, AdamW, grad-clip 1.0.
``gate_only`` trains only the kernel's ``new_param_names``; ``full`` trains
everything (optionally after ``--warmup_gate_steps`` gate-only steps).  The SFT
data is prepared on first use.  Checkpoints go to
``outputs/<kernel>/sft_<mode>/checkpoint-N`` (``model.pt``, ``optimizer.pt``,
``config.json`` recording the kernel) with ``train_log.jsonl`` next to them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data import DEFAULT_DATA_DIR, ensure_sft_data
from ..load_weights import DEFAULT_BASE_MODEL_DIR, REPO_ROOT, build_model
from ..registry import get_kernel
from ..sft_utils import TruncatedDataset, collate_fn, compute_loss, evaluate, find_latest_checkpoint

DEFAULT_OUTPUTS = REPO_ROOT / "outputs"


def add_args(ap):
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--modes", default="gate_only,full", help="comma list of gate_only / full")
    ap.add_argument("--output_dir", default=None, help="default outputs/<kernel>")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--init_ckpt", default=None, help="start from this checkpoint (e.g. a distill checkpoint)")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--data_max_length", type=int, default=262144, help="length the SFT data is prepared at")
    ap.add_argument("--max_length", type=int, default=131072, help="left-truncate examples to this many tokens")
    ap.add_argument("--gate_steps", type=int, default=100)
    ap.add_argument("--gate_lr", type=float, default=2e-4)
    ap.add_argument("--full_steps", type=int, default=50)
    ap.add_argument("--full_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_gate_steps", type=int, default=0, help="full mode: gate-only optimizer steps first")
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_batches", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--save_optimizer", type=int, default=1)
    ap.add_argument("--max_hours", type=float, default=None, help="stop (and save) after this wall time")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce_chunk_size", type=int, default=2048)


def checkpoint_config(model, args, mode):
    cfg = {k: (str(v) if k == "dtype" else v) for k, v in model.cfg.items()}
    cfg.update({
        "linear_kernel": model.kernel.name,
        "base_model_dir": str(args.base_model_dir),
        "sft_mode": mode,
        "new_param_names": list(model.kernel.new_param_names),
    })
    return cfg


def save_checkpoint(model, optimizer, step, output_dir, args, mode):
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    if args.save_optimizer:
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, ckpt_dir / "optimizer.pt")
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(checkpoint_config(model, args, mode), f, indent=1)
    return ckpt_dir


def log_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def train_one(args, mode, data_dir, out_dir) -> Path:
    """Run one SFT mode; returns the final checkpoint directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_steps = args.gate_steps if mode == "gate_only" else args.full_steps
    with open(out_dir / "args.json", "w") as f:
        json.dump({**vars(args), "mode": mode, "num_steps": num_steps, "data_dir": str(data_dir)}, f, indent=1)
    log_path = out_dir / "train_log.jsonl"

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    spec = get_kernel(args.kernel)
    print(f"[posttrain] kernel={spec.name} mode={mode} steps={num_steps} -> {out_dir}")
    model = build_model(args.kernel, base_model_dir=args.base_model_dir, device=device,
                        ckpt_dir=args.init_ckpt)
    if args.init_ckpt:
        print(f"  initialised from {args.init_ckpt}")
    model.train()
    model.gradient_checkpointing = True

    gate_params = [p for _, p in model.new_parameters()]
    gate_ids = {id(p) for p in gate_params}
    backbone_params = [p for p in model.parameters() if id(p) not in gate_ids]
    if mode == "gate_only":
        for p in backbone_params:
            p.requires_grad_(False)
        param_groups = [{"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0, "is_gate": True}]
    else:
        param_groups = [
            {"params": gate_params, "lr": args.full_lr, "weight_decay": 0.0, "is_gate": True},
            {"params": backbone_params, "lr": args.full_lr, "weight_decay": args.weight_decay, "is_gate": False},
        ]
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"  trainable params {n_train/1e6:.2f}M (gate/new {sum(p.numel() for p in gate_params)/1e6:.2f}M, "
          f"new_param_names={spec.new_param_names})")
    optimizer = AdamW(param_groups)

    global_step = 0
    latest = find_latest_checkpoint(out_dir)
    if latest is not None:
        print(f"  resuming from {latest.name}")
        model.load_state_dict(torch.load(latest / "model.pt", map_location="cpu", weights_only=True))
        if (latest / "optimizer.pt").exists():
            opt_state = torch.load(latest / "optimizer.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(opt_state["optimizer"])
            global_step = opt_state["step"]
        else:
            global_step = int(latest.name.split("-")[1])

    train_ds = TruncatedDataset(load_from_disk(Path(data_dir) / "train"), args.max_length)
    val_ds = TruncatedDataset(load_from_disk(Path(data_dir) / "validation"), args.max_length)
    g = torch.Generator().manual_seed(args.seed + global_step)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    train_iter = iter(train_loader)

    if global_step == 0 and args.eval_batches > 0:
        val_loss = evaluate(model, val_loader, max_batches=args.eval_batches, chunk_size=args.ce_chunk_size)
        print(f"  step 0: val_loss={val_loss:.4f}")
        log_jsonl(log_path, {"step": 0, "val_loss": val_loss})
        torch.cuda.empty_cache()

    start = time.time()
    stop_early = False
    final_ckpt = latest
    while global_step < num_steps and not stop_early:
        if mode == "full" and args.warmup_gate_steps > 0:
            for grp in optimizer.param_groups:
                if not grp["is_gate"]:
                    grp["lr"] = 0.0 if global_step < args.warmup_gate_steps else args.full_lr
                else:
                    grp["lr"] = args.gate_lr if global_step < args.warmup_gate_steps else args.full_lr

        accum_loss, n_tokens = 0.0, 0
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            n_tokens += int((batch["labels"] != -100).sum())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compute_loss(model, batch, chunk_size=args.ce_chunk_size, do_backward=True,
                                    loss_scale=1.0 / args.grad_accum_steps)
            accum_loss += loss

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for grp in optimizer.param_groups for p in grp["params"]], args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        elapsed = time.time() - start
        rec = {"step": global_step, "loss": accum_loss / args.grad_accum_steps, "grad_norm": float(grad_norm),
               "label_tokens": n_tokens, "elapsed_s": round(elapsed, 1),
               "lr": [grp["lr"] for grp in optimizer.param_groups]}
        print(f"  step {global_step}: loss={rec['loss']:.4f} grad_norm={rec['grad_norm']:.3f} "
              f"label_tokens={n_tokens} elapsed={elapsed/60:.1f}min", flush=True)
        log_jsonl(log_path, rec)

        if args.max_hours is not None and elapsed > args.max_hours * 3600:
            print(f"  reached max_hours={args.max_hours}; stopping.")
            stop_early = True
        last = stop_early or global_step == num_steps
        if global_step % args.eval_every == 0 or last:
            val_loss = evaluate(model, val_loader, max_batches=args.eval_batches, chunk_size=args.ce_chunk_size)
            print(f"  step {global_step}: val_loss={val_loss:.4f}", flush=True)
            log_jsonl(log_path, {"step": global_step, "val_loss": val_loss})
            torch.cuda.empty_cache()
        if global_step % args.save_every == 0 or last:
            final_ckpt = save_checkpoint(model, optimizer, global_step, out_dir, args, mode)
            print(f"  saved {final_ckpt}", flush=True)
            torch.cuda.empty_cache()

    print(f"[posttrain] {mode} complete at step {global_step}: {final_ckpt}")
    del model, optimizer
    torch.cuda.empty_cache()
    return final_ckpt


def main(args) -> list[Path]:
    get_kernel(args.kernel)  # validate early
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("gate_only", "full"):
            raise SystemExit(f"posttrain: unknown mode {m!r}")
    data_dir = ensure_sft_data(args.base_model_dir, args.data_max_length, args.data_dir)
    out_root = (Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUTS / args.kernel).resolve()
    ckpts = [train_one(args, mode, data_dir, out_root / f"sft_{mode}") for mode in modes]
    print("[posttrain] checkpoints: " + ", ".join(str(c) for c in ckpts))
    return ckpts
