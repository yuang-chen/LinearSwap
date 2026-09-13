"""Stage — export: write a swapped model as a Hugging Face checkpoint.

    linswap export --kernel kda --out hf/Qwen3.5-0.8B-KDA                 # base swap
    linswap export --ckpt outputs/kda/sft_full/checkpoint-50 --out hf/Qwen3.5-0.8B-KDA-sft

The result loads with ``AutoModelForCausalLM.from_pretrained`` after ``import linswap``
and can be pushed to the Hub with ``huggingface-cli upload`` / ``push_to_hub``.
"""

from ..load_weights import DEFAULT_BASE_MODEL_DIR, read_checkpoint_kernel


def add_args(ap):
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--ckpt", default=None, help="checkpoint dir (kernel read from its config.json)")
    ap.add_argument("--out", required=True, help="output directory (HF checkpoint)")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--no_tokenizer", action="store_true")


def main(args):
    from ..hf import export

    kernel = args.kernel or (read_checkpoint_kernel(args.ckpt) if args.ckpt else None)
    if kernel is None:
        raise SystemExit("export: --kernel or --ckpt is required")
    out = export(kernel, args.out, base_model_dir=args.base_model_dir, ckpt_dir=args.ckpt, save_tokenizer=not args.no_tokenizer)
    print(f"[export] wrote {out}")
    return out
