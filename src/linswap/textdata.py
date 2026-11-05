"""Generic-text corpora for the distillation steps.

Distillation runs on generic web text (DCLM by default, FineWeb-Edu as an alternative) packed into
fixed-length sequences — no instruction data and no chat template.  This module prepares such a corpus
once per tokenizer: documents are tokenised, an EOS token is appended to each, and the result is saved
as an HF ``datasets`` directory with ``train`` / ``validation`` splits of ``{"input_ids"}``
documents that ``train_utils.PackedDataset`` packs on the fly.

The directory is named after the corpus and its shard count only, so ``info.json`` records which
tokenizer produced it (``tokenizer_sha256``) and ``ensure_text_data`` refuses to hand token ids to a
backbone with a different one.
"""

import json
import os
import warnings
from pathlib import Path

from .manifest import tokenizer_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = REPO_ROOT / "data" / "hf_cache"

DEFAULT_TEXT_DIR = REPO_ROOT / "data" / "text"

CORPORA = {
    # name: (HF dataset repo, shard path pattern, text column)
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/{:03d}_00000.parquet", "text"),
    # DCLM-baseline; one shard ≈ 200 MB ≈ 110M tokens
    "dclm": ("mlfoundations/dclm-baseline-1.0-parquet",
             "filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/"
             "processed_data/global-shard_01_of_10/local-shard_0_of_10/shard_{:08d}_processed.parquet", "text"),
}


def text_data_dir(name, shards=1, out_dir=DEFAULT_TEXT_DIR) -> Path:
    return Path(out_dir) / f"{name}-{shards}shard"


def prepare_text_data(base_model_dir, name="dclm", shards=1, out_dir=DEFAULT_TEXT_DIR, val_docs=512,
                      num_proc=32, seed=0) -> Path:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    repo, pattern, col = CORPORA[name]
    files = [hf_hub_download(repo, pattern.format(i), repo_type="dataset", cache_dir=str(HF_CACHE_DIR))
             for i in range(shards)]
    tok = AutoTokenizer.from_pretrained(base_model_dir)
    eos = tok.eos_token_id
    ds = load_dataset("parquet", data_files=files, cache_dir=str(HF_CACHE_DIR))["train"]
    ds = ds.select_columns([col]).shuffle(seed=seed)

    def tokenize(batch):
        return {"input_ids": [ids + [eos] for ids in tok(batch[col], add_special_tokens=False)["input_ids"]]}

    ds = ds.map(tokenize, batched=True, num_proc=num_proc, remove_columns=[col], desc="tokenize")
    out = text_data_dir(name, shards, out_dir)
    ds.select(range(val_docs)).save_to_disk(str(out / "validation"))
    ds.select(range(val_docs, len(ds))).save_to_disk(str(out / "train"))
    n_tok = sum(len(x) for x in ds["input_ids"])
    with open(out / "info.json", "w") as f:
        json.dump({"corpus": name, "repo": repo, "shards": shards, "docs": len(ds), "tokens": n_tok,
                   "tokenizer": str(base_model_dir), "tokenizer_sha256": tokenizer_fingerprint(base_model_dir),
                   "validation_docs": val_docs, "shuffle_seed": seed}, f, indent=1)
    print(f"[textdata] {name}: {len(ds)} docs, {n_tok/1e6:.0f}M tokens -> {out}")
    return out


def ensure_text_data(base_model_dir, name="dclm", shards=1, out_dir=DEFAULT_TEXT_DIR) -> Path:
    out = text_data_dir(name, shards, out_dir)
    if (out / "info.json").exists():
        check_tokenizer(out, base_model_dir)
        return out
    return prepare_text_data(base_model_dir, name, shards, out_dir)


def check_tokenizer(data_dir, base_model_dir) -> None:
    """Stop if the corpus in ``data_dir`` was tokenised with another tokenizer than ``base_model_dir``'s.

    A corpus prepared before fingerprints were recorded names its tokenizer by path only: the fingerprint
    is then taken from that path (and written back when it matches), and a path that no longer exists
    leaves the corpus unverifiable, which is reported and let through."""
    data_dir = Path(data_dir)
    with open(data_dir / "info.json") as f:
        info = json.load(f)
    want = tokenizer_fingerprint(base_model_dir)
    have, backfill = info.get("tokenizer_sha256"), False
    if have is None and info.get("tokenizer"):
        recorded = Path(info["tokenizer"])
        for cand in (recorded, REPO_ROOT / recorded):
            if cand.is_dir():
                have, backfill = tokenizer_fingerprint(cand), True
                break
    if have is None or want is None:
        warnings.warn(f"{data_dir}: cannot check that it was tokenised for {base_model_dir} "
                      f"(recorded tokenizer: {info.get('tokenizer')!r})")
        return
    if have != want:
        raise SystemExit(f"[textdata] {data_dir} was tokenised with {info.get('tokenizer')!r} "
                         f"(sha256 {have[:12]}), not with the tokenizer of {base_model_dir} (sha256 {want[:12]}): "
                         f"its token ids mean something else for this backbone.  Pass another --text_dir, or "
                         f"remove the directory to tokenise the corpus again.")
    if backfill:
        info["tokenizer_sha256"] = have
        tmp = data_dir / f"info.json.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(info, f, indent=1)
            os.replace(tmp, data_dir / "info.json")
        except OSError:
            tmp.unlink(missing_ok=True)       # a read-only corpus stays as it is
