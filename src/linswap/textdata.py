"""Generic-text corpora for the distillation stages.

The Transformer→hybrid conversion literature (RADLADS, HALO, Retrieval-Aware Distillation,
"What matters in linearizing LMs") distils on generic web text (DCLM / FineWeb-Edu) packed into
fixed-length sequences, not on chat data.  This module prepares such a corpus once per
tokenizer: documents are tokenised, an EOS token is appended to each, and the result is saved
as an HF ``datasets`` directory with ``train`` / ``validation`` splits of ``{"input_ids"}``
documents that ``sft_utils.PackedDataset`` packs on the fly.
"""

import json
from pathlib import Path

from .data import HF_CACHE_DIR, REPO_ROOT

DEFAULT_TEXT_DIR = REPO_ROOT / "data" / "text"

CORPORA = {
    # name: (HF dataset repo, shard path pattern, text column)
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/{:03d}_00000.parquet", "text"),
}


def text_data_dir(name, shards=1, out_dir=DEFAULT_TEXT_DIR) -> Path:
    return Path(out_dir) / f"{name}-{shards}shard"


def prepare_text_data(base_model_dir, name="fineweb-edu", shards=1, out_dir=DEFAULT_TEXT_DIR, val_docs=512,
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
                   "tokenizer": str(base_model_dir), "validation_docs": val_docs}, f, indent=1)
    print(f"[textdata] {name}: {len(ds)} docs, {n_tok/1e6:.0f}M tokens -> {out}")
    return out


def ensure_text_data(base_model_dir, name="fineweb-edu", shards=1, out_dir=DEFAULT_TEXT_DIR) -> Path:
    out = text_data_dir(name, shards, out_dir)
    if (out / "info.json").exists():
        return out
    return prepare_text_data(base_model_dir, name, shards, out_dir)
