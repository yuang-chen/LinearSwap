"""Long-context SFT data: LongAlign-10k + LongAlpaca-12k + anti-haystack.

Each example is rendered with the backbone's chat format (`<|im_start|>role ... <|im_end|>`), tokenised, non-assistant
tokens are masked with -100 and the sequence is left-truncated to
``max_length`` (so the answer tail survives).  The three sets are concatenated,
shuffled (seed 42) and split 98/2 into ``{output_dir}/len{max_length}/{train,validation}``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "sft"
HF_CACHE_DIR = REPO_ROOT / "data" / "hf_cache"

DATASETS = {
    "longalign": ("zai-org/LongAlign-10k", lambda ex: ex["messages"]),
    "longalpaca": ("Yukang/LongAlpaca-12k", lambda ex: [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]),
    "antihaystack": ("wenbopan/anti-haystack", lambda ex: [
        {"role": "user", "content": ex["document"] + "\n\nQuestion: " + ex["question"]},
        {"role": "assistant", "content": ex["answer"]},
    ]),
}


def sft_data_dir(max_length: int, output_dir=DEFAULT_DATA_DIR, datasets=None) -> Path:
    """``data/sft/len{L}`` for the full mixture, ``data/sft-{a}+{b}/len{L}`` for a subset."""
    out = Path(output_dir)
    if datasets is not None and set(datasets) != set(DATASETS):
        out = out.with_name(out.name + "-" + "+".join(d for d in DATASETS if d in datasets))
    return out / f"len{max_length}"


def tokenize_messages(messages, tokenizer, max_length):
    input_ids, labels = [], []
    for msg in messages:
        text = f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend(ids if msg["role"] == "assistant" else [-100] * len(ids))
    if len(input_ids) > max_length:
        input_ids, labels = input_ids[-max_length:], labels[-max_length:]
    return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}


def prepare_sft_data(base_model_dir, max_length: int = 262144, output_dir=DEFAULT_DATA_DIR,
                     val_ratio: float = 0.02, num_proc: int = 8, datasets=None) -> Path:
    """Download, tokenise and save the SFT data; returns the ``len{max_length}`` directory."""
    from datasets import concatenate_datasets, load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    parts = []
    for name, (repo, formatter) in DATASETS.items():
        if datasets is not None and name not in datasets:
            continue
        ds = load_dataset(repo, cache_dir=str(HF_CACHE_DIR))["train"]
        ds = ds.map(lambda ex: tokenize_messages(formatter(ex), tokenizer, max_length),
                    remove_columns=ds.column_names, num_proc=num_proc, desc=f"tokenize {name}")
        print(f"  {name}: {len(ds)} examples")
        parts.append(ds)
    combined = concatenate_datasets(parts).shuffle(seed=42)
    n_val = max(1, int(len(combined) * val_ratio))
    val, train = combined.select(range(n_val)), combined.select(range(n_val, len(combined)))

    out = sft_data_dir(max_length, output_dir, datasets)
    out.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(out / "train"))
    val.save_to_disk(str(out / "validation"))
    with open(out / "info.json", "w") as f:
        json.dump({"max_length": max_length, "train": len(train), "validation": len(val),
                   "datasets": list(datasets or DATASETS)}, f)
    print(f"Saved {out}: train={len(train)}, validation={len(val)}")
    return out


def parse_datasets(spec):
    """'longalign,longalpaca' -> list; None / 'all' -> None (the full mixture)."""
    if spec is None or spec == "all":
        return None
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in DATASETS]
    if bad:
        raise ValueError(f"unknown datasets {bad}; known: {list(DATASETS)}")
    return names


def ensure_sft_data(base_model_dir, max_length: int = 262144, output_dir=DEFAULT_DATA_DIR, datasets=None) -> Path:
    datasets = parse_datasets(datasets) if isinstance(datasets, str) else datasets
    out = sft_data_dir(max_length, output_dir, datasets)
    if (out / "train").exists() and (out / "validation").exists():
        return out
    print(f"SFT data not found at {out}; preparing it (max_length={max_length}, datasets={datasets or 'all'})...")
    return prepare_sft_data(base_model_dir, max_length, output_dir, datasets=datasets)
