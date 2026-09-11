"""Preprocess the three long-context SFT datasets for the GDN2 Qwen3.5 model."""

import argparse
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


def load_tokenizer(model_dir: str):
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def tokenize_messages(messages, tokenizer, max_length):
    """Tokenize a list of {"role": ..., "content": ...} messages and mask non-assistant tokens."""
    input_ids = []
    labels = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Use the same formatting Qwen3.5's chat template uses for train sequences.
        text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        if role == "assistant":
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))

    # Truncate from the left so the question / answer tail is preserved.
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}


def format_longalign(example):
    return example["messages"]


def format_longalpaca(example):
    return [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]


def format_antihaystack(example):
    user_text = example["document"] + "\n\nQuestion: " + example["question"]
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": example["answer"]},
    ]


def process_dataset(name, tokenizer, max_length):
    if name == "longalign":
        ds = load_dataset("zai-org/LongAlign-10k", cache_dir="data/hf_cache")["train"]
        formatter = format_longalign
    elif name == "longalpaca":
        ds = load_dataset("Yukang/LongAlpaca-12k", cache_dir="data/hf_cache")["train"]
        formatter = format_longalpaca
    elif name == "antihaystack":
        ds = load_dataset("wenbopan/anti-haystack", cache_dir="data/hf_cache")["train"]
        formatter = format_antihaystack
    else:
        raise ValueError(f"Unknown dataset {name}")

    def map_fn(example):
        messages = formatter(example)
        return tokenize_messages(messages, tokenizer, max_length)

    ds = ds.map(
        map_fn,
        remove_columns=ds.column_names,
        num_proc=8,
        batched=False,
        desc=f"tokenize {name}",
    )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=str(Path(__file__).resolve().parents[1] / "models/Qwen3.5-0.8B"))
    parser.add_argument("--max_length", type=int, default=131072)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "data/sft"))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_dir)
    print(f"Preprocessing datasets with max_length={args.max_length}")

    parts = []
    for name in ["longalign", "longalpaca", "antihaystack"]:
        print(f"Processing {name}...")
        ds = process_dataset(name, tokenizer, args.max_length)
        parts.append(ds)
        print(f"  -> {len(ds)} examples")

    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=42)
    n_val = max(1, int(len(combined) * args.val_ratio))
    val = combined.select(range(n_val))
    train = combined.select(range(n_val, len(combined)))

    output_dir = Path(args.output_dir) / f"len{args.max_length}"
    output_dir.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(output_dir / "train"))
    val.save_to_disk(str(output_dir / "validation"))

    with open(output_dir / "info.json", "w") as f:
        json.dump({"max_length": args.max_length, "train": len(train), "validation": len(val)}, f)

    print(f"Saved {output_dir}: train={len(train)}, val={len(val)}")


if __name__ == "__main__":
    main()
