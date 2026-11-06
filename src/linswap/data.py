"""Distillation data: the kinds of input the teacher handles, as one stream of ready-to-run batches.

  text    generic web text (DCLM), documents concatenated and cut into fixed-length sequences
  chat    instruction data (Alpaca) rendered with the backbone's chat template, packed the same way
  image   image question answering (five subsets of The Cauldron: photos, scene text, plots, screenshots,
          image pairs), several images to a multi-turn chat, run through the upstream processor

Distillation matches the student to the teacher's activations, so none of this is used as labels: the
corpora only decide *where* in input space the two are compared.  The image subsets are not among the
evaluation benchmarks.

``stream`` interleaves the sources so that each receives its share of the tokens; a batch is the keyword
arguments of the upstream model's forward (``input_ids`` and, with images, ``pixel_values`` /
``image_grid_thw``), never padded.  ``validation`` is a fixed held-out set of the same kinds plus one long
sequence.  Corpora are fetched from the Hub and tokenised on first use, under ``data/``.
"""

from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path

import torch

from .manifest import tokenizer_fingerprint
from .model import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
TEXT = {"dclm": ("mlfoundations/dclm-baseline-1.0-parquet",
                 "filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/"
                 "processed_data/global-shard_01_of_10/local-shard_0_of_10/shard_{:08d}_processed.parquet", "text", 10)}
CHAT = {"alpaca": ("yahma/alpaca-cleaned", "alpaca_data_cleaned.json")}
CAULDRON = ("HuggingFaceM4/the_cauldron", ("vqav2", "textcaps", "plotqa", "screen2words", "nlvr2"))
VAL_DOCS = 512                 # documents held out of each text corpus
VAL_ROWS = 64                  # rows held out of each image subset
MAX_PIXELS = 768 * 768         # per training image: at most 576 visual tokens
# (source, sequence length, sequences per batch, share of the tokens)
MIX = (("text", 16384, 1, 0.3), ("text", 2048, 8, 0.3), ("chat", 2048, 8, 0.2), ("image", None, None, 0.2))


# ----------------------------------------------------------------------------- corpora
def text_corpus(name, base_model_dir) -> Path:
    """Tokenised documents of a text or chat corpus (``train`` / ``validation``), prepared on first use; one
    copy per tokenizer, so backbones with different vocabularies do not share token ids."""
    shards = TEXT[name][3] if name in TEXT else 1
    want = tokenizer_fingerprint(base_model_dir)
    out = DATA_DIR / "text" / f"{name}-{shards}shard-{(want or 'unknown')[:8]}"
    if (out / "info.json").exists():
        return out
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model_dir)
    if name in CHAT:
        repo, file = CHAT[name]
        ds = load_dataset("json", data_files=hf_hub_download(repo, file, repo_type="dataset"))["train"].shuffle(seed=0)

        def tokenize(batch):
            chats = [[{"role": "user", "content": i + ("\n\n" + x if x else "")}, {"role": "assistant", "content": o}]
                     for i, x, o in zip(batch["instruction"], batch["input"], batch["output"])]
            text = [tok.apply_chat_template(c, tokenize=False, enable_thinking=False) for c in chats]
            return {"input_ids": tok(text, add_special_tokens=False)["input_ids"]}
    else:
        repo, pattern, col, _ = TEXT[name]
        files = [hf_hub_download(repo, pattern.format(i), repo_type="dataset") for i in range(shards)]
        ds = load_dataset("parquet", data_files=files)["train"].select_columns([col]).shuffle(seed=0)

        def tokenize(batch):
            return {"input_ids": [ids + [tok.eos_token_id] for ids in tok(batch[col], add_special_tokens=False)["input_ids"]]}

    ds = ds.map(tokenize, batched=True, num_proc=16, remove_columns=ds.column_names, desc=f"tokenize {name}")
    ds.select(range(VAL_DOCS)).save_to_disk(str(out / "validation"))
    ds.select(range(VAL_DOCS, len(ds))).save_to_disk(str(out / "train"))
    (out / "info.json").write_text(json.dumps({"corpus": name, "repo": repo, "docs": len(ds), "tokenizer_sha256": want,
                                               "validation_docs": VAL_DOCS}, indent=1))
    return out


def image_corpus() -> list[Path]:
    """The first parquet shard of each Cauldron subset (0.3 – 0.6 GiB each, thousands of images)."""
    from huggingface_hub import HfApi, hf_hub_download

    repo, subsets = CAULDRON
    root = DATA_DIR / "image"
    files = [next(iter(sorted((root / s).glob("*.parquet"))), None) for s in subsets]
    if None in files:
        listing = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
        files = [Path(hf_hub_download(repo, next(f for f in listing if f.startswith(s + "/")), repo_type="dataset",
                                      local_dir=str(root))) for s in subsets]
    return files


# ------------------------------------------------------------------------------ sources
def packed(docs, length, batch, seed):
    """Endless ``{"input_ids": [batch, length]}``: documents in shuffled order, concatenated and cut."""
    rng, order, buf, rows = random.Random(seed), list(range(len(docs))), [], []
    while True:
        rng.shuffle(order)
        for i in order:
            buf.extend(docs[i]["input_ids"])
            while len(buf) >= length:
                rows.append(buf[:length])
                buf = buf[length:]
                if len(rows) == batch:
                    yield {"input_ids": torch.tensor(rows)}
                    rows = []


class ImageChats(torch.utils.data.IterableDataset):
    """Endless multi-turn chats, ``images`` images each: every turn shows the images of one row and asks its
    first two questions, answered with the corpus' answers."""

    def __init__(self, files, processor, seed, images=4, held_out=False):
        self.files, self.processor, self.seed, self.images, self.held_out = files, processor, seed, images, held_out

    def rows(self):
        import pyarrow.parquet as pq

        tables = [pq.ParquetFile(f) for f in self.files]
        spans = [range(t.metadata.num_rows - VAL_ROWS, t.metadata.num_rows) if self.held_out
                 else range(t.metadata.num_rows - VAL_ROWS) for t in tables]
        index = [(t, i) for t, span in enumerate(spans) for i in span]
        groups = [t.metadata.row_group(0).num_rows for t in tables]
        rng = random.Random(self.seed)
        cache = {}
        while True:
            if not self.held_out:
                rng.shuffle(index)
            for t, i in index:
                g = i // groups[t]                             # row groups are equal-sized except the last
                if (t, g) not in cache:
                    cache.clear()                              # shuffled access: keep one decoded group
                    cache[(t, g)] = tables[t].read_row_group(g, columns=["images", "texts"]).to_pylist()
                yield cache[(t, g)][i - g * groups[t]]
            if self.held_out:
                return

    @staticmethod
    def image(cell):
        from PIL import Image

        img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        w, h = img.size
        if w * h > MAX_PIXELS:
            s = math.sqrt(MAX_PIXELS / (w * h))
            img = img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.LANCZOS)
        return img

    def __iter__(self):
        messages, n = [], 0
        for row in self.rows():
            for j, qa in enumerate(row["texts"][:2]):
                content = [{"type": "image", "image": self.image(c)} for c in row["images"]] if j == 0 else []
                messages.append({"role": "user", "content": content + [{"type": "text", "text": qa["user"]}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": qa["assistant"]}]})
            n += len(row["images"])
            if n >= self.images:
                batch = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                                           enable_thinking=False)
                yield {k: v for k, v in batch.items() if k != "attention_mask"}     # nothing is padded
                messages, n = [], 0


def _docs(name, base_model_dir, split):
    from datasets import load_from_disk

    return load_from_disk(str(text_corpus(name, base_model_dir) / split))


def stream(processor, base_model_dir, seed):
    """Endless ``(source, batch)``: at every draw the entry of ``MIX`` furthest below its token share."""
    files = image_corpus()
    loader = torch.utils.data.DataLoader(ImageChats(files, processor, seed), batch_size=None, num_workers=2,
                                         prefetch_factor=4)
    docs = {"text": _docs("dclm", base_model_dir, "train"), "chat": _docs("alpaca", base_model_dir, "train")}
    sources = [iter(loader) if kind == "image" else packed(docs[kind], length, batch, seed + i)
               for i, (kind, length, batch, _) in enumerate(MIX)]
    drawn = [0] * len(MIX)
    while True:
        i = min(range(len(MIX)), key=lambda j: drawn[j] / MIX[j][3])
        batch = next(sources[i])
        drawn[i] += batch["input_ids"].numel()
        yield MIX[i][0], batch


def validation(processor, base_model_dir):
    """The held-out set: ``[(source, batch)]`` — text, chat, image chats, and one 32K text sequence."""
    import itertools

    text, chat = _docs("dclm", base_model_dir, "validation"), _docs("alpaca", base_model_dir, "validation")
    out = [("text", b) for b in itertools.islice(packed(text, 8192, 1, 0), 4)]
    out += [("chat", b) for b in itertools.islice(packed(chat, 2048, 4, 0), 2)]
    out += [("image", b) for b in itertools.islice(ImageChats(image_corpus(), processor, 0, held_out=True), 8)]
    out += [("long", b) for b in itertools.islice(packed(text, 32768, 1, 1), 1)]
    return out
