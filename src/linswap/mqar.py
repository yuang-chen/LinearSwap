"""In-context multi-query associative recall (MQAR-style) for pretrained language models.

The classic MQAR benchmark (Arora et al., "Zoology") trains models on synthetic key-value
sequences; for a pretrained LM we phrase the same task in text so no training is needed:

    apple => 4831 ; river => 2207 ; ... ; (N pairs, random words, random 4-digit values)
    apple => 4831 ; river => 2207 ; ...   (all keys queried again in random order)

The model is run teacher-forced over the whole sequence and a query counts as correct when
every value token is the arg-max prediction at its position.  Accuracy versus the number of
pairs measures how much the recurrent state can hold and retrieve — the mechanism-level
property that separates kernels with and without an erase term, at sequence lengths set by
``pairs`` (≈ 8 tokens per pair).
"""

from __future__ import annotations

import random

import torch

from .data import HF_CACHE_DIR


def _word_list():
    try:
        import wonderwords

        rw = wonderwords.RandomWord()
        return sorted(set(rw._categories["noun"] + rw._categories["adjective"]))
    except Exception:  # pragma: no cover
        return [f"key{i}" for i in range(5000)]


def make_example(tokenizer, n_pairs: int, seed: int):
    rng = random.Random(seed)
    words = rng.sample(_word_list(), n_pairs)
    values = [rng.randint(1000, 9999) for _ in words]
    context = " ; ".join(f"{w} => {v}" for w, v in zip(words, values))
    order = list(range(n_pairs))
    rng.shuffle(order)
    ids = tokenizer(context + " ; ", add_special_tokens=False).input_ids
    targets = []  # (positions of value tokens, value token ids)
    for i in order:
        q = tokenizer(f"{words[i]} => ", add_special_tokens=False).input_ids
        v = tokenizer(f"{values[i]}", add_special_tokens=False).input_ids
        ids += q
        targets.append((list(range(len(ids), len(ids) + len(v))), v))
        ids += v + tokenizer(" ; ", add_special_tokens=False).input_ids
    return ids, targets


@torch.no_grad()
def mqar_accuracy(model, tokenizer, n_pairs: int, n_samples: int = 10, seed: int = 0) -> dict:
    device = next(model.parameters()).device
    correct = total = 0
    length = 0
    for s in range(n_samples):
        ids, targets = make_example(tokenizer, n_pairs, seed * 1000 + s)
        length = max(length, len(ids))
        x = torch.tensor([ids], device=device)
        pred = model(x).argmax(-1)[0]           # pred[t] predicts token t+1
        for positions, value in targets:
            ok = all(int(pred[p - 1]) == v for p, v in zip(positions, value))
            correct += ok
            total += 1
    return {"pairs": n_pairs, "acc": correct / max(total, 1), "queries": total, "max_len": length}
