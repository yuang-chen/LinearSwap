"""Batching, packing and chunked losses for the distillation stages.

``collate_fn`` right-pads, ``PackedDataset`` concatenates documents into fixed-length sequences, and the
chunked cross-entropy / evaluation helpers compute the loss in vocabulary chunks so the full logits are
never materialised."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def collate_fn(batch, pad_id: int = 0):
    """Right-pad to the longest example.  Right padding keeps a causal model exact without an
    attention mask: padded positions come after every real token and carry label -100, so they
    never influence a supervised position (the recurrent state after the last real token is
    simply unused)."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels = [], []
    for b in batch:
        ids, lbl = b["input_ids"], b["labels"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lbl + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class PackedDataset(torch.utils.data.IterableDataset):
    """Concatenate shuffled examples into fixed-length sequences (no padding, no masks).

    Used for long-context distillation: a 64K packed sequence contains several documents, so
    the student is trained to reproduce the teacher across document boundaries — i.e. to forget
    what the teacher forgets.  Labels are the input ids (all positions supervised)."""

    def __init__(self, ds, length, seed=0):
        self.ds, self.length, self.seed = ds, length, seed

    def __iter__(self):
        import random

        rng = random.Random(self.seed)
        order = list(range(len(self.ds)))
        while True:
            rng.shuffle(order)
            buf = []
            for i in order:
                buf.extend(self.ds[i]["input_ids"])
                while len(buf) >= self.length:
                    seq, buf = buf[:self.length], buf[self.length:]
                    yield {"input_ids": seq, "labels": list(seq)}


def chunked_cross_entropy_with_backward(hidden_states, labels, lm_head, chunk_size=2048, ignore_index=-100,
                                        loss_scale=1.0):
    """Mean token cross-entropy with backward pass; returns the (unscaled) average loss.

    Both the LM-head gradient (accumulated by the per-chunk backward calls) and the
    hidden-state gradient are scaled by ``loss_scale / num_label_tokens`` so the
    result equals ``loss_scale * mean CE`` exactly.
    """
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()
    b, seq_len, d = shift_hidden.shape
    # Flatten batch and sequence together so chunks index hidden states and labels consistently.
    flat_hidden = shift_hidden.reshape(-1, d)
    flat_labels = shift_labels.reshape(-1)
    total_count = int((flat_labels != ignore_index).sum().item())
    grad_scale = loss_scale / max(total_count, 1)

    hidden_grad = torch.zeros_like(flat_hidden)
    total_loss = 0.0
    n_rows = flat_hidden.shape[0]
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        h_chunk = flat_hidden[start:end].detach().requires_grad_(True)
        logits_chunk = F.linear(h_chunk, lm_head.weight)
        loss_chunk = F.cross_entropy(logits_chunk.float(), flat_labels[start:end],
                                     ignore_index=ignore_index, reduction="sum")
        (loss_chunk * grad_scale).backward()
        hidden_grad[start:end] = h_chunk.grad
        total_loss += loss_chunk.item()

    if shift_hidden.requires_grad:
        shift_hidden.backward(hidden_grad.view_as(shift_hidden))
    return total_loss / total_count if total_count else 0.0


def chunked_cross_entropy_eval(hidden_states, labels, lm_head, chunk_size=2048, ignore_index=-100,
                               return_sum=False):
    """Mean (or, with ``return_sum``, (sum, count)) cross-entropy over the supervised positions."""
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
    flat_labels = labels[:, 1:].contiguous().view(-1)
    total_loss = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
    total_count = 0
    for start in range(0, flat_hidden.size(0), chunk_size):
        end = min(start + chunk_size, flat_hidden.size(0))
        logits_chunk = F.linear(flat_hidden[start:end], lm_head.weight)
        total_loss += F.cross_entropy(logits_chunk.float(), flat_labels[start:end],
                                      ignore_index=ignore_index, reduction="sum")
        total_count += int((flat_labels[start:end] != ignore_index).sum())
    if return_sum:
        return total_loss.item(), total_count
    return (total_loss / total_count).item() if total_count else 0.0


def compute_loss(model, batch, chunk_size=2048, do_backward=False, loss_scale=1.0, return_sum=False):
    hidden = model(batch["input_ids"], return_hidden=True)
    if do_backward:
        return chunked_cross_entropy_with_backward(hidden, batch["labels"], model.lm_head,
                                                   chunk_size=chunk_size, loss_scale=loss_scale)
    return chunked_cross_entropy_eval(hidden, batch["labels"], model.lm_head, chunk_size=chunk_size,
                                      return_sum=return_sum)


@torch.no_grad()
def evaluate(model, dataloader, max_batches=10, chunk_size=2048, token_weighted=False):
    """Validation cross-entropy.  Default: mean of per-batch means (the metric used in the docs);
    ``token_weighted=True``: total loss / total supervised tokens (corpus-level NLL)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if token_weighted:
            s, n = compute_loss(model, batch, chunk_size=chunk_size, return_sum=True)
            total += s
            count += n
        else:
            total += compute_loss(model, batch, chunk_size=chunk_size)
            count += 1
    if was_training:
        model.train()
    return total / max(count, 1)
