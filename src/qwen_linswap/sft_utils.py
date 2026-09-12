"""Training utilities shared by the posttrain and evaluate stages.

Memory-efficient chunked cross-entropy: the LM head is applied to 2048-token
chunks of the final hidden states, each chunk's loss is back-propagated
immediately, and the accumulated hidden-state gradient is propagated through
the model once — so the full ``[T, vocab]`` logits are never materialised.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels = [], []
    for b in batch:
        ids, lbl = b["input_ids"], b["labels"]
        pad = max_len - len(ids)
        input_ids.append([0] * pad + ids)
        labels.append([-100] * pad + lbl)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def truncate_left(example, max_length):
    if max_length is None or len(example["input_ids"]) <= max_length:
        return example
    return {"input_ids": example["input_ids"][-max_length:], "labels": example["labels"][-max_length:]}


class TruncatedDataset(torch.utils.data.Dataset):
    """Left-truncates every example to ``max_length`` tokens (keeps the assistant tail)."""

    def __init__(self, ds, max_length):
        self.ds, self.max_length = ds, max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return truncate_left(self.ds[i], self.max_length)


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
    flat_labels = shift_labels.view(-1)
    total_count = int((flat_labels != ignore_index).sum().item())
    grad_scale = loss_scale / max(total_count, 1)

    hidden_grad = torch.zeros_like(shift_hidden)
    total_loss = 0.0
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        h_chunk = shift_hidden[:, start:end, :].reshape(-1, d).detach().requires_grad_(True)
        logits_chunk = F.linear(h_chunk, lm_head.weight)
        loss_chunk = F.cross_entropy(logits_chunk.float(), flat_labels[start:end],
                                     ignore_index=ignore_index, reduction="sum")
        (loss_chunk * grad_scale).backward()
        hidden_grad[:, start:end, :] += h_chunk.grad.view(b, end - start, d)
        total_loss += loss_chunk.item()

    if shift_hidden.requires_grad:
        shift_hidden.backward(hidden_grad)
    return total_loss / total_count if total_count else 0.0


def chunked_cross_entropy_eval(hidden_states, labels, lm_head, chunk_size=2048, ignore_index=-100):
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
    return (total_loss / total_count).item() if total_count else 0.0


def compute_loss(model, batch, chunk_size=2048, do_backward=False, loss_scale=1.0):
    hidden = model(batch["input_ids"], return_hidden=True)
    if do_backward:
        return chunked_cross_entropy_with_backward(hidden, batch["labels"], model.out_head,
                                                   chunk_size=chunk_size, loss_scale=loss_scale)
    return chunked_cross_entropy_eval(hidden, batch["labels"], model.out_head, chunk_size=chunk_size)


@torch.no_grad()
def evaluate(model, dataloader, max_batches=10, chunk_size=2048):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        total += compute_loss(model, batch, chunk_size=chunk_size)
        count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


def find_latest_checkpoint(output_dir):
    ckpts = list(Path(output_dir).glob("checkpoint-*"))
    return max(ckpts, key=lambda p: int(p.name.split("-")[1])) if ckpts else None
