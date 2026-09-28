"""TensorBoard logging for ``linswap distill``.

``TBLogger`` wraps ``SummaryWriter`` (a no-op when ``tensorboard`` is not installed or logging is
disabled); the helpers compute what the dashboard shows beyond the loss / grad norm / lr that the
training loop already has:

  parameter groups     ``new`` (the kernel's ``new_param_names``: gates, decays, ...), ``linear_shared``
                       (the swapped layers' q/k/v/g/o projections, convolutions and output norm, copied
                       from the backbone) and ``backbone`` (everything else) -- per-group gradient norm,
                       parameter norm, drift from the swap init and the relative size of one update.
  output distribution  KL(teacher ‖ student), top-1 agreement and both entropies on the held-out
                       sequences, chunked over token positions with the full vocabulary in every chunk
                       (the normalisation is exact; nothing is chunked over the vocabulary).

    tensorboard --logdir outputs            # every distill run writes outputs/<kernel>/distill/tensorboard
"""

from __future__ import annotations

import json
import warnings

import torch
import torch.nn.functional as F

GROUPS = ("new", "linear_shared", "backbone")


class TBLogger:
    """``SummaryWriter`` with a no-op fallback.  ``purge_step`` is passed through on resume so the
    events a crashed run logged at or after the checkpoint's step are dropped by TensorBoard instead
    of drawing a second, overlapping curve."""

    def __init__(self, log_dir, enabled=True, purge_step=None):
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            warnings.warn("TensorBoard logging disabled: `pip install tensorboard`")
            return
        self.writer = SummaryWriter(str(log_dir), purge_step=purge_step)

    @property
    def enabled(self):
        return self.writer is not None

    def scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def scalars(self, values: dict, step, prefix=""):
        for k, v in values.items():
            self.scalar(prefix + k, v, step)

    def text(self, tag, text, step=0):
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def args(self, args, step=0):
        self.text("args", "```\n" + json.dumps(vars(args), indent=1, default=str) + "\n```", step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


# ------------------------------------------------------------------ parameter groups
def param_groups(model, params):
    """The trainable ``params`` split into ``new`` / ``linear_shared`` / ``backbone`` as ``(name, p)`` lists."""
    trainable = {id(p) for p in params}
    new = {id(p) for _, p in model.new_parameters()}
    groups = {g: [] for g in GROUPS}
    for n, p in model.named_parameters():
        if id(p) not in trainable:
            continue
        g = "new" if id(p) in new else "linear_shared" if ".linear_attn." in n else "backbone"
        groups[g].append((n, p))
    return groups


def _sqsum(tensors):
    return sum(float(t.float().pow(2).sum()) for t in tensors)


def group_grad_norms(groups):
    """L2 norm of the accumulated gradient per group (0 for a group with no trainable parameters)."""
    return {g: _sqsum(p.grad for _, p in ps if p.grad is not None) ** 0.5 for g, ps in groups.items()}


def group_param_norms(groups):
    return {g: _sqsum(p.detach() for _, p in ps) ** 0.5 for g, ps in groups.items() if ps}


def snapshot(groups):
    """Clones of the parameters, taken before ``optimizer.step()`` for :func:`update_ratio`."""
    return {g: [p.detach().clone() for _, p in ps] for g, ps in groups.items() if ps}


def update_ratio(groups, before):
    """‖w_after − w_before‖ / ‖w_before‖ per group, over the whole group."""
    out = {}
    for g, prev in before.items():
        num = sum(float((p.detach().float() - b.float()).pow(2).sum()) for (_, p), b in zip(groups[g], prev))
        den = _sqsum(prev)
        out[g] = (num ** 0.5) / max(den ** 0.5, 1e-12)
    return out


def init_state(model):
    """CPU copy of every parameter: the reference for :func:`drift_from_init`."""
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}


def drift_from_init(groups, init_sd):
    """sqrt(Σ‖W − W₀‖²) / sqrt(Σ‖W₀‖²) per group, W₀ from :func:`init_state`."""
    out = {}
    for g, ps in groups.items():
        if not ps:
            continue
        num = den = 0.0
        for n, p in ps:
            w0 = init_sd[n].to(device=p.device, dtype=torch.float32)
            num += float((p.detach().float() - w0).pow(2).sum())
            den += float(w0.pow(2).sum())
        out[g] = (num ** 0.5) / max(den ** 0.5, 1e-12)
    return out


# ------------------------------------------------------------------ output distribution
@torch.no_grad()
def distribution_stats(s_hidden, t_hidden, s_head, t_head, chunk_size=2048, temperature=1.0, pos_edges=(512, 2048)):
    """KL(teacher ‖ student), top-1 agreement and entropies, averaged over positions, plus the KL per
    position bucket (``pos_edges`` split ``[0, T)``).  Chunks run over token positions; each chunk's
    ``log_softmax`` covers the full vocabulary."""
    b, T, d = s_hidden.shape
    s_flat, t_flat = s_hidden.reshape(-1, d), t_hidden.reshape(-1, d)
    n = s_flat.shape[0]
    kl = torch.empty(n, device=s_flat.device)
    ent_s, ent_t = torch.empty_like(kl), torch.empty_like(kl)
    agree = torch.empty(n, device=s_flat.device, dtype=torch.bool)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        t_logp = F.log_softmax(F.linear(t_flat[start:end], t_head.weight).float() / temperature, dim=-1)
        s_logp = F.log_softmax(F.linear(s_flat[start:end], s_head.weight).float() / temperature, dim=-1)
        t_p = t_logp.exp()
        kl[start:end] = (t_p * (t_logp - s_logp)).sum(-1)
        ent_t[start:end] = -(t_p * t_logp).sum(-1)
        ent_s[start:end] = -(s_logp.exp() * s_logp).sum(-1)
        agree[start:end] = s_logp.argmax(-1) == t_logp.argmax(-1)
    out = {"kl_teacher_student": kl.mean().item(), "top1_agreement": agree.float().mean().item(),
           "entropy_student": ent_s.mean().item(), "entropy_teacher": ent_t.mean().item()}
    pos = torch.arange(n, device=kl.device) % T
    edges = [0, *(e for e in pos_edges if e < T), T]
    for lo, hi in zip(edges, edges[1:]):
        m = (pos >= lo) & (pos < hi)
        if m.any():
            out[f"kl_pos_{lo}-{hi}"] = kl[m].mean().item()
    return out


@torch.no_grad()
def evaluate_vs_teacher(student, teacher, dataloader, max_batches=10, chunk_size=2048, temperature=1.0,
                        pos_edges=(512, 2048)):
    """Teacher cross-entropy and :func:`distribution_stats` on the validation sequences, mean of
    per-batch values (the same convention as ``train_utils.evaluate``)."""
    from linswap.train_utils import chunked_cross_entropy_eval

    was_training = student.training
    student.eval()
    device = next(student.parameters()).device
    total, count = {}, 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        s_hidden = student(ids, return_hidden=True)
        t_hidden = teacher(ids, return_hidden=True)
        stats = distribution_stats(s_hidden, t_hidden, student.lm_head, teacher.lm_head, chunk_size, temperature, pos_edges)
        stats["teacher_loss"] = chunked_cross_entropy_eval(t_hidden, labels, teacher.lm_head, chunk_size=chunk_size)
        for k, v in stats.items():
            total[k] = total.get(k, 0.0) + v
        count += 1
    if was_training:
        student.train()
    return {k: v / max(count, 1) for k, v in total.items()}
