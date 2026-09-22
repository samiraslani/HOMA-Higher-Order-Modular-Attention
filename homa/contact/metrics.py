"""Precision at L/k for contact prediction, by sequence-separation range."""

from __future__ import annotations

import math

import numpy as np
import torch

#: TAPE's separation ranges.  The paper reports the long range, |i-j| >= 24.
RANGES = {"short": (6, 12), "medium": (12, 24), "long": (24, 10 ** 6)}


@torch.no_grad()
def precision_at_L_over_k(prob, target, length, ks=(1, 2, 5)) -> dict:
    """Precision of the top ``L/k`` predicted contacts in each range.

    Args:
        prob: ``(L, L)`` contact probabilities.
        target: ``(L, L)`` labels in ``{0, 1, -1}``; -1 is ignored.
        length: The true sequence length ``L``.
        ks: The ``k`` in ``L/k``.  The paper's metric is ``k = 5``, long range.

    Returns:
        ``{"P@L/<k>_<range>": precision}``; NaN where a range has no pairs.

    Each unordered pair is scored once (upper triangle), because the head is
    symmetric and counting both orders would double every prediction.
    """
    idx = torch.arange(length, device=prob.device)
    sep = (idx[:, None] - idx[None, :]).abs()
    upper = idx[:, None] < idx[None, :]
    out = {}
    for name, (lo, hi) in RANGES.items():
        m = upper & (sep >= lo) & (sep < hi) & (target >= 0)
        n = int(m.sum())
        for k in ks:
            key = f"P@L/{k}_{name}"
            if n == 0:
                out[key] = float("nan")
                continue
            take = min(max(1, length // k), n)
            sel = torch.topk(prob[m], take).indices
            out[key] = float(target[m].float()[sel].mean())
    return out


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None) -> dict:
    """Mean per-protein precision over a loader.

    Averaged per protein rather than pooled over pairs, so a long protein does
    not outweigh a short one.
    """
    model.eval()
    acc = {}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        tgt = batch["contacts"].to(device)
        logits = model(ids, ids.shape[1])
        prob = torch.softmax(logits.float(), dim=-1)[..., 1]
        for i, ln in enumerate(batch["lengths"].tolist()):
            r = precision_at_L_over_k(prob[i, :ln, :ln], tgt[i, :ln, :ln], ln)
            for k, v in r.items():
                if not math.isnan(v):
                    acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}
