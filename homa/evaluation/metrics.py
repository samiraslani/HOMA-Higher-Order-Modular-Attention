"""
Evaluation metrics for all published tasks.

Functions are stateless and operate on plain tensors or numpy arrays so they
can be called from any training loop without coupling to a specific framework.
"""

from typing import Optional

import torch
from scipy.stats import spearmanr


def accuracy_per_position(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Classification accuracy over non-padding positions.

    Args:
        logits: ``(B, L, num_classes)`` — raw model output.
        labels: ``(B, L)`` — ground-truth class indices.
            Positions with ``labels == ignore_index`` are excluded.
        ignore_index: Label value used for padding (default ``-100``).

    Returns:
        Fraction of correctly predicted positions in ``[0, 1]``.
    """
    preds = logits.argmax(dim=-1)          # (B, L)
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


def spearman_correlation(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Spearman rank correlation coefficient.

    Used for regression tasks (fluorescence, stability).

    Args:
        predictions: Predicted values — any shape (will be flattened).
        targets: Ground-truth values — same shape as ``predictions``.

    Returns:
        Spearman ρ in ``[-1, 1]``.  Returns ``0.0`` if all predictions or
        targets are constant (undefined correlation).
    """
    preds_np = predictions.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()
    rho, _ = spearmanr(preds_np, targets_np)
    # spearmanr returns nan when std == 0
    return float(rho) if rho == rho else 0.0
