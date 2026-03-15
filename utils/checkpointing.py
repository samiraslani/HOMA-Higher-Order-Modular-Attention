"""Checkpoint save / load helpers."""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        path: Full file path for the ``.pt`` file.
        model: The model whose ``state_dict`` is saved.
        optimizer: The optimizer whose ``state_dict`` is saved.
        epoch: Current epoch index (0-based).
        extra: Optional dict of additional items to store in the checkpoint
            (e.g. ``{"epoch_losses": [...]}``)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a checkpoint from disk and restore model (and optionally optimizer) state.

    Args:
        path: Path to the ``.pt`` file.
        model: Model instance to load weights into.
        optimizer: If provided, optimizer state is restored as well.
        device: Target device string (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        The full checkpoint dict so callers can access ``epoch``, loss
        histories, etc.
    """
    map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info("Checkpoint loaded from %s (epoch %d)", path, checkpoint.get("epoch", -1))
    return checkpoint
