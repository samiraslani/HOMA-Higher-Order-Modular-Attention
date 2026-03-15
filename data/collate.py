"""
Custom collate functions for padding variable-length protein sequences.

All sequences are right-padded to ``FIXED_LEN_SEQ`` (512) tokens.
``input_ids`` are padded with 0 (the PAD token).
``labels`` for classification tasks are padded with -100 (the
``ignore_index`` of ``nn.CrossEntropyLoss``).
``targets`` for regression tasks are stacked as-is (scalar per sample).
"""

import torch
import torch.nn.functional as F

FIXED_LEN_SEQ: int = 512


def _pad1d(tensor: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    """Right-pad a 1-D tensor to ``length`` with ``pad_value``."""
    return F.pad(tensor, (0, length - tensor.size(0)), value=pad_value)


def collate_ss3(batch):
    """Collate a batch of secondary-structure samples.

    Pads ``input_ids``, ``attention_mask``, and ``labels`` to
    ``FIXED_LEN_SEQ``.

    Args:
        batch: List of dicts with keys ``"input_ids"``, ``"attention_mask"``,
            ``"labels"``.

    Returns:
        Dict with stacked tensors of shape ``(B, FIXED_LEN_SEQ)``.
    """
    input_ids = torch.stack([_pad1d(x["input_ids"], FIXED_LEN_SEQ, 0) for x in batch])
    attention_mask = torch.stack(
        [_pad1d(x["attention_mask"], FIXED_LEN_SEQ, 0) for x in batch]
    )
    labels = torch.stack([_pad1d(x["labels"], FIXED_LEN_SEQ, -100) for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_regression(batch):
    """Collate a batch of regression samples (fluorescence / stability).

    Pads ``input_ids`` to ``FIXED_LEN_SEQ`` and stacks scalar ``targets``.

    Args:
        batch: List of dicts with keys ``"input_ids"`` and ``"targets"``.

    Returns:
        Dict with ``"input_ids"`` of shape ``(B, FIXED_LEN_SEQ)`` and
        ``"targets"`` of shape ``(B,)``.
    """
    input_ids = torch.stack([_pad1d(x["input_ids"], FIXED_LEN_SEQ, 0) for x in batch])
    targets = torch.stack([x["targets"] for x in batch])
    return {"input_ids": input_ids, "targets": targets}
