"""
Abstract base class and shared utilities for all attention mechanisms.

Concrete attention classes (attention_2d.py, attention_3d.py) inherit from
``AttentionBase`` and reuse the helper methods defined here to avoid
duplication.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class AttentionBase(nn.Module, ABC):
    """Abstract base for all attention mechanisms in this package.

    Subclasses must implement :meth:`forward`.  Helper methods
    ``_split_heads``, ``_sliding_blocks``, and ``_reconstruct_from_blocks``
    provide standard implementations that sliding-window variants can reuse
    or override.

    Args:
        num_heads: Number of parallel attention heads.
        d_model: Total model dimension (must be divisible by ``num_heads``).
    """

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention-weighted output.

        Args:
            x: Input tensor of shape ``(B, L, d_model)``.
            mask: Optional boolean/integer mask.  Shape conventions vary by
                subclass; ``1`` marks valid positions, ``0`` marks padding.

        Returns:
            Output tensor of shape ``(B, L, d_model)``.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, L, d_model)`` → ``(B, H, L, head_dim)``.

        Args:
            x: Tensor of shape ``(B, L, d_model)``.

        Returns:
            Tensor of shape ``(B, num_heads, L, head_dim)``.
        """
        B, L, _ = x.shape
        return (
            x.view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    def _sliding_blocks(
        self, x: torch.Tensor, block_size: int, stride: int
    ) -> torch.Tensor:
        """Extract overlapping sliding blocks from a sequence.

        Uses ``as_strided`` for a zero-copy view where possible.

        Args:
            x: Input sequence of shape ``(B, L, D)``.
            block_size: Length of each block (``l``).
            stride: Step between consecutive block start positions (``d``).

        Returns:
            Tensor of shape ``(B, num_blocks, block_size, D)`` where
            ``num_blocks = (L - block_size) // stride + 1``.
        """
        B, L, D = x.shape
        num_blocks = (L - block_size) // stride + 1
        shape = (B, num_blocks, block_size, D)
        strides = (
            x.stride(0),
            stride * x.stride(1),
            x.stride(1),
            x.stride(2),
        )
        return x.as_strided(shape, strides).contiguous()

    def _reconstruct_from_blocks(
        self,
        blocks: torch.Tensor,
        seq_len: int,
        stride: int,
    ) -> torch.Tensor:
        """Average overlapping block outputs back into a full sequence.

        Args:
            blocks: Tensor of shape ``(B, num_blocks, block_size, D)``.
            seq_len: Target sequence length ``L``.
            stride: The stride used when creating the blocks.

        Returns:
            Tensor of shape ``(B, seq_len, D)``.
        """
        block_size = blocks.shape[2]
        B, num_blocks, _, D = blocks.shape
        device = blocks.device

        start_idx = torch.arange(num_blocks, device=device) * stride
        block_offsets = torch.arange(block_size, device=device)
        # positions[b, block, offset] = position in [0, seq_len)
        positions = (start_idx[:, None] + block_offsets).unsqueeze(0).expand(B, -1, -1)

        flat_pos = positions.reshape(B, -1)              # (B, num_blocks * block_size)
        flat_blocks = blocks.reshape(B, -1, D)           # (B, num_blocks * block_size, D)

        out = torch.zeros(B, seq_len, D, device=device)
        counts = torch.zeros(B, seq_len, 1, device=device)

        out.scatter_add_(1, flat_pos.unsqueeze(-1).expand(-1, -1, D), flat_blocks)
        counts.scatter_add_(
            1,
            flat_pos.unsqueeze(-1),
            torch.ones(B, flat_pos.shape[1], 1, device=device),
        )
        return out / counts


def softmax_nd(x: torch.Tensor, dim) -> torch.Tensor:
    """Numerically stable softmax along one or multiple axes.

    Args:
        x: Input tensor (any shape).
        dim: Axis or tuple of axes over which to normalise.

    Returns:
        Softmax-normalised tensor of the same shape.
    """
    x = x - torch.amax(x, dim=dim, keepdim=True)
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)
