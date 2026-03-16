"""
2D attention baseline mechanisms.

Three classes are provided, all inheriting from ``AttentionBase``:

* ``MultiHeadAttn2D``      — standard scaled dot-product attention
* ``Attn2DBlockwise``      — sliding-window attention
* ``Attn2DLinformer``      — low-rank (Linformer) attention
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import AttentionBase


class MultiHeadAttn2D(AttentionBase):
    """Standard multi-head scaled dot-product self-attention.

    Implements the formulation from "Attention Is All You Need" (Vaswani et al.
    2017):

    .. math::

        \\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_h}}\\right) V

    Args:
        num_heads: Number of parallel attention heads.
        d_model: Model / embedding dimension.
    """

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__(num_heads, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _self_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """Scaled dot-product attention.

        Args:
            q: ``(B, H, L, head_dim)``
            k: ``(B, H, L, head_dim)``
            v: ``(B, H, L, head_dim)``
            mask: ``(B, 1, 1, L)`` or broadcastable; 0 masks padding.

        Returns:
            Tuple of (attention_scores, output), each ``(B, H, L, head_dim)``.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return attn, torch.matmul(attn, v)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, L, d_model)``
            mask: ``(B, 1, 1, L)`` broadcastable mask; 1 = valid, 0 = pad.

        Returns:
            ``(B, L, d_model)``
        """
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        _, out = self._self_attention(Q, K, V, mask)

        B, _, L, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


class Attn2DBlockwise(AttentionBase):
    """Sliding-window (blockwise) 2D self-attention.

    The input sequence is split into overlapping blocks of length
    ``block_size`` with step ``stride``.  Standard 2D attention is computed
    independently within each block, and the block outputs are averaged back
    into a full sequence.

    Computational complexity: ``O(L · block_size²)`` instead of ``O(L²)``
    for large sequences.

    Args:
        num_heads: Number of attention heads.
        d_model: Model dimension.
        stride: Step between consecutive block start positions.
        block_size: Number of tokens per block.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        stride: int = 15,
        block_size: int = 30,
    ) -> None:
        super().__init__(num_heads, d_model)
        self.stride = stride
        self.block_size = block_size

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape blocked tensor ``(B, Blk, L_b, D)`` → ``(B, Blk, H, L_b, head_dim)``."""
        B, Blk, L_b, _ = x.shape
        return (
            x.reshape(B, Blk, L_b, self.num_heads, self.head_dim)
            .transpose(2, 3)
        )

    def _block_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention within each block.

        Args:
            q/k/v: ``(B, Blk, H, L_b, head_dim)``
            mask: ``(B, Blk, 1, 1, L_b)`` or ``None``.

        Returns:
            ``(B, Blk, H, L_b, head_dim)``
        """
        print(f"[_block_attention] q: {q.shape}, k: {k.shape}, v: {v.shape}")
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        print(f"[_block_attention] scores: {scores.shape}")
        if mask is not None:
            print(f"[_block_attention] mask: {mask.shape}")
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, L, d_model)``
            mask: ``(B, Blk, 1, 1, L_b)`` block-shaped mask; 1 = valid, 0 = pad.

        Returns:
            ``(B, L, d_model)``
        """
        B, L, _ = x.shape
        print(f"[Attn2DBlockwise.forward] x: {x.shape}, mask: {mask.shape if mask is not None else None}")

        q_raw = self._sliding_blocks(self.W_q(x), self.block_size, self.stride)
        k_raw = self._sliding_blocks(self.W_k(x), self.block_size, self.stride)
        v_raw = self._sliding_blocks(self.W_v(x), self.block_size, self.stride)
        print(f"[Attn2DBlockwise.forward] blocks (B,Blk,L_b,D): {q_raw.shape}")

        Q = self._split_heads_blocks(q_raw)
        K = self._split_heads_blocks(k_raw)
        V = self._split_heads_blocks(v_raw)
        print(f"[Attn2DBlockwise.forward] Q/K/V (B,Blk,H,L_b,Dh): {Q.shape}")

        out_blocks = self._block_attention(Q, K, V, mask)

        # (B, Blk, H, L_b, Dh) → (B, Blk, L_b, D)
        B2, Blk, H, L_b, Dh = out_blocks.shape
        out_blocks = out_blocks.transpose(2, 3).contiguous().view(B2, Blk, L_b, self.d_model)

        out = self._reconstruct_from_blocks(out_blocks, L, self.stride)
        return self.W_o(out)


class Attn2DLinformer(AttentionBase):
    """Low-rank (Linformer) multi-head self-attention.

    Keys and values are projected to a low-rank space of dimension ``k``
    before computing attention, reducing complexity from ``O(L²)`` to
    ``O(L·k)`` (Wang et al., 2020 – "Linformer: Self-Attention with Linear
    Complexity").

    Args:
        num_heads: Number of attention heads.
        d_model: Model dimension.
        k: Low-rank projection size (k << L).
        len_seq: Input sequence length (required to define the projection
            matrices ``E`` and ``F``).
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        k: int,
        len_seq: int,
    ) -> None:
        super().__init__(num_heads, d_model)
        self.k = k

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Low-rank sequence projections (L → k)
        self.E = nn.Linear(len_seq, k)  # projects keys
        self.F = nn.Linear(len_seq, k)  # projects values

    def _project_low_rank(
        self,
        x: torch.Tensor,
        seq_proj: nn.Linear,
        feat_proj: nn.Linear,
    ) -> torch.Tensor:
        """Project ``x`` through a feature linear and a sequence linear.

        Args:
            x: ``(B, L, d_model)``
            seq_proj: Linear ``L → k``
            feat_proj: Linear ``d_model → d_model``

        Returns:
            ``(B, H, k, head_dim)``
        """
        x = feat_proj(x).permute(0, 2, 1)   # (B, D, L)
        x = seq_proj(x).permute(0, 2, 1)    # (B, k, D)
        return self._split_heads(x)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, L, d_model)``
            mask: ``(B, 1, L, 1)`` broadcastable; 1 = valid.

        Returns:
            ``(B, L, d_model)``
        """
        Q = self._split_heads(self.W_q(x))         # (B, H, L, Dh)
        K = self._project_low_rank(x, self.E, self.W_k)  # (B, H, k, Dh)
        V = self._project_low_rank(x, self.F, self.W_v)  # (B, H, k, Dh)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, L, k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, L, Dh)

        B, _, L, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)
