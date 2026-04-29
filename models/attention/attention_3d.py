"""
3D attention modules: HOMA and MultiHeadAttn3D (blockwise3d).

HOMA is the primary contribution (2D + 3D fusion).
MultiHeadAttn3D is a triadic-only blockwise attention with windowed (j, k)
interactions and a low-rank U-matrix — no 2D branch, no fusion MLP.

``HOMA`` is the primary novel mechanism introduced in
this work.  It extends standard 2D multi-head attention with a *third-order
interaction term* while keeping computational cost tractable through two
key design choices:

1. **Sliding-window (multi-ped) blocking** — the sequence is divided into
   overlapping blocks of length ``block_size`` with step ``stride``, so all
   tensor operations are local.
2. **Low-rank U-matrix** — the additional ``U`` projection required for 3D
   attention is factored as ``U = x W_{u_u} W_{u_v}`` with inner rank
   ``rank``, reducing its parameter count from ``d^2`` to ``2 d · rank``
   (e.g., 262 k → 8 k for d=512, rank=8).

Within each block the 3D interaction is further restricted to a local context
window of size ``window_size`` (default 7), so the per-block cost scales as
``O(L_b · w²)`` instead of ``O(L_b³)``.

Transfer learning
-----------------
Pre-trained 2D weights (W_q, W_k, W_v) can be loaded from an existing
checkpoint via ``load_pretrained_2d`` and optionally frozen, allowing the
model to be initialised from a strong 2D baseline and to only train the 3D
components (W_u_u, W_u_v, fusion_layer) from scratch.

Mathematical formulation
------------------------
Given input ``x ∈ R^{B × L × d}``:

  Q = x W_q,   K = x W_k,   V = x W_v         (standard 2D projections)
  U_mat = (x W_{u_u}) W_{u_v}                   (low-rank third projection)

After head-splitting and block-extraction (all tensors: B × Blk × H × L_b × d_h):

  **2D branch:**
    scores_2d[i, j] = Q[i] · K[j] / √d_h
    attn_2d = softmax(scores_2d, dim=-1) · V        shape (B Blk H L_b d_h)

  **3D branch (local window w):**
    For each position i, extract K_local, U_local, V_local from the w
    nearest neighbours (replicate-padded at boundaries):

    scores_3d[i, j, k] = (Q[i] ⊙ K_local[j] ⊙ U_local[k]).sum() / √d_h
                          j, k ∈ [0, w)

    attn_3d = softmax(scores_3d, dim=(-2, -1))      shape (B Blk H L_b w w)

    V_paired[j, k] = V_local[j] ⊙ V_local[k]       shape (B Blk H L_b w w d_h)
    res_3d[i] = Σ_{j,k} attn_3d[i,j,k] · V_paired[j,k]

  **Fusion:**
    combined = cat([attn_2d, res_3d], dim=-1)       shape (B Blk H L_b 2·d_h)
    out = MLP(combined)                              shape (B Blk H L_b d_h)

Blocks are reconstructed and averaged back to the full sequence, followed by
a final linear projection W_o.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .base import AttentionBase, softmax_nd

logger = logging.getLogger(__name__)


def _remap_legacy_u_projection_keys(state_dict, prefix: str) -> None:
    """Allow old checkpoints that used W_l_* names to load into W_u_* modules."""
    for old, new in (("W_l_u", "W_u_u"), ("W_l_v", "W_u_v")):
        for suffix in ("weight", "bias"):
            old_key = f"{prefix}{old}.{suffix}"
            new_key = f"{prefix}{new}.{suffix}"
            if old_key in state_dict:
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]


class HOMA(AttentionBase):
    """HOMA — Higher-Order MultiHead Attention with low-rank U-matrix and 2D transfer.

    See module docstring for the full mathematical description.

    Args:
        num_heads: Number of parallel attention heads.
        d_model: Model / embedding dimension.
        stride: Step between consecutive sliding-window block start positions.
        block_size: Number of tokens per block (``L_b``).
        window_size: Local context window size for 3D interactions (``w``).
            Must be odd (symmetric padding).
        rank: Inner rank of the low-rank U-matrix decomposition.
        load_from_pretrained_2d: If ``True``, load W_q/W_k/W_v weights from
            ``pretrained_2d_ckpt`` before any training.
        pretrained_2d_ckpt: Path to a ``.pt`` checkpoint file.  Required when
            ``load_from_pretrained_2d=True``.
        freeze_2d: If ``True``, freeze W_q/W_k/W_v after loading so only the
            3D-specific parameters are updated during training.
        prefix_hint: Optional key prefix used when matching parameter names
            inside the checkpoint state-dict.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        stride: int = 15,
        block_size: int = 40,
        window_size: int = 7,
        rank: int = 8,
        load_from_pretrained_2d: bool = False,
        pretrained_2d_ckpt: Optional[str] = None,
        freeze_2d: bool = False,
        prefix_hint: str = "",
    ) -> None:
        super().__init__(num_heads, d_model)
        self.stride = stride
        self.block_size = block_size
        self.window_size = window_size
        self.rank = rank

        # 2D projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Low-rank 3D projection  U = x W_{u_u} W_{u_v}
        self.W_u_u = nn.Linear(d_model, rank, bias=False)
        self.W_u_v = nn.Linear(rank, d_model, bias=False)

        # Fusion MLP that combines 2D and 3D outputs
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * self.head_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.head_dim),
        )

        if load_from_pretrained_2d:
            if pretrained_2d_ckpt is None:
                raise ValueError(
                    "pretrained_2d_ckpt must be provided when "
                    "load_from_pretrained_2d=True"
                )
            self.load_pretrained_2d(pretrained_2d_ckpt, prefix_hint)

        if freeze_2d:
            for layer in (self.W_q, self.W_k, self.W_v):
                for p in layer.parameters():
                    p.requires_grad_(False)
            logger.info("Froze pretrained W_q, W_k, W_v")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _remap_legacy_u_projection_keys(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def load_pretrained_2d(self, ckpt_path: str, prefix_hint: str = "") -> None:
        """Load W_q, W_k, W_v weights from a 2D attention checkpoint.

        Attempts a best-effort key match: a state-dict key is accepted if it
        ends with ``"W_q.weight"``, ``"W_q.bias"``, etc. (with an optional
        ``prefix_hint`` suffix).

        Args:
            ckpt_path: Path to the ``.pt`` checkpoint.
            prefix_hint: Optional string appended to the key suffix during
                matching, useful when the checkpoint uses layer-index prefixes.
        """
        logger.info("Loading pretrained 2D parameters from: %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        def _match(target: str):
            for k, v in state_dict.items():
                if k.endswith(target) or (
                    prefix_hint and k.endswith(prefix_hint + target)
                ):
                    return v
            return None

        for name, layer in (("W_q", self.W_q), ("W_k", self.W_k), ("W_v", self.W_v)):
            w = _match(f"{name}.weight")
            b = _match(f"{name}.bias")
            if w is not None and b is not None:
                layer.weight.data.copy_(w)
                layer.bias.data.copy_(b)
                logger.info("Loaded pretrained weights for %s", name)
            else:
                logger.warning("Could not find pretrained weights for %s", name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_heads_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, Blk, L_b, D)`` → ``(B, Blk, H, L_b, head_dim)``."""
        B, Blk, L_b, _ = x.shape
        return (
            x.view(B, Blk, L_b, self.num_heads, self.head_dim)
            .transpose(2, 3)
        )

    @staticmethod
    def _replicate_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
        """Replicate-pad along the token dimension (dim=3).

        Args:
            x: ``(B, Blk, H, L_b, D)``
            pad: Number of positions to pad on each side.

        Returns:
            ``(B, Blk, H, L_b + 2·pad, D)``
        """
        left = x[:, :, :, :1, :].expand(-1, -1, -1, pad, -1)
        right = x[:, :, :, -1:, :].expand(-1, -1, -1, pad, -1)
        return torch.cat([left, x, right], dim=3)

    def _build_window_mask(self, mask_blocks: torch.Tensor) -> torch.Tensor:
        """Build a 3D window mask from block-level padding indicators.

        Args:
            mask_blocks: ``(B, Blk, L_b)`` — 1 for valid tokens, 0 for padding.

        Returns:
            ``(B, Blk, 1, 1, L_b, w, w)`` mask for ``scores_w_w``.
        """
        B, Blk, L_b = mask_blocks.shape
        pad = self.window_size // 2
        left = mask_blocks[:, :, :1].expand(-1, -1, pad)
        right = mask_blocks[:, :, -1:].expand(-1, -1, pad)
        mask_pad = torch.cat([left, mask_blocks, right], dim=2)

        # (B, Blk, L_b, w)
        mask_local = mask_pad.unfold(dimension=2, size=self.window_size, step=1)

        mask_center = mask_blocks.unsqueeze(-1).unsqueeze(-1)   # (B, Blk, L_b, 1, 1)
        mask_k = mask_local.unsqueeze(-1)                       # (B, Blk, L_b, w, 1)
        mask_u = mask_local.unsqueeze(-2)                       # (B, Blk, L_b, 1, w)

        mask_window = mask_center * mask_k * mask_u             # (B, Blk, L_b, w, w)
        return mask_window.unsqueeze(2)                         # (B, Blk, 1, 1, L_b, w, w)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute HOMA attention.

        Args:
            x: Input sequence ``(B, L, d_model)``.
            mask: Optional ``(B, Blk, L_b)`` block-level mask produced by the
                model's ``generate_padding_mask``; 1 = valid, 0 = padding.

        Returns:
            Output sequence ``(B, L, d_model)``.
        """
        B, L, _ = x.shape
        w = self.window_size
        pad = w // 2

        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # Low-rank U-matrix
        u_mat = self.W_u_v(self.W_u_u(x))

        # Slide into blocks, then split heads → (B, Blk, H, L_b, Dh)
        def _to_blocks_heads(t):
            return self._split_heads_blocks(
                self._sliding_blocks(t, self.block_size, self.stride)
            )

        Q = _to_blocks_heads(q)
        K = _to_blocks_heads(k)
        V = _to_blocks_heads(v)
        U_m = _to_blocks_heads(u_mat)
        B2, Blk, H, L_b, Dh = Q.shape

        # ---- 2D attention branch ----------------------------------------
        scores_2d = torch.matmul(Q, K.transpose(-2, -1)) / (Dh ** 0.5)
        if mask is not None:
            # mask: (B, Blk, L_b) → (B, Blk, 1, 1, L_b) for broadcast
            scores_2d = scores_2d.masked_fill(
                mask.unsqueeze(2).unsqueeze(3) == 0, -1e9
            )
        attn_2d = torch.softmax(scores_2d, dim=-1)
        attn_out_2d = torch.matmul(attn_2d, V)              # (B, Blk, H, L_b, Dh)

        # ---- 3D local-window branch ------------------------------------
        K_pad = self._replicate_pad(K, pad)                  # (B, Blk, H, L_b+2p, Dh)
        U_pad = self._replicate_pad(U_m, pad)
        V_pad = self._replicate_pad(V, pad)

        # Local windows: (B, Blk, H, L_b, w, Dh)
        def _unfold_window(t):
            return t.unfold(dimension=3, size=w, step=1).permute(0, 1, 2, 3, 5, 4).contiguous()

        K_local = _unfold_window(K_pad)
        U_local = _unfold_window(U_pad)
        V_local = _unfold_window(V_pad)

        # Tri-linear interaction scores: (B, Blk, H, L_b, w, w)
        scores_3d = torch.einsum(
            "bphld,bphlwd,bphlvd->bphlwv", Q, K_local, U_local
        ) / (Dh ** 0.5)

        if mask is not None:
            mask_window = self._build_window_mask(mask).to(scores_3d.device)
            scores_3d = scores_3d.masked_fill(mask_window == 0, -1e9)

        attn_3d = softmax_nd(scores_3d, dim=(-2, -1))        # (B, Blk, H, L_b, w, w)

        # Outer-product value interaction: (B, Blk, H, L_b, w, w, Dh)
        VV_local = torch.einsum("bphlwd,bphlvd->bphlwvd", V_local, V_local)

        # Aggregate → (B, Blk, H, L_b, Dh)
        res_3d = torch.einsum("bphlwv,bphlwvd->bphld", attn_3d, VV_local)

        # ---- Fusion -------------------------------------------------------
        combined = torch.cat([attn_out_2d, res_3d], dim=-1)   # (B, Blk, H, L_b, 2·Dh)
        out = self.fusion_layer(combined)                     # (B, Blk, H, L_b, Dh)
        out = out.transpose(2, 3).contiguous().view(B2, Blk, L_b, self.d_model)

        # Reconstruct full sequence by averaging overlapping block outputs
        out_full = self._reconstruct_from_blocks(out, L, self.stride)
        return self.W_o(out_full)


class MultiHeadAttn3D(AttentionBase):
    """Triadic-only blockwise attention with windowed (j, k) interactions.

    Unlike HOMA there is no 2D branch and no fusion MLP — this module computes
    only the 3D triadic interaction with a low-rank U-matrix and local windows.

    Input sequence ``(B, L, d_model)`` is split into overlapping blocks of
    length ``block_size`` (step ``stride``).  Within each block, for every
    query position ``i``, attention is restricted to a local window of size
    ``window_size`` for both the ``j`` (key) and ``k`` (U-matrix) indices:

    .. math::

        \\text{scores}[i,j,k] = \\frac{(Q_i \\odot K_{j}^{(w)} \\odot U_{k}^{(w)}) \\cdot \\mathbf{1}}{\\sqrt{d_h}},
        \\quad j, k \\in [0, w)

    The per-block cost is ``O(L_b · w²)`` and the U-matrix is factored as
    ``U = x W_{u_u} W_{u_v}`` (rank ``rank``), keeping parameter count low.

    Args:
        num_heads: Number of parallel attention heads.
        d_model: Model / embedding dimension.
        block_size: Number of tokens per sliding block (``L_b``).
        stride: Step between consecutive block start positions.
        window_size: Local context window size ``w`` (must be odd).
        rank: Inner rank of the low-rank U-matrix decomposition.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        block_size: int = 30,
        stride: int = 15,
        window_size: int = 7,
        rank: int = 8,
    ) -> None:
        super().__init__(num_heads, d_model)

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd so that the local window is centred")

        self.block_size = block_size
        self.stride = stride
        self.window_size = window_size
        self.rank = rank

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Low-rank U-matrix:  U = x W_{u_u} W_{u_v}
        self.W_u_u = nn.Linear(d_model, rank, bias=False)
        self.W_u_v = nn.Linear(rank, d_model, bias=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _remap_legacy_u_projection_keys(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_heads_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, Blk, L_b, D)`` → ``(B, Blk, H, L_b, head_dim)``."""
        B, Blk, L_b, _ = x.shape
        return (
            x.view(B, Blk, L_b, self.num_heads, self.head_dim)
            .transpose(2, 3)
        )

    @staticmethod
    def _replicate_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
        """Replicate-pad along the token dimension (dim 3).

        Args:
            x: ``(B, Blk, H, L_b, D)``
            pad: Positions to pad on each side.

        Returns:
            ``(B, Blk, H, L_b + 2·pad, D)``
        """
        left = x[:, :, :, :1, :].expand(-1, -1, -1, pad, -1)
        right = x[:, :, :, -1:, :].expand(-1, -1, -1, pad, -1)
        return torch.cat([left, x, right], dim=3)

    def _build_window_mask(self, mask_blocks: torch.Tensor) -> torch.Tensor:
        """Build a 3D window mask from block-level padding indicators.

        Args:
            mask_blocks: ``(B, Blk, L_b)`` — 1 for valid tokens, 0 for padding.

        Returns:
            ``(B, Blk, 1, L_b, w, w)``
        """
        pad = self.window_size // 2
        left = mask_blocks[:, :, :1].expand(-1, -1, pad)
        right = mask_blocks[:, :, -1:].expand(-1, -1, pad)
        mask_pad = torch.cat([left, mask_blocks, right], dim=2)

        mask_local = mask_pad.unfold(dimension=2, size=self.window_size, step=1)  # (B, Blk, L_b, w)

        mask_center = mask_blocks.unsqueeze(-1).unsqueeze(-1)   # (B, Blk, L_b, 1, 1)
        mask_j = mask_local.unsqueeze(-1)                       # (B, Blk, L_b, w, 1)
        mask_k = mask_local.unsqueeze(-2)                       # (B, Blk, L_b, 1, w)

        mask_window = mask_center * mask_j * mask_k             # (B, Blk, L_b, w, w)
        return mask_window.unsqueeze(2)                         # (B, Blk, 1, L_b, w, w)

    def _triadic_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        mask_blocks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Windowed triadic self-attention.

        Args:
            q, k, u, v: ``(B, Blk, H, L_b, head_dim)``
            mask_blocks: ``(B, Blk, L_b)`` or ``None``

        Returns:
            ``(B, Blk, H, L_b, head_dim)``
        """
        pad = self.window_size // 2
        w = self.window_size
        scale = self.head_dim ** 0.5

        k_pad = self._replicate_pad(k, pad)
        u_pad = self._replicate_pad(u, pad)
        v_pad = self._replicate_pad(v, pad)

        # (B, Blk, H, L_b, w, head_dim)
        def _unfold(t):
            return t.unfold(dimension=3, size=w, step=1).permute(0, 1, 2, 3, 5, 4).contiguous()

        k_local = _unfold(k_pad)
        u_local = _unfold(u_pad)
        v_local = _unfold(v_pad)

        # scores: (B, Blk, H, L_b, w, w)
        scores = torch.einsum(
            "bphid,bphijd,bphikd->bphijk",
            q, k_local, u_local
        ) / scale

        if mask_blocks is not None:
            mask_window = self._build_window_mask(mask_blocks).to(scores.device)
            scores = scores.masked_fill(mask_window == 0, -1e9)

        attn = softmax_nd(scores, dim=(-2, -1))                 # (B, Blk, H, L_b, w, w)

        # Outer-product value interaction: (B, Blk, H, L_b, w, w, head_dim)
        v_paired = torch.einsum("bphijd,bphikd->bphijkd", v_local, v_local)

        # Aggregate: (B, Blk, H, L_b, head_dim)
        return torch.einsum("bphijk,bphijkd->bphid", attn, v_paired)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute blockwise 3D attention.

        Args:
            x: Input sequence ``(B, L, d_model)``.
            mask: Optional ``(B, Blk, L_b)`` block-level mask from
                ``_generate_mask``; 1 = valid, 0 = padding.

        Returns:
            Output sequence ``(B, L, d_model)``.
        """
        B, L, _ = x.shape

        u_mat = self.W_u_v(self.W_u_u(x))  # low-rank U projection

        def _to_blocks_heads(t):
            return self._split_heads_blocks(
                self._sliding_blocks(t, self.block_size, self.stride)
            )

        Q = _to_blocks_heads(self.W_q(x))
        K = _to_blocks_heads(self.W_k(x))
        U_m = _to_blocks_heads(u_mat)
        V = _to_blocks_heads(self.W_v(x))

        mask_blocks = mask.float() if mask is not None else None
        out = self._triadic_attention(Q, K, U_m, V, mask_blocks=mask_blocks)

        # (B, Blk, H, L_b, Dh) → (B, Blk, L_b, D)
        B2, Blk, H, L_b, Dh = out.shape
        out = out.transpose(2, 3).contiguous().view(B2, Blk, L_b, self.d_model)

        out_full = self._reconstruct_from_blocks(out, L, self.stride)
        return self.W_o(out_full)
