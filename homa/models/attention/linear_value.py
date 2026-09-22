"""Third-order routing with a first-order value: the fix implied by the
overfitting analysis.

The finding
-----------
On Match3 the triadic mechanisms overfit far harder than a pairwise one (train
0.925 vs 0.800 at test 0.698 vs 0.735, n = 6000).  Four hypotheses were tested
and three were rejected:

  * NOT parameter count - a pairwise model matched to HOMA's parameter count
    (d = 76, 31,070 params vs 30,882) has gap +0.075 against HOMA's +0.227.
  * NOT the number of interaction terms - the gap is flat from w = 3 (9 pairs,
    +0.198) to w = 15 (225 pairs, +0.227).
  * NOT the low-rank U - rank 8 -> 64 moves accuracy by under a point.
  * IT IS the value interaction.  ``uniform_pool_3d`` deletes the triadic
    softmax entirely and keeps only ``V_j (*) V_k``; the gap stays at +0.196.

The mechanism is an identity.  Under uniform attention the aggregation collapses

    sum_{j,k} (1/w^2) V_j (*) V_k = (mean_j V_j) (*) (mean_k V_k),

so *any* 3D variant emits a function QUADRATIC in the values, where a pairwise
branch emits a LINEAR one.  A quadratic form on ``head_dim`` coordinates carries
d(d+1)/2 degrees of freedom against d - 8.5x at head_dim 16 - at identical
parameter count.  That is capacity invisible to a parameter count and visible to
random-label memorisation, which is exactly what was measured (random-label
train accuracy: pairwise 0.755, Blockwise-3D 0.856, HOMA 0.940).

The fix
-------
The third-order *score* is what buys the Match3 representational win; the
second-order *value* is what costs the sample complexity.  Nothing forces them
to travel together - they are coupled only because the triadic branch was
written as a direct analogue of the pairwise one.  So keep the trilinear score
and the softmax over the (w, w) grid, and aggregate a value that is linear in V:

    quadratic (published)   res_i = sum_{j,k} A_ijk (V_j (*) V_k)
    linear-j                res_i = sum_j ( sum_k A_ijk ) V_j
    linear-sym              res_i = 1/2 [ sum_j (sum_k A_ijk) V_j
                                        + sum_k (sum_j A_ijk) V_k ]

``linear-j`` marginalises the triadic attention over ``k`` and uses the result
to weight ``V_j``.  Every position ``j`` that participates in a satisfying triple
receives mass, so third-order information still determines the routing; only the
value read out is first order.  ``linear-sym`` uses both marginals, which is not
redundant because the score is asymmetric in ``j`` and ``k`` (they are contracted
against ``K`` and ``U`` respectively).

Parameter count is identical to the published module in every case: marginalising
adds no weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention_3d import HOMA, MultiHeadAttn3D

AGGREGATIONS = ("quadratic", "linear_j", "linear_sym")


def aggregate(attn, v_local, mode: str) -> torch.Tensor:
    """Combine triadic attention with local values.

    Args:
        attn: ``(..., L_b, w, w)`` softmax over the (j, k) grid.
        v_local: ``(..., L_b, w, Dh)`` gathered values.
        mode: one of :data:`AGGREGATIONS`.

    Returns:
        ``(..., L_b, Dh)``
    """
    if mode == "quadratic":
        # the published aggregation, sum_{j,k} A_ijk V_j (*) V_k, computed
        # without materialising the (w, w, Dh) outer product
        return (torch.matmul(attn, v_local) * v_local).sum(dim=-2)
    if mode == "linear_j":
        return torch.matmul(attn.sum(dim=-1).unsqueeze(-2), v_local).squeeze(-2)
    if mode == "linear_sym":
        a_j = torch.matmul(attn.sum(dim=-1).unsqueeze(-2), v_local).squeeze(-2)
        a_k = torch.matmul(attn.sum(dim=-2).unsqueeze(-2), v_local).squeeze(-2)
        return 0.5 * (a_j + a_k)
    raise ValueError(f"unknown aggregation {mode!r}; expected one of {AGGREGATIONS}")


def make_linear_value_triadic():
    """``MultiHeadAttn3D`` with a selectable value aggregation."""

    class LinearValueAttn3D(MultiHeadAttn3D):

        def __init__(self, *args, value_agg: str = "linear_j", **kw):
            super().__init__(*args, **kw)
            if value_agg not in AGGREGATIONS:
                raise ValueError(value_agg)
            self.value_agg = value_agg

        def _triadic_attention(self, q, k, u, v, mask_blocks=None):
            pad, w = self.window_size // 2, self.window_size
            scale = self.head_dim ** 0.5

            def _unfold(t):
                return (t.unfold(dimension=3, size=w, step=1)
                         .permute(0, 1, 2, 3, 5, 4).contiguous())

            k_local = _unfold(self._replicate_pad(k, pad))
            u_local = _unfold(self._replicate_pad(u, pad))
            v_local = _unfold(self._replicate_pad(v, pad))

            qk_local = k_local * q.unsqueeze(-2)
            scores = torch.matmul(qk_local, u_local.transpose(-2, -1)) / scale
            if mask_blocks is not None:
                mask_window = self._build_window_mask(mask_blocks).to(scores.device)
                scores = scores.masked_fill(mask_window == 0, -1e9)
            attn = torch.softmax(scores.flatten(-2), dim=-1).view_as(scores)
            return aggregate(attn, v_local, self.value_agg)

    return LinearValueAttn3D


def make_linear_value_homa():
    """``HOMA`` with a selectable value aggregation on its triadic branch.

    ``forward`` is a copy of ``HOMA.forward`` with the aggregation swapped; the
    2D branch, the fusion and the block reconstruction are untouched.  The copy
    is pinned to the original numerically by :func:`verify_equivalence` at
    ``value_agg="quadratic"``.
    """

    class LinearValueHOMA(HOMA):

        def __init__(self, *args, value_agg: str = "linear_j", **kw):
            super().__init__(*args, **kw)
            if value_agg not in AGGREGATIONS:
                raise ValueError(value_agg)
            if self.uniform_pool_3d:
                raise NotImplementedError(
                    "uniform_pool_3d has no scores to marginalise")
            self.value_agg = value_agg

        def forward(self, x: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            B, L, _ = x.shape
            w, pad = self.window_size, self.window_size // 2

            q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
            u_mat = k if self.tie_u_to_k else self.W_u_v(self.W_u_u(x))

            def _to_blocks_heads(t):
                return self._split_heads_blocks(
                    self._sliding_blocks(t, self.block_size, self.stride))

            Q, K, V = map(_to_blocks_heads, (q, k, v))
            U_m = _to_blocks_heads(u_mat)
            B2, Blk, H, L_b, Dh = Q.shape

            # ---- 2D branch (unchanged) --------------------------------------
            scores_2d = torch.matmul(Q, K.transpose(-2, -1)) / (Dh ** 0.5)
            if mask is not None:
                scores_2d = scores_2d.masked_fill(
                    mask.unsqueeze(2).unsqueeze(3) == 0, -1e9)
            attn_out_2d = torch.matmul(torch.softmax(scores_2d, dim=-1), V)

            # ---- 3D branch: same score, selectable value --------------------
            def _unfold(t):
                return (t.unfold(dimension=3, size=w, step=1)
                         .permute(0, 1, 2, 3, 5, 4).contiguous())

            K_local = _unfold(self._replicate_pad(K, pad))
            U_local = _unfold(self._replicate_pad(U_m, pad))
            V_local = _unfold(self._replicate_pad(V, pad))

            scores_3d = torch.matmul(
                K_local * Q.unsqueeze(-2), U_local.transpose(-2, -1)) / (Dh ** 0.5)
            if mask is not None:
                mask_window = self._build_window_mask(mask).to(scores_3d.device)
                scores_3d = scores_3d.masked_fill(mask_window == 0, -1e9)
            attn_3d = torch.softmax(scores_3d.flatten(-2), dim=-1).view_as(scores_3d)

            p_u = attn_3d.sum(dim=-2)
            self.u_axis_entropy = -(p_u.clamp_min(1e-9)
                                    * p_u.clamp_min(1e-9).log()).sum(-1).mean()

            res_3d = aggregate(attn_3d, V_local, self.value_agg)

            if self.combine == "fusion":
                out = self.fusion_layer(torch.cat([attn_out_2d, res_3d], dim=-1))
            elif self.combine == "gated":
                out = attn_out_2d + self.gate * res_3d
            else:
                out = attn_out_2d + res_3d
            out = out.transpose(2, 3).contiguous().view(B2, Blk, L_b, self.d_model)
            return self.W_o(self._reconstruct_from_blocks(out, L, self.stride))

    return LinearValueHOMA


def verify_equivalence(*, num_heads=4, d_model=32, L=8, window=15, rank=8,
                       tol=1e-5, verbose=True) -> bool:
    """At ``value_agg="quadratic"`` the new modules must equal the originals.

    The same discipline as everywhere else in this study: a variant is not
    trusted until the configuration with a known answer reproduces it.  This is
    what pins the copied HOMA ``forward`` to the published one, so a difference
    measured later is attributable to the aggregation and to nothing else.
    """
    torch.manual_seed(0)
    x = torch.randn(3, L, d_model)
    ok = True

    ref3 = MultiHeadAttn3D(num_heads, d_model, block_size=L, stride=L,
                              window_size=window, rank=rank).eval()
    new3 = make_linear_value_triadic()(
        num_heads, d_model, block_size=L, stride=L, window_size=window,
        rank=rank, value_agg="quadratic").eval()
    new3.load_state_dict(ref3.state_dict())

    refh = HOMA(num_heads, d_model, stride=L, block_size=L,
                   window_size=window, rank=rank).eval()
    newh = make_linear_value_homa()(
        num_heads, d_model, stride=L, block_size=L, window_size=window,
        rank=rank, value_agg="quadratic").eval()
    newh.load_state_dict(refh.state_dict())

    with torch.no_grad():
        d3 = (ref3(x) - new3(x)).abs().max().item()
        dh = (refh(x) - newh(x)).abs().max().item()
    p3 = (sum(p.numel() for p in ref3.parameters()),
          sum(p.numel() for p in new3.parameters()))
    ph = (sum(p.numel() for p in refh.parameters()),
          sum(p.numel() for p in newh.parameters()))
    ok = d3 < tol and dh < tol and p3[0] == p3[1] and ph[0] == ph[1]

    # the linear aggregations must also leave the parameter count untouched
    for mode in ("linear_j", "linear_sym"):
        m = make_linear_value_homa()(
            num_heads, d_model, stride=L, block_size=L, window_size=window,
            rank=rank, value_agg=mode)
        ok &= sum(p.numel() for p in m.parameters()) == ph[0]

    if verbose:
        print(f"  Blockwise-3D  max|diff| = {d3:.2e}  params {p3[0]:,} vs {p3[1]:,}")
        print(f"  HOMA          max|diff| = {dh:.2e}  params {ph[0]:,} vs {ph[1]:,}")
        print(f"  equivalence at value_agg='quadratic': {'PASS' if ok else 'FAIL'}")
    return ok
