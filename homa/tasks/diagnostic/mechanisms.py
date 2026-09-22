"""One registry of every attention arm the diagnostic tasks use, by paper name.

Everything a diagnostic experiment can instantiate is defined here once, and
the runners select arms by name.  Keeping one definition means the same arm is
the same module in every experiment; in the original working tree each
experiment built its arms inline and the copies drifted (``homa_add`` existed
for PARITY but had to be patched in at run time for MATCH).

Naming
------
The left column is the string the runners and result files use; the right
column is how the paper prints it.

    plain2d        Pairwise-2D     dense softmax attention
    pairwise2d     Pairwise-2D     alias of plain2d, used by the MATCH runner
    blockwise2d    Blockwise-2D    sliding-window pairwise attention
    blockwise3d    Blockwise-3D    triadic only, low-rank U, windowed
    homa           HOMA            pairwise + triadic, fused by an MLP
    homa_add       HOMA-add        pairwise + triadic, plain sum

On ``blockwise2d``: the diagnostic runners pass ``block_size = L``, a single
block spanning the whole sequence, so that blocking is not a confound at these
sequence lengths.  At that setting Blockwise-2D and Pairwise-2D are the same
operator, which is why the paper's PARITY table labels the row Pairwise-2D
while the result files key it ``blockwise2d``.

Ablation arms (the component teardown, Table 3 of the paper), each changing one
component of HOMA, with the fusion MLP (``homa_*``) or without it
(``homa_add_*``):

    ..._uniform    triadic softmax -> uniform average over the window
                   (keeps the V_j (*) V_k value, removes selection)
    ..._tiedu      U := K, i.e. no dedicated third projection
"""

from __future__ import annotations

from ...models.attention import attention_2d as a2
from ...models.attention import attention_3d as a3

__all__ = ["build_attention", "MECHANISMS", "PAPER_NAME", "is_known"]

#: Every arm the runners accept.  ``homa_w<k>`` (HOMA with an explicit triadic
#: window k) is also accepted and is not listed because the window is part of
#: the name.
MECHANISMS = (
    "plain2d", "pairwise2d", "blockwise2d", "blockwise3d",
    "homa", "homa_add",
    "homa_uniform", "homa_tiedu",
    "homa_add_uniform", "homa_add_tiedu",
)

#: How each arm is printed in the paper.
PAPER_NAME = {
    "plain2d": "Pairwise-2D",
    "pairwise2d": "Pairwise-2D",
    "blockwise2d": "Blockwise-2D",
    "blockwise3d": "Blockwise-3D",
    "homa": "HOMA",
    "homa_add": "HOMA-add",
    "homa_uniform": "HOMA, uniform pooling",
    "homa_tiedu": "HOMA, U=K",
    "homa_add_uniform": "HOMA-add, uniform pooling",
    "homa_add_tiedu": "HOMA-add, U=K",
}


def is_known(mech: str) -> bool:
    return mech in MECHANISMS or mech.startswith("homa_w")


def build_attention(mech: str, *, heads: int, d_model: int, block_size: int,
                    stride: int, window: int, rank: int):
    """Instantiate one attention module by its arm name.

    Args:
        mech: One of :data:`MECHANISMS`, or ``homa_w<k>`` for HOMA with an
            explicit triadic window of ``k`` (used by the coverage ablation).
        heads: Number of attention heads.
        d_model: Model width.
        block_size: Tokens per sliding block.  The diagnostic runners pass the
            sequence length, giving a single full-sequence block.
        stride: Step between block starts.
        window: Triadic window ``w``; must be odd so it is centred on the query.
        rank: Inner rank of the low-rank U factorisation.

    Returns:
        An ``nn.Module`` implementing ``forward(x, mask=None)``.

    Raises:
        ValueError: If ``mech`` is not a known arm.
    """
    # Explicit-window HOMA, e.g. "homa_w3".  Checked first because the suffix
    # is digits and would otherwise be caught by a prefix match below.
    if mech.startswith("homa_w"):
        return a3.HOMA(heads, d_model, stride=stride, block_size=block_size,
                       window_size=int(mech[len("homa_w"):]), rank=rank)

    if mech in ("plain2d", "pairwise2d"):
        return a2.MultiHeadAttn2D(heads, d_model)

    if mech == "blockwise2d":
        return a2.Attn2DBlockwise(heads, d_model, stride=stride,
                                  block_size=block_size)

    if mech == "blockwise3d":
        return a3.MultiHeadAttn3D(heads, d_model, block_size=block_size,
                                  stride=stride, window_size=window, rank=rank)

    if mech.startswith("homa"):
        # "homa_add_uniform" -> combine="add", variant="uniform"
        rest = mech[len("homa"):].lstrip("_")
        combine = "fusion"
        if rest.startswith("add"):
            combine = "add"
            rest = rest[len("add"):].lstrip("_")
        base = dict(stride=stride, block_size=block_size, window_size=window,
                    rank=rank, combine=combine)

        if rest == "":
            return a3.HOMA(heads, d_model, **base)
        if rest == "uniform":
            return a3.HOMA(heads, d_model, uniform_pool_3d=True, **base)
        if rest == "tiedu":
            return a3.HOMA(heads, d_model, tie_u_to_k=True, **base)

    raise ValueError(
        f"unknown mechanism {mech!r}; expected one of {MECHANISMS} "
        f"or 'homa_w<window>'")
