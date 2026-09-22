"""One registry of every attention arm the paper reports, by its paper name.

Why this module exists
----------------------
In the original working tree each experiment built its own arms inline, and the
three copies drifted.  ``homa_add`` -- the HOMA-add column of Table 1 and
Table 2 -- existed in the parity runner but not in the match runner, and the
match sweep supplied it by monkey-patching ``build_model`` from inside the
notebook at run time.  A reader of the committed source could not have found
it.  Everything an experiment can instantiate is therefore defined here once,
and the runners select by name.

Naming
------
The left column is the string the runners and the result files use; the right
column is how the paper prints it.

    plain2d        Pairwise-2D     dense softmax attention
    blockwise2d    Blockwise-2D    sliding-window pairwise attention
    pairwise2d     Pairwise-2D     alias of plain2d, used by the match runner
    blockwise3d    Blockwise-3D    triadic only, low-rank U, windowed
    plain3d        --              dense triadic, full-rank U (no restrictions)
    homa           HOMA            2D + 3D, fused by an MLP
    homa_add       HOMA-add        2D + 3D, plain sum (parameter-matched to 3D)

One caution about ``blockwise2d``.  The synthetic runners pass
``block_size = L``, i.e. a single block spanning the whole sequence, so that
blocking is not a confound at these sequence lengths.  At that setting
Blockwise-2D and Pairwise-2D are the *same operator*, which is why Table 1
labels the row Pairwise-2D while the result files key it ``blockwise2d``.
The distinction is real only on the TAPE tasks, where the sequences are long
and the block size is genuinely smaller than the sequence.

Teardown arms (Table 3).  Each changes exactly one component of HOMA:

    ..._uniform    triadic softmax -> uniform average over the window
                   (keeps the V_j (*) V_k value interaction, drops selection)
    ..._tiedu      U := K, i.e. no dedicated third projection
    ..._linj       value read-out marginalised over k: sum_j (sum_k A_ijk) V_j

each available both with the fusion MLP (``homa_*``) and without it
(``homa_add_*``), which is the two halves of Table 3.
"""

from __future__ import annotations

from ..models.attention import attention_2d as a2
from ..models.attention import attention_3d as a3
from ..models.attention.linear_value import make_linear_value_homa

__all__ = ["build_attention", "MECHANISMS", "PAPER_NAME", "is_known"]

#: Every arm the runners accept.  ``homa_w<k>`` is also accepted and is not
#: listed because the window is part of the name.
MECHANISMS = (
    "plain2d", "pairwise2d", "blockwise2d", "blockwise3d", "plain3d",
    "homa", "homa_add",
    "homa_uniform", "homa_tiedu", "homa_linj",
    "homa_add_uniform", "homa_add_tiedu", "homa_add_linj",
)

#: How each arm is printed in the paper.  Arms that appear only in the
#: supplement or only as a control map to their own name.
PAPER_NAME = {
    "plain2d": "Pairwise-2D",
    "pairwise2d": "Pairwise-2D",
    "blockwise2d": "Blockwise-2D",
    "blockwise3d": "Blockwise-3D",
    "plain3d": "Dense-3D",
    "homa": "HOMA",
    "homa_add": "HOMA-add",
    "homa_uniform": "HOMA, uniform pooling",
    "homa_tiedu": "HOMA, U=K",
    "homa_linj": "HOMA, linear value",
    "homa_add_uniform": "HOMA-add, uniform pooling",
    "homa_add_tiedu": "HOMA-add, U=K",
    "homa_add_linj": "HOMA-add, linear value",
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
        block_size: Tokens per sliding block.  The synthetic runners pass the
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

    if mech == "plain3d":
        # Dense triadic attention: every (i, j, k) triple, full-rank third
        # projection, no blocking and no window.  It is the unrestricted
        # operator that blockwise3d and HOMA are tractable restrictions of, so
        # it is NOT parameter-matched to them -- a win here reads "the
        # unrestricted operator does better", never "the window hurt".
        return a3.MultiHeadAttn3DPlain(heads, d_model)

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
        if rest == "linj":
            return make_linear_value_homa()(heads, d_model,
                                            value_agg="linear_j", **base)

    raise ValueError(
        f"unknown mechanism {mech!r}; expected one of {MECHANISMS} "
        f"or 'homa_w<window>'")
