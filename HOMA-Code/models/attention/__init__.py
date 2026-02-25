"""
Attention mechanism factory.

Import ``get_attention`` to instantiate any supported attention class by name.
"""

import logging
from typing import Any

from .attention_2d import Attn2D_MultiPed, Attn2DLinformer, MultiHeadAttn2D
from .attention_3d import Attn2DMultiPed3Dslidingw

logger = logging.getLogger(__name__)

# Supported attention type strings
_ATTENTION_TYPES = ("plain2d", "multiped2d", "linformer2d", "sliding3d")


def get_attention(attn_type: str, **kwargs: Any):
    """Instantiate an attention module by name.

    Args:
        attn_type: One of ``"plain2d"``, ``"multiped2d"``, ``"linformer2d"``,
            or ``"sliding3d"``.
        **kwargs: Parameters forwarded to the attention constructor.  Common
            keys: ``num_heads``, ``d_model``, ``len_seq``, ``block_size``,
            ``stride``, ``linformer_k`` (mapped to ``k``), ``window_size``,
            ``rank``, ``load_from_pretrained_2d``, ``pretrained_2d_ckpt``,
            ``freeze_2d``.

    Returns:
        An ``nn.Module`` instance of the requested attention class.

    Raises:
        ValueError: If ``attn_type`` is not one of the supported strings.
    """
    attn_type = attn_type.lower()
    logger.info("Instantiating attention: %s", attn_type)

    if attn_type == "plain2d":
        return MultiHeadAttn2D(
            num_heads=kwargs["num_heads"],
            d_model=kwargs["d_model"],
        )

    if attn_type == "multiped2d":
        return Attn2D_MultiPed(
            num_heads=kwargs["num_heads"],
            d_model=kwargs["d_model"],
            block_size=kwargs["block_size"],
            stride=kwargs["stride"],
        )

    if attn_type == "linformer2d":
        return Attn2DLinformer(
            num_heads=kwargs["num_heads"],
            d_model=kwargs["d_model"],
            k=kwargs.get("linformer_k", kwargs.get("k", 50)),
            len_seq=kwargs["len_seq"],
        )

    if attn_type == "sliding3d":
        return Attn2DMultiPed3Dslidingw(
            num_heads=kwargs["num_heads"],
            d_model=kwargs["d_model"],
            stride=kwargs["stride"],
            block_size=kwargs["block_size"],
            window_size=kwargs.get("window_size", 7),
            rank=kwargs.get("rank", kwargs.get("rank_3d", 8)),
            load_from_pretrained_2d=kwargs.get("load_from_pretrained_2d", False),
            pretrained_2d_ckpt=kwargs.get("pretrained_2d_ckpt", None),
            freeze_2d=kwargs.get("freeze_2d", False),
            prefix_hint=kwargs.get("prefix_hint", ""),
        )

    raise ValueError(
        f"Unknown attention type: '{attn_type}'. "
        f"Supported types: {_ATTENTION_TYPES}"
    )


__all__ = [
    "get_attention",
    "MultiHeadAttn2D",
    "Attn2D_MultiPed",
    "Attn2DLinformer",
    "Attn2DMultiPed3Dslidingw",
]
