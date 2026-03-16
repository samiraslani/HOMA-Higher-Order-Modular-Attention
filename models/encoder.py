"""Single transformer encoder layer: attention + FFN with residual connections."""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .attention import get_attention
from .feedforward import FeedForward

logger = logging.getLogger(__name__)

# Attention types that use sliding-window blocking and therefore need seq-len
# to be aligned to (block_size, stride) before positional embeddings are
# created.  The Encoder adjusts len_seq in kwargs accordingly.
_SLIDING_ATTENTION_TYPES = {"blockwise2d", "homa"}


class Encoder(nn.Module):
    """Single encoder layer: Multi-Head Attention → Add & Norm → FFN → Add & Norm.

    The attention mechanism is injected at construction time via
    :func:`get_attention`, so the same ``Encoder`` class supports all
    published attention variants.

    Args:
        attn_type: Attention type string forwarded to ``get_attention``.
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Hidden dimension of the feed-forward network.
        dropout: Dropout probability for both sub-layers.
        pretrained_2d_ckpt: Optional checkpoint path forwarded to the
            attention module for 2D weight pre-loading (only used by
            ``homa``).
        load_ffn_pretrained: If ``True``, load FFN weights from
            ``pretrained_2d_ckpt`` as well.
        freeze_ffn: If ``True``, freeze FFN weights after loading.
        **attn_kwargs: Additional keyword arguments forwarded to
            ``get_attention`` (e.g. ``len_seq``, ``block_size``, ``stride``).
    """

    def __init__(
        self,
        attn_type: str,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pretrained_2d_ckpt: Optional[str] = None,
        load_ffn_pretrained: bool = False,
        freeze_ffn: bool = False,
        **attn_kwargs,
    ) -> None:
        super().__init__()

        # Adjust len_seq for sliding-window attention so that
        # (len_seq - block_size) % stride == 0.
        if attn_type.lower() in _SLIDING_ATTENTION_TYPES:
            block_size = attn_kwargs.get("block_size")
            stride = attn_kwargs.get("stride")
            len_seq = attn_kwargs.get("len_seq")
            if block_size and stride and len_seq:
                remainder = (len_seq - block_size) % stride
                pad_len = (stride - remainder) % stride
                if pad_len > 0:
                    attn_kwargs["len_seq"] = len_seq + pad_len

        self.mha = get_attention(
            attn_type,
            d_model=d_model,
            num_heads=num_heads,
            pretrained_2d_ckpt=pretrained_2d_ckpt,
            **attn_kwargs,
        )

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Optionally load & freeze FFN from a 2D checkpoint
        if load_ffn_pretrained and pretrained_2d_ckpt:
            self._load_ffn_weights(pretrained_2d_ckpt)
        if freeze_ffn:
            for p in self.ffn.parameters():
                p.requires_grad_(False)
            logger.info("Froze FFN parameters.")

    def _load_ffn_weights(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        ff_keys = [k for k in state_dict if "encoder_layers" in k and "ffn" in k]
        own_state = self.ffn.state_dict()
        loaded = 0
        for name in own_state:
            match = [k for k in ff_keys if k.endswith(name)]
            if match:
                own_state[name].copy_(state_dict[match[0]])
                loaded += 1
        logger.info("Loaded %d / %d FFN weight tensors from checkpoint.", loaded, len(own_state))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pre-norm transformer encoder step.

        Args:
            x: ``(B, L, d_model)``
            mask: Optional mask forwarded to the attention module.

        Returns:
            ``(B, L, d_model)``
        """
        x = x + self.dropout(self.mha(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
