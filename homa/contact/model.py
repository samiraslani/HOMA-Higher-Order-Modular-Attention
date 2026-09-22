"""Contact-prediction model: the package encoder plus a pairwise head."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AttentionConfig, ModelConfig
from ..models.protein_transformer import ProteinTransformer
from .data import VOCAB_SIZE


class PairwiseContactHead(nn.Module):
    """``(B, L, d)`` -> ``(B, L, L, 2)`` contact logits, symmetric in (i, j).

    The pair feature is the concatenation of the elementwise product and the
    difference of the two residue representations.  The output is symmetrised
    explicitly because contact is a symmetric relation and nothing upstream
    enforces that.

    Args:
        d_model: Width of the encoder output.
        dropout: Applied to the residue representations before pairing.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(2 * d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        prod = x[:, :, None, :] * x[:, None, :, :]
        diff = x[:, :, None, :] - x[:, None, :, :]
        z = self.project(torch.cat([prod, diff], dim=-1))
        return 0.5 * (z + z.transpose(1, 2))


class _AlwaysEqual:
    """Compares equal to everything, so an ``if x != flag`` guard never fires."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


def silence_padding_log(model):
    """Stop the encoder logging its block-safe pad target once per step.

    ``ProteinTransformer._pad_to_blocks`` prints whenever its target changes
    and remembers the last one.  Sequences are padded to the *batch* maximum
    rather than to ``max_len``, so that target changes almost every batch --
    thousands of lines that bury the loss.  Replacing the remembered value with
    something that compares equal to everything suppresses the print without
    touching the padding itself.
    """
    model.encoder._last_reported_target = _AlwaysEqual()
    return model


class ContactModel(nn.Module):
    """Encoder with an identity head, and the pairwise head applied outside it.

    ``ProteinTransformer`` pads each batch up to a block-safe length so the
    sliding-window mechanisms tile evenly.  Taking the encoder output with
    ``head=nn.Identity()`` and slicing back to the real length *before* the
    pairwise head means the ``L x L`` tensor is never built at the padded
    length -- and that tensor is what bounds memory here.
    """

    def __init__(self, model_cfg: ModelConfig, attn_cfg: AttentionConfig) -> None:
        super().__init__()
        self.encoder = ProteinTransformer(model_cfg, attn_cfg, head=nn.Identity())
        self.head = PairwiseContactHead(model_cfg.d_model, model_cfg.dropout)

    def forward(self, input_ids: torch.Tensor, real_len: int) -> torch.Tensor:
        h = self.encoder(input_ids)             # (B, L_pad, d)
        return self.head(h[:, :real_len])       # (B, L, L, 2)


def build_contact_model(attn_type: str, *, d_model: int = 128,
                        n_layers: int = 12, heads: int = 8,
                        block_size: int = 32, stride: int = 16,
                        window_size: int = 5, rank_3d: int = 8,
                        combine: str = "fusion", dropout: float = 0.1,
                        **_ignored) -> ContactModel:
    """Build one contact model in the published configuration.

    Defaults are the published ones: 12 layers, 8 heads, block 32 / stride 16,
    triadic window 5, rank 8.  ``**_ignored`` absorbs sweep bookkeeping such as
    ``max_len`` so a single config dict can be passed straight through.
    """
    model_cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=d_model,
                            num_layers=n_layers, num_heads=heads,
                            dim_feedforward=2 * d_model, dropout=dropout,
                            max_seq_length=None)
    attn_cfg = AttentionConfig(type=attn_type, block_size=block_size,
                               stride=stride, window_size=window_size,
                               rank_3d=rank_3d, combine=combine)
    return ContactModel(model_cfg, attn_cfg)


def pair_memory_gb(batch: int, L: int, d_model: int, bytes_per: int = 4,
                   training: bool = True) -> float:
    """Activation cost of the pairwise head alone, in GB.

    ``prod`` and ``diff`` are each ``(B, L, L, d)`` and their concatenation is
    ``(B, L, L, 2d)``.  Backprop keeps them alive, hence the factor of two.
    Useful for sizing a run before it runs out of memory eight hours in.
    """
    elems = batch * L * L * d_model * 4        # prod + diff + concat(2d)
    return elems * bytes_per / 1e9 * (2 if training else 1)
