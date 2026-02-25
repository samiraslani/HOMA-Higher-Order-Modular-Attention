"""Position-wise feed-forward network used inside each encoder layer."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network with ReLU activation.

    Applied independently to each sequence position after the attention
    sub-layer.  Follows the standard transformer FFN:

    .. math::

        \\text{FFN}(x) = \\text{Linear}_2(\\text{Dropout}(\\text{ReLU}(\\text{Linear}_1(x))))

    Args:
        d_model: Input / output dimension.
        d_ff: Hidden (intermediate) dimension.
        dropout: Dropout probability applied after the first ReLU.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, L, d_model)``

        Returns:
            ``(B, L, d_model)``
        """
        return self.layer2(self.dropout(self.relu(self.layer1(x))))
