"""
Centralized configuration for the TAPE BioTransformer package.

All hyperparameters live here. Pass these dataclass instances to model,
data, and training constructors rather than scattering magic numbers through
the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters shared across all tasks.

    Attributes:
        vocab_size: Number of tokens in the amino-acid vocabulary (TAPE IUPAC tokenizer).
        d_model: Embedding / hidden dimension throughout the transformer.
        num_layers: Number of stacked encoder layers.
        num_heads: Number of attention heads (must divide d_model evenly).
        dim_feedforward: Hidden width of the position-wise feed-forward network.
        dropout: Dropout probability applied after attention and FFN sub-layers.
        max_seq_length: Maximum padded sequence length fed to the model.
    """
    vocab_size: int = 30
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.4
    max_seq_length: int = 512


@dataclass
class AttentionConfig:
    """Configuration for the attention mechanism.

    The ``type`` field selects which attention class is instantiated via
    ``get_attention()``.  Only the parameters relevant to the chosen type
    need to be set; unused ones are silently ignored.

    Supported types
    ---------------
    ``"plain2d"``
        Standard multi-head scaled dot-product attention.
    ``"multiped2d"``
        Sliding-window 2D attention (block_size, stride).
    ``"linformer2d"``
        Low-rank 2D attention (linformer_k, max_seq_length from ModelConfig).
    ``"homa"``  ← **main contribution**
        HOMA (Higher-Order MultiHead Attention) with low-rank L-matrix and optional
        pretrained-2D transfer (block_size, stride, window_size, rank_3d,
        pretrained_ckpt, freeze_2d).

    Attributes:
        type: One of the strings listed above.
        block_size: Block length for sliding-window attention variants.
        stride: Step size between consecutive blocks.
        linformer_k: Low-rank projection dimension for Linformer2D.
        window_size: Local context window for HOMA attention.
        rank_3d: Rank of the low-rank L-matrix decomposition in homa.
        pretrained_ckpt: Path to a checkpoint whose W_q/W_k/W_v weights are
            loaded into the 2D projections of homa.
        freeze_2d: If True, freeze the loaded 2D projection weights so only
            the 3D-specific parameters (W_l_u, W_l_v, fusion_layer) are
            trained.
    """
    type: str = "plain2d"

    # Sliding-window (multiped2d / homa)
    block_size: int = 40
    stride: int = 15

    # Linformer
    linformer_k: int = 50

    # 3D sliding-window specific
    window_size: int = 7
    rank_3d: int = 8

    # Transfer-learning for homa
    pretrained_ckpt: Optional[str] = None
    freeze_2d: bool = False


@dataclass
class TrainingConfig:
    """Optimisation and bookkeeping settings.

    Attributes:
        batch_size: Mini-batch size for the data loaders.
        learning_rate: Initial Adam learning rate.
        epochs: Total number of training epochs.
        warmup_steps: Steps excluded from efficiency timing at the start of
            each epoch (avoids measuring JIT-compilation overhead).
        checkpoint_dir: Directory where ``*.pt`` checkpoints are saved.
        num_workers: DataLoader worker processes.
        device: ``"cuda"`` or ``"cpu"`` (auto-detected if not set).
    """
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 20
    warmup_steps: int = 5
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 0
    device: Optional[str] = None
