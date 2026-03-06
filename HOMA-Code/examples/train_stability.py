"""
Example: train a BioTransformer for protein stability prediction.

Usage
-----
From the repository root::

    python examples/train_stability.py

TAPE LMDB datasets are expected at the paths set in the ``DATA_DIR``
variable below.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tape.datasets import LMDBDataset
from tape.tokenizers import TAPETokenizer

from tape_biotransformer.config import AttentionConfig, ModelConfig, TrainingConfig
from tape_biotransformer.tasks.stability import StabilityTask
from tape_biotransformer.utils.seed import set_seed

DATA_DIR = os.environ.get("TAPE_DATA_DIR", "/path/to/tape/data/stability")
CHECKPOINT_DIR = "checkpoints-stability"

model_cfg = ModelConfig(
    vocab_size=30,
    d_model=512,
    num_layers=12,
    num_heads=8,
    dim_feedforward=1024,
    dropout=0.4,
    max_seq_length=512,
)

train_cfg = TrainingConfig(
    batch_size=16,
    learning_rate=1e-4,
    epochs=20,
    warmup_steps=5,
    checkpoint_dir=CHECKPOINT_DIR,
)

tokenizer = TAPETokenizer(vocab="iupac")
dataset_train = LMDBDataset(os.path.join(DATA_DIR, "train.lmdb"))
dataset_valid = LMDBDataset(os.path.join(DATA_DIR, "valid.lmdb"))


def run(attn_cfg: AttentionConfig, pretrained_ckpt=None) -> None:
    set_seed(42)
    print(f"\n{'='*60}")
    print(f"Attention type: {attn_cfg.type}")
    print(f"{'='*60}\n")

    task = StabilityTask(model_cfg, attn_cfg, train_cfg)
    model, history = task.train(
        train_lmdb=dataset_train,
        val_lmdb=dataset_valid,
        tokenizer=tokenizer,
        pretrained_2d_ckpt=pretrained_ckpt,
        track_efficiency=False,
    )

    best_val = max(history["val_metric"])
    print(f"\nBest validation Spearman ρ ({attn_cfg.type}): {best_val:.4f}\n")


if __name__ == "__main__":
    run(AttentionConfig(type="plain2d"))
    run(AttentionConfig(type="multiped2d", block_size=40, stride=15))

    pretrained = os.path.join(CHECKPOINT_DIR, "stability_multiped2d.pt")
    run(
        AttentionConfig(
            type="homa",
            block_size=40,
            stride=15,
            window_size=7,
            rank_3d=8,
            pretrained_ckpt=pretrained if os.path.exists(pretrained) else None,
            freeze_2d=False,
        ),
        pretrained_ckpt=pretrained if os.path.exists(pretrained) else None,
    )
