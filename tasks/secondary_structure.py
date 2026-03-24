"""Secondary structure (SS3) task wrapper."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import AttentionConfig, ModelConfig, TrainingConfig
from data.collate import collate_ss3
from data.datasets import SecondaryStructureDataset
from evaluation.metrics import accuracy_per_position
from models.protein_transformer import PerResidueHead, ProteinTransformer
from training.trainer import Trainer

NUM_CLASSES = 3  # H (helix), E (strand), C (coil)


class SecondaryStructureTask:
    """End-to-end wrapper for secondary structure prediction (SS3).

    Encapsulates dataset creation, model construction, and training so that
    an experiment can be launched with a minimal amount of glue code.

    Example::

        task = SecondaryStructureTask(model_cfg, attn_cfg, train_cfg)
        model, history = task.train(
            train_lmdb=dataset_train,
            val_lmdb=dataset_valid,
            tokenizer=tokenizer,
        )

    Args:
        model_cfg: Architecture configuration.
        attn_cfg: Attention type and its parameters.
        train_cfg: Training hyperparameters.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        attn_cfg: AttentionConfig,
        train_cfg: TrainingConfig,
    ) -> None:
        self.model_cfg = model_cfg
        self.attn_cfg = attn_cfg
        self.train_cfg = train_cfg

    def build_model(self, pretrained_2d_ckpt=None) -> ProteinTransformer:
        """Instantiate the transformer model with a per-residue classification head."""
        head = PerResidueHead(
            d_model=self.model_cfg.d_model,
            num_classes=NUM_CLASSES,
        )
        return ProteinTransformer(
            model_cfg=self.model_cfg,
            attn_cfg=self.attn_cfg,
            head=head
        )

    def make_loader(self, lmdb_dataset, tokenizer, shuffle: bool) -> DataLoader:
        """Create a DataLoader for the given LMDB split."""
        dataset = SecondaryStructureDataset(lmdb_dataset, tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=shuffle,
            collate_fn=collate_ss3,
            num_workers=self.train_cfg.num_workers,
        )

    def train(
        self,
        train_lmdb,
        val_lmdb,
        tokenizer,
        pretrained_2d_ckpt=None,
        track_efficiency: bool = False,
    ):
        """Build, train, and return the model.

        Args:
            train_lmdb: TAPE ``LMDBDataset`` for training.
            val_lmdb: TAPE ``LMDBDataset`` for validation.
            tokenizer: TAPE tokenizer.
            pretrained_2d_ckpt: Optional checkpoint for 2D weight init.
            track_efficiency: Enable per-step timing and memory tracking.

        Returns:
            ``(model, history)``
        """
        model = self.build_model(pretrained_2d_ckpt)
        train_loader = self.make_loader(train_lmdb, tokenizer, shuffle=True)
        val_loader = self.make_loader(val_lmdb, tokenizer, shuffle=False)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        metric_fn = lambda logits, labels: accuracy_per_position(logits, labels)

        trainer = Trainer(
            config=self.train_cfg,
            attn_name=self.attn_cfg.type,
            select_by="val_loss",
        )
        return trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            metric_fn=metric_fn,
            is_classification=True,
            track_efficiency=track_efficiency,
        )
