"""Stability prediction task wrapper (global regression)."""

import torch.nn as nn
from torch.utils.data import DataLoader

from config import AttentionConfig, ModelConfig, TrainingConfig
from data.collate import collate_regression
from data.datasets import StabilityDataset
from evaluation.metrics import spearman_correlation
from models.protein_transformer import GlobalRegressionHead, ProteinTransformer
from training.trainer import Trainer


class StabilityTask:
    """End-to-end wrapper for protein stability prediction.

    Predicts the stability score of protein variants (single scalar per
    sequence) using a ``GlobalRegressionHead`` (mean-pool + MLP).

    Example::

        task = StabilityTask(model_cfg, attn_cfg, train_cfg)
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
        """Instantiate the transformer model with a global regression head.

        ``model_cfg.max_seq_length`` must be set: it determines both the
        padding length applied to every batch and the width of the classifier's
        first linear layer (``max_seq_length * d_model``).
        """
        if self.model_cfg.max_seq_length is None:
            raise ValueError(
                "ModelConfig.max_seq_length must be set for StabilityTask. "
                "It controls both the padding length and the classifier width."
            )
        len_seq = self.model_cfg.max_seq_length
        head = GlobalRegressionHead(
            d_model=self.model_cfg.d_model,
            len_seq=len_seq,
            d_ff=self.model_cfg.dim_feedforward,
        )
        return ProteinTransformer(
            model_cfg=self.model_cfg,
            attn_cfg=self.attn_cfg,
            head=head,
            pretrained_2d_ckpt=pretrained_2d_ckpt,
        )

    def make_loader(self, lmdb_dataset, tokenizer, shuffle: bool) -> DataLoader:
        dataset = StabilityDataset(lmdb_dataset, tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=shuffle,
            collate_fn=collate_regression,
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

        Returns:
            ``(model, history)``
        """
        model = self.build_model(pretrained_2d_ckpt)
        train_loader = self.make_loader(train_lmdb, tokenizer, shuffle=True)
        val_loader = self.make_loader(val_lmdb, tokenizer, shuffle=False)

        criterion = nn.MSELoss()

        trainer = Trainer(
            config=self.train_cfg,
            attn_name=f"stability_{self.attn_cfg.type}",
            select_by="val_metric",
        )
        return trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            metric_fn=spearman_correlation,
            is_classification=False,
            track_efficiency=track_efficiency,
        )
