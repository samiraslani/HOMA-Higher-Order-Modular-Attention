"""Trainer that records a test-set trajectory alongside validation.

This is the trainer behind every published Secondary Structure and
Fluorescence number.  It was originally defined inside the sweep notebook; it
lives here so the protocol is in the package rather than in a cell.

Why a trajectory and not one test number.  Model selection is by validation
(val loss for Secondary Structure, val Spearman for Fluorescence), and the
reported test score is the test value *at the best-validation epoch*.  Getting
that requires the test metric at every epoch, recorded but never used for
selection.  :func:`test_at_best_val` then reads it out.

Protocol notes, because the config object can mislead here.  ``fit_with_test``
builds a plain Adam at ``config.learning_rate`` and passes no scheduler, so
``warmup_ratio`` and ``lr_scheduler`` on the config are **not used** -- the
learning rate is constant for the whole run.  ``grad_clip`` *is* honoured,
because the shared epoch loop reads it; the published runs left it at its
default of 0.0, i.e. no clipping.  Both facts matter for reproduction.
"""

from __future__ import annotations

import numpy as np
import torch.optim as optim

from .efficiency import EfficiencyTracker
from .trainer import Trainer

EFF_KEYS = ["tokens_per_sec_compute", "avg_step_ms_compute", "epoch_wall_s",
            "peak_mem_alloc_gb"]


class TrajectoryTrainer(Trainer):
    """``Trainer`` with a per-epoch test evaluation on every test split."""

    def fit_with_test(self, model, train_loader, val_loader, test_loaders,
                      criterion, metric_fn, is_classification,
                      track_efficiency=True):
        """Train and return ``(model, history)``.

        ``history`` holds ``train_metric``, ``val_metric``, ``val_loss``, one
        ``test_<split>`` list per test loader, and the efficiency counters.
        """
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        history = {"train_metric": [], "val_metric": [], "val_loss": []}
        for name in test_loaders:
            history[f"test_{name}"] = []
        tracker = None
        if track_efficiency:
            tracker = EfficiencyTracker(device=str(self.device),
                                        warmup_steps=self.config.warmup_steps)
            for k in EFF_KEYS:
                history[k] = []

        for epoch in range(self.config.epochs):
            tr_loss, tr_metric = self._train_epoch(
                model, train_loader, optimizer, criterion,
                is_classification, tracker, metric_fn, None)
            if tracker is not None:
                eff = tracker.end_epoch()
                for k in EFF_KEYS:
                    history[k].append(eff[k])
            va_loss, va_metric = self._validate(model, val_loader, criterion,
                                                metric_fn, is_classification)
            history["train_metric"].append(tr_metric)
            history["val_metric"].append(va_metric)
            history["val_loss"].append(va_loss)
            msg = f"  ep {epoch + 1:2d}/{self.config.epochs} | val {va_metric:.4f}"
            for name, loader in test_loaders.items():
                _, tm = self._validate(model, loader, criterion, metric_fn,
                                       is_classification)
                history[f"test_{name}"].append(tm)
                msg += f" | {name} {tm:.4f}"
            if tracker is not None:
                msg += (f" | {eff['tokens_per_sec_compute']:,.0f} tok/s"
                        f" | {eff['peak_mem_alloc_gb']:.2f} GB")
            print(msg, flush=True)
        return model, history


def best_val_epoch(history: dict, select_by: str) -> int:
    """Index of the best-validation epoch.  Selection never reads test."""
    if select_by == "val_loss":
        return int(np.nanargmax(-np.asarray(history["val_loss"], float)))
    return int(np.nanargmax(np.asarray(history["val_metric"], float)))


def test_at_best_val(history: dict, split: str, select_by: str) -> float:
    """The reported test score: test ``split`` at the best-validation epoch."""
    curve = history[f"test_{split}"]
    return float(curve[min(best_val_epoch(history, select_by), len(curve) - 1)])
