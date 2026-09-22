#!/usr/bin/env python
"""TAPE Secondary Structure and Fluorescence across mechanisms and widths.

Produces the Secondary Structure and Fluorescence panels of Figure 3 and
Supplementary Tables S8-S9 (CB513 / CASP12 / TS115, fluorescence at every
width, compute and peak memory, and the accuracy-matched comparison).  Contact
prediction is in ``run_contact.py``.

Protocol (as executed, which is not quite as Supplementary Table S3 states)
---------------------------------------------------------------------------
12 layers, 8 heads, FFN width 2 d_model, low-rank U rank 8,
block 30 / stride 15, triadic window 5.  Adam at a constant 1e-4, batch 16,
10 epochs, no weight decay, no learning-rate schedule, no gradient clipping.
Dropout 0.4 for Secondary Structure and 0.1 for Fluorescence.  Seeds 42, 456,
608.  Maximum length 512 for Secondary Structure and 237 for Fluorescence
(see ``TASK_SPEC``).

Supplementary Table S3 lists Fluorescence with a cosine schedule (6 % warm-up)
and gradient clipping at 1.0.  Every notebook that trained Secondary Structure
or Fluorescence used neither: the trainer builds a plain Adam and passes no
scheduler, and ``grad_clip`` was left at 0.  The warm-up, cosine and clipping
settings appear only in a later Stability-only run, and Stability is not in
the paper.  This runner reproduces what was executed.  ``--fl-schedule`` is
there if you want the protocol the table describes instead, but the numbers it
produces are not the published ones.

Selection and the reported number
---------------------------------
Model selection is by validation -- validation loss for Secondary Structure,
validation Spearman for Fluorescence -- and the reported test score is the
test metric *at the best-validation epoch*.  The trainer therefore evaluates
every test split at every epoch, records it, and never uses it for selection.

Data
----
TAPE LMDBs, one folder per task: ``--data-root/secondary_structure/`` and
``--data-root/fluorescence/``.  Files may be named ``<task>_<split>.lmdb`` (the
TAPE release) or ``<split>.lmdb``.  ``tape_proteins`` is not required.

Usage
-----
    python experiments/run_tape.py --data-root /path/to/tape
    python experiments/run_tape.py --data-root ... --tasks secondary_structure \\
        --widths 64 --arms homa --seeds 42
    python experiments/run_tape.py --data-root ... --quick
    python experiments/run_tape.py --tables-only --out results/tape.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from experiments.common import DEFAULT_RESULTS_DIR, git_commit, int_list, mean_sd, pick_device
from homa.config import AttentionConfig, ModelConfig, TrainingConfig
from homa.data.tape_compat import LMDBDataset, TAPETokenizer
from homa.evaluation.metrics import accuracy_per_position, spearman_correlation
from homa.tasks.fluorescence import FluorescenceTask
from homa.tasks.secondary_structure import SecondaryStructureTask
from homa.training.trajectory import TrajectoryTrainer, test_at_best_val
from homa.utils.seed import set_seed

N_LAYERS, NUM_HEADS, FF_MULT = 12, 8, 2
RANK = 8
BATCH_SIZE, LR, EPOCHS = 16, 1e-4, 10

ARMS = ["plain2d", "blockwise2d", "blockwise3d", "homa"]
SEEDS = [42, 456, 608]
WIDTHS = {"secondary_structure": [32, 64, 128, 256, 512],
          "fluorescence": [32, 64, 128, 256]}

TASK_SPEC = {
    "secondary_structure": dict(
        task_cls=SecondaryStructureTask, tests=["cb513", "casp12", "ts115"],
        is_classification=True, metric=accuracy_per_position,
        make_criterion=lambda: nn.CrossEntropyLoss(ignore_index=-100),
        select_by="val_loss", dropout=0.4, block=30, stride=15, window=5,
        max_len=512),
    "fluorescence": dict(
        task_cls=FluorescenceTask, tests=["test"],
        is_classification=False, metric=spearman_correlation,
        make_criterion=lambda: nn.MSELoss(),
        select_by="val_metric", dropout=0.1, block=30, stride=15, window=5,
        # 237, not 512.  It is the longest GFP sequence in the training split,
        # and it is what the published models used: their parameter counts
        # match this setting exactly at every width and every arm (16 of 16),
        # and match none at 512, because both the regression head's flattened
        # input and the positional table scale with it.  The head reads
        # max_len * d_model features, so this is not a padding detail -- it
        # changes the model.  encode() adds <cls> and <sep>, so inputs are 239
        # tokens and the encoder truncates them to 237, dropping the final
        # residue and <sep>.  Supplementary Table S1 states this as "fixed at
        # 237" (it said "pad to 512" before the reproduction check).
        max_len=237),
}

NAME = {"plain2d": "Pairwise-2D", "blockwise2d": "Blockwise-2D",
        "blockwise3d": "Blockwise-3D", "homa": "HOMA"}

TOKENIZER = TAPETokenizer(vocab="iupac")


def lmdb_path(root: Path, task: str, split: str) -> Path:
    for name in (f"{task}_{split}.lmdb", f"{split}.lmdb"):
        p = root / task / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no {task}_{split}.lmdb or {split}.lmdb under {root / task}")


def run_one(task_name, arm, d_model, seed, *, root, device, epochs, cap=None,
            fl_schedule=False):
    spec = TASK_SPEC[task_name]
    set_seed(seed)
    model_cfg = ModelConfig(d_model=d_model, num_layers=N_LAYERS,
                            num_heads=NUM_HEADS, dim_feedforward=FF_MULT * d_model,
                            dropout=spec["dropout"],
                            max_seq_length=spec["max_len"])
    attn_cfg = AttentionConfig(type=arm, block_size=spec["block"],
                               stride=spec["stride"], window_size=spec["window"],
                               rank_3d=RANK)
    extra = {}
    if fl_schedule and task_name == "fluorescence":
        extra = dict(grad_clip=1.0)   # the trainer ignores warmup/cosine; see docstring
    train_cfg = TrainingConfig(batch_size=BATCH_SIZE, learning_rate=LR,
                               epochs=epochs, device=device,
                               checkpoint_dir=str(DEFAULT_RESULTS_DIR / "ckpt"),
                               num_workers=0, **extra)
    task = spec["task_cls"](model_cfg, attn_cfg, train_cfg)
    model = task.build_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def loader(split, shuffle, n=None):
        ds = LMDBDataset(lmdb_path(root, task_name, split))
        if n is not None and n < len(ds):
            ds = Subset(ds, list(range(n)))
        return task.make_loader(ds, TOKENIZER, shuffle=shuffle)

    tl = loader("train", True, cap)
    vl = loader("valid", False, cap and max(8, cap // 4))
    tests = {s: loader(s, False, cap and max(8, cap // 4)) for s in spec["tests"]}

    trainer = TrajectoryTrainer(train_cfg,
                                attn_name=f"{task_name}_{arm}_d{d_model}_s{seed}")
    _, hist = trainer.fit_with_test(model, tl, vl, tests,
                                    spec["make_criterion"](), spec["metric"],
                                    spec["is_classification"])
    hist.update(n_params_trainable=n_params, batch_size=BATCH_SIZE,
                epochs=epochs, commit=git_commit())
    return hist


def summarise(results, task, arm, d, test):
    spec = TASK_SPEC[task]
    runs = [h for h in results.get(task, {}).get(arm, {}).get(str(d), {}).values()
            if isinstance(h, dict) and f"test_{test}" in h]
    return mean_sd([test_at_best_val(h, test, spec["select_by"]) for h in runs])


def print_tables(results, tasks, arms):
    for task in tasks:
        for test in TASK_SPEC[task]["tests"]:
            widths = sorted({int(d) for a in arms
                             for d in results.get(task, {}).get(a, {})})
            if not widths:
                continue
            metric = "Q3" if task == "secondary_structure" else "Spearman rho"
            print(f"\n{'=' * 78}\n  {task} / {test}: test {metric} at the "
                  f"best-validation epoch, mean +/- sd\n{'=' * 78}")
            print("  " + "mechanism".ljust(15)
                  + "".join(f"d={d}".rjust(19) for d in widths))
            for a in arms:
                row = "  " + NAME.get(a, a).ljust(15)
                for d in widths:
                    mu, sd, n = summarise(results, task, a, d, test)
                    row += ("--".rjust(19) if n == 0
                            else f"{mu:.3f}+/-{sd:.3f}({n})".rjust(19))
                print(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("TAPE_DATA_DIR"),
                   help="folder holding secondary_structure/ and fluorescence/")
    p.add_argument("--out", default=str(DEFAULT_RESULTS_DIR / "tape.json"))
    p.add_argument("--tasks", default="secondary_structure,fluorescence")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--widths", default=None,
                   help="comma list; default is the published range per task")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--fl-schedule", action="store_true",
                   help="clip gradients at 1.0 for fluorescence, as Table S3 "
                        "states; NOT what produced the published numbers")
    p.add_argument("--quick", action="store_true",
                   help="64 training sequences, 1 epoch: a pipeline check")
    p.add_argument("--tables-only", action="store_true")
    args = p.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    arms = [a for a in args.arms.split(",") if a]
    seeds = int_list(args.seeds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}

    if not args.tables_only:
        if not args.data_root:
            p.error("--data-root (or $TAPE_DATA_DIR) is required")
        root = Path(args.data_root)
        device = pick_device(args.device)
        epochs = 1 if args.quick else args.epochs
        cap = 64 if args.quick else None
        print(f"data   {root}\ndevice {device}   commit {git_commit()}")
        print(f"protocol: {N_LAYERS} layers, {NUM_HEADS} heads, FFN {FF_MULT}d, "
              f"batch {BATCH_SIZE}, Adam {LR} constant, {epochs} epochs\n")

        jobs = [(t, a, d, s) for t in tasks for s in seeds
                for d in (int_list(args.widths) if args.widths else WIDTHS[t])
                for a in arms]
        have = lambda t, a, d, s: str(s) in results.get(t, {}).get(a, {}).get(str(d), {})
        todo = [j for j in jobs if not have(*j)]
        print(f"{len(jobs)} runs, {len(jobs) - len(todo)} done, {len(todo)} to go")

        for t, a, d, s in todo:
            print(f"\n=== {t} / {a} / d={d} / seed={s} ===", flush=True)
            t0 = time.time()
            try:
                rec = run_one(t, a, d, s, root=root, device=device,
                              epochs=epochs, cap=cap, fl_schedule=args.fl_schedule)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # recorded, never retried at a smaller batch: that would
                # change the protocol for one cell of the table
                rec = {"oom": True, "batch_size": BATCH_SIZE}
                print("  OUT OF MEMORY -- recorded, batch NOT reduced")
            except Exception:
                traceback.print_exc()
                continue
            results.setdefault(t, {}).setdefault(a, {}).setdefault(str(d), {})[str(s)] = rec
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(results))
            tmp.replace(out)
            print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print_tables(results, tasks, arms)


if __name__ == "__main__":
    main()
