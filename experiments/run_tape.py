#!/usr/bin/env python
"""TAPE protein tasks: secondary structure, contact prediction, fluorescence.

One runner for the three protein-sequence tasks of the paper (Figure 3 and its
results tables): every attention mechanism at every width, over three seeds.

    python experiments/run_tape.py --data-root /path/to/tape
    python experiments/run_tape.py --data-root ... --tasks contact --widths 64
    python experiments/run_tape.py --data-root ... --quick     # pipeline check
    python experiments/run_tape.py --tables-only

Data layout under ``--data-root`` (or ``$TAPE_DATA_DIR``)::

    secondary_structure/   <task>_<split>.lmdb or <split>.lmdb
    fluorescence/          the same
    proteinnet/            proteinnet_{train,valid,test}.lmdb
                           (or pass --proteinnet, or set $PROTEINNET_DIR)

``tape_proteins`` is not required.  Results are written after every run to
``results/tape.json`` (secondary structure, fluorescence) and
``results/contact.json`` (contact prediction), and a restart skips what is
already there.

Protocol, as executed
---------------------
All tasks: 12 layers, 8 heads, FFN width 2 d_model, low-rank U rank 8,
triadic window 5.

Secondary structure and fluorescence: block 30 / stride 15, Adam at a constant
1e-4, batch 16, 10 epochs, no weight decay, no learning-rate schedule and no
gradient clipping; dropout 0.4 (secondary structure) and 0.1 (fluorescence);
maximum length 512 and 237; seeds 42, 456, 608.  The trainer builds a plain
Adam and passes no scheduler, so warm-up and cosine settings on the config are
not used, and ``grad_clip`` is left at 0.  The reported test score is the test
metric at the best-validation epoch: validation loss selects for secondary
structure, validation Spearman for fluorescence.

Contact prediction: block 32 / stride 16, crop 256, 8 epochs, batch 8, AdamW
at 1e-4 with a OneCycle schedule (10 % warm-up), weight decay 0.01, gradient
clipping at 1.0, contact-class weight 5, dropout 0.1, seeds 0, 1, 2.  The
contact class is about 2 % of scored pairs, so unweighted cross-entropy is
content to predict "no contact" everywhere.  The reported number is one test
pass after the last epoch; the best validation epoch is recorded for
inspection only.

Cost: secondary structure and fluorescence take hours per width on one GPU;
contact prediction about 21 A100-hours for the full grid.
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
from torch.utils.data import DataLoader, Subset

from experiments.common import DEFAULT_RESULTS_DIR, git_commit, int_list, mean_sd, pick_device
from config import AttentionConfig, ModelConfig, TrainingConfig
from data.tape_compat import LMDBDataset, TAPETokenizer
from evaluation.metrics import accuracy_per_position, spearman_correlation
from tasks.protein.contact_prediction import (ContactDataset, build_contact_model,
                                                   collate_contacts, evaluate,
                                                   find_proteinnet, pair_memory_gb,
                                                   silence_padding_log)
from tasks.protein.fluorescence import FluorescenceTask
from tasks.protein.secondary_structure import SecondaryStructureTask
from training.trajectory import TrajectoryTrainer, test_at_best_val
from utils.seed import set_seed

TASKS = ["secondary_structure", "contact", "fluorescence"]
ARMS = ["plain2d", "blockwise2d", "blockwise3d", "homa"]
NAME = {"plain2d": "Pairwise-2D", "blockwise2d": "Blockwise-2D",
        "blockwise3d": "Blockwise-3D", "homa": "HOMA"}
WIDTHS = {"secondary_structure": [32, 64, 128, 256, 512],
          "contact": [32, 64, 128, 256],
          "fluorescence": [32, 64, 128, 256]}
SEEDS = {"secondary_structure": [42, 456, 608], "fluorescence": [42, 456, 608],
         "contact": [0, 1, 2]}

# ---- secondary structure and fluorescence ---------------------------------
N_LAYERS, NUM_HEADS, FF_MULT = 12, 8, 2
RANK = 8
BATCH_SIZE, LR, EPOCHS = 16, 1e-4, 10

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
        # 237, not 512: the longest GFP sequence in the training split, and
        # what the published models used -- their parameter counts match this
        # setting at every width and arm and match none at 512, because the
        # regression head's flattened input and the positional table both
        # scale with it.  encode() adds <cls> and <sep>, so inputs are 239
        # tokens and the encoder truncates them to 237.
        max_len=237),
}

TOKENIZER = TAPETokenizer(vocab="iupac")

# ---- contact prediction ------------------------------------------------------
CONTACT = dict(n_layers=12, heads=8, max_len=256, epochs=8, batch=8,
               lr=1e-4, pos_weight=5.0)


def save_atomic(payload, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def lmdb_path(root: Path, task: str, split: str) -> Path:
    for name in (f"{task}_{split}.lmdb", f"{split}.lmdb"):
        p = root / task / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no {task}_{split}.lmdb or {split}.lmdb under {root / task}")


def run_one_tape(task_name, arm, d_model, seed, *, root, device, epochs, cap=None,
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


def print_tape_tables(results, tasks, arms):
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


def train_contact(attn_type, train_ds, val_ds, *, test_ds=None, seed=0, epochs=8,
              batch=8, lr=1e-4, pos_weight=5.0, device="cuda",
              eval_batches=60, log_every=1000, on_epoch=None, **kw) -> dict:
    """Train one mechanism and return a record complete enough to plot from.

    Everything a table might need is captured here rather than recomputed
    later: the per-epoch history, the best validation epoch, the end-of-training
    test pass, parameter counts, peak memory and throughput.  Re-running a run
    to recover a column that was not saved costs more than saving all of them.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                    collate_fn=collate_contacts, num_workers=0, drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch, shuffle=False,
                    collate_fn=collate_contacts, num_workers=0)

    model = silence_padding_log(build_contact_model(attn_type, **kw)).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(tl), pct_start=0.1)
    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=device), ignore_index=-1)

    hist, t0, n_res, n_step = [], time.time(), 0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        model.train()
        run, te = 0.0, time.time()
        for it, b in enumerate(tl):
            ids = b["input_ids"].to(device)
            tgt = b["contacts"].to(device)
            logits = model(ids, ids.shape[1])
            loss = lossf(logits.reshape(-1, 2), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            n_res += int(b["lengths"].sum())
            n_step += 1
            if log_every and it % log_every == 0:
                print(f"    ep{ep} it{it}/{len(tl)} loss {run / (it + 1):.4f}",
                      flush=True)
        m = evaluate(model, vl, device, max_batches=eval_batches)
        m.update(epoch=ep, train_loss=run / max(1, len(tl)),
                 epoch_s=round(time.time() - te, 1),
                 lr_end=sched.get_last_lr()[0])
        hist.append(m)
        print(f"  [{attn_type}] ep{ep}  loss {m['train_loss']:.4f}  "
              f"val P@L/5 long {m.get('P@L/5_long', float('nan')):.4f}",
              flush=True)
        if on_epoch is not None:
            on_epoch(hist)

    wall = time.time() - t0
    best_i = int(np.argmax([h.get("P@L/5_long", -1) for h in hist]))

    test_m = {}
    if test_ds is not None:
        tel = DataLoader(test_ds, batch_size=batch, shuffle=False,
                         collate_fn=collate_contacts, num_workers=0)
        test_m = evaluate(model, tel, device, max_batches=None)
        print(f"  [{attn_type}] TEST  P@L/5 long "
              f"{test_m.get('P@L/5_long', float('nan')):.4f}", flush=True)

    return {
        "attn": attn_type, "seed": seed, "d_model": kw.get("d_model", 128),
        "params": n_par, "params_total": n_tot,
        "history": hist, "final": hist[-1], "best": hist[best_i],
        "best_epoch": best_i,
        "best_long_L5": hist[best_i].get("P@L/5_long", float("nan")),
        "test": test_m,
        "wall_s": round(wall, 1),
        "epoch_s_mean": round(float(np.mean([h["epoch_s"] for h in hist])), 1),
        "steps_per_s": round(n_step / wall, 2),
        "residues_per_s": round(n_res / wall, 1),
        "peak_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                        if device == "cuda" else None),
        "config": dict(kw, epochs=epochs, batch=batch, lr=lr,
                       pos_weight=pos_weight, seed=seed),
    }


def contact_key(r) -> tuple:
    """Resume key: the epoch budget AND the seed.

    Without the budget, raising ``epochs`` would find old runs "done"; without
    the seed, every seed after the first would be skipped.
    """
    return (r["attn"], r["d_model"],
            r.get("config", {}).get("epochs", r.get("epochs")), r["seed"])


def print_contact_table(runs, arms, widths):
    print(f"\n{'=' * 78}\n  Contact prediction: test P@L/5, long range "
          f"(|i-j| >= 24), mean +/- sd over seeds\n{'=' * 78}")
    print("  " + "mechanism".ljust(15)
          + "".join(f"d={d}".rjust(19) for d in widths))
    for a in arms:
        row = "  " + NAME.get(a, a).ljust(15)
        for d in widths:
            v = [r["test"]["P@L/5_long"] for r in runs
                 if r["attn"] == a and r["d_model"] == d
                 and "P@L/5_long" in r.get("test", {})]
            mu, sd, n = mean_sd(v)
            row += ("--".rjust(19) if n == 0
                    else f"{mu:.3f}+/-{sd:.3f}({n})".rjust(19))
        print(row)

    print(f"\n  parameters and peak memory")
    for a in arms:
        row = "  " + NAME.get(a, a).ljust(15)
        for d in widths:
            rs = [r for r in runs if r["attn"] == a and r["d_model"] == d]
            if not rs:
                row += "--".rjust(19)
                continue
            mem = [r["peak_mem_gb"] for r in rs if r.get("peak_mem_gb")]
            memstr = f"{np.mean(mem):.2f}GB" if mem else "n/a"
            row += f"{rs[0]['params'] / 1e6:.2f}M {memstr}".rjust(19)
        print(row)

def tape_grid(tasks, args, root, device):
    out = Path(args.out_dir) / "tape.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    epochs = 1 if args.quick else args.epochs
    cap = 64 if args.quick else None
    jobs = [(t, a, d, s) for t in tasks
            for s in (int_list(args.seeds) if args.seeds else SEEDS[t])
            for d in (int_list(args.widths) if args.widths else WIDTHS[t])
            for a in args.arm_list]
    have = lambda t, a, d, s: str(s) in results.get(t, {}).get(a, {}).get(str(d), {})
    todo = [j for j in jobs if not have(*j)]
    print(f"secondary structure / fluorescence: {len(jobs)} runs, "
          f"{len(jobs) - len(todo)} done, {len(todo)} to go -> {out}")
    for t, a, d, s in todo:
        print(f"\n=== {t} / {a} / d={d} / seed={s} ===", flush=True)
        t0 = time.time()
        try:
            rec = run_one_tape(t, a, d, s, root=root, device=device,
                               epochs=epochs, cap=cap, fl_schedule=args.fl_schedule)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # recorded, never retried at a smaller batch: that would change
            # the protocol for one cell of the table
            rec = {"oom": True, "batch_size": BATCH_SIZE}
            print("  OUT OF MEMORY -- recorded, batch NOT reduced")
        except Exception:
            traceback.print_exc()
            continue
        results.setdefault(t, {}).setdefault(a, {}).setdefault(str(d), {})[str(s)] = rec
        save_atomic(results, out)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)


def contact_grid(args, root, device):
    out = Path(args.out_dir) / "contact.json"
    runs = json.loads(out.read_text()).get("runs", []) if out.exists() else []
    pn = (Path(args.proteinnet) if args.proteinnet
          else find_proteinnet([root / "proteinnet"] if root else []))
    epochs = 1 if args.quick else CONTACT["epochs"]
    batch = 2 if args.quick else CONTACT["batch"]
    L = CONTACT["max_len"]
    seeds = int_list(args.seeds) if args.seeds else SEEDS["contact"]
    widths = int_list(args.widths) if args.widths else WIDTHS["contact"]

    train_for = {s: ContactDataset(pn / "proteinnet_train.lmdb", max_len=L,
                                   train=True, seed=s) for s in seeds}
    val_ds = ContactDataset(pn / "proteinnet_valid.lmdb", max_len=L)
    test_ds = ContactDataset(pn / "proteinnet_test.lmdb", max_len=L)
    if args.quick:
        train_for = {s: Subset(ds, range(min(32, len(ds)))) for s, ds in train_for.items()}
        val_ds = Subset(val_ds, range(min(8, len(val_ds))))
        test_ds = Subset(test_ds, range(min(8, len(test_ds))))

    done = {contact_key(r) for r in runs}
    todo = [(s, d, a) for s in seeds for d in widths for a in args.arm_list
            if (a, d, epochs, s) not in done]
    print(f"contact prediction ({pn}): {len(todo)} runs to go -> {out}")
    for d in widths:
        print(f"  d={d:<4} pairwise-head activations ~{pair_memory_gb(batch, L, d):.2f} GB")

    for seed, d_model, arm in todo:
        print(f"\n=== contact / {arm} / d={d_model} / seed={seed} ===", flush=True)

        def on_epoch(hist, _a=arm, _d=d_model, _s=seed):
            save_atomic({"runs": runs, "commit": git_commit(),
                         "in_progress": {"attn": _a, "d_model": _d, "seed": _s,
                                         "epochs_done": len(hist)}}, out)

        rec = train_contact(arm, train_for[seed], val_ds, test_ds=test_ds,
                            seed=seed, epochs=epochs, batch=batch,
                            lr=CONTACT["lr"], pos_weight=CONTACT["pos_weight"],
                            device=device, d_model=d_model,
                            n_layers=CONTACT["n_layers"], heads=CONTACT["heads"],
                            max_len=L, eval_batches=4 if args.quick else 60,
                            on_epoch=on_epoch)
        runs.append(rec)
        save_atomic({"runs": runs, "commit": git_commit()}, out)
        print(f"  saved -> {len(runs)} runs | {rec['wall_s']}s | params {rec['params']:,}",
              flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("TAPE_DATA_DIR"),
                   help="folder holding secondary_structure/, fluorescence/, proteinnet/")
    p.add_argument("--proteinnet", default=None,
                   help="folder with proteinnet_*.lmdb, if not <data-root>/proteinnet")
    p.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--tasks", default=",".join(TASKS),
                   help="any of secondary_structure, contact, fluorescence")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--widths", default=None, help="default: the published range per task")
    p.add_argument("--seeds", default=None, help="default: the published seeds per task")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="secondary structure / fluorescence budget")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--fl-schedule", action="store_true",
                   help="clip gradients at 1.0 for fluorescence; NOT what produced "
                        "the published numbers")
    p.add_argument("--quick", action="store_true",
                   help="tiny subsets and 1 epoch: a pipeline check")
    p.add_argument("--tables-only", action="store_true")
    args = p.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        p.error(f"unknown task(s) {unknown}; choose from {TASKS}")
    args.arm_list = [a for a in args.arms.split(",") if a]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tape_tasks = [t for t in tasks if t != "contact"]

    if not args.tables_only:
        root = Path(args.data_root) if args.data_root else None
        if tape_tasks and root is None:
            p.error("--data-root (or $TAPE_DATA_DIR) is required")
        device = pick_device(args.device)
        print(f"data   {root}\ndevice {device}   commit {git_commit()}\n")
        if tape_tasks:
            tape_grid(tape_tasks, args, root, device)
        if "contact" in tasks:
            contact_grid(args, root, device)

    tape_out = Path(args.out_dir) / "tape.json"
    if tape_tasks and tape_out.exists():
        print_tape_tables(json.loads(tape_out.read_text()), tape_tasks, args.arm_list)
    contact_out = Path(args.out_dir) / "contact.json"
    if "contact" in tasks and contact_out.exists():
        print_contact_table(json.loads(contact_out.read_text()).get("runs", []),
                            args.arm_list, WIDTHS["contact"])


if __name__ == "__main__":
    main()
