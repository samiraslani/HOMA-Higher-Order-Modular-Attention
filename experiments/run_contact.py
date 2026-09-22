#!/usr/bin/env python
"""Contact prediction on ProteinNet across attention mechanisms and widths.

Produces the Contact Prediction panels of Figure 3 and its columns in
Supplementary Tables S8-S9: test P@L/5 over long-range pairs (|i-j| >= 24) on
the held-out CASP12 split, plus parameters, throughput and peak memory.

Data
----
The TAPE ProteinNet LMDBs (``proteinnet_{train,valid,test}.lmdb``).  The TAPE
S3 mirror now returns HTTP 403, so the data must be supplied locally; point
``--data`` or ``$PROTEINNET_DIR`` at the folder that directly contains
``proteinnet_train.lmdb``.  Coordinates are unit-checked on load (see
``homa.contact.data.check_units``).

Protocol
--------
12 layers, 8 heads, crop 256, 8 epochs, batch 8, AdamW at 1e-4 with a
OneCycle schedule (10 % warm-up), weight decay 0.01, gradient clipping at 1.0,
contact-class weight 5.  The contact class is ~2 % of scored pairs, so
unweighted cross-entropy is content to predict "no contact" everywhere, which
costs nothing in accuracy and everything in precision.

The published table reports the **end-of-training test** number, not the best
validation epoch: one test pass after the last epoch.  The best-validation
epoch is recorded too, for inspection only.

Cost
----
This is the expensive experiment.  The per-epoch times measured at d=128 on
an A100 (172-241 s per arm) extrapolate to roughly 7 hours per seed for the
4 arms x 4 widths grid, about 21 A100-hours for three seeds, and more in
practice since cost grows slightly faster than linearly in width.  It is resumable at run granularity and saves
after every epoch, so it can be spread over several sessions.

Usage
-----
    python experiments/run_contact.py --data /path/to/proteinnet
    python experiments/run_contact.py --data ... --widths 64 --arms homa
    python experiments/run_contact.py --data ... --quick    # pipeline check
    python experiments/run_contact.py --tables-only --out results/contact.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments.common import (DEFAULT_RESULTS_DIR, git_commit, int_list,
                                mean_sd, pick_device)
from homa.contact import (ContactDataset, build_contact_model,
                          collate_contacts, evaluate, find_proteinnet,
                          pair_memory_gb, silence_padding_log)

PUBLISHED = dict(n_layers=12, heads=8, max_len=256, epochs=8, batch=8,
                 lr=1e-4, pos_weight=5.0)
PUBLISHED_ARMS = ["plain2d", "blockwise2d", "blockwise3d", "homa"]
PUBLISHED_WIDTHS = [32, 64, 128, 256]
#: Seeds are per task family; the contact runs used these.
PUBLISHED_SEEDS = [0, 1, 2]

NAME = {"plain2d": "Pairwise-2D", "blockwise2d": "Blockwise-2D",
        "blockwise3d": "Blockwise-3D", "homa": "HOMA"}


def train_arm(attn_type, train_ds, val_ds, *, test_ds=None, seed=0, epochs=8,
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


def key_of(r) -> tuple:
    """Resume key: the epoch budget AND the seed.

    Without the budget, raising ``epochs`` would find old runs "done"; without
    the seed, every seed after the first would be skipped.
    """
    return (r["attn"], r["d_model"],
            r.get("config", {}).get("epochs", r.get("epochs")), r["seed"])


def save_atomic(payload, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def print_table(runs, arms, widths):
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None,
                   help="folder containing proteinnet_train.lmdb "
                        "(default: $PROTEINNET_DIR)")
    p.add_argument("--out", default=str(DEFAULT_RESULTS_DIR / "contact.json"))
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--arms", default=",".join(PUBLISHED_ARMS))
    p.add_argument("--widths", default=",".join(map(str, PUBLISHED_WIDTHS)))
    p.add_argument("--seeds", default=",".join(map(str, PUBLISHED_SEEDS)))
    p.add_argument("--epochs", type=int, default=PUBLISHED["epochs"])
    p.add_argument("--batch", type=int, default=PUBLISHED["batch"])
    p.add_argument("--quick", action="store_true",
                   help="tiny subsets and 1 epoch, to check the pipeline")
    p.add_argument("--tables-only", action="store_true")
    args = p.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    widths = int_list(args.widths)
    seeds = int_list(args.seeds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = json.loads(out.read_text()).get("runs", []) if out.exists() else []

    if not args.tables_only:
        device = pick_device(args.device)
        root = Path(args.data) if args.data else find_proteinnet()
        epochs = 1 if args.quick else args.epochs
        batch = 2 if args.quick else args.batch
        L = PUBLISHED["max_len"]

        print(f"proteinnet  {root}")
        print(f"device      {device}   commit {git_commit()}")
        print(f"protocol    {dict(PUBLISHED, epochs=epochs, batch=batch)}")
        for d in widths:
            print(f"  d={d:<4} pairwise-head activations "
                  f"~{pair_memory_gb(batch, L, d):.2f} GB")

        train_for = {s: ContactDataset(root / "proteinnet_train.lmdb",
                                       max_len=L, train=True, seed=s)
                     for s in seeds}
        val_ds = ContactDataset(root / "proteinnet_valid.lmdb", max_len=L)
        test_ds = ContactDataset(root / "proteinnet_test.lmdb", max_len=L)
        if args.quick:
            train_for = {s: Subset(ds, range(min(32, len(ds))))
                         for s, ds in train_for.items()}
            val_ds = Subset(val_ds, range(min(8, len(val_ds))))
            test_ds = Subset(test_ds, range(min(8, len(test_ds))))
        print(f"train {len(train_for[seeds[0]])} | valid {len(val_ds)} | "
              f"test {len(test_ds)}\n")

        done = {key_of(r) for r in runs}
        todo = [(s, d, a) for s in seeds for d in widths for a in arms
                if (a, d, epochs, s) not in done]
        print(f"{len(todo)} runs to go\n")

        for seed, d_model, arm in todo:
            print(f"\n=== {arm}  d_model={d_model}  seed={seed} ===", flush=True)

            def on_epoch(hist, _a=arm, _d=d_model, _s=seed):
                save_atomic({"runs": runs, "commit": git_commit(),
                             "in_progress": {"attn": _a, "d_model": _d,
                                             "seed": _s,
                                             "epochs_done": len(hist)}}, out)

            rec = train_arm(arm, train_for[seed], val_ds, test_ds=test_ds,
                            seed=seed, epochs=epochs, batch=batch,
                            lr=PUBLISHED["lr"],
                            pos_weight=PUBLISHED["pos_weight"], device=device,
                            d_model=d_model, n_layers=PUBLISHED["n_layers"],
                            heads=PUBLISHED["heads"], max_len=L,
                            eval_batches=4 if args.quick else 60,
                            on_epoch=on_epoch)
            runs.append(rec)
            save_atomic({"runs": runs, "commit": git_commit()}, out)
            print(f"  saved -> {len(runs)} runs | {rec['wall_s']}s | "
                  f"params {rec['params']:,}", flush=True)

    print_table(runs, arms, widths)


if __name__ == "__main__":
    main()
