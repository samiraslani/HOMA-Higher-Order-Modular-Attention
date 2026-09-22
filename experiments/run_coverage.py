#!/usr/bin/env python
"""Triadic window coverage on PARITY-3: the window's reach against the task's.

Produces Figure 4 of the paper: the window cliff and the depth ablation.

Two reaches, which must not be confused
---------------------------------------
The **interaction reach** R is a property of the task: PARITY-3 at reach R
labels position i by the bits at offsets (-R, 0, +R), so the label depends on
positions up to R away.

The **window reach** is a property of the mechanism: the triadic window has
size w and is centred on the query, so one layer sees offsets up to
(w - 1) / 2 on each side.  Stacking L layers composes windows, and the
receptive field grows to L (w - 1) / 2.

The triadic branch can see the whole interaction exactly when
R <= L (w - 1) / 2.  The ablation varies R past that line on purpose and
measures what crossing it costs.

Phases
------
``cliff``   one layer, windows w = 3 and 7, R = 1..6.  Blockwise-3D should be
            perfect up to R = (w - 1) / 2 and fall to chance beyond.  HOMA should
            not fall below its pairwise branch, which has no window.
``depth``   window w = 3 (one-layer reach 1), R = 1..3, depth 1, 2, 4.  Asks
            whether stacking recovers what one window cannot see.

Controls held fixed across every cell: PARITY-3, d_model 32, L = 24.  The
labelled positions are pinned to those valid at the *largest* reach (12 of
them), so widening R does not also shrink the number of labels per sequence,
which would confound "wider interaction" with "fewer training signals".

Usage
-----
    python experiments/run_coverage.py
    python experiments/run_coverage.py --phases cliff --quick
    python experiments/run_coverage.py --tables-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (banner, base_parser, fmt_cell, int_list,
                                mean_sd, pick_device, ResultStore, run_jobs)
from tasks.diagnostic import PAPER_NAME
from tasks.diagnostic.parity_majority import centers_for, make_offsets, run_one

PUBLISHED_CFG = dict(seq_len=24, reach=3, heads=4, rank=8, stride=8,
                     train_n=3000, test_n=800, epochs=40, batch_size=128,
                     lr=2e-3)
QUICK_CFG = dict(PUBLISHED_CFG, train_n=400, test_n=200, epochs=4)

K = 3
D_MODEL = 32
REACHES = [1, 2, 3, 4, 5, 6]
CLIFF_WINDOWS = [3, 7]
CLIFF_ARMS = ["blockwise2d", "blockwise3d", "homa_add", "homa"]
DEPTH_WINDOW = 3
DEPTH_REACHES = [1, 2, 3]
DEPTHS = [1, 2, 4]
DEPTH_ARMS = ["blockwise3d", "homa_add", "homa"]


def pinned_centers(seq_len: int, reaches) -> list[int]:
    """Positions valid at the widest reach, used at every reach."""
    return centers_for(make_offsets(K, max(reaches)), seq_len)


def key_of(j) -> str:
    """Key format kept identical to the published ``results_order_capacity.json``."""
    return (f"coverage|{j['phase']}|{K}|{j['mech']}|r{j['reach']}"
            f"|w{j['window']}|L{j['n_layers']}|{j['seed']}")


def label_of(j) -> str:
    return (f"{j['phase']:<6} {j['mech']:<12} R={j['reach']} w={j['window']} "
            f"L={j['n_layers']} s={j['seed']}")


def build_jobs(args) -> list[dict]:
    jobs = []
    if "cliff" in args.phase_list:
        jobs += [dict(phase="cliff", mech=m, reach=R, window=w, n_layers=1,
                      seed=s)
                 for w in CLIFF_WINDOWS for R in REACHES for m in CLIFF_ARMS
                 for s in args.seed_list]
    if "depth" in args.phase_list:
        jobs += [dict(phase="depth", mech=m, reach=R, window=DEPTH_WINDOW,
                      n_layers=L, seed=s)
                 for R in DEPTH_REACHES for L in DEPTHS for m in DEPTH_ARMS
                 for s in args.seed_list]
    return jobs


def cell(runs, seeds, **j):
    return mean_sd(runs[key_of(dict(j, seed=s))]["final"] for s in seeds
                   if key_of(dict(j, seed=s)) in runs
                   and "final" in runs[key_of(dict(j, seed=s))])


def print_cliff(runs, seeds):
    for w in CLIFF_WINDOWS:
        half = (w - 1) // 2
        print(f"\n{'=' * 78}\n  Window cliff: w = {w}, one layer; the window "
              f"reaches +/-{half}\n{'=' * 78}")
        print("  " + "mechanism".ljust(15)
              + "".join(f"R={R}{'*' if R > half else ''}".rjust(14)
                        for R in REACHES))
        for m in CLIFF_ARMS:
            print("  " + PAPER_NAME.get(m, m).ljust(15) + "".join(
                fmt_cell(*cell(runs, seeds, phase="cliff", mech=m, reach=R,
                               window=w, n_layers=1)) for R in REACHES))
    print("\n  * the interaction reaches past the window: the triadic branch "
          "cannot see all three bits")


def print_depth(runs, seeds):
    half = (DEPTH_WINDOW - 1) // 2
    print(f"\n{'=' * 78}\n  Depth against reach: w = {DEPTH_WINDOW}; "
          f"L layers reach +/-L*{half}\n{'=' * 78}")
    print("  " + "mechanism".ljust(15) + "depth".rjust(6)
          + "".join(f"R={R} ".rjust(14) for R in DEPTH_REACHES))
    for m in DEPTH_ARMS:
        for L in DEPTHS:
            row = "  " + PAPER_NAME.get(m, m).ljust(15) + str(L).rjust(6)
            for R in DEPTH_REACHES:
                mu, sd, n = cell(runs, seeds, phase="depth", mech=m, reach=R,
                                 window=DEPTH_WINDOW, n_layers=L)
                mark = " " if R <= L * half else "*"
                row += fmt_cell(mu, sd, n, 13) + mark
            print(row)
        print()
    print("  * the interaction reaches past what L stacked windows can see")


def main() -> None:
    p = base_parser(__doc__, "coverage.json")
    p.add_argument("--phases", default="cliff,depth")
    args = p.parse_args()
    args.phase_list = [x for x in args.phases.split(",") if x]
    args.seed_list = int_list(args.seeds)

    cfg = dict(QUICK_CFG if args.quick else PUBLISHED_CFG)
    centers = pinned_centers(cfg["seq_len"], REACHES)
    device = pick_device(args.device)
    store = ResultStore(args.out, protocol=dict(cfg, k=K, d_model=D_MODEL),
                        meta=dict(phases=args.phase_list, seeds=args.seed_list,
                                  centers=centers))

    if not args.tables_only:
        banner("Triadic window coverage on PARITY-3", device, cfg)
        print(f"  labelled positions pinned to {centers} "
              f"({len(centers)} per sequence at every reach)\n")

        def run(j):
            rec = run_one(j["mech"], family="parity", k=K, d_model=D_MODEL,
                          seed=j["seed"], n_layers=j["n_layers"], cfg=cfg,
                          device=device, residual=True, reach=j["reach"],
                          window=j["window"], centers=centers)
            rec.update(sweep="coverage", phase=j["phase"], family="parity",
                       k=K, mechanism=j["mech"], d_model=D_MODEL,
                       seed=j["seed"])
            return rec

        run_jobs(store, build_jobs(args), key_of, run, label_of)

    runs = json.loads(Path(args.out).read_text())["runs"]
    if "cliff" in args.phase_list:
        print_cliff(runs, args.seed_list)
    if "depth" in args.phase_list:
        print_depth(runs, args.seed_list)


if __name__ == "__main__":
    main()
