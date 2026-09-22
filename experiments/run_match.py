#!/usr/bin/env python
"""MATCH2 and MATCH3: does some pair / triple sum to zero modulo M?

MATCH3 at N = 6 and 8 across widths d = 8-256 (the default), and the MATCH2
control (``--orders 2``).

The task
--------
    y_i = 1  iff  exists j        with  x_i + x_j       = 0 (mod M)   [MATCH2]
    y_i = 1  iff  exists j, k     with  x_i + x_j + x_k = 0 (mod M)   [MATCH3]

This is the family of Sanford et al. (2024).  MATCH2 is realisable by one
self-attention unit; MATCH3 is the smallest case their separation covers, and
a single unit of third-order attention computes it.

Two things this runner holds fixed, both of which silently ruin the comparison
if they move:

``M`` is calibrated per (N, order) so the majority-class baseline sits near
0.50.  MATCH3 label density grows like ``1 - exp(-N^2 / (2M))``, so a fixed M
across lengths drives the positive rate to 1 and a constant predictor scores
whatever that rate is.  The published moduli are pinned in
``tasks.diagnostic.match.PUBLISHED_M`` and asserted against the
calibration at startup, because M is a property of the task: if it differed
between two arms in a column, the accuracies in that column would not be
comparable.

The embedding is a frozen Fourier basis, not a learned lookup table.  MATCH3
otherwise asks the model to discover modular arithmetic *and* to route three
values through it, and the first is the famously slow problem and not the one
under test -- with a learned table both a pairwise and a triadic model plateau
near 0.74 and then overfit, which says nothing about interaction order.  The
Fourier basis is also the representation the theory assumes: the indicator of
``s = 0 (mod M)`` is ``(1/M) sum_w exp(2*pi*i*w*s/M)``, and with these features
a trilinear form expresses ``cos(w(x_i + x_j + x_k))`` exactly while a bilinear
one has no such expansion.  The projection on top stays learnable, so what is
frozen is the modular structure, not the model's choice of frequencies.

Usage
-----
    python experiments/run_match.py                   # MATCH3
    python experiments/run_match.py --orders 2        # the MATCH2 control
    python experiments/run_match.py --quick           # ~3 min, pipeline check
    python experiments/run_match.py --lengths 8 --widths 64 --tables-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (banner, base_parser, fmt_cell, int_list,
                                mean_sd, pick_device, ResultStore, run_jobs)
from tasks.diagnostic import PAPER_NAME, PUBLISHED_M
from tasks.diagnostic.match import calibrate_M, full_window, run_one

#: The protocol behind every published MATCH number.
PUBLISHED_CFG = dict(heads=4, epochs=40, train_n=30000, test_n=2000,
                     lr=3e-3, rank=8)

PUBLISHED_WIDTHS = [8, 16, 32, 64, 128, 256]
TABLE2_WIDTHS = [16, 32, 64, 128]
PUBLISHED_ARMS = ["pairwise2d", "blockwise3d", "homa_add", "homa"]
PUBLISHED_LENGTHS = [6, 8]

QUICK_CFG = dict(PUBLISHED_CFG, epochs=3, train_n=2000, test_n=500)


def key_of(j) -> str:
    """Key format kept identical to the published ``match_q_sweep.json``."""
    return f"q{j['order']}|N{j['N']}|{j['mech']}|d{j['d_model']}|s{j['seed']}"


def label_of(j) -> str:
    return (f"MATCH{j['order']} N={j['N']:<3} {j['mech']:<12} "
            f"d={j['d_model']:<4} s={j['seed']}")


def resolve_moduli(orders, lengths, strict=True) -> dict:
    """Modulus per (order, N), published where one exists.

    ``strict`` re-derives every published modulus from the calibration and
    fails on a mismatch.  That check is the point: it is what would catch a
    change to the search silently moving a published column onto a different
    task.
    """
    table = {}
    for q in orders:
        for N in lengths:
            pub = PUBLISHED_M.get((q, N))
            if pub is not None:
                if strict:
                    got = calibrate_M(N, q)
                    if got != pub:
                        raise SystemExit(
                            f"calibration drift at MATCH{q}, N={N}: published "
                            f"M={pub} but calibrate_M gives {got}. The task "
                            f"itself has changed; refusing to run.")
                table[(q, N)] = pub
            else:
                table[(q, N)] = calibrate_M(N, q)
    return table


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def cell(runs, *, order, N, mech, d_model, seeds, field="final"):
    vals = []
    for s in seeds:
        r = runs.get(key_of(dict(order=order, N=N, mech=mech,
                                 d_model=d_model, seed=s)))
        if r and field in r:
            vals.append(r[field])
    return mean_sd(vals)


def width_at_threshold(runs, *, order, N, mech, seeds, widths, thresh=0.90):
    """Smallest tested width whose seed mean reaches ``thresh``.

    Returns ``>max`` only when every width was actually run, so an unfinished
    sweep reads as ``--`` rather than as a mechanism that failed.
    """
    ran = [d for d in widths
           if cell(runs, order=order, N=N, mech=mech, d_model=d,
                   seeds=seeds)[2] > 0]
    hit = [d for d in ran
           if cell(runs, order=order, N=N, mech=mech, d_model=d,
                   seeds=seeds)[0] >= thresh]
    if hit:
        return min(hit)
    return f">{max(widths)}" if len(ran) == len(widths) else None


def print_accuracy(runs, args, moduli):
    print(f"\n{'=' * 78}\n  MATCH: mean final test accuracy +/- sd over "
          f"{len(args.seed_list)} seeds\n{'=' * 78}")
    for q in args.order_list:
        for N in args.length_list:
            print(f"\n  MATCH{q}  N={N}  (M={moduli[(q, N)]}, "
                  f"triadic window {full_window(N)})")
            print("    " + "mechanism".ljust(15)
                  + "".join(f"d={d}".rjust(14) for d in args.width_list))
            for m in args.arm_list:
                row = "    " + PAPER_NAME.get(m, m).ljust(15)
                for d in args.width_list:
                    row += fmt_cell(*cell(runs, order=q, N=N, mech=m,
                                          d_model=d, seeds=args.seed_list))
                print(row)


def print_threshold(runs, args, thresh=0.90):
    """Smallest width reaching the threshold, and its parameter multiple."""
    print(f"\n{'=' * 78}\n  Smallest width reaching {thresh:.2f} on the seed "
          f"mean, and its parameter count\n  relative to Pairwise-2D at the "
          f"same width\n{'=' * 78}")
    print("    " + "cell".ljust(18)
          + "".join(PAPER_NAME.get(m, m).rjust(22) for m in args.arm_list))
    for q in args.order_list:
        for N in args.length_list:
            row = "    " + f"MATCH{q} N={N}".ljust(18)
            for m in args.arm_list:
                w = width_at_threshold(runs, order=q, N=N, mech=m,
                                       seeds=args.seed_list,
                                       widths=args.width_list, thresh=thresh)
                if w is None:
                    row += "--".rjust(22)
                elif isinstance(w, str):
                    row += w.rjust(22)
                else:
                    p = cell(runs, order=q, N=N, mech=m, d_model=w,
                             seeds=args.seed_list, field="params")[0]
                    ref = cell(runs, order=q, N=N, mech="pairwise2d",
                               d_model=w, seeds=args.seed_list,
                               field="params")[0]
                    mult = f"{p / ref:.2f}x" if ref and ref == ref else "?"
                    row += f"{w} ({mult})".rjust(22)
            print(row)


def print_baseline(runs, args, moduli):
    """Majority-class baseline per cell, which is what an accuracy is read against."""
    print(f"\n{'=' * 78}\n  Majority-class baseline per cell "
          f"(calibration target 0.50)\n{'=' * 78}")
    for q in args.order_list:
        for N in args.length_list:
            mu = cell(runs, order=q, N=N, mech=args.arm_list[0],
                      d_model=args.width_list[0], seeds=args.seed_list,
                      field="majority")[0]
            print(f"    MATCH{q} N={N:<3} M={moduli[(q, N)]:<6} "
                  f"majority={mu:.3f}")


def main() -> None:
    p = base_parser(__doc__, "match.json")
    p.add_argument("--orders", default="3")
    p.add_argument("--lengths", default=",".join(map(str, PUBLISHED_LENGTHS)))
    p.add_argument("--widths", default=",".join(map(str, PUBLISHED_WIDTHS)))
    p.add_argument("--mechanisms", default=",".join(PUBLISHED_ARMS))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--thresh", type=float, default=0.90)
    p.add_argument("--allow-calibration-drift", action="store_true",
                   help="do not fail when the calibration no longer "
                        "reproduces a published modulus")
    args = p.parse_args()

    args.order_list = int_list(args.orders)
    bad = [q for q in args.order_list if q not in (2, 3)]
    if bad:
        p.error(f"--orders must be 2 (MATCH2) and/or 3 (MATCH3); got {bad}")
    args.length_list = int_list(args.lengths)
    args.width_list = int_list(args.widths)
    args.seed_list = int_list(args.seeds)
    args.arm_list = [m for m in args.mechanisms.split(",") if m]

    cfg = dict(QUICK_CFG if args.quick else PUBLISHED_CFG)
    if args.epochs:
        cfg["epochs"] = args.epochs

    moduli = resolve_moduli(args.order_list, args.length_list,
                            strict=not args.allow_calibration_drift)

    device = pick_device(args.device)
    store = ResultStore(args.out, protocol=cfg,
                        meta=dict(orders=args.order_list,
                                  lengths=args.length_list,
                                  widths=args.width_list,
                                  arms=args.arm_list, seeds=args.seed_list,
                                  M={f"q{q}|N{n}": v
                                     for (q, n), v in moduli.items()}))

    if not args.tables_only:
        banner("MATCH2 / MATCH3 across sequence length and width", device, cfg)
        if args.quick:
            print("  !! --quick: reduced budget, these are NOT published "
                  "numbers\n")
        print("  moduli: " + ", ".join(f"MATCH{q} N={n} -> M={v}"
                                       for (q, n), v in sorted(moduli.items())))
        print()

        jobs = [dict(order=q, N=N, mech=m, d_model=d, seed=s)
                for s in args.seed_list
                for q in args.order_list
                for N in args.length_list
                for m in args.arm_list
                for d in args.width_list]

        def run(j):
            rec = run_one(j["mech"], order=j["order"], N=j["N"],
                          M=moduli[(j["order"], j["N"])],
                          d_model=j["d_model"], seed=j["seed"],
                          device=device, **cfg)
            rec.update(order=j["order"], N=j["N"],
                       M=moduli[(j["order"], j["N"])], mech=j["mech"],
                       d_model=j["d_model"], seed=j["seed"], **cfg)
            return rec

        run_jobs(store, jobs, key_of, run, label_of)

    runs = json.loads(Path(args.out).read_text())["runs"]
    print_accuracy(runs, args, moduli)
    print_threshold(runs, args, args.thresh)
    print_baseline(runs, args, moduli)


if __name__ == "__main__":
    main()
