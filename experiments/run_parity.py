#!/usr/bin/env python
"""PARITY-k and MAJORITY-k across interaction order, depth and width.

Accuracy and per-epoch convergence curves for every order, depth and width, for
PARITY and the MAJORITY control.  Three presets:

    --preset grid       order x depth x width, residual at every depth
                        (the default)
    --preset capacity   PARITY-3 at depth 1 across widths 8-128, a single
                        attention layer with NO residual path
    --preset long       Pairwise-2D on PARITY-5 at d=64, depths 1, 6, 12,
                        rerun for 120 epochs instead of 40

The question
------------
PARITY-k is the degree-k monomial on +/-1-encoded bits: every strict subset of
the k positions is statistically independent of the label, so only a genuine
k-way interaction determines it.  MAJORITY-k is a threshold on the same sum
over the same bits, so it is order 1 and a pairwise mechanism separates it at
every k.  Running both is what makes a result readable: an effect that appears
on parity *and* on majority is not about interaction order.

Confound control
----------------
The k offsets are spread evenly over a fixed interval [-reach, +reach], so the
reach and the span are constant as k grows and one centred triadic window of
size ``2*reach+1`` covers every k.  Only k varies.  Boundary positions, where
the offset pattern would run off the sequence, are labelled -100 and excluded
from both the loss and the metric.

The backbone is deliberately impoverished -- embedding, attention, linear
readout, and no feed-forward sublayer.  An FFN can compute parity by itself and
would mask what the attention contributes; with a linear readout, a mechanism
that can only build degree-2 features cannot separate parity-3 in one layer.
That is what makes this a statement about attention rather than about the
network around it.

Depth is run with residual connections at EVERY depth including depth 1, so the
depth axis compares like with like.  A depth-1 point built without the residual
path would differ from the depth-12 point in architecture as well as in depth.

Usage
-----
    python experiments/run_parity.py                 # the published grid
    python experiments/run_parity.py --preset capacity
    python experiments/run_parity.py --preset long
    python experiments/run_parity.py --quick         # ~2 min, pipeline check
    python experiments/run_parity.py --depths 1 --orders 3,4,5
    python experiments/run_parity.py --tables-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (banner, base_parser, fmt_cell, int_list,
                                mean_sd, pick_device, ResultStore, run_jobs)
from tasks.diagnostic import PAPER_NAME, epochs_to
from tasks.diagnostic.parity_majority import run_one

#: The protocol behind every published parity number.
PUBLISHED_CFG = dict(seq_len=16, reach=3, heads=4, rank=8, stride=8,
                     train_n=3000, test_n=800, epochs=40, batch_size=128,
                     lr=2e-3)

#: k=3-5 are the higher orders; k=1,2 are the lower-order check.
PUBLISHED_ORDERS = [1, 2, 3, 4, 5]
PUBLISHED_DEPTHS = [1, 6, 12]
PUBLISHED_WIDTHS = [32, 64]
PUBLISHED_ARMS = ["blockwise2d", "blockwise3d", "homa_add", "homa"]
PUBLISHED_FAMILIES = ["parity", "majority"]

QUICK_CFG = dict(PUBLISHED_CFG, train_n=400, test_n=200, epochs=4,
                 batch_size=128)

#: Axis defaults, residual setting and key format per preset.  Keys follow the
#: published files so ``--tables-only`` can read them directly:
#:   grid      results_depth.json
#:   capacity  results_order_capacity.json (its capacity records)
#:   long      results_depth_long.json
PRESETS = {
    "grid": dict(orders=PUBLISHED_ORDERS, depths=PUBLISHED_DEPTHS,
                 widths=PUBLISHED_WIDTHS, families=PUBLISHED_FAMILIES,
                 arms=PUBLISHED_ARMS, residual=True, epochs=40),
    # The width axis was run as one attention layer applied directly,
    # without the residual path the depth grid uses; its parameter counts
    # match only that (e.g. HOMA at d=64: 25,362 here, 25,490 with the residual).
    # plain2d runs alongside blockwise2d because at one block the two are the
    # same operator; they should agree run for run, a cheap check that the
    # single-block claim holds in the code.
    "capacity": dict(orders=[3], depths=[1], widths=[8, 16, 32, 64, 128],
                     families=["parity"],
                     arms=["plain2d", "blockwise2d", "blockwise3d", "homa"],
                     residual=False, epochs=40),
    # A rerun at three times the budget of the one cell where
    # Pairwise-2D was still climbing at epoch 40.
    "long": dict(orders=[5], depths=[1, 6, 12], widths=[64],
                 families=["parity"], arms=["blockwise2d"],
                 residual=True, epochs=120),
}


def key_of(j) -> str:
    """Key format kept identical to the published result files.

    ``results_depth.json`` from the original run keys every record
    ``order|L<depth>|<family>|k<k>|<mech>|d<width>|s<seed>``.  Matching it means
    a published file can be dropped in as a starting point and this runner will
    correctly skip what it already contains instead of re-running it.
    """
    preset = j.get("preset", "grid")
    if preset == "capacity":
        return (f"capacity|{j['family']}|{j['k']}|{j['mech']}|"
                f"{j['d_model']}|{j['seed']}")
    prefix = "depth|order" if preset == "long" else "order"
    return (f"{prefix}|L{j['depth']}|{j['family']}|k{j['k']}|"
            f"{j['mech']}|d{j['d_model']}|s{j['seed']}")


def label_of(j) -> str:
    return (f"{j['family']}-{j['k']} {j['mech']:<12} "
            f"d={j['d_model']:<4} L={j['depth']:<3} s={j['seed']}")


def build_jobs(args) -> list[dict]:
    return [dict(depth=L, family=fam, k=k, mech=m, d_model=d, seed=s,
                 preset=args.preset)
            for s in args.seed_list
            for d in args.width_list
            for L in args.depth_list
            for fam in args.family_list
            for k in args.order_list
            for m in args.arm_list]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def cell(runs, *, depth, family, k, mech, d_model, seeds, field="final",
         preset="grid"):
    vals = []
    for s in seeds:
        r = runs.get(key_of(dict(depth=depth, family=family, k=k, mech=mech,
                                 d_model=d_model, seed=s, preset=preset)))
        if r and field in r:
            vals.append(r[field])
    return mean_sd(vals)


def epochs_cell(runs, *, depth, family, k, mech, d_model, seeds, thresh=0.90,
                preset="grid"):
    """Mean epoch to reach ``thresh``, over the seeds that reach it.

    Seeds that never cross are excluded from the mean rather than counted as
    the budget, and the count that did cross is returned so the table can mark
    it.  Averaging in a censored value would report a number that no run
    produced.
    """
    hit = []
    for s in seeds:
        r = runs.get(key_of(dict(depth=depth, family=family, k=k, mech=mech,
                                 d_model=d_model, seed=s, preset=preset)))
        if r and "curve" in r:
            e = epochs_to(r["curve"], thresh)
            if e is not None:
                hit.append(e)
    if not hit:
        return None, 0
    return sum(hit) / len(hit), len(hit)


def print_grid(runs, args):
    """Accuracy by order, depth and width, one block per width."""
    for fam in args.family_list:
        print(f"\n{'=' * 74}\n  {fam.upper()}-k: mean final test accuracy "
              f"+/- sd over {len(args.seed_list)} seeds\n{'=' * 74}")
        head = "  " + "width".ljust(7) + "mechanism".ljust(15)
        for k in args.order_list:
            for L in args.depth_list:
                head += f"k{k},L{L}".rjust(14)
        print(head)
        for d in args.width_list:
            for m in args.arm_list:
                row = "  " + str(d).ljust(7) + PAPER_NAME.get(m, m).ljust(15)
                for k in args.order_list:
                    for L in args.depth_list:
                        row += fmt_cell(*cell(runs, depth=L, family=fam, k=k,
                                              mech=m, d_model=d,
                                              seeds=args.seed_list,
                                              preset=args.preset))
                print(row)
            print()


def print_epochs(runs, args, thresh=0.90):
    """Epochs to 0.90 on parity."""
    print(f"\n{'=' * 74}\n  PARITY: mean epochs to {thresh:.2f} "
          f"(seeds reaching it, of {len(args.seed_list)})\n{'=' * 74}")
    head = "  " + "width".ljust(7) + "mechanism".ljust(15)
    for k in args.order_list:
        for L in args.depth_list:
            head += f"k{k},L{L}".rjust(14)
    print(head)
    for d in args.width_list:
        for m in args.arm_list:
            row = "  " + str(d).ljust(7) + PAPER_NAME.get(m, m).ljust(15)
            for k in args.order_list:
                for L in args.depth_list:
                    ep, n = epochs_cell(runs, depth=L, family="parity", k=k,
                                        mech=m, d_model=d, seeds=args.seed_list,
                                        thresh=thresh, preset=args.preset)
                    cellstr = (f">{args.cfg['epochs']}" if ep is None
                               else f"{ep:.1f}({n})")
                    row += cellstr.rjust(14)
            print(row)
        print()


def print_substitution(runs, args):
    """The headline: does depth let a pairwise stack reach the triadic result?"""
    if "homa" not in args.arm_list or "blockwise2d" not in args.arm_list:
        return
    print(f"\n{'=' * 74}\n  Does depth substitute for interaction order?"
          f"\n{'=' * 74}")
    print(f"  PARITY, gap = HOMA - Pairwise-2D\n")
    for d in args.width_list:
        print(f"  d_model = {d}")
        print("    " + "depth".rjust(6)
              + "".join(f"k={k}".rjust(22) for k in args.order_list))
        print("    " + "".rjust(6)
              + "".join("2D".rjust(8) + "HOMA".rjust(7) + "gap".rjust(7)
                        for _ in args.order_list))
        for L in args.depth_list:
            row = "    " + str(L).rjust(6)
            for k in args.order_list:
                b = cell(runs, depth=L, family="parity", k=k,
                         mech="blockwise2d", d_model=d, seeds=args.seed_list,
                         preset=args.preset)[0]
                h = cell(runs, depth=L, family="parity", k=k, mech="homa",
                         d_model=d, seeds=args.seed_list,
                         preset=args.preset)[0]
                row += f"{b:>8.3f}{h:>7.3f}{h - b:>+7.3f}"
            print(row)
        print()


def main() -> None:
    p = base_parser(__doc__, "parity.json")
    # the default output file follows the preset, set below
    p.add_argument("--preset", default="grid", choices=sorted(PRESETS))
    p.add_argument("--orders", default=None)
    p.add_argument("--depths", default=None)
    p.add_argument("--widths", default=None)
    p.add_argument("--families", default=None)
    p.add_argument("--mechanisms", default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="override the epoch budget")
    p.add_argument("--thresh", type=float, default=0.90,
                   help="accuracy threshold for the epochs-to table")
    args = p.parse_args()

    pre = PRESETS[args.preset]
    pick = lambda given, key: given if given is not None else ",".join(map(str, pre[key]))
    args.order_list = int_list(pick(args.orders, "orders"))
    args.depth_list = int_list(pick(args.depths, "depths"))
    args.width_list = int_list(pick(args.widths, "widths"))
    args.seed_list = int_list(args.seeds)
    args.family_list = [f for f in pick(args.families, "families").split(",") if f]
    args.arm_list = [m for m in pick(args.mechanisms, "arms").split(",") if m]
    residual = pre["residual"]

    cfg = dict(QUICK_CFG if args.quick else dict(PUBLISHED_CFG, epochs=pre["epochs"]))
    if args.epochs:
        cfg["epochs"] = args.epochs
    args.cfg = cfg

    if args.out.endswith("/parity.json") and args.preset != "grid":
        args.out = args.out[:-len("parity.json")] + f"parity_{args.preset}.json"

    device = pick_device(args.device)
    store = ResultStore(args.out, protocol=dict(cfg, residual=residual),
                        meta=dict(orders=args.order_list,
                                  depths=args.depth_list,
                                  widths=args.width_list,
                                  families=args.family_list,
                                  arms=args.arm_list,
                                  seeds=args.seed_list))

    if not args.tables_only:
        banner(f"PARITY-k / MAJORITY-k  [preset {args.preset}, "
               f"residual={residual}]", device, cfg)
        if args.quick:
            print("  !! --quick: reduced budget, these are NOT published "
                  "numbers\n")

        def run(j):
            rec = run_one(j["mech"], family=j["family"], k=j["k"],
                          d_model=j["d_model"], seed=j["seed"],
                          n_layers=j["depth"], cfg=cfg, device=device,
                          residual=residual)
            # Field names follow the published records, including both
            # spellings of the arm ("mech", "mechanism") they use.
            rec.update(axis="order", depth=j["depth"], family=j["family"],
                       k=j["k"], mech=j["mech"], mechanism=j["mech"],
                       d_model=j["d_model"], seed=j["seed"])
            if args.preset != "grid":
                rec["sweep"] = {"capacity": "capacity", "long": "depth"}[args.preset]
            return rec

        run_jobs(store, build_jobs(args), key_of, run, label_of)

    runs = json.loads(Path(args.out).read_text())["runs"]
    print_grid(runs, args)
    if "parity" in args.family_list:
        print_epochs(runs, args, args.thresh)
        print_substitution(runs, args)


if __name__ == "__main__":
    main()
