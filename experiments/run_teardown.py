#!/usr/bin/env python
"""Component teardown of HOMA on PARITY-k and MATCH3.

Produces Table 3 of the main paper.  Each row changes exactly one component and
holds the others fixed, first with the fusion MLP in place and then again
without it, which is the two halves of the table.

The three components
--------------------
``pooling``       the triadic softmax over the (w, w) window, replaced by a
                  uniform average.  This removes *selection* while keeping the
                  V_j (*) V_k value interaction, so it separates "the triadic
                  attention chooses" from "the value product is rich".

``third factor``  the dedicated low-rank projection ``U``, replaced by reusing
                  the key: ``U := K``.  Isolates what a separate third
                  projection buys over the key it could have shared.

``value``         the quadratic read-out ``sum_{j,k} A_ijk (V_j (*) V_k)``,
                  replaced by marginalising the attention over ``k`` and
                  weighting ``V_j``: ``sum_j (sum_k A_ijk) V_j``.

The parity models here are one attention layer with **no residual path**, as in
the run that produced the published table.  Table 1's depth grid is built with
residual connections at every depth, including depth 1, so the two tables'
one-layer cells come from different models.

That last row is the one most easily misread, so it is worth being precise
about what it is not.  Marginalised attention is **not** pairwise attention.
Its attention row over ``j`` still shifts when a third token is perturbed,
which a pairwise score ``Q_i K_j^T`` cannot do; on the same input its
row-centred logit matrix has rank 23 against a pairwise ceiling of
``d_h + 1 = 5``.  The row ablates the value interaction, and does not revert to
a pairwise baseline.

``ep@0.90`` is averaged only over the seeds that reach 0.90 within the budget,
and the table marks how many did.  Averaging in the budget for a seed that
never crossed would report a number no run produced.

Usage
-----
    python experiments/run_teardown.py              # Table 3
    python experiments/run_teardown.py --quick
    python experiments/run_teardown.py --tasks parity
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (banner, base_parser, int_list, mean_sd,
                                pick_device, ResultStore, run_jobs)
from homa.synthetic import epochs_to
from homa.synthetic.match_tasks import run_one as match_run_one
from homa.synthetic.order_tasks import run_one as parity_run_one
from homa.synthetic.match_tasks import PUBLISHED_M

PARITY_CFG = dict(seq_len=16, reach=3, heads=4, rank=8, stride=8,
                  train_n=3000, test_n=800, epochs=40, batch_size=128,
                  lr=2e-3)
MATCH_CFG = dict(heads=4, epochs=40, train_n=30000, test_n=2000,
                 lr=3e-3, rank=8)

QUICK_PARITY = dict(PARITY_CFG, train_n=400, test_n=200, epochs=4)
QUICK_MATCH = dict(MATCH_CFG, epochs=3, train_n=2000, test_n=500)

#: Table 3 rows, in the order the table prints them.  The tuple is
#: (arm, pooling, third factor, value, fusion).
ROWS = [
    ("homa",             "softmax", "W(U)",      "Vj*Vk", "MLP"),
    ("homa_uniform",     "uniform", "W(U)",      "Vj*Vk", "MLP"),
    ("homa_tiedu",       "softmax", "W(U)=W(K)", "Vj*Vk", "MLP"),
    ("homa_linj",        "softmax", "W(U)",      "Vj",    "MLP"),
    ("homa_add",         "softmax", "W(U)",      "Vj*Vk", "sum"),
    ("homa_add_uniform", "uniform", "W(U)",      "Vj*Vk", "sum"),
    ("homa_add_tiedu",   "softmax", "W(U)=W(K)", "Vj*Vk", "sum"),
    ("homa_add_linj",    "softmax", "W(U)",      "Vj",    "sum"),
]

#: Published teardown setting: one attention layer, d_model 32, three seeds.
D_MODEL = 32
N_LAYERS = 1
MATCH_N = 6


def key_of(j) -> str:
    """Keys in the formats of the two published source files.

    The parity half of Table 3 is Sweep A of ``results_order_capacity.json``,
    keyed ``order|parity|<k>|<mech>|<width>|<seed>``; the MATCH3 half is
    ``match3_teardown.json``, keyed ``q3|N6|<mech>|d<width>|s<seed>``.  Using
    the same keys means either file can be read directly by ``--tables-only``.
    """
    t = j["task"]
    if t.startswith("parity"):
        return f"order|parity|{t[len('parity'):]}|{j['mech']}|{j['d_model']}|{j['seed']}"
    return f"q{t[len('match'):]}|N{MATCH_N}|{j['mech']}|d{j['d_model']}|s{j['seed']}"


def label_of(j) -> str:
    return f"{j['task']:<10} {j['mech']:<18} d={j['d_model']:<4} s={j['seed']}"


def build_jobs(args) -> list[dict]:
    return [dict(task=t, mech=m, d_model=args.width, seed=s)
            for s in args.seed_list
            for t in args.task_list
            for m in args.arm_list]


def print_table(runs, args, thresh=0.90):
    print(f"\n{'=' * 100}\n  Component teardown of HOMA "
          f"(d_model={args.width}, {N_LAYERS} layer, "
          f"{len(args.seed_list)} seeds)\n{'=' * 100}")
    head = ("  " + "pooling".ljust(9) + "third factor".ljust(13)
            + "value".ljust(8) + "fusion".ljust(8))
    for t in args.task_list:
        head += f"{t}: acc".rjust(14) + "ep@0.90".rjust(10)
    print(head)
    print("  " + "-" * (len(head) - 2))

    for mech, pool, third, value, fusion in ROWS:
        if mech not in args.arm_list:
            continue
        row = ("  " + pool.ljust(9) + third.ljust(13) + value.ljust(8)
               + fusion.ljust(8))
        for t in args.task_list:
            accs, eps = [], []
            for s in args.seed_list:
                r = runs.get(key_of(dict(task=t, mech=mech,
                                         d_model=args.width, seed=s)))
                if not r or "final" not in r:
                    continue
                accs.append(r["final"])
                e = epochs_to(r.get("curve", []), thresh)
                if e is not None:
                    eps.append(e)
            mu = mean_sd(accs)[0]
            row += ("--".rjust(14) if accs != accs or not accs
                    else f"{mu:.3f}".rjust(14))
            if not eps:
                row += f">{args.epoch_budget[t]}".rjust(10)
            else:
                mark = {len(args.seed_list): "", 1: "+", 2: "++"}.get(
                    len(eps), f"({len(eps)})")
                row += f"{sum(eps) / len(eps):.1f}{mark}".rjust(10)
        print(row)
    print("\n  + / ++ mark that one / two seeds reached the threshold; "
          "a bare number means all did.")
    print("  '>N' means no seed reached it within the N-epoch budget.")


def main() -> None:
    p = base_parser(__doc__, "teardown.json")
    p.add_argument("--tasks", default="parity3,parity4,parity5,match3")
    p.add_argument("--mechanisms", default=",".join(r[0] for r in ROWS))
    p.add_argument("--width", type=int, default=D_MODEL)
    p.add_argument("--thresh", type=float, default=0.90)
    args = p.parse_args()

    args.task_list = [t for t in args.tasks.split(",") if t]
    args.arm_list = [m for m in args.mechanisms.split(",") if m]
    args.seed_list = int_list(args.seeds)

    pcfg = dict(QUICK_PARITY if args.quick else PARITY_CFG)
    mcfg = dict(QUICK_MATCH if args.quick else MATCH_CFG)
    args.epoch_budget = {t: (mcfg if t.startswith("match") else pcfg)["epochs"]
                         for t in args.task_list}

    device = pick_device(args.device)
    store = ResultStore(args.out,
                        protocol={"parity": pcfg, "match": mcfg,
                                  "d_model": args.width, "n_layers": N_LAYERS},
                        meta=dict(tasks=args.task_list, arms=args.arm_list,
                                  seeds=args.seed_list))

    if not args.tables_only:
        banner("Component teardown of HOMA", device,
               {"parity": pcfg["epochs"], "match": mcfg["epochs"],
                "d_model": args.width})
        if args.quick:
            print("  !! --quick: reduced budget, these are NOT published "
                  "numbers\n")

        def run(j):
            task = j["task"]
            if task.startswith("parity"):
                k = int(task[len("parity"):])
                # residual=False: the published teardown is Sweep A, a single
                # attention layer applied directly, with no residual path.  Its
                # parameter counts match only this setting (8 of 8 arms).  The
                # depth grid of Table 1 uses residual=True at every depth, so a
                # one-layer HOMA-add cell there is a different model from the
                # HOMA-add row here and the two need not agree.
                rec = parity_run_one(j["mech"], family="parity", k=k,
                                     d_model=j["d_model"], seed=j["seed"],
                                     n_layers=N_LAYERS, cfg=pcfg,
                                     device=device, residual=False)
            elif task.startswith("match"):
                q = int(task[len("match"):])
                rec = match_run_one(j["mech"], order=q, N=MATCH_N,
                                    M=PUBLISHED_M[(q, MATCH_N)],
                                    d_model=j["d_model"], seed=j["seed"],
                                    device=device, **mcfg)
            else:
                raise ValueError(f"unknown task {task!r}")
            rec.update(task=task, mech=j["mech"], d_model=j["d_model"],
                       seed=j["seed"])
            return rec

        run_jobs(store, build_jobs(args), key_of, run, label_of)

    runs = json.loads(Path(args.out).read_text())["runs"]
    print_table(runs, args, args.thresh)


if __name__ == "__main__":
    main()
