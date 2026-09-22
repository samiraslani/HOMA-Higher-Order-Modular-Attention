#!/usr/bin/env python
"""Compare a results file against the published cells, and say what moved.

    python reproduce/verify.py --parity results/parity.json
    python reproduce/verify.py --match  results/match.json
    python reproduce/verify.py --parity results/parity.json --tol 0.02

What "reproduced" means here
----------------------------
Not bit-identity.  The published runs were executed on CUDA; floating-point
reduction order differs between CUDA, CPU and Apple MPS, and on a task whose
accuracy can sit on a knife edge -- a parity cell is usually either ~0.50 or
~1.00 -- a different reduction order can move a single seed by a few
thousandths, occasionally more.  Seeding makes a run repeatable on the *same*
backend, not across backends.

So the comparison is per cell, on the seed mean, against a tolerance, and the
report separates two kinds of disagreement:

  drift      the cell moved but stayed on the same side of the separation the
             paper draws from it -- a chance-level cell is still at chance, a
             solved cell is still solved
  conflict   the cell crossed that line, which would contradict the claim

Only conflicts are failures.  A cell that the paper reports at 0.500 and a
rerun puts at 0.513 is drift; one the paper reports at 1.000 and a rerun puts
at 0.55 is a conflict, and worth investigating before trusting anything else
in the file.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Midpoint between "at chance" and "solved" on these two-class tasks.  A cell
#: is a conflict only if the rerun lands on the other side of this line from
#: the published value, by more than the tolerance.
SEPARATION = 0.75


def seed_mean(runs: dict, prefix: str, seeds) -> tuple[float | None, int]:
    vals = [runs[f"{prefix}|s{s}"]["final"] for s in seeds
            if f"{prefix}|s{s}" in runs and "final" in runs[f"{prefix}|s{s}"]]
    if not vals:
        return None, 0
    return statistics.mean(vals), len(vals)


def classify(pub: float, got: float, tol: float) -> str:
    if abs(got - pub) <= tol:
        return "match"
    same_side = (pub >= SEPARATION) == (got >= SEPARATION)
    return "drift" if same_side else "conflict"


def verify(results_path: Path, expected_path: Path, tol: float,
           seeds) -> int:
    blob = json.loads(results_path.read_text())
    runs = blob["runs"]
    exp = json.loads(expected_path.read_text())
    cells = exp["cells"]

    print(f"results  {results_path}")
    print(f"expected {expected_path}  ({exp['source']}, "
          f"{len(cells)} published cells)")
    print(f"protocol {blob.get('cfg')}")
    print(f"commit   {blob.get('commit', 'unknown')}")
    print(f"tolerance {tol:.3f} on the seed mean\n")

    rows, counts = [], {"match": 0, "drift": 0, "conflict": 0}
    for prefix, ref in sorted(cells.items()):
        got, n = seed_mean(runs, prefix, seeds)
        if got is None:
            continue                      # not run here; not a failure
        verdict = classify(ref["mean"], got, tol)
        counts[verdict] += 1
        rows.append((prefix, ref["mean"], ref["sd"], got, n, verdict))

    if not rows:
        print("No overlapping cells: this results file does not cover any "
              "published cell.")
        return 1

    print(f"{'cell':<44}{'published':>18}{'reproduced':>14}{'':>4}verdict")
    for prefix, pm, ps, got, n, verdict in rows:
        flag = {"match": "", "drift": "  ~", "conflict": "  !!"}[verdict]
        print(f"{prefix:<44}{pm:>10.3f}+/-{ps:<5.3f}{got:>14.3f}"
              f"{'(' + str(n) + ')':>4}{flag} {verdict}")

    total = len(rows)
    print(f"\n{total} cells compared: {counts['match']} match, "
          f"{counts['drift']} drift, {counts['conflict']} conflict")

    if counts["conflict"]:
        print("\nFAIL: at least one cell crossed the separation the paper "
              "draws from it.")
        return 1
    if counts["drift"]:
        print(f"\nPASS with drift: every cell stayed on the published side of "
              f"{SEPARATION:.2f}.\nDrift of a few thousandths between CUDA and "
              f"CPU/MPS is expected; see the module docstring.")
        return 0
    print("\nPASS: every compared cell is within tolerance.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parity", help="a results file from run_parity.py")
    p.add_argument("--match", help="a results file from run_match.py")
    p.add_argument("--tol", type=float, default=0.02,
                   help="allowed absolute difference on the seed mean")
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    if not args.parity and not args.match:
        p.error("pass --parity and/or --match")

    rc = 0
    if args.parity:
        print("=" * 78)
        print("  PARITY / MAJORITY")
        print("=" * 78)
        rc |= verify(Path(args.parity), HERE / "expected_parity.json",
                     args.tol, seeds)
    if args.match:
        print("\n" + "=" * 78)
        print("  MATCH")
        print("=" * 78)
        rc |= verify(Path(args.match), HERE / "expected_match.json",
                     args.tol, seeds)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
