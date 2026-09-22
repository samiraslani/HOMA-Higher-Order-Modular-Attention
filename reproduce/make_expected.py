#!/usr/bin/env python
"""Derive the published reference cells from the original result files.

This is the only script here that reads the original working tree.  It exists
so that ``reproduce/expected_*.json`` is *derived* rather than transcribed:
every number in those files is computed from the run records that produced the
paper, by the same aggregation the paper uses (mean and sample ``(n-1)``
standard deviation over seeds).  Nothing is typed in by hand, so a reference
value cannot drift from the table it is supposed to pin.

It is not part of reproduction and does not need to run on a fresh checkout --
``expected_parity.json`` and ``expected_match.json`` are committed.  Re-run it
only if the published result files change:

    python reproduce/make_expected.py --source /path/to/original/downloads
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: What each reference file pins, and where it comes from.
PARITY_SOURCE = "results_depth.json"
MATCH_SOURCE = "match_q_sweep.json"

PARITY_ARMS = ["blockwise2d", "blockwise3d", "homa_add", "homa"]
PARITY_WIDTHS = [32, 64]
PARITY_DEPTHS = [1, 6, 12]
PARITY_ORDERS = [1, 2, 3, 4, 5]
PARITY_FAMILIES = ["parity", "majority"]

MATCH_ARMS = ["pairwise2d", "blockwise3d", "homa_add", "homa"]
MATCH_WIDTHS = [8, 16, 32, 64, 128, 256]
MATCH_LENGTHS = [6, 8, 12, 16, 24]


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"mean": statistics.mean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def build_parity(src: Path, seeds=(0, 1, 2)) -> dict:
    runs = json.loads((src / PARITY_SOURCE).read_text())["runs"]
    cells = {}
    for fam in PARITY_FAMILIES:
        for d in PARITY_WIDTHS:
            for L in PARITY_DEPTHS:
                for k in PARITY_ORDERS:
                    for m in PARITY_ARMS:
                        got = agg(runs.get(f"order|L{L}|{fam}|k{k}|{m}|d{d}|s{s}",
                                           {}).get("final") for s in seeds)
                        if got:
                            cells[f"order|L{L}|{fam}|k{k}|{m}|d{d}"] = got
    return {"source": PARITY_SOURCE,
            "metric": "final test accuracy, mean and sample sd over seeds",
            "seeds": list(seeds), "cells": cells}


def build_match(src: Path, seeds=(0, 1, 2)) -> dict:
    blob = json.loads((src / MATCH_SOURCE).read_text())
    runs = blob["runs"]
    cells = {}
    for q in (2, 3, 4, 5):
        for N in MATCH_LENGTHS:
            for m in MATCH_ARMS:
                for d in MATCH_WIDTHS:
                    got = agg(runs.get(f"q{q}|N{N}|{m}|d{d}|s{s}",
                                       {}).get("final") for s in seeds)
                    if got:
                        cells[f"q{q}|N{N}|{m}|d{d}"] = got
    return {"source": MATCH_SOURCE,
            "metric": "final test accuracy, mean and sample sd over seeds",
            "seeds": list(seeds),
            "M": blob.get("M", {}),
            "cells": cells}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True,
                   help="directory holding the original result JSON files")
    args = p.parse_args()
    src = Path(args.source)

    for name, builder in (("expected_parity.json", build_parity),
                          ("expected_match.json", build_match)):
        blob = builder(src)
        (HERE / name).write_text(json.dumps(blob, indent=1))
        print(f"wrote {name}  ({len(blob['cells'])} cells from {blob['source']})")


if __name__ == "__main__":
    main()
