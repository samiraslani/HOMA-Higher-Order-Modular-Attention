"""Shared plumbing for every experiment runner: resumable stores and devices.

The one rule these runners follow is that a result is on disk the moment it
exists.  Every sweep here is hours long and most were originally run on
preemptible Colab GPUs, so a runner that only writes at the end loses the whole
session to one disconnect.  :class:`ResultStore` writes after every run, via a
temp file and a rename so an interrupted write cannot truncate the file, and
skips on restart anything already keyed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

def pick_device(requested: str | None = None) -> str:
    """Resolve the device, preferring CUDA then Apple MPS then CPU.

    MPS is offered because it is what a laptop reproduction will actually use,
    but note that the published numbers were produced on CUDA.  Floating-point
    reduction order differs between backends, so a single run can land a few
    thousandths away from the published cell; the seed mean over three seeds is
    what should be compared.
    """
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    import torch

    if device == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return device


def git_commit() -> str:
    """Short HEAD of this repo, recorded in every result file."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse",
                              "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Resumable result store
# --------------------------------------------------------------------------

class ResultStore:
    """A JSON file of ``{key: record}`` that survives being interrupted.

    Args:
        path: Where to write.  Created with its parents if missing.
        protocol: The run configuration.  Stored alongside the runs and
            compared on reopen: a file written under a different protocol is
            refused rather than appended to, because mixing two protocols in
            one file produces a table whose cells are not comparable and
            nothing downstream would notice.
        meta: Extra fields to record (axis values, arms, seeds).
    """

    def __init__(self, path: str | Path, protocol: dict,
                 meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol
        self.meta = meta or {}
        self.runs: dict[str, Any] = {}

        if self.path.exists():
            blob = json.loads(self.path.read_text())
            self.runs = blob.get("runs", {})
            prev = blob.get("cfg")
            if prev is not None and prev != protocol:
                raise SystemExit(
                    f"{self.path} holds {len(self.runs)} runs under a "
                    f"DIFFERENT protocol:\n  saved: {prev}\n  now:   {protocol}\n"
                    "Delete or rename that file, or pass --out elsewhere.")
            print(f"resuming: {len(self.runs)} runs already in {self.path}")

    def __contains__(self, key: str) -> bool:
        return key in self.runs

    def get(self, key: str, default=None):
        return self.runs.get(key, default)

    def put(self, key: str, record: dict) -> None:
        self.runs[key] = record
        self.flush()

    def flush(self) -> None:
        payload = {"runs": self.runs, "cfg": self.protocol,
                   "commit": git_commit(), **self.meta}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path)   # atomic: never leaves a truncated file


def run_jobs(store: ResultStore, jobs: list, key_of: Callable,
             run: Callable, label: Callable) -> None:
    """Execute the jobs not already in ``store``, saving after each.

    A run that raises is recorded as an error record rather than killing the
    sweep, so one out-of-memory cell at the top of a width axis does not cost
    the rest of the grid.  Errors are counted in the closing summary.
    """
    todo = [j for j in jobs if key_of(j) not in store]
    print(f"{len(jobs)} runs total, {len(jobs) - len(todo)} done, "
          f"{len(todo)} to go\n")

    t0 = time.time()
    for n, job in enumerate(todo, 1):
        print(f"[{n:>4}/{len(todo)}] {label(job):<58} ", end="", flush=True)
        try:
            rec = run(job)
            note = (f"final={rec['final']:.3f}" if "final" in rec else "done")
        except Exception as exc:                  # noqa: BLE001 - recorded, not raised
            rec = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            note = f"FAILED: {type(exc).__name__}"
            _empty_cache()
        store.put(key_of(job), rec)
        el = (time.time() - t0) / 60
        eta = el / n * (len(todo) - n)
        print(f"{note}   [{el:.0f}m elapsed, eta {eta:.0f}m]", flush=True)

    errs = [k for k, v in store.runs.items() if "error" in v]
    print(f"\nwrote {store.path}  ({len(store.runs)} runs, {len(errs)} failed)")
    for k in errs[:5]:
        print(f"  failed: {k} - {store.runs[k]['error'][:80]}")


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Aggregation helpers used by the printed tables
# --------------------------------------------------------------------------

def mean_sd(values: Iterable[float]) -> tuple[float, float, int]:
    """Mean, sample ``(n-1)`` standard deviation, and count.

    The sample form is what the paper reports, so it is what the runners
    print.  ``numpy.std`` defaults to the population form and would print a
    visibly smaller spread on three seeds.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def fmt_cell(mu: float, sd: float, n: int, width: int = 14) -> str:
    if n == 0:
        return "--".rjust(width)
    return f"{mu:.3f}+/-{sd:.3f}".rjust(width)


# --------------------------------------------------------------------------
# Argument parsing shared by the synthetic runners
# --------------------------------------------------------------------------

def base_parser(description: str, default_out: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_RESULTS_DIR / default_out),
                   help="results JSON (resumable)")
    p.add_argument("--device", default=None,
                   choices=["cpu", "cuda", "mps"],
                   help="default: cuda, else mps, else cpu")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--quick", action="store_true",
                   help="tiny budget to check the pipeline end to end; the "
                        "numbers it produces are NOT the published ones")
    p.add_argument("--tables-only", action="store_true",
                   help="print the tables from an existing results file and exit")
    return p


def int_list(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x != ""]


def banner(title: str, device: str, protocol: dict) -> None:
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)
    print(f"  device   {describe_device(device)}")
    print(f"  commit   {git_commit()}")
    print(f"  protocol {protocol}")
    print()
