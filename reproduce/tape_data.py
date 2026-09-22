"""Loader for the TAPE tables (used by make_tables.py).

Pulls the capacity sweep and the separately-run contact sweep into one shape:
    D[task][arm][d_model] = dict(acc, std, params, gpu, mem)
`acc` is the held-out test score -- for secondary structure and fluorescence
that is the test trajectory read at the best-validation epoch; contact logged
only one end-of-training test number, which is what it contributes.
"""
import json, os
import numpy as np

DL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

C  = {"plain2d":"#CC79A7","blockwise2d":"#D55E00","blockwise3d":"#0072B2","homa":"#009E73"}
MK = {"plain2d":"s","blockwise2d":"o","blockwise3d":"^","homa":"D"}
NM = {"plain2d":"Pairwise-2D","blockwise2d":"Blockwise-2D",
      "blockwise3d":"Blockwise-3D","homa":"HOMA"}
# Row labels in the heatmap.  Kept identical to NM so the panel labels and the
# figure legend name the same mechanism the same way.
SH = {"plain2d":"Pairwise-2D","blockwise2d":"Blockwise-2D",
      "blockwise3d":"Blockwise-3D","homa":"HOMA"}
ORD = ["plain2d","blockwise2d","blockwise3d","homa"]
PW  = ["plain2d","blockwise2d"]          # the pairwise arms
HI  = ["blockwise3d","homa"]             # the higher-order arms
TESTS = {"secondary_structure":["cb513","casp12","ts115"],"fluorescence":["test"]}
SEL   = {"secondary_structure":"val_loss","fluorescence":"val_metric"}
YLAB  = {"secondary_structure":"mean Q3","fluorescence":"Spearman $\\rho$",
         "contact":"test P@L/5 long"}
TITLE = {"secondary_structure":"Secondary Structure","fluorescence":"Fluorescence",
         "contact":"Contact Prediction"}


def load(src=None, csrc=None, ss_test=None):
    """`ss_test` picks ONE secondary-structure test set instead of averaging the
    three.  The default (None) keeps the historical behaviour -- the mean of
    CB513, CASP12 and TS115 -- which is what the figures reported before this
    argument existed.  The spread across the three is 0.024-0.046, larger than
    several differences the figure asks the reader to resolve, so which
    convention is in force has to be stated in the caption either way.
    """
    src  = src  or f"{DL}/tape.json"
    csrc = csrc or f"{DL}/contact.json"
    R = json.load(open(src))
    tests = dict(TESTS)
    if ss_test:
        assert ss_test in TESTS["secondary_structure"], ss_test
        tests["secondary_structure"] = [ss_test]

    def traj(h, t):
        cs = [np.array(h[f"test_{s}"], float) for s in tests[t] if f"test_{s}" in h]
        E = min(len(c) for c in cs)
        return np.mean([c[:E] for c in cs], axis=0)

    def best_ep(h, t):
        v = (-np.array(h["val_loss"], float) if SEL[t] == "val_loss"
             else np.array(h["val_metric"], float))
        return int(np.nanargmax(v))

    D = {}
    for t in ("secondary_structure", "fluorescence"):
        for m in ORD:
            for d in sorted(R[t].get(m, {}), key=int):
                hs = [h for h in R[t][m][d].values()
                      if isinstance(h, dict) and not h.get("oom")]
                if not hs: continue
                a = [float(traj(h, t)[min(best_ep(h, t), len(traj(h, t)) - 1)]) for h in hs]
                D.setdefault(t, {}).setdefault(m, {})[int(d)] = dict(
                    acc=float(np.mean(a)), std=float(np.std(a)),
                    params=float(np.mean([h["n_params_trainable"] for h in hs])),
                    gpu=float(np.mean([sum(h["epoch_wall_s"]) for h in hs])),
                    tput=float(np.mean([np.mean(h["tokens_per_sec_compute"]) for h in hs])),
                    mem=float(np.mean([max(h["peak_mem_alloc_gb"]) for h in hs])))

    if os.path.exists(csrc):          # contact: three seeds, one test number each
        raw = json.load(open(csrc))["runs"]
        # Two schemas exist in the wild: the original single-seed flattened
        # export, and the multi-seed sweep written by the resumable notebook.
        # Normalise both, then aggregate over seeds.
        by = {}
        for r in raw:
            if "attn" in r:                                   # multi-seed schema
                rec = dict(arm=r["attn"], d=int(r["d_model"]),
                           acc=float(r["test"]["P@L/5_long"]),
                           params=float(r["params"]), wall=float(r["wall_s"]),
                           mem=float(r["peak_mem_gb"]))
            else:                                             # flattened export
                rec = dict(arm=r["model"], d=int(r["d_model"]),
                           acc=float(r["test_p_at_l_over_5_long"]),
                           params=float(r["parameters"]),
                           wall=float(r["runtime_seconds"]),
                           mem=float(r["peak_memory_gb"]))
            by.setdefault((rec["arm"], rec["d"]), []).append(rec)

        for (arm, dm), rs in by.items():
            acc = [x["acc"] for x in rs]
            # No tokens/sec was logged for contact.  Every contact run trains the
            # same data for the same fixed 8 epochs, so the token count is a
            # constant across runs and 1/wall-clock is exactly proportional to
            # throughput -- correct for the RATIOS this is used for, but with no
            # absolute scale, so TPUT_ABS marks the task as relative-only.
            wall = float(np.mean([x["wall"] for x in rs]))
            D.setdefault("contact", {}).setdefault(arm, {})[dm] = dict(
                acc=float(np.mean(acc)), std=float(np.std(acc)),
                params=float(np.mean([x["params"] for x in rs])),
                gpu=wall, tput=1.0 / wall,
                mem=float(np.mean([x["mem"] for x in rs])))

    rows = [t for t in ("secondary_structure", "contact", "fluorescence") if t in D]
    return D, rows


# tasks whose `tput` is a real tokens/sec figure; elsewhere it is proportional
# to throughput but carries no absolute scale
TPUT_ABS = {"secondary_structure", "fluorescence"}


K_TOL = 1.5            # equivalence margin, in pooled seed s.d.
EPS_REL = 0.005        # absolute equivalence margin, as a fraction of the target


def _matches(v, tau, ref_std, k=K_TOL):
    """Does this cell match the reference accuracy, allowing for seed noise?

    A strict `acc >= tau` treats a shortfall inside the noise as a failure, and
    because width moves in 4x steps that can quadruple the reported multiple.
    On CB513 secondary structure, Blockwise-2D at d=256 scores 0.6218 against
    HOMA's 0.6251 at d=32 -- a 0.0032 gap on a pooled s.d. of 0.0030, i.e.
    1.1 sigma -- and the strict rule skipped it for d=512, turning 44x into
    175x.  A cell counts as matching when it is not more than k pooled standard
    deviations below the reference.

    K_TOL is 1.5 rather than 1.0 because that cell sits at 1.09 sigma: at k=1 it
    misses by 0.0003 and the reported multiple flips between 44x and 175x on
    that.  A 1.5-sigma margin is an ordinary equivalence band, it is the
    conservative direction (rivals match more easily, so HOMA's advantage is
    reported smaller).

    No threshold removes the problem, it only moves it: the multiple is a step
    function over a 4x width grid, so some cell always lands on the cut.  At
    k=1.5 the boundary case is contact prediction, HOMA d=128 against
    Blockwise-3D d=128, at exactly 1.50 sigma; at k=1.0 it was secondary
    structure, HOMA d=32 against Blockwise-2D d=256, at 1.09 sigma.  The claims
    that do not depend on the threshold are the ones separated by 3 sigma or
    more -- on secondary structure, that no pairwise width matches HOMA above
    d=64 (2.9 to 4.9 sigma).  Cells near the cut should be read as ties.

    A cell also counts as matching when it is within EPS_REL of the target in
    the task's own units, whatever the seed spread.  The sigma rule alone
    misreads a cell whose reference is unusually reproducible: on fluorescence
    HOMA at d=128 has a seed s.d. of 0.0008, so the pairwise arms' shortfalls
    of 0.0020 and 0.0024 -- three tenths of one percent of a Spearman rho --
    come out at 3.0 and 2.9 sigma and are rejected, although no one would call
    them a real difference.

    EPS_REL is 0.5% because the data leave a gap there.  Ranked by relative
    shortfall, the unmatched cells run 0.30%, 0.36%, then 1.41% and upward, so
    any threshold between roughly 0.4% and 1.4% admits exactly those two and
    nothing else; 0.5% sits in that gap rather than on either edge.  Widening
    the sigma margin instead would not work: reaching those two cells needs
    k = 3.1, which also admits five others, among them a 17% shortfall on
    contact prediction.
    """
    sd = np.sqrt((ref_std ** 2 + v.get("std", 0.0) ** 2) / 2)
    return v["acc"] >= tau - k * sd or v["acc"] >= tau * (1.0 - EPS_REL)


def iso_cost(D, t, m, tau, key, hi_better=False, ref_std=0.0, k=K_TOL):
    """the best this arm can do on `key` while still matching accuracy `tau`.

    A min (or max, on an axis where higher is better) over widths, not 'the
    smallest width that reaches it': accuracy is not perfectly monotone in
    width, and the claim is about the best model you could have shipped.
    """
    c = [v[key] for v in D[t].get(m, {}).values() if _matches(v, tau, ref_std, k)]
    if not c: return -np.inf if hi_better else np.inf
    return max(c) if hi_better else min(c)


def iso_group(D, t, arms, tau, key, hi_better=False, ref_std=0.0, k=K_TOL):
    """the best config across a group of arms, and the width it sits at."""
    vals = [iso_cost(D, t, m, tau, key, hi_better, ref_std, k)
            for m in arms if m in D[t]]
    c = (max(vals) if hi_better else min(vals)) if vals else (
        -np.inf if hi_better else np.inf)
    if not np.isfinite(c): return c, None
    d = min(d2 for m in arms if m in D[t] for d2, v in D[t][m].items()
            if _matches(v, tau, ref_std, k) and v[key] == c)
    return c, d
