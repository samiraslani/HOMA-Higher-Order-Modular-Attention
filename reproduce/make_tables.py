#!/usr/bin/env python
"""Every supplementary table body, rebuilt from result files.

This is the script that generated the tables in the supplement, with its inputs
made configurable.  Run on the published result files it reproduces the
supplement's table bodies exactly; run on the files this package's runners
write, it rebuilds the same tables from a reproduction.

    python reproduce/make_tables.py                  # from results/
    python reproduce/make_tables.py --published DIR  # from the original files
    python reproduce/make_tables.py > tables.tex

Inputs (defaults are what the runners write under results/):

    --depth           run_parity.py                    results/parity.json
    --order-capacity  run_parity.py --preset capacity  results/parity_capacity.json
                      and run_coverage.py              results/coverage.json
                      (comma-separated; merged, since the original run kept
                      both in one file)
    --long            run_parity.py --preset long      results/parity_long.json
    --match           run_match.py                     results/match.json
    --tape            run_tape.py                      results/tape.json
    --contact         run_contact.py                   results/contact.json

Output is LaTeX: each block is a table body preceded by a ``% ---- TAB <name>``
marker, using the supplement's ``\\sd{}`` macro.

Two conventions are fixed here and stated in the document:

* epochs-to-threshold is the mean over the seeds that CROSS, annotated with the
  crossing count when it is not 3/3 -- the convention of the main paper's
  teardown table.  The convergence figure charges a non-crossing seed the
  budget instead, so the two differ by construction and are never mixed.
* a cell is reported only where the per-epoch curve was retained.  In the
  original depth grid 339 of 792 runs were recovered from logs and carry the
  final value only, so `n/a` in an epoch table means the curve is gone, not
  that nothing crossed.  A rerun with this package keeps every curve.
"""
import argparse
import os
import sys
import json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, HERE)

#: File names of the original result files, used by --published.
PUBLISHED_NAMES = dict(
    depth="results_depth.json",
    order_capacity="homa_order_capacity/results_order_capacity.json",
    long="results_depth_long.json",
    match="match_q_sweep.json",
    tape="homa_capacity_sweep.json",
    contact="contact_capacity.json",
)


def _args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--published", metavar="DIR",
                   help="read the original result files from DIR")
    p.add_argument("--depth", default=f"{RESULTS}/parity.json")
    p.add_argument("--order-capacity",
                   default=f"{RESULTS}/parity_capacity.json,{RESULTS}/coverage.json")
    p.add_argument("--long", default=f"{RESULTS}/parity_long.json")
    p.add_argument("--match", default=f"{RESULTS}/match.json")
    p.add_argument("--tape", default=f"{RESULTS}/tape.json")
    p.add_argument("--contact", default=f"{RESULTS}/contact.json")
    a = p.parse_args()
    if a.published:
        for k, name in PUBLISHED_NAMES.items():
            setattr(a, k, os.path.join(a.published, name))
    return a


ARGS = _args()


def _load_runs(paths):
    """One ``{"runs": ...}`` blob from one or several comma-separated files."""
    runs = {}
    for path in str(paths).split(","):
        blob = json.load(open(path))
        runs.update(blob["runs"])
    return {"runs": runs}

SYN = ["blockwise2d", "blockwise3d", "homa_add", "homa"]
SYN_NM = {"blockwise2d": "Pairwise-2D", "blockwise3d": "Blockwise-3D",
          "homa_add": "HOMA-add", "homa": "HOMA"}
MQ = ["pairwise2d", "blockwise3d", "homa_add", "homa"]
MQ_NM = {"pairwise2d": "Pairwise-2D", "blockwise3d": "Blockwise-3D",
         "homa_add": "HOMA-add", "homa": "HOMA"}
LAYERS = [1, 6, 12]

OUT = []
FAILED_CHECKS = []


def check(ok, what):
    """A statement the paper makes about these numbers, checked on this data.

    The original generator asserted these, which is right when the input is the
    published data and wrong for a rerun: a reproduction that does not support
    a claim should still produce its tables, and say plainly which claim it
    does not support.  Failures go to stderr and are summarised at the end.
    """
    if not ok:
        FAILED_CHECKS.append(what)
        print(f"CHECK FAILED: {what}", file=sys.stderr)


def emit(tag, lines):
    OUT.append("%% ---- %s %s" % (tag, "-" * max(3, 66 - len(tag))))
    OUT.extend(lines)
    OUT.append("")


def sdf(x):
    return "\\sd{%s}" % (f"{x:.3f}"[1:] if x < 1 else f"{x:.3f}")


def sd(v):
    """Sample standard deviation over seeds, matching the main paper's tables.

    numpy defaults to the population form; the main paper prints the n-1 form,
    and on three seeds the two differ by sqrt(3/2), which is visible in the
    third decimal place of every cell the two documents share.
    """
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1)) if v.size > 1 else 0.0


def acc(mu, s):
    return f"{mu:.3f}" + sdf(s)


# ===========================================================================
# PARITY / MAJORITY -- depth sweep
# ===========================================================================
_d = _load_runs(ARGS.depth)
# 72 of the 720 (family, k, mech, d, layers, seed) records appear twice -- all of
# them PARITY k=3 at d = 32 and 64, where an earlier k=3 series overlaps the
# depth sweep.  Counting a seed twice leaves the mean alone but corrupts the
# seed standard deviation and the seed counts, so keep one record per seed: the
# one with the retained per-epoch curve, then the one carrying a parameter count.
_seen = {}
for r in _d["runs"].values():
    key = (r["family"], r["k"], r["mech"], r["d_model"], r["n_layers"], r["seed"])
    rank = (len(r.get("curve") or []), 1 if r.get("params") else 0)
    if key not in _seen or rank > _seen[key][0]:
        _seen[key] = (rank, r)
PAR = defaultdict(list)
for (fam, k, mech, dm, lay, _), (_, r) in _seen.items():
    PAR[(fam, k, mech, dm, lay)].append(r)
for _k, _v in PAR.items():
    assert len({r["seed"] for r in _v}) == len(_v), _k

TH = 0.90


def ep_stat(runs):
    """(mean epoch over crossing seeds, n_crossed, n_with_curve, n_runs)"""
    cs = [np.asarray(r["curve"], float) for r in runs if len(r.get("curve", [])) > 1]
    hits = []
    for c in cs:
        w = np.flatnonzero(c >= TH)
        if w.size:
            hits.append(int(w[0]) + 1)
    return (float(np.mean(hits)) if hits else None), len(hits), len(cs), len(runs)


def ep_cell(runs):
    if not runs:
        return "--"
    mu, n, ncur, tot = ep_stat(runs)
    if ncur == 0:
        return r"\textit{n/a}"
    if mu is None:
        return r"$>$40"
    if n == ncur == tot:
        return f"{mu:.1f}"
    return r"%.1f\,{\tiny(%d/%d)}" % (mu, n, tot)


def grid_table(ks, ds, cellfn, lead_col=True):
    """`lead_col=False` drops the $d$ column, which is a constant when only one
    width is shown and costs about 25pt of a table that has to fit 11 data
    columns inside the text block."""
    o = 3 if lead_col else 2
    L = [r"\begin{tabular}{@{}" + ("ll " if lead_col else "l ")
         + " G ".join(["ccc"] * len(ks)) + r"@{}}",
         r"\toprule",
         ("& " * (o - 1)) + " & ".join(
             r"\multicolumn{3}{c}{$k{=}%d$}" % k for k in ks) + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (o + 3 * i, o + 2 + 3 * i)
                 for i in range(len(ks))),
         ("$d$ & mechanism & " if lead_col else "mechanism & ") + " & ".join(
             " & ".join(("$L{=}1$" if j == 0 else f"${v}$")
                        for j, v in enumerate(LAYERS)) for _ in ks) + r" \\",
         r"\midrule"]
    for di, d in enumerate(ds):
        if di:
            L.append(r"\addlinespace[2pt]\midrule")
        for mi, m in enumerate(SYN):
            cells = [cellfn(PAR[("parity", k, m, d, lay)])
                     for k in ks for lay in LAYERS]
            lead = ((r"$\mathbf{%d}$ & " % d) if mi == 0 else "& ") if lead_col else ""
            L.append(f"{lead}{SYN_NM[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


def acc_cell(runs):
    if not runs:
        return "--"
    f = [r["final"] for r in runs]
    return acc(float(np.mean(f)), sd(f))


emit("TAB parity_k12", grid_table([1, 2], [32, 64], acc_cell))
# d_model = 128 is not reported: the epochs table is d = 64 only, and with a
# single width the constant $d$ column is dropped (see grid_table).
emit("TAB parity_epochs", grid_table([3, 4, 5], [64], ep_cell, lead_col=False))


def majority_table():
    """MAJORITY-k across order, depth and width, in the layout of the PARITY
    panel.  Cells are seed means; the seed spread is below 0.001 everywhere,
    so it is stated once in the caption rather than printed 60 times.  A
    (width, depth, order, mechanism) that was never run prints "--"."""
    ks, ds = [1, 2, 3, 4, 5], [32, 64]
    L = [r"\begin{tabular}{@{}ll " + " G ".join(["ccc"] * len(ks)) + r"@{}}",
         r"\toprule",
         "& & " + " & ".join(r"\multicolumn{3}{c}{$k{=}%d$}" % k for k in ks) + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (3 + 3 * i, 5 + 3 * i) for i in range(len(ks))),
         "$d$ & mechanism & " + " & ".join(
             " & ".join(("$L{=}1$" if j == 0 else f"${v}$")
                        for j, v in enumerate(LAYERS)) for _ in ks) + r" \\",
         r"\midrule"]
    worst, spread = 1.0, 0.0
    for di, d in enumerate(ds):
        if di:
            L.append(r"\addlinespace[2pt]\midrule")
        for mi, m in enumerate(SYN):
            cells = []
            for k in ks:
                for lay in LAYERS:
                    f = [r["final"] for r in PAR[("majority", k, m, d, lay)]]
                    if not f:
                        cells.append("--"); continue
                    worst = min(worst, min(f)); spread = max(spread, sd(f))
                    cells.append(f"{np.mean(f):.3f}")
            lead = (r"$\mathbf{%d}$ & " % d) if mi == 0 else "& "
            L.append(f"{lead}{SYN_NM[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    # the paper states the control is solved everywhere; the caption states the spread
    assert worst >= 0.999, worst
    assert spread < 0.001, spread
    return L


emit("TAB majority", majority_table())


# --------------------------------------------------------------- capacity --
_oc = _load_runs(ARGS.order_capacity)
CAP = defaultdict(list)
for r in _oc["runs"].values():
    if r.get("sweep") == "capacity":
        CAP[(r["mechanism"], r["d_model"])].append(r)
# Pairwise-2D and Blockwise-2D are one block at L=16 and agree run for run.
for d in sorted({d for (_, d) in CAP}):
    a = sorted(r["final"] for r in CAP[("plain2d", d)])
    b = sorted(r["final"] for r in CAP[("blockwise2d", d)])
    check(a == b, f"Pairwise-2D and Blockwise-2D (one block) should agree run "
                  f"for run at d_model={d}; got {a} vs {b}")


CAP_DS_MAX = 64          # d_model = 128 is not reported on PARITY


def capacity_table():
    arms = ["blockwise2d", "blockwise3d", "homa"]
    ds = sorted({d for (_, d) in CAP if d <= CAP_DS_MAX})
    L = [r"\begin{tabular}{@{}r rcc rcc rcc@{}}", r"\toprule",
         "& " + " & ".join(r"\multicolumn{3}{c}{%s}" % SYN_NM[m] for m in arms) + r" \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
         r"$d_{\mathrm{model}}$ & params & acc & ep@0.90"
         r" & params & acc & ep@0.90 & params & acc & ep@0.90 \\", r"\midrule"]
    for d in ds:
        cells = []
        for m in arms:
            rs = CAP[(m, d)]
            p = int(rs[0]["params"])
            f = [r["final"] for r in rs]
            cells += [f"{p:,}".replace(",", "{,}"),
                      acc(float(np.mean(f)), sd(f)), ep_cell(rs)]
        L.append(f"${d}$ & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


emit("TAB parity3_capacity", capacity_table())


# ------------------------------------------------------ 120-epoch rerun ----
def long_table():
    R = _load_runs(ARGS.long)
    rs = list(R["runs"].values()) if isinstance(R.get("runs"), dict) else R["runs"]
    by = defaultdict(list)
    for r in rs:
        by[r["n_layers"]].append(r)
    L = [r"\begin{tabular}{@{}r cc c c@{}}", r"\toprule",
         r"depth $L$ & 40 epochs & 120 epochs & per-seed at 120"
         r" & gain over final 10 \\", r"\midrule"]
    for lay in sorted(by):
        rr = by[lay]
        c = np.stack([np.asarray(r["curve"], float)[:120] for r in rr])
        base = [r["final"] for r in PAR[("parity", 5, "blockwise2d", 64, lay)]]
        fin = c[:, -1]
        L.append("$%d$ & %.3f & %s%.3f%s & %s & $+%.3f$ \\\\" % (
            lay, float(np.mean(base)),
            r"\textbf{" if lay == 6 else "", float(np.mean(fin)),
            "}" if lay == 6 else "",
            ", ".join(f"{v:.3f}" for v in sorted(fin)),
            float(np.mean(c[:, -1] - c[:, -11]))))
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


# Table S6 (the 120-epoch rerun) was removed from the supplement on Overleaf;
# long_table() is kept so the numbers stay reproducible.


# ===========================================================================
# MATCH2 / MATCH3
# ===========================================================================
_mq = json.load(open(ARGS.match))
MQR = defaultdict(list)
_mqruns = list(_mq["runs"].values()) if isinstance(_mq.get("runs"), dict) else _mq["runs"]
for r in _mqruns:
    MQR[(r["order"], r["N"], r["mech"], r["d_model"])].append(r)
MQ_M = {(r["order"], r["N"]): r["M"] for r in _mqruns}


N_MAX = 8                # the lengths reported in the main paper


def match_len_table(order):
    Ns = sorted({N for (o, N, _, _) in MQR if o == order and N <= N_MAX})
    ds = sorted({d for (o, _, _, d) in MQR if o == order})
    L = [r"\begin{tabular}{@{}ll " + "c" * len(ds) + r"@{}}", r"\toprule",
         r"& & \multicolumn{%d}{c}{test accuracy at $d_{\mathrm{model}}$} \\" % len(ds),
         r"\cmidrule(l){3-%d}" % (2 + len(ds)),
         "$N\\,(M)$ & mechanism & " + " & ".join(f"${d}$" for d in ds) + r" \\",
         r"\midrule"]
    for ni, N in enumerate(Ns):
        if ni:
            L.append(r"\addlinespace[3pt]")
        for mi, m in enumerate(MQ):
            cells = []
            for d in ds:
                rs = MQR[(order, N, m, d)]
                if not rs:
                    cells.append("--"); continue
                f = [r["final"] for r in rs]
                cells.append(acc(float(np.mean(f)), sd(f)))
            lead = ("$%d\\,(%d)$" % (N, MQ_M[(order, N)])) if mi == 0 else ""
            L.append(f"{lead} & {MQ_NM[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


emit("TAB match2", match_len_table(2))
emit("TAB match3", match_len_table(3))




# ===========================================================================
# TAPE capacity study
# ===========================================================================
import tape_data as T                                         # noqa: E402

CAPJ = json.load(open(ARGS.tape))
CONJ = json.load(open(ARGS.contact))["runs"]
ORD, NM, TITLE = T.ORD, T.NM, T.TITLE


def ss_by_test(test):
    """mean and seed s.d. of one secondary-structure test set, read at the
    best-validation-loss epoch, per arm and width"""
    out = {}
    for m in ORD:
        for d in sorted(CAPJ["secondary_structure"].get(m, {}), key=int):
            hs = [h for h in CAPJ["secondary_structure"][m][d].values()
                  if isinstance(h, dict) and not h.get("oom")]
            if not hs:
                continue
            v = []
            for h in hs:
                e = int(np.nanargmax(-np.asarray(h["val_loss"], float)))
                c = np.asarray(h[f"test_{test}"], float)
                v.append(float(c[min(e, len(c) - 1)]))
            out[(m, int(d))] = (float(np.mean(v)), sd(v), len(v))
    return out


def ss_tests_table():
    tests = ["cb513", "casp12", "ts115"]
    LB = {"cb513": "CB513", "casp12": "CASP12", "ts115": "TS115"}
    S = {t: ss_by_test(t) for t in tests}
    ds = sorted({d for (_, d) in S["cb513"]})
    L = [r"\begin{tabular}{@{}ll " + "c" * len(ds) + r"@{}}", r"\toprule",
         r"& & \multicolumn{%d}{c}{Q3 accuracy at $d_{\mathrm{model}}$} \\" % len(ds),
         r"\cmidrule(l){3-%d}" % (2 + len(ds)),
         "test set & mechanism & " + " & ".join(f"${d}$" for d in ds) + r" \\",
         r"\midrule"]
    for ti, t in enumerate(tests):
        if ti:
            L.append(r"\addlinespace[3pt]")
        best = {d: max((S[t][(m, d)][0] for m in ORD if (m, d) in S[t]),
                       default=None) for d in ds}
        for mi, m in enumerate(ORD):
            cells = []
            for d in ds:
                if (m, d) not in S[t]:
                    cells.append("--"); continue
                mu, s, _ = S[t][(m, d)]
                cell = (r"\textbf{%.3f}" % mu + sdf(s)
                        if best[d] is not None and abs(mu - best[d]) < 1e-12
                        else acc(mu, s))
                cells.append(cell)
            lead = LB[t] if mi == 0 else ""
            L.append(f"{lead} & {NM[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L




def contact_fluor_table():
    D, _ = T.load(src=ARGS.tape, csrc=ARGS.contact, ss_test="cb513")
    L = [r"\begin{tabular}{@{}ll cccc@{}}", r"\toprule",
         r"& & \multicolumn{4}{c}{score at $d_{\mathrm{model}}$} \\",
         r"\cmidrule(l){3-6}",
         r"task & mechanism & $32$ & $64$ & $128$ & $256$ \\", r"\midrule"]
    for ti, t in enumerate(["contact", "fluorescence"]):
        if ti:
            L.append(r"\addlinespace[3pt]")
        best = {d: max((D[t][m][d]["acc"] for m in ORD
                        if m in D[t] and d in D[t][m]), default=None)
                for d in (32, 64, 128, 256)}
        for mi, m in enumerate(ORD):
            cells = []
            for d in (32, 64, 128, 256):
                v = D[t].get(m, {}).get(d)
                if v is None:
                    cells.append("--"); continue
                cell = (r"\textbf{%.3f}" % v["acc"] + sdf(v["std"])
                        if best[d] is not None and abs(v["acc"] - best[d]) < 1e-12
                        else acc(v["acc"], v["std"]))
                cells.append(cell)
            lead = ("Contact (P@$L/5$)" if t == "contact"
                    else r"Fluorescence ($\rho$)") if mi == 0 else ""
            L.append(f"{lead} & {NM[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


def tape_tests_table():
    """Secondary structure (three test sets), contact and fluorescence in one
    table over the union of tested widths.  Contact and fluorescence were not
    run at d_model = 512, so those cells are a dash.  Every cell is formatted
    exactly as in the two tables this replaces."""
    tests = ["cb513", "casp12", "ts115"]
    LB = {"cb513": "Sec.\\ struct.\\ CB513", "casp12": "Sec.\\ struct.\\ CASP12",
          "ts115": "Sec.\\ struct.\\ TS115"}
    S = {t: ss_by_test(t) for t in tests}
    D, _ = T.load(src=ARGS.tape, csrc=ARGS.contact, ss_test="cb513")
    ds = sorted({d for (_, d) in S["cb513"]}
                | {d for t in ("contact", "fluorescence") for m in D[t] for d in D[t][m]})
    L = [r"\begin{tabular}{@{}ll " + "c" * len(ds) + r"@{}}", r"\toprule",
         r"& & \multicolumn{%d}{c}{score at $d_{\mathrm{model}}$} \\" % len(ds),
         r"\cmidrule(l){3-%d}" % (2 + len(ds)),
         "task & mechanism & " + " & ".join(f"${d}$" for d in ds) + r" \\",
         r"\midrule"]

    def best_cell(mu, s_, best):
        return (r"\textbf{%.3f}" % mu + sdf(s_)
                if best is not None and abs(mu - best) < 1e-12 else acc(mu, s_))

    for ti, t in enumerate(tests):
        if ti:
            L.append(r"\addlinespace[3pt]")
        best = {d: max((S[t][(m, d)][0] for m in ORD if (m, d) in S[t]),
                       default=None) for d in ds}
        for mi, m in enumerate(ORD):
            cells = ["--" if (m, d) not in S[t] else
                     best_cell(S[t][(m, d)][0], S[t][(m, d)][1], best[d]) for d in ds]
            lead = LB[t] if mi == 0 else ""
            L.append(f"{lead} & {NM[m]} & " + " & ".join(cells) + r" \\")
    for t in ("contact", "fluorescence"):
        L.append(r"\addlinespace[3pt]")
        best = {d: max((D[t][m][d]["acc"] for m in ORD
                        if m in D[t] and d in D[t][m]), default=None) for d in ds}
        for mi, m in enumerate(ORD):
            cells = []
            for d in ds:
                v = D[t].get(m, {}).get(d)
                cells.append("--" if v is None else best_cell(v["acc"], v["std"], best[d]))
            lead = ("Contact (P@$L/5$)" if t == "contact"
                    else r"Fluorescence ($\rho$)") if mi == 0 else ""
            L.append(f"{lead} & {NM[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


emit("TAB tape_tests", tape_tests_table())


# ---------------------------------------------- cost: params, tput, memory --
COST = {}
for t in ("secondary_structure", "fluorescence"):
    for m in ORD:
        for d in sorted(CAPJ[t].get(m, {}), key=int):
            hs = [h for h in CAPJ[t][m][d].values()
                  if isinstance(h, dict) and not h.get("oom")]
            if hs:
                COST[(t, m, int(d))] = dict(
                    params=float(np.mean([h["n_params_trainable"] for h in hs])),
                    tput=float(np.mean([np.mean(h["tokens_per_sec_compute"])
                                        for h in hs])),
                    mem=float(np.mean([max(h["peak_mem_alloc_gb"]) for h in hs])))
_cb = defaultdict(list)
for r in CONJ:
    _cb[(r["attn"], int(r["d_model"]))].append(r)
for (m, d), rs in _cb.items():
    COST[("contact", m, d)] = dict(
        params=float(np.mean([r["params"] for r in rs])),
        tput=float(np.mean([r["residues_per_s"] for r in rs])),
        mem=float(np.mean([r["peak_mem_gb"] for r in rs])))


def kfmt(v):
    return f"{v/1e3:.1f}" if v < 1e5 else f"{v/1e3:.0f}"


def pfmt(v):
    return (f"{v/1e6:.1f}M" if v >= 1e6 else
            f"{v/1e3:.0f}k" if v >= 1e4 else f"{v/1e3:.1f}k")


def cost_table(t):
    ds = sorted({d for (tt, _, d) in COST if tt == t})
    L = [r"\begin{tabular}{@{}r " + " ".join(["rrr"] * len(ORD)) + r"@{}}",
         r"\toprule",
         "& " + " & ".join(r"\multicolumn{3}{c}{%s}" % NM[m] for m in ORD) + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i)
                 for i in range(len(ORD))),
         r"$d_{\mathrm{model}}$ & " + " & ".join(
             [r"par & tput & mem"] * len(ORD)) + r" \\",
         r"\midrule"]
    for d in ds:
        cells = []
        for m in ORD:
            c = COST.get((t, m, d))
            cells += ["--", "--", "--"] if c is None else [
                pfmt(c["params"]), kfmt(c["tput"]), f"{c['mem']:.2f}"]
        L.append(f"${d}$ & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


def cost_all_table():
    """The three per-task cost tables as one, with a heading row per task.
    Rows within a task are exactly those of cost_table(); a task lists only
    the widths it was run at."""
    TNAME = {"secondary_structure": "Secondary structure",
             "contact": "Contact prediction", "fluorescence": "Fluorescence"}
    ncol = 1 + 3 * len(ORD)
    out = []
    for i, t in enumerate(("secondary_structure", "contact", "fluorescence")):
        body = cost_table(t)
        if i == 0:
            head = body[:body.index(r"\midrule") + 1]
            out += head
        else:
            out.append(r"\addlinespace[3pt]")
        out.append(r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (ncol, TNAME[t]))
        out += body[body.index(r"\midrule") + 1:body.index(r"\bottomrule")]
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


emit("TAB cost_all", cost_all_table())


# ------------------------------------------- accuracy-matched configurations
D_T, _ROWS = T.load(src=ARGS.tape, csrc=ARGS.contact, ss_test="cb513")
BASE = [m for m in ORD if m != "homa"]
SCORE = {"secondary_structure": "Q3 (CB513)", "contact": "P@$L/5$ long",
         "fluorescence": r"Spearman $\rho$"}


def _match(t, m, tau, ref):
    return [d for d in sorted(D_T[t][m]) if T._matches(D_T[t][m][d], tau, ref)]


def mult(v):
    return (f"{v:.0f}$\\times$" if v >= 10 else
            f"{v:.1f}$\\times$" if v >= 1 else f"{v:.2f}$\\times$")


def matched_table(t):
    """one block per HOMA width: which baselines reach that score, and at what
    width, parameter count, throughput and peak memory"""
    levels = sorted(D_T[t]["homa"], key=lambda d: D_T[t]["homa"][d]["acc"])
    L = [r"\begin{tabular}{@{}l r r rl rl rl@{}}", r"\toprule",
         r"mechanism & $d_{\mathrm{model}}$ & " + SCORE[t] +
         r" & \multicolumn{2}{c}{params (M)} & \multicolumn{2}{c}{tput (k/s)}"
         r" & \multicolumn{2}{c@{}}{peak mem (GB)} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
         r" & & & & {\tiny vs HOMA} & & {\tiny vs HOMA} & & {\tiny vs HOMA} \\",
         r"\midrule"]
    for i, hd in enumerate(levels):
        h = D_T[t]["homa"][hd]
        tau, ref = h["acc"], h.get("std", 0.0)
        hits = [(m, _match(t, m, tau, ref)[0]) for m in BASE
                if _match(t, m, tau, ref)]
        if i:
            L.append(r"\addlinespace[3pt]")
        L.append(r"\multicolumn{9}{@{}l}{\textit{target: HOMA at "
                 r"$d_{\mathrm{model}}{=}%d$, %.4f}; %s} \\"
                 % (hd, tau, "no baseline matches" if not hits else
                    "%d of %d baselines match" % (len(hits), len(BASE))))
        L.append(r"\textbf{HOMA} & \textbf{%d} & %.4f & \textbf{%.2f} & & %s & & "
                 r"\textbf{%.2f} & \\" % (hd, tau, h["params"] / 1e6,
                                          kfmt(COST[(t, "homa", hd)]["tput"]), h["mem"]))
        for m, bd in hits:
            b = D_T[t][m][bd]
            L.append("%s & %d & %.4f & %.2f & %s & %s & %s & %.2f & %s \\\\"
                     % (NM[m], bd, b["acc"], b["params"] / 1e6,
                        mult(b["params"] / h["params"]), kfmt(COST[(t, m, bd)]["tput"]),
                        mult(COST[(t, "homa", hd)]["tput"] / COST[(t, m, bd)]["tput"]),
                        b["mem"], mult(h["mem"] / b["mem"])))
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


# The accuracy-matched tables (formerly S10-S12) were removed from the
# supplement on Overleaf; matched_table() is kept so they stay reproducible.


# ===========================================================================
# Triadic window coverage
# ===========================================================================
COV = defaultdict(list)
for r in _oc["runs"].values():
    if r.get("sweep") == "coverage":
        COV[(r.get("phase"), r["mechanism"], r["reach"], r["window"],
             r["n_layers"])].append(r)

CLIFF_NM = {"blockwise2d": "Pairwise-2D", "blockwise3d": "Blockwise-3D",
            "homa_add": "HOMA-add", "homa": "HOMA"}
CLIFF_ARMS = ["blockwise2d", "blockwise3d", "homa_add", "homa"]


def cliff_table():
    ws = sorted({w for (p, _, _, w, _) in COV if p == "cliff"})
    Rs = sorted({R for (p, _, R, _, _) in COV if p == "cliff"})
    L = [r"\begin{tabular}{@{}ll " + "c" * len(Rs) + r"@{}}", r"\toprule",
         r"& & \multicolumn{%d}{c}{interaction reach $R$} \\" % len(Rs),
         r"\cmidrule(l){3-%d}" % (2 + len(Rs)),
         "window & mechanism & " + " & ".join(f"${R}$" for R in Rs) + r" \\",
         r"\midrule"]
    for wi, w in enumerate(ws):
        if wi:
            L.append(r"\addlinespace[3pt]")
        Rw = (w - 1) // 2
        for mi, m in enumerate(CLIFF_ARMS):
            cells = []
            for R in Rs:
                rs = COV[("cliff", m, R, w, 1)]
                if not rs:
                    cells.append("--"); continue
                f = [x["final"] for x in rs]
                cells.append(acc(float(np.mean(f)), sd(f)))
            lead = (r"$w{=}%d$ {\tiny($R_w{=}%d$)}" % (w, Rw)) if mi == 0 else ""
            L.append(f"{lead} & {CLIFF_NM[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


emit("TAB coverage_cliff", cliff_table())


def depth_table():
    Rs = sorted({R for (p, _, R, _, _) in COV if p == "depth"})
    Ls = sorted({lay for (p, _, _, _, lay) in COV if p == "depth"})
    arms = ["blockwise3d", "homa_add", "homa"]
    L = [r"\begin{tabular}{@{}ll " + "c" * len(Ls) + r"@{}}", r"\toprule",
         r"& & \multicolumn{%d}{c}{attention layers} \\" % len(Ls),
         r"\cmidrule(l){3-%d}" % (2 + len(Ls)),
         "reach & mechanism & " + " & ".join(f"${v}$" for v in Ls) + r" \\",
         r"\midrule"]
    for ri, R in enumerate(Rs):
        if ri:
            L.append(r"\addlinespace[3pt]")
        for mi, m in enumerate(arms):
            cells = []
            for lay in Ls:
                rs = COV[("depth", m, R, 3, lay)]
                if not rs:
                    cells.append("--"); continue
                f = [x["final"] for x in rs]
                cells.append(acc(float(np.mean(f)), sd(f)))
            lead = (r"$R{=}%d$" % R) if mi == 0 else ""
            L.append(f"{lead} & {CLIFF_NM[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


emit("TAB coverage_depth", depth_table())


# ===========================================================================
# Component teardown and the marginalised-attention control
# ===========================================================================

print("\n".join(OUT))
if FAILED_CHECKS:
    print(f"\n{len(FAILED_CHECKS)} check(s) on the paper's claims FAILED on this "
          f"data; the tables above were still built:", file=sys.stderr)
    for w in FAILED_CHECKS:
        print(f"  - {w}", file=sys.stderr)
else:
    print("all checks on the paper's claims passed", file=sys.stderr)
