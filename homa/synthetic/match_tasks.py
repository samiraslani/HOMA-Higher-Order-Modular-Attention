"""MATCH-$q$ on integer sequences: does some $q$-tuple sum to zero mod $M$?

The task family of Sanford et al. (2024).  MATCH2 is realisable by one
self-attention unit; MATCH3 is the smallest case their separation covers, and a
single unit of third-order attention computes it.  Generalised here to any
order, because MATCH-q for q > 3 asks a different question --- whether a
third-order operator *composes* across depth to reach an order it cannot
express in one layer.

Kept in ``synthetic/`` alongside ``order_tasks`` and for the same reason: it
needs only numpy and torch, so it stays runnable in a bare Colab VM.

    from homa.synthetic.match_tasks import run_one, calibrate_M
    M = calibrate_M(N=8, order=4)
    rec = run_one("homa", order=4, N=8, M=M, d_model=64, seed=0,
                  device="cuda")
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from .mechanisms import build_attention, is_known


def match_labels(X, M, order, distinct=False):
    """Labels for MATCH-q, any ``order`` >= 2.

        y_i = 1  iff  exists j_1..j_{q-1}  with  x_i + x_{j_1} + ... = 0 (mod M)

    Indices are drawn with repetition over all N positions (``distinct=False``,
    the plain "exists" reading and the only one consistent with a
    replicate-padded window).  Because they are, the set of achievable
    (q-1)-sums does not depend on the anchor i, so it is computed once per
    sequence and indexed:

        S_1 = {x_t},   S_{j+1}[r] = OR_t S_j[(r - x_t) mod M]
        y_i = S_{q-1}[(-x_i) mod M]

    O(q*N*n*M), which replaces the O(N^(q-1)) tensor an explicit enumeration
    would need.  At q = 2, 3 this returns exactly what direct enumeration does;
    ``tests_match_labels`` checks that.
    """
    if order < 2:
        raise ValueError("order must be >= 2")
    if distinct:
        raise NotImplementedError(
            "distinct=True is not supported: the DP assumes repetition, which "
            "is what makes the reachable-sum set independent of the anchor.")

    n, N = X.shape
    X = np.asarray(X) % M
    rows, cols = np.arange(n)[:, None], np.arange(M)[None, :]

    S = np.zeros((n, M), dtype=bool)
    S[rows, X] = True                                   # S_1
    for _ in range(order - 2):                          # lift to S_{q-1}
        nxt = np.zeros_like(S)
        for t in range(N):
            nxt |= np.take_along_axis(S, (cols - X[:, t:t + 1]) % M, axis=1)
        S = nxt
    return S[rows, (-X) % M].astype(np.int64)


def tuple_count(N, order):
    """Number of (q-1)-tuples a query can draw, with repetition."""
    return N ** (order - 1) / math.factorial(order - 1)


def base_rate(N, M, order, seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, M, size=(n, N))
    return float(match_labels(X, M, order).mean())


#: Modulus used for every published MATCH cell, pinned rather than recomputed.
#:
#: M is what makes the two classes balanced, and it is a property of the task,
#: not of the model -- so it has to be identical across every arm and seed in a
#: column or the accuracies in that column are not comparable.  Pinning it also
#: means a change to the calibration search can never silently move a published
#: number.  :func:`calibrate_M` reproduces every entry below; the runner asserts
#: that it still does.
PUBLISHED_M = {
    (2, 6): 9, (2, 8): 14,
    (3, 6): 30, (3, 8): 56, (3, 12): 116, (3, 16): 206, (3, 24): 465,
    (4, 8): 186, (4, 12): 575,
    (5, 8): 520, (5, 12): 1922,
}


def calibrate_M(N, order, target=0.5, seed=0):
    """Pick the modulus that makes the two classes as close to balanced as possible.

    Without this the task silently becomes trivial: MATCH-q label density grows
    like 1 - exp(-N^(q-1) / ((q-1)! M)), so holding M fixed while sweeping N
    drives the positive rate to 1 and a constant predictor scores whatever that
    rate is.  Balancing per (N, order) keeps the majority-class baseline at
    ~0.50 for every cell, so accuracies are comparable across the sweep.

    Two search brackets, and the reason it is not one.  Every published cell at
    order 2 and 3 was calibrated with the closed-form bracket
    ``[N^2/8, 4N^2]``; the order-4 and order-5 cells were added later and used
    a bracket scaled by the tuple count ``T = N^(q-1)/(q-1)!``, because the
    closed form returns an endpoint for every order above 3.  The two do not
    agree at order 3 -- the tuple-count bracket gives M = 31 at N = 6 where the
    published value is 30 -- so collapsing them into one formula would shift
    five published columns onto a different task.  Each order therefore keeps
    the bracket that generated its published numbers.  :data:`PUBLISHED_M`
    records the answers and ``tests/test_match.py`` pins them.
    """
    if order <= 3:
        lo, hi = ((max(2, int(N / 4)), max(8, 8 * N)) if order == 2
                  else (max(2, int(N * N / 8)), max(8, 4 * N * N)))
    else:
        T = tuple_count(N, order)
        lo, hi = max(2, int(T / 8)), max(16, int(8 * T))

    best, best_err = None, 9.9
    for M in np.unique(np.geomspace(lo, hi, 40).astype(int)):
        err = abs(base_rate(N, int(M), order, seed) - target)
        if err < best_err:
            best, best_err = int(M), err
    return best


def modulus_for(N, order):
    """The published modulus if there is one, otherwise a calibrated one."""
    return PUBLISHED_M.get((order, N)) or calibrate_M(N, order)


def make_data(n, N, M, order, seed, distinct=False):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, M, size=(n, N))
    Y = match_labels(X, M, order, distinct)
    return torch.from_numpy(X).long(), torch.from_numpy(Y).long()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def full_window(N):
    """Smallest odd centred window through which every query sees every position.

    The window is centred on the query and replicate-padded, so a window of
    size w reaches only +/-(w-1)/2.  A query at position 0 therefore needs
    (w-1)/2 >= N-1 to see position N-1: full coverage means w = 2N-1, NOT w = N.
    Getting this wrong makes the triadic branch blind to most of the sequence
    at the edges, and the model fails for a reason that has nothing to do with
    interaction order.
    """
    return 2 * N - 1


def fourier_table(M, n_freq, seed=0):
    """Frozen embedding of Z_M in a Fourier basis: rows are cos/sin features.

    Why this is not cheating, and why the experiment needs it.  Match-k asks
    whether k values sum to zero *modulo M*, so a model trained on a learned
    lookup table has to solve two problems at once: discover a representation
    of modular arithmetic, and then route and combine k of them.  The first is
    the famously slow one, and it is not the one under test -- with a learned
    table both a pairwise and a triadic model plateau around 0.74 and then
    overfit, which says nothing about interaction order.

    Freezing the embedding to a Fourier basis removes the arithmetic and leaves
    only the routing.  It is also the representation the theory assumes: the
    indicator of ``s = 0 (mod M)`` is exactly ``(1/M) sum_w exp(2*pi*i*w*s/M)``,
    so with these features a *trilinear* form can express
    ``cos(w(x_i + x_j + x_k))`` exactly, via

        cos(A+B+C) = cosA cosB cosC - cosA sinB sinC
                     - sinA cosB sinC - sinA sinB cosC,

    which is a sum of four products of three coordinates -- precisely the shape
    of the triadic score ``sum_c Q_ic K_jc U_kc``.  A bilinear form ``Q_i . K_j``
    has no such expansion for a three-way sum.  So this embedding is what makes
    the architectural question sharp rather than what answers it.
    """
    K = max(1, min(n_freq, (M - 1) // 2))
    rng = np.random.default_rng(seed)
    freqs = (np.arange(1, K + 1) if K >= (M - 1) // 2
             else rng.choice(np.arange(1, (M - 1) // 2 + 1), size=K, replace=False))
    ang = 2 * np.pi * np.arange(M)[:, None] * np.sort(freqs)[None, :] / M
    E = np.concatenate([np.cos(ang), np.sin(ang)], axis=1)      # (M, 2K)
    return torch.tensor(E, dtype=torch.float32)


class FourierEmbed(nn.Module):
    """Frozen Fourier features followed by a learnable linear map.

    The projection is learnable so the model can still choose which frequencies
    matter; the modular structure it is projecting is what is held fixed.
    """

    def __init__(self, M, d_model, seed=0):
        super().__init__()
        table = fourier_table(M, d_model // 2, seed)
        self.emb = nn.Embedding.from_pretrained(table, freeze=True)
        self.proj = nn.Linear(table.shape[1], d_model)

    def forward(self, x):
        return self.proj(self.emb(x))


class TinyModel(nn.Module):
    """Embedding -> attention layers -> per-position readout.  No FFN.

    The feed-forward block is omitted on purpose: an MLP after attention can
    build products of its inputs by itself, so leaving it in would let a
    pairwise model fake a third-order interaction and the comparison would stop
    being about attention.  Positional embeddings are off by default because
    Match2/Match3 depend only on the multiset of values, not on where they sit.
    """

    def __init__(self, attns, d_model, vocab, L, use_pos=False, embed="fourier",
                 seed=0):
        super().__init__()
        self.tok = (FourierEmbed(vocab, d_model, seed) if embed == "fourier"
                    else nn.Embedding(vocab, d_model))
        self.pos = nn.Embedding(L, d_model) if use_pos else None
        self.attns = nn.ModuleList(attns)
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in attns)
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, 2)
        self.register_buffer("pos_ids", torch.arange(L).unsqueeze(0))

    def forward(self, x):
        h = self.tok(x)
        if self.pos is not None:
            h = h + self.pos(self.pos_ids)
        for attn, norm in zip(self.attns, self.norms):
            h = h + attn(norm(h), None)
        return self.readout(self.final_norm(h))


def build_model(mech, *, d_model, heads, N, vocab, rank=8, n_layers=1,
                embed="fourier", seed=0):
    """Stack ``n_layers`` copies of one mechanism over the Fourier embedding.

    The window is ``full_window(N) = 2N-1`` for every triadic arm, so the
    triadic branch sees the whole sequence from every query and the comparison
    is about interaction order rather than reach.  ``block_size = stride = N``
    gives a single full-sequence block for the same reason.

    Arms come from :mod:`homa.synthetic.mechanisms`.  That matters here: the
    published HOMA-add column of Table 2 was produced by patching this function
    at run time from the sweep notebook, so ``homa_add`` was reachable in the
    run but not in the committed source.  It is a first-class arm now.
    """
    if not is_known(mech):
        raise ValueError(f"unknown mechanism {mech!r}")
    W = full_window(N)

    def one():
        return build_attention(mech, heads=heads, d_model=d_model,
                               block_size=N, stride=N, window=W, rank=rank)

    return TinyModel([one() for _ in range(n_layers)], d_model, vocab, N,
                     embed=embed, seed=seed)


def auto_batch(N, heads, budget=3.0e7, cap=256):
    """Batch size that keeps the (w, w) score tensor inside a memory budget.

    The triadic branch materialises (B, H, N, w, w) with w = 2N-1, i.e. it
    grows like B*H*N^3.  Holding the batch fixed across an N sweep runs out of
    memory at the top end; holding this product fixed does not.
    """
    w = full_window(N)
    return int(max(8, min(cap, budget / (heads * N * w * w))))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def run_one(mech, *, order, N, M, d_model, heads, seed, device,
            epochs=30, train_n=6000, test_n=1500, lr=2e-3, rank=8,
            batch=None, distinct=False, embed="fourier"):
    set_seed(seed)
    Xtr, Ytr = make_data(train_n, N, M, order, seed, distinct)
    Xte, Yte = make_data(test_n, N, M, order, seed + 991, distinct)
    Xtr, Ytr, Xte, Yte = (t.to(device) for t in (Xtr, Ytr, Xte, Yte))

    model = build_model(mech, d_model=d_model, heads=heads, N=N,
                        vocab=M, rank=rank, embed=embed, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    bs = batch or auto_batch(N, heads)
    ebs = max(8, bs)

    t0, curve = time.time(), []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        for s in range(0, Xtr.shape[0], bs):
            idx = perm[s:s + bs]
            loss = lossf(model(Xtr[idx]).reshape(-1, 2), Ytr[idx].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, Xte.shape[0], ebs):
                p = model(Xte[s:s + ebs]).argmax(-1)
                correct += (p == Yte[s:s + ebs]).sum().item()
        curve.append(correct / Yte.numel())

    maj = float(max(Yte.float().mean().item(), 1 - Yte.float().mean().item()))
    return dict(curve=curve, final=curve[-1], best=max(curve),
                majority=maj, params=sum(p.numel() for p in model.parameters()),
                batch=bs, wall_s=round(time.time() - t0, 1))
