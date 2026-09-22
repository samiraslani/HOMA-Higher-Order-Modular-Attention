"""Contact prediction on ProteinNet (TAPE): the third protein-sequence task.

It sits beside ``secondary_structure`` and ``fluorescence`` and offers the same
interface -- a task class with ``build_model()`` and ``make_loader()`` -- plus
the pieces that are specific to a pairwise label:

    ContactDataset, collate_contacts    L x L contact maps from ProteinNet LMDBs
    ContactModel, PairwiseContactHead   package encoder + symmetric pairwise head
    precision_at_L_over_k, evaluate     precision at L/k by separation range

The reported metric is precision at L/5 over long-range pairs (|i-j| >= 24).
Needs ``lmdb`` in addition to the core dependencies.

Two things here are easy to get wrong and expensive to get wrong quietly.

**Coordinate units.**  ProteinNet's raw distribution stores coordinates in
picometers, the TAPE-processed LMDBs in Angstroms.  An 8.0 threshold applied to
picometers does not raise -- it yields almost no contacts, and training
proceeds on an all-negative label matrix.  :func:`check_units` measures the
median consecutive-residue distance (C-alpha spacing is 3.8 A) and
:class:`ContactDataset` calls it on construction.

**LMDB handles.**  LMDB refuses to open the same file twice in one process, so
every open goes through one cache keyed by path *and* PID; DataLoader workers
fork and must not inherit the parent's handle.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import AttentionConfig, ModelConfig, TrainingConfig
from models.protein_transformer import ProteinTransformer


# ===========================================================================
# Data
# ===========================================================================
#: TAPE IUPAC vocabulary, inlined so this does not depend on ``tape_proteins``
#: being installable.  Matches ``ModelConfig.vocab_size = 30``.  Note there is
#: no CLS or SEP: token index i is residue i, which is what keeps the contact
#: matrix aligned with the sequence.
IUPAC = (["<pad>", "<mask>", "<cls>", "<sep>", "<unk>"]
         + list("ABCDEFGHIKLMNOPQRSTUVWXYZ"))
TOK = {a: i for i, a in enumerate(IUPAC)}
UNK = TOK["<unk>"]
VOCAB_SIZE = len(IUPAC)

#: Two residues are in contact when their C-alpha atoms are within this many
#: Angstroms, following TAPE.
CONTACT_THRESHOLD_ANGSTROM = 8.0

#: Pairs closer than this along the sequence are excluded from loss and metrics
#: -- they are trivially in contact and would dominate any precision number.
MIN_SEQUENCE_SEPARATION = 6

_ENVS: dict = {}


def open_lmdb(path):
    """Open an LMDB read-only, once per (path, process).

    Args:
        path: Path to the ``.lmdb`` directory.

    Returns:
        An open ``lmdb.Environment``.
    """
    import lmdb

    key = (str(path), os.getpid())
    if key not in _ENVS:
        _ENVS[key] = lmdb.open(str(path), readonly=True, lock=False,
                               readahead=False, meminit=False)
    return _ENVS[key]


def find_proteinnet(extra_candidates=()) -> Path:
    """Locate the directory that directly contains ``proteinnet_train.lmdb``.

    Searched in order: ``$PROTEINNET_DIR``, any paths passed in, then a few
    conventional locations.  Raises with the list it tried rather than falling
    through to a download, because the TAPE S3 mirror now returns HTTP 403 and
    a silent failure here is worse than a loud one.
    """
    candidates = [Path(p) for p in (
        *( [os.environ["PROTEINNET_DIR"]] if os.environ.get("PROTEINNET_DIR") else [] ),
        *extra_candidates,
        "./data/proteinnet",
        "/content/drive/MyDrive/TAPE_data/proteinnet",
        "/content/proteinnet",
        "~/Downloads/proteinnet",
    )]
    for c in candidates:
        c = c.expanduser()
        if (c / "proteinnet_train.lmdb").exists():
            return c
        if c.exists():
            hits = list(c.glob("*/proteinnet_train.lmdb"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError(
        "Could not find proteinnet_train.lmdb. Set PROTEINNET_DIR to the "
        "folder that directly contains it. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates))


def check_units(lmdb_path, n_records: int = 400) -> float:
    """Median consecutive-residue distance, which must be ~3.8 A.

    Returns:
        The measured median, in whatever unit the file uses.

    Raises:
        ValueError: If it is not in the Angstrom range, meaning the 8.0
            threshold would be meaningless against these coordinates.
    """
    steps = []
    with open_lmdb(lmdb_path).begin() as t:
        n = pickle.loads(t.get(b"num_examples"))
        for i in range(min(n_records, n)):
            item = pickle.loads(t.get(str(i).encode()))
            if item["protein_length"] < 40:
                continue
            c = np.asarray(item["tertiary"], np.float32)
            v = np.asarray(item["valid_mask"], bool)
            d = np.linalg.norm(np.diff(c, axis=0), axis=1)[v[:-1] & v[1:]]
            if d.size:
                steps.append(np.median(d))
    med = float(np.median(steps))
    if not 3.0 < med < 5.0:
        raise ValueError(
            f"coordinates are not in Angstroms: median consecutive-residue "
            f"distance is {med:.1f}, expected ~3.8 (C-alpha spacing). "
            f"A {CONTACT_THRESHOLD_ANGSTROM} threshold against these would "
            f"yield almost no contacts rather than an error.")
    return med


class ContactDataset(Dataset):
    """ProteinNet with binary contact maps.

    Long proteins are cropped to ``max_len``.  The crop is random for training,
    so different epochs see different windows, and centred and deterministic
    for validation and test, so evaluation is repeatable.

    Each seed needs its **own** training instance: ``seed`` drives the random
    crop, so sharing one instance across seeds would give every seed identical
    crops and hide that source of variance.

    Args:
        lmdb_path: Path to one ``proteinnet_*.lmdb``.
        max_len: Crop length.  The pairwise head builds an ``L x L x 2d``
            tensor, so this is the dominant memory term.
        min_len: Proteins shorter than this are dropped -- they hold no pair at
            the minimum sequence separation and contribute nothing to score.
        train: Random crop if True, centred crop if False.
        seed: Seeds the crop RNG.
        verify_units: Check the coordinate units on construction.
    """

    def __init__(self, lmdb_path, max_len: int = 256, min_len: int = 32,
                 train: bool = False, seed: int = 0,
                 verify_units: bool = True) -> None:
        self.path = str(lmdb_path)
        self.max_len, self.min_len, self.train = max_len, min_len, train
        self.rng = np.random.default_rng(seed)

        if verify_units:
            check_units(self.path)

        with open_lmdb(self.path).begin() as t:
            self.n = pickle.loads(t.get(b"num_examples"))

        # Building the length index unpickles every record, which is fast on a
        # local disk and slow over a network mount, so it is cached beside the
        # data.
        cache = Path(f"/tmp/pn_len_{Path(self.path).name}.json")
        if cache.exists():
            lens = json.loads(cache.read_text())
        else:
            with open_lmdb(self.path).begin() as t:
                lens = [pickle.loads(t.get(str(i).encode()))["protein_length"]
                        for i in range(self.n)]
            try:
                cache.write_text(json.dumps(lens))
            except OSError:
                pass
        self.index = [i for i, L in enumerate(lens) if L >= min_len]

    @property
    def env(self):
        return open_lmdb(self.path)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        with self.env.begin() as t:
            item = pickle.loads(t.get(str(self.index[i]).encode()))
        seq = item["primary"]
        L = len(seq)
        coords = np.asarray(item["tertiary"], np.float32).reshape(L, -1)[:, :3]
        valid = np.asarray(item["valid_mask"], bool)

        if L > self.max_len:
            lo = (self.rng.integers(0, L - self.max_len + 1) if self.train
                  else (L - self.max_len) // 2)
            sl = slice(lo, lo + self.max_len)
            seq, coords, valid = seq[sl], coords[sl], valid[sl]
            L = self.max_len

        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        cmap = (dist < CONTACT_THRESHOLD_ANGSTROM).astype(np.int64)
        sep = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
        ignored = (~(valid[:, None] & valid[None, :])
                   | (sep < MIN_SEQUENCE_SEPARATION))
        cmap[ignored] = -1                  # ignored by both loss and metrics

        return {"input_ids": torch.tensor([TOK.get(c, UNK) for c in seq],
                                          dtype=torch.long),
                "contacts": torch.from_numpy(cmap),
                "length": L}


def collate_contacts(batch) -> dict:
    """Pad sequences to the batch maximum and contact maps with -1.

    The label is a matrix, so it cannot go through the package's 1-D label
    padding: that pads one axis and would leave the other ragged.  Padding with
    -1 keeps every padded pair ignored.
    """
    B = len(batch)
    Lm = max(b["length"] for b in batch)
    ids = torch.zeros(B, Lm, dtype=torch.long)
    cm = torch.full((B, Lm, Lm), -1, dtype=torch.long)
    lens = torch.tensor([b["length"] for b in batch])
    for i, b in enumerate(batch):
        L = b["length"]
        ids[i, :L] = b["input_ids"]
        cm[i, :L, :L] = b["contacts"]
    return {"input_ids": ids, "contacts": cm, "lengths": lens}


# ===========================================================================
# Model
# ===========================================================================

class PairwiseContactHead(nn.Module):
    """``(B, L, d)`` -> ``(B, L, L, 2)`` contact logits, symmetric in (i, j).

    The pair feature is the concatenation of the elementwise product and the
    difference of the two residue representations.  The output is symmetrised
    explicitly because contact is a symmetric relation and nothing upstream
    enforces that.

    Args:
        d_model: Width of the encoder output.
        dropout: Applied to the residue representations before pairing.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(2 * d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        prod = x[:, :, None, :] * x[:, None, :, :]
        diff = x[:, :, None, :] - x[:, None, :, :]
        z = self.project(torch.cat([prod, diff], dim=-1))
        return 0.5 * (z + z.transpose(1, 2))


class _AlwaysEqual:
    """Compares equal to everything, so an ``if x != flag`` guard never fires."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


def silence_padding_log(model):
    """Stop the encoder logging its block-safe pad target once per step.

    ``ProteinTransformer._pad_to_blocks`` prints whenever its target changes
    and remembers the last one.  Sequences are padded to the *batch* maximum
    rather than to ``max_len``, so that target changes almost every batch --
    thousands of lines that bury the loss.  Replacing the remembered value with
    something that compares equal to everything suppresses the print without
    touching the padding itself.
    """
    model.encoder._last_reported_target = _AlwaysEqual()
    return model


class ContactModel(nn.Module):
    """Encoder with an identity head, and the pairwise head applied outside it.

    ``ProteinTransformer`` pads each batch up to a block-safe length so the
    sliding-window mechanisms tile evenly.  Taking the encoder output with
    ``head=nn.Identity()`` and slicing back to the real length *before* the
    pairwise head means the ``L x L`` tensor is never built at the padded
    length -- and that tensor is what bounds memory here.
    """

    def __init__(self, model_cfg: ModelConfig, attn_cfg: AttentionConfig) -> None:
        super().__init__()
        self.encoder = ProteinTransformer(model_cfg, attn_cfg, head=nn.Identity())
        self.head = PairwiseContactHead(model_cfg.d_model, model_cfg.dropout)

    def forward(self, input_ids: torch.Tensor, real_len: int) -> torch.Tensor:
        h = self.encoder(input_ids)             # (B, L_pad, d)
        return self.head(h[:, :real_len])       # (B, L, L, 2)


def build_contact_model(attn_type: str, *, d_model: int = 128,
                        n_layers: int = 12, heads: int = 8,
                        block_size: int = 32, stride: int = 16,
                        window_size: int = 5, rank_3d: int = 8,
                        combine: str = "fusion", dropout: float = 0.1,
                        **_ignored) -> ContactModel:
    """Build one contact model in the published configuration.

    Defaults are the published ones: 12 layers, 8 heads, block 32 / stride 16,
    triadic window 5, rank 8.  ``**_ignored`` absorbs sweep bookkeeping such as
    ``max_len`` so a single config dict can be passed straight through.
    """
    model_cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=d_model,
                            num_layers=n_layers, num_heads=heads,
                            dim_feedforward=2 * d_model, dropout=dropout,
                            max_seq_length=None)
    attn_cfg = AttentionConfig(type=attn_type, block_size=block_size,
                               stride=stride, window_size=window_size,
                               rank_3d=rank_3d, combine=combine)
    return ContactModel(model_cfg, attn_cfg)


def pair_memory_gb(batch: int, L: int, d_model: int, bytes_per: int = 4,
                   training: bool = True) -> float:
    """Activation cost of the pairwise head alone, in GB.

    ``prod`` and ``diff`` are each ``(B, L, L, d)`` and their concatenation is
    ``(B, L, L, 2d)``.  Backprop keeps them alive, hence the factor of two.
    Useful for sizing a run before it runs out of memory eight hours in.
    """
    elems = batch * L * L * d_model * 4        # prod + diff + concat(2d)
    return elems * bytes_per / 1e9 * (2 if training else 1)


# ===========================================================================
# Metrics
# ===========================================================================

#: TAPE's separation ranges.  The paper reports the long range, |i-j| >= 24.
RANGES = {"short": (6, 12), "medium": (12, 24), "long": (24, 10 ** 6)}


@torch.no_grad()
def precision_at_L_over_k(prob, target, length, ks=(1, 2, 5)) -> dict:
    """Precision of the top ``L/k`` predicted contacts in each range.

    Args:
        prob: ``(L, L)`` contact probabilities.
        target: ``(L, L)`` labels in ``{0, 1, -1}``; -1 is ignored.
        length: The true sequence length ``L``.
        ks: The ``k`` in ``L/k``.  The paper's metric is ``k = 5``, long range.

    Returns:
        ``{"P@L/<k>_<range>": precision}``; NaN where a range has no pairs.

    Each unordered pair is scored once (upper triangle), because the head is
    symmetric and counting both orders would double every prediction.
    """
    idx = torch.arange(length, device=prob.device)
    sep = (idx[:, None] - idx[None, :]).abs()
    upper = idx[:, None] < idx[None, :]
    out = {}
    for name, (lo, hi) in RANGES.items():
        m = upper & (sep >= lo) & (sep < hi) & (target >= 0)
        n = int(m.sum())
        for k in ks:
            key = f"P@L/{k}_{name}"
            if n == 0:
                out[key] = float("nan")
                continue
            take = min(max(1, length // k), n)
            sel = torch.topk(prob[m], take).indices
            out[key] = float(target[m].float()[sel].mean())
    return out


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None) -> dict:
    """Mean per-protein precision over a loader.

    Averaged per protein rather than pooled over pairs, so a long protein does
    not outweigh a short one.
    """
    model.eval()
    acc = {}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        tgt = batch["contacts"].to(device)
        logits = model(ids, ids.shape[1])
        prob = torch.softmax(logits.float(), dim=-1)[..., 1]
        for i, ln in enumerate(batch["lengths"].tolist()):
            r = precision_at_L_over_k(prob[i, :ln, :ln], tgt[i, :ln, :ln], ln)
            for k, v in r.items():
                if not math.isnan(v):
                    acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ===========================================================================
# Task wrapper, the same interface as the other protein tasks
# ===========================================================================

class ContactPredictionTask:
    """Model construction and data loading for contact prediction.

    Mirrors ``SecondaryStructureTask`` and ``FluorescenceTask``::

        task = ContactPredictionTask(model_cfg, attn_cfg, train_cfg)
        model = task.build_model()
        loader = task.make_loader(ContactDataset(path), shuffle=True)

    ``model_cfg.vocab_size`` is overridden to the 30-token IUPAC vocabulary
    this task uses, and the head's dropout follows ``model_cfg.dropout``.
    """

    def __init__(self, model_cfg: ModelConfig, attn_cfg: AttentionConfig,
                 train_cfg: TrainingConfig) -> None:
        self.model_cfg = model_cfg
        self.attn_cfg = attn_cfg
        self.train_cfg = train_cfg

    def build_model(self) -> "ContactModel":
        self.model_cfg.vocab_size = VOCAB_SIZE
        return ContactModel(self.model_cfg, self.attn_cfg)

    def make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.train_cfg.batch_size,
                          shuffle=shuffle, drop_last=shuffle,
                          collate_fn=collate_contacts,
                          num_workers=self.train_cfg.num_workers)
