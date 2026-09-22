"""ProteinNet contact maps: dataset, tokenizer and collation.

Contact prediction is the one TAPE task in the paper that never had library
code -- it lived entirely in a sweep notebook, which is why it is written out
here.  The behaviour is kept identical to the notebook that produced the
published numbers; what changed is that it is now importable and testable.

Two things in here are easy to get wrong and expensive to get wrong quietly.

**Coordinate units.**  ProteinNet's raw distribution stores coordinates in
picometers, and the TAPE-processed LMDBs store Angstroms.  An 8.0 threshold
applied to picometers does not raise -- it yields essentially no contacts, and
training proceeds happily on an all-negative label matrix.  :func:`check_units`
measures the median distance between consecutive residues, which must be the
3.8 A C-alpha spacing, and is called by :class:`ContactDataset` on construction.

**LMDB handles.**  LMDB refuses to open the same file twice in one process, so
every open goes through one cache keyed by path *and* PID -- DataLoader workers
fork and must not inherit the parent's handle.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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
    -1 keeps every synthetic pair ignored.
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
