"""Drop-in replacements for the two pieces of ``tape_proteins`` this package uses.

``tape_proteins`` (last released 2020) no longer imports under NumPy 2: its
compiled dependencies fail with "numpy.dtype size changed, may indicate binary
incompatibility".  The package only ever used two things from it -- the IUPAC
tokenizer and the LMDB dataset reader -- and both are small, so they are
reproduced here line for line against the ``tape`` source.  The behaviour is
identical, including the edge cases:

* ``encode`` wraps the sequence as ``<cls> ... <sep>``, which is what the
  Fluorescence dataset feeds the model.  ``convert_tokens_to_ids`` does not add
  them, which is what the Secondary Structure dataset uses so that token i stays
  aligned with residue label i.
* An unrecognised residue raises ``KeyError``.  It is not mapped to ``<unk>``.
* A record without an ``id`` field is given ``str(index)``.

``tests/test_tape_compat.py`` checks the vocabulary against the published one.
If ``tape_proteins`` does import in your environment, the two are
interchangeable.
"""

from __future__ import annotations

import pickle as pkl
from collections import OrderedDict
from pathlib import Path
from typing import List, Union

import numpy as np
from torch.utils.data import Dataset

#: Identical to ``tape.tokenizers.IUPAC_VOCAB``.
IUPAC_VOCAB = OrderedDict(
    [("<pad>", 0), ("<mask>", 1), ("<cls>", 2), ("<sep>", 3), ("<unk>", 4)]
    + [(a, 5 + i) for i, a in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZ")])


class TAPETokenizer:
    """The IUPAC tokenizer of ``tape.tokenizers.TAPETokenizer``."""

    def __init__(self, vocab: str = "iupac") -> None:
        if vocab != "iupac":
            raise ValueError("only the 'iupac' vocabulary is provided")
        self.vocab = IUPAC_VOCAB
        self.tokens = list(self.vocab.keys())
        self._vocab_type = vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<cls>"

    @property
    def stop_token(self) -> str:
        return "<sep>"

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(t) for t in tokens]

    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        return [self.start_token] + tokens + [self.stop_token]

    def encode(self, text: str) -> np.ndarray:
        tokens = self.add_special_tokens(self.tokenize(text))
        return np.array(self.convert_tokens_to_ids(tokens), np.int64)


class LMDBDataset(Dataset):
    """The LMDB reader of ``tape.datasets.LMDBDataset``."""

    def __init__(self, data_file: Union[str, Path], in_memory: bool = False):
        import lmdb

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b"num_examples"))
        if in_memory:
            self._cache = [None] * num_examples
        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        if self._in_memory and self._cache[index] is not None:
            return self._cache[index]
        with self._env.begin(write=False) as txn:
            item = pkl.loads(txn.get(str(index).encode()))
            if "id" not in item:
                item["id"] = str(index)
            if self._in_memory:
                self._cache[index] = item
        return item
