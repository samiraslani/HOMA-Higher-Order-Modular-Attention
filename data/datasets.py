"""
PyTorch Dataset wrappers for the three published TAPE benchmark tasks.

All datasets expect a TAPE ``LMDBDataset`` object (from the ``tape`` library)
as input.  The TAPE tokenizer (``tape.tokenizers.TAPETokenizer``) handles
amino-acid tokenization.

Tasks covered
-------------
* ``SecondaryStructureDataset`` — per-residue 3-class labels (H / E / C)
* ``FluorescenceDataset``       — per-sequence log-fluorescence regression
* ``StabilityDataset``          — per-sequence stability regression
"""

import torch
from torch.utils.data import Dataset


class SecondaryStructureDataset(Dataset):
    """Wraps a TAPE LMDB dataset for secondary structure prediction (SS3).

    Each sample yields ``input_ids``, ``attention_mask``, and ``labels``
    (per-residue 3-class indices: 0 = helix, 1 = strand, 2 = coil).

    Args:
        lmdb_dataset: A ``tape.datasets.LMDBDataset`` for the SS3 task.
            Expected sample keys: ``"primary"`` (str), ``"valid_mask"``
            (list[int]), ``"ss3"`` (list[int]).
        tokenizer: A ``tape.tokenizers.TAPETokenizer`` instance.
    """

    def __init__(self, lmdb_dataset, tokenizer) -> None:
        self.data = lmdb_dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        tokens = list(sample["primary"])
        input_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long
        )
        attention_mask = torch.tensor(sample["valid_mask"], dtype=torch.long)
        labels = torch.tensor(sample["ss3"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class FluorescenceDataset(Dataset):
    """Wraps a TAPE LMDB dataset for fluorescence prediction.

    Each sample yields an ``input_ids`` tensor and a scalar ``target``
    (log-fluorescence value).

    Args:
        lmdb_dataset: A ``tape.datasets.LMDBDataset`` for the fluorescence
            task.  Expected keys: ``"primary"`` (str),
            ``"log_fluorescence"`` (list[float] of length 1).
        tokenizer: A ``tape.tokenizers.TAPETokenizer`` instance.
    """

    def __init__(self, lmdb_dataset, tokenizer) -> None:
        self.data = lmdb_dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        entry = self.data[idx]
        token_ids = self.tokenizer.encode(entry["primary"])
        target = entry["log_fluorescence"][0]
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.float32),
        }


class StabilityDataset(Dataset):
    """Wraps a TAPE LMDB dataset for protein stability prediction.

    Each sample yields an ``input_ids`` tensor and a scalar ``target``
    (stability score).

    Args:
        lmdb_dataset: A ``tape.datasets.LMDBDataset`` for the stability
            task.  Expected keys: ``"primary"`` (str),
            ``"stability_score"`` (list[float] of length 1).
        tokenizer: A ``tape.tokenizers.TAPETokenizer`` instance.
    """

    def __init__(self, lmdb_dataset, tokenizer) -> None:
        self.data = lmdb_dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        entry = self.data[idx]
        token_ids = self.tokenizer.encode(entry["primary"])
        target = entry["stability_score"][0]
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.float32),
        }
