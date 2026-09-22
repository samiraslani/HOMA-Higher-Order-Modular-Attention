"""Contact prediction on ProteinNet (TAPE), P@L/5 over long-range pairs.

Needs ``lmdb`` in addition to the core dependencies.
"""

from .data import (CONTACT_THRESHOLD_ANGSTROM, IUPAC, MIN_SEQUENCE_SEPARATION,
                   VOCAB_SIZE, ContactDataset, check_units, collate_contacts,
                   find_proteinnet, open_lmdb)
from .metrics import RANGES, evaluate, precision_at_L_over_k
from .model import (ContactModel, PairwiseContactHead, build_contact_model,
                    pair_memory_gb, silence_padding_log)

__all__ = [
    "CONTACT_THRESHOLD_ANGSTROM", "IUPAC", "MIN_SEQUENCE_SEPARATION",
    "VOCAB_SIZE", "ContactDataset", "check_units", "collate_contacts",
    "find_proteinnet", "open_lmdb",
    "RANGES", "evaluate", "precision_at_L_over_k",
    "ContactModel", "PairwiseContactHead", "build_contact_model",
    "pair_memory_gb", "silence_padding_log",
]
