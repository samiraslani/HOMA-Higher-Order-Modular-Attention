"""Protein-sequence tasks from TAPE.

``secondary_structure``   per-residue 3-class labels; Q3 on CB513, CASP12, TS115
``contact_prediction``    pairwise contact maps; P@L/5 over long-range pairs
``fluorescence``          per-sequence regression; Spearman rho
"""

from .contact_prediction import ContactPredictionTask
from .fluorescence import FluorescenceTask
from .secondary_structure import SecondaryStructureTask

__all__ = ["SecondaryStructureTask", "ContactPredictionTask", "FluorescenceTask"]
