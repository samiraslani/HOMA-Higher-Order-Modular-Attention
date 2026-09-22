"""HOMA -- Higher-Order Modular Attention.

The attention mechanisms live in ``homa.models.attention`` and are the only
thing that varies across every experiment in the paper.  Everything else --
backbone, optimiser, data, budget -- is held fixed by construction.

Subpackages are deliberately NOT imported here.  ``homa.tasks`` pulls in the
TAPE data stack and ``homa.contact`` pulls in lmdb, neither of which the
synthetic experiments need, and both of which are awkward to install.  Import
what you use::

    from homa.models.attention import get_attention      # always available
    from homa.synthetic import order_tasks               # numpy + torch only
    from homa.tasks import SecondaryStructureTask        # needs tape_proteins
    from homa.contact import ContactDataset              # needs lmdb
"""

__version__ = "1.0.0"
