"""HOMA -- Higher-Order Modular Attention.

The attention mechanisms live in ``homa.models.attention`` and are the only
thing that varies across the experiments.  Everything else -- backbone,
optimiser, data, budget -- is held fixed by construction.

Subpackages are deliberately not imported here.  The protein tasks pull in the
TAPE data stack and lmdb, which the diagnostic tasks do not need.  Import what
you use::

    from homa.models.attention import get_attention                 # always available
    from homa.tasks.diagnostic import run_one, match_run_one        # numpy + torch only
    from homa.tasks.protein import SecondaryStructureTask           # needs scipy, lmdb
"""

__version__ = "1.0.0"
