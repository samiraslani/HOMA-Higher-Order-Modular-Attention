"""Synthetic interaction-order tasks.

Kept free of the protein stack on purpose: these need only numpy and torch, so
they stay runnable in a bare environment such as a fresh Colab VM, and they are
what the order argument in the paper actually rests on.

``mechanisms``    every attention arm the paper reports, by its paper name.
``order_tasks``   PARITY-k / MAJORITY-k -- interaction order at fixed access.
``match_tasks``   MATCH-q on Z_M, the task family of Sanford et al. (2024).

Names common to both task modules (``run_one``, ``make_data``, ``build_model``,
``TinyModel``) are exported unprefixed from ``order_tasks`` and with a
``match_`` prefix from ``match_tasks``.
"""

from .mechanisms import MECHANISMS, PAPER_NAME, build_attention, is_known

from .order_tasks import (DEFAULT_CFG, TinyModel, build_model, centers_for,
                          chance_level, epochs_to, make_data, make_offsets,
                          n_params, run_one, set_seed)

from .match_tasks import (PUBLISHED_M, TinyModel as MatchTinyModel, auto_batch,
                          base_rate, build_model as match_build_model,
                          calibrate_M, full_window, fourier_table,
                          make_data as match_make_data, match_labels,
                          modulus_for, run_one as match_run_one, tuple_count)

__all__ = [
    # mechanisms
    "MECHANISMS", "PAPER_NAME", "build_attention", "is_known",
    # parity / majority
    "DEFAULT_CFG", "TinyModel", "build_model", "centers_for", "chance_level",
    "epochs_to", "make_data", "make_offsets", "n_params", "run_one",
    "set_seed",
    # match-q
    "PUBLISHED_M", "MatchTinyModel", "auto_batch", "base_rate",
    "match_build_model", "calibrate_M", "full_window", "fourier_table",
    "match_make_data", "match_labels", "modulus_for", "match_run_one",
    "tuple_count",
]
