"""Diagnostic tasks: interaction order under controlled conditions.

``mechanisms``        every attention arm, by its paper name
``parity_majority``   PARITY-k and MAJORITY-k: interaction order at fixed access
``match``             MATCH2 and MATCH3 on Z_M (Sanford et al., 2024)

Names common to both task modules (``run_one``, ``make_data``, ``build_model``,
``TinyModel``) are exported unprefixed from ``parity_majority`` and with a
``match_`` prefix from ``match``.
"""

from .mechanisms import MECHANISMS, PAPER_NAME, build_attention, is_known

from .parity_majority import (DEFAULT_CFG, TinyModel, build_model, centers_for,
                              chance_level, epochs_to, make_data, make_offsets,
                              n_params, run_one, set_seed)

from .match import (PUBLISHED_M, TinyModel as MatchTinyModel, auto_batch,
                    base_rate, build_model as match_build_model, calibrate_M,
                    full_window, fourier_table, make_data as match_make_data,
                    match_labels, modulus_for, run_one as match_run_one)

__all__ = [
    "MECHANISMS", "PAPER_NAME", "build_attention", "is_known",
    "DEFAULT_CFG", "TinyModel", "build_model", "centers_for", "chance_level",
    "epochs_to", "make_data", "make_offsets", "n_params", "run_one", "set_seed",
    "PUBLISHED_M", "MatchTinyModel", "auto_batch", "base_rate",
    "match_build_model", "calibrate_M", "full_window", "fourier_table",
    "match_make_data", "match_labels", "modulus_for", "match_run_one",
]
