"""Every arm builds, and each architecture has the published parameter count.

A parameter count is a cheap fingerprint of an architecture: a changed width, a
missing residual path, a different head or an extra projection all move it.
Every expected value below is taken from the result records that produced the
paper, so a failure here means the code no longer builds the published model.
"""

import pytest
import torch

from homa.tasks.diagnostic import MECHANISMS, build_model, match_build_model, n_params


@pytest.mark.parametrize("mech", MECHANISMS + ("homa_w3",))
def test_every_arm_builds_and_runs(mech):
    m = build_model(mech, 32, 4, 16, 7, 8, 8, n_layers=2)
    out = m(torch.randint(0, 2, (3, 16)))
    assert out.shape == (3, 16, 2)
    assert torch.isfinite(out).all()


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError):
        build_model("homa_typo", 32, 4, 16, 7, 8, 8)


# Table 1: depth grid, residual at every depth
@pytest.mark.parametrize("mech,expected", [
    ("blockwise2d", 18178), ("blockwise3d", 19202),
    ("homa_add", 19202), ("homa", 25490)])
def test_table1_parity_d64(mech, expected):
    assert n_params(build_model(mech, 64, 4, 16, 7, 8, 8, n_layers=1,
                                residual=True)) == expected


def test_homa_add_is_parameter_matched_to_blockwise3d():
    a = n_params(build_model("homa_add", 32, 4, 16, 7, 8, 8))
    b = n_params(build_model("blockwise3d", 32, 4, 16, 7, 8, 8))
    assert a == b


# Table 3: one layer, NO residual
@pytest.mark.parametrize("mech,expected", [
    ("homa", 8650), ("homa_uniform", 8138), ("homa_tiedu", 8138),
    ("homa_add", 5442), ("homa_add_uniform", 4930), ("homa_add_tiedu", 4930)])
def test_table3_teardown_d32(mech, expected):
    assert n_params(build_model(mech, 32, 4, 16, 7, 8, 8, n_layers=1,
                                residual=False)) == expected


# Table 2: MATCH3 at N=6
def test_table2_match_homa_d32():
    m = match_build_model("homa", d_model=32, heads=4, N=6, vocab=30)
    assert sum(p.numel() for p in m.parameters()) == 9906
