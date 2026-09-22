import torch

from homa.tasks.diagnostic.parity_majority import (DEFAULT_CFG, centers_for, chance_level,
                                        make_data, make_offsets)


def test_offsets_hold_reach_constant_across_k():
    for k in (1, 2, 3, 4, 5):
        offs = make_offsets(k, 3)
        assert len(offs) == k
        assert max(abs(o) for o in offs) <= 3
    assert make_offsets(3, 3) == (-3, 0, 3)


def test_boundaries_are_ignored():
    X, Y, offs, centers = make_data(50, "parity", 3, 3, 16, seed=0)
    assert centers == centers_for(offs, 16)
    outside = [i for i in range(16) if i not in centers]
    assert (Y[:, outside] == -100).all()


def test_parity_label_is_the_xor():
    X, Y, offs, centers = make_data(200, "parity", 3, 3, 16, seed=1)
    for i in centers:
        want = sum(X[:, i + o] for o in offs) % 2
        assert torch.equal(Y[:, i], want)


def test_published_chance_level_is_reproduced():
    """Test data for order|L1|parity|k4|*|d64|s0: 4035 of 8000 labels are 1.

    The published record stores 0.5043749809..., a float32 mean; the exact
    fraction is 0.504375.  Compared at float32 resolution, since the mean's
    last bit depends on the backend's reduction order.
    """
    _, Y, _, _ = make_data(DEFAULT_CFG["test_n"], "parity", 4, 3, 16, seed=0 + 999)
    lab = Y[Y != -100]
    assert lab.numel() == 8000 and int(lab.sum()) in (4035, 8000 - 4035)
    assert abs(chance_level(Y) - 0.504375) < 1e-6
