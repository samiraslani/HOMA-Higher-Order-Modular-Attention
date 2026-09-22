import itertools

import numpy as np
import pytest

from homa.tasks.diagnostic.match import (PUBLISHED_M, calibrate_M, full_window,
                                        match_labels)


@pytest.mark.parametrize("cell", sorted(PUBLISHED_M))
def test_calibration_reproduces_every_published_modulus(cell):
    q, N = cell
    assert calibrate_M(N, q) == PUBLISHED_M[cell]


@pytest.mark.parametrize("q", [2, 3])
def test_labels_match_brute_force(q):
    rng = np.random.default_rng(0)
    M, N = 11, 5
    X = rng.integers(0, M, size=(40, N))
    Y = match_labels(X, M, q)
    for s in range(len(X)):
        for i in range(N):
            want = any((X[s, i] + sum(X[s, list(js)])) % M == 0
                       for js in itertools.product(range(N), repeat=q - 1))
            assert Y[s, i] == int(want)


def test_full_window_covers_every_position():
    assert full_window(6) == 11
