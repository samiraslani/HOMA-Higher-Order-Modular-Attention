"""Re-run two published cells from scratch and compare with the published number.

These are the only tests that train, about 10 s each on a laptop CPU.  One cell
is from the component teardown (Table 3), one from the window-coverage ablation
(Figure 4).  Their purpose is to catch a silent change to data generation,
seeding, initialisation order or the forward pass, any of which would move the
numbers while every structural test still passed.

How close to expect.  Under torch 2.7.1 on CPU both cells reproduce the
published record exactly.  Kernels change between torch releases, and a parity
model trained to a knife edge can amplify the difference by a few test labels,
so the tolerance is 0.005 (40 of 8,000 labels).
"""

TOL = 0.005

import pytest

from homa.tasks.diagnostic.parity_majority import centers_for, make_offsets, run_one

PARITY = dict(seq_len=16, reach=3, heads=4, rank=8, stride=8, train_n=3000,
              test_n=800, epochs=40, batch_size=128, lr=2e-3)


@pytest.mark.slow
def test_table3_cell_homa_add_uniform_parity4_seed0():
    r = run_one("homa_add_uniform", family="parity", k=4, d_model=32, seed=0,
                n_layers=1, cfg=PARITY, device="cpu", residual=False)
    assert r["params"] == 4930
    assert abs(r["final"] - 0.5002) <= TOL


@pytest.mark.slow
def test_coverage_cell_blockwise3d_reach2_window3_seed0():
    cfg = dict(PARITY, seq_len=24)
    centers = centers_for(make_offsets(3, 6), 24)
    r = run_one("blockwise3d", family="parity", k=3, d_model=32, seed=0,
                n_layers=1, cfg=cfg, device="cpu", residual=True, reach=2,
                window=3, centers=centers)
    assert r["covered"] is False
    assert abs(r["final"] - 0.5081) <= TOL
