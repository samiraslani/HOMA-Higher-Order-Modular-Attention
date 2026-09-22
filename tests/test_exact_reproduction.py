"""Re-run two published cells from scratch and compare with the published number.

These are the only tests that train, about 10 s each on a laptop CPU.

How close to expect.  Under torch 2.7.1 on CPU both cells reproduce the
published record exactly, to every printed digit.  Under torch 2.14 the S10
cell is still exact and the Table 3 cell lands at 0.8465 against 0.8460 --
four of 8,000 test labels -- because kernels change between releases and a
parity model trained to a knife edge amplifies the difference.  So the
tolerance is 0.005 (40 labels): tight enough that a change to data generation,
seeding, initialisation order or the forward pass fails, loose enough not to
fail on a torch upgrade.
"""

TOL = 0.005

import pytest

from homa.synthetic.order_tasks import centers_for, make_offsets, run_one

PARITY = dict(seq_len=16, reach=3, heads=4, rank=8, stride=8, train_n=3000,
              test_n=800, epochs=40, batch_size=128, lr=2e-3)


@pytest.mark.slow
def test_table3_cell_homa_add_linj_parity4_seed0():
    r = run_one("homa_add_linj", family="parity", k=4, d_model=32, seed=0,
                n_layers=1, cfg=PARITY, device="cpu", residual=False)
    assert r["params"] == 5442
    assert abs(r["final"] - 0.8460) <= TOL


@pytest.mark.slow
def test_table_s10_cell_blockwise3d_reach2_window3_seed0():
    cfg = dict(PARITY, seq_len=24)
    centers = centers_for(make_offsets(3, 6), 24)
    r = run_one("blockwise3d", family="parity", k=3, d_model=32, seed=0,
                n_layers=1, cfg=cfg, device="cpu", residual=True, reach=2,
                window=3, centers=centers)
    assert r["covered"] is False
    assert abs(r["final"] - 0.5081) <= TOL
