import torch

from homa.tasks.protein.contact_prediction import (build_contact_model,
                                                   precision_at_L_over_k,
                                                   silence_padding_log)


def test_published_contact_parameter_counts_d32():
    expected = {"plain2d": 234818, "blockwise2d": 234818,
                "blockwise3d": 240962, "homa": 260978}
    for arm, n in expected.items():
        m = build_contact_model(arm, d_model=32)
        assert sum(p.numel() for p in m.parameters() if p.requires_grad) == n


def test_contact_head_is_symmetric():
    m = silence_padding_log(build_contact_model("homa", d_model=32)).eval()
    out = m(torch.randint(5, 30, (1, 40)), 40)
    assert torch.allclose(out, out.transpose(1, 2), atol=1e-6)


def test_precision_counts_only_long_range_upper_triangle():
    L = 60
    prob = torch.zeros(L, L)
    tgt = torch.zeros(L, L, dtype=torch.long)
    prob[0, 40] = prob[40, 0] = 1.0          # one confident long-range pair
    tgt[0, 40] = tgt[40, 0] = 1
    r = precision_at_L_over_k(prob, tgt, L)
    assert abs(r["P@L/5_long"] - 1 / (L // 5)) < 1e-6   # 1 hit in the top L/5
