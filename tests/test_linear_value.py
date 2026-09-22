from homa.models.attention.linear_value import verify_equivalence


def test_linear_value_module_equals_published_at_quadratic():
    """The copied HOMA forward must equal the original when nothing is ablated."""
    assert verify_equivalence(verbose=False)
