from homa.data.tape_compat import IUPAC_VOCAB, TAPETokenizer
from homa.tasks.protein.contact_prediction import IUPAC


def test_vocab_is_the_published_iupac():
    assert len(IUPAC_VOCAB) == 30
    assert IUPAC_VOCAB["<pad>"] == 0 and IUPAC_VOCAB["<cls>"] == 2
    assert IUPAC_VOCAB["A"] == 5 and IUPAC_VOCAB["Z"] == 29
    assert list(IUPAC_VOCAB) == IUPAC


def test_encode_adds_cls_and_sep_and_ids_do_not():
    t = TAPETokenizer()
    assert list(t.encode("MKV")) == [2, 16, 14, 25, 3]
    assert t.convert_tokens_to_ids(list("MKV")) == [16, 14, 25]


def test_unknown_residue_raises_as_tape_does():
    import pytest
    with pytest.raises(KeyError):
        TAPETokenizer().convert_tokens_to_ids(["J"])
