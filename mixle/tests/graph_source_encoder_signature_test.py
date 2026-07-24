"""Regression test: GraphDataEncoder's __eq__ and __str__ used to disagree about which fields
matter, so its structural signature string could not distinguish two differently-configured
encoders.

``GraphDataEncoder.__eq__`` compared both ``directed`` and ``fallback_assignments``, but
``__str__`` rendered only ``directed``. Two encoders with fallback assignments ``[0, 1]`` and
``[1, 0]`` therefore compared unequal (``__eq__``) yet produced an identical ``str()``.
``save_encoded``/``load_encoded`` (``mixle/data/encoded_io.py``) key their compatibility check
specifically on ``str(encoder)`` (a ``DataSequenceEncoder`` subclass is expected to make
``__str__`` structural, mirroring ``__eq__`` -- see that module's docstrings), so an encoded
payload saved under one fallback-assignment policy could be loaded under a different one without
any error.

The fix routes ``__eq__`` and ``__str__`` through a single versioned canonical signature
(``GraphDataEncoder._signature()``) so the two checks cannot silently diverge again.
"""

import pytest

import mixle.stats  # noqa: F401  -- fully initialize the package to avoid a circular import
from mixle.data import load_encoded, save_encoded
from mixle.data.sources.graph_source import GraphDataEncoder


def test_eq_and_signature_agree_on_fallback_assignments():
    """Two encoders differing only in fallback_assignments must be unequal AND must now also
    differ in their canonical signature and str() -- the two checks must not be able to disagree
    about which fields matter, which is exactly the bug this regresses."""
    enc_ab = GraphDataEncoder(directed=False, fallback_assignments=[0, 1])
    enc_ba = GraphDataEncoder(directed=False, fallback_assignments=[1, 0])

    assert enc_ab != enc_ba
    assert enc_ab._signature() != enc_ba._signature()
    assert str(enc_ab) != str(enc_ba)


def test_eq_and_signature_agree_when_equal():
    """Negative control: encoders with the SAME directed/fallback_assignments compare equal and
    produce an identical structural string, whether or not they are the same instance."""
    enc_1 = GraphDataEncoder(directed=True, fallback_assignments=[0, 1, 1])
    enc_2 = GraphDataEncoder(directed=True, fallback_assignments=[0, 1, 1])

    assert enc_1 is not enc_2
    assert enc_1 == enc_2
    assert enc_1._signature() == enc_2._signature()
    assert str(enc_1) == str(enc_2)


def test_signature_also_reflects_directed():
    # Not the field that was broken, but a canonical spec that silently dropped `directed` instead
    # would be just as bad -- pin both fields down.
    enc_a = GraphDataEncoder(directed=False, fallback_assignments=[0, 1])
    enc_b = GraphDataEncoder(directed=True, fallback_assignments=[0, 1])
    assert enc_a != enc_b
    assert str(enc_a) != str(enc_b)


def test_no_fallback_assignments_still_agree_and_distinguish_from_some():
    enc_none = GraphDataEncoder(directed=False)
    enc_none_2 = GraphDataEncoder(directed=False)
    assert enc_none == enc_none_2
    assert str(enc_none) == str(enc_none_2)

    enc_some = GraphDataEncoder(directed=False, fallback_assignments=[0])
    assert enc_none != enc_some
    assert str(enc_none) != str(enc_some)


def test_save_load_round_trip_detects_fallback_assignment_mismatch(tmp_path):
    """save_encoded/load_encoded key their compatibility check on str(encoder); confirm a
    fallback_assignments-only difference is now caught. It previously was not: __str__ omitted the
    field, so the saved and requested signatures compared equal even though the encoders did not."""
    enc_saved_under = GraphDataEncoder(directed=False, fallback_assignments=[0, 1])
    enc_requested_as = GraphDataEncoder(directed=False, fallback_assignments=[1, 0])
    path = str(tmp_path / "payload.bin")

    save_encoded((1, 2, 3), path, encoder=enc_saved_under)

    with pytest.raises(ValueError, match="encoder mismatch"):
        load_encoded(path, encoder=enc_requested_as)


def test_save_load_round_trip_accepts_matching_fallback_assignments(tmp_path):
    """Negative control: loading under an equal (same fields, different instance) encoder must
    still succeed -- the fix must not make legitimate matching loads fail."""
    enc_saved_under = GraphDataEncoder(directed=False, fallback_assignments=[0, 1])
    enc_requested_as = GraphDataEncoder(directed=False, fallback_assignments=[0, 1])
    path = str(tmp_path / "payload.bin")

    save_encoded((1, 2, 3), path, encoder=enc_saved_under)

    assert load_encoded(path, encoder=enc_requested_as) == (1, 2, 3)
