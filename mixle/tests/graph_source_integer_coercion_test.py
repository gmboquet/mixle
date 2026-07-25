"""Regression test: graph coercion used to truncate fractional edge endpoints, block assignments,
and num_nodes instead of rejecting them.

Edge endpoints (``_edge_list_to_adjacency``) and block assignments (``_as_assignments``, and
``GraphDataEncoder``'s own ``fallback_assignments`` constructor argument, which has the identical
call shape) were converted with bare ``int(...)``/``dtype=np.int64`` casts before integrality was
checked. The edge ``(0.9, 1.9)`` silently became the integer edge ``(0, 1)``, and assignments
``[0.9, 1.9]`` silently became ``[0, 1]`` -- changing graph topology and latent block labels without
any error. Casting an existing NumPy float array (as opposed to a fresh Python list) straight to
``int64`` is worse: a non-finite entry (``nan``/``inf``) does not raise at all, it silently becomes
an unspecified, platform-dependent integer.

``num_nodes`` (``_edge_list_to_adjacency``'s own parameter, and the ``_coerce_mapping``
"edges"+"num_nodes" mapping path that feeds it, e.g. ``GraphDataEncoder().seq_encode([{"edges": [],
"num_nodes": 5.5}])``) has the identical bare ``int(...)`` pattern but was out of scope for the fix
described above, which named only "edge endpoints and block assignments." ``int(5.5)`` silently
produced a wrong-sized graph instead of raising; ``int(float("nan"))``/``int(float("inf"))`` already
raised (``ValueError``/``OverflowError`` respectively, straight out of the bare ``int()`` call) but
with generic messages that neither name ``num_nodes`` nor share a common exception type.

The fix validates finite, exact integrality on the ORIGINAL representation before any cast, via
``_require_exact_int`` (scalar: edge endpoints, num_nodes) and ``_require_exact_int_array`` (array:
block assignments, fallback_assignments).
"""

import numpy as np
import pytest

import mixle.stats  # noqa: F401  -- fully initialize the package to avoid a circular import
from mixle.data.sources.graph_source import GraphDataEncoder, _as_assignments, _edge_list_to_adjacency

# --------------------------------------------------------------------------- edge endpoints


def test_fractional_edge_endpoint_raises():
    with pytest.raises(ValueError, match="exact integer"):
        _edge_list_to_adjacency([(0.9, 1.9)], 2, directed=True)


def test_fractional_edge_endpoint_raises_via_public_seq_encode():
    enc = GraphDataEncoder(directed=True)
    with pytest.raises(ValueError, match="exact integer"):
        enc.seq_encode([{"edges": [(0.9, 1.9)], "num_nodes": 2}])


def test_edge_endpoint_nan_raises():
    with pytest.raises(ValueError):
        _edge_list_to_adjacency([(float("nan"), 1)], 2, directed=True)


def test_edge_endpoint_inf_raises():
    with pytest.raises(ValueError):
        _edge_list_to_adjacency([(float("inf"), 1)], 2, directed=True)


def test_edge_endpoint_negative_control_integers_still_work():
    """Legitimate integer edges, as plain ints, exact-valued floats, and numpy integers, must be
    unaffected by the stricter check."""
    adj_ints = _edge_list_to_adjacency([(0, 1)], 2, directed=True)
    assert adj_ints[0, 1] == 1.0

    # 2.0 is an exact integer even though its Python type is float -- must NOT be rejected.
    adj_exact_float = _edge_list_to_adjacency([(0.0, 1.0)], 2, directed=True)
    assert adj_exact_float[0, 1] == 1.0

    adj_numpy_int = _edge_list_to_adjacency([(np.int64(0), np.int64(1))], 2, directed=True)
    assert adj_numpy_int[0, 1] == 1.0


# --------------------------------------------------------------------------- num_nodes
# (_edge_list_to_adjacency's num_nodes parameter, and the "edges"+"num_nodes" mapping path in
# _coerce_mapping that feeds it -- same bare int(...) truncation hazard as edge endpoints above, but
# out of scope for the original fix, which named only "edge endpoints and block assignments.")


def test_fractional_num_nodes_raises():
    with pytest.raises(ValueError, match="exact integer"):
        _edge_list_to_adjacency([], 5.5, directed=True)


def test_fractional_num_nodes_raises_via_public_seq_encode():
    enc = GraphDataEncoder(directed=True)
    with pytest.raises(ValueError, match="exact integer"):
        enc.seq_encode([{"edges": [], "num_nodes": 5.5}])


def test_num_nodes_nan_raises():
    with pytest.raises(ValueError, match="finite"):
        _edge_list_to_adjacency([], float("nan"), directed=True)


def test_num_nodes_inf_raises():
    with pytest.raises(ValueError, match="finite"):
        _edge_list_to_adjacency([], float("inf"), directed=True)


def test_num_nodes_negative_control_integers_still_work():
    """Legitimate integer num_nodes, as a plain int, an exact-valued float, and a numpy integer,
    must be unaffected by the stricter check."""
    adj_ints = _edge_list_to_adjacency([(0, 1)], 2, directed=True)
    assert adj_ints.shape == (2, 2)

    # 2.0 is an exact integer even though its Python type is float -- must NOT be rejected.
    adj_exact_float = _edge_list_to_adjacency([(0, 1)], 2.0, directed=True)
    assert adj_exact_float.shape == (2, 2)

    adj_numpy_int = _edge_list_to_adjacency([(0, 1)], np.int64(2), directed=True)
    assert adj_numpy_int.shape == (2, 2)


# --------------------------------------------------------------------------- block assignments


def test_fractional_block_assignments_raise():
    with pytest.raises(ValueError, match="exact integer"):
        _as_assignments([0.9, 1.9], 2)


def test_fractional_block_assignments_raise_via_public_seq_encode():
    enc = GraphDataEncoder(directed=False)
    with pytest.raises(ValueError, match="exact integer"):
        enc.seq_encode([{"adjacency": [[0, 1], [1, 0]], "block_assignments": [0.9, 1.9]}])


def test_block_assignments_nan_raises_for_list_input():
    with pytest.raises(ValueError, match="finite"):
        _as_assignments([float("nan"), 1.0], 2)


def test_block_assignments_nan_raises_for_existing_ndarray_input():
    # The dangerous path: casting an EXISTING float64 ndarray straight to int64 does not raise on
    # NaN/inf at all (a silent, platform-dependent result), unlike constructing directly from a
    # Python list of floats. Both input shapes are real (callers may hold either), so both must be
    # checked explicitly rather than relying on whatever numpy's cast happens to do.
    with pytest.raises(ValueError, match="finite"):
        _as_assignments(np.array([float("nan"), 1.0]), 2)


def test_block_assignments_inf_raises_for_existing_ndarray_input():
    with pytest.raises(ValueError, match="finite"):
        _as_assignments(np.array([float("inf"), 1.0]), 2)


def test_block_assignments_negative_control_integers_still_work():
    result_ints = _as_assignments([0, 1], 2)
    np.testing.assert_array_equal(result_ints, [0, 1])

    # Exact-valued floats must still be accepted (only the fractional case is new-rejected).
    result_exact_float = _as_assignments([0.0, 1.0], 2)
    np.testing.assert_array_equal(result_exact_float, [0, 1])

    result_numpy_int_array = _as_assignments(np.array([0, 1], dtype=np.int64), 2)
    np.testing.assert_array_equal(result_numpy_int_array, [0, 1])


# --------------------------------------------------------------------------- fallback_assignments
# (GraphDataEncoder's own constructor argument: same "block assignments" concept, same call shape,
# same pre-existing truncation bug, fixed alongside the two call sites named above.)


def test_fractional_fallback_assignments_raises_at_construction():
    with pytest.raises(ValueError, match="exact integer"):
        GraphDataEncoder(directed=False, fallback_assignments=[0.9, 1.9])


def test_fallback_assignments_nan_raises_at_construction():
    with pytest.raises(ValueError, match="finite"):
        GraphDataEncoder(directed=False, fallback_assignments=np.array([float("nan"), 1.0]))


def test_fallback_assignments_negative_control_integers_still_work():
    enc = GraphDataEncoder(directed=False, fallback_assignments=[0.0, 1.0])
    assert enc.fallback_assignments == (0, 1)
