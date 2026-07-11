"""P4 (experimental) -- MPS/tensor-network leaf: exact conditioning + entanglement receipts.

Receipts: the MPS normalizes to a proper distribution; its marginals and conditionals are exact by
contraction (verified against brute-force enumeration); the entanglement entropy at any cut is
bounded by ``log(bond dimension)`` and hits its known values (0 for a product state, ``log 2`` for
a Bell state); and bond truncation error is exactly the discarded Schmidt weight -- the receipt
that says how much long-range structure a truncation throws away.
"""

from __future__ import annotations

import itertools

import numpy as np

from mixle.experimental.tensor_network import (
    MPS,
    entanglement_entropy,
    product_mps,
    random_mps,
    truncate_error,
)


def _brute_conditional(mps, query, evidence):
    d, length = mps.d, mps.length
    p = mps.all_probabilities()
    out = np.zeros(d)
    for i, seq in enumerate(itertools.product(range(d), repeat=length)):
        if all(seq[k] == v for k, v in evidence.items()):
            out[seq[query]] += p[i]
    return out / out.sum()


def test_normalizes_to_a_proper_distribution() -> None:
    p = random_mps(8, bond=4, seed=0).all_probabilities()
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p >= 0.0)


def test_exact_conditioning_matches_brute_force() -> None:
    mps = random_mps(8, bond=4, seed=1)
    for query, evidence in [(3, {0: 1, 1: 0}), (5, {7: 1}), (0, {4: 0, 6: 1})]:
        exact = mps.conditional(query, evidence)
        brute = _brute_conditional(mps, query, evidence)
        assert np.allclose(exact, brute), f"query={query} ev={evidence}: {exact} vs {brute}"


def test_exact_marginal_matches_brute_force() -> None:
    mps = random_mps(7, bond=3, seed=2)
    p = mps.all_probabilities()
    seqs = list(itertools.product(range(2), repeat=7))
    for evidence in [{0: 1}, {2: 0, 5: 1}]:
        exact = mps.marginal(evidence)
        brute = sum(p[i] for i, s in enumerate(seqs) if all(s[k] == v for k, v in evidence.items()))
        assert np.isclose(exact, brute), f"ev={evidence}: {exact} vs {brute}"


def test_entanglement_bounded_by_log_bond() -> None:
    bond = 4
    mps = random_mps(8, bond=bond, seed=3)
    for cut in range(1, 8):
        s = entanglement_entropy(mps, cut)
        assert s <= np.log(bond) + 1e-9, f"cut={cut}: S={s} exceeds log(bond)={np.log(bond)}"


def test_product_state_has_zero_entanglement() -> None:
    prod = product_mps([[0.6, 0.8], [0.5, 0.5], [0.9, 0.1], [0.3, 0.7]])
    for cut in range(1, 4):
        assert abs(entanglement_entropy(prod, cut)) < 1e-9


def test_bell_state_has_maximal_entanglement() -> None:
    """psi ∝ |00> + |11> has Schmidt spectrum (1/sqrt2, 1/sqrt2) -> S = log 2."""
    a1 = np.zeros((1, 2, 2))
    a1[0, 0, 0] = 1.0
    a1[0, 1, 1] = 1.0
    a2 = np.zeros((2, 2, 1))
    a2[0, 0, 0] = 1.0
    a2[1, 1, 0] = 1.0
    bell = MPS([a1, a2])
    assert np.isclose(entanglement_entropy(bell, 1), np.log(2.0))


def test_truncation_error_tracks_discarded_entanglement() -> None:
    mps = random_mps(8, bond=4, seed=4)
    discarded, tvs = [], []
    for chi in (4, 3, 2, 1):
        disc, tv = truncate_error(mps, 4, chi)
        discarded.append(disc)
        tvs.append(tv)
    assert discarded[0] < 1e-12 and tvs[0] < 1e-9  # no truncation (chi == full rank) -> no error
    # Both grow monotonically as we keep fewer Schmidt values.
    assert all(discarded[i] < discarded[i + 1] for i in range(len(discarded) - 1))
    assert all(tvs[i] < tvs[i + 1] for i in range(len(tvs) - 1))


def test_determinism() -> None:
    a = random_mps(6, bond=3, seed=7).all_probabilities()
    b = random_mps(6, bond=3, seed=7).all_probabilities()
    assert np.array_equal(a, b)
