"""Worklist Q5.6 -- random-state discipline: reproducible and isolated random behavior.

Q5.6 asks that reproducibility claims state their *exact strength* and pass repeated
tests. This file pins the strength precisely, and no more:

  * **Isolation** -- a stable fit given an explicit ``rng=`` must NOT secretly depend on
    the global NumPy (or ``random``) RNG. Poisoning the global state with different
    seeds before the fit leaves the result unchanged.
  * **Same-seed reproducibility** -- the same explicit seed produces a bit-identical fit
    (identical parameter fingerprint), including through a nested model tree, so seed
    propagation reaches every leaf.
  * **Seed-conditional, not unconditional** -- different seeds may reach different EM
    local optima. Reproducibility is guaranteed *given the seed*, not across seeds. We
    assert exactly that and do NOT overclaim statistical equivalence across seeds, which
    is false for a general mixture (one seed here converges, another collapses).

Process-count invariance ("where promised") is covered separately by the MPI route
equivalence test; seed propagation to distributed workers is that test's concern.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from mixle.inference.estimation import optimize
from mixle.inference.reproduce import param_fingerprint
from mixle.stats import GaussianEstimator, MixtureEstimator


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """These tests deliberately poison the global RNG; restore it so nothing leaks."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


# Two well-separated clusters -- a fit that uses randomness at initialization.
_DATA = [float(x) for x in np.random.RandomState(0).normal(0.0, 1.0, 300)] + [
    float(x) for x in np.random.RandomState(1).normal(6.0, 1.0, 300)
]


def _mixture() -> MixtureEstimator:
    return MixtureEstimator([GaussianEstimator(), GaussianEstimator()])


def _fit_fingerprint(fit_seed: int, *, global_seed: int | None = None, max_its: int = 60) -> str:
    if global_seed is not None:
        np.random.seed(global_seed)  # deliberately poison the global RNGs
        random.seed(global_seed)
    model = optimize(_DATA, _mixture(), out=None, rng=np.random.RandomState(fit_seed), max_its=max_its)
    return param_fingerprint(model)


def test_stable_fit_is_isolated_from_global_rng() -> None:
    """A fit with an explicit rng must be identical no matter the global RNG state."""
    fp_under_a = _fit_fingerprint(fit_seed=7, global_seed=111)
    fp_under_b = _fit_fingerprint(fit_seed=7, global_seed=999)
    assert fp_under_a == fp_under_b, (
        "the fit changed when only the global NumPy/random seed changed -- a stable path "
        "given an explicit rng is leaking a dependence on global RNG state (Q5.6)"
    )


def test_same_seed_is_bit_identical() -> None:
    """The same explicit seed reproduces the exact same fit, repeatedly."""
    fps = {_fit_fingerprint(fit_seed=7) for _ in range(3)}
    assert len(fps) == 1, f"same seed gave non-identical fits: {fps}"


def test_seed_propagates_through_a_model_tree() -> None:
    """A nested mixture-of-mixtures is fully seeded: same seed -> identical, repeatedly."""
    nested = lambda: MixtureEstimator(  # noqa: E731
        [
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
            GaussianEstimator(),
        ]
    )
    fp1 = param_fingerprint(optimize(_DATA, nested(), out=None, rng=np.random.RandomState(3), max_its=40))
    fp2 = param_fingerprint(optimize(_DATA, nested(), out=None, rng=np.random.RandomState(3), max_its=40))
    assert fp1 == fp2, "seed did not propagate deterministically through the nested model tree"


def test_reproducibility_is_seed_conditional_not_unconditional() -> None:
    """Reproducibility is guaranteed given the seed, not across seeds.

    This documents the exact strength of the claim: two different seeds can reach
    different EM optima, so we must NOT promise seed-independent results.
    """
    fp_seed1 = _fit_fingerprint(fit_seed=1)
    fp_seed2 = _fit_fingerprint(fit_seed=2)
    # Deterministic RNG => this inequality is stable, not flaky.
    assert fp_seed1 != fp_seed2, (
        "if all seeds always coincided we could claim seed-independence; here they differ, "
        "so the honest claim is same-seed reproducibility only"
    )
    # ...but each seed is individually reproducible.
    assert _fit_fingerprint(fit_seed=1) == fp_seed1
    assert _fit_fingerprint(fit_seed=2) == fp_seed2
