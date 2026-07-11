"""Worklist T3.1 -- the ``smoke`` tier: is the package fundamentally working?

These tests answer one question quickly, on a base install with no optional extras:
did something break so badly that nothing else is worth running? Import health, the
public fit entry point, a distribution's density/sampler, and a serialization round-trip.

Budget: the whole file must run in well under the smoke tier's 30 s target (it is
milliseconds in practice). Keep it dependency-light -- no torch, no backends, no network.
Run just this tier with ``pytest -m smoke``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_top_level_import_is_healthy() -> None:
    import mixle

    # A few load-bearing entry points must be reachable from the package root or its
    # documented submodules -- a broken import cycle shows up here first.
    from mixle.inference import optimize  # noqa: F401
    from mixle.stats import GaussianDistribution, GaussianEstimator  # noqa: F401

    assert hasattr(mixle, "__version__")


def test_public_fit_path_runs() -> None:
    """The headline `optimize(data)` path must infer a model and fit it."""
    from mixle.inference import optimize

    records = [(1.9, "paid", True), (0.4, "free", False), (2.1, "paid", True), (0.7, "free", False)]
    model = optimize(records, out=None)
    assert model is not None
    ld = model.log_density(records[0])
    assert isinstance(ld, float) or hasattr(ld, "__float__")


def test_distribution_density_and_sampler() -> None:
    from mixle.stats import GaussianDistribution

    g = GaussianDistribution(1.0, 2.0)
    assert isinstance(float(g.log_density(1.0)), float)
    drawn = g.sampler().sample(3)
    assert len(list(drawn)) == 3


def test_serialization_round_trip() -> None:
    from mixle.stats import GaussianDistribution
    from mixle.utils.serialization import from_serializable, to_serializable

    g = GaussianDistribution(1.5, 2.5)
    g2 = from_serializable(to_serializable(g))
    assert float(g2.log_density(0.7)) == pytest.approx(float(g.log_density(0.7)))
