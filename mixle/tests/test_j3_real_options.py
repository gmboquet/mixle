"""J3 DoD -- real options & decision-under-uncertainty (notes/exec/workstream-J.md).

The Definition of Done asks for exactly two things about the deferral option:

1. Its value strictly exceeds the naive ``npv_dist.mean`` under high price volatility.
2. It collapses to ``~= max(npv_dist.mean, 0)`` as volatility -> 0.

Both are asserted below on the same underlying (positive-mean) project, plus a handful of supporting
checks on the rest of this task's public API (``OptionValue``'s shape, ``expand``/``abandon`` kinds,
and ``voi_dollars``' non-negativity) that aren't part of the DoD command itself but guard against
regressions in code this task also ships.

Repo-boundary note: J2's ``mixle.analysis.valuation.NPVDistribution`` had not landed on ``release/0.8.0``
as of this PR (see ``mixle/analysis/real_options.py``'s module docstring), so these tests build the
minimal duck-typed stand-in the real work order itself types as a forward reference (``"NPVDistribution"``)
rather than importing a class that doesn't exist yet on this branch.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from mixle.analysis.real_options import OptionValue, real_option_value, voi_dollars


class _FakeNPVDistribution(NamedTuple):
    """Minimal duck-typed stand-in for J2's NPVDistribution -- only `.mean` is consumed."""

    samples: np.ndarray
    mean: float


def _npv_dist(mean: float, spread: float = 5.0, n: int = 2000, seed: int = 0) -> _FakeNPVDistribution:
    rng = np.random.default_rng(seed)
    samples = rng.normal(mean, spread, size=n)
    return _FakeNPVDistribution(samples=samples, mean=float(mean))


def test_defer_option_exceeds_npv_under_high_volatility():
    npv_dist = _npv_dist(mean=10.0)
    opt = real_option_value(npv_dist, volatility=0.6, horizon=5, kind="defer", rate=0.05)
    assert isinstance(opt, OptionValue)
    assert opt.value > npv_dist.mean
    assert opt.premium_over_npv == pytest.approx(opt.value - npv_dist.mean)


def test_defer_option_collapses_to_naive_npv_as_volatility_to_zero():
    npv_dist = _npv_dist(mean=10.0)
    opt = real_option_value(npv_dist, volatility=1e-9, horizon=5, kind="defer", rate=0.05)
    assert opt.value == pytest.approx(max(npv_dist.mean, 0.0), abs=1e-6)


def test_defer_option_monotone_in_volatility():
    npv_dist = _npv_dist(mean=10.0)
    low = real_option_value(npv_dist, volatility=0.05, horizon=5, kind="defer", rate=0.05)
    high = real_option_value(npv_dist, volatility=0.6, horizon=5, kind="defer", rate=0.05)
    assert high.value > low.value


def test_defer_option_on_negative_mean_project_still_floors_at_zero_as_volatility_vanishes():
    npv_dist = _npv_dist(mean=-8.0)
    opt = real_option_value(npv_dist, volatility=1e-9, horizon=5, kind="defer", rate=0.05)
    assert opt.value == pytest.approx(0.0, abs=1e-6)
    # but with real volatility, the option to wait for an upswing is worth strictly more than 0
    opt_vol = real_option_value(npv_dist, volatility=0.6, horizon=5, kind="defer", rate=0.05)
    assert opt_vol.value > 0.0


def test_exercise_boundary_has_one_entry_per_lattice_step():
    npv_dist = _npv_dist(mean=10.0)
    opt = real_option_value(npv_dist, volatility=0.4, horizon=6, kind="defer", rate=0.05)
    assert opt.exercise_boundary.shape == (7,)


@pytest.mark.parametrize("kind", ["defer", "expand", "abandon"])
def test_all_kinds_run_and_return_option_value(kind):
    npv_dist = _npv_dist(mean=10.0)
    opt = real_option_value(npv_dist, volatility=0.3, horizon=4, kind=kind, rate=0.05)
    assert isinstance(opt, OptionValue)
    assert np.isfinite(opt.value)


def test_unknown_kind_raises():
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError):
        real_option_value(npv_dist, volatility=0.3, horizon=4, kind="bogus", rate=0.05)


class _ToyPosterior:
    """A minimal IC-1-conforming posterior: an independent Gaussian belief over one grade parameter."""

    def __init__(self, mean: float, std: float):
        self._mean = mean
        self._std = std

    def samples(self, n, rng):
        return rng.normal(self._mean, self._std, size=(n, 1))

    @property
    def mean(self):
        return np.array([self._mean])

    @property
    def cov(self):
        return np.array([[self._std**2]])

    def credible_interval(self, level):
        z = self._std * 1.6448536269514722  # ~ inverse-CDF fudge, not exercised by this test
        return self.mean - z, self.mean + z

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError


def _decision_value(samples: np.ndarray) -> float:
    """Toy decision rule: a single risk-neutral go/no-go choice, priced at the belief's mean."""
    return float(max(np.mean(samples[:, 0]), 0.0))


def test_voi_dollars_is_non_negative():
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    rng = np.random.default_rng(0)
    voi = voi_dollars(posterior, _decision_value, {"variance_reduction": 0.7}, rng=rng)
    assert voi >= 0.0


def test_voi_dollars_grows_with_variance_reduction():
    rng = np.random.default_rng(0)
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    voi_small = voi_dollars(posterior, _decision_value, {"variance_reduction": 0.1}, rng=np.random.default_rng(1))
    voi_large = voi_dollars(posterior, _decision_value, {"variance_reduction": 0.9}, rng=np.random.default_rng(1))
    assert voi_large >= voi_small
