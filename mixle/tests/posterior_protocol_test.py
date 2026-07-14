"""IC-1 -- the shared `Posterior` protocol (frozen conformance check; work-plan Sec.5).

Named with the repo's own `*_test.py` suffix convention (``pyproject.toml``'s
``[tool.pytest.ini_options] python_files``), not the ``test_*.py`` prefix contracts.md's copy-paste
stub illustrates -- content is otherwise the frozen conformance test verbatim.
"""

import inspect

import numpy as np

from mixle.reason.posterior_protocol import DerivedQuantity, Posterior


class _DQ:
    def __init__(self, s):
        self.samples = np.asarray(s, float)
        self.prior_dominated = False

    def credible_interval(self, level):
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, 0), np.quantile(self.samples, 1 - a, 0)


class _Conforming:
    def samples(self, n, rng):
        return rng.standard_normal((n, 3))

    @property
    def mean(self):
        return np.zeros(3)

    @property
    def cov(self):
        return np.eye(3)

    def credible_interval(self, level):
        return -np.ones(3), np.ones(3)

    def derived_quantity(self, fn, n, rng):
        return _DQ(fn(self.samples(n, rng)))


def test_protocols_are_runtime_checkable():
    assert isinstance(_Conforming(), Posterior)
    assert isinstance(_DQ([1.0, 2.0]), DerivedQuantity)


def test_frozen_signatures():
    assert list(inspect.signature(Posterior.samples).parameters) == ["self", "n", "rng"]
    assert list(inspect.signature(Posterior.credible_interval).parameters) == ["self", "level"]
    assert list(inspect.signature(Posterior.derived_quantity).parameters) == ["self", "fn", "n", "rng"]


def test_derived_quantity_carries_flag_and_interval():
    p = _Conforming()
    dq = p.derived_quantity(lambda m: m.sum(1), 128, np.random.default_rng(0))
    lo, hi = dq.credible_interval(0.9)
    assert dq.prior_dominated is False and np.all(lo <= hi)
