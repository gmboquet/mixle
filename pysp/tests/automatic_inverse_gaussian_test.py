"""Acceptance tests for the inverse_gaussian automatic-selection detector.

RECOVERY: in-family inverse Gaussian data is recommended as an InverseGaussianDistribution.
NO-STEAL: Gaussian(0,1) and Exponential(scale=1) data are NOT recommended as inverse Gaussian.
"""

import numpy as np

from pysp.inference.estimation import fit
from pysp.stats import InverseGaussianDistribution
from pysp.utils.automatic import get_estimator


def _recommend(data):
    return fit(data, get_estimator(data), max_its=25, out=None)


def test_recovers_inverse_gaussian():
    rng = np.random.RandomState(0)
    for mu, lam in [(1.0, 3.0), (4.0, 8.0)]:
        # numpy's Wald is the inverse Gaussian with (mean=mu, scale=lam).
        data = [float(x) for x in rng.wald(mu, lam, size=5000)]
        model = _recommend(data)
        assert isinstance(model, InverseGaussianDistribution), (
            "expected InverseGaussianDistribution for Wald(%s, %s), got %s" % (mu, lam, type(model).__name__)
        )


def test_does_not_steal_gaussian():
    rng = np.random.RandomState(1)
    data = [float(x) for x in rng.normal(0.0, 1.0, size=5000)]
    model = _recommend(data)
    assert not isinstance(model, InverseGaussianDistribution), (
        "inverse_gaussian stole Gaussian(0,1) data"
    )


def test_does_not_steal_exponential():
    rng = np.random.RandomState(2)
    data = [float(x) for x in rng.exponential(1.0, size=5000)]
    model = _recommend(data)
    assert not isinstance(model, InverseGaussianDistribution), (
        "inverse_gaussian stole Exponential(scale=1) data"
    )
