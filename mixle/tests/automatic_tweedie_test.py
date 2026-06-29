"""Acceptance tests for the tweedie automatic-selection detector.

RECOVERY: in-family Tweedie (compound Poisson-Gamma) data is recommended as a TweedieDistribution.
NO-STEAL: Gaussian(0,1) and Exponential(scale=1) data are NOT recommended as Tweedie -- neither has the
Tweedie signature (an exact-zero atom plus a positive continuous part), so the detector must not fire.
"""

import numpy as np

from mixle.inference.estimation import fit
from mixle.stats import TweedieDistribution
from mixle.utils.automatic import get_estimator


def _recommend(data):
    return fit(data, get_estimator(data), max_its=25, out=None)


def test_recovers_tweedie():
    for mu, phi in [(7.0, 1.3), (8.0, 1.4)]:
        # Sample exactly from the TRUE Tweedie via its compound Poisson-Gamma sampler.
        data = [float(x) for x in TweedieDistribution(mu, phi, 1.5).sampler(seed=0).sample(size=5000)]
        # Sanity: the in-family draw genuinely carries a zero atom plus a positive part.
        arr = np.asarray(data)
        assert np.any(arr == 0.0) and np.any(arr > 0.0)
        model = _recommend(data)
        assert isinstance(model, TweedieDistribution), (
            "expected TweedieDistribution for Tweedie(mu=%s, phi=%s), got %s" % (mu, phi, type(model).__name__)
        )


def test_does_not_steal_gaussian():
    rng = np.random.RandomState(1)
    data = [float(x) for x in rng.normal(0.0, 1.0, size=5000)]
    model = _recommend(data)
    assert not isinstance(model, TweedieDistribution), "tweedie stole Gaussian(0,1) data"


def test_does_not_steal_exponential():
    rng = np.random.RandomState(2)
    data = [float(x) for x in rng.exponential(1.0, size=5000)]
    model = _recommend(data)
    assert not isinstance(model, TweedieDistribution), "tweedie stole Exponential(scale=1) data"
