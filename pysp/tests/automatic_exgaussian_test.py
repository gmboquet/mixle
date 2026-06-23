"""Acceptance tests for the exgaussian automatic-selection detector: recovery + no-steal."""

import numpy as np
import pytest
from scipy import stats

from pysp.inference.estimation import fit
from pysp.stats import ExponentiallyModifiedGaussianDistribution
from pysp.utils.automatic import get_estimator


def _recommended(data):
    return fit(data, get_estimator(data), max_its=25, out=None)


# scipy exponnorm: K = 1/(lam*sigma), loc = mu, scale = sigma. Both settings sit well inside the
# ex-Gaussian's own skew band (skew = 2/(1+(lam*sigma)^2)^1.5): K=1 -> skew ~ 0.707, K=1.5 -> skew ~ 0.99.
@pytest.mark.parametrize("k, mu, sigma", [(1.0, 0.0, 1.0), (1.5, 2.0, 1.5)])
def test_recovery_exgaussian(k, mu, sigma):
    data = stats.exponnorm.rvs(k, loc=mu, scale=sigma, size=5000, random_state=17).tolist()
    model = _recommended(data)
    assert isinstance(model, ExponentiallyModifiedGaussianDistribution), type(model)


def test_no_steal_gaussian():
    rng = np.random.RandomState(23)
    data = rng.normal(0.0, 1.0, size=5000).tolist()
    model = _recommended(data)
    assert not isinstance(model, ExponentiallyModifiedGaussianDistribution), type(model)


def test_no_steal_exponential():
    rng = np.random.RandomState(29)
    data = rng.exponential(scale=1.0, size=5000).tolist()
    model = _recommended(data)
    assert not isinstance(model, ExponentiallyModifiedGaussianDistribution), type(model)
