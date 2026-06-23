"""Acceptance tests for the skew_normal automatic-selection detector: recovery + no-steal."""

import numpy as np
import pytest
from scipy import stats

from pysp.inference.estimation import fit
from pysp.stats import SkewNormalDistribution
from pysp.utils.automatic import get_estimator


def _recommended(data):
    return fit(data, get_estimator(data), max_its=25, out=None)


@pytest.mark.parametrize(
    "shape, loc, scale",
    [(4.0, 0.0, 1.0), (-3.0, 5.0, 2.0)],
)
def test_recovery_skew_normal(shape, loc, scale):
    data = stats.skewnorm.rvs(shape, loc=loc, scale=scale, size=5000, random_state=11).tolist()
    model = _recommended(data)
    assert isinstance(model, SkewNormalDistribution), type(model)


def test_no_steal_gaussian():
    rng = np.random.RandomState(23)
    data = rng.normal(0.0, 1.0, size=5000).tolist()
    model = _recommended(data)
    assert not isinstance(model, SkewNormalDistribution), type(model)


def test_no_steal_exponential():
    rng = np.random.RandomState(29)
    data = rng.exponential(scale=1.0, size=5000).tolist()
    model = _recommended(data)
    assert not isinstance(model, SkewNormalDistribution), type(model)
