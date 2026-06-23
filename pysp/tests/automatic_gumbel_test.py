"""Acceptance tests for the gumbel automatic-selection detector: recovery + no-steal."""

import numpy as np
import pytest

from pysp.inference.estimation import fit
from pysp.stats import GumbelDistribution
from pysp.utils.automatic import get_estimator


def _recommended(data):
    return fit(data, get_estimator(data), max_its=25, out=None)


@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (3.0, 2.5)])
def test_recovery_gumbel(loc, scale):
    rng = np.random.RandomState(17)
    data = rng.gumbel(loc=loc, scale=scale, size=5000).tolist()
    model = _recommended(data)
    assert isinstance(model, GumbelDistribution), type(model)


def test_no_steal_gaussian():
    rng = np.random.RandomState(23)
    data = rng.normal(0.0, 1.0, size=5000).tolist()
    model = _recommended(data)
    assert not isinstance(model, GumbelDistribution), type(model)


def test_no_steal_exponential():
    rng = np.random.RandomState(29)
    data = rng.exponential(scale=1.0, size=5000).tolist()
    model = _recommended(data)
    assert not isinstance(model, GumbelDistribution), type(model)
