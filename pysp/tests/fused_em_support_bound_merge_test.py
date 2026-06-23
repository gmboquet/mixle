"""Regression: threaded fused-EM merge must min/max-combine support bounds.

Pareto stores its support minimum in stats[2] (init +inf) and Uniform stores
the min/max in stats[1]/stats[2] (init +/-inf). These are non-additive: when
the n_threads>1 path merges per-chunk buffers, summing them (the old bug)
turns chunk minima 2.0 and 3.0 into 5.0, corrupting ParetoEstimator's xm and
UniformEstimator's [low, high]. The merge must use np.minimum / np.maximum.
"""

import numpy as np

from pysp.stats.compute.fused_kernels import CompiledMixture
from pysp.stats.univariate.continuous.pareto import ParetoDistribution
from pysp.stats.univariate.continuous.uniform import UniformDistribution


def _threaded_vs_serial(model, data):
    cm = CompiledMixture(model)
    enc = cm.encode(data)
    n = enc[0]
    gamma = np.ones((n, 1), dtype=np.float64)
    # force several chunks across multiple threads so the merge path runs
    serial = cm.weighted_suff_stats(enc, gamma, n_threads=1)
    threaded = cm.weighted_suff_stats(enc, gamma, n_threads=4)
    return serial, threaded


def test_pareto_threaded_merge_preserves_support_minimum():
    data = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    model = ParetoDistribution(2.0, 1.5)
    serial, threaded = _threaded_vs_serial(model, data)
    # legacy Pareto ss = (nobs, sum_log_x, min_x)
    assert threaded[2] == 2.0, "threaded merge corrupted the support minimum"
    assert threaded[2] == serial[2]
    assert np.isclose(threaded[0], serial[0])
    assert np.isclose(threaded[1], serial[1])


def test_uniform_threaded_merge_preserves_support_bounds():
    data = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    model = UniformDistribution(1.0, 10.0)
    serial, threaded = _threaded_vs_serial(model, data)
    # legacy Uniform ss = (nobs, min_x, max_x)
    assert threaded[1] == 2.0, "threaded merge corrupted the support minimum"
    assert threaded[2] == 9.0, "threaded merge corrupted the support maximum"
    assert threaded[1] == serial[1]
    assert threaded[2] == serial[2]
    assert np.isclose(threaded[0], serial[0])


def test_pareto_estimate_agrees_across_thread_counts():
    rng = np.random.RandomState(0)
    data = (rng.pareto(2.0, size=4000) + 1.0) * 2.0  # xm=2.0, alpha=2.0
    model = ParetoDistribution(2.0, 2.0)
    serial, threaded = _threaded_vs_serial(model, data.tolist())
    # the recovered support minimum (xm) must be identical, not summed
    assert threaded[2] == serial[2]
    assert threaded[2] == float(np.min(data))
