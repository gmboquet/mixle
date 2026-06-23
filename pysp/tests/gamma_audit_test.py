"""Regression test for the Gamma pseudo_count log-mean prior bug.

GammaDistribution.estimator(pseudo_count) must store the log-mean target as
E[log x] = digamma(k) + log(theta) (a contribution to the accumulated
sum-of-logs), NOT exp(E[log x]). Storing the exponentiated value biases the
shape estimate away from the prior.
"""

import numpy as np
from numpy.random import RandomState

from pysp.stats.univariate.continuous.gamma import GammaDistribution


def test_pseudo_count_logmean_prior_pulls_toward_prior_shape():
    """A pseudo_count prior centered at the truth must not push the shape away.

    With the buggy exp() form, a prior centered exactly at (k=3, theta=2)
    pushed the estimated shape ABOVE the MLE (toward ~3.44) instead of pulling
    it toward the prior mean of 3.0.
    """
    true_k, true_theta = 3.0, 2.0
    rng = RandomState(1)
    data = rng.gamma(shape=true_k, scale=true_theta, size=2000).tolist()

    base = GammaDistribution(true_k, true_theta)
    enc = base.dist_to_encoder()
    x = enc.seq_encode(data)

    # MLE (no prior).
    mle_est = base.estimator()
    acc_mle = mle_est.accumulator_factory().make()
    acc_mle.seq_update(x, np.ones(len(data)), None)
    fit_mle = mle_est.estimate(None, acc_mle.value())

    # Prior centered exactly at the truth, strong-ish weight.
    prior_est = base.estimator(pseudo_count=10.0)
    acc_prior = prior_est.accumulator_factory().make()
    acc_prior.seq_update(x, np.ones(len(data)), None)
    fit_prior = prior_est.estimate(None, acc_prior.value())

    # A prior centered at k=3.0 must pull the MAP shape toward 3.0, i.e. it must
    # lie between the prior mean (3.0) and the MLE -- never further from the
    # prior than the MLE itself.
    assert abs(fit_prior.k - true_k) <= abs(fit_mle.k - true_k) + 1e-9, (
        f"prior-shape {fit_prior.k} is further from prior mean {true_k} than the MLE {fit_mle.k}"
    )


def test_estimator_stores_logmean_not_exp():
    """The second stored suff_stat must equal E[log x], not exp(E[log x])."""
    from pysp.utils.special import digamma

    k, theta = 3.0, 2.0
    est = GammaDistribution(k, theta).estimator(pseudo_count=5.0)
    expected_logmean = digamma(k) + np.log(theta)
    assert np.isclose(est.suff_stat[1], expected_logmean), (
        f"stored log-mean {est.suff_stat[1]} != E[log x] {expected_logmean}"
    )
