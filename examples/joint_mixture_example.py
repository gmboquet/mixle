"""Fit a joint mixture: two heterogeneous views of the same object, coupled by a joint weight matrix.

A ``JointMixtureDistribution`` models observations that arrive as a PAIR of views -- here view 1 is a
fixed-length record ``(category, real)`` and view 2 is a variable-length sequence of ``(real, positive)``
pairs. Each view gets its own component set, and a full ``joint_weights`` matrix (not a product of two
marginals) says how the two latent labels co-occur: the mass concentrates on the diagonal, so knowing
view 1's cluster is informative about view 2's. That coupling is the point -- fitting the two views
separately would throw it away.

Takeaway: the fit is scored against the generating model with :func:`empirical_kl_divergence` on a
held-out split, so the printed number is a measured recovery statistic rather than an assertion.
It is the held-out estimate of ``KL(True || Estimate)`` -- truth's expected log-density advantage
over the fit, non-negative in expectation, lower is better. (An earlier revision passed the
arguments the other way round, printing ``-KL(True||Estimate)`` under a ``KL[Estimate||True]``
label with "lower is better" -- a metric whose sign, orientation, AND improvement direction were
all wrong: on a Gaussian oracle it printed -0.317 where the two true KLs are +0.318 and +0.807 --
STAT-RR22-03.)

Note on ``len_normalized``: the sequence views deliberately do NOT set it. A length-normalized sequence
density is a geometric-mean *training objective*, not a generative probability law, and mixture
components must be generative -- ``JointMixtureDistribution`` rejects likelihood factors outright.
"""

import numpy as np

from mixle.inference import best_of
from mixle.stats import *
from mixle.utils.evaluation import empirical_kl_divergence, partition_data

if __name__ == "__main__":
    rng = np.random.RandomState(1)

    # --- the generating model -------------------------------------------------------------------
    # View 1: three (category, real) components; the category identifies the cluster exactly.
    d11 = CompositeDistribution(
        [CategoricalDistribution({"a": 1.0, "b": 0.0, "c": 0.0}), GaussianDistribution(mu=-6.0, sigma2=1.0)]
    )
    d12 = CompositeDistribution(
        [CategoricalDistribution({"a": 0.0, "b": 1.0, "c": 0.0}), GaussianDistribution(mu=0.0, sigma2=1.0)]
    )
    d13 = CompositeDistribution(
        [CategoricalDistribution({"a": 0.0, "b": 0.0, "c": 1.0}), GaussianDistribution(mu=6.0, sigma2=1.0)]
    )

    # View 2: three variable-length bags of (real, positive) pairs, Poisson-distributed length.
    d21 = SequenceDistribution(
        CompositeDistribution([GaussianDistribution(mu=-6.0, sigma2=1.0), GammaDistribution(1.0, 3.0)]),
        PoissonDistribution(3.0),
    )
    d22 = SequenceDistribution(
        CompositeDistribution([GaussianDistribution(mu=0.0, sigma2=1.0), GammaDistribution(3.0, 3.0)]),
        PoissonDistribution(3.0),
    )
    d23 = SequenceDistribution(
        CompositeDistribution([GaussianDistribution(mu=6.0, sigma2=1.0), GammaDistribution(1.0, 3.0)]),
        PoissonDistribution(3.0),
    )

    # Rows index view 1's component, columns view 2's; diagonal-heavy => the views are dependent.
    joint_weights = [[0.48, 0.06, 0.06], [0.03, 0.24, 0.03], [0.01, 0.01, 0.08]]
    dist = JointMixtureDistribution(
        [d11, d12, d13],
        [d21, d22, d23],
        joint_weights=joint_weights,
    )

    sampler = dist.sampler(seed=1)
    data = sampler.sample(10000)

    train_data, valid_data = partition_data(data, [0.9, 0.1], rng)

    # --- the estimator mirrors the model tree, one estimator per view ----------------------------
    est1 = CompositeEstimator([CategoricalEstimator(pseudo_count=1.0), GaussianEstimator()])
    est2 = SequenceEstimator(CompositeEstimator([GaussianEstimator(), GammaEstimator()]), PoissonEstimator())
    est = JointMixtureEstimator([est1] * 3, [est2] * 3, pseudo_count=(0.001, 0.001, 0.001))

    # best_of runs 5 random restarts of EM (<=100 its each) and keeps the one the held-out split likes.
    _, mm = best_of(train_data, valid_data, est, 5, 100, 0.01, 1.0e-8, rng)

    # Score the recovered model against the TRUE one on held-out truth samples. Argument order is
    # the estimand (STAT-RR22-03): with X ~ truth, empirical_kl_divergence(dist1=truth, dist2=fit)
    # estimates E_truth[log truth - log fit] = KL(True || Estimate) >= 0, lower is better. The
    # reversed order estimates the NEGATIVE of that quantity and rewards a worse fit for moving
    # "down".
    enc_vdata = seq_encode(valid_data, model=mm)
    kl, _, _ = empirical_kl_divergence(dist, mm, enc_vdata)

    print("KL(True || Estimate), held-out estimate (nats; lower is better) = %f" % (kl))

    print(str(mm))
