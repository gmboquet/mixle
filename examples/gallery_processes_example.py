"""Gallery: point / counting / partition process families, each built / sampled / re-estimated.

Renewal and Hawkes (self-exciting) processes and an inhomogeneous Poisson process each draw one
realization per ``sample()`` call, so we fit a list of realizations. The Chinese restaurant process
and birth-death sampler draw i.i.d. observations directly. Random data only.
"""

import numpy as np

from mixle.stats import (
    BirthDeathSamplingDistribution,
    ChineseRestaurantProcessDistribution,
    GammaDistribution,
    HawkesProcessDistribution,
    InhomogeneousPoissonProcessDistribution,
    RenewalProcessDistribution,
)
from mixle.inference import estimate

if __name__ == '__main__':
    print('# RenewalProcess (Gamma inter-arrival times on [0, window))')
    d = RenewalProcessDistribution(GammaDistribution(k=3.0, theta=0.5), window=20.0)
    fit = estimate([d.sampler(s).sample() for s in range(400)], d.estimator())
    print('  fit: %s' % fit)

    print('# InhomogeneousPoissonProcess (piecewise-constant rate over 3 bins)')
    d = InhomogeneousPoissonProcessDistribution([2.0, 0.5, 4.0], t_max=3.0)
    fit = estimate([d.sampler(s).sample() for s in range(500)], d.estimator())
    print('  true rates [2.0, 0.5, 4.0] -> fit %s' % np.round(np.asarray(fit.rates), 2))

    print('# HawkesProcess (self-exciting: mu baseline, alpha jump, beta decay)')
    d = HawkesProcessDistribution(mu=0.6, alpha=0.7, beta=1.3, window=50.0)
    fit = estimate([np.sort(d.sampler(seed=s).sample()) for s in range(80)], d.estimator())
    print('  true (mu=0.6, alpha=0.7, beta=1.3) -> fit %s' % fit)

    print('# ChineseRestaurantProcess (cluster-count partitions, concentration alpha)')
    d = ChineseRestaurantProcessDistribution(2.0, 20)
    fit = estimate(list(d.sampler(seed=1).sample(8000)), d.estimator())
    print('  true alpha 2.0 -> fit %s' % fit)

    print('# BirthDeathSampling (birth/death/sampling-rate trajectories)')
    d = BirthDeathSamplingDistribution(0.5, 0.3, 0.2)
    fit = estimate(list(d.sampler(0).sample(400)), d.estimator())
    print('  true (0.5, 0.3, 0.2) -> fit %s' % fit)
