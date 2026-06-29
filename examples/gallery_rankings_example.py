"""Gallery: ranking / permutation families, each built / sampled / re-estimated.

Mallows (a modal ranking with a dispersion), Plackett-Luce (sequential choice from item utilities),
and Matching (a Gibbs model over bipartite matchings). Each exposes a matching estimator via
``.estimator()``. The Spearman ranking model has its own example (spearman_example.py). Random data.
"""

import numpy as np

from mixle.stats import MallowsDistribution, MatchingDistribution, PlackettLuceDistribution
from mixle.inference import estimate

CASES = [
    ('Mallows',      MallowsDistribution([2, 0, 1, 3], theta=0.8)),
    ('PlackettLuce', PlackettLuceDistribution(np.log([0.4, 0.3, 0.2, 0.1]))),
    ('Matching',     MatchingDistribution(np.array([[2.0, 0.5, 0.1], [0.2, 2.0, 0.3], [0.1, 0.4, 2.0]]))),
]

if __name__ == '__main__':
    for label, true_dist in CASES:
        fit = estimate(list(true_dist.sampler(seed=0).sample(3000)), true_dist.estimator())
        print('%-13s' % label)
        print('  true: %s' % true_dist)
        print('  fit : %s' % fit)
