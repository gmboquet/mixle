"""Gallery: ranking / permutation families, each built / sampled / re-estimated.

Mallows (a modal ranking with a dispersion), Plackett-Luce (sequential choice from item utilities),
and Matching (a Gibbs model over bipartite matchings). Each exposes a matching estimator via
``.estimator()``, so a round trip -- build, sample, refit, compare -- is one line per family.

Takeaway: permutations are ordinary mixle observations. The same ``estimate`` call that fits a
Gaussian fits a distribution over orderings, and the recovered parameters can be read straight off
the printed model.
"""

import numpy as np

from mixle.inference import estimate
from mixle.stats import (
    MallowsDistribution,
    MatchingDistribution,
    MatchingEstimator,
    PlackettLuceDistribution,
)

# (label, true distribution, estimator or None to use the family's own ``.estimator()``)
CASES = [
    ("Mallows", MallowsDistribution([2, 0, 1, 3], theta=0.8), None),
    ("PlackettLuce", PlackettLuceDistribution(np.log([0.4, 0.3, 0.2, 0.1])), None),
    # Matching's fit is first-order dual ascent on the edge marginals. The default budget
    # (max_steps=500) stops at ~7e-5 marginal error, short of the 1e-7 default tolerance, and the
    # estimator then raises rather than returning a half-converged model -- so give it a real budget.
    # 2000 steps converge here in well under a second.
    (
        "Matching",
        MatchingDistribution(np.array([[2.0, 0.5, 0.1], [0.2, 2.0, 0.3], [0.1, 0.4, 2.0]])),
        MatchingEstimator(dim=3),
    ),
]

if __name__ == "__main__":
    for label, true_dist, estimator in CASES:
        fit = estimate(list(true_dist.sampler(seed=0).sample(3000)), estimator or true_dist.estimator())
        print("%-13s" % label)
        print("  true: %s" % true_dist)
        print("  fit : %s" % fit)
