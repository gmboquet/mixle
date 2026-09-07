"""Gallery: directional families on the circle and the sphere, each built / sampled / re-estimated.

Circular angles (VonMises, WrappedNormal, WrappedCauchy, ProjectedNormal) and axial data on the
sphere (Watson). Each true distribution exposes a matching estimator via ``.estimator()``, so the
fit is a one-liner. (VonMisesFisher lives in the multivariate gallery.) Random data only.
"""

import numpy as np

from mixle.inference import estimate, optimize
from mixle.stats import (
    ProjectedNormalDistribution,
    VonMisesDistribution,
    WatsonDistribution,
    WrappedCauchyDistribution,
    WrappedNormalDistribution,
)

# (label, true distribution, one_pass): the closed-form families are fitted by the one-pass
# ``estimate`` helper; the projected normal has no closed form (its estimator is an EM step whose
# first pass only points the resultant along the data mean), so it runs ``optimize`` to convergence.
CASES = [
    ("VonMises (circle)", VonMisesDistribution(0.7, 4.0), True),
    ("WrappedNormal (circle)", WrappedNormalDistribution(0.7, 0.8), True),
    ("WrappedCauchy (circle)", WrappedCauchyDistribution(0.7, 0.6), True),
    ("ProjectedNormal (circle)", ProjectedNormalDistribution(1.5, -0.5), False),
    ("Watson (axes on sphere)", WatsonDistribution([0.0, 0.0, 1.0], 4.0), True),
]

if __name__ == "__main__":
    for label, true_dist, one_pass in CASES:
        data = list(true_dist.sampler(seed=1).sample(8000))
        if one_pass:
            fit = estimate(data, true_dist.estimator())
        else:
            fit = optimize(data, true_dist.estimator(), max_its=200, rng=np.random.RandomState(1))
        print("%-24s" % label)
        print("  true: %s" % true_dist)
        print("  fit : %s" % fit)
