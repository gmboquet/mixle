"""Gallery: directional families on the circle and the sphere, each built / sampled / re-estimated.

Circular angles (VonMises, WrappedNormal, WrappedCauchy, ProjectedNormal) and axial data on the
sphere (Watson). Each true distribution exposes a matching estimator via ``.estimator()``, so the
fit is a one-liner. (VonMisesFisher lives in the multivariate gallery.) Random data only.
"""

from pysp.stats import (
    ProjectedNormalDistribution,
    VonMisesDistribution,
    WatsonDistribution,
    WrappedCauchyDistribution,
    WrappedNormalDistribution,
)
from pysp.inference import estimate

CASES = [
    ('VonMises (circle)',       VonMisesDistribution(0.7, 4.0)),
    ('WrappedNormal (circle)',  WrappedNormalDistribution(0.7, 0.8)),
    ('WrappedCauchy (circle)',  WrappedCauchyDistribution(0.7, 0.6)),
    ('ProjectedNormal (circle)', ProjectedNormalDistribution(1.5, -0.5)),
    ('Watson (axes on sphere)', WatsonDistribution([0.0, 0.0, 1.0], 4.0)),
]

if __name__ == '__main__':
    for label, true_dist in CASES:
        fit = estimate(list(true_dist.sampler(seed=1).sample(8000)), true_dist.estimator())
        print('%-24s' % label)
        print('  true: %s' % true_dist)
        print('  fit : %s' % fit)
