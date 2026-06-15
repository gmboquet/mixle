"""Gallery: the structural combinators and wrappers that compose other distributions.

Shows how leaf families are assembled into richer models: Composite (a fixed-length record of
heterogeneous fields), Sequence (i.i.d. bags with a length model), Optional (values that may be
missing), Transform (a fixed bijection of a base density), Weighted (per-observation weights),
Select (route each observation to a sub-model), Ignored (carry-along fields excluded from the
likelihood), PointMass (a constant), and DiracLengthMixture (a length model mixed with a spike).
Self-contained random data only.
"""
import numpy as np

from pysp.stats import *
from pysp.stats.combinator.transform import AffineTransform
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    print('# Composite: a record of (Gaussian, Categorical, Poisson)')
    d = CompositeDistribution((GaussianDistribution(2.0, 1.0),
                               CategoricalDistribution({'a': 0.7, 'b': 0.3}),
                               PoissonDistribution(5.0)))
    fit = estimate(d.sampler(seed=1).sample(4000),
                   CompositeEstimator((GaussianEstimator(), CategoricalEstimator(), PoissonEstimator())))
    print('  fit: %s' % fit)

    print('# Sequence: i.i.d. Categorical draws with a Poisson length model')
    d = SequenceDistribution(CategoricalDistribution({'a': 0.6, 'b': 0.4}), len_dist=PoissonDistribution(4.0))
    fit = estimate(d.sampler(seed=1).sample(3000),
                   SequenceEstimator(CategoricalEstimator(), len_estimator=PoissonEstimator()))
    print('  fit base=%s len=%s' % (fit.dist, fit.len_dist))

    print('# Optional: a Gaussian that is missing with probability p')
    d = OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.3)
    fit = estimate(d.sampler(seed=1).sample(4000),
                   OptionalEstimator(GaussianEstimator(), est_prob=True))
    print('  fit: %s' % fit)

    print('# Transform: y = 2*x + 5 with x ~ N(0,1), affine bijection fixed and known')
    d = TransformDistribution(GaussianDistribution(0.0, 1.0), transform=AffineTransform(loc=5.0, scale=2.0))
    fit = estimate(d.sampler(seed=1).sample(4000),
                   TransformEstimator(GaussianEstimator(), transform=AffineTransform(loc=5.0, scale=2.0)))
    print('  recovered base: %s' % fit.dist)

    print('# Weighted: a base Gaussian carrying per-observation weights')
    d = WeightedDistribution(GaussianDistribution(1.0, 2.0))
    data = d.sampler(seed=1).sample(4000)
    w_est = WeightedEstimator(GaussianEstimator())
    fit = estimate(data, w_est, prev_estimate=initialize(data, w_est, np.random.RandomState(1)))
    print('  recovered base: %s' % fit.dist)

    print('# Select: density(x) = dists[choice(x)].density(x); route each value by sign')
    choose = lambda x: 0 if x < 0.0 else 1
    data = (list(GaussianDistribution(-5.0, 1.0).sampler(seed=1).sample(2000))
            + list(GaussianDistribution(5.0, 1.0).sampler(seed=2).sample(2000)))
    fit = estimate(data, SelectEstimator([GaussianEstimator(), GaussianEstimator()], choose))
    print('  recovered: %s' % ', '.join(str(c) for c in fit.dists))

    print('# Ignored: a field carried along but excluded from the likelihood')
    d = IgnoredDistribution(GaussianDistribution(0.0, 1.0))
    fit = estimate(d.sampler(seed=1).sample(100), IgnoredEstimator())
    print('  fit: %s' % fit)

    print('# PointMass: a degenerate distribution on a single value')
    d = PointMassDistribution(42.0)
    fit = estimate(d.sampler(seed=1).sample(100), PointMassEstimator(42.0))
    print('  fit: %s' % fit)

    print('# DiracLengthMixture: a Poisson length model mixed with a spike at 0')
    d = DiracLengthMixtureDistribution(PoissonDistribution(6.0), p=0.7, v=0)
    fit = optimize(d.sampler(seed=1).sample(4000),
                   DiracLengthMixtureEstimator(PoissonEstimator(), v=0),
                   max_its=50, rng=np.random.RandomState(1))
    print('  fit: %s' % fit)
