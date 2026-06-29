"""Gallery: vector / set / count distribution families, each built / sampled / re-estimated.

Covers multivariate and diagonal Gaussians, the von Mises-Fisher directional model, the Dirichlet,
multinomial counts (object-keyed and integer), Bernoulli sets (object-keyed and integer-indexed),
and the integer uniform-spike model. Self-contained random data only.
"""
import numpy as np

from mixle.stats import *
from mixle.inference import estimate

if __name__ == '__main__':
    print('# MultivariateGaussian')
    d = MultivariateGaussianDistribution([1.0, -1.0], [[2.0, 0.8], [0.8, 1.0]])
    fit = estimate(d.sampler(seed=1).sample(5000), MultivariateGaussianEstimator())
    print('  fit mu=%s' % np.round(fit.mu, 2))

    print('# DiagonalGaussian')
    d = DiagonalGaussianDistribution([0.0, 3.0, -2.0], [1.0, 2.0, 0.5])
    fit = estimate(d.sampler(seed=1).sample(5000), DiagonalGaussianEstimator())
    print('  fit mu=%s covar=%s' % (np.round(fit.mu, 2), np.round(fit.covar, 2)))

    print('# VonMisesFisher (directions on the unit circle)')
    d = VonMisesFisherDistribution([0.6, 0.8], 8.0)
    fit = estimate(d.sampler(seed=1).sample(5000), VonMisesFisherEstimator())
    print('  fit mu=%s kappa=%.2f' % (np.round(fit.mu, 2), fit.kappa))

    print('# Dirichlet (points on the simplex)')
    d = DirichletDistribution([2.0, 4.0, 1.0])
    fit = estimate(d.sampler(seed=1).sample(5000), DirichletEstimator(dim=3))
    print('  fit alpha=%s' % np.round(fit.alpha, 2))

    print('# Multinomial (object-keyed counts)')
    d = MultinomialDistribution(CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2}),
                                len_dist=CategoricalDistribution({8: 1.0}))
    fit = estimate(d.sampler(seed=1).sample(3000),
                   MultinomialEstimator(CategoricalEstimator(), len_estimator=CategoricalEstimator()))
    print('  fit pmap=%s' % {k: round(v, 2) for k, v in fit.dist.pmap.items()})

    print('# IntegerMultinomial (integer-indexed counts)')
    d = IntegerMultinomialDistribution(0, [0.5, 0.3, 0.2], len_dist=CategoricalDistribution({8: 1.0}))
    fit = estimate(d.sampler(seed=1).sample(3000),
                   IntegerMultinomialEstimator(min_val=0, len_estimator=CategoricalEstimator()))
    print('  fit p_vec=%s' % np.round(fit.p_vec, 2))

    print('# BernoulliSet (presence/absence over a label universe)')
    d = BernoulliSetDistribution({'x': 0.8, 'y': 0.5, 'z': 0.1})
    fit = estimate(d.sampler(seed=1).sample(5000), BernoulliSetEstimator())
    print('  fit pmap=%s' % {k: round(v, 2) for k, v in fit.pmap.items()})

    print('# IntegerBernoulliSet (presence/absence over indices 0..n-1)')
    d = IntegerBernoulliSetDistribution(np.log([0.8, 0.5, 0.1, 0.3]))
    fit = estimate(d.sampler(seed=1).sample(5000), IntegerBernoulliSetEstimator(4))
    print('  fit p=%s' % np.round(np.exp(fit.log_pvec), 2))

    print('# IntegerUniformSpike (spike at k mixed with a uniform background)')
    d = IntegerUniformSpikeDistribution(3, 10, 0.6)
    fit = estimate(d.sampler(seed=1).sample(5000), IntegerUniformSpikeEstimator())
    print('  fit k=%d p=%.2f' % (fit.k, fit.p))
