"""Gallery: every univariate (scalar) distribution family, each built / sampled / re-estimated.

For each family we instantiate a known distribution, draw an i.i.d. sample, fit a fresh estimator
with the one-pass ``estimate`` helper, and print the true vs. recovered parameters. Fully
self-contained (random data only). This is the quickest tour of pysparkplug's scalar leaf families.
"""

from pysp.stats import *

# (label, true distribution, estimator) -- continuous then discrete.
CASES = [
    ('Gaussian',          GaussianDistribution(1.5, 4.0),               GaussianEstimator()),
    ('LogGaussian',       LogGaussianDistribution(0.5, 0.25),           LogGaussianEstimator()),
    ('Gamma',             GammaDistribution(3.0, 2.0),                  GammaEstimator()),
    ('Beta',              BetaDistribution(2.0, 5.0),                   BetaEstimator()),
    ('Exponential',       ExponentialDistribution(2.0),                 ExponentialEstimator()),
    ('Weibull',           WeibullDistribution(1.5, 2.0),                WeibullEstimator()),
    ('Rayleigh',          RayleighDistribution(2.0),                    RayleighEstimator()),
    ('Laplace',           LaplaceDistribution(1.0, 2.0),                LaplaceEstimator()),
    ('Logistic',          LogisticDistribution(1.0, 0.5),               LogisticEstimator()),
    ('StudentT (df=5)',   StudentTDistribution(5.0, 1.0, 2.0),          StudentTEstimator(df=5.0)),
    ('Pareto',            ParetoDistribution(1.0, 3.0),                 ParetoEstimator()),
    ('Uniform',           UniformDistribution(-2.0, 3.0),               UniformEstimator()),
    ('Poisson',           PoissonDistribution(4.0),                     PoissonEstimator()),
    ('Geometric',         GeometricDistribution(0.3),                   GeometricEstimator()),
    ('Binomial',          BinomialDistribution(0.4, 10),                BinomialEstimator(min_val=0)),
    ('NegativeBinomial',  NegativeBinomialDistribution(4.0, 0.4),       NegativeBinomialEstimator(r=4.0)),
    ('Bernoulli',         BernoulliDistribution(0.7),                   BernoulliEstimator()),
    ('Categorical',       CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2}), CategoricalEstimator()),
    ('IntegerCategorical', IntegerCategoricalDistribution(0, [0.5, 0.3, 0.2]),     IntegerCategoricalEstimator()),
]

if __name__ == '__main__':
    for label, true_dist, estimator in CASES:
        data = true_dist.sampler(seed=1).sample(5000)
        fit = estimate(data, estimator)
        print('%-20s' % label)
        print('  true: %s' % true_dist)
        print('  fit : %s' % fit)
