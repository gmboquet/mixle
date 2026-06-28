"""The pysp.ppl dialect: declare a model with ``free`` placeholders, observe data, and ``.fit()``.

This is the concise, equation-style surface over pysparkplug's families. A ``free`` token marks a
parameter to estimate; literals stay fixed. The same RandomVariable answers the query verbs
(``.sample`` / ``.mean`` / ``.posterior``) before and after fitting. No torch or Stan required --
the fit is closed-form EM under the hood.
"""

import numpy as np

from pysp.ppl import Exponential, Mix, Normal, Poisson, Seq, free

if __name__ == '__main__':
    rng = np.random.RandomState(1)

    # 1. Scalar MLE: mark the rate free, fit to data.
    pois = Poisson(free).fit(list(rng.poisson(3.5, size=20000).astype(float)), max_its=50)
    print('Poisson rate      true 3.50   fit %.3f' % pois.dist.lam)

    expo = Exponential(free).fit(list(rng.exponential(1.0 / 0.7, size=20000)), max_its=50)
    print('Exponential rate  true 0.70   fit %.3f' % (1.0 / expo.dist.beta))

    # 2. A two-component Gaussian mixture, every parameter free.
    data = list(np.concatenate([rng.normal(-5.0, 1.0, 8000), rng.normal(5.0, 1.0, 8000)]))
    gmm = Mix([Normal(free, free), Normal(free, free)]).fit(data, max_its=80, rng=np.random.RandomState(7))
    means = sorted(c.mu for c in gmm.dist.components)
    print('Mixture means     true [-5.0, 5.0]  fit [%.2f, %.2f]' % (means[0], means[1]))
    resp = gmm.posterior([-5.0, 5.0])  # responsibilities for two query points
    print('  P(component | x=-5) = %s' % np.round(resp[0], 3))

    # 3. Variable-length sequences: wrap a leaf in Seq.
    seqs = [list(rng.normal(2.0, 1.5, size=rng.randint(5, 15))) for _ in range(2000)]
    seq = Seq(Normal(free, free)).fit(seqs, max_its=40)
    print('Seq leaf          true (2.0, 1.5)  fit (%.2f, %.2f)' % (seq.dist.dist.mu, np.sqrt(seq.dist.dist.sigma2)))

    # 4. RandomVariable algebra answers moments with no data at all.
    print('E[Normal(0,1).exp()]       = %.3f  (lognormal exp(0.5) = %.3f)' % (Normal(0, 1).exp().mean(), np.exp(0.5)))
    print('E[Normal(0,1) + Normal(5,2)] = %.3f' % (Normal(0, 1) + Normal(5, 2)).mean())
