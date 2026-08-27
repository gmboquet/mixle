"""mixle.dist — the distribution families.

The objects: the leaf distributions, multivariate families, combinators, latent-variable models and
priors. A friendly namespace alias of :mod:`mixle.stats` during the concern-oriented reorg
(``docs/ARCHITECTURE.md``) — a re-export, so every ``mixle.stats`` import keeps working. The
cross-cutting *concerns* live in :mod:`mixle.enumeration`, :mod:`mixle.inference`,
and :mod:`mixle.ops` (sampling is intrinsic behavior, not a concern: ``mixle.stats.sample``).

Naming note (T2-06): the log-normal family is named **LogGaussian** in this catalog
(:class:`LogGaussianDistribution` / :class:`LogGaussianEstimator`) but **LogNormal** in the
:mod:`mixle.ppl` dialect (:func:`mixle.ppl.LogNormal`, with :func:`mixle.ppl.LogGaussian` as its
alias). So that a search under either name finds the family on either surface, this module also
exports :data:`LogNormalDistribution` / :data:`LogNormalEstimator` as aliases of the LogGaussian
classes — the same objects, not subclasses.

Parameterization note (T2-05): :mod:`mixle.ppl` constructors use the *conventional* textbook
parameterizations, which for a few families differ from the catalog classes here. Measured
divergences (all other shared families agree argument-for-argument):

==============  ================================  =====================================  =========================
family          ppl constructor                   catalog class (this module)            conversion
==============  ================================  =====================================  =========================
Normal          ``Normal(mean, sd)``              ``GaussianDistribution(mu, sigma2)``   ``sigma2 = sd**2``
LogNormal       ``LogNormal(mu, sigma)``          ``LogGaussianDistribution(mu,          ``sigma2 = sigma**2``
                                                  sigma2)``
EMG             ``EMG(mu, sigma, rate)``          ``ExponentiallyModifiedGaussian-       ``sigma2 = sigma**2``
                                                  Distribution(mu, sigma2, lam)``        (``lam = rate``)
Gamma           ``Gamma(shape, rate)``            ``GammaDistribution(k, theta)``        ``theta = 1 / rate``
Exponential     ``Exponential(rate)``             ``ExponentialDistribution(beta)``      ``beta = 1 / rate``
Binomial        ``Binomial(n, p)``                ``BinomialDistribution(p, n)``         same meanings, positional
                                                                                         order swapped
==============  ================================  =====================================  =========================

Example — the same model written on both surfaces::

    from mixle.ppl import Normal            # ppl dialect: standard deviation
    from mixle.dist import GaussianDistribution  # catalog: variance
    Normal(0.0, 2.0)                        # sd = 2.0 ...
    GaussianDistribution(0.0, 4.0)          # ... is sigma2 = 2.0**2 = 4.0 here

The full dialect-side table lives in :mod:`mixle.ppl.distributions`.
"""

from __future__ import annotations

from mixle.stats import *  # noqa: F401,F403  (re-export the distribution family surface)
from mixle.stats import LogGaussianDistribution as LogGaussianDistribution
from mixle.stats import LogGaussianEstimator as LogGaussianEstimator
from mixle.stats import __all__ as _stats_all

# T2-06: catalog-side aliases so the family is findable under its ppl-dialect name too. Aliases of
# the same classes (not subclasses), so isinstance/identity checks agree across the two names.
LogNormalDistribution = LogGaussianDistribution
LogNormalEstimator = LogGaussianEstimator

# A NEW list: ``mixle.stats.__all__`` is a shared mutable object and must not grow these aliases.
__all__ = [*_stats_all, "LogNormalDistribution", "LogNormalEstimator"]
