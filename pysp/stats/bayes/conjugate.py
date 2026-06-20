"""Conjugate-prior posterior inference, derived from the exponential-family map.

Every exponential-family likelihood ``p(x | eta) = h(x) exp(<eta, T(x)> - A(eta))`` has a
conjugate prior ``p(eta | chi, nu) propto exp(<eta, chi> - nu A(eta))``; after observing data the
posterior is simply

    chi' = chi + sum_i T(x_i),    nu' = nu + n.

That update is *generic* -- it needs only the sufficient statistics ``T(x)`` and the pseudo-counts,
both of which :mod:`pysp.stats.exp_family` already exposes for ~22 families. This module turns that
into usable Bayesian inference:

* for the *regular* conjugate families (Gaussian, multivariate Gaussian, Poisson, Exponential,
  Bernoulli/Binomial/Geometric, Categorical, Gamma-rate) it realises the posterior in its familiar
  closed form -- exact parameter samples, marginal likelihood (evidence), and posterior predictive;
* for every other exponential-family leaf it falls back to the generic natural-parameter posterior,
  giving the posterior mean of the mean-parameters and a moment-matched point estimate.

The public entry point is :func:`conjugate_posterior`.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from scipy.special import betaln, gammaln, logsumexp, multigammaln

__all__ = [
    "ConjugatePosterior",
    "MixtureConjugatePosterior",
    "conjugate_posterior",
    "mixture_conjugate_posterior",
]


def _as_weighted_array(data: Any, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(data), dtype=np.float64) if not isinstance(data, np.ndarray) else data.astype(np.float64)
    if weights is None:
        w = np.ones(len(x), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != x.shape[0]:
            raise ValueError("weights length does not match data length")
    return x, w


class ConjugatePosterior:
    """Base class for a closed-form conjugate posterior over a likelihood's parameters.

    Subclasses fill in the family-specific hyperparameters and the four capabilities: ``mean`` (the
    posterior mean of the parameters), ``sample`` (exact draws of the parameters), ``point_estimate``
    (a fitted distribution at the posterior mean), ``log_marginal_likelihood`` (the evidence of the
    observed data under the prior), and ``posterior_predictive`` (the distribution of a new draw).
    """

    family: str = "conjugate"
    log_base: float = 0.0  # sum_i log h(x_i): the base-measure term of the absolute marginal likelihood

    # -- to be provided by subclasses -------------------------------------
    def mean(self) -> dict[str, Any]:
        raise NotImplementedError

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def point_estimate(self):
        raise NotImplementedError

    def log_marginal_likelihood(self) -> float:
        raise NotImplementedError

    def posterior_predictive(self):
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        return {"family": self.family, "mean": self.mean(), "hyper": self.hyper()}

    def hyper(self) -> dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "%s(%s)" % (type(self).__name__, self.hyper())


# ---------------------------------------------------------------------------
# Beta posterior (Bernoulli / Binomial / Geometric likelihoods)
# ---------------------------------------------------------------------------
class BetaPosterior(ConjugatePosterior):
    family = "Beta"

    def __init__(self, a: float, b: float, kind: str = "bernoulli", n_trials: int = 1):
        self.a = float(a)
        self.b = float(b)
        self.kind = kind
        self.n_trials = int(n_trials)
        self._a0 = None  # prior hyperparameters, set by the builder for the evidence term
        self._b0 = None

    def mean(self) -> dict[str, Any]:
        return {"p": self.a / (self.a + self.b)}

    def variance(self) -> float:
        a, b = self.a, self.b
        return a * b / ((a + b) ** 2 * (a + b + 1.0))

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        rng = rng or np.random.RandomState()
        return {"p": rng.beta(self.a, self.b, size=n)}

    def point_estimate(self):
        p = self.a / (self.a + self.b)
        if self.kind == "geometric":
            from pysp.stats.leaf.geometric import GeometricDistribution

            return GeometricDistribution(p)
        if self.kind == "binomial":
            from pysp.stats.leaf.binomial import BinomialDistribution

            return BinomialDistribution(p, self.n_trials)
        from pysp.stats.leaf.bernoulli import BernoulliDistribution

        return BernoulliDistribution(p)

    def posterior_predictive(self):
        # The plug-in predictive at the posterior mean (Beta-Bernoulli predictive prob == mean).
        return self.point_estimate()

    def hyper(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b}

    def _set_prior(self, a0: float, b0: float, n: float, s: float, extra: float) -> None:
        # cache enough to compute the marginal likelihood; ``extra`` carries failure mass / n_trials
        self._a0, self._b0, self._n, self._s, self._extra = a0, b0, n, s, extra

    def log_marginal_likelihood(self) -> float:
        if self._a0 is None:
            raise ValueError("prior hyperparameters were not recorded")
        # Beta-Bernoulli/Binomial evidence: B(a0+s, b0+f) / B(a0, b0); ``log_base`` carries the
        # product of binomial coefficients for Binomial data (0 for Bernoulli/Geometric).
        return float(self.log_base + betaln(self.a, self.b) - betaln(self._a0, self._b0))


def _build_bernoulli(dist, data, weights, prior) -> BetaPosterior:
    x, w = _as_weighted_array(data, weights)
    n = float(w.sum())
    s = float(np.dot(w, x))
    a0, b0 = (prior or {}).get("a", 1.0), (prior or {}).get("b", 1.0)
    post = BetaPosterior(a0 + s, b0 + n - s, kind="bernoulli")
    post._set_prior(a0, b0, n, s, n - s)
    return post


def _build_geometric(dist, data, weights, prior) -> BetaPosterior:
    # Geometric(p) on {1,2,...}: likelihood propto p^N (1-p)^(sum_x - N).
    x, w = _as_weighted_array(data, weights)
    n = float(w.sum())
    sx = float(np.dot(w, x))
    a0, b0 = (prior or {}).get("a", 1.0), (prior or {}).get("b", 1.0)
    post = BetaPosterior(a0 + n, b0 + sx - n, kind="geometric")
    post._set_prior(a0, b0, n, n, sx - n)
    return post


def _build_binomial(dist, data, weights, prior) -> BetaPosterior:
    n_trials = int(getattr(dist, "n", getattr(dist, "n_trials", 1)))
    x, w = _as_weighted_array(data, weights)
    N = float(w.sum())
    s = float(np.dot(w, x))
    a0, b0 = (prior or {}).get("a", 1.0), (prior or {}).get("b", 1.0)
    failures = n_trials * N - s
    post = BetaPosterior(a0 + s, b0 + failures, kind="binomial", n_trials=n_trials)
    post.log_base = float(np.dot(w, gammaln(n_trials + 1.0) - gammaln(x + 1.0) - gammaln(n_trials - x + 1.0)))
    post._set_prior(a0, b0, N, s, failures)
    return post


# ---------------------------------------------------------------------------
# Gamma posterior (Poisson / Exponential / Gamma-known-shape rate)
# ---------------------------------------------------------------------------
class GammaRatePosterior(ConjugatePosterior):
    family = "Gamma"

    def __init__(self, shape: float, rate: float, kind: str = "poisson", known_shape: float = 1.0):
        self.shape = float(shape)  # posterior Gamma shape (A)
        self.rate = float(rate)  # posterior Gamma rate (B)
        self.kind = kind
        self.known_shape = float(known_shape)
        self._a0 = None

    def mean(self) -> dict[str, Any]:
        return {"rate": self.shape / self.rate}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        rng = rng or np.random.RandomState()
        return {"rate": rng.gamma(self.shape, 1.0 / self.rate, size=n)}

    def point_estimate(self):
        lam = self.shape / self.rate
        if self.kind == "poisson":
            from pysp.stats.leaf.poisson import PoissonDistribution

            return PoissonDistribution(lam)
        if self.kind == "exponential":
            from pysp.stats.leaf.exponential import ExponentialDistribution

            return ExponentialDistribution(1.0 / lam)  # ExponentialDistribution is mean-parameterised
        from pysp.stats.leaf.gamma import GammaDistribution

        return GammaDistribution(self.known_shape, 1.0 / lam)  # k=shape, theta=scale=1/rate

    def posterior_predictive(self):
        if self.kind == "poisson":
            # Poisson-Gamma predictive is Negative-Binomial(r=A, p=B/(B+1)).
            from pysp.stats.leaf.negative_binomial import NegativeBinomialDistribution

            return NegativeBinomialDistribution(self.shape, self.rate / (self.rate + 1.0))
        return self.point_estimate()

    def hyper(self) -> dict[str, Any]:
        return {"shape": self.shape, "rate": self.rate}

    def _set_prior(self, a0, b0, n, sx):
        self._a0, self._b0, self._n, self._sx = a0, b0, n, sx

    def log_marginal_likelihood(self) -> float:
        if self._a0 is None:
            raise ValueError("prior hyperparameters were not recorded")
        # Gamma-conjugate evidence; ``log_base`` carries sum_i log h(x_i) (-sum log x_i! for Poisson, 0
        # for Exponential), making this the absolute log marginal likelihood usable across families.
        return float(
            self.log_base
            + gammaln(self.shape)
            - gammaln(self._a0)
            + self._a0 * math.log(self._b0)
            - self.shape * math.log(self.rate)
        )


def _build_poisson(dist, data, weights, prior) -> GammaRatePosterior:
    x, w = _as_weighted_array(data, weights)
    n = float(w.sum())
    sx = float(np.dot(w, x))
    a0, b0 = (prior or {}).get("shape", 1e-3), (prior or {}).get("rate", 1e-3)
    post = GammaRatePosterior(a0 + sx, b0 + n, kind="poisson")
    post.log_base = float(-np.dot(w, gammaln(x + 1.0)))  # sum_i log(1/x_i!)
    post._set_prior(a0, b0, n, sx)
    return post


def _build_exponential(dist, data, weights, prior) -> GammaRatePosterior:
    x, w = _as_weighted_array(data, weights)
    n = float(w.sum())
    sx = float(np.dot(w, x))
    a0, b0 = (prior or {}).get("shape", 1e-3), (prior or {}).get("rate", 1e-3)
    post = GammaRatePosterior(a0 + n, b0 + sx, kind="exponential")
    post._set_prior(a0, b0, n, sx)
    return post


# ---------------------------------------------------------------------------
# Dirichlet posterior (Categorical / IntegerCategorical)
# ---------------------------------------------------------------------------
class DirichletPosterior(ConjugatePosterior):
    family = "Dirichlet"

    def __init__(self, alpha: np.ndarray, support: list, kind: str = "categorical", min_val: int = 0):
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.support = list(support)
        self.kind = kind
        self.min_val = int(min_val)
        self._alpha0 = None

    def mean(self) -> dict[str, Any]:
        p = self.alpha / self.alpha.sum()
        return {"probs": p, "map": dict(zip(self.support, p))}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        rng = rng or np.random.RandomState()
        return {"probs": rng.dirichlet(self.alpha, size=n)}

    def point_estimate(self):
        p = self.alpha / self.alpha.sum()
        if self.kind == "integer_categorical":
            from pysp.stats.leaf.integer_categorical import IntegerCategoricalDistribution

            return IntegerCategoricalDistribution(self.min_val, p)
        from pysp.stats.leaf.categorical import CategoricalDistribution

        return CategoricalDistribution(dict(zip(self.support, p)))

    def posterior_predictive(self):
        return self.point_estimate()

    def hyper(self) -> dict[str, Any]:
        return {"alpha": self.alpha}

    def _set_prior(self, alpha0):
        self._alpha0 = np.asarray(alpha0, dtype=np.float64)

    def log_marginal_likelihood(self) -> float:
        if self._alpha0 is None:
            raise ValueError("prior hyperparameters were not recorded")
        a0, an = self._alpha0, self.alpha
        # Dirichlet-multinomial evidence (multinomial coefficient dropped -- constant in the parameter)
        return float(gammaln(a0.sum()) - gammaln(an.sum()) + (gammaln(an).sum() - gammaln(a0).sum()))


def _categorical_counts(dist, data, weights):
    x = list(data)
    if weights is None:
        weights = np.ones(len(x))
    counts: Counter = Counter()
    for v, w in zip(x, np.asarray(weights, dtype=np.float64)):
        counts[v] += float(w)
    return counts


def _build_categorical(dist, data, weights, prior) -> DirichletPosterior:
    # support comes from the distribution if available, else from the data
    support = list(getattr(dist, "pmap", {}).keys()) if getattr(dist, "pmap", None) else None
    counts = _categorical_counts(dist, data, weights)
    if support is None:
        support = sorted(counts.keys(), key=lambda v: (str(type(v)), v))
    a0_scalar = (prior or {}).get("alpha", 1.0)
    alpha0 = np.full(len(support), float(a0_scalar))
    cvec = np.array([counts.get(s, 0.0) for s in support], dtype=np.float64)
    post = DirichletPosterior(alpha0 + cvec, support, kind="categorical")
    post._set_prior(alpha0)
    return post


def _build_integer_categorical(dist, data, weights, prior) -> DirichletPosterior:
    x, w = _as_weighted_array(data, weights)
    min_val = int(getattr(dist, "min_val", int(x.min())))
    k = len(getattr(dist, "p_vec", getattr(dist, "prob_vec", [])))
    if k == 0:
        k = int(x.max()) - min_val + 1
    support = list(range(min_val, min_val + k))
    cvec = np.zeros(k, dtype=np.float64)
    for xi, wi in zip(x.astype(int), w):
        if 0 <= xi - min_val < k:
            cvec[xi - min_val] += wi
    a0_scalar = (prior or {}).get("alpha", 1.0)
    alpha0 = np.full(k, float(a0_scalar))
    post = DirichletPosterior(alpha0 + cvec, support, kind="integer_categorical", min_val=min_val)
    post._set_prior(alpha0)
    return post


# ---------------------------------------------------------------------------
# Normal-Inverse-Gamma posterior (Gaussian, unknown mean AND variance)
# ---------------------------------------------------------------------------
class NormalInverseGammaPosterior(ConjugatePosterior):
    family = "NormalInverseGamma"

    def __init__(self, m: float, kappa: float, a: float, b: float):
        self.m = float(m)  # posterior mean location
        self.kappa = float(kappa)  # mean pseudo-count
        self.a = float(a)  # inverse-gamma shape
        self.b = float(b)  # inverse-gamma scale
        self._prior = None

    def mean(self) -> dict[str, Any]:
        return {"mu": self.m, "sigma2": self.b / (self.a - 1.0) if self.a > 1.0 else float("inf")}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        rng = rng or np.random.RandomState()
        sigma2 = 1.0 / rng.gamma(self.a, 1.0 / self.b, size=n)  # InvGamma(a,b)
        mu = rng.normal(self.m, np.sqrt(sigma2 / self.kappa))
        return {"mu": mu, "sigma2": sigma2}

    def point_estimate(self):
        from pysp.stats.leaf.gaussian import GaussianDistribution

        sigma2 = self.b / (self.a - 1.0) if self.a > 1.0 else self.b / self.a
        return GaussianDistribution(self.m, sigma2)

    def posterior_predictive(self):
        # Marginalising (mu, sigma2) gives a Student-t: df=2a, loc=m, scale^2=b(kappa+1)/(a*kappa).
        from pysp.stats.leaf.student_t import StudentTDistribution

        scale = math.sqrt(self.b * (self.kappa + 1.0) / (self.a * self.kappa))
        return StudentTDistribution(2.0 * self.a, self.m, scale)

    def hyper(self) -> dict[str, Any]:
        return {"m": self.m, "kappa": self.kappa, "a": self.a, "b": self.b}

    def _set_prior(self, m0, k0, a0, b0, n):
        self._prior = (m0, k0, a0, b0, n)

    def log_marginal_likelihood(self) -> float:
        if self._prior is None:
            raise ValueError("prior hyperparameters were not recorded")
        m0, k0, a0, b0, n = self._prior
        return float(
            gammaln(self.a)
            - gammaln(a0)
            + a0 * math.log(b0)
            - self.a * math.log(self.b)
            + 0.5 * (math.log(k0) - math.log(self.kappa))
            - 0.5 * n * math.log(2.0 * math.pi)
        )


def _build_gaussian(dist, data, weights, prior) -> NormalInverseGammaPosterior:
    x, w = _as_weighted_array(data, weights)
    n = float(w.sum())
    sx = float(np.dot(w, x))
    sx2 = float(np.dot(w, x * x))
    xbar = sx / n if n > 0 else 0.0
    p = prior or {}
    m0 = float(p.get("m", 0.0))
    k0 = float(p.get("kappa", 1e-3))
    a0 = float(p.get("a", 1e-3))
    b0 = float(p.get("b", 1e-3))
    kn = k0 + n
    mn = (k0 * m0 + sx) / kn
    an = a0 + 0.5 * n
    # b_n = b0 + 0.5*sum w (x-xbar)^2 + 0.5 k0 n (xbar-m0)^2 / kn
    ss = sx2 - n * xbar * xbar
    bn = b0 + 0.5 * ss + 0.5 * k0 * n * (xbar - m0) ** 2 / kn
    post = NormalInverseGammaPosterior(mn, kn, an, bn)
    post._set_prior(m0, k0, a0, b0, n)
    return post


# ---------------------------------------------------------------------------
# Normal-Inverse-Wishart posterior (multivariate Gaussian, unknown mean AND covariance)
# ---------------------------------------------------------------------------
class NormalInverseWishartPosterior(ConjugatePosterior):
    family = "NormalInverseWishart"

    def __init__(self, m: np.ndarray, kappa: float, nu: float, psi: np.ndarray):
        self.m = np.asarray(m, dtype=np.float64)
        self.kappa = float(kappa)
        self.nu = float(nu)
        self.psi = np.asarray(psi, dtype=np.float64)
        self.d = self.m.shape[0]
        self._prior = None

    def mean(self) -> dict[str, Any]:
        cov = self.psi / (self.nu - self.d - 1.0) if self.nu > self.d + 1 else None
        return {"mean": self.m, "cov": cov}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        rng = rng or np.random.RandomState()
        means = np.empty((n, self.d))
        covs = np.empty((n, self.d, self.d))
        psi_inv = np.linalg.inv(self.psi)
        for i in range(n):
            sigma = _sample_inverse_wishart(self.nu, psi_inv, rng)
            covs[i] = sigma
            means[i] = rng.multivariate_normal(self.m, sigma / self.kappa)
        return {"mean": means, "cov": covs}

    def point_estimate(self):
        from pysp.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

        cov = self.psi / (self.nu - self.d - 1.0) if self.nu > self.d + 1 else self.psi / self.nu
        return MultivariateGaussianDistribution(self.m, cov)

    def posterior_predictive(self):
        from pysp.stats.multivariate.multivariate_student_t import MultivariateStudentTDistribution

        df = self.nu - self.d + 1.0
        shape = self.psi * (self.kappa + 1.0) / (self.kappa * df)
        return MultivariateStudentTDistribution(df, self.m, shape)

    def hyper(self) -> dict[str, Any]:
        return {"m": self.m, "kappa": self.kappa, "nu": self.nu, "psi": self.psi}

    def _set_prior(self, m0, k0, nu0, psi0, n):
        self._prior = (np.asarray(m0), k0, nu0, np.asarray(psi0), n)

    def log_marginal_likelihood(self) -> float:
        if self._prior is None:
            raise ValueError("prior hyperparameters were not recorded")
        m0, k0, nu0, psi0, n = self._prior
        d = self.d
        return float(
            multigammaln(self.nu / 2.0, d)
            - multigammaln(nu0 / 2.0, d)
            + (nu0 / 2.0) * _logdet(psi0)
            - (self.nu / 2.0) * _logdet(self.psi)
            + (d / 2.0) * (math.log(k0) - math.log(self.kappa))
            - (n * d / 2.0) * math.log(math.pi)
        )


def _logdet(a: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(a)
    return float(ld)


def _sample_inverse_wishart(nu: float, psi_inv: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # Sample W ~ Wishart(nu, psi_inv) via Bartlett, then invert to get Inverse-Wishart(nu, psi).
    d = psi_inv.shape[0]
    chol = np.linalg.cholesky(psi_inv)
    a = np.zeros((d, d))
    for i in range(d):
        a[i, i] = math.sqrt(rng.chisquare(nu - i))
        for j in range(i):
            a[i, j] = rng.normal()
    w = chol @ a @ a.T @ chol.T
    return np.linalg.inv(w)


def _build_mvn(dist, data, weights, prior) -> NormalInverseWishartPosterior:
    x = np.asarray(list(data), dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    d = x.shape[1]
    w = np.ones(x.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    n = float(w.sum())
    xbar = (w[:, None] * x).sum(axis=0) / n
    diff = x - xbar
    scatter = (w[:, None] * diff).T @ diff
    p = prior or {}
    m0 = np.asarray(p.get("m", np.zeros(d)), dtype=np.float64)
    k0 = float(p.get("kappa", 1e-3))
    nu0 = float(p.get("nu", d + 2.0))
    psi0 = np.asarray(p.get("psi", np.eye(d) * 1e-3), dtype=np.float64)
    kn = k0 + n
    mn = (k0 * m0 + n * xbar) / kn
    nun = nu0 + n
    dm = xbar - m0
    psin = psi0 + scatter + (k0 * n / kn) * np.outer(dm, dm)
    post = NormalInverseWishartPosterior(mn, kn, nun, psin)
    post._set_prior(m0, k0, nu0, psi0, n)
    return post


# ---------------------------------------------------------------------------
# Generic exponential-family fallback (any family with a to_exponential_family map)
# ---------------------------------------------------------------------------
class GenericExpFamilyPosterior(ConjugatePosterior):
    family = "ExponentialFamily(generic)"

    def __init__(self, dist, chi: np.ndarray, nu: float):
        self.dist = dist
        self.chi = np.asarray(chi, dtype=np.float64)
        self.nu = float(nu)

    def mean_parameters(self) -> np.ndarray:
        """Posterior mean of the mean-parameters E[T(x)] = chi / nu."""
        return self.chi / self.nu

    def mean(self) -> dict[str, Any]:
        return {"mean_parameters": self.mean_parameters()}

    def point_estimate(self):
        """Distribution whose mean-parameters match the posterior mean (moment matching)."""
        from pysp.stats.exp_family import to_exponential_family

        form = to_exponential_family(self.dist)
        target = self.mean_parameters()
        # Where a closed-form dual map exists, invert grad A; else fall back to the source MLE shape.
        recon = getattr(form, "from_mean_parameters", None)
        if callable(recon):
            d = recon(target)
            if d is not None:
                return d
        return self.dist  # best-effort: caller still has the natural posterior hyperparameters

    def hyper(self) -> dict[str, Any]:
        return {"chi": self.chi, "nu": self.nu}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None):
        raise NotImplementedError(
            "generic posterior sampling requires the family's dual map A(eta); not available for "
            "%s. Its closed-form conjugate (if any) is not yet registered." % type(self.dist).__name__
        )

    def log_marginal_likelihood(self) -> float:
        raise NotImplementedError("generic marginal likelihood requires the family's prior normaliser")

    def posterior_predictive(self):
        return self.point_estimate()


def _build_generic(dist, data, weights, prior) -> GenericExpFamilyPosterior:
    from pysp.stats.exp_family import to_exponential_family

    form = to_exponential_family(dist)
    if form is None:
        raise TypeError(
            "%s is not an exponential family with a registered map; no conjugate posterior available."
            % type(dist).__name__
        )
    x = list(data)
    stats = np.asarray(form.engine.to_numpy(form.sufficient_statistics(x)), dtype=np.float64)
    if weights is None:
        sum_t = stats.sum(axis=0)
        n = float(len(x))
    else:
        w = np.asarray(weights, dtype=np.float64)
        sum_t = (w[:, None] * stats).sum(axis=0)
        n = float(w.sum())
    p = prior or {}
    chi0 = np.asarray(p.get("chi", np.zeros(stats.shape[1])), dtype=np.float64)
    nu0 = float(p.get("nu", 0.0))
    return GenericExpFamilyPosterior(dist, chi0 + sum_t, nu0 + n)


# ---------------------------------------------------------------------------
# Registry + dispatch
# ---------------------------------------------------------------------------
def _registry():
    from pysp.stats.leaf.bernoulli import BernoulliDistribution
    from pysp.stats.leaf.binomial import BinomialDistribution
    from pysp.stats.leaf.categorical import CategoricalDistribution
    from pysp.stats.leaf.exponential import ExponentialDistribution
    from pysp.stats.leaf.gaussian import GaussianDistribution
    from pysp.stats.leaf.geometric import GeometricDistribution
    from pysp.stats.leaf.integer_categorical import IntegerCategoricalDistribution
    from pysp.stats.leaf.poisson import PoissonDistribution
    from pysp.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

    return {
        BernoulliDistribution: _build_bernoulli,
        BinomialDistribution: _build_binomial,
        GeometricDistribution: _build_geometric,
        PoissonDistribution: _build_poisson,
        ExponentialDistribution: _build_exponential,
        CategoricalDistribution: _build_categorical,
        IntegerCategoricalDistribution: _build_integer_categorical,
        GaussianDistribution: _build_gaussian,
        MultivariateGaussianDistribution: _build_mvn,
    }


def conjugate_posterior(dist, data, prior: dict | None = None, weights: np.ndarray | None = None) -> ConjugatePosterior:
    """Closed-form conjugate posterior over the parameters of ``dist`` given ``data``.

    Args:
        dist: A pysp likelihood distribution instance whose *type* selects the conjugate family
            (its parameter values are ignored except for fixed structure such as a Binomial's
            number of trials or a Categorical's support). Any exponential-family leaf is accepted;
            the regular conjugates get a closed-form realiser, the rest the generic fallback.
        data: A sequence of observations of the kind ``dist`` scores.
        prior: Optional dict of conjugate-prior hyperparameters (family specific, e.g.
            ``{"a": 1.0, "b": 1.0}`` for Beta, ``{"m":0,"kappa":1e-3,"a":1e-3,"b":1e-3}`` for
            Gaussian). ``None`` uses a weak proper prior.
        weights: Optional per-observation weights (e.g. EM responsibilities).

    Returns:
        A :class:`ConjugatePosterior` exposing ``mean``, ``sample``, ``point_estimate``,
        ``log_marginal_likelihood``, ``posterior_predictive`` and ``summary``.
    """
    builder = _registry().get(type(dist))
    if builder is not None:
        return builder(dist, data, weights, prior)
    return _build_generic(dist, data, weights, prior)


# ---------------------------------------------------------------------------
# Mixtures of conjugate priors (Diaconis-Ylvisaker: a mixture of conjugates is conjugate)
# ---------------------------------------------------------------------------
class MixtureConjugatePosterior(ConjugatePosterior):
    """Posterior under a prior that is itself a mixture of conjugate priors.

    Diaconis-Ylvisaker (1979): if the prior is ``sum_m w_m * pi_m(theta)`` with each ``pi_m``
    conjugate to the likelihood, the posterior is again a mixture of the component conjugate
    posteriors, ``sum_m w'_m * pi_m^post(theta)``, with the mixing weights reweighted by each
    component's marginal likelihood ``Z_m``:

        w'_m  proportional to  w_m * Z_m.

    Everything stays closed-form, so a prior can be multimodal (e.g. "the rate is near 1 OR near
    10") or robust (a heavy-tailed prior as a mixture of conjugates) without losing exact inference.
    """

    family = "MixtureOfConjugates"

    def __init__(self, components, post_weights, prior_weights, comp_log_evidence):
        self.components: list[ConjugatePosterior] = list(components)
        self.weights = np.asarray(post_weights, dtype=np.float64)  # posterior mixing weights (normalised)
        self.prior_weights = np.asarray(prior_weights, dtype=np.float64)
        self.comp_log_evidence = np.asarray(comp_log_evidence, dtype=np.float64)

    def mean(self) -> dict[str, Any]:
        # posterior mean of theta = sum_m w'_m E_m[theta]; per-key weighted average of the components.
        out: dict[str, Any] = {}
        keys = self.components[0].mean().keys()
        for k in keys:
            vals = [c.mean()[k] for c in self.components]
            try:
                out[k] = sum(w * np.asarray(v, dtype=np.float64) for w, v in zip(self.weights, vals))
            except (TypeError, ValueError):
                out[k] = vals[int(np.argmax(self.weights))]  # non-numeric (e.g. dicts): take the MAP component
        return out

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        rng = rng or np.random.RandomState()
        counts = rng.multinomial(n, self.weights)  # how many draws come from each component
        gathered: dict[str, list] = {}
        for m, cm in enumerate(counts):
            if cm == 0:
                continue
            s = self.components[m].sample(int(cm), rng)
            for key, arr in s.items():
                gathered.setdefault(key, []).append(np.asarray(arr))
        return {key: np.concatenate(parts) for key, parts in gathered.items()}

    def point_estimate(self):
        """The maximum-a-posteriori *component*'s point estimate (the dominant conjugate mode)."""
        return self.components[int(np.argmax(self.weights))].point_estimate()

    def posterior_predictive(self):
        """A pysp MixtureDistribution of the component predictives, weighted by the posterior weights."""
        from pysp.stats.latent.mixture import MixtureDistribution

        return MixtureDistribution([c.posterior_predictive() for c in self.components], list(self.weights))

    def log_marginal_likelihood(self) -> float:
        # evidence of the whole data under the mixture prior: sum_m w_m Z_m
        return float(logsumexp(np.log(self.prior_weights) + self.comp_log_evidence))

    def hyper(self) -> dict[str, Any]:
        return {"weights": self.weights, "components": [c.hyper() for c in self.components]}


def mixture_conjugate_posterior(
    dist,
    data,
    priors: list[dict],
    prior_weights: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> MixtureConjugatePosterior:
    """Posterior under a prior that is a mixture of conjugate priors (Diaconis-Ylvisaker).

    Args:
        dist: The likelihood distribution (must map to a *closed-form* conjugate realiser, since the
            reweighting needs each component's marginal likelihood).
        data: The observations.
        priors: One hyperparameter dict per mixture component (same keys :func:`conjugate_posterior`
            accepts for this family).
        prior_weights: Prior mixing weights ``w_m`` (default uniform); normalised internally.
        weights: Optional per-observation weights.

    Returns:
        A :class:`MixtureConjugatePosterior`: the exact posterior, again a mixture of conjugate
        posteriors with weights ``w'_m proportional to w_m * Z_m``.
    """
    m = len(priors)
    if m == 0:
        raise ValueError("need at least one prior component")
    pw = np.ones(m) / m if prior_weights is None else np.asarray(prior_weights, dtype=np.float64)
    pw = pw / pw.sum()
    components = [conjugate_posterior(dist, data, prior=pr, weights=weights) for pr in priors]
    try:
        log_evidence = np.array([c.log_marginal_likelihood() for c in components])
    except NotImplementedError as exc:
        raise TypeError(
            "mixture_conjugate_posterior needs a closed-form family (it reweights by each "
            "component's marginal likelihood); %s has only the generic posterior." % type(dist).__name__
        ) from exc
    log_post = np.log(pw) + log_evidence
    post_weights = np.exp(log_post - logsumexp(log_post))
    return MixtureConjugatePosterior(components, post_weights, pw, log_evidence)
