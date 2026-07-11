"""Automatic dependency-structure learning for heterogeneous records -- the tagline, taken literally.

``CompositeDistribution`` models a record's fields as *independent* (Naive-Bayes under a mixture). But real
heterogeneous data has cross-field dependence -- a category shifts a real's mean, a count's rate tracks another
field -- and modeling it is worth a great deal of likelihood (a blatant category->Gaussian link is ~1000 nats on
600 rows). No mainstream tool discovers that structure across *arbitrary* families: Stan/PyMC make you write it,
sklearn/pomegranate mixtures assume independence, bnlearn/pgmpy are discrete-or-Gaussian only.

This module closes the gap. Dependence is detected by modeling it: fit ``P(child)`` vs
``P(child | parent)`` and compare description length (:func:`dependency_gain`). The winning edges are assembled
into a :class:`DependencyTreeDistribution` -- a directed forest over the record where each field is either a
marginal or a per-parent-value conditional (a real :class:`~mixle.stats.combinator.conditional.ConditionalDistribution`
edge) -- and :func:`learn_structure` picks the forest and fits it automatically. The result scores, samples, and
composes like any mixle distribution, but *models the dependence a composite drops*.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.inference.estimation import fit
from mixle.stats.combinator.conditional import ConditionalDistributionEstimator


def _columns(data: Sequence[tuple]) -> list[list[Any]]:
    """Transpose a list of record tuples into per-field columns."""
    n = len(data[0])
    return [[row[i] for row in data] for i in range(n)]


def _is_discrete(column: Sequence[Any], *, max_levels: int = 64) -> bool:
    """A field can be a *parent* if it takes few discrete values (categorical, boolean, or small-range integer)."""
    vals = set()
    for v in column:
        if isinstance(v, float):
            return False
        vals.add(v)
        if len(vals) > max_levels:
            return False
    return True


def _field_estimator(column: Sequence[Any]) -> Any:
    """Infer a single mixle estimator template for a field from its values (via the automatic detector)."""
    from mixle.utils.automatic import get_estimator

    return get_estimator(list(column))


def dependency_gain(
    parent: Sequence[Any],
    child: Sequence[Any],
    child_estimator: Any,
    *,
    max_its: int = 30,
    penalty: str = "bic",
    rng: np.random.RandomState | None = None,
) -> float:
    """Description-length gain (nats) of modeling ``child`` conditioned on a discrete ``parent`` vs. independently.

    Fits the marginal ``P(child)`` and the conditional ``P(child | parent)`` (a child model per parent value) on
    the same data and returns ``LL_cond - LL_marginal`` minus a complexity penalty for the extra parameters
    (BIC: ``0.5 * (levels - 1) * k * ln n``). Positive means the dependence is worth modeling. This is a
    model-based dependency test -- it works across *any* pair of families, unlike a same-type MI estimate.
    ``rng`` seeds the fits' EM initializations (``None`` = a fixed seed: deterministic by default; matters when
    the child family needs a randomized init, e.g. a mixture).
    """
    rng = np.random.RandomState(0) if rng is None else rng
    child = list(child)
    n = len(child)
    levels = sorted(set(parent))
    marginal = fit(child, _clone(child_estimator), max_its=max_its, out=None, rng=rng)
    ll_marginal = float(np.sum(marginal.seq_log_density(marginal.dist_to_encoder().seq_encode(child))))

    pairs = list(zip(parent, child))
    cond_est = ConditionalDistributionEstimator(
        estimator_map={lv: _clone(child_estimator) for lv in levels}, given_estimator=None
    )
    cond = fit(pairs, cond_est, max_its=max_its, out=None, rng=rng)
    enc = cond.dist_to_encoder().seq_encode(pairs)
    ll_cond = float(np.sum(cond.seq_log_density(enc)))

    if penalty == "bic":
        k = _num_free_params(marginal)
        extra = (len(levels) - 1) * k
        pen = 0.5 * extra * np.log(max(n, 2))
    else:
        pen = 0.0
    return ll_cond - ll_marginal - pen


def _is_numeric(column: Sequence[Any]) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in column)


class LinearGaussianEdge:
    """A *regression* edge: ``P(child | parent) = Normal(a + b*parent, sigma2)``.

    Where the current per-parent-bin conditional needs ``bins * k`` parameters (and coarse bins) to model a smooth
    continuous dependence, this captures it with a single slope ``b`` — far more statistically efficient and exact
    for a linear relationship. Slots into :class:`DependencyTreeDistribution` as a factor with an identity binner
    (the raw parent value drives the conditional)."""

    __pysp_serializable__ = True  # opt in to mixle JSON serialization (a/b/sigma2 round-trip via __dict__)

    def __init__(self, a: float, b: float, sigma2: float) -> None:
        self.a, self.b, self.sigma2 = float(a), float(b), max(float(sigma2), 1e-12)

    def log_density(self, x: tuple) -> float:
        """Evaluate ``log p(child | parent)`` for one parent-child pair."""
        parent, child = x
        resid = float(child) - (self.a + self.b * float(parent))
        return float(-0.5 * np.log(2.0 * np.pi * self.sigma2) - 0.5 * resid * resid / self.sigma2)

    def dist_to_encoder(self) -> Any:
        """Return the encoder for parent-child pairs."""
        return _EdgeEncoder()

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        """Evaluate log densities for encoded parent-child pairs."""
        parent, child = encoded
        resid = np.asarray(child, dtype=float) - (self.a + self.b * np.asarray(parent, dtype=float))
        return -0.5 * np.log(2.0 * np.pi * self.sigma2) - 0.5 * resid * resid / self.sigma2

    def sampler(self, seed: int | None = None) -> Any:
        """Return a sampler for the linear Gaussian edge."""
        return _LinearGaussianEdgeSampler(self, seed)

    def __str__(self) -> str:
        return f"LinearGaussianEdge(child ~ N({self.a:.3g} + {self.b:.3g}*parent, {self.sigma2:.3g}))"


class _EdgeEncoder:
    def seq_encode(self, pairs: Sequence[tuple]) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(pairs, dtype=float)
        return arr[:, 0], arr[:, 1]


class _LinearGaussianEdgeSampler:
    def __init__(self, edge: LinearGaussianEdge, seed: int | None) -> None:
        self.edge = edge
        self.rng = np.random.RandomState(seed)

    def sample_given(self, parent: Any) -> float:
        mean = self.edge.a + self.edge.b * float(parent)
        return float(self.rng.normal(mean, np.sqrt(self.edge.sigma2)))


def fit_linear_gaussian_edge(pairs: Sequence[tuple]) -> LinearGaussianEdge:
    """OLS fit of a linear-Gaussian conditional ``child ~ a + b*parent`` (closed form)."""
    arr = np.asarray(pairs, dtype=float)
    p, c = arr[:, 0], arr[:, 1]
    pm, cm = float(p.mean()), float(c.mean())
    denom = float(np.sum((p - pm) ** 2))
    b = float(np.sum((p - pm) * (c - cm)) / denom) if denom > 1e-12 else 0.0
    a = cm - b * pm
    resid = c - (a + b * p)
    return LinearGaussianEdge(a, b, float(np.var(resid)))


def regression_gain(
    parent: Sequence[Any],
    child: Sequence[Any],
    child_estimator: Any,
    *,
    max_its: int = 30,
    penalty: str = "bic",
    rng: np.random.RandomState | None = None,
) -> float:
    """Description-length gain (nats) of a linear-Gaussian *regression* edge ``child ~ a + b*parent`` over the
    child marginal. One extra parameter (the slope) vs. the ``bins * k`` a binned conditional spends — so for a
    real linear dependence this beats binning decisively. Returns ``-inf`` when a regression is undefined.
    ``rng`` seeds the marginal fit's EM initialization (``None`` = a fixed seed: deterministic by default)."""
    rng = np.random.RandomState(0) if rng is None else rng
    p = np.asarray(parent, dtype=float)
    c = np.asarray(child, dtype=float)
    n = len(c)
    if n < 3 or float(np.var(p)) < 1e-12:
        return float("-inf")
    edge = fit_linear_gaussian_edge(list(zip(p.tolist(), c.tolist())))
    ll_reg = float(np.sum(edge.seq_log_density((p, c))))
    marginal = fit(list(child), _clone(child_estimator), max_its=max_its, out=None, rng=rng)
    ll_marginal = float(np.sum(marginal.seq_log_density(marginal.dist_to_encoder().seq_encode(list(child)))))
    pen = 0.5 * 1.0 * np.log(max(n, 2)) if penalty == "bic" else 0.0  # a single extra parameter: the slope
    return ll_reg - ll_marginal - pen


def _is_count(column: Sequence[Any]) -> bool:
    return len({v for v in column}) > 2 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) >= 0 and float(v) == int(v) for v in column
    )


def _is_binary(column: Sequence[Any]) -> bool:
    return _is_numeric(column) and {float(v) for v in column} <= {0.0, 1.0}


def _family_logpmf(name: str, y: Any, mu: np.ndarray, phi: float) -> np.ndarray:
    from scipy import stats

    y = np.asarray(y, dtype=float)
    if name == "poisson":
        return stats.poisson.logpmf(y, np.clip(mu, 1e-12, None))
    if name == "binomial":
        m = np.clip(mu, 1e-12, 1.0 - 1e-12)
        return y * np.log(m) + (1.0 - y) * np.log(1.0 - m)
    return stats.norm.logpdf(y, mu, np.sqrt(max(phi, 1e-12)))


class GLMEdge:
    """A generalized-linear regression edge: a *count* child's rate ``= exp(a + b*parent)`` (Poisson log-link), a
    *binary* child's probability ``= logit^-1(a + b*parent)`` (logistic) — the heterogeneous generalization of
    :class:`LinearGaussianEdge` (McCullagh & Nelder 1989). One slope parameter, fit by IRLS via
    :func:`mixle.inference.glm.glm`. Models a count/binary child driven by a continuous parent far better than a
    coarse per-bin conditional."""

    __pysp_serializable__ = (
        True  # opt in to mixle JSON serialization (custom state: the link fn is rebuilt, not stored)
    )

    def __init__(self, family: str, beta: Any, link: str, phi: float = 1.0) -> None:
        from mixle.inference.glm import _LINKS

        self.family, self.beta, self.link, self.phi = family, np.asarray(beta, dtype=float), link, float(phi)
        self._inv = _LINKS[link].inv

    def __pysp_getstate__(self) -> dict[str, Any]:
        # ``_inv`` is a live link function -- store the named parameters and rebuild it on decode instead.
        return {"family": self.family, "beta": self.beta, "link": self.link, "phi": self.phi}

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        from mixle.inference.glm import _LINKS

        self.family = state["family"]
        self.beta = np.asarray(state["beta"], dtype=float)
        self.link = state["link"]
        self.phi = float(state["phi"])
        self._inv = _LINKS[self.link].inv

    def _mu(self, parent: Any) -> np.ndarray:
        return self._inv(self.beta[0] + self.beta[1] * np.asarray(parent, dtype=float))

    def log_density(self, x: tuple) -> float:
        """Evaluate ``log p(child | parent)`` for one GLM edge pair."""
        parent, child = x
        return float(_family_logpmf(self.family, [float(child)], self._mu(np.array([float(parent)])), self.phi)[0])

    def dist_to_encoder(self) -> Any:
        """Return the encoder for parent-child pairs."""
        return _EdgeEncoder()

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        """Evaluate log densities for encoded GLM edge pairs."""
        parent, child = encoded
        return _family_logpmf(
            self.family, np.asarray(child, dtype=float), self._mu(np.asarray(parent, dtype=float)), self.phi
        )

    def sampler(self, seed: int | None = None) -> Any:
        """Return a sampler for the GLM edge."""
        return _GLMEdgeSampler(self, seed)

    def __str__(self) -> str:
        return f"GLMEdge({self.family}/{self.link}: child ~ {self.beta[0]:.3g} + {self.beta[1]:.3g}*parent)"


class _GLMEdgeSampler:
    def __init__(self, edge: GLMEdge, seed: int | None) -> None:
        self.edge = edge
        self.rng = np.random.RandomState(seed)

    def sample_given(self, parent: Any) -> Any:
        mu = float(self.edge._mu(np.array([float(parent)]))[0])
        if self.edge.family == "poisson":
            return int(self.rng.poisson(max(mu, 0.0)))
        if self.edge.family == "binomial":
            return int(self.rng.binomial(1, min(max(mu, 0.0), 1.0)))
        return float(self.rng.normal(mu, np.sqrt(self.edge.phi)))


def fit_glm_edge(pairs: Sequence[tuple], family: str) -> GLMEdge:
    """Fit a GLM conditional ``child ~ g^-1(a + b*parent)`` (family's canonical link) by IRLS."""
    from mixle.inference.glm import glm

    arr = np.asarray(pairs, dtype=float)
    p, c = arr[:, 0], arr[:, 1]
    result = glm(np.column_stack([np.ones_like(p), p]), c, family=family)
    return GLMEdge(result.family, result.coef, result.link, result.dispersion)


def glm_gain(
    parent: Sequence[Any],
    child: Sequence[Any],
    child_estimator: Any,
    family: str,
    *,
    max_its: int = 30,
    penalty: str = "bic",
    rng: np.random.RandomState | None = None,
) -> float:
    """Description-length gain (nats) of a GLM ``family`` regression edge over the child marginal (one extra slope
    parameter). Returns ``-inf`` when the fit is undefined or non-finite. ``rng`` seeds the marginal fit's EM
    initialization (``None`` = a fixed seed: deterministic by default)."""
    rng = np.random.RandomState(0) if rng is None else rng
    p = np.asarray(parent, dtype=float)
    c = np.asarray(child, dtype=float)
    n = len(c)
    if n < 3 or float(np.var(p)) < 1e-12:
        return float("-inf")
    try:
        edge = fit_glm_edge(list(zip(p.tolist(), c.tolist())), family)
        ll_glm = float(np.sum(edge.seq_log_density((p, c))))
    except Exception:  # noqa: BLE001
        return float("-inf")
    if not np.isfinite(ll_glm):
        return float("-inf")
    marginal = fit(list(child), _clone(child_estimator), max_its=max_its, out=None, rng=rng)
    ll_marginal = float(np.sum(marginal.seq_log_density(marginal.dist_to_encoder().seq_encode(list(child)))))
    pen = 0.5 * 1.0 * np.log(max(n, 2)) if penalty == "bic" else 0.0
    return ll_glm - ll_marginal - pen


def _numeric_edge_candidate(
    parent_col: Sequence[Any],
    child_col: Sequence[Any],
    template: Any,
    *,
    max_its: int = 30,
    rng: np.random.RandomState | None = None,
) -> tuple[float | None, str | None]:
    """The best regression/GLM edge of ``child`` on a numeric ``parent``: ``(gain, kind)`` or ``(None, None)``.
    Dispatches on the child's type — binary -> logistic, count -> Poisson, real -> linear-Gaussian."""
    if not _is_numeric(parent_col) or float(np.var(np.asarray(parent_col, dtype=float))) < 1e-12:
        return None, None
    if not _is_numeric(child_col):  # a categorical child stays a binned conditional
        return None, None
    if _is_binary(child_col):
        return glm_gain(parent_col, child_col, template, "binomial", max_its=max_its, rng=rng), "glm:binomial"
    if _is_count(child_col):
        return glm_gain(parent_col, child_col, template, "poisson", max_its=max_its, rng=rng), "glm:poisson"
    if _is_numeric(child_col):
        return regression_gain(parent_col, child_col, template, max_its=max_its, rng=rng), "regression"
    return None, None


def _safe_log_density(fac: Any, value: Any) -> float:
    """One factor's log-density with out-of-support semantics: zero probability -> ``-inf``, not a crash.

    Field factors are chosen automatically from the *training* column, so a support-restricted family
    (Weibull, Gamma, ...) can be picked from a slice that happened to satisfy its support; scoring new
    data outside that support must then report ``log 0 = -inf`` (the model's support semantics), not raise.
    """
    try:
        ld = float(fac.log_density(value))
    except (ValueError, TypeError, KeyError, FloatingPointError, OverflowError):
        return float("-inf")
    return ld if not np.isnan(ld) else float("-inf")


def _safe_seq_log_density(fac: Any, values: list[Any]) -> np.ndarray:
    """Vectorized factor scoring with the same out-of-support -> ``-inf`` semantics as
    :func:`_safe_log_density`; falls back to element-wise scoring only when the batch encode rejects."""
    try:
        out = np.asarray(fac.seq_log_density(fac.dist_to_encoder().seq_encode(values)), dtype=np.float64)
    except (ValueError, TypeError, KeyError, FloatingPointError, OverflowError):
        return np.array([_safe_log_density(fac, v) for v in values], dtype=np.float64)
    out[np.isnan(out)] = -np.inf
    return out


class DependencyTreeDistribution:
    """A directed-forest joint over a heterogeneous record: each field is a marginal or a conditional on its parent.

    ``log_density(record) = sum_root log P(f_root) + sum_child log P(f_child | f_parent)``. The dependence a
    :class:`~mixle.stats.combinator.composite.CompositeDistribution` assumes away is modeled here as
    per-parent-value conditionals -- while it still scores, samples, and composes like any mixle distribution.
    """

    __pysp_serializable__ = True  # parents/factors/binners/order round-trip via __dict__ (factors self-serialize)

    def __init__(
        self, parents: Sequence[int | None], factors: Sequence[Any], binners: Sequence[Any] | None = None
    ) -> None:
        # factors[i]: a marginal distribution if parents[i] is None, else a ConditionalDistribution over (key, i),
        # where the parent value is mapped to a conditioning key by binners[i] (identity for a discrete parent,
        # a quantile bin for a continuous one).
        self.parents = list(parents)
        self.factors = list(factors)
        self.binners = list(binners) if binners is not None else [None] * len(self.parents)
        self.order = _topo_order(self.parents)

    def _key(self, i: int, parent_value: Any) -> Any:
        binner = self.binners[i]
        return parent_value if binner is None else binner(parent_value)

    def __str__(self) -> str:
        edges = [f"{p}->{i}" for i, p in enumerate(self.parents) if p is not None]
        return f"DependencyTreeDistribution(fields={len(self.parents)}, edges=[{', '.join(edges) or 'none'}])"

    def log_density(self, x: tuple) -> float:
        """Evaluate the dependency-tree joint log density for one record."""
        total = 0.0
        for i, parent in enumerate(self.parents):
            if parent is None:
                total += _safe_log_density(self.factors[i], x[i])
            else:
                total += _safe_log_density(self.factors[i], (self._key(i, x[parent]), x[i]))
        return total

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        """Evaluate dependency-tree joint log density for encoded records."""
        cols, n = encoded
        out = np.zeros(n, dtype=np.float64)
        for i, parent in enumerate(self.parents):
            fac = self.factors[i]
            if parent is None:
                out += _safe_seq_log_density(fac, list(cols[i]))
            else:
                pairs = [(self._key(i, pv), cv) for pv, cv in zip(cols[parent], cols[i])]
                out += _safe_seq_log_density(fac, pairs)
        return out

    def dist_to_encoder(self) -> Any:
        """Return the encoder for dependency-tree record batches."""
        return _DependencyEncoder(len(self.parents))

    def sampler(self, seed: int | None = None) -> Any:
        """Return a sampler for the dependency tree."""
        return _DependencyTreeSampler(self, seed)

    def edges(self) -> list[tuple[int, int]]:
        """The learned dependency edges ``(parent_field, child_field)``."""
        return [(p, i) for i, p in enumerate(self.parents) if p is not None]


class _DependencyEncoder:
    def __init__(self, n_fields: int) -> None:
        self.n_fields = n_fields

    def seq_encode(self, data: Sequence[tuple]) -> tuple[list[list[Any]], int]:
        return _columns(list(data)), len(data)


class _DependencyTreeSampler:
    """Ancestral sampling: draw each root marginal, then each child from its parent-conditioned distribution."""

    def __init__(self, dist: DependencyTreeDistribution, seed: int | None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int = 1) -> list[tuple]:
        rows = []
        for _ in range(size):
            vals: list[Any] = [None] * len(self.dist.parents)
            for i in self.dist.order:
                parent = self.dist.parents[i]
                fac = self.dist.factors[i]
                seed = int(self.rng.randint(0, 2**31 - 1))
                if parent is None:
                    vals[i] = fac.sampler(seed).sample(1)[0]
                else:
                    vals[i] = fac.sampler(seed).sample_given(self.dist._key(i, vals[parent]))
            rows.append(tuple(vals))
        return rows


class MixtureOfDependencyTrees:
    """A latent mixture whose components each carry their *own* discovered dependency structure.

    ``log p(x) = logsumexp_k ( log w_k + log p_k(x) )`` where each ``p_k`` is a :class:`DependencyTreeDistribution`.
    This is the deep form of the tagline: it discovers both the clustering *and* the within-cluster cross-field
    dependence -- so the same category can map to different reals in different clusters, which neither a single
    dependency tree (one relationship) nor a mixture of independent composites (no within-cluster dependence) can
    represent. Fit by :func:`learn_mixture_structure`.
    """

    __pysp_serializable__ = True  # components/weights/log_weights round-trip via __dict__

    def __init__(self, components: Sequence[DependencyTreeDistribution], weights: Sequence[float]) -> None:
        self.components = list(components)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.log_weights = np.log(np.clip(self.weights, 1e-300, None))

    @property
    def w(self) -> np.ndarray:
        """Mixture-convention alias for ``weights`` -- lets mixture-generic tooling (e.g.
        ``mixle.utils.hvis.model_fit_health`` / ``hvis_map``) accept this model unchanged."""
        return self.weights

    @property
    def log_w(self) -> np.ndarray:
        """Mixture-convention alias for ``log_weights`` (see :attr:`w`)."""
        return self.log_weights

    def __str__(self) -> str:
        return f"MixtureOfDependencyTrees(k={len(self.components)}, weights={np.round(self.weights, 3).tolist()})"

    def _component_ll(self, encoded: Any) -> np.ndarray:
        return np.stack([c.seq_log_density(encoded) for c in self.components], axis=1)  # (n, K)

    def log_density(self, x: tuple) -> float:
        """Evaluate mixture log density for one dependency-tree record."""
        from scipy.special import logsumexp

        return float(logsumexp(self.log_weights + np.array([c.log_density(x) for c in self.components])))

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        """Evaluate mixture log density for encoded dependency-tree records."""
        from scipy.special import logsumexp

        return logsumexp(self._component_ll(encoded) + self.log_weights[None, :], axis=1)

    def dist_to_encoder(self) -> Any:
        """Return the record encoder shared by all tree components."""
        return _DependencyEncoder(len(self.components[0].parents))

    def responsibilities(self, data: Sequence[tuple]) -> np.ndarray:
        """Posterior ``p(component | record)`` for each record -- the E-step and a soft cluster assignment."""
        enc = self.dist_to_encoder().seq_encode(list(data))
        joint = self._component_ll(enc) + self.log_weights[None, :]
        joint -= joint.max(axis=1, keepdims=True)
        r = np.exp(joint)
        return r / r.sum(axis=1, keepdims=True)

    def sampler(self, seed: int | None = None) -> Any:
        """Return a sampler for the mixture of dependency trees."""
        return _MixtureTreeSampler(self, seed)

    @property
    def n_components(self) -> int:
        """Return the number of mixture components."""
        return len(self.components)


class _MixtureTreeSampler:
    def __init__(self, dist: MixtureOfDependencyTrees, seed: int | None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int = 1) -> list[tuple]:
        ks = self.rng.choice(self.dist.n_components, size=size, p=self.dist.weights)
        rows = []
        for k in ks:
            s = int(self.rng.randint(0, 2**31 - 1))
            rows.append(self.dist.components[int(k)].sampler(s).sample(1)[0])
        return rows


def learn_mixture_structure(
    data: Sequence[tuple],
    n_components: int,
    *,
    restarts: int = 3,
    max_iter: int = 15,
    seed: int = 0,
    min_gain: float = 0.0,
    n_bins: int = 4,
    max_its: int = 30,
    field_estimators: Sequence[Any] | None = None,
) -> MixtureOfDependencyTrees:
    """Fit a :class:`MixtureOfDependencyTrees` by hard EM: discover clusters and each cluster's dependency graph.

    Each iteration re-learns a dependency forest per cluster on its currently-assigned points (M-step), then
    reassigns every record to its most-probable cluster (E-step), until assignments stabilize. Runs ``restarts``
    random initializations and returns the highest-likelihood fit. Empty/tiny clusters are re-seeded so a
    component never collapses. Deterministic given ``seed``: the one ``RandomState`` drives the k-means/random
    initializations AND every per-cluster fit's EM init (via :func:`learn_structure`'s ``rng``).

    ``field_estimators`` pins each field's family (forwarded to :func:`learn_structure`) instead of re-running
    the automatic detector per cluster per iteration. Beyond the speedup, pinning matters for identifiability:
    the detector models a multimodal column with a Gaussian MIXTURE, which lets ONE cluster absorb what the
    caller intended as two -- with per-field families pinned to unimodal models, regimes that differ in level
    must separate into different components to score well.
    random initializations and returns the highest-likelihood fit. Empty or very small clusters are re-seeded so a
    component never collapses.
    """
    data = list(data)
    n = len(data)
    rng = np.random.RandomState(seed)
    min_size = max(10, n // (4 * n_components))
    best: MixtureOfDependencyTrees | None = None
    best_ll = -np.inf

    def learn(subset: list[tuple]) -> DependencyTreeDistribution:
        return learn_structure(
            subset, field_estimators=field_estimators, min_gain=min_gain, n_bins=n_bins, max_its=max_its, rng=rng
        )

    # seed the first restarts with k-means (numeric-level split, then full-feature split), rest random
    inits = [
        _kmeans_init(data, n_components, rng, numeric_only=True),
        _kmeans_init(data, n_components, rng, numeric_only=False),
    ]
    inits += [rng.randint(0, n_components, n) for _ in range(max(0, restarts - len(inits)))]
    for assign in inits:
        model: MixtureOfDependencyTrees | None = None
        prev = None
        for _it in range(max_iter):
            comps, counts = [], []
            for k in range(n_components):
                idx = np.flatnonzero(assign == k)
                if len(idx) < min_size:  # re-seed a starved component from random points
                    idx = rng.choice(n, size=min_size, replace=False)
                comps.append(learn([data[i] for i in idx]))
                counts.append(len(idx))
            weights = np.asarray(counts, dtype=np.float64)
            weights /= weights.sum()
            model = MixtureOfDependencyTrees(comps, weights)
            new = model.responsibilities(data).argmax(axis=1)
            if prev is not None and np.array_equal(new, prev):
                break
            prev, assign = assign, new
        assert model is not None
        ll = float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))
        if ll > best_ll:
            best_ll, best = ll, model
    assert best is not None
    return best


def _split_separation(values: np.ndarray) -> tuple[float, float]:
    """Deterministic 1-D 2-means ``(separation, minority share)`` -- the same construction (sign
    split + Lloyd, gap over pooled within-std) and calibration as the merged-regime detector in
    ``mixle.utils.hvis.topology.model_fit_health``, so the ``2.65 + 6/sqrt(n)`` threshold carries
    over. Returns ``(0, 0)`` when 2-means degenerates to one cluster."""
    proj = np.asarray(values, dtype=np.float64)
    proj = proj - proj.mean()
    assign = proj > 0.0
    for _ in range(15):
        if assign.all() or (~assign).all():
            return 0.0, 0.0
        c1, c0 = float(proj[assign].mean()), float(proj[~assign].mean())
        new_assign = np.abs(proj - c1) < np.abs(proj - c0)
        if bool(np.all(new_assign == assign)):
            break
        assign = new_assign
    if not 0 < int(assign.sum()) < len(proj):
        return 0.0, 0.0
    minority = min(float(assign.mean()), float(1.0 - assign.mean()))
    within_var = float(
        np.average([proj[assign].var(), proj[~assign].var()], weights=[assign.mean(), 1 - assign.mean()])
    )
    sep = abs(float(proj[assign].mean() - proj[~assign].mean())) / max(np.sqrt(within_var), 1.0e-12)
    return sep, minority


def mixture_structure_health(
    mot: MixtureOfDependencyTrees, data: Sequence[tuple], *, merged_sep_threshold: float | None = None
) -> dict:
    """The identifiability receipt for a fitted :class:`MixtureOfDependencyTrees`.

    The trap it names: a flexible per-field family (a Gaussian-mixture conditional, a heavy-tailed
    catch-all marginal) lets ONE component absorb what the caller intended as SEVERAL regimes --
    and because the absorbed fit scores well, likelihood-level receipts look healthy. Measured
    concretely: on two planted regimes, a one-component fit matches the two-component fit's
    likelihood, so nothing downstream of the density can see the difference.

    The check that can: STRUCTURE-CONDITIONAL multimodality. For every continuous field of every
    component, group that component's dominated points by the field's own learned conditioning
    (parent level for a binned conditional, regression residuals for a linear edge, the whole
    fiber for a root marginal) -- after the tree has explained what it can, each group must be
    unimodal. A group that still splits (deterministic 2-means separation over the same
    ``2.65 + 6/sqrt(n)`` finite-sample threshold the hvis merged-regime detector is calibrated
    to, minority share >= 20%) is a regime split the component is hiding, whichever family
    absorbed it.

    Returns ``{"components": [...], "diagnosis": [str, ...]}`` -- empty ``diagnosis`` means no
    component hides multimodal structure. Per component, ``multimodal_fields`` lists
    ``(field, where, separation)`` and ``mixture_factors`` lists factors the detector fitted as
    mixture families (supporting detail: where the absorbed structure went). The fix when it
    fires is more components or pinned unimodal ``field_estimators`` (see
    :func:`learn_mixture_structure`). Complementary to ``mixle.utils.hvis.model_fit_health``,
    which audits density calibration and accepts this model directly.
    """
    from mixle.stats import MixtureDistribution
    from mixle.stats.combinator.conditional import ConditionalDistribution

    data = list(data)
    cols = _columns(data)
    dominant = mot.responsibilities(data).argmax(axis=1)
    components, diagnosis = [], []
    for k, tree in enumerate(mot.components):
        rows_k = np.flatnonzero(dominant == k)
        mixture_factors: list[tuple[int, str]] = []
        multimodal: list[tuple[int, str, float]] = []
        for j, factor in enumerate(tree.factors):
            if isinstance(factor, MixtureDistribution):
                mixture_factors.append((j, "marginal"))
            elif isinstance(factor, ConditionalDistribution):
                mixture_factors.extend(
                    (j, f"conditional level {lv!r}")
                    for lv, child in factor.dmap.items()
                    if isinstance(child, MixtureDistribution)
                )

            if not _is_numeric(cols[j]):
                continue
            vals = np.asarray([cols[j][i] for i in rows_k], dtype=np.float64)
            parent = tree.parents[j]
            if parent is None:
                groups = [("marginal", vals)]
            elif isinstance(factor, LinearGaussianEdge):
                pv = np.asarray([cols[parent][i] for i in rows_k], dtype=np.float64)
                groups = [("regression residuals", vals - (factor.a + factor.b * pv))]
            elif isinstance(factor, ConditionalDistribution):
                binner = tree.binners[j]
                keys = [cols[parent][i] if binner is None else binner(cols[parent][i]) for i in rows_k]
                by_key: dict = {}
                for key, v in zip(keys, vals):
                    by_key.setdefault(key, []).append(float(v))
                groups = [
                    (f"level {key!r}", np.asarray(g)) for key, g in sorted(by_key.items(), key=lambda t: repr(t[0]))
                ]
            else:  # GLM edges have discrete children; nothing continuous to test
                continue
            for where, g in groups:
                if len(g) < 20:
                    continue
                sep, minority = _split_separation(g)
                threshold = (
                    merged_sep_threshold if merged_sep_threshold is not None else 2.65 + 6.0 / np.sqrt(float(len(g)))
                )
                if sep > threshold and minority >= 0.2:
                    multimodal.append((j, where, sep))
                    diagnosis.append(
                        f"component {k} field {j} ({where}): still splits after conditioning (2-means "
                        f"separation {sep:.1f}, minority {minority:.0%}) -- this component is absorbing "
                        "multiple regimes; consider more components or pinned field_estimators."
                    )
        components.append({"mixture_factors": mixture_factors, "multimodal_fields": multimodal})
    return {"components": components, "diagnosis": diagnosis}


# --- structure search + fitting ------------------------------------------------------------------------------


def learn_structure(
    data: Sequence[tuple],
    *,
    field_estimators: Sequence[Any] | None = None,
    min_gain: float = 0.0,
    max_levels: int = 64,
    n_bins: int = 4,
    max_its: int = 30,
    rng: np.random.RandomState | None = None,
) -> DependencyTreeDistribution:
    """Discover the dependency forest for heterogeneous ``data`` and return the fitted joint model.

    Any field can be a parent: a discrete one conditions directly, a continuous one is quantile-binned into
    ``n_bins`` conditioning levels (so a real can drive a count, a category, or another real). Scores every
    ``(parent -> child)`` pair by :func:`dependency_gain`, greedily builds a maximum-gain acyclic forest (each
    field at most one parent), and fits each factor. Falls back to independent marginals where no dependence
    clears ``min_gain`` -- never worse than a composite, much better when structure exists. This is "automatic
    inference for composable models of heterogeneous data" made real.

    ``rng`` seeds every internal fit's EM initialization; ``None`` resolves to a FIXED seed, so two calls on
    the same data return the same model. (Before this knob, fits whose detected family needs a randomized
    init -- a Gaussian-mixture conditional, say -- drew fresh OS entropy per call, so the learned model
    itself was nondeterministic.)
    """
    rng = np.random.RandomState(0) if rng is None else rng
    data = list(data)
    cols = _columns(data)
    n_fields = len(cols)
    templates = list(field_estimators) if field_estimators is not None else [_field_estimator(c) for c in cols]
    discrete = [_is_discrete(c, max_levels=max_levels) for c in cols]

    # a conditioning key per candidate parent: identity for a discrete field, a quantile bin for a continuous one
    binners = [None if discrete[p] else _quantile_binner(cols[p], n_bins) for p in range(n_fields)]
    keyed = [cols[p] if binners[p] is None else [binners[p](v) for v in cols[p]] for p in range(n_fields)]

    # a numeric parent + a real/count/binary child can use a linear-Gaussian or GLM REGRESSION edge (1 slope
    # param) instead of a coarse per-bin conditional; each edge takes whichever scores the higher DL gain.
    candidates: list[tuple[float, int, int, str]] = []
    for p in range(n_fields):
        for c in range(n_fields):
            if c == p:
                continue
            gain = dependency_gain(keyed[p], cols[c], templates[c], max_its=max_its, rng=rng)
            kind = "binned"
            ngain, nkind = _numeric_edge_candidate(cols[p], cols[c], templates[c], max_its=max_its, rng=rng)
            if ngain is not None and ngain > gain:
                gain, kind = ngain, nkind
            if gain > min_gain:
                candidates.append((gain, p, c, kind))
    candidates.sort(key=lambda t: t[0], reverse=True)

    # greedy maximum-gain forest: each child at most one parent, keep it acyclic (union-find on undirected links)
    parents: list[int | None] = [None] * n_fields
    edge_kind: list[str] = ["binned"] * n_fields
    uf = _UnionFind(n_fields)
    for _gain, p, c, kind in candidates:
        if parents[c] is not None or uf.connected(p, c):
            continue
        parents[c] = p
        edge_kind[c] = kind
        uf.union(p, c)

    # fit each factor: roots as marginals, children as per-parent-key conditionals (or regression edges)
    factors: list[Any] = [None] * n_fields
    edge_binners: list[Any] = [None] * n_fields
    for i in range(n_fields):
        if parents[i] is None:
            factors[i] = fit(cols[i], _clone(templates[i]), max_its=max_its, out=None, rng=rng)
        elif edge_kind[i] == "regression":
            factors[i] = fit_linear_gaussian_edge(list(zip(cols[parents[i]], cols[i])))
            edge_binners[i] = None  # the raw parent value drives the regression
        elif edge_kind[i].startswith("glm:"):
            factors[i] = fit_glm_edge(list(zip(cols[parents[i]], cols[i])), edge_kind[i].split(":", 1)[1])
            edge_binners[i] = None  # the raw parent value drives the GLM
        else:
            p = parents[i]
            keys = keyed[p]
            est = ConditionalDistributionEstimator(
                estimator_map={lv: _clone(templates[i]) for lv in sorted(set(keys))}, given_estimator=None
            )
            factors[i] = fit(list(zip(keys, cols[i])), est, max_its=max_its, out=None, rng=rng)
            edge_binners[i] = binners[p]
    return DependencyTreeDistribution(parents, factors, edge_binners)


def _init_matrix(data: list[tuple], *, numeric_only: bool) -> np.ndarray:
    """Featurize records for the init k-means: standardized numerics, plus one-hot categoricals unless ``numeric_only``.

    ``numeric_only`` clusters on the continuous fields alone -- which carry the regime's *level* -- and avoids
    k-means latching onto an observed categorical (clustering by a data field, not the latent). The full-feature
    variant is tried as a separate restart for cases where a categorical marginal defines the clusters.
    """
    cols = _columns(data)
    numerics, onehots = [], []
    for c in cols:
        if _is_discrete(c):
            levels = sorted(set(c))
            onehots.append(np.array([[1.0 if v == lv else 0.0 for lv in levels] for v in c]))
        else:
            arr = np.asarray(c, dtype=np.float64)
            sd = arr.std() or 1.0
            numerics.append(((arr - arr.mean()) / sd)[:, None])
    blocks = numerics if (numeric_only and numerics) else (numerics + onehots)
    return np.hstack(blocks) if blocks else np.zeros((len(data), 1))


def _kmeans_init(data: list[tuple], k: int, rng: np.random.RandomState, *, numeric_only: bool) -> np.ndarray:
    """k-means (Lloyd, few iters) on init features -> an initial hard cluster assignment."""
    x = _init_matrix(data, numeric_only=numeric_only)
    n = x.shape[0]
    centers = x[rng.choice(n, size=k, replace=False)]
    assign = np.zeros(n, dtype=int)
    for _ in range(10):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(axis=0)
    return assign


class _QuantileBinner:
    """Map a continuous value to a bin label ``bK`` by fixed quantile edges -- a continuous field as a parent."""

    __pysp_serializable__ = True  # edges (a list of floats) round-trip via __dict__

    def __init__(self, edges: Sequence[float]) -> None:
        self.edges = list(edges)

    def __call__(self, value: Any) -> str:
        return f"b{int(np.searchsorted(self.edges, float(value)))}"


def _quantile_binner(column: Sequence[Any], n_bins: int) -> _QuantileBinner:
    arr = np.asarray(column, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(arr, qs)) if len(arr) else np.array([0.0])
    return _QuantileBinner(edges)


# --- small helpers -------------------------------------------------------------------------------------------


def _clone(estimator: Any) -> Any:
    """A fresh, independent copy of an estimator template so structure-search candidates never share state.

    The older ``eval(str(estimator))`` path failed for estimators with the default ``<object at 0x...>`` repr
    and fell back to sharing the same object -- safe only because estimators are stateless
    templates. ``deepcopy`` gives real isolation with no source-level eval; the fallback keeps the old
    same-object behavior for any estimator that can't be copied (e.g. one holding an uncopyable handle)."""
    try:
        return copy.deepcopy(estimator)
    except Exception:  # noqa: BLE001 - some estimators hold uncopyable state; sharing was the prior behavior
        return estimator


def _num_free_params(dist: Any) -> int:
    """An approximate parameter count for the BIC penalty (used only to scale the complexity term).

    Composes over :class:`~mixle.stats.combinator.composite.CompositeDistribution` (sums each field's
    own count) rather than falling through to a flat constant -- a composite of any size used to score
    the same as a single scalar leaf, which made the network-vs-composite BIC comparison in
    :func:`mixle.inference.estimation._maybe_structured_model` nearly meaningless for multi-field data.

    Counts a fitted categorical-family leaf (``pmap``/``p_vec``) as ``K - 1`` (the true simplex free
    parameter count), not the flat constant every other attribute check fell through to -- undercounting
    complexity there let :func:`mixle.inference.bayesian_network.learn_bayesian_network`'s greedy search
    accept spurious edges between fields with no real dependence, since the BIC penalty for the extra
    per-parent-config categorical table barely grew with the number of categories.

    The single-scalar-parameter families below (Poisson rate, Bernoulli/Binomial success probability)
    count 1, not 2 -- the old code doubled every matched attribute uniformly, correct only for a
    location+scale pair (Gaussian's mean+variance) and an overcount for these. NegativeBinomialDistribution
    is a deliberate exception: it has both ``r`` and ``p`` attributes, and ``NegativeBinomialEstimator``
    fits both by default (``estimate_r=True``), so it counts 2 even though ``p`` alone would otherwise
    match the single-scalar bucket below. Other leaf families (Weibull, Gumbel, Beta, ...) use the
    conservative flat-2 fallback unless they expose a more specific parameter-count hook.
    """
    name = type(dist).__name__
    if name == "CompositeDistribution":
        return sum(_num_free_params(d) for d in dist.dists)
    if hasattr(dist, "pmap"):  # CategoricalDistribution: a K-outcome simplex has K-1 free params
        return max(1, len(dist.pmap) - 1)
    if hasattr(dist, "p_vec"):  # IntegerCategoricalDistribution: same simplex parameterization
        return max(1, int(np.asarray(dist.p_vec).size) - 1)
    if hasattr(dist, "r") and hasattr(dist, "p"):  # NegativeBinomialDistribution: r and p both estimated
        return 2
    if hasattr(dist, "mu"):  # location+scale family (Gaussian, ...): mean + variance
        try:
            return max(1, int(np.asarray(dist.mu).size) * 2)
        except (TypeError, ValueError):
            return 2
    for attr in ("lam", "p"):  # single free scalar: Poisson rate, Bernoulli/Binomial/NegBinom success prob
        if hasattr(dist, attr):
            return 1
    return 2


def _topo_order(parents: Sequence[int | None]) -> list[int]:
    order, seen = [], set()

    def visit(i: int) -> None:
        if i in seen:
            return
        if parents[i] is not None:
            visit(parents[i])
        seen.add(i)
        order.append(i)

    for i in range(len(parents)):
        visit(i)
    return order


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)
