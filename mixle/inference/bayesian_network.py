"""Heterogeneous Bayesian network learning -- a directed graph over mixed-type fields with parametric edges.

This deepens :mod:`mixle.inference.structure` (a single-parent forest with quantile-binned continuous parents)
into the real thing: a **DAG** where a field may have *several* parents, and continuous dependence is a
*parametric* conditional, not a binning. A continuous child is a conditional-linear-Gaussian node -- ``child ~
N(w . [continuous parents, one-hot(discrete parents)] + b, sigma^2)`` -- so a real driven by two reals, or by a
category and a real, is modeled exactly and cheaply (closed-form least squares). A discrete/count child with
all-discrete parents conditions on their joint configuration (marginal backoff for unseen configs); with a
continuous driver it becomes a GLM node (logistic / Poisson log-link / multinomial softmax), so category<->real
dependence is representable in BOTH orientations and BIC picks the cheaper one.

``learn_bayesian_network`` grows each node's parent set greedily by description-length gain, up to ``max_parents``,
keeping the graph acyclic. The result scores, samples, and composes like any mixle distribution -- the moat:
automatic discovery *and* fitting of a heterogeneous graphical model across arbitrary families, which no
mainstream tool does (Stan/PyMC: you write it; sklearn/pomegranate: independence; bnlearn/pgmpy: discrete-or-Gaussian).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.inference.estimation import fit
from mixle.inference.structure import _clone, _columns, _field_estimator, _is_discrete, _num_free_params

_LOG_2PI = float(np.log(2.0 * np.pi))


# --- factors: each models P(field_i | its parents) with a uniform score/sample interface --------------------


class _MarginalFactor:
    """A root field: ``P(x_i)`` under a fitted marginal distribution."""

    __pysp_serializable__ = True

    def __init__(self, child: int, dist: Any) -> None:
        self.child = child
        self.parents: list[int] = []
        self.dist = dist

    def seq_log_density(self, cols: list[list[Any]]) -> np.ndarray:
        enc = self.dist.dist_to_encoder().seq_encode(cols[self.child])
        return np.asarray(self.dist.seq_log_density(enc), dtype=np.float64)

    def log_density(self, x: tuple) -> float:
        return float(self.dist.log_density(x[self.child]))

    def sample(self, x: list, rng: np.random.RandomState) -> Any:
        return self.dist.sampler(int(rng.randint(0, 2**31 - 1))).sample(1)[0]

    def n_params(self) -> int:
        return _num_free_params(self.dist)

    def describe(self, field_names: Sequence[str] | None = None) -> dict[str, Any]:
        """A real summary of the fitted root marginal: its family/parameters via ``str()``, not a placeholder."""
        return {"field": _name(self.child, field_names), "kind": "marginal", "parents": [], "fitted": str(self.dist)}


def _name(i: int, field_names: Sequence[str] | None) -> str:
    return str(field_names[i]) if field_names is not None else str(i)


def _design_row(parents: list[int], values: Sequence[Any], discrete: dict, vec_dims: dict) -> np.ndarray:
    """A design row from a record's parent values: vector parents contribute all components, discrete
    parents one-hot (drop-first), continuous parents their raw value, then a trailing intercept."""
    feats: list[float] = []
    for p, v in zip(parents, values):
        if p in vec_dims:
            feats.extend(float(c) for c in np.asarray(v, dtype=np.float64).ravel())
        elif p in discrete:
            feats.extend(1.0 if v == lv else 0.0 for lv in discrete[p][1:])  # drop-first
        else:
            feats.append(float(v))
    feats.append(1.0)
    return np.asarray(feats, dtype=np.float64)


def _design_feature_names(
    parents: list[int], discrete: dict[int, list[Any]], vec_dims: dict[int, int], field_names: Sequence[str] | None
) -> list[str]:
    """Feature-column names for a fitted ``coef``/``weights`` vector, in the exact order :func:`_design_row`
    builds it -- so a :meth:`describe` report can label each coefficient by the parent (and, for a
    multi-level categorical or vector parent, the specific level/component) it belongs to, not just a bare
    index. Mirrors ``_design_row``'s three branches one-for-one, plus the trailing intercept term."""
    names: list[str] = []
    for p in parents:
        pname = _name(p, field_names)
        if p in vec_dims:
            names.extend(f"{pname}[{k}]" for k in range(vec_dims[p]))
        elif p in discrete:
            names.extend(f"{pname}={lv!r}" for lv in discrete[p][1:])  # drop-first, matches _design_row
        else:
            names.append(pname)
    names.append("intercept")
    return names


class _VectorMarginalFactor:
    """A vector-valued field's marginal: a multivariate Gaussian (closed-form mean + covariance)."""

    __pysp_serializable__ = True

    def __init__(self, child: int, mean: np.ndarray, cov: np.ndarray) -> None:
        self.child = child
        self.parents: list[int] = []
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self._prec = np.linalg.inv(self.cov)
        self._logdet = float(np.linalg.slogdet(self.cov)[1])
        self._chol = np.linalg.cholesky(self.cov)

    def _mvn_ll(self, y: np.ndarray) -> np.ndarray:
        d = self.mean.shape[0]
        delta = y - self.mean
        quad = np.einsum("ni,ij,nj->n", delta, self._prec, delta)
        return -0.5 * (d * _LOG_2PI + self._logdet + quad)

    def seq_log_density(self, cols: list[list[Any]]) -> np.ndarray:
        return self._mvn_ll(np.asarray(cols[self.child], dtype=np.float64))

    def log_density(self, x: tuple) -> float:
        return float(self._mvn_ll(np.asarray(x[self.child], dtype=np.float64)[None, :])[0])

    def sample(self, x: list, rng: np.random.RandomState) -> np.ndarray:
        return self.mean + self._chol @ rng.randn(self.mean.shape[0])

    def describe(self, field_names: Sequence[str] | None = None) -> dict[str, Any]:
        """The fitted multivariate-Gaussian marginal: mean vector and the covariance diagonal (variance
        per component) -- enough to see each component's scale without dumping the full covariance matrix."""
        return {
            "field": _name(self.child, field_names),
            "kind": "vector-marginal",
            "parents": [],
            "dim": int(self.mean.shape[0]),
            "mean": [round(float(v), 6) for v in self.mean],
            "variance": [round(float(v), 6) for v in np.diag(self.cov)],
        }

    @classmethod
    def fit(cls, child: int, cols: list[list[Any]], weights: np.ndarray | None = None):
        y = np.asarray(cols[child], dtype=np.float64)
        if weights is None:
            mean = y.mean(axis=0)
            cov = np.cov(y, rowvar=False)
        else:
            w = np.asarray(weights, dtype=np.float64)
            sw = w.sum()
            mean = (w[:, None] * y).sum(axis=0) / max(sw, 1e-12)
            delta = y - mean
            cov = (w[:, None, None] * np.einsum("ni,nj->nij", delta, delta)).sum(axis=0) / max(sw, 1e-12)
        cov = np.atleast_2d(cov) + 1e-6 * np.eye(y.shape[1])
        return cls(child, mean, cov)

    def n_params(self) -> int:
        d = self.mean.shape[0]
        return d + d * (d + 1) // 2


class _VectorCLGFactor:
    """A vector child as a multivariate linear-Gaussian of its parents: ``Y ~ N(W x + b, Sigma)``.

    Closed-form multivariate least squares for ``[W|b]`` and a residual covariance for ``Sigma`` --
    the vector-valued generalization of :class:`_LinearGaussianFactor`, so an embedding field can be
    driven by (or drive) any other field in the graph."""

    __pysp_serializable__ = True

    def __init__(self, child, parents, discrete, vec_dims, coef, cov) -> None:
        self.child = child
        self.parents = list(parents)
        self.discrete = discrete
        self.vec_dims = vec_dims
        self.coef = np.asarray(coef, dtype=np.float64)  # (p_feats, d)
        self.cov = np.asarray(cov, dtype=np.float64)  # (d, d)
        self._prec = np.linalg.inv(self.cov)
        self._logdet = float(np.linalg.slogdet(self.cov)[1])
        self._chol = np.linalg.cholesky(self.cov)

    def _design(self, cols: list[list[Any]]) -> np.ndarray:
        n = len(cols[self.child])
        return np.stack(
            [
                _design_row(self.parents, [cols[p][j] for p in self.parents], self.discrete, self.vec_dims)
                for j in range(n)
            ]
        )

    def _mvn_ll(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        d = y.shape[1]
        delta = y - mu
        quad = np.einsum("ni,ij,nj->n", delta, self._prec, delta)
        return -0.5 * (d * _LOG_2PI + self._logdet + quad)

    def seq_log_density(self, cols: list[list[Any]]) -> np.ndarray:
        y = np.asarray(cols[self.child], dtype=np.float64)
        return self._mvn_ll(y, self._design(cols) @ self.coef)

    def log_density(self, x: tuple) -> float:
        row = _design_row(self.parents, [x[p] for p in self.parents], self.discrete, self.vec_dims)[None, :]
        y = np.asarray(x[self.child], dtype=np.float64)[None, :]
        return float(self._mvn_ll(y, row @ self.coef)[0])

    def sample(self, x: list, rng: np.random.RandomState) -> np.ndarray:
        row = _design_row(self.parents, [x[p] for p in self.parents], self.discrete, self.vec_dims)
        return row @ self.coef + self._chol @ rng.randn(self.coef.shape[1])

    @classmethod
    def fit(cls, child, parents, cols, discrete, vec_dims, weights=None):
        proto = cls(child, parents, discrete, vec_dims, np.zeros((1, 1)), np.eye(1))
        x = proto._design(cols)
        y = np.asarray(cols[child], dtype=np.float64)
        if weights is None:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ coef
            cov = np.cov(resid, rowvar=False)
        else:
            w = np.asarray(weights, dtype=np.float64)
            sw = np.sqrt(np.maximum(w, 0.0))
            coef, *_ = np.linalg.lstsq(x * sw[:, None], y * sw[:, None], rcond=None)
            resid = y - x @ coef
            cov = (w[:, None, None] * np.einsum("ni,nj->nij", resid, resid)).sum(axis=0) / max(w.sum(), 1e-12)
        cov = np.atleast_2d(cov) + 1e-6 * np.eye(y.shape[1])
        return cls(child, parents, discrete, vec_dims, coef, cov)

    def n_params(self) -> int:
        d = self.coef.shape[1]
        return int(self.coef.size) + d * (d + 1) // 2

    def describe(self, field_names: Sequence[str] | None = None) -> dict[str, Any]:
        """The fitted multivariate linear-Gaussian edge: ``coef`` labeled by feature name (rows) and output
        component index (columns), plus the residual variance per output component (the diagonal of ``cov``)."""
        feat_names = _design_feature_names(self.parents, self.discrete, self.vec_dims, field_names)
        return {
            "field": _name(self.child, field_names),
            "kind": "vector-linear-gaussian",
            "parents": [_name(p, field_names) for p in self.parents],
            "output_dim": int(self.coef.shape[1]),
            "coefficients": {n: [round(float(v), 6) for v in row] for n, row in zip(feat_names, self.coef)},
            "residual_variance": [round(float(v), 6) for v in np.diag(self.cov)],
        }


class _LinearGaussianFactor:
    """A continuous child as a linear-Gaussian of its parents (continuous raw + one-hot discrete): the CLG node."""

    __pysp_serializable__ = True

    def __init__(
        self,
        child: int,
        parents: list[int],
        discrete: dict[int, list[Any]],
        coef: np.ndarray,
        sigma: float,
        vec_dims: dict | None = None,
    ):
        self.child = child
        self.parents = list(parents)
        self.discrete = discrete  # parent idx -> its sorted levels (one-hot, drop-first); absent => continuous
        self.vec_dims = vec_dims or {}  # parent idx -> dim, for vector-valued parents
        self.coef = coef  # (d+1,) : weights then intercept
        self.sigma = float(sigma)

    def _row(self, values: Sequence[Any]) -> np.ndarray:
        return _design_row(self.parents, values, self.discrete, self.vec_dims)

    def _design(self, cols: list[list[Any]]) -> np.ndarray:
        return np.stack([self._row([cols[p][j] for p in self.parents]) for j in range(len(cols[self.child]))])

    def seq_log_density(self, cols: list[list[Any]]) -> np.ndarray:
        y = np.asarray(cols[self.child], dtype=np.float64)
        mu = self._design(cols) @ self.coef
        return -0.5 * _LOG_2PI - np.log(self.sigma) - 0.5 * ((y - mu) / self.sigma) ** 2

    def log_density(self, x: tuple) -> float:
        mu = float(self._row([x[p] for p in self.parents]) @ self.coef)
        return -0.5 * _LOG_2PI - np.log(self.sigma) - 0.5 * ((float(x[self.child]) - mu) / self.sigma) ** 2

    def sample(self, x: list, rng: np.random.RandomState) -> float:
        mu = float(self._row([x[p] for p in self.parents]) @ self.coef)
        return float(mu + self.sigma * rng.randn())

    @classmethod
    def fit(
        cls,
        child: int,
        parents: list[int],
        cols: list[list[Any]],
        discrete: dict[int, list[Any]],
        weights: np.ndarray | None = None,
        vec_dims: dict | None = None,
    ):
        proto = cls(child, parents, discrete, np.zeros(1), 1.0, vec_dims=vec_dims)
        x = proto._design(cols)
        y = np.asarray(cols[child], dtype=np.float64)
        if weights is None:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ coef
            sigma = float(np.sqrt(max(resid.var(), 1e-6)))
        else:
            w = np.asarray(weights, dtype=np.float64)
            sw = np.sqrt(np.maximum(w, 0.0))
            coef, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
            resid = y - x @ coef
            var = float(np.sum(w * resid**2) / max(np.sum(w), 1e-12))
            sigma = float(np.sqrt(max(var, 1e-6)))
        return cls(child, parents, discrete, coef, sigma, vec_dims=vec_dims)

    def n_params(self) -> int:
        return self.coef.shape[0] + 1

    def describe(self, field_names: Sequence[str] | None = None) -> dict[str, Any]:
        """The fitted linear-Gaussian edge: every regression coefficient labeled by the exact feature it
        weights (a continuous parent's raw value, or ``parent=level`` for a one-hot discrete level), plus
        the intercept and residual noise scale -- e.g. ``{"hours.per.week": 2.3, "intercept": 40.1}``
        means each extra weekly hour raises the fitted mean by 2.3, holding the other parents fixed."""
        feat_names = _design_feature_names(self.parents, self.discrete, self.vec_dims, field_names)
        return {
            "field": _name(self.child, field_names),
            "kind": "linear-gaussian",
            "parents": [_name(p, field_names) for p in self.parents],
            "coefficients": {n: round(float(c), 6) for n, c in zip(feat_names, self.coef)},
            "sigma": round(float(self.sigma), 6),
        }


class _GLMFactor:
    """A discrete child with at least one CONTINUOUS parent — the edge the greedy search used to refuse.

    The child's kind picks the family: exactly two levels -> Bernoulli logistic; nonnegative integer
    counts -> Poisson log-link; K>2 categorical -> multinomial logistic (softmax, class 0 reference)
    with a small ridge so perfectly separable data keeps a finite, deterministic optimum. The design
    matrix mirrors the CLG node: continuous parents raw + one-hot(drop-first) discrete parents + 1.
    """

    __pysp_serializable__ = True

    def __init__(
        self,
        child: int,
        parents: list[int],
        discrete: dict[int, list[Any]],
        kind: str,
        levels: list[Any],
        weights: np.ndarray,
        vec_dims: dict | None = None,
    ) -> None:
        self.child = child
        self.parents = list(parents)
        self.discrete = discrete
        self.vec_dims = vec_dims or {}  # parent idx -> dim, for vector-valued parents
        self.kind = kind  # 'binomial' | 'poisson' | 'multinomial'
        self.levels = list(levels)  # child levels (binomial/multinomial); [] for poisson
        self.weights = np.asarray(weights, dtype=np.float64)  # (d,) or (K-1, d)

    _row = _LinearGaussianFactor._row
    _design = _LinearGaussianFactor._design

    def _log_pmf_rows(self, x_mat: np.ndarray, values: Sequence[Any]) -> np.ndarray:
        out = np.full(len(values), -np.inf, dtype=np.float64)
        if self.kind == "poisson":
            z = np.clip(x_mat @ self.weights, -700.0, 700.0)
            for j, v in enumerate(values):
                if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and int(v) >= 0:
                    from scipy.special import gammaln as _gammaln

                    out[j] = float(v) * z[j] - np.exp(z[j]) - float(_gammaln(float(v) + 1.0))
            return out
        if self.kind == "binomial":
            z = x_mat @ self.weights
            lp1 = -np.logaddexp(0.0, -z)  # log sigmoid(z)
            lp0 = -np.logaddexp(0.0, z)
            for j, v in enumerate(values):
                if v == self.levels[1]:
                    out[j] = lp1[j]
                elif v == self.levels[0]:
                    out[j] = lp0[j]
            return out
        logits = np.concatenate([np.zeros((len(values), 1)), x_mat @ self.weights.T], axis=1)
        logp = logits - _logsumexp_rows(logits)[:, None]
        index = {lv: i for i, lv in enumerate(self.levels)}
        for j, v in enumerate(values):
            i = index.get(v)
            if i is not None:
                out[j] = logp[j, i]
        return out

    def seq_log_density(self, cols: list[list[Any]]) -> np.ndarray:
        return self._log_pmf_rows(self._design(cols), cols[self.child])

    def log_density(self, x: tuple) -> float:
        row = self._row([x[p] for p in self.parents])[None, :]
        return float(self._log_pmf_rows(row, [x[self.child]])[0])

    def sample(self, x: list, rng: np.random.RandomState) -> Any:
        row = self._row([x[p] for p in self.parents])
        if self.kind == "poisson":
            return int(rng.poisson(float(np.exp(np.clip(row @ self.weights, -700.0, 700.0)))))
        if self.kind == "binomial":
            p1 = 1.0 / (1.0 + np.exp(-float(row @ self.weights)))
            return self.levels[1] if rng.rand() < p1 else self.levels[0]
        logits = np.concatenate([[0.0], self.weights @ row])
        p = np.exp(logits - np.max(logits))
        return self.levels[int(rng.choice(len(self.levels), p=p / p.sum()))]

    @classmethod
    def fit(
        cls,
        child: int,
        parents: list[int],
        cols: list[list[Any]],
        discrete: dict[int, list[Any]],
        weights: np.ndarray | None = None,
        vec_dims: dict | None = None,
    ):
        proto = cls(child, parents, discrete, "binomial", [], np.zeros(1), vec_dims=vec_dims)
        x = proto._design(cols)
        col = cols[child]
        levels = sorted(set(col), key=repr)
        is_count = all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) and int(v) >= 0 for v in col)
        if len(levels) == 2:
            from mixle.inference.glm import glm

            y01 = np.asarray([1.0 if v == levels[1] else 0.0 for v in col])
            with np.errstate(all="ignore"):  # separable data diverges IRLS; the finite-gain guard rejects it
                beta = glm(x, y01, family="binomial", weights=weights).coef
            return cls(child, parents, discrete, "binomial", levels, beta, vec_dims=vec_dims)
        if is_count and len(levels) > 2:
            from mixle.inference.glm import glm

            with np.errstate(all="ignore"):
                beta = glm(x, np.asarray(col, dtype=np.float64), family="poisson", weights=weights).coef
            return cls(child, parents, discrete, "poisson", [], beta, vec_dims=vec_dims)
        w = _fit_multinomial_logistic(x, np.asarray([levels.index(v) for v in col]), len(levels), weights=weights)
        return cls(child, parents, discrete, "multinomial", levels, w, vec_dims=vec_dims)

    def n_params(self) -> int:
        return int(self.weights.size)

    def describe(self, field_names: Sequence[str] | None = None) -> dict[str, Any]:
        """The fitted GLM edge: coefficients labeled by feature name, plus the family (binomial ->
        logistic log-odds of ``levels[1]`` vs. the ``levels[0]`` reference; poisson -> log-rate; multinomial
        -> one log-odds row per non-reference level, vs. ``levels[0]``). Real fitted numbers, not a
        placeholder -- e.g. a positive binomial coefficient means that feature raises the odds of the
        positive level, holding the others fixed."""
        feat_names = _design_feature_names(self.parents, self.discrete, self.vec_dims, field_names)
        base = {
            "field": _name(self.child, field_names),
            "kind": f"glm-{self.kind}",
            "parents": [_name(p, field_names) for p in self.parents],
        }
        if self.kind == "multinomial":
            base["reference_level"] = self.levels[0]
            base["coefficients"] = {
                str(level): {n: round(float(w), 6) for n, w in zip(feat_names, row)}
                for level, row in zip(self.levels[1:], self.weights)
            }
        else:
            base["coefficients"] = {n: round(float(w), 6) for n, w in zip(feat_names, self.weights)}
            if self.kind == "binomial":
                base["positive_level"] = self.levels[1]
                base["reference_level"] = self.levels[0]
        return base


def _logsumexp_rows(a: np.ndarray) -> np.ndarray:
    m = np.max(a, axis=1)
    return m + np.log(np.sum(np.exp(a - m[:, None]), axis=1))


def _fit_multinomial_logistic(
    x: np.ndarray, y_idx: np.ndarray, k: int, ridge: float = 1e-6, weights: np.ndarray | None = None
) -> np.ndarray:
    """Softmax regression (class 0 reference) by L-BFGS on the convex ridge-penalized (weighted) NLL."""
    from scipy.optimize import minimize

    n, d = x.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y_idx] = 1.0
    w_obs = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    def nll_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
        w = flat.reshape(k - 1, d)
        logits = np.concatenate([np.zeros((n, 1)), x @ w.T], axis=1)
        lse = _logsumexp_rows(logits)
        nll = float(np.sum(w_obs * (lse - logits[np.arange(n), y_idx])) + 0.5 * ridge * np.sum(w * w))
        p = np.exp(logits - lse[:, None])
        grad = ((p[:, 1:] - onehot[:, 1:]) * w_obs[:, None]).T @ x + ridge * w
        return nll, grad.ravel()

    res = minimize(nll_grad, np.zeros((k - 1) * d), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x.reshape(k - 1, d)


class _DiscreteConditionalFactor:
    """A discrete/count child: a fitted child distribution per joint configuration of its (discrete) parents."""

    __pysp_serializable__ = True

    def __init__(self, child: int, parents: list[int], table: dict[tuple, Any], backoff: Any) -> None:
        self.child = child
        self.parents = list(parents)
        self.table = table  # config-tuple -> fitted child distribution
        self.backoff = backoff  # marginal child distribution for unseen configs

    def _config(self, values: Sequence[Any]) -> tuple:
        return tuple(values)

    def _dist(self, config: tuple) -> Any:
        return self.table.get(config, self.backoff)

    def seq_log_density(self, cols: list[list[Any]]) -> np.ndarray:
        out = np.empty(len(cols[self.child]), dtype=np.float64)
        for j in range(len(out)):
            d = self._dist(self._config([cols[p][j] for p in self.parents]))
            out[j] = d.log_density(cols[self.child][j])
        return out

    def log_density(self, x: tuple) -> float:
        return float(self._dist(self._config([x[p] for p in self.parents])).log_density(x[self.child]))

    def sample(self, x: list, rng: np.random.RandomState) -> Any:
        d = self._dist(self._config([x[p] for p in self.parents]))
        return d.sampler(int(rng.randint(0, 2**31 - 1))).sample(1)[0]

    @classmethod
    def fit(
        cls,
        child: int,
        parents: list[int],
        cols: list[list[Any]],
        template: Any,
        max_its: int,
        weights: np.ndarray | None = None,
    ):
        n = len(cols[child])
        backoff = _leaf_fit(cols[child], template, max_its, weights)
        groups: dict[tuple, list[int]] = {}
        for j in range(n):
            groups.setdefault(tuple(cols[p][j] for p in parents), []).append(j)
        table = {
            cfg: _leaf_fit(
                [cols[child][j] for j in idx],
                template,
                max_its,
                None if weights is None else weights[idx],
            )
            for cfg, idx in ((cfg, np.asarray(idx)) for cfg, idx in groups.items())
        }
        return cls(child, parents, table, backoff)

    def n_params(self) -> int:
        return _num_free_params(self.backoff) * max(1, len(self.table))

    def describe(self, field_names: Sequence[str] | None = None, *, max_configurations: int = 20) -> dict[str, Any]:
        """The fitted per-configuration table: how many distinct parent-value combinations were observed,
        the marginal fallback used for any unseen combination, and (up to ``max_configurations``) each
        observed combination's own fitted child distribution -- real fitted distributions, not a placeholder."""
        parent_names = [_name(p, field_names) for p in self.parents]
        items = sorted(self.table.items(), key=lambda kv: repr(kv[0]))
        configurations = [
            {"given": dict(zip(parent_names, cfg)), "fitted": str(dist)} for cfg, dist in items[:max_configurations]
        ]
        return {
            "field": _name(self.child, field_names),
            "kind": "discrete-conditional",
            "parents": parent_names,
            "n_configurations": len(self.table),
            "backoff_for_unseen_configurations": str(self.backoff),
            "configurations": configurations,
            "configurations_truncated": len(self.table) > max_configurations,
        }


def _leaf_fit(values: list, template: Any, max_its: int, weights: np.ndarray | None) -> Any:
    """Fit a leaf template to ``values``; with ``weights`` the sufficient statistics are accumulated
    directly (one weighted pass — exact for the simple exponential-family leaves the templates are)."""
    if weights is None:
        return fit(values, _clone(template), max_its=max_its, out=None)
    est = _clone(template)
    enc = est.accumulator_factory().make().acc_to_encoder().seq_encode(values)
    acc = est.accumulator_factory().make()
    acc.seq_update(enc, np.asarray(weights, dtype=np.float64), None)
    return est.estimate(None, acc.value())


class HeterogeneousBayesianNetwork:
    """A DAG joint over a heterogeneous record: ``log p(x) = sum_i log P(x_i | parents(i))`` over fitted factors."""

    __pysp_serializable__ = True

    def __init__(self, factors: Sequence[Any]) -> None:
        self.factors = list(sorted(factors, key=lambda f: f.child))
        self.order = _topo_order([f.parents for f in self.factors])

    def __str__(self) -> str:
        e = [f"{p}->{f.child}" for f in self.factors for p in f.parents]
        return f"HeterogeneousBayesianNetwork(fields={len(self.factors)}, edges=[{', '.join(e) or 'none'}])"

    def edges(self) -> list[tuple[int, int]]:
        """Return DAG edges as ``(parent_field, child_field)`` pairs."""
        return [(p, f.child) for f in self.factors for p in f.parents]

    def describe(self, field_names: Sequence[str] | None = None) -> dict[str, Any]:
        """A real, structured explanation of what the fit found -- every field's per-factor report
        (:meth:`_MarginalFactor.describe` and friends: fitted regression coefficients, GLM weights, the
        conditional table, or the marginal summary), split into ``edges`` (fields with >=1 parent) and
        ``roots`` (independent fields). Pass ``field_names`` (e.g. the tuple position -> column name map)
        to label everything by name instead of integer index. Every value is read off this network's own
        fitted parameters -- nothing here is a placeholder or a re-derived approximation.
        """
        if field_names is not None and len(field_names) != len(self.factors):
            raise ValueError(
                f"field_names has {len(field_names)} entries but this network has {len(self.factors)} fields"
            )
        fields = [f.describe(field_names) for f in self.factors]
        return {
            "model_type": type(self).__name__,
            "n_fields": len(self.factors),
            "edges": [f for f in fields if f["parents"]],
            "roots": [f for f in fields if not f["parents"]],
        }

    def log_density(self, x: tuple) -> float:
        """Evaluate the joint log density of one record."""
        return float(sum(f.log_density(x) for f in self.factors))

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        """Evaluate joint log density for encoded records."""
        cols, n = encoded
        out = np.zeros(n, dtype=np.float64)
        for f in self.factors:
            out += f.seq_log_density(cols)
        return out

    def dist_to_encoder(self) -> Any:
        """Return the encoder for record batches consumed by ``seq_log_density``."""
        return _BNEncoder(len(self.factors))

    def sampler(self, seed: int | None = None) -> Any:
        """Return a sampler for the fitted Bayesian network."""
        return _BNSampler(self, seed)


class _BNEncoder:
    def __init__(self, n_fields: int) -> None:
        self.n_fields = n_fields

    def seq_encode(self, data: Sequence[tuple]) -> tuple[list[list[Any]], int]:
        return _columns(list(data)), len(data)


class _BNSampler:
    def __init__(self, net: HeterogeneousBayesianNetwork, seed: int | None) -> None:
        self.net = net
        self.rng = np.random.RandomState(seed)
        self._by_child = {f.child: f for f in net.factors}

    def sample(self, size: int = 1) -> list[tuple]:
        rows = []
        for _ in range(size):
            vals: list[Any] = [None] * len(self.net.factors)
            for i in self.net.order:
                vals[i] = self._by_child[i].sample(vals, self.rng)
            rows.append(tuple(vals))
        return rows


class MixtureOfBayesianNetworks:
    """A latent mixture whose components each carry their own heterogeneous DAG (regression edges and all).

    The fullest form of the moat: it discovers the clustering *and* each cluster's cross-field graphical model,
    so the *slope* of one field on another (or the whole dependency graph) can differ between clusters -- which a
    single network cannot represent. ``log p(x) = logsumexp_k(log w_k + log p_k(x))`` over
    :class:`HeterogeneousBayesianNetwork` components. Fit by :func:`learn_mixture_bayesian_network`.
    """

    def __init__(self, components: Sequence[HeterogeneousBayesianNetwork], weights: Sequence[float]) -> None:
        self.components = list(components)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.log_weights = np.log(np.clip(self.weights, 1e-300, None))

    def __str__(self) -> str:
        return f"MixtureOfBayesianNetworks(k={len(self.components)}, weights={np.round(self.weights, 3).tolist()})"

    def _component_ll(self, encoded: Any) -> np.ndarray:
        return np.stack([c.seq_log_density(encoded) for c in self.components], axis=1)  # (n, K)

    def log_density(self, x: tuple) -> float:
        """Evaluate the mixture joint log density of one record."""
        from scipy.special import logsumexp

        return float(logsumexp(self.log_weights + np.array([c.log_density(x) for c in self.components])))

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        """Evaluate mixture joint log density for encoded records."""
        from scipy.special import logsumexp

        return logsumexp(self._component_ll(encoded) + self.log_weights[None, :], axis=1)

    def dist_to_encoder(self) -> Any:
        """Return the record encoder shared by all mixture components."""
        return _BNEncoder(len(self.components[0].factors))

    def responsibilities(self, data: Sequence[tuple]) -> np.ndarray:
        """Return posterior component probabilities for each record."""
        enc = self.dist_to_encoder().seq_encode(list(data))
        joint = self._component_ll(enc) + self.log_weights[None, :]
        joint -= joint.max(axis=1, keepdims=True)
        r = np.exp(joint)
        return r / r.sum(axis=1, keepdims=True)

    def sampler(self, seed: int | None = None) -> Any:
        """Return a sampler for the mixture of Bayesian networks."""
        return _BNMixtureSampler(self, seed)

    @property
    def n_components(self) -> int:
        """Return the number of mixture components."""
        return len(self.components)


class _BNMixtureSampler:
    def __init__(self, mix: MixtureOfBayesianNetworks, seed: int | None) -> None:
        self.mix = mix
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int = 1) -> list[tuple]:
        ks = self.rng.choice(self.mix.n_components, size=size, p=self.mix.weights)
        return [self.mix.components[int(k)].sampler(int(self.rng.randint(0, 2**31 - 1))).sample(1)[0] for k in ks]


def learn_mixture_bayesian_network(
    data: Sequence[tuple],
    n_components: int,
    *,
    restarts: int = 3,
    max_iter: int = 12,
    seed: int = 0,
    max_parents: int = 2,
    min_gain: float = 0.0,
    max_its: int = 30,
    em: str = "hard",
) -> MixtureOfBayesianNetworks:
    """Fit a :class:`MixtureOfBayesianNetworks` by EM: discover clusters and each cluster's DAG.

    ``em="hard"`` (default): each iteration re-learns a network per cluster on its assigned points and
    reassigns every record to its most-probable cluster, until assignments stabilize. ``em="soft"``:
    proper EM -- every record contributes to EVERY cluster with its responsibility, each component's
    structure search and factor fits are responsibility-weighted (``learn_bayesian_network(weights=)``),
    and convergence is on the observed-data log-likelihood; boundary points shape both clusters instead
    of whipsawing between them. Both run from the same restarts (k-means-seeded + random); best final
    log-likelihood wins. Starved clusters are re-seeded (hard) / responsibility-floored (soft).
    """
    if em not in ("hard", "soft"):
        raise ValueError(f"em must be 'hard' or 'soft', got {em!r}")
    from mixle.inference.structure import _kmeans_init

    data = list(data)
    n = len(data)
    rng = np.random.RandomState(seed)
    min_size = max(10, n // (4 * n_components))

    def learn(subset: list[tuple], w: np.ndarray | None = None) -> HeterogeneousBayesianNetwork:
        return learn_bayesian_network(subset, max_parents=max_parents, min_gain=min_gain, max_its=max_its, weights=w)

    inits = [
        _kmeans_init(data, n_components, rng, numeric_only=True),
        _kmeans_init(data, n_components, rng, numeric_only=False),
    ]
    inits += [rng.randint(0, n_components, n) for _ in range(max(0, restarts - len(inits)))]

    best: MixtureOfBayesianNetworks | None = None
    best_ll = -np.inf
    for assign in inits:
        if em == "soft":
            model, ll = _soft_em_run(data, n_components, assign, learn, max_iter)
        else:
            model, ll = _hard_em_run(data, n_components, assign, learn, max_iter, min_size, rng)
        if ll > best_ll:
            best_ll, best = ll, model
    assert best is not None
    return best


def _hard_em_run(data, n_components, assign, learn, max_iter, min_size, rng):
    n = len(data)
    model = None
    prev = None
    for _it in range(max_iter):
        comps, counts = [], []
        for k in range(n_components):
            idx = np.flatnonzero(assign == k)
            if len(idx) < min_size:
                idx = rng.choice(n, size=min_size, replace=False)
            comps.append(learn([data[i] for i in idx]))
            counts.append(len(idx))
        weights = np.asarray(counts, dtype=np.float64)
        weights /= weights.sum()
        model = MixtureOfBayesianNetworks(comps, weights)
        new = model.responsibilities(data).argmax(axis=1)
        if prev is not None and np.array_equal(new, prev):
            break
        prev, assign = assign, new
    assert model is not None
    ll = float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))
    return model, ll


def _soft_em_run(data, n_components, assign, learn, max_iter, tol: float = 1e-5):
    n = len(data)
    # a soft one-hot init: 0.95 on the seeded cluster, the rest spread — enough overlap to move points,
    # This is not a running floor: a persistent floor makes every component carry a sliver of every regime,
    # inflating its variance and dragging the whole mixture below the hard-EM fit).
    r = np.full((n, n_components), 0.05 / max(n_components - 1, 1))
    r[np.arange(n), assign] = 0.95
    model = None
    prev_ll = -np.inf
    for _it in range(max_iter):
        mass = r.sum(axis=0)
        for k in np.flatnonzero(mass < 2.0):  # a starved component restarts from a random slice
            r[:, k] = np.random.RandomState(int(mass.sum()) + k).dirichlet(np.ones(n)) * n * 0.05
        r = np.clip(r, 1e-12, None)
        r /= r.sum(axis=1, keepdims=True)
        comps = [learn(data, w=r[:, k]) for k in range(n_components)]
        mix_w = r.mean(axis=0)
        model = MixtureOfBayesianNetworks(comps, mix_w)
        enc = model.dist_to_encoder().seq_encode(data)
        ll = float(np.sum(model.seq_log_density(enc)))
        r = model.responsibilities(data)
        if ll - prev_ll < tol * max(n, 1):
            break
        prev_ll = ll
    assert model is not None
    ll = float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))
    return model, ll


def bayesian_network_bic(model: Any, data: Sequence[tuple]) -> float:
    """BIC (lower is better) for a fitted network or mixture: ``-2 LL + n_params log n``."""
    data = list(data)
    enc = model.dist_to_encoder().seq_encode(data)
    ll = float(np.sum(model.seq_log_density(enc)))
    if isinstance(model, MixtureOfBayesianNetworks):
        k = sum(sum(f.n_params() for f in c.factors) for c in model.components) + (len(model.components) - 1)
    else:
        k = sum(f.n_params() for f in model.factors)
    return -2.0 * ll + k * float(np.log(max(len(data), 2)))


def select_mixture_components(
    data: Sequence[tuple],
    k_values: Sequence[int] = (1, 2, 3, 4),
    *,
    em: str = "hard",
    seed: int = 0,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Model selection over the number of clusters by BIC.

    Fits :func:`learn_bayesian_network` for ``k=1`` and :func:`learn_mixture_bayesian_network` for each
    larger ``k``, scores each by :func:`bayesian_network_bic`, and returns ``(best_model, report)`` where
    ``report = {"k": chosen, "bic": {k: score, ...}}``. BIC's ``n_params log n`` penalty is what stops a
    mixture of per-cluster DAGs from always preferring more clusters.
    """
    data = list(data)
    scores: dict[int, float] = {}
    models: dict[int, Any] = {}
    for k in k_values:
        if int(k) <= 1:
            models[1] = learn_bayesian_network(data, **{a: v for a, v in kwargs.items() if a != "restarts"})
            scores[1] = bayesian_network_bic(models[1], data)
        else:
            models[int(k)] = learn_mixture_bayesian_network(data, int(k), em=em, seed=seed, **kwargs)
            scores[int(k)] = bayesian_network_bic(models[int(k)], data)
    k_best = min(scores, key=scores.get)  # type: ignore[arg-type]
    return models[k_best], {"k": k_best, "bic": scores}


# --- structure search ---------------------------------------------------------------------------------------


def learn_bayesian_network(
    data: Sequence[tuple],
    *,
    max_parents: int = 2,
    min_gain: float = 0.0,
    max_its: int = 30,
    weights: Sequence[float] | None = None,
) -> HeterogeneousBayesianNetwork:
    """Discover a heterogeneous DAG for ``data`` and return the fitted network.

    Each field greedily gains up to ``max_parents`` parents by BIC-penalized conditional likelihood, keeping the
    graph acyclic. Continuous children become conditional-linear-Gaussian factors (regression on continuous +
    one-hot discrete parents); discrete/count children condition on the joint config of their discrete parents,
    or become GLM nodes when a driver is continuous. With ``weights`` (soft-EM responsibilities), every factor
    fit and the BIC search itself are responsibility-weighted, with effective sample size ``sum(weights)``.
    """
    data = list(data)
    cols = _columns(data)
    n_fields = len(cols)
    n = len(data)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    n_eff = float(n) if w is None else float(np.sum(w))
    import mixle.stats as st

    # a vector-valued field (fixed-length numeric sequence, e.g. an embedding) is neither discrete nor a
    # scalar Gaussian -- it becomes a multivariate-Gaussian marginal / multivariate CLG node, and its
    # components splice into the design matrix when it is a parent in a cross-modal graph.
    vec_dims: dict[int, int] = {i: _vector_dim(cols[i]) for i in range(n_fields) if _is_vector_col(cols[i])}
    discrete = [(i not in vec_dims and _is_discrete(c)) for i, c in enumerate(cols)]
    # continuous fields get a Gaussian marginal (defined on all of R, consistent with the CLG children) so a
    # mixture component can score every point; discrete fields keep the automatic family (categorical/count).
    templates = [
        None if i in vec_dims else (_field_estimator(cols[i]) if discrete[i] else st.GaussianEstimator())
        for i in range(n_fields)
    ]
    # key=repr (not bare comparison): a discrete column may carry a missing sentinel (``None``) beside
    # str/int/bool levels, and ``None`` has no ``<`` against those types (TypeError). repr gives a total,
    # deterministic order regardless of level type mix -- same guard `_GLMFactor.fit` already applies below.
    levels = {i: sorted(set(cols[i]), key=repr) for i in range(n_fields) if discrete[i]}

    def _wsum(ll: np.ndarray) -> float:
        return float(np.sum(ll) if w is None else np.dot(w, ll))

    parents: list[list[int]] = [[] for _ in range(n_fields)]
    factors: list[Any] = [None] * n_fields
    base_ll = np.zeros(n_fields)
    for c in range(n_fields):
        factors[c] = _fit_factor(c, [], cols, discrete, levels, templates[c], max_its, w, vec_dims)
        base_ll[c] = _wsum(factors[c].seq_log_density(cols))

    # global greedy: each round add the single best-penalized-gain edge over the whole graph, so the cheaper
    # (fewer-parameter) orientation of a dependence wins instead of whichever node happened to be visited first.
    log_n = np.log(max(n_eff, 2.0))
    while True:
        best = (min_gain, -1, -1, None)  # (gain, child, parent, factor)
        for c in range(n_fields):
            if len(parents[c]) >= max_parents:
                continue
            for q in range(n_fields):
                if q == c or q in parents[c] or _would_cycle(parents, q, c):
                    continue
                cand = _fit_factor(c, [*parents[c], q], cols, discrete, levels, templates[c], max_its, w, vec_dims)
                with np.errstate(all="ignore"):
                    ll = _wsum(cand.seq_log_density(cols))
                if not np.isfinite(ll):
                    continue  # a diverged candidate fit (e.g. separable-data IRLS) loses outright
                gain = ll - base_ll[c] - 0.5 * (cand.n_params() - factors[c].n_params()) * log_n
                if gain > best[0]:
                    best = (gain, c, q, cand)
        _, c, q, cand = best
        if cand is None:
            break
        parents[c].append(q)
        factors[c] = cand
        base_ll[c] = _wsum(cand.seq_log_density(cols))

    return HeterogeneousBayesianNetwork(factors)


def _is_vector_col(col: Sequence[Any]) -> bool:
    """A field is vector-valued iff every value is the same-length (>=2) numeric sequence."""
    first = col[0]
    if isinstance(first, str) or np.isscalar(first):
        return False
    try:
        d = len(first)
    except TypeError:
        return False
    if d < 2 or not isinstance(first[0], (int, float, np.integer, np.floating)):
        return False
    return all(hasattr(v, "__len__") and len(v) == d and not isinstance(v, str) for v in col)


def _vector_dim(col: Sequence[Any]) -> int:
    return len(col[0])


def _fit_factor(child, parents, cols, discrete, levels, template, max_its, weights=None, vec_dims=None):
    vec_dims = vec_dims or {}
    if child in vec_dims:  # a vector-valued child: multivariate-Gaussian marginal or multivariate CLG
        if not parents:
            return _VectorMarginalFactor.fit(child, cols, weights)
        disc = {p: levels[p] for p in parents if discrete[p]}
        vpar = {p: vec_dims[p] for p in parents if p in vec_dims}
        return _VectorCLGFactor.fit(child, parents, cols, disc, vpar, weights)
    if not parents:
        return _MarginalFactor(child, _leaf_fit(cols[child], template, max_its, weights))
    disc = {p: levels[p] for p in parents if discrete[p]}
    vpar = {p: vec_dims[p] for p in parents if p in vec_dims}
    if discrete[child]:
        if len(disc) == len(parents):  # all-discrete parents: the exact per-config table
            return _DiscreteConditionalFactor.fit(child, parents, cols, template, max_its, weights)
        return _GLMFactor.fit(child, parents, cols, disc, weights, vec_dims=vpar)  # a continuous/vector driver
    return _LinearGaussianFactor.fit(child, parents, cols, disc, weights, vec_dims=vpar)


def _would_cycle(parents: list[list[int]], new_parent: int, child: int) -> bool:
    """Adding ``new_parent -> child`` cycles iff ``child`` is already an ancestor of ``new_parent``."""
    stack, seen = [new_parent], set()
    while stack:
        node = stack.pop()
        if node == child:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(parents[node])
    return False


def _topo_order(parents_of: Sequence[Sequence[int]]) -> list[int]:
    order, seen = [], set()

    def visit(i: int) -> None:
        if i in seen:
            return
        for p in parents_of[i]:
            visit(p)
        seen.add(i)
        order.append(i)

    for i in range(len(parents_of)):
        visit(i)
    return order
