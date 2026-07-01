"""Automatic dependency-structure learning for heterogeneous records -- the tagline, taken literally.

``CompositeDistribution`` models a record's fields as *independent* (Naive-Bayes under a mixture). But real
heterogeneous data has cross-field dependence -- a category shifts a real's mean, a count's rate tracks another
field -- and modeling it is worth a great deal of likelihood (a blatant category->Gaussian link is ~1000 nats on
600 rows). No mainstream tool discovers that structure across *arbitrary* families: Stan/PyMC make you write it,
sklearn/pomegranate mixtures assume independence, bnlearn/pgmpy are discrete-or-Gaussian only.

This module closes the gap. Dependence is detected the mixle way -- **by modeling it**: fit ``P(child)`` vs
``P(child | parent)`` and compare description length (:func:`dependency_gain`). The winning edges are assembled
into a :class:`DependencyTreeDistribution` -- a directed forest over the record where each field is either a
marginal or a per-parent-value conditional (a real :class:`~mixle.stats.combinator.conditional.ConditionalDistribution`
edge) -- and :func:`learn_structure` picks the forest and fits it automatically. The result scores, samples, and
composes like any mixle distribution, but *models the dependence a composite drops*.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.inference.estimation import fit
from mixle.stats.combinator.conditional import ConditionalDistributionEstimator
from mixle.stats.combinator.null_dist import NullDistribution


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
) -> float:
    """Description-length gain (nats) of modeling ``child`` conditioned on a discrete ``parent`` vs. independently.

    Fits the marginal ``P(child)`` and the conditional ``P(child | parent)`` (a child model per parent value) on
    the same data and returns ``LL_cond - LL_marginal`` minus a complexity penalty for the extra parameters
    (BIC: ``0.5 * (levels - 1) * k * ln n``). Positive means the dependence is worth modeling. This is a
    model-based dependency test -- it works across *any* pair of families, unlike a same-type MI estimate.
    """
    child = list(child)
    n = len(child)
    levels = sorted(set(parent))
    marginal = fit(child, _clone(child_estimator), max_its=max_its, out=None)
    ll_marginal = float(np.sum(marginal.seq_log_density(marginal.dist_to_encoder().seq_encode(child))))

    pairs = list(zip(parent, child))
    cond_est = ConditionalDistributionEstimator(
        estimator_map={lv: _clone(child_estimator) for lv in levels}, given_estimator=None
    )
    cond = fit(pairs, cond_est, max_its=max_its, out=None)
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

    def __init__(self, a: float, b: float, sigma2: float) -> None:
        self.a, self.b, self.sigma2 = float(a), float(b), max(float(sigma2), 1e-12)

    def log_density(self, x: tuple) -> float:
        parent, child = x
        resid = float(child) - (self.a + self.b * float(parent))
        return float(-0.5 * np.log(2.0 * np.pi * self.sigma2) - 0.5 * resid * resid / self.sigma2)

    def dist_to_encoder(self) -> Any:
        return _EdgeEncoder()

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        parent, child = encoded
        resid = np.asarray(child, dtype=float) - (self.a + self.b * np.asarray(parent, dtype=float))
        return -0.5 * np.log(2.0 * np.pi * self.sigma2) - 0.5 * resid * resid / self.sigma2

    def sampler(self, seed: int | None = None) -> Any:
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
    parent: Sequence[Any], child: Sequence[Any], child_estimator: Any, *, max_its: int = 30, penalty: str = "bic"
) -> float:
    """Description-length gain (nats) of a linear-Gaussian *regression* edge ``child ~ a + b*parent`` over the
    child marginal. One extra parameter (the slope) vs. the ``bins * k`` a binned conditional spends — so for a
    real linear dependence this beats binning decisively. Returns ``-inf`` when a regression is undefined."""
    p = np.asarray(parent, dtype=float)
    c = np.asarray(child, dtype=float)
    n = len(c)
    if n < 3 or float(np.var(p)) < 1e-12:
        return float("-inf")
    edge = fit_linear_gaussian_edge(list(zip(p.tolist(), c.tolist())))
    ll_reg = float(np.sum(edge.seq_log_density((p, c))))
    marginal = fit(list(child), _clone(child_estimator), max_its=max_its, out=None)
    ll_marginal = float(np.sum(marginal.seq_log_density(marginal.dist_to_encoder().seq_encode(list(child)))))
    pen = 0.5 * 1.0 * np.log(max(n, 2)) if penalty == "bic" else 0.0  # a single extra parameter: the slope
    return ll_reg - ll_marginal - pen


class DependencyTreeDistribution:
    """A directed-forest joint over a heterogeneous record: each field is a marginal or a conditional on its parent.

    ``log_density(record) = sum_root log P(f_root) + sum_child log P(f_child | f_parent)``. The dependence a
    :class:`~mixle.stats.combinator.composite.CompositeDistribution` assumes away is modeled here as
    per-parent-value conditionals -- while it still scores, samples, and composes like any mixle distribution.
    """

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
        total = 0.0
        for i, parent in enumerate(self.parents):
            if parent is None:
                total += self.factors[i].log_density(x[i])
            else:
                total += self.factors[i].log_density((self._key(i, x[parent]), x[i]))
        return total

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        cols, n = encoded
        out = np.zeros(n, dtype=np.float64)
        for i, parent in enumerate(self.parents):
            fac = self.factors[i]
            if parent is None:
                out += fac.seq_log_density(fac.dist_to_encoder().seq_encode(cols[i]))
            else:
                pairs = [(self._key(i, pv), cv) for pv, cv in zip(cols[parent], cols[i])]
                out += fac.seq_log_density(fac.dist_to_encoder().seq_encode(pairs))
        return out

    def dist_to_encoder(self) -> Any:
        return _DependencyEncoder(len(self.parents))

    def sampler(self, seed: int | None = None) -> Any:
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

    def __init__(self, components: Sequence[DependencyTreeDistribution], weights: Sequence[float]) -> None:
        self.components = list(components)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.log_weights = np.log(np.clip(self.weights, 1e-300, None))

    def __str__(self) -> str:
        return f"MixtureOfDependencyTrees(k={len(self.components)}, weights={np.round(self.weights, 3).tolist()})"

    def _component_ll(self, encoded: Any) -> np.ndarray:
        return np.stack([c.seq_log_density(encoded) for c in self.components], axis=1)  # (n, K)

    def log_density(self, x: tuple) -> float:
        from scipy.special import logsumexp

        return float(logsumexp(self.log_weights + np.array([c.log_density(x) for c in self.components])))

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        from scipy.special import logsumexp

        return logsumexp(self._component_ll(encoded) + self.log_weights[None, :], axis=1)

    def dist_to_encoder(self) -> Any:
        return _DependencyEncoder(len(self.components[0].parents))

    def responsibilities(self, data: Sequence[tuple]) -> np.ndarray:
        """Posterior ``p(component | record)`` for each record -- the E-step and a soft cluster assignment."""
        enc = self.dist_to_encoder().seq_encode(list(data))
        joint = self._component_ll(enc) + self.log_weights[None, :]
        joint -= joint.max(axis=1, keepdims=True)
        r = np.exp(joint)
        return r / r.sum(axis=1, keepdims=True)

    def sampler(self, seed: int | None = None) -> Any:
        return _MixtureTreeSampler(self, seed)

    @property
    def n_components(self) -> int:
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
) -> MixtureOfDependencyTrees:
    """Fit a :class:`MixtureOfDependencyTrees` by hard EM -- discover clusters AND each cluster's dependency graph.

    Each iteration re-learns a dependency forest per cluster on its currently-assigned points (M-step), then
    reassigns every record to its most-probable cluster (E-step), until assignments stabilize. Runs ``restarts``
    random initializations and returns the highest-likelihood fit. Empty/tiny clusters are re-seeded so a
    component never collapses.
    """
    data = list(data)
    n = len(data)
    rng = np.random.RandomState(seed)
    min_size = max(10, n // (4 * n_components))
    best: MixtureOfDependencyTrees | None = None
    best_ll = -np.inf

    def learn(subset: list[tuple]) -> DependencyTreeDistribution:
        return learn_structure(subset, min_gain=min_gain, n_bins=n_bins, max_its=max_its)

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


# --- structure search + fitting ------------------------------------------------------------------------------


def learn_structure(
    data: Sequence[tuple],
    *,
    field_estimators: Sequence[Any] | None = None,
    min_gain: float = 0.0,
    max_levels: int = 64,
    n_bins: int = 4,
    max_its: int = 30,
) -> DependencyTreeDistribution:
    """Discover the dependency forest for heterogeneous ``data`` and return the fitted joint model.

    Any field can be a parent: a discrete one conditions directly, a continuous one is quantile-binned into
    ``n_bins`` conditioning levels (so a real can drive a count, a category, or another real). Scores every
    ``(parent -> child)`` pair by :func:`dependency_gain`, greedily builds a maximum-gain acyclic forest (each
    field at most one parent), and fits each factor. Falls back to independent marginals where no dependence
    clears ``min_gain`` -- never worse than a composite, much better when structure exists. This is "automatic
    inference for composable models of heterogeneous data" made real.
    """
    data = list(data)
    cols = _columns(data)
    n_fields = len(cols)
    templates = list(field_estimators) if field_estimators is not None else [_field_estimator(c) for c in cols]
    discrete = [_is_discrete(c, max_levels=max_levels) for c in cols]

    # a conditioning key per candidate parent: identity for a discrete field, a quantile bin for a continuous one
    binners = [None if discrete[p] else _quantile_binner(cols[p], n_bins) for p in range(n_fields)]
    keyed = [cols[p] if binners[p] is None else [binners[p](v) for v in cols[p]] for p in range(n_fields)]

    # a continuous child on a numeric parent can use a linear-Gaussian REGRESSION edge (1 slope param) instead of
    # a coarse per-bin conditional; each edge takes whichever scores the higher description-length gain.
    numeric = [_is_numeric(cols[i]) for i in range(n_fields)]

    candidates: list[tuple[float, int, int, str]] = []
    for p in range(n_fields):
        for c in range(n_fields):
            if c == p:
                continue
            gain = dependency_gain(keyed[p], cols[c], templates[c], max_its=max_its)
            kind = "binned"
            if not discrete[c] and not discrete[p] and numeric[c] and numeric[p]:
                rgain = regression_gain(cols[p], cols[c], templates[c], max_its=max_its)
                if rgain > gain:
                    gain, kind = rgain, "regression"
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
            factors[i] = fit(cols[i], _clone(templates[i]), max_its=max_its, out=None)
        elif edge_kind[i] == "regression":
            factors[i] = fit_linear_gaussian_edge(list(zip(cols[parents[i]], cols[i])))
            edge_binners[i] = None  # the raw parent value drives the regression
        else:
            p = parents[i]
            keys = keyed[p]
            est = ConditionalDistributionEstimator(
                estimator_map={lv: _clone(templates[i]) for lv in sorted(set(keys))}, given_estimator=None
            )
            factors[i] = fit(list(zip(keys, cols[i])), est, max_its=max_its, out=None)
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
    """A fresh copy of an estimator template (eval/str round-trip; falls back to the same object if unsupported)."""
    try:
        return eval(str(estimator), _estimator_eval_scope())  # noqa: S307 - estimator repr is mixle-controlled
    except Exception:  # noqa: BLE001 - some estimators are already stateless/reusable
        return estimator


def _estimator_eval_scope() -> dict[str, Any]:
    import mixle.stats as st

    scope = {name: getattr(st, name) for name in dir(st) if not name.startswith("_")}
    scope["NullDistribution"] = NullDistribution
    return scope


def _num_free_params(dist: Any) -> int:
    """A rough parameter count for the BIC penalty (used only to scale the complexity term)."""
    for attr in ("mu", "p", "lam", "beta", "alpha"):
        if hasattr(dist, attr):
            v = getattr(dist, attr)
            try:
                return max(1, int(np.asarray(v).size) * 2)
            except (TypeError, ValueError):
                return 2
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
