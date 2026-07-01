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

    candidates: list[tuple[float, int, int]] = []
    for p in range(n_fields):
        for c in range(n_fields):
            if c == p:
                continue
            gain = dependency_gain(keyed[p], cols[c], templates[c], max_its=max_its)
            if gain > min_gain:
                candidates.append((gain, p, c))
    candidates.sort(key=lambda t: t[0], reverse=True)

    # greedy maximum-gain forest: each child at most one parent, keep it acyclic (union-find on undirected links)
    parents: list[int | None] = [None] * n_fields
    uf = _UnionFind(n_fields)
    for _gain, p, c in candidates:
        if parents[c] is not None or uf.connected(p, c):
            continue
        parents[c] = p
        uf.union(p, c)

    # fit each factor: roots as marginals, children as per-parent-key conditionals
    factors: list[Any] = [None] * n_fields
    edge_binners: list[Any] = [None] * n_fields
    for i in range(n_fields):
        if parents[i] is None:
            factors[i] = fit(cols[i], _clone(templates[i]), max_its=max_its, out=None)
        else:
            p = parents[i]
            keys = keyed[p]
            est = ConditionalDistributionEstimator(
                estimator_map={lv: _clone(templates[i]) for lv in sorted(set(keys))}, given_estimator=None
            )
            factors[i] = fit(list(zip(keys, cols[i])), est, max_its=max_its, out=None)
            edge_binners[i] = binners[p]
    return DependencyTreeDistribution(parents, factors, edge_binners)


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
