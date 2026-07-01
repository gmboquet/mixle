"""Heterogeneous Bayesian network learning -- a directed graph over mixed-type fields with parametric edges.

This deepens :mod:`mixle.inference.structure` (a single-parent forest with quantile-binned continuous parents)
into the real thing: a **DAG** where a field may have *several* parents, and continuous dependence is a
*parametric* conditional, not a binning. A continuous child is a conditional-linear-Gaussian node -- ``child ~
N(w . [continuous parents, one-hot(discrete parents)] + b, sigma^2)`` -- so a real driven by two reals, or by a
category and a real, is modeled exactly and cheaply (closed-form least squares). A discrete/count child conditions
on the joint configuration of its discrete parents (with a marginal backoff for unseen configs).

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


class _LinearGaussianFactor:
    """A continuous child as a linear-Gaussian of its parents (continuous raw + one-hot discrete): the CLG node."""

    def __init__(self, child: int, parents: list[int], discrete: dict[int, list[Any]], coef: np.ndarray, sigma: float):
        self.child = child
        self.parents = list(parents)
        self.discrete = discrete  # parent idx -> its sorted levels (one-hot, drop-first); absent => continuous
        self.coef = coef  # (d+1,) : weights then intercept
        self.sigma = float(sigma)

    def _row(self, values: Sequence[Any]) -> np.ndarray:
        feats: list[float] = []
        for p, v in zip(self.parents, values):
            if p in self.discrete:
                levels = self.discrete[p]
                feats.extend(1.0 if v == lv else 0.0 for lv in levels[1:])  # drop-first
            else:
                feats.append(float(v))
        return np.asarray([*feats, 1.0], dtype=np.float64)

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
    def fit(cls, child: int, parents: list[int], cols: list[list[Any]], discrete: dict[int, list[Any]]):
        stub = cls(child, parents, discrete, np.zeros(1), 1.0)
        x = stub._design(cols)
        y = np.asarray(cols[child], dtype=np.float64)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        sigma = float(np.sqrt(max(resid.var(), 1e-6)))
        return cls(child, parents, discrete, coef, sigma)

    def n_params(self) -> int:
        return self.coef.shape[0] + 1


class _DiscreteConditionalFactor:
    """A discrete/count child: a fitted child distribution per joint configuration of its (discrete) parents."""

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
    def fit(cls, child: int, parents: list[int], cols: list[list[Any]], template: Any, max_its: int):
        n = len(cols[child])
        backoff = fit(cols[child], _clone(template), max_its=max_its, out=None)
        groups: dict[tuple, list[Any]] = {}
        for j in range(n):
            groups.setdefault(tuple(cols[p][j] for p in parents), []).append(cols[child][j])
        table = {cfg: fit(vals, _clone(template), max_its=max_its, out=None) for cfg, vals in groups.items()}
        return cls(child, parents, table, backoff)

    def n_params(self) -> int:
        return _num_free_params(self.backoff) * max(1, len(self.table))


class HeterogeneousBayesianNetwork:
    """A DAG joint over a heterogeneous record: ``log p(x) = sum_i log P(x_i | parents(i))`` over fitted factors."""

    def __init__(self, factors: Sequence[Any]) -> None:
        self.factors = list(sorted(factors, key=lambda f: f.child))
        self.order = _topo_order([f.parents for f in self.factors])

    def __str__(self) -> str:
        e = [f"{p}->{f.child}" for f in self.factors for p in f.parents]
        return f"HeterogeneousBayesianNetwork(fields={len(self.factors)}, edges=[{', '.join(e) or 'none'}])"

    def edges(self) -> list[tuple[int, int]]:
        return [(p, f.child) for f in self.factors for p in f.parents]

    def log_density(self, x: tuple) -> float:
        return float(sum(f.log_density(x) for f in self.factors))

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        cols, n = encoded
        out = np.zeros(n, dtype=np.float64)
        for f in self.factors:
            out += f.seq_log_density(cols)
        return out

    def dist_to_encoder(self) -> Any:
        return _BNEncoder(len(self.factors))

    def sampler(self, seed: int | None = None) -> Any:
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
        from scipy.special import logsumexp

        return float(logsumexp(self.log_weights + np.array([c.log_density(x) for c in self.components])))

    def seq_log_density(self, encoded: Any) -> np.ndarray:
        from scipy.special import logsumexp

        return logsumexp(self._component_ll(encoded) + self.log_weights[None, :], axis=1)

    def dist_to_encoder(self) -> Any:
        return _BNEncoder(len(self.components[0].factors))

    def responsibilities(self, data: Sequence[tuple]) -> np.ndarray:
        enc = self.dist_to_encoder().seq_encode(list(data))
        joint = self._component_ll(enc) + self.log_weights[None, :]
        joint -= joint.max(axis=1, keepdims=True)
        r = np.exp(joint)
        return r / r.sum(axis=1, keepdims=True)

    def sampler(self, seed: int | None = None) -> Any:
        return _BNMixtureSampler(self, seed)

    @property
    def n_components(self) -> int:
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
) -> MixtureOfBayesianNetworks:
    """Fit a :class:`MixtureOfBayesianNetworks` by hard EM -- discover clusters AND each cluster's DAG.

    Each iteration re-learns a heterogeneous network per cluster on its assigned points (M-step) and reassigns
    every record to its most-probable cluster (E-step), until assignments stabilize; the best of ``restarts``
    (k-means-seeded + random) initializations wins. Starved clusters are re-seeded so none collapses.
    """
    from mixle.inference.structure import _kmeans_init

    data = list(data)
    n = len(data)
    rng = np.random.RandomState(seed)
    min_size = max(10, n // (4 * n_components))

    def learn(subset: list[tuple]) -> HeterogeneousBayesianNetwork:
        return learn_bayesian_network(subset, max_parents=max_parents, min_gain=min_gain, max_its=max_its)

    inits = [
        _kmeans_init(data, n_components, rng, numeric_only=True),
        _kmeans_init(data, n_components, rng, numeric_only=False),
    ]
    inits += [rng.randint(0, n_components, n) for _ in range(max(0, restarts - len(inits)))]

    best: MixtureOfBayesianNetworks | None = None
    best_ll = -np.inf
    for assign in inits:
        model: MixtureOfBayesianNetworks | None = None
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
        if ll > best_ll:
            best_ll, best = ll, model
    assert best is not None
    return best


# --- structure search ---------------------------------------------------------------------------------------


def learn_bayesian_network(
    data: Sequence[tuple],
    *,
    max_parents: int = 2,
    min_gain: float = 0.0,
    max_its: int = 30,
) -> HeterogeneousBayesianNetwork:
    """Discover a heterogeneous DAG for ``data`` and return the fitted network.

    Each field greedily gains up to ``max_parents`` parents by BIC-penalized conditional likelihood, keeping the
    graph acyclic. Continuous children become conditional-linear-Gaussian factors (regression on continuous +
    one-hot discrete parents); discrete/count children condition on the joint config of their discrete parents.
    """
    data = list(data)
    cols = _columns(data)
    n_fields = len(cols)
    n = len(data)
    import mixle.stats as st

    discrete = [_is_discrete(c) for c in cols]
    # continuous fields get a Gaussian marginal (defined on all of R, consistent with the CLG children) so a
    # mixture component can score every point; discrete fields keep the automatic family (categorical/count).
    templates = [_field_estimator(cols[i]) if discrete[i] else st.GaussianEstimator() for i in range(n_fields)]
    levels = {i: sorted(set(cols[i])) for i in range(n_fields) if discrete[i]}

    parents: list[list[int]] = [[] for _ in range(n_fields)]
    factors: list[Any] = [None] * n_fields
    base_ll = np.zeros(n_fields)
    for c in range(n_fields):
        factors[c] = _fit_factor(c, [], cols, discrete, levels, templates[c], max_its)
        base_ll[c] = float(np.sum(factors[c].seq_log_density(cols)))

    # global greedy: each round add the single best-penalized-gain edge over the WHOLE graph, so the cheaper
    # (fewer-parameter) orientation of a dependence wins instead of whichever node happened to be visited first.
    log_n = np.log(max(n, 2))
    while True:
        best = (min_gain, -1, -1, None)  # (gain, child, parent, factor)
        for c in range(n_fields):
            if len(parents[c]) >= max_parents:
                continue
            for q in range(n_fields):
                if q == c or q in parents[c] or _would_cycle(parents, q, c):
                    continue
                if discrete[c] and not discrete[q]:
                    continue  # a discrete child conditions on discrete parents (a continuous driver -> a CLG child)
                cand = _fit_factor(c, [*parents[c], q], cols, discrete, levels, templates[c], max_its)
                ll = float(np.sum(cand.seq_log_density(cols)))
                gain = ll - base_ll[c] - 0.5 * (cand.n_params() - factors[c].n_params()) * log_n
                if gain > best[0]:
                    best = (gain, c, q, cand)
        _, c, q, cand = best
        if cand is None:
            break
        parents[c].append(q)
        factors[c] = cand
        base_ll[c] = float(np.sum(cand.seq_log_density(cols)))

    return HeterogeneousBayesianNetwork(factors)


def _fit_factor(child, parents, cols, discrete, levels, template, max_its):
    if not parents:
        return _MarginalFactor(child, fit(cols[child], _clone(template), max_its=max_its, out=None))
    if discrete[child]:
        return _DiscreteConditionalFactor.fit(child, parents, cols, template, max_its)
    disc = {p: levels[p] for p in parents if discrete[p]}
    return _LinearGaussianFactor.fit(child, parents, cols, disc)


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
