"""P15 (experimental) -- active causal discovery: buying the interventions that matter.

A chain ``X0->X1->X2``, the reverse ``X2->X1->X0``, and the fork ``X0<-X1->X2`` share the skeleton
``0-1-2`` (one Markov equivalence class). They are parameterized as factorizations of the same
observational joint Gaussian, so observation cannot orient them.
Interventions break the tie decisively -- ``do(X1)`` moves ``X2`` under the chain, ``X0`` under the
reverse, and both under the fork. This module chooses *which* intervention to run by expected
information gain (EIG) over a posterior on candidate causal structures, and identifies the true
structure with fewer experiments than random or observation-only selection.

Everything is exact linear-Gaussian, so the structure likelihood is closed form and the synthetic
ground truth is known -- active causal discovery can be *graded exactly*, which the field mostly
cannot do.

Exploratory ``mixle.experimental`` code (P15 card).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class LinearGaussianSCM:
    """A linear-Gaussian structural causal model over ``n_nodes`` variables.

    ``parents[j]`` lists node ``j``'s parents; each contributes ``weight`` to ``j``'s mean. Nodes
    with no parents are exogenous ``N(0, noise^2)``.
    """

    name: str
    n_nodes: int
    parents: dict[int, list[int]]
    weight: float = 0.9
    noise: float = 1.0
    edge_weights: dict[tuple[int, int], float] | None = None
    noise_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string.")
        if isinstance(self.n_nodes, (bool, np.bool_)) or not isinstance(self.n_nodes, (int, np.integer)):
            raise ValueError("n_nodes must be a positive integer.")
        self.n_nodes = int(self.n_nodes)
        if self.n_nodes <= 0:
            raise ValueError("n_nodes must be a positive integer.")
        if not np.isfinite(self.weight):
            raise ValueError("weight must be finite.")
        if not np.isfinite(self.noise) or self.noise <= 0.0:
            raise ValueError("noise must be finite and positive.")

        canonical_parents: dict[int, list[int]] = {}
        for raw_child, raw_parents in self.parents.items():
            child = self._validated_node(raw_child, "child")
            if isinstance(raw_parents, (str, bytes)) or not hasattr(raw_parents, "__iter__"):
                raise ValueError(f"parents[{child}] must be an iterable of node IDs.")
            node_parents = [self._validated_node(parent, "parent") for parent in raw_parents]
            if child in node_parents:
                raise ValueError(f"node {child} cannot be its own parent.")
            if len(node_parents) != len(set(node_parents)):
                raise ValueError(f"parents[{child}] contains duplicate node IDs.")
            canonical_parents[child] = node_parents
        self.parents = canonical_parents

        declared_edges = {(parent, child) for child, node_parents in self.parents.items() for parent in node_parents}
        edge_weights = {} if self.edge_weights is None else dict(self.edge_weights)
        if not set(edge_weights).issubset(declared_edges):
            raise ValueError("edge_weights contains an edge that is not declared in parents.")
        if any(not np.isfinite(value) for value in edge_weights.values()):
            raise ValueError("edge weights must be finite.")
        self.edge_weights = edge_weights

        if self.noise_scales is not None:
            scales = np.asarray(self.noise_scales, dtype=float)
            if scales.shape != (self.n_nodes,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
                raise ValueError("noise_scales must contain one finite positive scale per node.")
            self.noise_scales = tuple(float(scale) for scale in scales)

        self._topo_order()  # Validate acyclicity at construction, before simulation or scoring.

    def _validated_node(self, node: object, role: str) -> int:
        if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)):
            raise ValueError(f"{role} node IDs must be integers.")
        value = int(node)
        if value < 0 or value >= self.n_nodes:
            raise ValueError(f"{role} node ID {value} is outside [0, {self.n_nodes}).")
        return value

    def _edge_weight(self, parent: int, child: int) -> float:
        assert self.edge_weights is not None
        return float(self.edge_weights.get((parent, child), self.weight))

    def _noise_scale(self, node: int) -> float:
        return float(self.noise if self.noise_scales is None else self.noise_scales[node])

    def _validated_intervention(self, intervention: tuple[int, float] | None) -> tuple[int, float] | None:
        if intervention is None:
            return None
        if not isinstance(intervention, tuple) or len(intervention) != 2:
            raise ValueError("intervention must be None or a (node, finite_value) tuple.")
        node = self._validated_node(intervention[0], "intervention")
        value = intervention[1]
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError("intervention value must be a finite scalar.")
        return node, float(value)

    def _topo_order(self) -> list[int]:
        order: list[int] = []
        seen: set[int] = set()
        active: set[int] = set()

        def visit(j: int) -> None:
            if j in seen:
                return
            if j in active:
                raise ValueError("parents must define an acyclic graph.")
            active.add(j)
            for p in self.parents.get(j, []):
                visit(p)
            active.remove(j)
            seen.add(j)
            order.append(j)

        for j in range(self.n_nodes):
            visit(j)
        return order

    def simulate(self, n: int, rng: np.random.Generator, intervention: tuple[int, float] | None = None) -> np.ndarray:
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        if not hasattr(rng, "standard_normal"):
            raise TypeError("rng must provide standard_normal().")
        intervention = self._validated_intervention(intervention)
        x = np.zeros((int(n), self.n_nodes))
        for j in self._topo_order():
            if intervention is not None and intervention[0] == j:
                x[:, j] = intervention[1]
                continue
            mu = sum((self._edge_weight(p, j) * x[:, p] for p in self.parents.get(j, [])), start=np.zeros(int(n)))
            x[:, j] = mu + self._noise_scale(j) * rng.standard_normal(int(n))
        return x

    def log_likelihood(self, x: np.ndarray, intervention: tuple[int, float] | None = None) -> float:
        """Total log-density of rows ``x`` under this SCM given the intervention regime."""
        intervention = self._validated_intervention(intervention)
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.n_nodes or not np.all(np.isfinite(x)):
            raise ValueError(f"x must be a non-empty finite array with shape (n, {self.n_nodes}).")
        if intervention is not None and not np.all(x[:, intervention[0]] == intervention[1]):
            return float("-inf")
        total = 0.0
        for j in range(self.n_nodes):
            if intervention is not None and intervention[0] == j:
                continue  # an intervened node is set, not modeled
            mu = sum((self._edge_weight(p, j) * x[:, p] for p in self.parents.get(j, [])), start=np.zeros(x.shape[0]))
            s2 = self._noise_scale(j) ** 2
            total += float(np.sum(-0.5 * (_LOG_2PI + np.log(s2) + (x[:, j] - mu) ** 2 / s2)))
        return total

    def observational_covariance(self) -> np.ndarray:
        """Return the exact observational covariance implied by the acyclic SCM."""
        coefficients = np.zeros((self.n_nodes, self.n_nodes))
        for child, node_parents in self.parents.items():
            for parent in node_parents:
                coefficients[child, parent] = self._edge_weight(parent, child)
        transform = np.linalg.inv(np.eye(self.n_nodes) - coefficients)
        noise_covariance = np.diag([self._noise_scale(node) ** 2 for node in range(self.n_nodes)])
        return transform @ noise_covariance @ transform.T


@dataclass
class StructurePosterior:
    """Posterior over a fixed list of candidate SCMs, updated by experiment likelihoods."""

    candidates: list[LinearGaussianSCM]
    log_w: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.candidates = list(self.candidates)
        if not self.candidates:
            raise ValueError("candidates must be non-empty.")
        if len({candidate.name for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate names must be unique.")
        n_nodes = self.candidates[0].n_nodes
        if any(candidate.n_nodes != n_nodes for candidate in self.candidates):
            raise ValueError("all candidates must have the same number of nodes.")
        if self.log_w is None:
            self.log_w = np.full(len(self.candidates), -np.log(len(self.candidates)))
        else:
            self.log_w = np.asarray(self.log_w, dtype=float).copy()
            if self.log_w.shape != (len(self.candidates),):
                raise ValueError("log_w must contain one value per candidate.")
            if np.any(np.isnan(self.log_w)) or np.all(np.isneginf(self.log_w)) or np.any(np.isposinf(self.log_w)):
                raise ValueError("log_w must define at least one finite candidate weight.")

    @property
    def probs(self) -> np.ndarray:
        m = self.log_w - self.log_w.max()
        p = np.exp(m)
        total = p.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("candidate weights cannot be normalized.")
        return p / total

    def entropy(self) -> float:
        p = self.probs
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def update(self, x: np.ndarray, intervention: tuple[int, float] | None) -> None:
        lls = np.array([g.log_likelihood(x, intervention) for g in self.candidates])
        updated = self.log_w + lls
        if np.all(np.isneginf(updated)) or np.any(np.isnan(updated)) or np.any(np.isposinf(updated)):
            raise ValueError("experiment has zero or invalid likelihood under every candidate.")
        self.log_w = updated - updated.max()

    def argmax(self) -> int:
        return int(np.argmax(self.probs))

    def copy(self) -> StructurePosterior:
        return StructurePosterior(self.candidates, self.log_w.copy())


def expected_information_gain(
    posterior: StructurePosterior,
    regime: tuple[int, float] | None,
    *,
    n_batch: int,
    rng: np.random.Generator,
    n_outcomes: int = 6,
) -> float:
    """EIG of running ``regime`` once: current entropy minus the expected posterior entropy.

    Outcomes are simulated from each candidate weighted by the current posterior (the
    posterior-predictive), so no ground truth leaks into the design.
    """
    if isinstance(n_batch, (bool, np.bool_)) or not isinstance(n_batch, (int, np.integer)) or int(n_batch) <= 0:
        raise ValueError("n_batch must be a positive integer.")
    if (
        isinstance(n_outcomes, (bool, np.bool_))
        or not isinstance(n_outcomes, (int, np.integer))
        or int(n_outcomes) <= 0
    ):
        raise ValueError("n_outcomes must be a positive integer.")
    if not hasattr(rng, "standard_normal"):
        raise TypeError("rng must provide standard_normal().")
    if regime is not None:
        regime = posterior.candidates[0]._validated_intervention(regime)

    p = posterior.probs
    h_now = posterior.entropy()
    expected_h = 0.0
    for g_idx, g in enumerate(posterior.candidates):
        if p[g_idx] <= 0:
            continue
        h_g = 0.0
        for _ in range(int(n_outcomes)):
            batch = g.simulate(int(n_batch), rng, regime)
            post = posterior.copy()
            post.update(batch, regime)
            h_g += post.entropy()
        expected_h += p[g_idx] * (h_g / int(n_outcomes))
    return float(h_now - expected_h)


def default_regimes(n_nodes: int, value: float = 2.0) -> list[tuple[int, float] | None]:
    """Observation plus ``do(node = value)`` for each node."""
    if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) <= 0:
        raise ValueError("n_nodes must be a positive integer.")
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError("value must be a finite scalar.")
    return [None, *[(j, float(value)) for j in range(int(n_nodes))]]


@dataclass
class DiscoveryResult:
    identified: int
    correct: bool
    n_experiments: int
    final_probs: list[float]


def active_discovery(
    true_scm: LinearGaussianSCM,
    candidates: list[LinearGaussianSCM],
    *,
    strategy: str = "eig",
    n_batch: int = 8,
    threshold: float = 0.95,
    max_experiments: int = 40,
    seed: int = 0,
    value: float = 2.0,
) -> DiscoveryResult:
    """Run the act-observe-update loop until the posterior concentrates or the budget runs out.

    ``strategy="eig"`` picks the max-EIG regime each step; ``"random"`` picks uniformly; ``"obs"``
    only ever observes. Returns which structure was identified and how many experiments it took.
    """
    if strategy not in {"eig", "random", "obs"}:
        raise ValueError(f"unknown strategy {strategy!r}")
    if isinstance(n_batch, (bool, np.bool_)) or not isinstance(n_batch, (int, np.integer)) or int(n_batch) <= 0:
        raise ValueError("n_batch must be a positive integer.")
    if (
        isinstance(max_experiments, (bool, np.bool_))
        or not isinstance(max_experiments, (int, np.integer))
        or int(max_experiments) <= 0
    ):
        raise ValueError("max_experiments must be a positive integer.")
    if (
        isinstance(threshold, (bool, np.bool_))
        or not np.isscalar(threshold)
        or not np.isfinite(threshold)
        or not 0.0 < float(threshold) <= 1.0
    ):
        raise ValueError("threshold must be finite and in (0, 1].")

    rng = np.random.default_rng(seed)
    posterior = StructurePosterior(candidates)
    regimes = default_regimes(true_scm.n_nodes, value)
    if true_scm.n_nodes != posterior.candidates[0].n_nodes:
        raise ValueError("true_scm and candidates must have the same number of nodes.")
    matches = [i for i, candidate in enumerate(posterior.candidates) if candidate.name == true_scm.name]
    if len(matches) != 1:
        raise ValueError("true_scm.name must identify exactly one candidate.")
    true_idx = matches[0]

    for step in range(1, int(max_experiments) + 1):
        if strategy == "eig":
            eigs = [expected_information_gain(posterior, r, n_batch=n_batch, rng=rng) for r in regimes]
            regime = regimes[int(np.argmax(eigs))]
        elif strategy == "random":
            regime = regimes[rng.integers(len(regimes))]
        elif strategy == "obs":
            regime = None
        batch = true_scm.simulate(int(n_batch), rng, regime)
        posterior.update(batch, regime)
        if posterior.probs.max() >= threshold:
            return DiscoveryResult(posterior.argmax(), posterior.argmax() == true_idx, step, posterior.probs.tolist())

    return DiscoveryResult(
        posterior.argmax(), posterior.argmax() == true_idx, int(max_experiments), posterior.probs.tolist()
    )


def _factorize_covariance(
    name: str,
    parents: dict[int, list[int]],
    covariance: np.ndarray,
    *,
    fallback_weight: float,
    fallback_noise: float,
) -> LinearGaussianSCM:
    """Factor a compatible covariance into one DAG's linear-Gaussian conditionals."""
    n_nodes = covariance.shape[0]
    edge_weights: dict[tuple[int, int], float] = {}
    noise_scales = np.empty(n_nodes)
    for child in range(n_nodes):
        node_parents = parents.get(child, [])
        if node_parents:
            parent_covariance = covariance[np.ix_(node_parents, node_parents)]
            coefficients = np.linalg.solve(parent_covariance, covariance[node_parents, child])
            for parent, coefficient in zip(node_parents, coefficients):
                edge_weights[(parent, child)] = float(coefficient)
            residual_variance = covariance[child, child] - covariance[child, node_parents] @ coefficients
        else:
            residual_variance = covariance[child, child]
        if not np.isfinite(residual_variance) or residual_variance <= 0.0:
            raise ValueError("covariance is not compatible with the requested DAG factorization.")
        noise_scales[child] = np.sqrt(residual_variance)
    return LinearGaussianSCM(
        name,
        n_nodes,
        parents,
        fallback_weight,
        fallback_noise,
        edge_weights=edge_weights,
        noise_scales=tuple(noise_scales),
    )


def markov_equivalent_triple(weight: float = 0.9, noise: float = 1.0) -> list[LinearGaussianSCM]:
    """The chain / reverse-chain / fork over 3 nodes -- observationally equivalent, do()-separable."""
    if not np.isfinite(weight) or weight == 0.0:
        raise ValueError("weight must be finite and non-zero so interventions separate the candidates.")
    if not np.isfinite(noise) or noise <= 0.0:
        raise ValueError("noise must be finite and positive.")

    base = LinearGaussianSCM("base", 3, {1: [0], 2: [1]}, weight, noise)
    covariance = base.observational_covariance()
    candidates = [
        _factorize_covariance("chain", {1: [0], 2: [1]}, covariance, fallback_weight=weight, fallback_noise=noise),
        _factorize_covariance("reverse", {1: [2], 0: [1]}, covariance, fallback_weight=weight, fallback_noise=noise),
        _factorize_covariance("fork", {0: [1], 2: [1]}, covariance, fallback_weight=weight, fallback_noise=noise),
    ]
    if not all(np.allclose(candidate.observational_covariance(), covariance) for candidate in candidates):
        raise RuntimeError("internal error: candidate factorizations are not observationally equivalent.")
    return candidates
