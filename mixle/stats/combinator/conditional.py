"""Conditional distributions over paired observations.

Data type: (Tuple[T0, T1]): The ConditionalDistribution if given by density,
    P(X0,X1) = P_cond(X1|X0)*P_given(X0).

The ConditionalDistribution allows for user defined conditional distributions P_cond(X1|X0), and given distributions
P_given(X0).
"""

import heapq
import itertools
import math
from collections.abc import Iterator, Sequence
from typing import Any, Optional, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream
from mixle.stats.combinator.composite import _distribute_child_prior
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    ConditionalSampler,
    ContractError,
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
    prefix_contract_error,
)

T0 = TypeVar("T0")
T1 = TypeVar("T1")

E0 = TypeVar("E0")
E1 = TypeVar("E1")
E = tuple[int, tuple[T0, ...], tuple[np.ndarray, ...], tuple[E0, ...], E1 | None]
SS0 = TypeVar("SS0")
SS1 = TypeVar("SS1")
SS2 = TypeVar("SS2")


def _conditional_zero_stats(component_estimators: Sequence[Any], num_components: int) -> tuple[Any, ...]:
    """Return per-component zero-value legacy stats from child estimators."""
    if len(component_estimators) != num_components or any(est is None for est in component_estimators):
        return tuple(None for _ in range(num_components))
    return tuple(est.accumulator_factory().make().value() for est in component_estimators)


def _conditional_add_stats(left: Any, right: Any, engine: Any) -> Any:
    """Add component-stacked child sufficient-stat payloads with matching structure."""
    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left.keys()) | set(right.keys())
        return {key: _conditional_add_stats(left.get(key), right.get(key), engine) for key in keys}
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple(_conditional_add_stats(a, b, engine) for a, b in zip(left, right))
    if isinstance(left, list) and isinstance(right, list):
        return [_conditional_add_stats(a, b, engine) for a, b in zip(left, right)]
    return left + right


class ConditionalDistribution(SequenceEncodableProbabilityDistribution):
    """ConditionalDistribution models pairs (x0, x1) with density P_cond(x1 | x0) * P_given(x0),
    where the conditional distributions are looked up from a dictionary keyed by x0."""

    def __init__(
        self,
        dmap: dict[Any, SequenceEncodableProbabilityDistribution] | list[SequenceEncodableProbabilityDistribution],
        default_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        given_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        keys: str | None = None,
        prior: tuple[dict[Any, Any], Any, Any] | None = None,
    ) -> None:
        """Create a conditional distribution over observations ``x=(given, value)``.

        P(x) = P_cond(x[1] | x[0])*P_given(x[0]), where

        ``p_cond(x[1] | x[0])`` is defined by ``dmap``, whose keys have data type ``T0`` and whose values are
        child distributions compatible with data type ``T1``.

        P_given(x[0]) is defined as the given distribution. If None is provided, it is assumed that P_given(x[0]) = 1
        for all x[0].

        default_dist defines the distribution for the case where x[0] is not a key in dmap. That is, x[0] is not in the
        support of P_cond(X_1 | X_0). If None is provided we assume that P_cond(X1 | X0) = 0, for all X0 not in dmap.

        Args:
            dmap Union[Dict[Any, SequenceEncodableProbabilityDistribution],
                List[SequenceEncodableProbabilityDistribution]]): Used to create dictionary of
                SequenceEncodableProbabilityDistribution objects. Type T0 is inferred to be type of dmap keys if dict,
                else the T0 is inferred to integer.
            default_dist (Optional[SequenceEncodableProbabilityDistribution]): Branch used when ``x[0]`` is not a
                key in ``dmap``.
            given_dist (Optional[SequenceEncodableProbabilityDistribution]): Marginal distribution over the given
                value ``x[0]``.
            name (Optional[str]): Name assigned to object.
            keys (Optional[str]): All ConditionalDistribution objects with same keys value are the same distribution.

        Attributes:
            dmap (Dict[T0, SequenceEncodableProbabilityDistribution]): T0 is integer if dmap arg was list, else T0 is
                data type of the "given" or conditional.
            default_dist (SequenceEncodableProbabilityDistribution): Set to NullDistribution if None is passed as arg.
            given_dist (SequenceEncodableProbabilityDistribution): Set to NullDistribution if None is passed as arg.
            has_default (bool): True if default distribution is not NullDistribution, else False.
            has_given (bool): True if given_dist is not NullDistribution, else False.
            name (Optional[str]): Name assigned to object.
            keys (Optional[str]): All ConditionalDistribution objects with same keys value are the same distribution.

        """
        if isinstance(dmap, list):
            dmap = dict(zip(range(len(dmap)), dmap))

        self.dmap = dmap
        self.default_dist = default_dist if default_dist is not None else NullDistribution()
        self.given_dist = given_dist if given_dist is not None else NullDistribution()

        self.has_default = not supports(self.default_dist, Neutral)
        self.has_given = not supports(self.given_dist, Neutral)
        self.name = name
        self.keys = keys
        self.set_prior(prior)

    def get_prior(self) -> tuple[dict[Any, Any], Any, Any]:
        """Return the joint prior as ``(per_branch_priors, default_prior, given_prior)``.

        ``per_branch_priors`` is a dict keyed like ``dmap`` of each conditional branch's prior.
        """
        branch = {k: v.get_prior() for k, v in self.dmap.items()}
        return branch, self.default_dist.get_prior(), self.given_dist.get_prior()

    def set_prior(self, prior: tuple[dict[Any, Any], Any, Any] | None) -> None:
        """Distribute per-branch priors to the conditional branches, default, and given distributions.

        ``prior=None`` is a no-op (children keep their existing priors, leaving the MLE path
        byte-identical). Otherwise ``prior`` is ``(per_branch_priors, default_prior, given_prior)``: each
        ``dmap`` branch prior is pushed to the matching child via ``set_prior``, and the default/given
        priors are pushed to ``default_dist``/``given_dist``.
        """
        if prior is None:
            return
        branch, default_prior, given_prior = prior
        for k, p in branch.items():
            if k in self.dmap:
                self.dmap[k].set_prior(p)
        self.default_dist.set_prior(default_prior)
        self.given_dist.set_prior(given_prior)

    def expected_log_density(self, x: tuple[T0, T1]) -> float:
        """Prior-expected log-density: selected branch ``expected_log_density`` + given term.

        Mirrors ``log_density``: unmatched conditioning values with no default score ``-inf``.
        """
        if self.has_default:
            rv = self.dmap.get(x[0], self.default_dist).expected_log_density(x[1])
        else:
            if x[0] in self.dmap:
                rv = self.dmap[x[0]].expected_log_density(x[1])
            else:
                return -np.inf

        rv += self.given_dist.expected_log_density(x[0])
        return rv

    def seq_expected_log_density(self, x: E0) -> np.ndarray:
        """Vectorized prior-expected log-density mirroring ``seq_log_density``."""
        sz, cond_vals, eobs_vals, idx_vals, given_enc = x
        rv = np.zeros(sz, dtype=float)

        for i in range(len(cond_vals)):
            if self.has_default:
                rv[idx_vals[i]] = self.dmap.get(cond_vals[i], self.default_dist).seq_expected_log_density(eobs_vals[i])
            else:
                if cond_vals[i] in self.dmap:
                    rv[idx_vals[i]] += self.dmap[cond_vals[i]].seq_expected_log_density(eobs_vals[i])
                else:
                    rv[idx_vals[i]] = -np.inf

        if self.has_given:
            rv += self.given_dist.seq_expected_log_density(given_enc)

        return rv

    def compute_capabilities(self):
        """Declare generated-compute support joined across branch, default, and given models."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = list(self.dmap.values())
        if self.has_default:
            children.append(self.default_dist)
        if self.has_given:
            children.append(self.given_dist)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(tuple(children)), kernel_status="generic")

    def compute_declaration(self):
        """Return the generated-compute declaration for the conditional distribution."""
        from mixle.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        children = []
        roles = []
        for key, dist in self.dmap.items():
            declaration = declaration_for(dist)
            if declaration is not None:
                children.append(declaration)
                roles.append("condition_%s" % repr(key))
        if self.has_default:
            declaration = declaration_for(self.default_dist)
            if declaration is not None:
                children.append(declaration)
                roles.append("default")
        if self.has_given:
            declaration = declaration_for(self.given_dist)
            if declaration is not None:
                children.append(declaration)
                roles.append("given")
        return DistributionDeclaration(
            name="conditional",
            distribution_type=type(self),
            parameters=(),
            statistics=(
                StatisticSpec("conditions", kind="mapping"),
                StatisticSpec("default", kind="child_stat"),
                StatisticSpec("given", kind="child_stat"),
            ),
            support="conditional_pair",
            children=tuple(children),
            child_roles=tuple(roles),
            differentiable=all(child.differentiable for child in children),
        )

    def __str__(self) -> str:
        """Return a constructor-style representation of the conditional distribution."""
        s1 = repr(self.dmap)
        s2 = repr(self.default_dist)
        s3 = repr(self.given_dist)
        s4 = repr(self.name)
        s5 = repr(self.keys)

        return "ConditionalDistribution(%s, default_dist=%s, given_dist=%s, name=%s, keys=%s)" % (s1, s2, s3, s4, s5)

    def density(self, x: tuple[T0, T1]) -> float:
        """Evaluates density of ConditionalDistribution at Tuple x.

        Calls log_density() and returns the exponentiated result. See log_density() for details.

        Args:
            x (Tuple[T0, T1]): T0 data type much match keys of dmap, T1 much match value of dmap distribution for key
                value.

        Returns:
            Density of ConditionalDistribution at Tuple x

        """
        return math.exp(self.log_density(x))

    def log_density(self, x: tuple[T0, T1]) -> float:
        """Evaluate log-density of ConditionalDistribution at Tuple x.

        Log-density:
            log(P(x)) = log(P_cond(x[1] | x[0])) + log(P_given(x[0])), where
            log(P_cond(x[1] | x[0])) is defined from dmap, and log(P_given(x[0])) is defined from given_dist.

        Note: Log-density is evaluated to -np.inf, if x[0] not in dmap and default_dist is NullDistribution().

        Args:
            x (Tuple[T0, T1]): T0 data type much match keys of dmap, T1 much match value of dmap distribution for key
                value.

        Returns:
            Log-density of ConditionalDistribution at Tuple x.

        """
        if self.has_default:
            rv = self.dmap.get(x[0], self.default_dist).log_density(x[1])
        else:
            if x[0] in self.dmap:
                rv = self.dmap[x[0]].log_density(x[1])
            else:
                return -np.inf

        rv += self.given_dist.log_density(x[0])

        return rv

    def seq_log_density(self, x: E0) -> np.ndarray:
        """Arkouda vectorized evaluation of the log-density on sequence encoded data x.

        x Tuple of length 5:
            x[0] (int): length of x (i.e. total observations).
            x[1] (Tuple[T0]): Unique conditional values in data.
            x[2] (Tuple[E0,...]): Tuple of encoded data sequences for each given key.
            x[3] (Tuple[ak.pdarray,...]): Tuple containing idxs for observation corresponding to x[1] values.
            x[4] (Optional[Encoded[T0]]): If the given_encoder is not the NullDataEncoder, the
                observed conditional values of data type T0 are sequence encoded by given_encoder. Else return None.

        Args:
            x: See above for details.

        Returns:
            Numpy array of log-density evaluated at each encoded data point.

        """
        sz, cond_vals, eobs_vals, idx_vals, given_enc = x
        rv = np.zeros(sz, dtype=float)

        for i in range(len(cond_vals)):
            if self.has_default:
                rv[idx_vals[i]] = self.dmap.get(cond_vals[i], self.default_dist).seq_log_density(eobs_vals[i])
            else:
                if cond_vals[i] in self.dmap:
                    rv[idx_vals[i]] += self.dmap[cond_vals[i]].seq_log_density(eobs_vals[i])
                else:
                    rv[idx_vals[i]] = -np.inf

        if self.has_given:
            rv += self.given_dist.seq_log_density(given_enc)

        return rv

    def backend_seq_log_density(self, x: E0, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for grouped conditional encodings."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, cond_vals, eobs_vals, idx_vals, given_enc = x
        rv = engine.zeros(sz)
        for i in range(len(cond_vals)):
            key = cond_vals[i]
            idx = engine.asarray(idx_vals[i])
            if key in self.dmap:
                scores = backend_seq_log_density(self.dmap[key], eobs_vals[i], engine)
            elif self.has_default:
                scores = backend_seq_log_density(self.default_dist, eobs_vals[i], engine)
            else:
                scores = engine.zeros(len(idx_vals[i])) + float("-inf")
            rv = engine.index_add(rv, idx, scores)

        if self.has_given and given_enc is not None:
            rv = rv + backend_seq_log_density(self.given_dist, given_enc, engine)

        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["ConditionalDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked child routes for homogeneous conditional mixtures."""
        from mixle.stats.compute.stacked import stacked_component_params

        keys = tuple(dists[0].dmap.keys())
        has_default = bool(dists[0].has_default)
        has_given = bool(dists[0].has_given)
        if any(
            tuple(dist.dmap.keys()) != keys
            or bool(dist.has_default) != has_default
            or bool(dist.has_given) != has_given
            for dist in dists
        ):
            raise ValueError("Stacked ConditionalDistribution components require matching key/default/given layout.")

        routes = {}
        for key in keys:
            child_dists = [dist.dmap[key] for dist in dists]
            try:
                routes[key] = stacked_component_params(child_dists, engine)
            except ValueError as exc:
                raise ValueError(
                    "Conditional key %s child %s is not stackable: %s" % (repr(key), type(child_dists[0]).__name__, exc)
                )

        default_route = None
        if has_default:
            try:
                default_route = stacked_component_params([dist.default_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "Conditional default child %s is not stackable: %s" % (type(dists[0].default_dist).__name__, exc)
                )

        given_route = None
        if has_given:
            try:
                given_route = stacked_component_params([dist.given_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "Conditional given child %s is not stackable: %s" % (type(dists[0].given_dist).__name__, exc)
                )

        return {
            "routes": routes,
            "default_route": default_route,
            "given_route": given_route,
            "has_default": has_default,
            "has_given": has_given,
            "keys": keys,
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: E0, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of conditional log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        sz, cond_vals, eobs_vals, idx_vals, given_enc = x
        rv = engine.zeros((sz, int(params["num_components"])))
        for i in range(len(cond_vals)):
            key = cond_vals[i]
            idx = engine.asarray(idx_vals[i])
            route = params["routes"].get(key, params["default_route"])
            if route is None:
                scores = engine.zeros((len(idx_vals[i]), int(params["num_components"]))) + float("-inf")
            else:
                scores = stacked_component_log_density(eobs_vals[i], route, engine)
            rv = engine.index_add(rv, idx, scores)

        if params["given_route"] is not None and given_enc is not None:
            rv = rv + stacked_component_log_density(given_enc, params["given_route"], engine)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: E0, weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[dict[Any, Any], Any, Any]:
        """Return per-component legacy conditional sufficient statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        sz, cond_vals, eobs_vals, idx_vals, given_enc = x
        ww = engine.asarray(weights)
        num_components = int(tuple(getattr(ww, "shape", (0, 0)))[1])
        outer_estimators = tuple(getattr(estimator, "estimators", ()))
        group_pos = {key: pos for pos, key in enumerate(cond_vals)}

        per_key_stats: dict[Any, tuple[Any, ...]] = {}
        for key, route in params["routes"].items():
            component_estimators = tuple(
                getattr(component_est, "estimator_map", {}).get(key) for component_est in outer_estimators
            )
            pos = group_pos.get(key)
            if pos is None:
                per_key_stats[key] = _conditional_zero_stats(component_estimators, num_components)
                continue
            child_estimator = (
                StackedEstimatorView(component_estimators) if len(component_estimators) == num_components else None
            )
            child_weights = ww[engine.asarray(idx_vals[pos])]
            child_stats = stacked_component_sufficient_statistics(
                eobs_vals[pos], child_weights, route, engine, child_estimator
            )
            per_key_stats[key] = unstack_component_stats(child_stats, num_components)

        default_stats = None
        if params["default_route"] is not None:
            default_estimators = tuple(
                getattr(component_est, "default_estimator", None) for component_est in outer_estimators
            )
            for pos, key in enumerate(cond_vals):
                if key in params["routes"]:
                    continue
                child_estimator = (
                    StackedEstimatorView(default_estimators) if len(default_estimators) == num_components else None
                )
                child_weights = ww[engine.asarray(idx_vals[pos])]
                child_stats = stacked_component_sufficient_statistics(
                    eobs_vals[pos], child_weights, params["default_route"], engine, child_estimator
                )
                default_stats = _conditional_add_stats(default_stats, child_stats, engine)
            if default_stats is None:
                default_by_component = _conditional_zero_stats(default_estimators, num_components)
            else:
                default_by_component = unstack_component_stats(default_stats, num_components)
        else:
            default_by_component = tuple(None for _ in range(num_components))

        if params["given_route"] is not None and given_enc is not None:
            given_estimators = tuple(
                getattr(component_est, "given_estimator", None) for component_est in outer_estimators
            )
            given_estimator = (
                StackedEstimatorView(given_estimators) if len(given_estimators) == num_components else None
            )
            given_stats = stacked_component_sufficient_statistics(
                given_enc, ww, params["given_route"], engine, given_estimator
            )
            given_by_component = unstack_component_stats(given_stats, num_components)
        else:
            given_by_component = tuple(None for _ in range(num_components))

        return tuple(
            (
                {key: per_key_stats[key][component] for key in params["keys"]},
                default_by_component[component],
                given_by_component[component],
            )
            for component in range(num_components)
        )

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from mixle.stats.compute.gradient import ConditionalGradientFitState

        dmap = {key: recurse(child, engine, torch, leaves) for key, child in self.dmap.items()}
        default_child = recurse(self.default_dist, engine, torch, leaves) if self.has_default else None
        given_child = recurse(self.given_dist, engine, torch, leaves) if self.has_given else None
        return ConditionalGradientFitState(self, dmap, default_child, given_child)

    def sampler(self, seed: int | None = None) -> "ConditionalDistributionSampler":
        """Create a sampler for this conditional distribution.

        Args:
            seed (Optional[int]): Seed for the sampler's random number generator.

        Returns:
            ConditionalDistributionSampler configured from this distribution.

        """
        return ConditionalDistributionSampler(self, seed=seed)

    def estimator(self, pseudo_count: float | None = None) -> "ConditionalDistributionEstimator":
        """Create an estimator initialized from this conditional distribution.

        Used to estimate a ConditionalDistribution from data observations.

        Args:
            pseudo_count (Optional[float]): Used to inflate the sufficient statistics of ConditionalDistribution.

        Returns:
            ConditionalDistributionEstimator with estimators for the branch, default, and given distributions.

        """
        est_map = {k: v.estimator(pseudo_count) for k, v in self.dmap.items()}
        default_est = self.default_dist.estimator(pseudo_count)
        given_est = self.given_dist.estimator(pseudo_count)

        return ConditionalDistributionEstimator(
            estimator_map=est_map,
            default_estimator=default_est,
            given_estimator=given_est,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "ConditionalDistributionDataEncoder":
        """Create a data encoder for conditional-distribution observations."""
        encoder_map = {k: v.dist_to_encoder() for k, v in self.dmap.items()}
        default_encoder = NullDataEncoder() if not self.has_default else self.default_dist.dist_to_encoder()
        given_encoder = NullDataEncoder() if not self.has_given else self.given_dist.dist_to_encoder()

        return ConditionalDistributionDataEncoder(
            encoder_map=encoder_map, default_encoder=default_encoder, given_encoder=given_encoder
        )

    def enumerator(self) -> "ConditionalDistributionEnumerator":
        """Creates a ConditionalDistributionEnumerator iterating (given, value) pairs in
        descending joint probability order.

        Requires an enumerable given distribution and enumerable conditional distributions;
        raises EnumerationError otherwise.

        Returns:
            ConditionalDistributionEnumerator object.

        """
        return ConditionalDistributionEnumerator(self)


class ConditionalDistributionEnumerator(DistributionEnumerator):
    """Enumerates (given, value) pairs of a ConditionalDistribution in descending joint
    probability order."""

    def __init__(self, dist: ConditionalDistribution) -> None:
        """Enumerates (given, value) pairs in descending joint probability order.

        The joint log-density log P(x0, x1) = log P_cond(x1|x0) + log P_given(x0) factors over a
        per-given-value stream of pairs, so this is a lazy k-way merge of the per-x0 streams.
        Streams are instantiated in descending P_given(x0) order; since the head of stream x0 is
        bounded above by log P_given(x0), a buffered head is emitted only once it beats the next
        un-instantiated stream's bound, which makes the global order correct.

        Raises EnumerationError if the given distribution is unspecified (NullDistribution) or
        not enumerable, or if any conditional (or the default, when present) is not enumerable.
        Given values whose conditional has zero mass everywhere (no dmap entry and no default)
        contribute no pairs.

        Args:
            dist (ConditionalDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)

        if not dist.has_given:
            raise EnumerationError(
                dist,
                reason="the given distribution is unspecified "
                "(NullDistribution), so the support over (given, value) "
                "pairs is not enumerable",
            )

        self._given_stream = BufferedStream(child_enumerator(dist.given_dist, "ConditionalDistribution.given_dist"))

        # Fail fast: build one enumerator per conditional up front. Each given value appears at
        # most once in the given enumeration, so each stored enumerator is consumed at most once.
        self._cond_enums = {
            k: child_enumerator(v, "ConditionalDistribution.dmap[%s]" % repr(k)) for k, v in dist.dmap.items()
        }
        if dist.has_default:
            child_enumerator(dist.default_dist, "ConditionalDistribution.default_dist")

        self._next_rank = 0
        self._counter = itertools.count()
        self._heap: list[tuple[float, int, int]] = []  # (-head_lp, counter, stream id)
        self._heads: dict[int, tuple[Any, float]] = {}
        self._streams: dict[int, Iterator[tuple[Any, float]]] = {}

    def _make_stream(self, x0: Any, lp0: float) -> Iterator[tuple[Any, float]] | None:
        """Return a sorted stream of ((x0, x1), lp0 + lp1) pairs, or None when x0 has no mass."""
        if x0 in self._cond_enums:
            child = self._cond_enums.pop(x0)
        elif self.dist.has_default:
            child = child_enumerator(self.dist.default_dist, "ConditionalDistribution.default_dist")
        else:
            return None
        return (((x0, x1), lp0 + lp1) for x1, lp1 in child)

    def _pop(self) -> tuple[Any, float]:
        """Emit the best instantiated head and advance its stream."""
        _, _, sid = heapq.heappop(self._heap)
        value, lp = self._heads.pop(sid)
        try:
            nxt = next(self._streams[sid])
            self._heads[sid] = nxt
            heapq.heappush(self._heap, (-nxt[1], next(self._counter), sid))
        except StopIteration:
            del self._streams[sid]
        return (value, lp)

    def __next__(self) -> tuple[Any, float]:
        while True:
            frontier = self._given_stream.get(self._next_rank)
            if frontier is None:
                if self._heap:
                    return self._pop()
                raise StopIteration
            if self._heap and -self._heap[0][0] >= frontier[1]:
                return self._pop()
            x0, lp0 = frontier
            sid = self._next_rank
            self._next_rank += 1
            stream = self._make_stream(x0, lp0)
            if stream is None:
                continue
            try:
                head = next(stream)
            except StopIteration:
                continue
            self._streams[sid] = stream
            self._heads[sid] = head
            heapq.heappush(self._heap, (-head[1], next(self._counter), sid))


class ConditionalDistributionSampler(ConditionalSampler, DistributionSampler):
    """ConditionalDistributionSampler draws (given, value) pairs from a ConditionalDistribution,
    or values conditioned on a fixed given value via sample_given()."""

    def __init__(self, dist: ConditionalDistribution, seed: int | None = None) -> None:
        """Create a sampler for direct or given-conditioned draws from a conditional distribution.

        Args:
            dist (ConditionalDistribution): Conditional distribution to draw samples from.
            seed (Optional[int]): Used to set the seed of random number generator used in sampling.

        Attributes:
            dist (ConditionalDistribution): Conditional distribution to draw samples from.
            default_sampler (DistributionSampler): Sampler for ``default_dist`` of
                ConditionalDistribution.
            has_default_sampler (bool): True if default sampler is not NullDistribution, else False.
            given_sampler (DistributionSampler): Sampler for ``given_dist`` of
                ConditionalDistribution.
            has_given_sampler (bool): True if given sampler is not NullDistribution, else False.
            samplers (Dict[T0,DistributionSampler]): Dictionary of samplers for sampling from ConditionalDistribution,
                given a key of data type T0. Note returns List[T1] or T1.

        """
        self.dist = dist
        rng = np.random.RandomState(seed)

        loc_seed = rng.randint(0, maxrandint)

        self.has_default_sampler = dist.has_default
        self.default_sampler = dist.default_dist.sampler(loc_seed)

        loc_seed = rng.randint(0, maxrandint)
        self.given_sampler = dist.given_dist.sampler(loc_seed)
        self.has_given_sampler = not supports(dist.given_dist, Neutral)

        self.samplers = {k: u.sampler(rng.randint(0, maxrandint)) for k, u in self.dist.dmap.items()}

    def single_sample(self) -> tuple[Any, Any]:
        """Draw one ``(given, conditional)`` pair from the conditional distribution.

        The first element is sampled from the given distribution; the second is sampled from the branch selected by
        that value, or from the default branch when the value is not present in ``dmap``.

        Returns:
            Tuple[T0, T1] as defined from dmap and given_distribution types in dist (ConditionalDistribution instance).

        """
        x0 = self.given_sampler.sample()
        if x0 in self.samplers:
            x1 = self.samplers[x0].sample()
        else:
            x1 = self.default_sampler.sample()
        return x0, x1

    def sample(self, size: int | None = None, *, batched: bool = True) -> tuple[Any, Any] | list[tuple[Any, Any]]:
        """Sample 'size' independent samples from ConditionalDistribution.

        Sequence of 'size' calls to single_sample(). If size is None, size is taken to be 1.

        Data type returned is a Tuple[T0, T1], where T0 and T1 are the respective data types of the given_dist and
        dmap defined in the CompositeDistribution instance 'dist'.

        Args:
            size (Optional[int]): Number of independent samples to draw from ConditionalDistribution.

        Returns:
            A list of 'size' tuples of Tuple[T0, T1], or a single Tuple[T0, T1].

        """

        if size is None:
            return self.single_sample()
        else:
            return [self.single_sample() for i in range(size)]

    def sample_given(self, x: T0) -> Any:
        """Sample from the conditional distribution for a supplied given value.

        Return data type T1 as defined for dictionary of ConditionalDistribution instance.

        Args:
            x (T0): Value of given/conditional value for ConditionalDistribution.

        Returns:
            Single sample from ConditionalDistribution object 'dist.dmap' given x.

        """
        if x in self.samplers:
            return self.samplers[x].sample()

        elif self.has_default_sampler:
            return self.default_sampler.sample()

        else:
            raise ValueError("Conditional default distribution unspecified.")


class ConditionalDistributionAccumulator(SequenceEncodableStatisticAccumulator):
    """ConditionalDistributionAccumulator accumulates sufficient statistics for each conditional
    distribution, the default distribution, and the given distribution."""

    def __init__(
        self,
        accumulator_map: dict[T0, SequenceEncodableStatisticAccumulator],
        default_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        given_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: str | None = None,
    ) -> None:
        """ConditionalDistributionAccumulator used for aggregating sufficient statistics of ConditionalDistribution.

        The sufficient statistics are defined through the accumulator_map dictionary, which is a dictionary with keys
        of data type T0 for the given type. Each value of the dict contains a SequenceEncodableStatisticAccumulator for
        accumulating respective sufficient statistics.

        The sufficient statistics for the default distribution are stored in ``default_accumulator``. If
        ``default_accumulator`` is None, it is set to ``NullAccumulator``.

        The sufficient statistics for ``given_distribution`` are stored in ``given_accumulator``. This is set to
        ``NullAccumulator`` if no ``given_accumulator`` is specified.

        Args:
            accumulator_map (Dict[T0, SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of each
                conditional distribution for a given key value of data type T0.
            default_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of
                distribution for case where key not in accumulator_map.
            given_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of
                given distribution if provided.
            keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

        Attributes:
            accumulator_map (Dict[T0, SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of each
                conditional distribution for a given key value of data type T0.
            default_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of
                distribution for case where key not in accumulator_map.
            given_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of
                given distribution if provided.
            has_default (bool): True if default_accumulator is not NullAccumulator.
            has_given (bool): True if given_accumulator is not NullAccumulator.
            key (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

            _init_rng (bool): False unless a single call to initialize or seq_initialize has been made.
            _acc_rng (Optional[Dict[T0, RandomState]]): Used to seed RandomState calls of accumulator_map.
            _default_rng (Optional[RandomState]): Used to seed RandomState calls for default accumulator initialization.
            _given_rng (Optional[RandomState]): Used to seed RandomState calls for given accumulator initialization.

        """
        self.accumulator_map = accumulator_map
        self.default_accumulator = default_accumulator if default_accumulator is not None else NullAccumulator()
        self.given_accumulator = given_accumulator if given_accumulator is not None else NullAccumulator()

        self.has_default = not supports(default_accumulator, Neutral)
        self.has_given = not supports(given_accumulator, Neutral)

        self.keys = keys

        # Seeds for initializers.
        self._init_rng = False
        self._acc_rng: dict[T0, RandomState] | None = None
        self._default_rng: RandomState | None = None
        self._given_rng: RandomState | None = None

    def update(self, x: tuple[T0, T1], weight: float, estimate: Optional["ConditionalDistribution"]) -> None:
        """Updates sufficient statistics of ConditionalDistributionAccumulator for one weighted observation x.

        Single weighted observation used to update the sufficient statistics of ConditionalDistributionAccumulator.

        Args:
            x (Tuple[T0, T1]): Tuple observation of ConditionalDistribution.
            weight (float): Weight for observation.
            estimate (Optional['ConditionalDistribution']): Sufficient statistics from ConditionalDistribution
                are aggregated with weighted observation x.

        Returns:
            None.

        """

        if x[0] in self.accumulator_map:
            if estimate is None:
                self.accumulator_map[x[0]].update(x[1], weight, None)
            else:
                self.accumulator_map[x[0]].update(x[1], weight, estimate.dmap[x[0]])
        else:
            if self.has_default:
                if estimate is None:
                    self.default_accumulator.update(x[1], weight, None)
                else:
                    self.default_accumulator.update(x[1], weight, estimate.default_dist)

        if self.has_given:
            if estimate is None:
                self.given_accumulator.update(x[0], weight, None)
            else:
                self.given_accumulator.update(x[0], weight, estimate.given_dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initializes protected rng members for first call to initialize or seq_initialize."""
        self._acc_rng = dict()
        for acc_key in self.accumulator_map.keys():
            self._acc_rng[acc_key] = RandomState(seed=rng.randint(2**31))

        self._default_rng = RandomState(seed=rng.randint(2**31))
        self._given_rng = RandomState(seed=rng.randint(2**31))

    def initialize(self, x: tuple[T0, T1], weight: float, rng: RandomState) -> None:
        """Initialize ConditionalDistributionAccumulator with single weighted observation.

        Note: _rng_initialize is called if _init_rng is False. This allows consistency between seq_initialize and
        initialize.

        Args:
            x (Tuple[T0, T1]): Tuple observation of ConditionalDistribution.
            weight (float): Weight for observation.
            rng (RandomState): RandomState used to set seed in initialize calls to member accumulators.

        Returns:
            None

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if x[0] in self.accumulator_map:
            self.accumulator_map[x[0]].initialize(x[1], weight, self._acc_rng[x[0]])
        else:
            if self.has_default:
                self.default_accumulator.initialize(x[1], weight, self._default_rng)

        if self.has_given:
            self.given_accumulator.initialize(x[0], weight, self._given_rng)

    def seq_initialize(self, x: E0, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialize ConditionalDistributionAccumulator for a sequence encoded x.

        Input x must be an encoded sequence produces from ConditionalDistributionDataEncoder.seq_encode() called on
        a list iid sequence of ConditionalDistribution observations.

        Calls seq_initialize on accumulator_map, default_accumulator, and given_accumulator.

        Note: _rng_initialize is called if _init_rng is False. This allows consistency between seq_initialize and
        initialize.

        E Tuple of length 5:
            E[0] (int): length of x (i.e. total observations).
            E[1] (Tuple[T0]): Unique conditional values in data.
            E[2] (Tuple[Encoded[T1]): Tuple of sequence encoded data of type T1 encoded by
                encoder_map[key] or default_encoder if key not in default_encoder and default_encoder is not
                the NullDataEncoder.
            E[3] (Tuple[np.ndarray,...]): Tuple of length equal to the number of unique conditional
                values encountered in the data. Each entry contains a numpy array for the indices of x that correspond
                to a unique conditional value.
            E[4] (Optional[Encoded[T0]]): If the given_encoder is not the NullDataEncoder, the
                observed conditional values of data type T0 are sequence encoded by given_encoder. Else return None.

        Args:
            x: See description above for details.
            weights (ndarray): Numpy array of floats containing weights for each observation.
            rng (RandomState): RandomState used to set seed in initialize calls to member accumulators.

        Returns:
            None.

        """
        sz, cond_vals, eobs_vals, idx_vals, given_enc = x

        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_initialize(
                    eobs_vals[i], weights[idx_vals[i]], self._acc_rng[cond_vals[i]]
                )
            else:
                if self.has_default:
                    self.default_accumulator.seq_initialize(eobs_vals[i], weights[idx_vals[i]], self._default_rng)

        if self.has_given:
            self.given_accumulator.seq_initialize(given_enc, weights, self._given_rng)

    def seq_update(self, x: E0, weights: np.ndarray, estimate: "ConditionalDistribution") -> None:
        """Vectorized update of sufficient statistics of ConditionalDistributionAccumulator for a sequence encoded
        x.

        Input x must be an encoded sequence produces from ConditionalDistributionDataEncoder.seq_encode() called on
        a list iid sequence of ConditionalDistribution observations.

        Calls seq_update on accumulator_map, default_accumulator, and given_accumualtor.

        E Tuple of length 5:
            E[0] (int): length of x (i.e. total observations).
            E[1] (Tuple[T0]): Unique conditional values in data.
            E[2] (Tuple[Encoded[T1]): Tuple of sequence encoded data of type T1 encoded by
                encoder_map[key] or default_encoder if key not in default_encoder and default_encoder is not
                the NullDataEncoder.
            E[3] (Tuple[np.ndarray,...]): Tuple of length equal to the number of unique conditional
                values encountered in the data. Each entry contains a numpy array for the indices of x that correspond
                to a unique conditional value.
            E[4] (Optional[Encoded[T0]]): If the given_encoder is not the NullDataEncoder, the
                observed conditional values of data type T0 are sequence encoded by given_encoder. Else return None.

        Args:
            x: See description above for details.
            weights (ndarray): Numpy array of floats containing weights for each observation.
            estimate (Optional['ConditionalDistribution']): Sufficient statistics from ConditionalDistribution
                are used in merged with aggregated statistics from input x.

        Returns:
            None.

        """
        sz, cond_vals, eobs_vals, idx_vals, given_enc = x

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_update(
                    eobs_vals[i], weights[idx_vals[i]], estimate.dmap[cond_vals[i]]
                )
            else:
                if self.has_default:
                    if estimate is None:
                        self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], None)
                    else:
                        self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], estimate.default_dist)

        if self.has_given:
            if estimate is None:
                self.given_accumulator.seq_update(given_enc, weights, None)
            else:
                self.given_accumulator.seq_update(given_enc, weights, estimate.given_dist)

    def seq_update_engine(self, x: E0, weights: Any, estimate: "ConditionalDistribution", engine: Any) -> None:
        """Engine-resident E-step: per-conditional-value subgroup weights are gathered on the active
        engine and the matching child accumulators (and the given accumulator) are routed through
        the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        sz, cond_vals, eobs_vals, idx_vals, given_enc = x
        w_eng = engine.asarray(weights)

        for i in range(len(cond_vals)):
            wi = w_eng[np.asarray(idx_vals[i], dtype=np.int64)]
            if cond_vals[i] in self.accumulator_map:
                child_seq_update(
                    self.accumulator_map[cond_vals[i]],
                    eobs_vals[i],
                    wi,
                    estimate.dmap[cond_vals[i]] if estimate is not None else None,
                    engine,
                )
            elif self.has_default:
                child_seq_update(
                    self.default_accumulator,
                    eobs_vals[i],
                    wi,
                    None if estimate is None else estimate.default_dist,
                    engine,
                )

        if self.has_given:
            child_seq_update(
                self.given_accumulator, given_enc, w_eng, None if estimate is None else estimate.given_dist, engine
            )

    def combine(self, suff_stat: tuple[dict[T0, SS0], SS1 | None, SS2 | None]) -> "ConditionalDistributionAccumulator":
        """Aggregate sufficient statistics (suff_stat) with sufficient statistics of ConditionalDistributionAccumulator
            instance.

        Args:
            suff_stat: Tuple of length 3 containing the sufficient statistics of the conditional distributions,
                default distribution, and given distribution.

        Returns:
            ConditionalDistributionAccumulator with aggregated sufficient statistics.

        """
        for k, v in suff_stat[0].items():
            if k in self.accumulator_map:
                self.accumulator_map[k].combine(v)
            else:
                self.accumulator_map[k].from_value(v)

        if self.has_default and suff_stat[1] is not None:
            self.default_accumulator.combine(suff_stat[1])

        if self.has_given and suff_stat[2] is not None:
            self.given_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> tuple[dict[Any, Any], Any | None, Any | None]:
        """Get sufficient statistics of CompositeDistributionAccumulator."""
        rv3 = self.given_accumulator.value()
        rv2 = self.default_accumulator.value()
        rv1 = {k: v.value() for k, v in self.accumulator_map.items()}

        return rv1, rv2, rv3

    def from_value(self, x: tuple[dict[T0, SS0], SS1 | None, SS1 | None]) -> "ConditionalDistributionAccumulator":
        """Set ConditionalDistributionAccumulator member instances to x.

        Input x must be sufficient statistic tuple compatible with ConditionalDistributionAccumulator.

        Args:
            x: Tuple of length 3 containing the sufficient statistics of ConditionalDistributionAccumulator.

        Returns:
            ConditionalDistributionAccumulator object.

        """
        for k, v in x[0].items():
            self.accumulator_map[k].from_value(v)

        if self.has_default and x[1] is not None:
            self.default_accumulator.from_value(x[1])

        if self.has_given and x[2] is not None:
            self.given_accumulator.from_value(x[2])

        return self

    def scale(self, c: float) -> "ConditionalDistributionAccumulator":
        """Scale every branch, default, and given accumulator statistic in place."""
        for accumulator in self.accumulator_map.values():
            accumulator.scale(c)
        if self.has_default:
            self.default_accumulator.scale(c)
        if self.has_given:
            self.given_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Aggregate the sufficient statistics of ConditionalDistributionAccumulator with member instance key in
            stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Key of dict are the 'keys' for
                ConditionalDistributionAccumulator that represent the same distribution.

        Returns:
            None

        """
        for k, v in self.accumulator_map.items():
            v.key_merge(stats_dict)

        if self.has_default:
            self.default_accumulator.key_merge(stats_dict)

        if self.has_given:
            self.given_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Invoke key_replace on each member SequenceEncodableStatisticAccumulator of
            ConditionalDistributionAccumulator instance.

        Args:
            stats_dict (Dict[str, Any]): Key of dict are the 'keys' for
                ConditionalDistributionAccumulator that represent the same distribution.

        Returns:
            None

        """
        for k, v in self.accumulator_map.items():
            v.key_replace(stats_dict)

        if self.has_default:
            self.default_accumulator.key_replace(stats_dict)

        if self.has_given:
            self.given_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "ConditionalDistributionDataEncoder":
        """Create a data encoder from the branch accumulator encoders."""

        encoder_map = {k: v.acc_to_encoder() for k, v in self.accumulator_map.items()}
        default_encoder = self.default_accumulator.acc_to_encoder()
        given_encoder = self.given_accumulator.acc_to_encoder()

        return ConditionalDistributionDataEncoder(
            encoder_map=encoder_map, default_encoder=default_encoder, given_encoder=given_encoder
        )


class ConditionalDistributionAccumulatorFactory(StatisticAccumulatorFactory):
    """ConditionalDistributionAccumulatorFactory creates ConditionalDistributionAccumulator
    objects from the per-key, default, and given factories."""

    def __init__(
        self,
        factory_map: dict[T0, StatisticAccumulatorFactory],
        default_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        given_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: str | None = None,
    ) -> None:
        """Create a factory for conditional-distribution accumulators.

        Args:
            factory_map (Dict[T0, StatisticAccumulatorFactory]): Dictionary of StatisticAccumulatorFactory objects for
                creating SequenceEncodableStatisticAccumulator objects in ConditionalDistributionAccumulator
            default_factory (StatisticAccumulatorFactory): Used to create SequenceEncodableStatisticAccumulator for
                default_accumulator in ConditionalDistributionAccumulator.
            given_factory (StatisticAccumulatorFactory): Used to create SequenceEncodableStatisticAccumulator for
                given_accumulator in ConditionalDistributionAccumulator.
            keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

        Attributes:
            factory_map (Dict[T0, StatisticAccumulatorFactory]): Dictionary of StatisticAccumulatorFactory objects for
                creating SequenceEncodableStatisticAccumulator objects in ConditionalDistributionAccumulator
            default_factory (StatisticAccumulatorFactory): Used to create SequenceEncodableStatisticAccumulator for
                default_accumulator in ConditionalDistributionAccumulator.
            given_factory (StatisticAccumulatorFactory): Used to create SequenceEncodableStatisticAccumulator for
                given_accumulator in ConditionalDistributionAccumulator.
            keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

        """
        self.factory_map = factory_map
        self.default_factory = default_factory
        self.given_factory = given_factory
        self.keys = keys

    def make(self) -> "ConditionalDistributionAccumulator":
        """Create a conditional accumulator from member factories.

        SequenceEncodableStatisticAccumulator objects are created for accumulator_map, default_accumulator, and
        given_accumulator in ConditionalAccumulator object.

        Returns:
            ConditionalDistributionAccumulator

        """
        acc = {k: v.make() for k, v in self.factory_map.items()}
        def_acc = self.default_factory.make()
        given_acc = self.given_factory.make()

        return ConditionalDistributionAccumulator(acc, def_acc, given_acc, self.keys)


class ConditionalDistributionEstimator(ParameterEstimator):
    """ConditionalDistributionEstimator estimates a ConditionalDistribution from aggregated
    sufficient statistics."""

    def __init__(
        self,
        estimator_map: dict[T0, ParameterEstimator],
        default_estimator: ParameterEstimator | None = NullEstimator(),
        given_estimator: ParameterEstimator | None = NullEstimator(),
        name: str | None = None,
        keys: str | None = None,
        prior: tuple[dict[Any, Any], Any, Any] | None = None,
    ) -> None:
        """Create an estimator for a conditional distribution from aggregated data.

        If None is passed for default_estimator, default_estimator is set to NullEstimator().
        If None is passed for given_estimator, given_estimator is set to NullEstimator().

        Args:
            estimator_map (Dict[T0, ParameterEstimator]):
            default_estimator (Optional[ParameterEstimator]): ParameterEstimator for default_distribution, can be None.
            given_estimator (Optional[ParameterEstimator]): ParameterEstimator for given_distribution, can be None.
            name (Optional[str]): Name the ConditionalDistributionEstimator object.
            keys (Optional[str]): ConditionalDistributionEstimator with matching 'keys' will be aggregated.

        Attributes:
            estimator_map (Dict[T0, ParameterEstimator]):
            default_estimator (ParameterEstimator): ParameterEstimator for default_distribution set to NullEstimator,
                if None is passed as arg.
            given_estimator (ParameterEstimator): ParameterEstimator for given_distribution set to NullEstimator
                if None is passed as arg.
            name (Optional[str]): Name the ConditionalDistributionEstimator object.
            keys (Optional[str]): ConditionalDistributionEstimator with matching 'keys' will be aggregated.

        """
        self.estimator_map = estimator_map
        self.default_estimator = default_estimator if default_estimator is not None else NullEstimator()
        self.keys = keys
        self.given_estimator = given_estimator if given_estimator is not None else NullEstimator()
        self.name = name
        self.set_prior(prior)

    def get_prior(self) -> tuple[dict[Any, Any], Any, Any]:
        """Return the joint prior as ``(per_branch_priors, default_prior, given_prior)`` from child estimators."""
        branch = {k: v.get_prior() for k, v in self.estimator_map.items()}
        return branch, self.default_estimator.get_prior(), self.given_estimator.get_prior()

    def set_prior(self, prior: tuple[dict[Any, Any], Any, Any] | None) -> None:
        """Distribute per-branch priors to the child estimators (branches, default, given).

        ``prior=None`` is a no-op. Each branch prior is pushed to the matching estimator via
        ``set_prior``; default/given priors go to the default/given estimators.
        """
        if prior is None:
            return
        branch, default_prior, given_prior = prior
        for k, p in branch.items():
            if k in self.estimator_map:
                _distribute_child_prior(self.estimator_map[k], p)
        _distribute_child_prior(self.default_estimator, default_prior)
        _distribute_child_prior(self.given_estimator, given_prior)

    def model_log_density(self, model: "ConditionalDistribution") -> float:
        """Sum each branch's estimator ``model_log_density`` plus the default and given terms."""
        rv = 0.0
        for k, est in self.estimator_map.items():
            if k in model.dmap:
                rv += est.model_log_density(model.dmap[k])
        rv += self.default_estimator.model_log_density(model.default_dist)
        rv += self.given_estimator.model_log_density(model.given_dist)
        return rv

    def accumulator_factory(self) -> "ConditionalDistributionAccumulatorFactory":
        """Return an accumulator factory for the given, branch, and default estimators."""
        emap_items = {k: v.accumulator_factory() for k, v in self.estimator_map.items()}
        def_factory = self.default_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()

        return ConditionalDistributionAccumulatorFactory(emap_items, def_factory, given_factory, self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[dict[T0, SS0], SS1 | None, SS2 | None]
    ) -> "ConditionalDistribution":
        """Estimate a ConditionalDistribution from aggregated data.

        Calls the estimate() member function of each ParameterEstimator instance for estimator_map, default_estimator,
        and given_estimator.

        Input suff_stat if a Tuple of size three containing sufficient statistics compatible with each respective
        ParameterEstimator. Entry one of the Tuple must be a dict with keys of data type T0, matching the data type
        for the given distribution.

        Return the conditional distribution estimated from the sufficient statistics in ``suff_stat``.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency.
            suff_stat: See description above.

        Returns:
            ConditionalDistribution object.

        """
        if not isinstance(suff_stat, (tuple, list)) or len(suff_stat) != 3:
            raise ContractError(
                "ConditionalDistributionEstimator.estimate(suff_stat)",
                "a 3-tuple (dist_map_suff_stats, default_suff_stat, given_suff_stat)",
                "%s%s"
                % (
                    type(suff_stat).__name__,
                    " of length %d" % len(suff_stat) if isinstance(suff_stat, (tuple, list)) else "",
                ),
                "pass the 3-tuple produced by ConditionalDistributionAccumulator.value(), not a bare "
                "component sufficient statistic.",
            )
        if not isinstance(suff_stat[0], dict):
            raise ContractError(
                "ConditionalDistributionEstimator.estimate(suff_stat[0])",
                "a dict mapping each conditioning value to its sufficient statistic",
                "%s" % type(suff_stat[0]).__name__,
                "suff_stat[0] must be the dict-of-sufficient-statistics produced by "
                "ConditionalDistributionAccumulator.value(), keyed by conditioning value.",
            )

        try:
            default_dist = self.default_estimator.estimate(None, suff_stat[1])
        except ContractError as e:
            raise prefix_contract_error("ConditionalDistribution.default_dist", e) from None
        try:
            given_dist = self.given_estimator.estimate(None, suff_stat[2])
        except ContractError as e:
            raise prefix_contract_error("ConditionalDistribution.given_dist", e) from None

        dist_map = {}
        for k, v in suff_stat[0].items():
            if k not in self.estimator_map:
                raise ContractError(
                    "ConditionalDistributionEstimator.estimator_map[%r]" % (k,),
                    "a conditioning value present in estimator_map",
                    "conditioning value %r, not in estimator_map" % (k,),
                    "suff_stat[0] carries a key not covered by this estimator's estimator_map -- "
                    "check the accumulator/estimator pairing, or add %r to estimator_map." % (k,),
                )
            try:
                dist_map[k] = self.estimator_map[k].estimate(None, v)
            except ContractError as e:
                raise prefix_contract_error("ConditionalDistribution.estimator_map[%r]" % (k,), e) from None

        return ConditionalDistribution(
            dist_map, default_dist=default_dist, given_dist=given_dist, name=self.name, keys=self.keys
        )


class ConditionalDistributionDataEncoder(DataSequenceEncoder):
    """ConditionalDistributionDataEncoder encodes sequences of (given, value) pairs, grouping the
    values by given value and delegating each group to the matching conditional encoder."""

    def __init__(
        self,
        encoder_map: dict[T0, DataSequenceEncoder],
        default_encoder: DataSequenceEncoder = NullDataEncoder(),
        given_encoder: DataSequenceEncoder = NullDataEncoder(),
    ) -> None:
        """ConditionalDistributionDataEncoder used to encode sequence of data.

        Data type should be Tuple[T0, T1] where T0 is the type of the conditional value in ConditionalDistribution.
        I.e.,
        p_mat(X1|X0), should have x_mat as type T0, and Y as type T1.

        Args:
            encoder_map (Dict[T0, DataSequenceEncoder]): Dictionary of DataSequenceEncoder objects for each conditional
                value of data type T0. Data types of the encoders must be of type T1.
            default_encoder (DataSequenceEncoder): DataSequenceEncoder compatible with data type T1.
            given_encoder ((DataSequenceEncoder): DataSequenceEncoder compatible with data type T0.

        Attributes:
            encoder_map (Dict[T0, DataSequenceEncoder]): Dictionary of DataSequenceEncoder objects for each conditional
                value of data type T0. Data types of the encoders must be of type T1.
            default_encoder (DataSequenceEncoder): DataSequenceEncoder compatible with data type T1.
            given_encoder (DataSequenceEncoder): DataSequenceEncoder compatible with data type T0.
            null_default_encoder (bool): True if default_encoder is instance of NullDataEncoder, else false.
            null_given_encoder (bool): True if default_encoder is instance of NullDataEncoder, else false.

        """
        self.encoder_map = encoder_map
        self.default_encoder = default_encoder
        self.given_encoder = given_encoder

        self.null_default_encoder = supports(self.default_encoder, Neutral)
        self.null_given_encoder = supports(self.given_encoder, Neutral)

    def __str__(self) -> str:
        """Return a constructor-style representation of the conditional encoder."""
        encoder_items = list(self.encoder_map.items())
        encoder_str = "ConditionalDataEncoder("
        for k, v in encoder_items[:-1]:
            encoder_str += str(k) + ":" + str(v) + ","
        encoder_str += str(encoder_items[-1][0]) + ":" + str(encoder_items[-1][1])

        if not self.null_default_encoder:
            encoder_str += ",default=" + str(self.default_encoder)
        else:
            encoder_str += ",default=None"

        if not self.null_given_encoder:
            encoder_str += ",given=" + str(self.given_encoder)
        else:
            encoder_str += ",given=None)"

        return encoder_str

    def __eq__(self, other) -> bool:
        """Return whether another encoder is equivalent to this encoder.

        The object must match each encoder in this ConditionalDistributionDataEncoder. That is, it must match
        encoder_map, default_encoder, and given_encoder. If any condition does not match, equality does not hold.

        Args:
            other (object): Object to be compared to instance of ConditionalDistributionDataEncoder.

        Returns:
            True if object is equivalent to instance of ConditionalDistributionDataEncoder, else False.

        """
        if not isinstance(other, ConditionalDistributionDataEncoder):
            return False
        else:
            if not self.encoder_map == other.encoder_map:
                return False

            if not self.default_encoder == other.default_encoder:
                return False

            if not self.given_encoder == other.given_encoder:
                return False

        return True

    def seq_encode(
        self, x: list[tuple[T0, T1]]
    ) -> tuple[int, tuple[Any, ...], tuple[Any, ...], tuple[np.ndarray, ...], Any | None]:
        """Encode sequence of iid observations from ConditionalDistribution for vectorized "seq_" function calls.

        Data must be a List of Tuple of two types, T0 and T1. T0 is the data type compatible with the conditional
        values of the ConditionalDistribution. T1 must be consistent with the data type of the conditional
        distributions.

        E Tuple of length 5:
            E[0] (int): length of x (i.e. total observations).
            E[1] (Tuple[T0]): Unique conditional values in data.
            E[2] (Tuple[Encoded[T1]): Tuple of sequence encoded data of type T1 encoded by
                encoder_map[key] or default_encoder if key not in default_encoder and default_encoder is not
                the NullDataEncoder.
            E[3] (Tuple[np.ndarray,...]): Tuple of length equal to the number of unique conditional
                values encountered in the data. Each entry contains a numpy array for the indices of x that correspond
                to a unique conditional value.
            E[4] (Optional[Encoded[T0]]): If the given_encoder is not the NullDataEncoder, the
                observed conditional values of data type T0 are sequence encoded by given_encoder. Else return None.

        Args:
            x (List[Tuple[T0, T1]]): List of data observations.

        Returns:
            Returns rv (see description for details)

        """
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise ContractError(
                "ConditionalDistribution.seq_encode",
                "a sequence of (given, value) pairs",
                "%s" % type(x).__name__,
                "pass a list of 2-tuples, e.g. [(given0, value0), (given1, value1), ...].",
            )

        cond_enc = dict()

        given_vals = []

        for i in range(len(x)):
            xx = x[i]
            if not isinstance(xx, (tuple, list, np.ndarray)) or len(xx) != 2:
                raise ContractError(
                    "ConditionalDistribution.seq_encode (row %d)" % i,
                    "a 2-tuple (given_value, observed_value)",
                    "%s%s"
                    % (
                        type(xx).__name__,
                        " of length %d" % len(xx) if isinstance(xx, (tuple, list, np.ndarray)) else "",
                    ),
                    "each row must be a (given, value) pair -- check row %d for a missing/extra field." % i,
                )
            given_vals.append(xx[0])
            if xx[0] not in cond_enc:
                cond_enc[xx[0]] = [[xx[1]], [i]]
            else:
                cond_enc_loc = cond_enc[xx[0]]
                cond_enc_loc[0].append(xx[1])
                cond_enc_loc[1].append(i)

        cond_enc_items = list(cond_enc.items())
        cond_vals = tuple([u[0] for u in cond_enc_items])

        eobs_vals = []
        idx_vals = []

        for u in cond_enc_items:
            field_path = "ConditionalDistribution.estimator_map[%r]" % (u[0],)
            try:
                if self.null_default_encoder:
                    if u[0] in self.encoder_map:
                        eobs_vals.append(self.encoder_map[u[0]].seq_encode(u[1][0]))
                    else:
                        # No encoder and no default for this conditioning value: append a
                        # sentinel so eobs_vals stays aligned with cond_vals/idx_vals.
                        # seq_log_density/seq_update guard on the cond key, so it is never
                        # dereferenced (the group scores -inf / is skipped).
                        eobs_vals.append(None)
                else:
                    eobs_vals.append(self.encoder_map.get(u[0], self.default_encoder).seq_encode(u[1][0]))
            except ContractError as e:
                raise prefix_contract_error(field_path, e) from None
            except (TypeError, ValueError, IndexError, KeyError) as e:
                raise ContractError(
                    field_path,
                    "values compatible with the conditional distribution registered for given=%r" % (u[0],),
                    "data that raised %s: %s" % (type(e).__name__, e),
                    "check that every value observed under given=%r matches the data type expected "
                    "by its conditional distribution." % (u[0],),
                ) from e

            idx_vals.append(np.asarray(u[1][1]))

        try:
            given_enc = self.given_encoder.seq_encode(given_vals)
        except ContractError as e:
            raise prefix_contract_error("ConditionalDistribution.given_dist", e) from None
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise ContractError(
                "ConditionalDistribution.given_dist",
                "given-values compatible with the given distribution's data type",
                "data that raised %s: %s" % (type(e).__name__, e),
                "check that every given-value matches the data type expected by given_dist (%s)." % self.given_encoder,
            ) from e

        return len(x), cond_vals, tuple(eobs_vals), tuple(idx_vals), given_enc


# --- Backward-compatible API naming aliases ---
ConditionalAccumulator = ConditionalDistributionAccumulator
ConditionalAccumulatorFactory = ConditionalDistributionAccumulatorFactory
ConditionalDataEncoder = ConditionalDistributionDataEncoder
ConditionalEnumerator = ConditionalDistributionEnumerator
ConditionalEstimator = ConditionalDistributionEstimator
