"""Create, estimate, and sample from a select distribution.

Defines the SelectDistribution, SelectSampler, SelectEstimatorAccumulator,
SelectEstimatorAccumulatorFactory, SelectEstimator, SelectDataEncoder, and SelectEnumerator
classes for use with pysparkplug.

Data type: T (any type accepted by every child distribution). The SelectDistribution routes an
observation x to one of its child distributions through a user-supplied choice function
c(x) -> {0, ..., len(dists)-1}, and evaluates the density of the selected child,

    p(x) = p_{c(x)}(x).

The choice function partitions the data space, so each child distribution is estimated only from
the observations routed to it.

"""

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)
from pysp.utils.enumeration import BufferedStream, best_first_union_max

T = TypeVar("T")


def _child_accumulator_factory(estimator: ParameterEstimator) -> StatisticAccumulatorFactory:
    """Return estimator.accumulator_factory(), falling back to the legacy camelCase name.

    Modern estimators define accumulator_factory(); a few legacy estimators only define
    accumulatorFactory(). This helper supports both so SelectEstimator works with either.

    Args:
        estimator (ParameterEstimator): Child estimator to obtain a factory from.

    Returns:
        StatisticAccumulatorFactory created by the child estimator.

    """
    factory_fn = getattr(estimator, "accumulator_factory", None)
    if factory_fn is None:
        factory_fn = estimator.accumulatorFactory
    return factory_fn()


class SelectDistribution(SequenceEncodableProbabilityDistribution):
    """SelectDistribution routes each observation to one child distribution via a choice function.

    The density of an observation x is dists[choice_function(x)].density(x).
    """

    def __init__(
        self, dists: Sequence[SequenceEncodableProbabilityDistribution], choice_function: Callable[[T], int]
    ) -> None:
        """SelectDistribution object for observations routed to child distributions.

        Args:
            dists (Sequence[SequenceEncodableProbabilityDistribution]): Child distributions, each
                compatible with the observations the choice function routes to it.
            choice_function (Callable[[T], int]): Maps an observation to the index of the child
                distribution that models it. Must return values in {0, ..., len(dists)-1}.

        Attributes:
            dists (Sequence[SequenceEncodableProbabilityDistribution]): Child distributions.
            choice_function (Callable[[T], int]): Observation-to-child routing function.
            count (int): Number of child distributions.

        """
        self.dists = dists
        self.choice_function = choice_function
        self.count = len(dists)

    def compute_capabilities(self):
        from pysp.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(engine_ready=intersect_engine_ready(tuple(self.dists)), kernel_status="generic")

    def compute_declaration(self):
        from pysp.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        children = tuple(declaration_for(d) for d in self.dists)
        children = tuple(d for d in children if d is not None)
        return DistributionDeclaration(
            name="select",
            distribution_type=type(self),
            parameters=(),
            statistics=(StatisticSpec("children", kind="choice_child_stats"),),
            support="choice_partition",
            children=children,
            child_roles=tuple("choice_%d" % i for i in range(len(children))),
            differentiable=all(child.differentiable for child in children),
        )

    def __str__(self) -> str:
        """Returns string representation of SelectDistribution object."""
        return "SelectDistribution(" + ",".join([str(u) for u in self.dists]) + ")"

    def density(self, x: T) -> float:
        """Density of the child distribution selected for observation x.

        Args:
            x (T): Observation compatible with the child selected by the choice function.

        Returns:
            Density of the selected child distribution at x.

        """
        idx = self.choice_function(x)
        return self.dists[idx].density(x)

    def log_density(self, x: T) -> float:
        """Log-density of the child distribution selected for observation x.

        Args:
            x (T): Observation compatible with the child selected by the choice function.

        Returns:
            Log-density of the selected child distribution at x.

        """
        idx = self.choice_function(x)
        return self.dists[idx].log_density(x)

    def seq_log_density(self, x: tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[Any, ...]]) -> np.ndarray:
        """Vectorized evaluation of the log-density on sequence encoded data x.

        The encoding groups observations by choice index: x[1][i] is the choice index of group i,
        x[0][i] holds the original positions of the group's observations, and x[2][i] is the
        group's data encoded by the matching child encoder. Each group is scored by the child
        distribution its choice index selects.

        Args:
            x: Sequence encoded data produced by SelectDataEncoder.seq_encode().

        Returns:
            Numpy array of log-densities, one entry per encoded observation.

        """
        xi, idx, enc_tuple = x
        sz = sum(len(u) for u in xi)
        rv = np.zeros(sz)
        for i in range(len(idx)):
            rv[xi[i]] = self.dists[idx[i]].seq_log_density(enc_tuple[i])
        return rv

    def backend_seq_log_density(
        self, x: tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[Any, ...]], engine: Any
    ) -> Any:
        """Engine-neutral vectorized log-density for choice-grouped encodings."""
        from pysp.stats.compute.backend import backend_seq_log_density

        xi, idx, enc_tuple = x
        sz = sum(len(u) for u in xi)
        rv = engine.zeros(sz)
        for i in range(len(idx)):
            child_scores = backend_seq_log_density(self.dists[idx[i]], enc_tuple[i], engine)
            rv = engine.index_add(rv, engine.asarray(xi[i]), child_scores)
        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["SelectDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked child parameters for homogeneous select-wrapper mixtures."""
        from pysp.stats.compute.stacked import stacked_component_params

        count = dists[0].count
        choice_function = dists[0].choice_function
        if any(d.count != count or d.choice_function is not choice_function for d in dists):
            raise ValueError("Stacked SelectDistribution components require matching choice routing.")
        children = []
        for i in range(count):
            child_dists = [d.dists[i] for d in dists]
            try:
                children.append(stacked_component_params(child_dists, engine))
            except ValueError as exc:
                raise ValueError(
                    "Select choice %d child %s is not stackable: %s" % (i, type(child_dists[0]).__name__, exc)
                )
        return {"children": tuple(children), "choice_function": choice_function, "num_components": len(dists)}

    @classmethod
    def backend_stacked_log_density(
        cls, x: tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[Any, ...]], params: dict[str, Any], engine: Any
    ) -> Any:
        """Return an ``(n, k)`` matrix of choice-routed select log densities."""
        from pysp.stats.compute.stacked import stacked_component_log_density

        xi, idx, enc_tuple = x
        sz = sum(len(u) for u in xi)
        rv = engine.zeros((sz, int(params["num_components"])))
        for i in range(len(idx)):
            child_scores = stacked_component_log_density(enc_tuple[i], params["children"][idx[i]], engine)
            rv = engine.index_add(rv, engine.asarray(xi[i]), child_scores)
        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls,
        x: tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[Any, ...]],
        weights: Any,
        params: dict[str, Any],
        engine: Any,
        estimator: Any,
    ) -> tuple[list[tuple[Any, Any]], ...]:
        """Return per-component legacy select sufficient statistics."""
        from pysp.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        xi, idx, enc_tuple = x
        ww = engine.asarray(weights)
        num_components = int(tuple(getattr(ww, "shape", (0, 0)))[1])
        outer_estimators = tuple(getattr(estimator, "estimators", ()))
        by_choice = {choice: pos for pos, choice in enumerate(idx)}
        per_component: list[list[tuple[Any, Any]]] = [[None] * len(params["children"]) for _ in range(num_components)]

        for choice, route in enumerate(params["children"]):
            component_estimators = tuple(
                getattr(component_est, "estimators", ())[choice]
                for component_est in outer_estimators
                if len(getattr(component_est, "estimators", ())) > choice
            )
            group_pos = by_choice.get(choice)
            if group_pos is None:
                child_counts = [0.0] * num_components
                child_stats_by_component = tuple(
                    _child_accumulator_factory(component_est).make().value() for component_est in component_estimators
                )
                if len(child_stats_by_component) != num_components:
                    child_stats_by_component = tuple(None for _ in range(num_components))
            else:
                row_index = engine.asarray(xi[group_pos])
                group_weights = ww[row_index]
                child_counts = engine.sum(group_weights, axis=0)
                child_estimator = (
                    StackedEstimatorView(component_estimators) if len(component_estimators) == num_components else None
                )
                child_stats = stacked_component_sufficient_statistics(
                    enc_tuple[group_pos], group_weights, route, engine, child_estimator
                )
                child_stats_by_component = unstack_component_stats(child_stats, num_components)
            for component in range(num_components):
                per_component[component][choice] = (
                    child_counts[component],
                    child_stats_by_component[component],
                )
        return tuple(per_component)

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from pysp.stats.compute.gradient import SelectGradientFitState

        return SelectGradientFitState(self, [recurse(dist, engine, torch, leaves) for dist in self.dists])

    def sampler(self, seed: int | None = None) -> "SelectSampler":
        """Creates a SelectSampler object for sampling from the child distributions.

        Args:
            seed (Optional[int]): Seed for the random number generator used in sampling.

        Returns:
            SelectSampler object.

        """
        return SelectSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SelectEstimator":
        """Creates a SelectEstimator with one child estimator per child distribution.

        Args:
            pseudo_count (Optional[float]): Passed through to each child estimator.

        Returns:
            SelectEstimator object.

        """
        return SelectEstimator([d.estimator(pseudo_count=pseudo_count) for d in self.dists], self.choice_function)

    def dist_to_encoder(self) -> "SelectDataEncoder":
        """Creates a SelectDataEncoder object for encoding sequences of SelectDistribution data.

        Returns:
            SelectDataEncoder object.

        """
        encoders = [d.dist_to_encoder() for d in self.dists]
        return SelectDataEncoder(encoders=encoders, choice_function=self.choice_function)

    def enumerator(self) -> "SelectEnumerator":
        """Creates a SelectEnumerator iterating the union of child supports in descending
        select-density order. All children must support enumeration."""
        return SelectEnumerator(self)


class SelectEnumerator(DistributionEnumerator):
    """Enumerates the union of the child supports in descending select-density order."""

    def __init__(self, dist: SelectDistribution) -> None:
        """Enumerates the union of the child supports in descending select-density order.

        Candidates are pulled best-first from the child enumerations, de-duplicated, and re-scored
        exactly with the select log-density p(x) = p_{c(x)}(x). A candidate is emitted only once
        its exact score beats the bound max over child stream heads, which upper-bounds any
        not-yet-seen value because such a value has not yet been pulled from the stream of its own
        selected child. Values whose selected child assigns zero probability are skipped.

        Requires every child to support enumeration, and requires the choice function to be
        defined on every value in every child's support.

        Args:
            dist (SelectDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        streams = [
            BufferedStream(child_enumerator(d, "SelectDistribution.dists[%d]" % i)) for i, d in enumerate(dist.dists)
        ]
        log_offsets = [0.0] * len(streams)

        # Zero-probability candidates (selected child assigns no mass) are re-scored to -inf and
        # skipped; suppress the harmless log(0) warning that some children emit in that case.
        def exact_log_density(x):
            with np.errstate(divide="ignore"):
                return dist.log_density(x)

        self._union = best_first_union_max(streams, log_offsets, exact_log_density)

    def __next__(self) -> tuple[Any, float]:
        return next(self._union)


class SelectSampler(DistributionSampler):
    """SelectSampler draws samples from each child distribution of a SelectDistribution."""

    def __init__(self, dist: SelectDistribution, seed: int | None = None) -> None:
        """SelectSampler object used to generate samples from the children of a SelectDistribution.

        Args:
            dist (SelectDistribution): SelectDistribution to draw samples from.
            seed (Optional[int]): Seed for the random number generator used in sampling.

        Attributes:
            dist (SelectDistribution): SelectDistribution to draw samples from.
            rng (RandomState): RandomState with seed set if provided.
            dist_samplers (List[DistributionSampler]): One sampler per child distribution.

        """
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_samplers = [d.sampler(seed=self.rng.randint(maxint)) for d in dist.dists]

    def sample(self, size: int | None = None):
        """Draw one sample from every child distribution.

        Note: this samples each child independently and groups the draws, returning a tuple with
        one entry per child (or a list of such tuples when size is given). It is not a draw from
        the select density itself, since the select density conditions on the choice function.

        Args:
            size (Optional[int]): Number of grouped draws. If None a single tuple is returned.

        Returns:
            Tuple with one sample per child if size is None, else a list of 'size' such tuples.

        """
        if size is None:
            return tuple([d.sample(size=size) for d in self.dist_samplers])
        else:
            return list(zip(*[d.sample(size=size) for d in self.dist_samplers]))


class SelectEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """SelectEstimatorAccumulator accumulates sufficient statistics for each child distribution,
    routing each observation to one child accumulator via the choice function."""

    def __init__(
        self, accumulators: Sequence[SequenceEncodableStatisticAccumulator], choice_function: Callable[[T], int]
    ) -> None:
        """SelectEstimatorAccumulator object for aggregating sufficient statistics.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): One accumulator per
                child distribution.
            choice_function (Callable[[T], int]): Maps an observation to the index of the child
                accumulator that receives it.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Child accumulators.
            choice_function (Callable[[T], int]): Observation-to-child routing function.
            weights (List[float]): Total weight routed to each child.
            count (int): Number of child accumulators.

            _rng_init (bool): True once the per-child RandomStates have been seeded.
            _acc_rng (Optional[List[RandomState]]): Per-child RandomStates used by initialize.

        """
        self.accumulators = accumulators
        self.choice_function = choice_function
        self.weights = [zero] * len(accumulators)
        self.count = len(accumulators)

        self._rng_init = False
        self._acc_rng: list[RandomState] | None = None

    def update(self, x: T, weight: float, estimate: SelectDistribution | None) -> None:
        """Route one weighted observation to the accumulator of the selected child.

        Args:
            x (T): Observation routed by the choice function.
            weight (float): Weight for the observation.
            estimate (Optional[SelectDistribution]): Previous estimate; the matching child
                distribution is passed through to the child accumulator if provided.

        Returns:
            None.

        """
        idx = self.choice_function(x)
        self.accumulators[idx].update(x, weight, estimate.dists[idx] if estimate is not None else None)
        self.weights[idx] += weight

    def _rng_initialize(self, rng: RandomState) -> None:
        """Seed one RandomState per child accumulator for consistent initialize calls."""
        self._acc_rng = [RandomState(seed=rng.randint(0, maxrandint)) for xx in range(self.count)]
        self._rng_init = True

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        """Initialize the accumulator of the selected child with one weighted observation.

        Args:
            x (T): Observation routed by the choice function.
            weight (float): Weight for the observation.
            rng (RandomState): RandomState used to seed the per-child RandomStates.

        Returns:
            None.

        """
        if not self._rng_init:
            self._rng_initialize(rng)

        idx = self.choice_function(x)
        self.accumulators[idx].initialize(x, weight, self._acc_rng[idx])
        self.weights[idx] += weight

    def seq_update(
        self,
        x: tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[Any, ...]],
        weights: np.ndarray,
        estimate: SelectDistribution | None,
    ) -> None:
        """Vectorized update of the child accumulators from sequence encoded data x.

        Each encoded group i carries its choice index x[1][i]; the group's encoded data and
        weights are routed to the child accumulator at that choice index.

        Args:
            x: Sequence encoded data produced by SelectDataEncoder.seq_encode().
            weights (np.ndarray): Weights for each encoded observation.
            estimate (Optional[SelectDistribution]): Previous estimate; the matching child
                distributions are passed through to the child accumulators if provided.

        Returns:
            None.

        """
        xi, idx, enc_tuple = x
        for i in range(len(idx)):
            j = idx[i]
            w = weights[xi[i]]
            self.accumulators[j].seq_update(enc_tuple[i], w, estimate.dists[j] if estimate is not None else None)
            self.weights[j] += np.sum(w)

    def seq_update_engine(
        self,
        x: tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[Any, ...]],
        weights: Any,
        estimate: SelectDistribution | None,
        engine: Any,
    ) -> None:
        """Engine-resident E-step: each group's weights are gathered on the active engine and routed
        to the chosen child accumulator through the engine. Matches seq_update.
        """
        from pysp.stats.compute.backend import child_seq_update

        xi, idx, enc_tuple = x
        w_eng = engine.asarray(weights)
        for i in range(len(idx)):
            j = idx[i]
            w = w_eng[np.asarray(xi[i], dtype=np.int64)]
            child_seq_update(
                self.accumulators[j], enc_tuple[i], w, estimate.dists[j] if estimate is not None else None, engine
            )
            self.weights[j] += float(engine.to_numpy(engine.sum(w)))

    def seq_initialize(
        self, x: tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[Any, ...]], weights: np.ndarray, rng: RandomState
    ) -> None:
        """Vectorized initialization of the child accumulators from sequence encoded data x.

        Each encoded group i carries its choice index x[1][i]; the group's encoded data and
        weights are routed to the child accumulator at that choice index.

        Args:
            x: Sequence encoded data produced by SelectDataEncoder.seq_encode().
            weights (np.ndarray): Weights for each encoded observation.
            rng (RandomState): RandomState used to seed the per-child RandomStates.

        Returns:
            None.

        """
        if not self._rng_init:
            self._rng_initialize(rng)

        xi, idx, enc_tuple = x
        for i in range(len(idx)):
            j = idx[i]
            w = weights[xi[i]]
            self.accumulators[j].seq_initialize(enc_tuple[i], w, self._acc_rng[j])
            self.weights[j] += np.sum(w)

    def combine(self, suff_stat: Sequence[tuple[float, Any]]) -> "SelectEstimatorAccumulator":
        """Aggregate sufficient statistics suff_stat with this accumulator's statistics.

        Args:
            suff_stat (Sequence[Tuple[float, Any]]): One (weight, child sufficient statistic)
                pair per child, as returned by value().

        Returns:
            SelectEstimatorAccumulator with combined sufficient statistics.

        """
        for i in range(0, self.count):
            self.weights[i] += suff_stat[i][0]
            self.accumulators[i].combine(suff_stat[i][1])

        return self

    def value(self) -> list[tuple[float, Any]]:
        """Returns the sufficient statistics as a list of (weight, child value) pairs."""
        return [(w, acc.value()) for w, acc in zip(self.weights, self.accumulators)]

    def from_value(self, x: Sequence[tuple[float, Any]]) -> "SelectEstimatorAccumulator":
        """Set the accumulator's sufficient statistics to x.

        Args:
            x (Sequence[Tuple[float, Any]]): One (weight, child sufficient statistic) pair per
                child, as returned by value().

        Returns:
            SelectEstimatorAccumulator object.

        """
        for i, u in enumerate(x):
            self.weights[i] = u[0]
            self.accumulators[i].from_value(u[1])

        return self

    def scale(self, c: float) -> "SelectEstimatorAccumulator":
        for i in range(self.count):
            self.weights[i] *= c
            self.accumulators[i].scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Invoke key_merge on each child accumulator.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to shared sufficient statistics.

        Returns:
            None.

        """
        for acc in self.accumulators:
            acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Invoke key_replace on each child accumulator.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to shared sufficient statistics.

        Returns:
            None.

        """
        for acc in self.accumulators:
            acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> "SelectDataEncoder":
        """Creates a SelectDataEncoder object for encoding sequences of SelectDistribution data.

        Returns:
            SelectDataEncoder object.

        """
        encoders = [acc.acc_to_encoder() for acc in self.accumulators]
        return SelectDataEncoder(encoders=encoders, choice_function=self.choice_function)


class SelectEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """SelectEstimatorAccumulatorFactory creates SelectEstimatorAccumulator objects from the
    child estimators."""

    def __init__(self, estimators: Sequence[ParameterEstimator], choice_function: Callable[[T], int]) -> None:
        """SelectEstimatorAccumulatorFactory object.

        Args:
            estimators (Sequence[ParameterEstimator]): One estimator per child distribution.
            choice_function (Callable[[T], int]): Observation-to-child routing function.

        Attributes:
            estimators (Sequence[ParameterEstimator]): One estimator per child distribution.
            choice_function (Callable[[T], int]): Observation-to-child routing function.

        """
        self.estimators = estimators
        self.choice_function = choice_function

    def make(self) -> "SelectEstimatorAccumulator":
        """Creates a SelectEstimatorAccumulator with one child accumulator per child estimator.

        Returns:
            SelectEstimatorAccumulator object.

        """
        return SelectEstimatorAccumulator(
            [_child_accumulator_factory(x).make() for x in self.estimators], self.choice_function
        )


class SelectEstimator(ParameterEstimator):
    """SelectEstimator estimates a SelectDistribution from child sufficient statistics."""

    def __init__(self, estimators: Sequence[ParameterEstimator], choice_function: Callable[[T], int]) -> None:
        """SelectEstimator object.

        Args:
            estimators (Sequence[ParameterEstimator]): One estimator per child distribution.
            choice_function (Callable[[T], int]): Observation-to-child routing function.

        Attributes:
            estimators (Sequence[ParameterEstimator]): One estimator per child distribution.
            choice_function (Callable[[T], int]): Observation-to-child routing function.
            count (int): Number of child estimators.

        """
        self.estimators = estimators
        self.choice_function = choice_function
        self.count = len(estimators)

    def accumulator_factory(self) -> "SelectEstimatorAccumulatorFactory":
        """Creates a SelectEstimatorAccumulatorFactory from the child estimators.

        Returns:
            SelectEstimatorAccumulatorFactory object.

        """
        return SelectEstimatorAccumulatorFactory(self.estimators, self.choice_function)

    def estimate(self, nobs: float | None, suff_stat: Sequence[tuple[float, Any]]) -> "SelectDistribution":
        """Estimate a SelectDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency.
            suff_stat (Sequence[Tuple[float, Any]]): One (weight, child sufficient statistic)
                pair per child, as returned by SelectEstimatorAccumulator.value().

        Returns:
            SelectDistribution object.

        """
        return SelectDistribution(
            [est.estimate(ss[0], ss[1]) for est, ss in zip(self.estimators, suff_stat)], self.choice_function
        )


class SelectDataEncoder(DataSequenceEncoder):
    """SelectDataEncoder encodes sequences of SelectDistribution data, grouping observations by
    their choice index and delegating each group to the matching child encoder."""

    def __init__(self, encoders: Sequence[DataSequenceEncoder], choice_function: Callable[[T], int]) -> None:
        """SelectDataEncoder object.

        Args:
            encoders (Sequence[DataSequenceEncoder]): One encoder per child distribution.
            choice_function (Callable[[T], int]): Observation-to-child routing function.

        Attributes:
            encoders (Sequence[DataSequenceEncoder]): One encoder per child distribution.
            choice_function (Callable[[T], int]): Observation-to-child routing function.

        """
        self.encoders = encoders
        self.choice_function = choice_function

    def __str__(self) -> str:
        """Returns string representation of SelectDataEncoder with its child encoders."""
        return "SelectDataEncoder(" + ",".join([str(encoder) for encoder in self.encoders]) + ")"

    def __eq__(self, other: object) -> bool:
        """Checks if an object is an equivalent SelectDataEncoder.

        Note: assumes that the choice functions of the two encoders are equal; only the child
        encoders are compared.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a SelectDataEncoder with equal child encoders, else False.

        """
        if isinstance(other, SelectDataEncoder):
            if len(other.encoders) != len(self.encoders):
                return False

            for i, encoder in enumerate(self.encoders):
                if other.encoders[i] != encoder:
                    return False

            return True

        else:
            return False

    def seq_encode(self, x: Sequence[T]) -> tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[Any, ...]]:
        """Encode a sequence of iid SelectDistribution observations for vectorized "seq_" calls.

        Observations are grouped by their choice index, in order of first appearance. The
        encoding is a tuple of three aligned tuples (one entry per observed choice index):

            rv[0] (Tuple[np.ndarray, ...]): Original positions of each group's observations.
            rv[1] (Tuple[int, ...]): The choice index of each group.
            rv[2] (Tuple[Any, ...]): Each group's data encoded by the matching child encoder.

        Args:
            x (Sequence[T]): Sequence of iid observations.

        Returns:
            See description above.

        """
        cnt = 0
        idx_dict = dict()

        for i, xx in enumerate(x):
            idx = self.choice_function(xx)
            if idx not in idx_dict:
                idx_dict[idx] = [[], []]
            idx_dict[idx][1].append(xx)
            idx_dict[idx][0].append(i)
            cnt += 1

        idx_keys = []
        idx_xi = []
        idx_enc_vals = []

        for keys, vals in idx_dict.items():
            idx_keys.append(keys)
            idx_xi.append(np.asarray(vals[0]))
            idx_enc_vals.append(self.encoders[keys].seq_encode(vals[1]))

        return tuple(idx_xi), tuple(idx_keys), tuple(idx_enc_vals)


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
SelectAccumulator = SelectEstimatorAccumulator
SelectAccumulatorFactory = SelectEstimatorAccumulatorFactory
