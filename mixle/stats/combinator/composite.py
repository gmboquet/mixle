"""Independent product distributions over tuple-valued observations.

Data type: (Tuple[T_0, ... T_{n-1}]): The CompositeDistribution of size 'n' is a joint distribution for
independent observations of 'n'-tupled data. Each component 'k' of the CompositeDistribution has data type T_k that
must be compatible with data type T_k.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import (
    BufferedStream,
    LazyQuantizedEnumerationIndex,
    ProductEnumerator,
    QuantizedCrossIndex,
)
from mixle.inference.fisher import Path
from mixle.stats.compute.pdist import (
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

T = tuple[Any, ...]
E = TypeVar("E")
SS = TypeVar("SS")


from mixle.inference.fisher import FixedFisherView, SufficientStatisticVectorizer, to_fisher


def _distribute_child_prior(child: Any, prior: Any) -> None:
    """Push a parameter prior onto a child distribution/estimator (structural-wrapper helper).

    Prefers the child's ``set_prior`` when it exists. Leaf *estimators* that predate the unified
    Bayesian surface expose only a ``prior`` attribute (set in ``__init__``); for those this writes
    the attribute directly and refreshes ``has_conj_prior`` so the child's ``estimate`` takes its
    conjugate branch (the prior attached to such a leaf is its conjugate prior by construction).
    """
    set_prior = getattr(child, "set_prior", None)
    if callable(set_prior):
        set_prior(prior)
        return
    child.prior = prior
    if hasattr(child, "has_conj_prior"):
        child.has_conj_prior = prior is not None


class CompositeDistribution(SequenceEncodableProbabilityDistribution):
    """Product distribution over heterogeneous component variables."""

    def compute_capabilities(self):
        """Return compute-backend metadata shared by all component distributions."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(
            engine_ready=intersect_engine_ready(tuple(self.dists)), kernel_status="numba_adapter"
        )

    def __init__(
        self,
        dists: Sequence[SequenceEncodableProbabilityDistribution],
        prior: Sequence[SequenceEncodableProbabilityDistribution] | None = None,
    ) -> None:
        """CompositeDistribution for modeling independent distributions of from (Dist_0,Dist_1,...,Dist_{n-1}).

        Data type must be (T_0, T_1, ..., T_{n-1}), where data type T_k is consistent with distribution Dist_k. The
        density for a single observation tuple x = (x_0,x_1,...,x_{n-1}) is given by,

        p_mat(x) = p_mat(x_0 | Dist_0)*p_mat(x_1 | Dist_1)*...*p_mat(x_{n-1} | Dist_{n-1}).

        Args:
            dists (Sequence[SequenceEncodableProbabilityDistribution]): Distributions given by Dist_k above.
            prior (Optional): Per-component parameter priors. ``CompositeDistribution`` is a structural
                wrapper, so the joint prior factors over the components: a sequence of one prior per
                component (in component order) is distributed to the children via ``set_prior``. ``None``
                (default) leaves every child a plain point model (existing behavior byte-identical).

        Attributes:
            dists: (Sequence[SequenceEncodableProbabilityDistribution]): Distributions given by Dist_k above.
            counts (int): Number of components (i.e. len(dists)).

        """
        self.dists = dists
        self.count = len(dists)
        self.set_prior(prior)

    def get_prior(self) -> list[SequenceEncodableProbabilityDistribution | None]:
        """Return the joint prior as the list of per-component child priors (in component order)."""
        return [d.get_prior() for d in self.dists]

    def set_prior(self, prior: Sequence[SequenceEncodableProbabilityDistribution | None] | None) -> None:
        """Distribute per-component parameter priors to the wrapped child distributions.

        ``CompositeDistribution`` owns no parameters of its own; the joint prior factors over the
        independent components. ``prior=None`` is a no-op (children keep their existing priors,
        leaving the MLE path byte-identical); otherwise ``prior`` must be a sequence of exactly
        ``count`` child priors that are pushed to the children via their own ``set_prior``.
        """
        if prior is None:
            return
        prior = list(prior)
        if len(prior) != self.count:
            raise ValueError(
                "CompositeDistribution.set_prior expected %d priors but got %d." % (self.count, len(prior))
            )
        for d, p in zip(self.dists, prior):
            d.set_prior(p)

    def expected_log_density(self, x: tuple[Any, ...]) -> float:
        """Prior-expected log-density: sum of the component ``expected_log_density`` values at ``x``."""
        self._check_arity(x)
        rv = self.dists[0].expected_log_density(x[0])
        for i in range(1, self.count):
            rv += self.dists[i].expected_log_density(x[i])
        return rv

    def seq_expected_log_density(self, x: E) -> np.ndarray:
        """Vectorized prior-expected log-density: sum of the component ``seq_expected_log_density`` values."""
        rv = self.dists[0].seq_expected_log_density(x[0])
        for i in range(1, self.count):
            rv += self.dists[i].seq_expected_log_density(x[i])
        return rv

    def compute_declaration(self):
        """Return the symbolic declaration for the product of component distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        children = tuple(declaration_for(d) for d in self.dists)
        children = tuple(d for d in children if d is not None)
        return DistributionDeclaration(
            name="composite",
            distribution_type=type(self),
            parameters=(),
            statistics=(StatisticSpec("components", kind="tuple"),),
            support="product",
            children=children,
            child_roles=tuple("field_%d" % i for i in range(len(children))),
            differentiable=all(child.differentiable for child in children),
        )

    def __str__(self) -> str:
        """Returns str name of CompositeDistribution with each dist as well."""
        return "CompositeDistribution((%s))" % (",".join(map(str, self.dists)))

    def marginal(self, indices: Sequence[int]) -> CompositeDistribution:
        """The marginal sub-composite over the given component ``indices``.

        Because the components are independent, the marginal of a subset of coordinates is just the
        sub-product over those coordinates. Used (with :meth:`condition`) by ``MixtureDistribution.conditional``
        to score the observed coordinates of a partial observation."""
        idx = sorted(indices)
        return CompositeDistribution([self.dists[i] for i in idx])

    def condition(self, observed: dict[int, Any]) -> CompositeDistribution:
        """The conditional sub-composite over the UNobserved components given ``observed``.

        ``observed`` maps a component index to its (present) value. Since the components are independent,
        conditioning leaves the unobserved factors unchanged -- the conditional is the sub-product over the
        coordinates not in ``observed`` (the observed values do not enter). This is the per-component piece
        that makes ``MixtureDistribution.conditional`` return the posterior/imputation over the missing
        fields of a partial observation."""
        obs = set(observed)
        return CompositeDistribution([self.dists[i] for i in range(self.count) if i not in obs])

    def _check_arity(self, x: tuple[Any, ...]) -> None:
        """A too-short ``x`` would otherwise crash with a bare ``IndexError`` deep in the per-field
        loop below, with no indication of which field or how many were expected; a too-long ``x``
        would otherwise be silently accepted with the extra fields never read at all. Both are real
        caller mistakes (e.g. an extra/missing column) that should surface immediately, at the call
        site, not as a wrong log-likelihood with no signal anything was off."""
        if len(x) != self.count:
            raise ValueError(
                "CompositeDistribution observation has %d fields but this composite has %d components."
                % (len(x), self.count)
            )

    def density(self, x: tuple[Any, ...]) -> float:
        """Evaluates density of CompositeDistribution for single observation tuple x.

        p_mat(x) = p_mat(x_0 | dist_0)*p_mat(x_1 | dist_1)*...*p_mat(x_{n-1} | dist_{n-1}),

        where dist_k is the k^{th} element of member variable dists and is consistent with data type type(x[k]).

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k^{th} data type must be consistent with dists[k].

        Returns:
            Density as float.

        """
        self._check_arity(x)
        rv = self.dists[0].density(x[0])

        for i in range(1, self.count):
            rv *= self.dists[i].density(x[i])

        return rv

    def density_semantics(self):
        """Return joined density semantics over all component distributions."""
        from mixle.stats.compute.pdist import join_density_semantics

        return join_density_semantics(c.density_semantics() for c in self.dists)

    def log_density(self, x: tuple[Any, ...]) -> float:
        """Evaluates log-density of CompositeDistribution for single observation tuple x.

        log(p_mat(x)) = log(p_mat(x_0 | dist_0)) + log(p_mat(x_1 | dist_1)) + ... + log(p_mat(x_{n-1} | dist_{n-1})),

        where dist_k is the k^{th} element of member variable dists and is consistent with data type type(x[k]).

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k^{th} data type must be consistent with dists[k].

        Returns:
            Log-density as float.

        """
        self._check_arity(x)
        rv = self.dists[0].log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].log_density(x[i])

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log density for Tuple of dist encoded data.

        Each entry of x is an encoded sequence, encoded by the DataSequenceEncoder of dist[k].dist_to_encoder().

        Note: len(x) == len(dists).
        Args:
            x (E): Tuple of length = len(dists), with k^{th} entry given by encoded sequence of dist[k]'s.

        Returns:
            np.ndarray of log_density evaluated at all encoded data points.

        """
        rv = self.dists[0].seq_log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].seq_log_density(x[i])

        return rv

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density by composing child distributions."""
        from mixle.stats.compute.backend import backend_seq_log_density

        rv = backend_seq_log_density(self.dists[0], x[0], engine)
        for i in range(1, self.count):
            rv = rv + backend_seq_log_density(self.dists[i], x[i], engine)
        return rv

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from mixle.stats.compute.gradient import CompositeGradientFitState

        return CompositeGradientFitState(self, [recurse(dist, engine, torch, leaves) for dist in self.dists])

    @classmethod
    def backend_stacked_params(cls, dists: Sequence[CompositeDistribution], engine: Any) -> dict[str, Any]:
        """Return stacked child parameters for homogeneous composite mixtures."""
        from mixle.stats.compute.stacked import stacked_component_params

        count = dists[0].count
        if any(d.count != count for d in dists):
            raise ValueError("Stacked CompositeDistribution components require equal arity.")
        children = []
        for i in range(count):
            child_dists = [d.dists[i] for d in dists]
            try:
                children.append(stacked_component_params(child_dists, engine))
            except ValueError as exc:
                raise ValueError("Composite child %s is not stackable: %s" % (type(child_dists[0]).__name__, exc))
        return {"children": tuple(children)}

    @classmethod
    def backend_stacked_log_density(cls, x: E, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of composite log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        children = params["children"]
        rv = stacked_component_log_density(x[0], children[0], engine)
        for i in range(1, len(children)):
            rv = rv + stacked_component_log_density(x[i], children[i], engine)
        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: E, weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy composite sufficient statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        ww = engine.asarray(weights)
        num_components = int(tuple(getattr(ww, "shape", (0, 0)))[1])
        outer_estimators = tuple(getattr(estimator, "estimators", ()))
        child_payloads = []
        for i, route in enumerate(params["children"]):
            component_estimators = tuple(
                getattr(component_est, "estimators", ())[i]
                for component_est in outer_estimators
                if len(getattr(component_est, "estimators", ())) > i
            )
            child_estimator = (
                StackedEstimatorView(component_estimators) if len(component_estimators) == num_components else None
            )
            child_stats = stacked_component_sufficient_statistics(x[i], ww, route, engine, child_estimator)
            child_payloads.append(unstack_component_stats(child_stats, num_components))
        return tuple(tuple(child[i] for child in child_payloads) for i in range(num_components))

    def support_size(self) -> int | None:
        """Product of child support sizes (``None`` if any child is infinite)."""
        if self.count == 0:
            return 1
        total = 1
        for d in self.dists:
            s = d.support_size()
            if s is None:
                return None
            total *= s
        return total

    def to_fisher(self, **kwargs):
        """Structural Fisher view (product of child views)."""
        if hasattr(self, "dists"):
            return CompositeFisherView(self)
        return super().to_fisher(**kwargs)

    def to_exponential_family(self, engine: Any = None):
        """Return the product exponential-family view, or ``None``.

        A composite is an exponential family iff every child is: the canonical pieces
        concatenate (``eta``, ``T``) and add (``A``, ``log h``).  Returns ``None`` when
        any child is not a (single) exponential family.
        """
        from mixle.engines import NUMPY_ENGINE
        from mixle.stats.compute.exp_family import ProductExponentialFamilyForm, to_exponential_family

        eng = NUMPY_ENGINE if engine is None else engine
        children = [to_exponential_family(d, engine=eng) for d in self.dists]
        if any(c is None for c in children):
            return None
        return ProductExponentialFamilyForm(
            distribution=self,
            components=tuple(children),
            engine=eng,
        )

    def sampler(self, seed: int | None = None) -> CompositeSampler:
        """Return a sampler that samples each component independently.

        Args:
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Returns:
            CompositeSampler object.

        """
        return CompositeSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> CompositeEstimator:
        """Return a composite estimator built from the component estimators.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics in estimation.

        Returns:
            CompositeEstimator object.

        """
        return CompositeEstimator([d.estimator(pseudo_count=pseudo_count) for d in self.dists])

    def decomposition(self):
        """Composite factors are independent: split along the factor axis, sufficient stats SUM-reduce."""
        from mixle.stats.compute.decomposition import DecompAxis, Decomposition, ReductionOp

        return Decomposition(
            axis=DecompAxis.FACTOR,
            num_units=self.count,
            reduction=ReductionOp.SUM,
            exact=True,
            child_roles=("factor",) * self.count,
        )

    def dist_to_encoder(self) -> CompositeDataEncoder:
        """Return a tuple encoder assembled from the component encoders.

        Passes 'encoders', which is a list of DataSequenceEncoders for each component of the CompositeDistribution.

        Returns:
            CompositeDataEncoder object.

        """
        encoders = tuple([d.dist_to_encoder() for d in self.dists])

        return CompositeDataEncoder(encoders=encoders)

    def enumerator(self) -> CompositeEnumerator:
        """Creates CompositeEnumerator iterating tuples in descending joint probability order."""
        return CompositeEnumerator(self)

    def conditional_enumerator(self, given: Mapping[int, Any]) -> CompositeConditionalEnumerator:
        """Enumerate complete tuples consistent with the fixed positions in ``given``, best-first.

        ``given`` is a mapping ``{position: value}`` pinning a subset of coordinates (most-probable
        completion / imputation). Because the components are independent, descending order over the
        *free* coordinates is descending conditional order; each yielded tuple has the fixed positions
        filled in and carries the full joint ``log_density`` (the fixed positions are a constant
        offset). Raises ValueError for an out-of-range position.
        """
        if not isinstance(given, Mapping):
            raise TypeError("given must be a mapping of {position: value}.")
        bad = [k for k in given if not isinstance(k, (int, np.integer)) or k < 0 or k >= self.count]
        if bad:
            raise ValueError("given names positions outside [0, %d): %r" % (self.count, bad))
        return CompositeConditionalEnumerator(self, {int(k): v for k, v in given.items()})

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> LazyQuantizedEnumerationIndex:
        """Build a bounded index with a DP over additive quantized child costs.

        Each child item is assigned an integer cost ceil(bits/bin_width_bits). The
        composite cost is the sum of those integer costs, so the bin counts are a
        convolution of child cost-bin counts. Items are unranked lazily from the child
        bin offsets when requested; the returned log probability is still the exact
        joint log-density.
        """
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        max_bin = int(math.floor(float(max_bits) / float(bin_width_bits) + 1.0e-12))
        if self.count == 0:
            counts = {0: 1} if max_bin >= 0 else {}

            def empty_getter(bin_id: int, offset: int) -> tuple[tuple[Any, ...], float]:
                if bin_id != 0 or offset != 0:
                    raise IndexError("offset outside indexed bin.")
                return (), 0.0

            return LazyQuantizedEnumerationIndex(
                counts, bin_width_bits=bin_width_bits, max_bits=max_bits, truncated=False, getter=empty_getter
            )

        child_bins: list[dict[int, list[tuple[Any, float]]]] = []
        truncated = False
        for i, dist in enumerate(self.dists):
            try:
                child_index = dist.quantized_index(max_bits=max_bits, bin_width_bits=bin_width_bits)
            except EnumerationError as e:
                path = "CompositeDistribution.dists[%d]" % i
                new_path = path if not e.path else "%s -> %s" % (path, e.path)
                raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None

            truncated = truncated or child_index.truncated
            bins_i: dict[int, list[tuple[Any, float]]] = defaultdict(list)
            for value, log_prob in child_index.iter_from():
                bits = max(0.0, -float(log_prob) / math.log(2.0))
                qbin = int(math.ceil(bits / float(bin_width_bits) - 1.0e-12))
                if qbin <= max_bin:
                    bins_i[qbin].append((value, float(log_prob)))
                else:
                    truncated = True
            child_bins.append(dict(bins_i))

        if any(len(bins_i) == 0 for bins_i in child_bins):

            def empty_getter(bin_id: int, offset: int) -> tuple[tuple[Any, ...], float]:
                raise IndexError("offset outside indexed bin.")

            return LazyQuantizedEnumerationIndex(
                {}, bin_width_bits=bin_width_bits, max_bits=max_bits, truncated=True, getter=empty_getter
            )

        plans: dict[int, list[tuple[tuple[int, ...], int]]] = {0: [((), 1)]}
        for bins_i in child_bins:
            next_plans: dict[int, list[tuple[tuple[int, ...], int]]] = defaultdict(list)
            for partial_bin, partial_plans in plans.items():
                for child_bin in sorted(bins_i):
                    new_bin = partial_bin + child_bin
                    if new_bin > max_bin:
                        truncated = True
                        continue
                    child_count = len(bins_i[child_bin])
                    for prefix, count in partial_plans:
                        next_plans[new_bin].append((prefix + (child_bin,), count * child_count))
            plans = dict(next_plans)

        counts = {b: sum(count for _, count in plan_list) for b, plan_list in plans.items() if plan_list}
        plans_by_bin = {b: plan_list for b, plan_list in plans.items() if plan_list}

        def getter(bin_id: int, offset: int) -> tuple[tuple[Any, ...], float]:
            if offset < 0:
                raise IndexError("offset must be non-negative.")
            for plan, plan_count in plans_by_bin.get(bin_id, []):
                if offset >= plan_count:
                    offset -= plan_count
                    continue
                values = []
                log_prob = 0.0
                local = offset
                item_offsets = [0] * len(plan)
                for j in range(len(plan) - 1, -1, -1):
                    n = len(child_bins[j][plan[j]])
                    item_offsets[j] = local % n
                    local //= n
                for j, item_offset in enumerate(item_offsets):
                    value, lp = child_bins[j][plan[j]][item_offset]
                    values.append(value)
                    log_prob += lp
                return tuple(values), float(log_prob)
            raise IndexError("offset outside indexed bin.")

        return LazyQuantizedEnumerationIndex(
            counts, bin_width_bits=bin_width_bits, max_bits=max_bits, truncated=truncated, getter=getter
        )

    def structural_fine_bucket(self, value, quantizer) -> int:
        """Sum of child structural buckets -- mirrors the count index's child convolution."""
        return sum(self.dists[i].structural_fine_bucket(value[i], quantizer) for i in range(self.count))

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """Structural count index: the ADDITIVE law -- the carrier's n-ary product over children.

        The complete log density is the sum of independent child log densities, so the joint count
        histogram is the ``times``/``product`` (convolution) of the child histograms in the
        witness-retaining count semiring (mixle.enumeration.quantization.semiring). Children are consumed
        by their *counts* and lazy unranker -- never drained -- so a child with astronomically large
        support (e.g. a Sequence) composes without being materialized. Swapping the carrier (e.g. a
        tropical one) would reuse this same reduction.
        """
        from mixle.enumeration.quantization.semiring import CountSemiring

        semiring = CountSemiring()
        if self.count == 0:
            return semiring.one(), False

        children = []
        truncated = False
        for i, dist in enumerate(self.dists):
            try:
                child_index, child_truncated = dist.quantized_count_index(quantizer, max_fine_bucket)
            except EnumerationError as e:
                path = "CompositeDistribution.dists[%d]" % i
                new_path = path if not e.path else "%s -> %s" % (path, e.path)
                raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None
            children.append(child_index)
            truncated = truncated or child_truncated

        return semiring.product(children, quantizer, max_fine_bucket), truncated

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an aligned cross-bin view for compatible composite distributions."""
        dists = [self] + list(others)
        if any(not isinstance(dist, CompositeDistribution) for dist in dists):
            raise EnumerationError(self, reason="composite cross-index requires CompositeDistribution objects")
        if any(dist.count != self.count for dist in dists):
            raise EnumerationError(self, reason="composite cross-index requires equal tuple arity")
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = tuple([float(max_bits)] * len(dists))
        if len(max_bits_tuple) != len(dists):
            raise ValueError("max_bits length must match the number of distributions.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        child_crosses = []
        truncated = False
        for i in range(self.count):
            try:
                child_cross = self.dists[i].quantized_multi_cross_index(
                    [dist.dists[i] for dist in dists[1:]], max_bits=max_bits_tuple, bin_width_bits=bin_width_bits
                )
            except EnumerationError as e:
                path = "CompositeDistribution.dists[%d]" % i
                new_path = path if not e.path else "%s -> %s" % (path, e.path)
                raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None
            child_crosses.append(child_cross)
            truncated = truncated or child_cross.truncated

        partials: list[tuple[tuple[Any, ...], tuple[float, ...]]] = [((), tuple([0.0] * len(dists)))]
        log2 = math.log(2.0)
        for child_cross in child_crosses:
            next_partials: list[tuple[tuple[Any, ...], tuple[float, ...]]] = []
            for prefix, lp_prefix in partials:
                for value, lps in child_cross.iter_items():
                    new_lps = tuple(float(lp_prefix[j] + lps[j]) for j in range(len(dists)))
                    bits = tuple(np.inf if lp == -np.inf else max(0.0, -lp / log2) for lp in new_lps)
                    if any(bits[j] <= max_bits_tuple[j] + 1.0e-12 for j in range(len(dists))):
                        next_partials.append((prefix + (value,), new_lps))
            partials = next_partials
            if not partials:
                break

        items = [(values, lps) for values, lps in partials]
        return QuantizedCrossIndex.from_items(
            items, max_bits=max_bits_tuple, bin_width_bits=bin_width_bits, truncated=truncated
        )

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an aligned cross-bin view for two compatible composite distributions."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class CompositeEnumerator(DistributionEnumerator):
    """Best-first enumerator over the Cartesian product of component supports."""

    def __init__(self, dist: CompositeDistribution) -> None:
        """Enumerates tuples of the component supports in descending joint probability order.

        Joint log-density is the sum of component log-densities, so this is a best-first
        search over the product of the (sorted) component enumerations. All components
        must support enumeration.

        Args:
            dist (CompositeDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        streams = [
            BufferedStream(child_enumerator(d, "CompositeDistribution.dists[%d]" % i)) for i, d in enumerate(dist.dists)
        ]
        self._product = ProductEnumerator(streams, combine=tuple)

    def __next__(self) -> tuple[tuple[Any, ...], float]:
        return next(self._product)


class CompositeConditionalEnumerator(DistributionEnumerator):
    """Best-first enumerator for complete tuples consistent with fixed component values."""

    def __init__(self, dist: CompositeDistribution, given: dict[int, Any]) -> None:
        """Enumerate complete tuples consistent with the fixed positions ``given``, best-first.

        Best-first over the product of the *free* coordinates' enumerations, offset by the fixed
        coordinates' summed log-density so each emitted score is the full joint ``log_density``. An
        impossible fixed value (``-inf`` under its component) makes the support empty.

        Args:
            dist (CompositeDistribution): Distribution whose conditional support is enumerated.
            given (dict): Fixed ``{position: value}`` assignments (already validated by the caller).
        """
        super().__init__(dist)
        free_idx = [i for i in range(dist.count) if i not in given]
        with np.errstate(divide="ignore"):
            fixed_lp = sum(dist.dists[i].log_density(v) for i, v in given.items())
        if fixed_lp == -np.inf:
            self._product: Any = iter(())
            return

        def combine(free_values: Sequence[Any], _given=given, _free_idx=free_idx) -> tuple[Any, ...]:
            slots = dict(_given)
            slots.update(zip(_free_idx, free_values))
            return tuple(slots[i] for i in range(len(slots)))

        streams = [
            BufferedStream(child_enumerator(dist.dists[i], "CompositeDistribution.dists[%d]" % i)) for i in free_idx
        ]
        self._product = ProductEnumerator(streams, combine=combine, offset=float(fixed_lp))

    def __next__(self) -> tuple[tuple[Any, ...], float]:
        return next(self._product)


class CompositeSampler(DistributionSampler):
    """Sampler that draws each component independently from its own sampler."""

    def __init__(self, dist: CompositeDistribution, seed: int | None = None) -> None:
        """CompositeSampler used to generate samples from CompositeDistribution.

        Args:
            dist (CompositeDistribution): CompositeDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Attributes:
            dist (CompositeDistribution): CompositeDistribution to draw samples from.
            rng (RandomState): RandomState with seed set if provided.
            dist_samplers (List[DistributionSamplers]): List of DistributionSamplers for each component
                (len=len(dists)).
        """
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_samplers = [d.sampler(seed=self.rng.randint(maxrandint)) for d in dist.dists]

    def sample(self, size: int | None = None) -> list[tuple[Any, ...]] | tuple[Any, ...]:
        """Generate independent samples from a CompositeDistribution.

        If size is None, draw one sample and return as Tuple of length = len(dists). If size > 0,
        draw size samples and return a list of length size containing tuples of len(dists).

        Args:
            size (Optional[int]): If None, draw 1 sample. Else, draw size number of iid samples.

        Returns:
            A tuple of length = len(dists) or a list of length size containing tuples of length = len(dists).

        """
        if size is None:
            return tuple([d.sample(size=size) for d in self.dist_samplers])

        else:
            return list(zip(*[d.sample(size=size) for d in self.dist_samplers]))


class CompositeAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator that delegates sufficient-statistic updates to component accumulators."""

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator], keys: str | None = None) -> None:
        """Create an accumulator for the component sufficient statistics of a CompositeDistribution.

        Args:
            accumulators (List[SequenceEncodableStatisticAccumulator]):
            keys (Optional[str]): All CompositeAccumulators with same keys will have suff-stats merged.

        Attributes:
            accumulators (List[SequenceEncodableStatisticAccumulator]): List of SequenceEncodableStatisticAccumulator
                objects for accumulating sufficient statsitics for each component of the CompositeDistribution.
            count (int): Length of accumulators.
            keys (Optional[str]): All CompositeAccumulators with same keys will have suff-stats merged.
            _init_rng (bool): Is True if _acc_rng has been set by a single function call to initialize.
            _acc_rng (List[RandomState]): Random states generated from seeds set by ``rng`` in ``initialize``.

        """
        self.accumulators = accumulators
        self.count = len(accumulators)
        self.keys = keys

        ### variables for initialization
        self._init_rng = False
        self._acc_rng: list[RandomState] | None = None

    def update(self, x: T, weight: float, estimate: CompositeDistribution | None) -> None:
        """Calls update on each CompositeAccumulator component[k], passing x[k] and weight along with estimate
            if provided.

        Component-wise update() calls to accumulator for each component of x. The same weight is passed to each update
        call, along with the corresponded component-distribution estimate, if estimate is provided.

        Args:
            x (Any): Category label.
            weight (float): Weight for the observation x.
            estimate (Optional['CategoricalDistribution']): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        if estimate is not None:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, estimate.dists[i])

        else:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, None)

    def _rng_initialize(self, rng: RandomState) -> None:
        seeds = rng.randint(2**31, size=self.count)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]

    def initialize(self, x: tuple[Any, ...], weight: float, rng: np.random.RandomState) -> None:
        """Initialize each accumulator of CompositeAccumulator with component x[i] of x and weight.

        Note: rng is used to set List[RandomState]: _acc_rng. This is done to ensure iteration over observations of data,
        produces the same initialization as seq_initialize().

        Args:
            x (Tuple[Any, ...]): Observation Tuple of length count, that is component-wise compatible with
                CompositeAccumulator member variable accumulators.
            weight (float): Weight for the observation x.
            rng (RandomState): Used to set seed of _acc_rng if not set.

        Returns:
            None

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(0, self.count):
            self.accumulators[i].initialize(x[i], weight, self._acc_rng[i])

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of each accumulator of CompositeAccumulator with encoded data x.

        Note: rng is used to set List[RandomState]: _acc_rng. This is done to ensure iteration over observations of
        data, produces the same initialization as seq_initialize().

        Args:
            x (E): Tuple of component wise sequence encoding of data.
            weights (np.ndarray): Numpy array weights for the encoded observations.
            rng (RandomState): Used to set seed of _acc_rng if not set.

        Returns:
            None

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(0, self.count):
            self.accumulators[i].seq_initialize(x[i], weights, self._acc_rng[i])

    def get_seq_lambda(self) -> list[Any]:
        """Return low-level sequence kernels from all component accumulators."""
        rv = []
        for i in range(self.count):
            rv.extend(self.accumulators[i].get_seq_lambda())
        return rv

    def seq_update(self, x: tuple[Any, ...], weights: np.ndarray, estimate: CompositeDistribution | None) -> None:
        """Vectorized aggregation of sufficient statistics for each component of CompositeAccumulator.

        Requires sequence encoded input x, from CompositeDataEncoder.seq_encode(data).

        Args:
            x (Tuple[Any, ...]): Encoded sequence Tuple of length count, that is a component wise sequence encoding of
                data.
            weights (np.ndarray): Numpy array weights for the encoded observations.
            estimate:

        Returns:
            None.

        """
        for i in range(self.count):
            self.accumulators[i].seq_update(x[i], weights, estimate.dists[i] if estimate is not None else None)

    def seq_update_engine(
        self, x: tuple[Any, ...], weights: Any, estimate: CompositeDistribution | None, engine: Any
    ) -> None:
        """Engine-resident E-step: route each component accumulator through the active engine so
        nested families stay resident. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        for i in range(self.count):
            child_seq_update(
                self.accumulators[i], x[i], weights, estimate.dists[i] if estimate is not None else None, engine
            )

    def combine(self, suff_stat: SS) -> CompositeAccumulator:
        """Aggregate the sufficient statistics of CompositeAccumulator with input suff_stat.

        Args:
            suff_stat (SS): Tuple of sufficient statistics for each component of the CompositeAccumulator.

        Returns:
            None

        """
        for i in range(0, self.count):
            self.accumulators[i].combine(suff_stat[i])

        return self

    def value(self) -> tuple[Any, ...]:
        """Return one sufficient-statistic value per component accumulator."""
        return tuple([x.value() for x in self.accumulators])

    def from_value(self, x: SS) -> CompositeAccumulator:
        """Set CompositeAccumulator instance sufficient statistics to x.

        Args:
            x (SS): Tuple of length equal to member variable count, containing sufficient statistics
                for each component.

        Returns:
            CompositeAccumulator

        """
        self.accumulators = [self.accumulators[i].from_value(x[i]) for i in range(len(x))]
        self.count = len(x)

        return self

    def scale(self, c: float) -> CompositeAccumulator:
        """Scale each child accumulator using its family-specific protocol."""
        for acc in self.accumulators:
            acc.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Combines the sufficient statistics of CompositeAccumulators that have the same key value.

        If key is not in the stats_dict (dictionary), the key and accumulator are added to the dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary for mapping keys to CompositeAccumulators.

        Returns:
            None

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Set CompositeAccumulator sufficient statistic attributes values to suff stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Maps member variable key to
                CompositeAccumulator with same key.

        Returns:
            None

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> CompositeDataEncoder:
        """Return a tuple encoder assembled from the child accumulator encoders.

        encoders is a list of DataSequenceEncoders for each component of the CompositeDistribution.

        Returns:
            CompositeDataEncoder

        """
        encoders = tuple([acc.acc_to_encoder() for acc in self.accumulators])

        return CompositeDataEncoder(encoders=encoders)


class CompositeAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for composite accumulators built from component accumulator factories."""

    def __init__(self, factories: Sequence[StatisticAccumulatorFactory], keys: str | None = None) -> None:
        """CompositeAccumulatorFactory used for lightweight creation of CompositeAccumulator.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): List of StatisticAccumulatorFactory objects for each
                component.
            keys (Optional[str]): Declare keys for merging sufficient statistics of CompositeAccumulator objects.

        Attributes:
            factories (List[StatisticAccumulatorFactory]): List of StatisticAccumulatorFactory objects for each
                component.
            keys (Optional[str]): Declare keys for merging sufficient statistics of CompositeAccumulator objects.
        """
        self.factories = factories
        self.keys = keys

    def make(self) -> CompositeAccumulator:
        """Create a composite accumulator from the component factories.

        Returns:
            CompositeAccumulator

        """
        return CompositeAccumulator([u.make() for u in self.factories], self.keys)


class CompositeEstimator(ParameterEstimator):
    """Estimator that fits each independent component distribution from its own statistics."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        keys: str | None = None,
        prior: Sequence[Any] | None = None,
    ) -> None:
        """Create an estimator for a composite distribution from component sufficient statistics.

        Args:
            estimators (List[ParameterEstimator]): List of ParameterEstimator objects for each component of
                CompositeEstimator.
            keys (Optional[str]): Keys used for merging sufficient statistics of CompositeEstimator objects.
            prior (Optional): Per-component parameter priors distributed to the child estimators via
                ``set_prior``. ``None`` (default) leaves each child estimator's prior untouched, so the
                MLE path stays byte-identical. Each child estimator performs its own conjugate update.

        Attributes:
            estimators (List[ParameterEstimator]): List of ParameterEstimator objects for each component of
                CompositeEstimator.
            keys (Optional[str]): Keys used for merging sufficient statistics of CompositeEstimator objects.
            count (int): Number of components in CompositeEstimator.

        """
        self.estimators = estimators
        self.count = len(estimators)
        self.keys = keys
        self.set_prior(prior)

    def get_prior(self) -> list[Any]:
        """Return the joint prior as the list of per-component child estimator priors (in order)."""
        return [est.get_prior() for est in self.estimators]

    def set_prior(self, prior: Sequence[Any] | None) -> None:
        """Distribute per-component parameter priors to the child estimators.

        ``prior=None`` is a no-op (children keep their existing priors). Otherwise ``prior`` must be a
        sequence of exactly ``count`` priors pushed to the children via their own ``set_prior``.
        """
        if prior is None:
            return
        prior = list(prior)
        if len(prior) != self.count:
            raise ValueError("CompositeEstimator.set_prior expected %d priors but got %d." % (self.count, len(prior)))
        for est, p in zip(self.estimators, prior):
            _distribute_child_prior(est, p)

    def model_log_density(self, model: CompositeDistribution) -> float:
        """Sum the child estimators' ``model_log_density`` on the corresponding child models (ELBO global term)."""
        rv = 0.0
        for est, d in zip(self.estimators, model.dists):
            rv += est.model_log_density(d)
        return rv

    def accumulator_factory(self) -> CompositeAccumulatorFactory:
        """Return an accumulator factory assembled from the child estimators."""
        return CompositeAccumulatorFactory([u.accumulator_factory() for u in self.estimators], self.keys)

    def estimate(self, nobs: float | None, suff_stat: SS) -> CompositeDistribution:
        """Estimate a CompositeDistribution from an aggregated sufficient statistics Tuple for a given number of
            observations (nobs).

        Args:
            nobs (Optional[float]): Weighted number of observations used to form suff_stat.
            suff_stat (SS): Tuple of sufficient statistics for each ParameterEstimator of estimators.

        Returns:
            CompositeDistribution estimated from argument aggregated sufficient statistics (suff_stat), from a given
                number of observation (nobs).

        """
        if not isinstance(suff_stat, (tuple, list)):
            raise ContractError(
                "CompositeEstimator.estimate(suff_stat)",
                "a tuple of %d component sufficient statistics" % self.count,
                "%s" % type(suff_stat).__name__,
                "pass the tuple produced by CompositeAccumulator.value(), not a single component's "
                "sufficient statistic.",
            )
        if len(suff_stat) != self.count:
            raise ContractError(
                "CompositeEstimator.estimate(suff_stat)",
                "a tuple of length %d (one sufficient statistic per component)" % self.count,
                "a tuple of length %d" % len(suff_stat),
                "the suff_stat tuple must have exactly %d entries, matching CompositeEstimator's "
                "%d component estimators -- a mismatched CompositeAccumulator/CompositeEstimator "
                "pairing is the usual cause." % (self.count, self.count),
            )
        components = []
        for i, (est, ss) in enumerate(zip(self.estimators, suff_stat)):
            try:
                components.append(est.estimate(nobs, ss))
            except ContractError as e:
                raise prefix_contract_error("CompositeEstimator.estimators[%d]" % i, e) from None
        return CompositeDistribution(tuple(components))


class CompositeDataEncoder(DataSequenceEncoder):
    """Encoder that applies each component encoder to the corresponding tuple field."""

    def __init__(self, encoders: Sequence[DataSequenceEncoder]) -> None:
        """CompositeDataEncoder used for encoding data.

        Data must be of form Sequence[Tuple[Any,...]]. Each encoder component must be compatible with each data
            component of the data.

        Args:
            encoders (Sequence[DataSequenceEncoder]): DataSequenceEncoders for each component of the
                CompositeDistribution.

        Attributes:
            encoders (Sequence[DataSequenceEncoder]): DataSequenceEncoders for each component of the
                CompositeDistribution.

        """
        self.encoders = encoders

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent composite data encoder.

        If other is CompositeDataEncoder, it must also have an equivalent encoder for each
        component of encoder member variable.

        Args:
            other (object): Object to be compared to CompositeDataEncoder.

        Returns:
            True if other can produce and equivalent encoding to instance of CompositeDataEncoder.

        """
        if not isinstance(other, CompositeDataEncoder):
            return False

        else:
            for i, encoder in enumerate(self.encoders):
                if not encoder == other.encoders[i]:
                    return False

        return True

    def __str__(self) -> str:
        """Return a constructor-style representation of the component encoders."""

        s = "CompositeDataEncoder(["

        for d in self.encoders[:-1]:
            s += str(d) + ","

        s += str(self.encoders[-1]) + "])"

        return s

    def seq_encode(self, x: Sequence[tuple[Any, ...]]) -> tuple[Any, ...]:
        """Encode tuple-valued observations for vectorized ``seq_*`` methods.

        The input x must be a Sequence of Tuples of length equal to the length of encoders. Each component tuple
        observation of x, say x[i], must be component-wise compatible with encoders.

        Args:
            x (Sequence[Tuple[Any, ...]]): Sequence of tuples of length equal to len(encoders).

        Returns:
            Tuple of length equal to len(encoders), with entry i, containing the sequence encoding from encoder[i]
            for all observations of component i from x.

        """
        count = len(self.encoders)
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise ContractError(
                "CompositeDistribution.seq_encode",
                "a sequence of %d-tuples" % count,
                "%s" % type(x).__name__,
                "pass a list/tuple of observations, e.g. [(x0, x1, ...), ...].",
            )
        # Validation in one C-speed pass (set(map(len, x)) -- the per-row python loop this replaces was
        # the single largest encode cost at 1M rows); the loop below runs only on the ERROR path, so the
        # per-row contract messages are byte-identical when something is actually wrong.
        try:
            row_lens = set(map(len, x))
            row_types_ok = all(issubclass(tp, (tuple, list, np.ndarray)) for tp in set(map(type, x)))
        except TypeError:
            row_lens, row_types_ok = None, False
        if row_lens is None or row_lens - {count} or not row_types_ok:
            for row_idx, u in enumerate(x):
                if not isinstance(u, (tuple, list, np.ndarray)):
                    raise ContractError(
                        "CompositeDistribution.dists (row %d)" % row_idx,
                        "a tuple of %d fields (one per component distribution)" % count,
                        "%s" % type(u).__name__,
                        "wrap the observation in a %d-tuple matching the component distributions." % count,
                    )
                if len(u) != count:
                    raise ContractError(
                        "CompositeDistribution.dists (row %d)" % row_idx,
                        "a tuple of length %d" % count,
                        "a tuple of length %d" % len(u),
                        "check row %d for a missing or extra field -- every row must have exactly %d "
                        "entries, one per component distribution." % (row_idx, count),
                    )

        enc_data = []

        for i, encoder in enumerate(self.encoders):
            field_path = "CompositeDistribution.dists[%d]" % i
            try:
                enc_data.append(encoder.seq_encode([u[i] for u in x]))
            except ContractError as e:
                raise prefix_contract_error(field_path, e) from None
            except (TypeError, ValueError, IndexError, KeyError) as e:
                raise ContractError(
                    field_path,
                    "data compatible with component %d's data type" % i,
                    "data that raised %s: %s" % (type(e).__name__, e),
                    "check that field %d of every row matches the data type expected by component %d "
                    "(%s)." % (i, i, encoder),
                ) from e

        return tuple(enc_data)


# --- Fisher view(s) co-located with this family ---
class CompositeFisherView(FixedFisherView):
    """Fisher view that concatenates component sufficient-statistic vectors."""

    def __init__(self, dist: Any) -> None:
        self.child_views = [to_fisher(d) for d in dist.dists]
        labels: list[Path] = []
        for i, view in enumerate(self.child_views):
            labels.extend((str(i),) + label for label in view.vectorizer.labels)
        super().__init__(dist, labels)

    def _refresh_labels(self) -> None:
        labels: list[Path] = []
        for i, view in enumerate(self.child_views):
            labels.extend((str(i),) + label for label in view.vectorizer.labels)
        self.labels = labels
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        mats = []
        ests = [None] * len(self.child_views) if estimate is None else estimate.dists
        for i, view in enumerate(self.child_views):
            child_data = [x[i] for x in data]
            mats.append(view.expected_statistics_matrix(data=child_data, estimate=ests[i]))
        self._refresh_labels()
        return np.hstack(mats) if mats else np.zeros((len(data), 0), dtype=np.float64)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        mats = []
        ests = [None] * len(self.child_views) if estimate is None else estimate.dists
        for i, view in enumerate(self.child_views):
            mats.append(view.seq_expected_statistics(enc_data[i], estimate=ests[i]))
        self._refresh_labels()
        n = mats[0].shape[0] if mats else 0
        return np.hstack(mats) if mats else np.zeros((n, 0), dtype=np.float64)

    def _model_mean(self) -> np.ndarray:
        return np.concatenate([view.mean_statistics() for view in self.child_views])

    def _model_fisher(self) -> np.ndarray:
        blocks = [np.asarray(view.fisher_information(ridge=0.0), dtype=np.float64) for view in self.child_views]
        dim = sum(block.shape[0] for block in blocks)
        out = np.zeros((dim, dim), dtype=np.float64)
        pos = 0
        for block in blocks:
            n = block.shape[0]
            out[pos : pos + n, pos : pos + n] = block
            pos += n
        return out
