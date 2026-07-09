"""Hidden association models over grouped-count set pairs.

An observation is a pair ``(given_counts, emitted_counts)`` where each side is
a list of ``(value, count)`` entries. The first side defines an empirical
mixture over latent values, and the second side is generated from a conditional
emission model marginalized over that empirical mixture.

For grouped counts ``x`` and emissions ``y``, the model scores:

    log p(x, y) = log p_given(x) + log p_len(|y|)
        + sum_y count_y log sum_x empirical_x cond(y | x)

This representation keeps repeated values compact while preserving the same
distributional contract as iid samples expanded by count.
"""

import math
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, frontier_merge
from mixle.stats.combinator.conditional import (
    ConditionalDistribution,
    ConditionalDistributionAccumulator,
    ConditionalDistributionAccumulatorFactory,
    ConditionalDistributionEstimator,
)
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.utils.optsutil import count_by_value

T = TypeVar("T")  ### value data type
SS1 = TypeVar("SS1")  ### Data type for suff stats of conditional
SS2 = TypeVar("SS2")  ### Data type for suff stats of given
SS3 = TypeVar("SS3")  ### Data type for suff stats of length


class HiddenAssociationDistribution(SequenceEncodableProbabilityDistribution):
    """Hidden association model: values of a second set are emitted conditionally on values drawn from a first set."""

    def __init__(
        self,
        cond_dist: ConditionalDistribution,
        given_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create a hidden association distribution.

        Args:
            cond_dist: Conditional emission distribution for emitted values
                given latent/given values.
            given_dist: Distribution for grouped counts on the given side.
            len_dist: Distribution for the total emitted count.
            name: Optional diagnostic name.
            keys: Optional merge keys for child sufficient statistics.

        Attributes:
            cond_dist: Conditional emission distribution.
            given_dist: Distribution for the given grouped counts.
            len_dist: Distribution for emitted count.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic merge keys.

        """
        self.cond_dist = cond_dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.given_dist = given_dist if given_dist is not None else NullDistribution()
        self.name = name
        self.keys = keys if keys is not None else (None, None)

    def compute_capabilities(self):
        """Return backend capability metadata for this concrete hidden association model."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(
            engine_ready=intersect_engine_ready((self.cond_dist, self.given_dist, self.len_dist)),
            kernel_status="generic_latent",
        )

    def compute_declaration(self):
        """Return the generated-compute declaration for hidden association."""
        from mixle.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        conditional = declaration_for(self.cond_dist)
        given = None if supports(self.given_dist, Neutral) else declaration_for(self.given_dist)
        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = tuple(child for child in (conditional, given, length) if child is not None)
        roles = ()
        if conditional is not None:
            roles += ("conditional",)
        if given is not None:
            roles += ("given",)
        if length is not None:
            roles += ("length",)
        return DistributionDeclaration(
            name="hidden_association",
            distribution_type=type(self),
            parameters=(),
            statistics=(
                StatisticSpec("conditional", kind="child_stat"),
                StatisticSpec("given", kind="child_stat"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="hidden_association_grouped_counts",
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        s1 = repr(self.cond_dist)
        s2 = repr(self.given_dist)
        s3 = repr(self.len_dist)
        s4 = repr(self.name)
        s5 = repr(self.keys)
        return "HiddenAssociationDistribution(%s, given_dist=%s, len_dist=%s, name=%s, keys=%s)" % (s1, s2, s3, s4, s5)

    def log_density(self, x: tuple[list[tuple[T, float]], list[tuple[T, float]]]) -> float:
        """Log-density of the hidden association model at observation x.

        For each emitted value in x[1], marginalizes the conditional emission density over the given
        values in x[0] weighted by their counts, then adds the log-density of the given set under
        given_dist and of the total emission count under len_dist.

        Args:
            x (Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]): Grouped-count observation
                ([(given value, count)], [(emitted value, count)]).

        Returns:
            Log-density at observation x.

        """
        rv = 0
        nn = 0
        for x1, c1 in x[1]:
            cc = 0  ## count for counts in given
            nn += c1
            ll = -np.inf
            for x0, c0 in x[0]:
                tt = self.cond_dist.log_density((x0, x1)) + math.log(c0)
                cc += c0

                if tt == -np.inf:
                    continue

                if ll > tt:
                    ll = math.log1p(math.exp(tt - ll)) + ll
                else:
                    ll = math.log1p(math.exp(ll - tt)) + tt

            if ll == -np.inf or cc <= 0:
                # Empty/all-zero given-set: no association mass for this emitted
                # value. Matches backend_seq_log_density's -inf handling and
                # avoids -inf - (-inf) = NaN from the log(cc) normalizer.
                rv += -np.inf * c1
            else:
                ll -= math.log(cc)
                rv += ll * c1

        rv += self.given_dist.log_density(x[0])
        rv += self.len_dist.log_density(nn)

        return rv

    def seq_log_density(self, x: list[tuple[list[tuple[T, float]], list[tuple[T, float]]]]) -> np.ndarray:
        """Evaluation of log-density at sequence encoded input x (loops over log_density).

        Args:
            x (List[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): Sequence encoded observations from
                HiddenAssociationDataEncoder.seq_encode() (the observations themselves).

        Returns:
            Numpy array of log-density values, one per observation.

        """
        return np.asarray([self.log_density(xx) for xx in x])

    def backend_seq_log_density(
        self, x: Sequence[tuple[list[tuple[T, float]], list[tuple[T, float]]]], engine: Any
    ) -> Any:
        """Evaluate encoded log-densities through distribution-owned backend composition."""
        from mixle.stats.compute.backend import backend_seq_log_density

        assoc_scores = []
        emit_lengths = []

        for given_obs, emitted_obs in x:
            emit_counts = [c1 for _, c1 in emitted_obs]
            emit_lengths.append(sum(emit_counts))

            if not emitted_obs:
                assoc_scores.append(engine.asarray(0.0))
                continue
            if not given_obs:
                assoc_scores.append(engine.asarray(float("-inf")))
                continue

            pairs = []
            given_counts = []
            for x0, c0 in given_obs:
                given_counts.append(c0)
            for x1, _ in emitted_obs:
                for x0, _ in given_obs:
                    pairs.append((x0, x1))

            cond_enc = self.cond_dist.dist_to_encoder().seq_encode(pairs)
            pair_scores = backend_seq_log_density(self.cond_dist, cond_enc, engine)
            pair_scores = pair_scores.reshape((len(emitted_obs), len(given_obs)))
            given_count_array = engine.asarray(np.asarray(given_counts, dtype=np.float64))
            emit_count_array = engine.asarray(np.asarray(emit_counts, dtype=np.float64))

            weighted_scores = pair_scores + engine.log(given_count_array).reshape((1, -1))
            per_emitted = engine.logsumexp(weighted_scores, axis=1) - engine.log(engine.sum(given_count_array))
            assoc_scores.append(engine.sum(per_emitted * emit_count_array))

        rv = engine.stack(assoc_scores) if assoc_scores else engine.zeros(0)

        given_enc = self.given_dist.dist_to_encoder().seq_encode([xx[0] for xx in x])
        rv = rv + backend_seq_log_density(self.given_dist, given_enc, engine)

        len_enc = self.len_dist.dist_to_encoder().seq_encode(emit_lengths)
        rv = rv + backend_seq_log_density(self.len_dist, len_enc, engine)

        return rv

    def emission_mixture(self, s1: list[tuple[T, float]]) -> MixtureDistribution | None:
        """The per-emission distribution ``q(.|S1)`` as a mixture, or None for an empty/degenerate S1.

        ``q(emitted|S1) = sum_u (c_u/n1) P(emitted|u)`` is a finite mixture of the conditional emission
        distributions ``cond_dist.dmap[u]`` weighted by the given-bag's normalized counts -- enumerable
        whenever those component distributions are. Requires ``cond_dist`` to be a
        :class:`ConditionalDistribution` (so the per-given components are available).
        """
        if not isinstance(self.cond_dist, ConditionalDistribution):
            raise EnumerationError(self, reason="enumeration requires cond_dist to be a ConditionalDistribution")
        n1 = float(sum(c for _, c in s1))
        if not s1 or n1 <= 0.0:
            return None
        comps, weights = [], []
        for u, c in s1:
            comps.append(self.cond_dist.dmap.get(u, self.cond_dist.default_dist))
            weights.append(c / n1)
        return MixtureDistribution(comps, weights)

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate ``(S1, S2)`` observations in descending probability order.

        The model factors as ``given_dist(S1) * [prod_e q(e|S1)^{c_e}] * P_len(n)``: the emitted bag S2
        is drawn iid from the per-given mixture ``q(.|S1)`` (see :meth:`emission_mixture`). Enumeration
        is a conditional product -- the outer stream enumerates S1 from ``given_dist`` and, for each S1,
        the inner stream enumerates S2 as a multiset best-first search over ``q(.|S1)``'s own enumeration
        under ``len_dist``, merged by descending total score with ``given_dist(S1)`` as the frontier
        bound. Requires an enumerable non-null ``given_dist`` and a ConditionalDistribution ``cond_dist``.
        """
        if supports(self.given_dist, Neutral):
            raise EnumerationError(self, reason="enumeration requires a non-null given_dist over the S1 bags")
        if not isinstance(self.cond_dist, ConditionalDistribution):
            raise EnumerationError(self, reason="enumeration requires cond_dist to be a ConditionalDistribution")
        return HiddenAssociationEnumerator(self)

    def sampler(self, seed: int | None = None) -> "HiddenAssociationSampler":
        """Return a sampler for iid grouped-count pairs.

        Requires non-null given_dist and len_dist.

        Args:
            seed: Optional random seed.

        Returns:
            A configured ``HiddenAssociationSampler``.

        """
        return HiddenAssociationSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HiddenAssociationEstimator":
        """Return an estimator built from the child estimators.

        Args:
            pseudo_count: Unused; accepted for protocol consistency.

        Returns:
            A configured ``HiddenAssociationEstimator``.

        """
        return HiddenAssociationEstimator(
            cond_estimator=self.cond_dist.estimator(),
            given_estimator=self.given_dist.estimator(),
            len_estimator=self.len_dist.estimator(),
            name=self.name,
        )

    def dist_to_encoder(self) -> "HiddenAssociationDataEncoder":
        """Return an encoder for grouped-count hidden-association observations."""
        return HiddenAssociationDataEncoder()


class HiddenAssociationEnumerator(DistributionEnumerator):
    """Enumerate hidden-association support pairs in best-first order."""

    def __init__(self, dist: "HiddenAssociationDistribution") -> None:
        """Conditional-product enumeration of ``(S1, S2)`` (S1 from given_dist, S2 from q(.|S1)).

        Args:
            dist (HiddenAssociationDistribution): Distribution whose support is enumerated.
        """
        super().__init__(dist)
        # Imported lazily: int_plsi decorates numba kernels at import time, and this
        # enumerator is rarely constructed -- keep `import mixle.stats` lightweight.
        from mixle.stats.latent.integer_probabilistic_latent_semantic_indexing import bag_stream

        len_dist = dist.len_dist

        def make_inner(s1, lp1):
            q = dist.emission_mixture(s1)
            if q is None:
                return iter(())  # degenerate empty S1

            def combine(pairs, s1=s1):
                return (s1, list(pairs))

            element_stream = child_enumerator(q, "HiddenAssociationDistribution.cond_dist")
            return ((value, lp1 + lp2) for value, lp2 in bag_stream(element_stream, len_dist, combine))

        outer = BufferedStream(child_enumerator(dist.given_dist, "HiddenAssociationDistribution.given_dist"))
        self._merge = frontier_merge(outer, make_inner)

    def __next__(self):
        return next(self._merge)


class HiddenAssociationSampler(DistributionSampler):
    """Sampler for grouped-count set pairs from a hidden-association distribution."""

    def __init__(self, dist: HiddenAssociationDistribution, seed: int | None = None) -> None:
        """Create a sampler for a hidden-association distribution.

        Args:
            dist (HiddenAssociationDistribution): Distribution to sample from. Must have non-null
                ``given_dist`` and ``len_dist``.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            dist (HiddenAssociationDistribution): Distribution to sample from.
            cond_sampler (ConditionalSampler): Sampler for the conditional emission distribution.
            idx_sampler (RandomState): RandomState for drawing latent given-value indices.
            len_sampler (DistributionSampler): Sampler for the number of emitted values.
            given_sampler (DistributionSampler): Sampler for the given set.

        """
        if supports(dist.given_dist, Neutral):
            raise Exception("HiddenAssociationSampler requires attribute dist.given_dist.")
        if supports(dist.len_dist, Neutral):
            raise Exception("HiddenAssociationSampler requires attribute dist.len_dist.")

        self.rng = np.random.RandomState(seed)
        self.dist = dist

        self.cond_sampler = dist.cond_dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.idx_sampler = np.random.RandomState(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.given_sampler = self.dist.given_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(
        self, size: int | None = None
    ) -> (
        Sequence[tuple[list[tuple[Any, float]], list[tuple[Any, float]]]]
        | tuple[list[tuple[Any, float]], list[tuple[Any, float]]]
    ):
        """Draw iid grouped-count observations from the hidden association model.

        Args:
            size (Optional[int]): Number of observations to draw. If None, a single observation is returned.

        Returns:
            A ([(given value, count)], [(emitted value, count)]) tuple if size is None, else a list of
            such tuples of length size.

        """
        if size is None:
            prev_obs = self.given_sampler.sample()
            cnt = self.len_sampler.sample()
            rng = np.random.RandomState(self.idx_sampler.randint(0, maxrandint))
            rv = []
            pp = np.asarray([u[1] for u in prev_obs], dtype=float)
            pp /= pp.sum()

            for i in rng.choice(len(prev_obs), p=pp, size=cnt):
                rv.append(self.cond_sampler.sample_given(prev_obs[i][0]))

            rv = list(count_by_value(rv).items())

            return prev_obs, rv

        else:
            return [self.sample() for i in range(size)]

    def sample_given(self, x: list[tuple[T, float]]):
        """Draw an emitted grouped-count set conditioned on the given set x.

        Args:
            x (List[Tuple[T, float]]): Given set as (value, count) pairs.

        Returns:
            List of (emitted value, count) pairs.

        """
        cnt = self.len_sampler.sample()
        rng = np.random.RandomState(self.idx_sampler.randint(0, maxrandint))
        rv = []
        pp = np.asarray([u[1] for u in x], dtype=float)
        pp /= pp.sum()

        for i in rng.choice(len(x), p=pp, size=cnt):
            rv.append(self.cond_sampler.sample_given(x[i][0]))

        rv = list(count_by_value(rv).items())

        return rv


class HiddenAssociationAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for hidden-association sufficient statistics."""

    def __init__(
        self,
        cond_acc: ConditionalDistributionAccumulator,
        given_acc: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        size_acc: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create an accumulator for hidden-association sufficient statistics.

        Args:
            cond_acc: Accumulator for the conditional emission distribution.
            given_acc: Accumulator for the given grouped counts.
            size_acc: Accumulator for the emitted count.
            name: Optional diagnostic name.
            keys: Optional merge keys for child sufficient statistics.

        Attributes:
            cond_accumulator: Accumulator for conditional emissions.
            given_accumulator: Accumulator for given grouped counts.
            size_accumulator: Accumulator for emitted count.
            init_key: Merge key for given-side statistics.
            trans_key: Merge key for conditional statistics.
            name: Optional diagnostic name.

        """
        self.cond_accumulator = cond_acc
        self.given_accumulator = given_acc if given_acc is not None else NullAccumulator()
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        self.init_key, self.trans_key = keys if keys is not None else (None, None)
        self.name = name
        # Data log-likelihood accumulated as a byproduct of the E-step (the per-observation log_density),
        # only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

    def update(
        self,
        x: tuple[list[tuple[T, float]], list[tuple[T, float]]],
        weight: float,
        estimate: HiddenAssociationDistribution,
    ) -> None:
        """Update sufficient statistics with the posterior assignment of emitted values to given values.

        Args:
            x (Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]): Grouped-count observation
                ([(given value, count)], [(emitted value, count)]).
            weight (float): Weight for the observation.
            estimate (HiddenAssociationDistribution): Previous estimate used to compute posteriors.

        """
        nn = 0
        pv = np.zeros(len(x[0]))
        # Data log-density of this observation (== HiddenAssociationDistribution.log_density(x)),
        # reusing the per-emitted logsumexp normalizer ``ll`` the E-step already computes. Only
        # materialized when the fused-EM fast path requests it (_track_ll).
        track = self._track_ll
        obs_ll = 0.0

        for x1, c1 in x[1]:
            cc = 0
            nn += c1
            ll = -np.inf

            for i, (x0, c0) in enumerate(x[0]):
                tt = estimate.cond_dist.log_density((x0, x1)) + math.log(c0)
                cc += c0
                pv[i] = tt

                if tt == -np.inf:
                    continue

                if ll > tt:
                    ll = math.log1p(math.exp(tt - ll)) + ll
                else:
                    ll = math.log1p(math.exp(ll - tt)) + tt

            if ll == -np.inf or cc <= 0:
                # Empty/all-zero given-set: no association mass. Avoid the
                # -inf - (-inf) = NaN normalizer that would corrupt sufficient
                # statistics, and contribute -inf to the tracked log-density to
                # match HiddenAssociationDistribution.log_density.
                if track:
                    obs_ll += -np.inf * c1
                continue

            if track:
                obs_ll += (ll - math.log(cc)) * c1

            pv -= ll
            np.exp(pv, out=pv)

            for i, (x0, c0) in enumerate(x[0]):
                self.cond_accumulator.update((x0, x1), pv[i] * c1 * weight, estimate.cond_dist)

        if track:
            obs_ll += estimate.given_dist.log_density(x[0])
            obs_ll += estimate.len_dist.log_density(nn)
            self._seq_ll += weight * obs_ll

        if self.given_accumulator is not None:
            given_dist = None if estimate is None else estimate.given_dist
            self.given_accumulator.update(x[0], weight, given_dist)

        if self.size_accumulator is not None:
            len_dist = None if estimate is None else estimate.len_dist
            self.size_accumulator.update(nn, weight, len_dist)

    def initialize(
        self, x: tuple[list[tuple[T, float]], list[tuple[T, float]]], weight: float, rng: np.random.RandomState
    ) -> None:
        """Initialize sufficient statistics with random (Dirichlet) assignments of emitted to given values.

        Args:
            x (Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]): Grouped-count observation
                ([(given value, count)], [(emitted value, count)]).
            weight (float): Weight for the observation.
            rng (np.random.RandomState): Random number generator for the random assignments.

        """
        w = rng.dirichlet(np.ones(len(x[0])), size=len(x[1]))
        nn = 0
        for j, (x1, c1) in enumerate(x[1]):
            nn += c1
            for i, (x0, c0) in enumerate(x[0]):
                self.cond_accumulator.initialize((x0, x1), w[j, i] * c1 * weight, rng)

        if self.given_accumulator is not None:
            self.given_accumulator.initialize(x[0], weight, rng)

        if self.size_accumulator is not None:
            self.size_accumulator.initialize(nn, weight, rng)

    def seq_initialize(
        self,
        x: Sequence[tuple[list[tuple[T, float]], list[tuple[T, float]]]],
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize sufficient statistics from sequence encoded observations (loops over initialize()).

        Args:
            x (Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): Sequence encoded observations.
            weights (np.ndarray): Weights, one per observation.
            rng (np.random.RandomState): Random number generator for the random assignments.

        """
        for i, xx in enumerate(x):
            self.initialize(xx, weights[i], rng)

    def seq_update(
        self,
        x: Sequence[tuple[list[tuple[T, float]], list[tuple[T, float]]]],
        weights: np.ndarray,
        estimate: HiddenAssociationDistribution,
    ) -> None:
        """Update sufficient statistics from sequence encoded observations (loops over update()).

        Args:
            x (Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): Sequence encoded observations.
            weights (np.ndarray): Weights, one per observation.
            estimate (HiddenAssociationDistribution): Previous estimate used to compute posteriors.

        """
        for xx, ww in zip(x, weights):
            self.update(xx, ww, estimate)

    def seq_update_engine(
        self,
        x: Sequence[tuple[list[tuple[T, float]], list[tuple[T, float]]]],
        weights: np.ndarray,
        estimate: HiddenAssociationDistribution,
        engine: Any,
    ) -> None:
        """Engine-resident E-step for the hidden association model.

        Builds the batched cross product of (given, emitted) pairs across the whole minibatch,
        scores them through the conditional distribution's engine kernel, and forms the per-emitted
        posterior over given values with a segmented softmax (global-max shift + ``index_add``) on
        the active engine. The engine-computed posterior weights are fed to the conditional
        accumulator; the given/size accumulators are updated per observation. Mirrors ``update``.
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)

        pairs = []
        log_c0 = []
        c1_list = []
        w_list = []
        group_id = []
        emit_lengths = []
        num_groups = 0
        for o, (given_obs, emitted_obs) in enumerate(x):
            emit_lengths.append(sum(c1 for _, c1 in emitted_obs))
            if not emitted_obs or not given_obs:
                continue
            wo = float(weights_np[o])
            log_gc = np.log(np.asarray([c0 for _, c0 in given_obs], dtype=np.float64))
            for x1, c1 in emitted_obs:
                for k, (x0, _) in enumerate(given_obs):
                    pairs.append((x0, x1))
                    log_c0.append(log_gc[k])
                    c1_list.append(c1)
                    w_list.append(wo)
                    group_id.append(num_groups)
                num_groups += 1

        if pairs:
            cond_enc = estimate.cond_dist.dist_to_encoder().seq_encode(pairs)
            pair_scores = backend_seq_log_density(estimate.cond_dist, cond_enc, engine)
            scored = pair_scores + engine.asarray(np.asarray(log_c0, dtype=np.float64))
            gid = engine.asarray(np.asarray(group_id, dtype=np.int64))
            shift = engine.max(scored, axis=0)
            e = engine.exp(scored - shift)
            denom = engine.index_add(engine.zeros(num_groups), gid, e)
            pv = e / denom[gid]
            pair_w = (
                pv
                * engine.asarray(np.asarray(c1_list, dtype=np.float64))
                * engine.asarray(np.asarray(w_list, dtype=np.float64))
            )
            pair_w_np = np.asarray(engine.to_numpy(pair_w), dtype=np.float64)
            self.cond_accumulator.seq_update(cond_enc, pair_w_np, estimate.cond_dist)

        if not supports(self.given_accumulator, Neutral):
            given_enc = self.given_accumulator.acc_to_encoder().seq_encode([xx[0] for xx in x])
            self.given_accumulator.seq_update(given_enc, weights_np, estimate.given_dist)

        if not supports(self.size_accumulator, Neutral):
            size_enc = self.size_accumulator.acc_to_encoder().seq_encode(emit_lengths)
            self.size_accumulator.seq_update(size_enc, weights_np, estimate.len_dist)

    def combine(self, suff_stat: tuple[SS1, SS2 | None, SS3 | None]) -> "HiddenAssociationAccumulator":
        """Merge sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[SS1, Optional[SS2], Optional[SS3]]): Conditional, given, and size suff stats.

        Returns:
            This HiddenAssociationAccumulator.

        """
        cond_acc, given_acc, size_acc = suff_stat

        self.cond_accumulator.combine(cond_acc)
        self.given_accumulator.combine(given_acc)
        self.size_accumulator.combine(size_acc)

        return self

    def value(self) -> tuple[Any, Any | None, Any | None]:
        """Returns the sufficient statistics: (conditional, given, size) accumulator values."""
        return self.cond_accumulator.value(), self.given_accumulator.value(), self.size_accumulator.value()

    def from_value(self, x: tuple[SS1, SS2 | None, SS3 | None]) -> "HiddenAssociationAccumulator":
        """Set the sufficient statistics of this accumulator from x.

        Args:
            x (Tuple[SS1, Optional[SS2], Optional[SS3]]): Conditional, given, and size suff stats.

        Returns:
            This HiddenAssociationAccumulator.

        """
        cond_acc, given_acc, size_acc = x

        self.cond_accumulator.from_value(cond_acc)
        self.given_accumulator.from_value(given_acc)
        self.size_accumulator.from_value(size_acc)

        return self

    def scale(self, c: float) -> "HiddenAssociationAccumulator":
        """Scale sufficient statistics by delegating to child accumulators."""
        self.cond_accumulator.scale(c)
        self.given_accumulator.scale(c)
        self.size_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed statistics of the conditional, given, and size accumulators into stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        self.cond_accumulator.key_merge(stats_dict)
        self.given_accumulator.key_merge(stats_dict)
        self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace keyed statistics of the conditional, given, and size accumulators with those in stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        self.cond_accumulator.key_replace(stats_dict)
        self.given_accumulator.key_replace(stats_dict)
        self.size_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "HiddenAssociationDataEncoder":
        """Return an encoder compatible with hidden-association observations."""
        return HiddenAssociationDataEncoder()


class HiddenAssociationAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for hidden-association accumulators."""

    def __init__(
        self,
        cond_factory: ConditionalDistributionAccumulatorFactory,
        given_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create an accumulator factory.

        Args:
            cond_factory: Factory for the conditional emission accumulator.
            given_factory: Factory for the given grouped-count accumulator.
            len_factory: Factory for the emitted-count accumulator.
            name: Optional diagnostic name.
            keys: Optional merge keys for child sufficient statistics.

        Attributes:
            cond_factory: Factory for conditional emissions.
            given_factory: Factory for given grouped counts.
            len_factory: Factory for emitted counts.
            keys: Optional sufficient-statistic merge keys.
            name: Optional diagnostic name.

        """
        self.cond_factory = cond_factory
        self.given_factory = given_factory if given_factory is not None else NullAccumulatorFactory()
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def make(self) -> "HiddenAssociationAccumulator":
        """Return a fresh hidden-association accumulator."""
        return HiddenAssociationAccumulator(
            self.cond_factory.make(), self.given_factory.make(), self.len_factory.make(), self.name, self.keys
        )


class HiddenAssociationEstimator(ParameterEstimator):
    """Estimator for hidden association distributions."""

    def __init__(
        self,
        cond_estimator: ConditionalDistributionEstimator,
        given_estimator: ParameterEstimator | None = NullEstimator(),
        len_estimator: ParameterEstimator | None = NullEstimator(),
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create an estimator for hidden-association sufficient statistics.

        Args:
            cond_estimator: Estimator for conditional emissions.
            given_estimator: Estimator for the given grouped counts.
            len_estimator: Estimator for emitted count.
            pseudo_count: Unused; accepted for protocol consistency.
            name: Optional diagnostic name.
            keys: Optional merge keys for child sufficient statistics.

        Attributes:
            cond_estimator: Estimator for conditional emissions.
            given_estimator: Estimator for given grouped counts.
            len_estimator: Estimator for emitted count.
            pseudo_count: Parameter retained for estimator protocol compatibility.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic merge keys.

        """
        self.keys = keys if keys is not None else (None, None)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.cond_estimator = cond_estimator
        self.given_estimator = given_estimator if given_estimator is not None else NullEstimator()
        self.name = name

    def accumulator_factory(self) -> "HiddenAssociationAccumulatorFactory":
        """Return an accumulator factory matching this estimator."""
        len_factory = self.len_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()
        cond_factory = self.cond_estimator.accumulator_factory()
        return HiddenAssociationAccumulatorFactory(
            cond_factory=cond_factory,
            given_factory=given_factory,
            len_factory=len_factory,
            name=self.name,
            keys=self.keys,
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[SS1, SS2 | None, SS3 | None]
    ) -> "HiddenAssociationDistribution":
        """Estimate a HiddenAssociationDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations, passed to the given and length estimators.
            suff_stat (Tuple[SS1, Optional[SS2], Optional[SS3]]): Conditional, given, and size suff stats.

        Returns:
            HiddenAssociationDistribution object.

        """
        cond_stats, given_stats, size_stats = suff_stat

        cond_dist = self.cond_estimator.estimate(None, cond_stats)
        given_dist = self.given_estimator.estimate(nobs, given_stats)
        len_dist = self.len_estimator.estimate(nobs, size_stats)

        return HiddenAssociationDistribution(
            cond_dist=cond_dist, given_dist=given_dist, len_dist=len_dist, name=self.name
        )


class HiddenAssociationDataEncoder(DataSequenceEncoder):
    """Encode grouped-count hidden-association observations."""

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return "HiddenAssociationDataEncoder"

    def __eq__(self, other) -> bool:
        """Return whether ``other`` is a hidden-association data encoder."""
        return isinstance(other, HiddenAssociationDataEncoder)

    def seq_encode(
        self, x: Sequence[tuple[list[tuple[T, float]], list[tuple[T, float]]]]
    ) -> Sequence[tuple[list[tuple[T, float]], list[tuple[T, float]]]]:
        """Encode a sequence of iid grouped-count observations (identity encoding).

        Args:
            x (Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): Sequence of iid
                ([(given value, count)], [(emitted value, count)]) observations.

        Returns:
            The observations unchanged (seq_log_density and seq_update loop over them).

        """
        return x


def _register_hidden_association_engine_kernel():
    """Register the engine-resident hidden-association kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class HiddenAssociationKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("HiddenAssociationKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class HiddenAssociationKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return HiddenAssociationKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(HiddenAssociationDistribution, HiddenAssociationKernelFactory())


_register_hidden_association_engine_kernel()
