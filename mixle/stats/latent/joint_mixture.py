"""Joint mixtures over paired observations.

This module models observations of the form ``(x1, x2)`` with separate
component families for each side and a learned conditional association between
their latent component states.

For components ``f_i`` on ``X1`` and ``g_j`` on ``X2``, the paired density is:

    p(x1, x2) = sum_i w1_i f_i(x1) sum_j tau12_ij g_j(x2)

The reverse conditional table ``tau21`` is stored as well so the fitted object
can expose both directions of the paired latent association.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator, best_first_union
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)

T0 = TypeVar("T0")
T1 = TypeVar("T1")
E0 = TypeVar("E0")
E1 = TypeVar("E1")
SS0 = TypeVar("SS0")
SS1 = TypeVar("SS1")


from mixle.inference.fisher import (
    FisherView,
    FixedFisherView,
    SufficientStatisticVectorizer,
    to_fisher,
)
from mixle.stats.latent.mixture import MixtureFisherView
from mixle.utils.aliasing import broadcast_pseudo_count


class JointMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Joint mixture distribution over paired observations.

    Observations are ``(x1, x2)`` tuples. The first tuple element is scored by
    ``components1`` and the second by ``components2``.

    """

    def __init__(
        self,
        components1: Sequence[SequenceEncodableProbabilityDistribution],
        components2: Sequence[SequenceEncodableProbabilityDistribution],
        w1: Sequence[float] | np.ndarray,
        w2: Sequence[float] | np.ndarray,
        taus12: list[list[float]] | np.ndarray,
        taus21: list[list[float]] | np.ndarray,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """Create a paired latent mixture distribution.

        Args:
            components1: Component distributions for the first field ``X1``.
            components2: Component distributions for the second field ``X2``.
            w1: Marginal mixture weights for ``components1``.
            w2: Marginal mixture weights for ``components2``.
            taus12: Conditional weights for ``X2`` component ``j`` given ``X1``
                component ``i``; rows correspond to ``components1``.
            taus21: Conditional weights for ``X1`` component ``i`` given ``X2``
                component ``j``; rows correspond to ``components1``.
            keys: Optional merge keys for joint weights, ``X1`` component
                accumulators, and ``X2`` component accumulators.
            name: Optional diagnostic name.

        Attributes:
            components1: First-field component distributions.
            components2: Second-field component distributions.
            w1: Marginal weights for ``components1``.
            w2: Marginal weights for ``components2``.
            num_components1: Number of first-field components.
            num_components2: Number of second-field components.
            taus12: Conditional ``X2``-given-``X1`` component weights.
            taus21: Conditional ``X1``-given-``X2`` component weights.
            log_w1: Log of ``w1``.
            log_w2: Log of ``w2``.
            log_taus12: Log of ``taus12``.
            log_taus21: Log of ``taus21``.
            keys: Optional sufficient-statistic merge keys.
            name: Optional diagnostic name.

        """
        with np.errstate(divide="ignore"):
            self.components1 = components1
            self.components2 = components2
            self.w1 = vec.make(w1)
            self.w2 = vec.make(w2)
            self.num_components1 = len(components1)
            self.num_components2 = len(components2)
            self.taus12 = np.reshape(taus12, (self.num_components1, self.num_components2))
            self.taus21 = np.reshape(taus21, (self.num_components1, self.num_components2))
            self.log_w1 = np.log(self.w1)
            self.log_w2 = np.log(self.w2)
            self.log_taus12 = np.log(self.taus12)
            self.log_taus21 = np.log(self.taus21)
            self.keys = keys if keys is not None else (None, None, None)
            self.name = name

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        s1 = ",".join([str(u) for u in self.components1])
        s2 = ",".join([str(u) for u in self.components2])
        s3 = ",".join(map(str, self.w1))
        s4 = ",".join(map(str, self.w2))
        s5 = ",".join(map(str, self.taus12.flatten()))
        s6 = ",".join(map(str, self.taus21.flatten()))
        s7 = repr(self.name)

        return "JointMixtureDistribution([%s], [%s], [%s], [%s], [%s], [%s], name=%s)" % (s1, s2, s3, s4, s5, s6, s7)

    def compute_capabilities(self):
        """Intersect generated-compute backend support across all child components."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = tuple(self.components1) + tuple(self.components2)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def compute_declaration(self):
        """Return the generated-compute declaration for the paired latent mixture."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        children1 = tuple(declaration_for(component) for component in self.components1)
        children2 = tuple(declaration_for(component) for component in self.components2)
        children = tuple(child for child in children1 + children2 if child is not None)
        roles = tuple("x1_component_%d" % i for i, child in enumerate(children1) if child is not None)
        roles += tuple("x2_component_%d" % i for i, child in enumerate(children2) if child is not None)
        return DistributionDeclaration(
            name="joint_mixture",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("w1", constraint="simplex_vector"),
                ParameterSpec("w2", constraint="simplex_vector"),
                ParameterSpec("taus12", constraint="row_simplex_matrix"),
                ParameterSpec("taus21", constraint="row_simplex_matrix"),
            ),
            statistics=(
                StatisticSpec("component_counts1"),
                StatisticSpec("component_counts2"),
                StatisticSpec("joint_counts"),
                StatisticSpec("components1", kind="tuple"),
                StatisticSpec("components2", kind="tuple"),
            ),
            support="paired_mixture",
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def density(self, x: tuple[T0, T1]) -> float:
        """Evaluate the density of a joint mixture observation x.

        See log_density() for details.

        Args:
            x (Tuple[T0, T1]): A single (X1, X2) observation.

        Returns:
            Density evaluated at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: tuple[T0, T1]) -> float:
        """Evaluate the log-density of a joint mixture observation x.

        The log-density at x = (x1, x2) is

            log(sum_{i=1}^{N} w_i * f_i(x1) * sum_{j=1}^{M} tau12_{ij} * g_j(x2)),

        evaluated with a log-sum-exp for numerical stability.

        Args:
            x (Tuple[T0, T1]): A single (X1, X2) observation.

        Returns:
            Log-density evaluated at x.

        """
        ll1 = np.zeros((1, self.num_components1))
        ll2 = np.zeros((1, self.num_components2))

        for i in range(self.num_components1):
            ll1[0, i] = self.components1[i].log_density(x[0]) + self.log_w1[i]
        for i in range(self.num_components2):
            ll2[0, i] += self.components2[i].log_density(x[1])

        max1 = ll1.max()
        ll1 -= max1
        np.exp(ll1, out=ll1)

        max2 = np.max(ll2)
        ll2 -= max2
        np.exp(ll2, out=ll2)

        ll12 = np.dot(ll1, self.taus12)
        ll2 *= ll12

        rv = np.log(ll2.sum()) + max1 + max2

        return rv

    def seq_log_density(self, x: tuple[int, E0, E1]) -> np.ndarray:
        """Vectorized evaluation of the log-density for an encoded sequence of observations x.

        Encoded sequence 'x' is a Tuple of length 3 containing:
            x[0] (int): Number of observations.
            x[1] (E0): Encoded sequence of X1 values.
            x[2] (E1): Encoded sequence of X2 values.

        Args:
            x: Encoded sequence of iid joint mixture observations.

        Returns:
            Log-density evaluated at each observation in the encoded sequence x.

        """
        sz, enc_data1, enc_data2 = x
        ll_mat1 = np.zeros((sz, self.num_components1))
        ll_mat2 = np.zeros((sz, self.num_components2))

        for i in range(self.num_components1):
            ll_mat1[:, i] = self.components1[i].seq_log_density(enc_data1)
            ll_mat1[:, i] += self.log_w1[i]

        for i in range(self.num_components2):
            ll_mat2[:, i] = self.components2[i].seq_log_density(enc_data2)

        with np.errstate(divide="ignore", invalid="ignore"):  # -inf max on impossible rows -> handled below
            ll_max1 = ll_mat1.max(axis=1, keepdims=True)
            ll_mat1 -= ll_max1
            np.exp(ll_mat1, out=ll_mat1)

            ll_max2 = ll_mat2.max(axis=1, keepdims=True)
            ll_mat2 -= ll_max2
            np.exp(ll_mat2, out=ll_mat2)

            ll_mat12 = np.dot(ll_mat1, self.taus12)
            ll_mat2 *= ll_mat12

            rv = np.log(ll_mat2.sum(axis=1)) + ll_max1[:, 0] + ll_max2[:, 0]
        # an observation outside the support of either component set has max log-density -inf, which
        # produces nan above; such observations have zero probability
        rv[~(np.isfinite(ll_max1[:, 0]) & np.isfinite(ll_max2[:, 0]))] = -np.inf

        return rv

    def backend_seq_log_density(self, x: tuple[int, E0, E1], engine: Any) -> Any:
        """Engine-neutral log-density for encoded joint-mixture observations."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, enc_data1, enc_data2 = x
        if sz == 0:
            return engine.zeros(0)

        ll1 = []
        for i in range(self.num_components1):
            ll1.append(backend_seq_log_density(self.components1[i], enc_data1, engine))
        ll1 = engine.stack(ll1, axis=1) + engine.asarray(self.log_w1)

        ll2 = []
        for j in range(self.num_components2):
            ll2.append(backend_seq_log_density(self.components2[j], enc_data2, engine))
        ll2 = engine.stack(ll2, axis=1)

        pair_scores = ll1[:, :, None] + engine.asarray(self.log_taus12)[None, :, :] + ll2[:, None, :]
        return engine.logsumexp(pair_scores, axis=(1, 2))

    def to_fisher(self, **kwargs):
        """Structural Fisher view for the joint mixture."""
        if hasattr(self, "components1") and hasattr(self, "components2"):
            return JointMixtureFisherView(self)
        return super().to_fisher(**kwargs)

    def density_semantics(self):
        """Return exact-or-approximate density semantics joined from child components."""
        from mixle.stats.compute.pdist import DensitySemantics, join_density_semantics

        children = list(self.components1) + list(self.components2)
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics")]
        return join_density_semantics(sems) if sems else DensitySemantics.EXACT

    def sampler(self, seed: int | None = None) -> JointMixtureSampler:
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional random seed.

        Returns:
            A configured ``JointMixtureSampler``.

        """
        return JointMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> JointMixtureEstimator:
        """Return an estimator initialized from this distribution's components.

        Args:
            pseudo_count: Optional smoothing count for latent-state counts.

        Returns:
            A ``JointMixtureEstimator``.

        """
        estimators1 = [comp1.estimator() for comp1 in self.components1]
        estimators2 = [comp2.estimator() for comp2 in self.components2]

        return JointMixtureEstimator(
            estimators1=estimators1, estimators2=estimators2, pseudo_count=pseudo_count, keys=self.keys, name=self.name
        )

    def dist_to_encoder(self) -> DataSequenceEncoder:
        """Return an encoder for paired joint-mixture observations."""
        encoder1 = self.components1[0].dist_to_encoder()
        encoder2 = self.components2[0].dist_to_encoder()
        return JointMixtureDataEncoder(encoder1=encoder1, encoder2=encoder2)

    def enumerator(self) -> JointMixtureEnumerator:
        """Return an enumerator over pairs in descending probability order."""
        return JointMixtureEnumerator(self)


class JointMixtureEnumerator(DistributionEnumerator):
    """Enumerates the support of a JointMixtureDistribution in descending probability order."""

    def __init__(self, dist: JointMixtureDistribution) -> None:
        """Enumerates the union of pairwise product supports in descending joint probability order.

        A joint mixture is a mixture over component pairs (i, j) with weight w1_i * tau12_ij and
        product density f_i(x1) * g_j(x2). Each positive-weight pair contributes a best-first
        product stream over the (shared, buffered) component enumerations. Pair supports may
        overlap, so candidates are de-duplicated and re-scored exactly with the joint mixture
        log-density before being emitted (the mixture best-first-union algorithm). Zero-weight
        pairs are never asked to enumerate.

        Args:
            dist (JointMixtureDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        buf1: dict[int, BufferedStream] = {}
        buf2: dict[int, BufferedStream] = {}
        streams = []
        log_offsets = []
        kept_pairs = []

        for i in range(dist.num_components1):
            if dist.w1[i] <= 0.0:
                continue
            for j in range(dist.num_components2):
                if dist.taus12[i, j] <= 0.0:
                    continue
                if i not in buf1:
                    buf1[i] = BufferedStream(
                        child_enumerator(dist.components1[i], "JointMixtureDistribution.components1[%d]" % i)
                    )
                if j not in buf2:
                    buf2[j] = BufferedStream(
                        child_enumerator(dist.components2[j], "JointMixtureDistribution.components2[%d]" % j)
                    )
                streams.append(BufferedStream(ProductEnumerator([buf1[i], buf2[j]], combine=tuple)))
                log_offsets.append(dist.log_w1[i] + dist.log_taus12[i, j])
                kept_pairs.append((i, j))

        log_pair_w = np.asarray(log_offsets, dtype=np.float64)

        # Equivalent to dist.log_density but restricted to positive-weight pairs, so a
        # zero-weight component never sees (possibly type-incompatible) candidate values.
        def exact_log_density(x):
            with np.errstate(divide="ignore"):
                ll = np.asarray(
                    [
                        dist.components1[i].log_density(x[0]) + dist.components2[j].log_density(x[1])
                        for i, j in kept_pairs
                    ]
                )
                return vec.log_sum(ll + log_pair_w)

        self._union = best_first_union(streams, log_offsets, exact_log_density)

    def __next__(self) -> tuple[Any, float]:
        return next(self._union)


class JointMixtureSampler(DistributionSampler):
    """Sampler for paired observations from a joint mixture distribution."""

    def __init__(self, dist: JointMixtureDistribution, seed: int | None = None) -> None:
        """Create a sampler for a joint mixture distribution.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.

        Attributes:
            rng: Random state used for component-state draws.
            dist: Distribution to sample from.
            comp_sampler1: Samplers for the ``X1`` components.
            comp_sampler2: Samplers for the ``X2`` components.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.comp_sampler1 = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in self.dist.components1]
        self.comp_sampler2 = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in self.dist.components2]

    def sample(self, size: int | None = None) -> tuple[Any, Any] | Sequence[tuple[Any, Any]]:
        """Draw iid ``(X1, X2)`` samples from the joint mixture.

        The X1 component state is drawn from w1, X1 is sampled from that component, the X2
        component state is drawn from taus12 given the X1 state, and X2 is sampled from the
        corresponding X2 component.

        Args:
            size: Number of iid samples to draw. ``None`` returns a scalar pair.

        Returns:
            A scalar pair when ``size`` is ``None``; otherwise a list of pairs.

        """
        if size is None:
            comp_state1 = self.rng.choice(range(0, self.dist.num_components1), replace=True, p=self.dist.w1)
            f1 = self.comp_sampler1[comp_state1].sample()
            comp_state2 = self.rng.choice(
                range(0, self.dist.num_components2), replace=True, p=self.dist.taus12[comp_state1, :]
            )
            f2 = self.comp_sampler2[comp_state2].sample()

            return f1, f2
        else:
            return [self.sample() for i in range(size)]


class JointMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for joint-mixture EM sufficient statistics."""

    def __init__(
        self,
        accumulators1: Sequence[SequenceEncodableStatisticAccumulator],
        accumulators2: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """Create an accumulator for paired-mixture sufficient statistics.

        Args:
            accumulators1: Component accumulators for ``X1``.
            accumulators2: Component accumulators for ``X2``.
            keys: Optional merge keys for joint counts, ``X1`` accumulators, and
                ``X2`` accumulators.
            name: Optional diagnostic name.

        Attributes:
            accumulators1: Component accumulators for ``X1``.
            accumulators2: Component accumulators for ``X2``.
            keys: Optional sufficient-statistic merge keys.
            num_components1: Number of ``X1`` components.
            num_components2: Number of ``X2`` components.
            comp_counts1: Weighted latent-state counts for ``X1``.
            comp_counts2: Weighted latent-state counts for ``X2``.
            joint_counts: Weighted joint counts for ``(X1_state, X2_state)``.
            name: Optional diagnostic name.

        """
        self.accumulators1 = accumulators1
        self.accumulators2 = accumulators2
        self.keys = keys if keys is not None else (None, None, None)
        self.num_components1 = len(accumulators1)
        self.num_components2 = len(accumulators2)
        self.comp_counts1 = vec.zeros(self.num_components1)
        self.comp_counts2 = vec.zeros(self.num_components2)
        self.joint_counts = vec.zeros((self.num_components1, self.num_components2))
        self.name = name
        # Data log-likelihood accumulated as a byproduct of the E-step (the posterior normalizer),
        # only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        self._rng_init = False
        self._idx1_rng: RandomState | None = None
        self._idx2_rng: RandomState | None = None
        self._acc1_rng: list[RandomState] | None = None
        self._acc2_rng: list[RandomState] | None = None

    def update(self, x: tuple[T0, T1], weight: float, estimate: JointMixtureDistribution) -> None:
        """Update sufficient statistics with a single weighted observation.

        Encodes the single observation and delegates to seq_update() so that the scalar and
        vectorized estimation paths agree.

        Args:
            x (Tuple[T0, T1]): A single (X1, X2) observation.
            weight (float): Weight for the observation.
            estimate (JointMixtureDistribution): Previous estimate from EM algorithm.

        Returns:
            None.

        """
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize member random states for ``initialize`` and ``seq_initialize`` consistency.

        Args:
            rng (RandomState): Random state used to generate member seeds.

        Returns:
            None.

        """
        self._idx1_rng = RandomState(seed=rng.randint(0, maxrandint))
        self._idx2_rng = RandomState(seed=rng.randint(0, maxrandint))
        self._acc1_rng = [RandomState(seed=rng.randint(0, maxrandint)) for i in range(self.num_components1)]
        self._acc2_rng = [RandomState(seed=rng.randint(0, maxrandint)) for i in range(self.num_components2)]
        self._rng_init = True

    def initialize(self, x: tuple[T0, T1], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with a single weighted observation.

        A component state is drawn uniformly at random for each of X1 and X2, and the
        corresponding component accumulators are initialized.

        Args:
            x (Tuple[T0, T1]): A single (X1, X2) observation.
            weight (float): Weight for the observation.
            rng: Random state used to seed child accumulator initializers.

        """
        if not self._rng_init:
            self._rng_initialize(rng)

        idx1 = self._idx1_rng.choice(self.num_components1)
        idx2 = self._idx2_rng.choice(self.num_components2)

        self.joint_counts[idx1, idx2] += weight

        for i in range(self.num_components1):
            w = weight if i == idx1 else 0.0
            self.accumulators1[i].initialize(x[0], w, self._acc1_rng[i])
            self.comp_counts1[i] += w
        for i in range(self.num_components2):
            w = weight if i == idx2 else 0.0
            self.accumulators2[i].initialize(x[1], w, self._acc2_rng[i])
            self.comp_counts2[i] += w

    def seq_initialize(self, x: tuple[int, E0, E1], weights, rng) -> None:
        """Vectorized initialization of sufficient statistics from an encoded sequence x.

        Note: Calls _rng_initialize() to ensure equivalence between seq_initialize() and initialize().

        Args:
            x (Tuple[int, E0, E1]): Encoded sequence of iid joint mixture observations.
            weights (np.ndarray): Weights for the observations.
            rng: Random state used to seed child accumulator initializers.

        """
        sz, enc1, enc2 = x

        if not self._rng_init:
            self._rng_initialize(rng)

        idx1 = self._idx1_rng.choice(self.num_components1, size=sz)
        idx2 = self._idx2_rng.choice(self.num_components2, size=sz)

        temp = np.bincount(
            idx1 * self.num_components2 + idx2, weights=weights, minlength=self.num_components1 * self.num_components2
        )
        self.joint_counts += np.reshape(temp, (self.num_components1, self.num_components2))

        for i in range(self.num_components1):
            w = np.zeros(sz)
            w[idx1 == i] = weights[idx1 == i]
            self.accumulators1[i].seq_initialize(enc1, w, self._acc1_rng[i])
            self.comp_counts1[i] += np.sum(w)

        for i in range(self.num_components2):
            w = np.zeros(sz)
            w[idx2 == i] = weights[idx2 == i]
            self.accumulators2[i].seq_initialize(enc2, w, self._acc2_rng[i])
            self.comp_counts2[i] += np.sum(w)

    def seq_update(self, x: tuple[int, E0, E1], weights: np.ndarray, estimate: JointMixtureDistribution) -> None:
        """Vectorized update of sufficient statistics from an encoded sequence x.

        The joint posterior over component pairs (i, j) is computed under the previous estimate,
        and the marginal posteriors are passed as weights into the component accumulators.

        Args:
            x (Tuple[int, E0, E1]): Encoded sequence of iid joint mixture observations.
            weights (np.ndarray): Weights for the observations.
            estimate (JointMixtureDistribution): Previous estimate from EM algorithm.

        Returns:
            None.

        """
        sz, enc_data1, enc_data2 = x
        ll_mat1 = np.zeros((sz, self.num_components1, 1))
        ll_mat2 = np.zeros((sz, 1, self.num_components2))
        log_w = estimate.log_w1

        for i in range(estimate.num_components1):
            ll_mat1[:, i, 0] = estimate.components1[i].seq_log_density(enc_data1)
            ll_mat1[:, i, 0] += log_w[i]

        for i in range(estimate.num_components2):
            ll_mat2[:, 0, i] = estimate.components2[i].seq_log_density(enc_data2)

        with np.errstate(invalid="ignore"):  # -inf max on impossible rows -> rows zeroed below
            ll_max1 = ll_mat1.max(axis=1, keepdims=True)
            ll_mat1 -= ll_max1
            np.exp(ll_mat1, out=ll_mat1)

            ll_max2 = ll_mat2.max(axis=2, keepdims=True)
            ll_mat2 -= ll_max2
            np.exp(ll_mat2, out=ll_mat2)

        # an observation outside the support of either component set has max log-density -inf, which makes
        # the exponentiated matrices nan; zero those rows so impossible observations contribute no
        # responsibility (rather than poisoning the whole batch's counts with nan)
        ll_mat1[~np.isfinite(np.broadcast_to(ll_max1, ll_mat1.shape))] = 0.0
        ll_mat2[~np.isfinite(np.broadcast_to(ll_max2, ll_mat2.shape))] = 0.0

        ll_joint = ll_mat1 * ll_mat2
        ll_joint *= estimate.taus12

        gamma_2 = np.sum(ll_joint, axis=1, keepdims=True)
        sf = np.sum(gamma_2, axis=2, keepdims=True)
        sf_safe = np.where(sf > 0.0, sf, 1.0)  # impossible rows have sf==0; their gammas are already 0
        ww = np.reshape(weights, [-1, 1, 1])

        # Capture per-row data log-likelihood (== seq_log_density) by reusing the joint posterior
        # normalizer sf already computed here: row_ll = log(sf) + rowmax1 + rowmax2. Free except an
        # O(n) log/dot, and only when the fused-EM fast path requests it (_track_ll).
        if self._track_ll:
            with np.errstate(divide="ignore"):
                row_ll = np.log(sf[:, 0, 0]) + ll_max1[:, 0, 0] + ll_max2[:, 0, 0]
            self._seq_ll += float(np.dot(weights, row_ll))

        gamma_1 = np.sum(ll_joint, axis=2, keepdims=True)
        gamma_1 *= ww / sf_safe
        gamma_2 *= ww / sf_safe

        ll_joint *= ww / sf_safe

        self.comp_counts1 += np.sum(gamma_1, axis=0).flatten()
        self.comp_counts2 += np.sum(gamma_2, axis=0).flatten()
        self.joint_counts += ll_joint.sum(axis=0)

        for i in range(self.num_components1):
            self.accumulators1[i].seq_update(enc_data1, gamma_1[:, i, 0], estimate.components1[i])

        for i in range(self.num_components2):
            self.accumulators2[i].seq_update(enc_data2, gamma_2[:, 0, i], estimate.components2[i])

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident E-step: component scoring and the joint-posterior arithmetic run on the
        active engine (numpy or torch); the marginal/joint counts and the per-component
        responsibility weights match the host seq_update.
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, enc_data1, enc_data2 = x
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)

        ll1 = engine.stack(
            [backend_seq_log_density(estimate.components1[i], enc_data1, engine) for i in range(self.num_components1)],
            axis=1,
        )  # (sz, C1)
        ll1 = ll1 + engine.asarray(estimate.log_w1)
        e1 = engine.exp(ll1 - engine.max(ll1, axis=1)[:, None])
        ll2 = engine.stack(
            [backend_seq_log_density(estimate.components2[i], enc_data2, engine) for i in range(self.num_components2)],
            axis=1,
        )  # (sz, C2)
        e2 = engine.exp(ll2 - engine.max(ll2, axis=1)[:, None])

        taus12 = engine.asarray(estimate.taus12)  # (C1, C2)
        ll_joint = e1[:, :, None] * e2[:, None, :] * taus12[None, :, :]  # (sz, C1, C2)
        sf = engine.sum(engine.sum(ll_joint, axis=2), axis=1)  # (sz,)
        ww = engine.asarray(weights_np) / sf  # (sz,)

        gamma_1 = engine.sum(ll_joint, axis=2) * ww[:, None]  # (sz, C1)
        gamma_2 = engine.sum(ll_joint, axis=1) * ww[:, None]  # (sz, C2)
        joint = ll_joint * ww[:, None, None]

        self.comp_counts1 += np.asarray(engine.to_numpy(engine.sum(gamma_1, axis=0))).flatten()
        self.comp_counts2 += np.asarray(engine.to_numpy(engine.sum(gamma_2, axis=0))).flatten()
        self.joint_counts += np.asarray(engine.to_numpy(engine.sum(joint, axis=0)))

        g1 = np.asarray(engine.to_numpy(gamma_1))
        g2 = np.asarray(engine.to_numpy(gamma_2))
        for i in range(self.num_components1):
            self.accumulators1[i].seq_update(enc_data1, g1[:, i], estimate.components1[i])
        for i in range(self.num_components2):
            self.accumulators2[i].seq_update(enc_data2, g2[:, i], estimate.components2[i])

    def combine(
        self, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[E0, ...], tuple[E1, ...]]
    ) -> JointMixtureEstimatorAccumulator:
        """Merge aggregated joint-mixture sufficient statistics into this accumulator.

        The tuple is interpreted as ``(x1_counts, x2_counts, joint_counts,
        x1_child_stats, x2_child_stats)``.

        Args:
            suff_stat: Aggregated sufficient statistics.

        Returns:
            This accumulator.

        """
        cc1, cc2, jc, s1, s2 = suff_stat

        self.joint_counts += jc
        self.comp_counts1 += cc1
        for i in range(self.num_components1):
            self.accumulators1[i].combine(s1[i])
        self.comp_counts2 += cc2
        for i in range(self.num_components2):
            self.accumulators2[i].combine(s2[i])

        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...], tuple[Any, ...]]:
        """Return accumulated sufficient statistics."""
        return (
            self.comp_counts1,
            self.comp_counts2,
            self.joint_counts,
            tuple([u.value() for u in self.accumulators1]),
            tuple([u.value() for u in self.accumulators2]),
        )

    def from_value(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[E0, ...], tuple[E1, ...]]
    ) -> JointMixtureEstimatorAccumulator:
        """Replace this accumulator's sufficient statistics.

        Args:
            x: Aggregated sufficient statistics in ``value`` format.

        Returns:
            This accumulator.

        """
        cc1, cc2, jc, s1, s2 = x

        self.comp_counts1 = cc1
        self.comp_counts2 = cc2
        self.joint_counts = jc

        for i in range(self.num_components1):
            self.accumulators1[i].from_value(s1[i])
        for i in range(self.num_components2):
            self.accumulators2[i].from_value(s2[i])

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under configured keys.

        Merges the count statistics if the weight key is set, and the X1/X2 component
        sufficient statistics if the corresponding accumulator keys are set.

        Args:
            stats_dict: Mapping from merge keys to sufficient statistics.

        """
        weight_key, acc1_key, acc2_key = self.keys

        if weight_key is not None:
            if weight_key in stats_dict:
                x1, x2, x3 = stats_dict[weight_key]
                stats_dict[weight_key] = (x1 + self.comp_counts1, x2 + self.comp_counts2, x3 + self.joint_counts)
            else:
                stats_dict[weight_key] = (self.comp_counts1, self.comp_counts2, self.joint_counts)

        if acc1_key is not None:
            if acc1_key in stats_dict:
                acc = stats_dict[acc1_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators1[i].value())
            else:
                stats_dict[acc1_key] = self.accumulators1

        if acc2_key is not None:
            if acc2_key in stats_dict:
                acc = stats_dict[acc2_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators2[i].value())
            else:
                stats_dict[acc2_key] = self.accumulators2

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's sufficient statistics from matching keys.

        Args:
            stats_dict: Mapping from merge keys to sufficient statistics.

        """
        weight_key, acc1_key, acc2_key = self.keys

        if weight_key is not None:
            if weight_key in stats_dict:
                x1, x2, x3 = stats_dict[weight_key]
                self.comp_counts1 = x1
                self.comp_counts2 = x2
                self.joint_counts = x3

        if acc1_key is not None:
            if acc1_key in stats_dict:
                self.accumulators1 = stats_dict[acc1_key]

        if acc2_key is not None:
            if acc2_key in stats_dict:
                self.accumulators2 = stats_dict[acc2_key]

    def acc_to_encoder(self) -> DataSequenceEncoder:
        """Return an encoder compatible with paired joint-mixture observations."""
        encoder1 = self.accumulators1[0].acc_to_encoder()
        encoder2 = self.accumulators2[0].acc_to_encoder()
        return JointMixtureDataEncoder(encoder1=encoder1, encoder2=encoder2)


class JointMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for joint-mixture EM accumulators."""

    def __init__(
        self,
        factories1: Sequence[StatisticAccumulatorFactory],
        factories2: Sequence[StatisticAccumulatorFactory],
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """Create an accumulator factory.

        Args:
            factories1: Component accumulator factories for ``X1``.
            factories2: Component accumulator factories for ``X2``.
            keys: Optional merge keys for joint counts, ``X1`` accumulators, and
                ``X2`` accumulators.
            name: Optional diagnostic name.

        Attributes:
            factories1: Component accumulator factories for ``X1``.
            factories2: Component accumulator factories for ``X2``.
            keys: Optional sufficient-statistic merge keys.
            name: Optional diagnostic name.

        """
        self.factories1 = factories1
        self.factories2 = factories2
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def make(self) -> JointMixtureEstimatorAccumulator:
        """Return a fresh joint-mixture accumulator."""
        f1 = [self.factories1[i].make() for i in range(len(self.factories1))]
        f2 = [self.factories2[i].make() for i in range(len(self.factories2))]
        return JointMixtureEstimatorAccumulator(f1, f2, keys=self.keys, name=self.name)


class JointMixtureEstimator(ParameterEstimator):
    """Estimator for paired latent mixture distributions."""

    def __init__(
        self,
        estimators1: Sequence[ParameterEstimator],
        estimators2: Sequence[ParameterEstimator],
        suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[E0, ...], tuple[E1, ...]] | None = None,
        pseudo_count: float | tuple[float, float, float] | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """Create an estimator for paired-mixture sufficient statistics.

        Args:
            estimators1: Component estimators for ``X1``.
            estimators2: Component estimators for ``X2``.
            suff_stat: Optional prior sufficient statistics used with ``pseudo_count``.
            pseudo_count: Optional smoothing counts for state and joint counts. A scalar is
                broadcast to all three slots.
            keys: Optional merge keys for joint counts, ``X1`` accumulators, and
                ``X2`` accumulators.
            name: Optional diagnostic name.

        Attributes:
            estimators1: Component estimators for ``X1``.
            estimators2: Component estimators for ``X2``.
            suff_stat: Optional prior sufficient statistics.
            pseudo_count: Optional smoothing counts.
            keys: Optional sufficient-statistic merge keys.
            name: Optional diagnostic name.

        """
        self.num_components1 = len(estimators1)
        self.num_components2 = len(estimators2)
        self.estimators1 = estimators1
        self.estimators2 = estimators2
        pseudo_count = broadcast_pseudo_count(pseudo_count, 3)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def accumulator_factory(self) -> JointMixtureEstimatorAccumulatorFactory:
        """Return an accumulator factory matching this estimator."""
        est_factories1 = [u.accumulator_factory() for u in self.estimators1]
        est_factories2 = [u.accumulator_factory() for u in self.estimators2]
        return JointMixtureEstimatorAccumulatorFactory(est_factories1, est_factories2, self.keys)

    def estimate(
        self, nobs, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[E0, ...], tuple[E1, ...]]
    ) -> JointMixtureDistribution:
        """Estimate a joint mixture distribution from aggregated sufficient statistics.

        The tuple is interpreted as ``(x1_counts, x2_counts, joint_counts,
        x1_child_stats, x2_child_stats)``.

        Args:
            nobs: Weighted number of observations, accepted for the estimator interface.
            suff_stat: Aggregated joint-mixture sufficient statistics.

        Returns:
            A fitted joint mixture distribution.

        """
        num_components1 = self.num_components1
        num_components2 = self.num_components2
        counts1, counts2, joint_counts, comp_suff_stats1, comp_suff_stats2 = suff_stat

        components1 = [self.estimators1[i].estimate(counts1[i], comp_suff_stats1[i]) for i in range(num_components1)]
        components2 = [self.estimators2[i].estimate(counts2[i], comp_suff_stats2[i]) for i in range(num_components2)]

        if self.pseudo_count is not None and self.suff_stat is None:
            p1 = self.pseudo_count[0] / float(self.num_components1)
            p2 = self.pseudo_count[1] / float(self.num_components2)
            p3 = self.pseudo_count[2] / float(self.num_components2 * self.num_components1)

            w1 = (counts1 + p1) / (counts1.sum() + self.pseudo_count[0])
            w2 = (counts2 + p2) / (counts2.sum() + self.pseudo_count[1])
            taus = joint_counts + p3

            taus12_sum = np.sum(taus, axis=1, keepdims=True)
            taus12_sum[taus12_sum == 0] = 1.0
            taus12 = taus / taus12_sum

            taus21_sum = np.sum(taus, axis=0, keepdims=True)
            taus21_sum[taus21_sum == 0] = 1.0
            taus21 = taus / taus21_sum

        else:
            counts1_sum = counts1.sum()
            counts2_sum = counts2.sum()
            w1 = (
                np.full(self.num_components1, 1.0 / self.num_components1)
                if counts1_sum <= 0.0
                else counts1 / counts1_sum
            )
            w2 = (
                np.full(self.num_components2, 1.0 / self.num_components2)
                if counts2_sum <= 0.0
                else counts2 / counts2_sum
            )
            taus = joint_counts

            taus12_sum = np.sum(taus, axis=1, keepdims=True)
            taus12_sum[taus12_sum == 0] = 1.0
            taus12 = taus / taus12_sum

            taus21_sum = np.sum(taus, axis=0, keepdims=True)
            taus21_sum[taus21_sum == 0] = 1.0
            taus21 = taus / taus21_sum

        return JointMixtureDistribution(components1, components2, w1, w2, taus12, taus21, name=self.name)


class JointMixtureDataEncoder(DataSequenceEncoder):
    """Encode paired observations for vectorized joint-mixture scoring and EM."""

    def __init__(self, encoder1: DataSequenceEncoder, encoder2: DataSequenceEncoder) -> None:
        """Create an encoder for paired observations.

        Args:
            encoder1: Encoder for the first field.
            encoder2: Encoder for the second field.

        Attributes:
            encoder1: Encoder for the first field.
            encoder2: Encoder for the second field.

        """
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return "JointMixtureDataEncoder(encoder0=" + str(self.encoder1) + ",encoder1=" + str(self.encoder2) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` has the same component encoders.

        Args:
            other: Object to compare.

        Returns:
            ``True`` when both field encoders are equivalent.

        """
        if isinstance(other, JointMixtureDataEncoder):
            return self.encoder2 == other.encoder2 and self.encoder1 == other.encoder1
        else:
            return False

    def seq_encode(self, x: Sequence[tuple[T0, T1]]) -> tuple[int, Any, Any]:
        """Encode a sequence of iid joint mixture observations for vectorized functions.

        Return value 'rv' is a Tuple containing:
            rv[0] (int): Number of observations.
            rv[1] (E0): Encoded sequence of X1 values.
            rv[2] (E1): Encoded sequence of X2 values.

        Args:
            x (Sequence[Tuple[T0, T1]]): Sequence of (X1, X2) observations.

        Returns:
            See above for details.

        """
        rv0 = len(x)
        rv1 = self.encoder1.seq_encode([u[0] for u in x])
        rv2 = self.encoder2.seq_encode([u[1] for u in x])

        return rv0, rv1, rv2


# --- Backward-compatible API naming aliases ---
JointMixtureAccumulator = JointMixtureEstimatorAccumulator
JointMixtureAccumulatorFactory = JointMixtureEstimatorAccumulatorFactory


def _register_joint_mixture_engine_kernel():
    """Register the engine-resident joint-mixture kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class JointMixtureKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("JointMixtureKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class JointMixtureKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return JointMixtureKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(JointMixtureDistribution, JointMixtureKernelFactory())


_register_joint_mixture_engine_kernel()


# --- Fisher view(s) co-located with this family ---
class _PairProductFisherView(FixedFisherView):
    """Fisher view for a product of two component views."""

    def __init__(self, left: FisherView, right: FisherView) -> None:
        self.left = left
        self.right = right
        labels = [("0",) + label for label in left.vectorizer.labels]
        labels.extend(("1",) + label for label in right.vectorizer.labels)
        FixedFisherView.__init__(self, (left.dist, right.dist), labels)

    def _refresh_labels(self) -> None:
        labels = [("0",) + label for label in self.left.vectorizer.labels]
        labels.extend(("1",) + label for label in self.right.vectorizer.labels)
        self.labels = labels
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        left_data = [x[0] for x in data]
        right_data = [x[1] for x in data]
        left_est = None if estimate is None else estimate[0]
        right_est = None if estimate is None else estimate[1]
        left = self.left.expected_statistics_matrix(data=left_data, estimate=left_est)
        right = self.right.expected_statistics_matrix(data=right_data, estimate=right_est)
        self._refresh_labels()
        return np.hstack((left, right))

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        left_est = None if estimate is None else estimate[0]
        right_est = None if estimate is None else estimate[1]
        left = self.left.seq_expected_statistics(enc_data[0], estimate=left_est)
        right = self.right.seq_expected_statistics(enc_data[1], estimate=right_est)
        self._refresh_labels()
        return np.hstack((left, right))

    def _model_mean(self) -> np.ndarray:
        return np.concatenate((self.left.mean_statistics(), self.right.mean_statistics()))

    def _model_fisher(self) -> np.ndarray:
        blocks = [
            np.asarray(self.left.fisher_information(ridge=0.0), dtype=np.float64),
            np.asarray(self.right.fisher_information(ridge=0.0), dtype=np.float64),
        ]
        dim = sum(block.shape[0] for block in blocks)
        out = np.zeros((dim, dim), dtype=np.float64)
        pos = 0
        for block in blocks:
            n = block.shape[0]
            out[pos : pos + n, pos : pos + n] = block
            pos += n
        return out


class JointMixtureFisherView(MixtureFisherView):
    """Complete-data Fisher view for joint mixtures without concrete proxies."""

    def __init__(self, dist: Any) -> None:
        self.pair_indices: list[tuple[int, int]] = []
        weights = []
        child_views = []
        for i, component1 in enumerate(dist.components1):
            if float(dist.w1[i]) <= 0.0:
                continue
            for j, component2 in enumerate(dist.components2):
                weight = float(dist.w1[i]) * float(dist.taus12[i, j])
                if weight <= 0.0:
                    continue
                self.pair_indices.append((i, j))
                weights.append(weight)
                child_views.append(_PairProductFisherView(to_fisher(component1), to_fisher(component2)))
        if not child_views:
            raise ValueError("JointMixtureFisherView requires at least one positive-weight component pair.")
        self.child_views = child_views
        self._pair_weights = np.asarray(weights, dtype=np.float64)
        self._pair_weights /= self._pair_weights.sum()
        labels = self._labels_from_children()
        FixedFisherView.__init__(self, dist, labels)

    @property
    def num_pairs(self) -> int:
        """Number of positive-weight component pairs represented by this Fisher view."""
        return len(self.pair_indices)

    def _pair_log_scores_from_data(self, data: Sequence[Any]) -> np.ndarray:
        rows = []
        for x in data:
            scores = []
            for i, j in self.pair_indices:
                scores.append(
                    self.dist.log_w1[i]
                    + self.dist.log_taus12[i, j]
                    + self.dist.components1[i].log_density(x[0])
                    + self.dist.components2[j].log_density(x[1])
                )
            rows.append(scores)
        return np.asarray(rows, dtype=np.float64)

    def _pair_log_scores_from_encoded(self, enc_data: Any) -> np.ndarray:
        sz, enc1, enc2 = enc_data
        scores = np.zeros((int(sz), len(self.pair_indices)), dtype=np.float64)
        left_cache: dict[int, np.ndarray] = {}
        right_cache: dict[int, np.ndarray] = {}
        for k, (i, j) in enumerate(self.pair_indices):
            if i not in left_cache:
                left_cache[i] = np.asarray(self.dist.components1[i].seq_log_density(enc1), dtype=np.float64)
            if j not in right_cache:
                right_cache[j] = np.asarray(self.dist.components2[j].seq_log_density(enc2), dtype=np.float64)
            scores[:, k] = self.dist.log_w1[i] + self.dist.log_taus12[i, j] + left_cache[i] + right_cache[j]
        return scores

    @staticmethod
    def _posterior_from_scores(scores: np.ndarray) -> np.ndarray:
        mx = np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores - mx)
        return weights / np.sum(weights, axis=1, keepdims=True)

    def _posterior_from_data(self, data: Sequence[Any]) -> np.ndarray:
        return self._posterior_from_scores(self._pair_log_scores_from_data(data))

    def _posterior_from_encoded(self, enc_data: Any) -> np.ndarray:
        return self._posterior_from_scores(self._pair_log_scores_from_encoded(enc_data))

    def log_density(self, x: Any) -> float:
        """Evaluate the Fisher-view mixture log-density for one paired observation."""
        scores = self._pair_log_scores_from_data([x])[0]
        mx = float(np.max(scores))
        return float(mx + np.log(np.exp(scores - mx).sum()))

    def _component_stats_from_data(self, data: Sequence[Any]) -> list[np.ndarray]:
        return [view.expected_statistics_matrix(data=data) for view in self.child_views]

    def _component_stats_from_encoded(self, enc_data: Any) -> list[np.ndarray]:
        _, enc1, enc2 = enc_data
        return [view.seq_expected_statistics((enc1, enc2)) for view in self.child_views]

    def structured_statistics(self, x: Any, estimate: Any | None = None, weight: float = 1.0) -> Any:
        """Return posterior pair weights and child sufficient statistics for one observation."""
        if estimate is not None and estimate is not self.dist:
            return to_fisher(estimate).structured_statistics(x, weight=weight)
        z = self._posterior_from_data([x])[0]
        child_values = tuple(z[k] * self.child_views[k].sufficient_statistics(x) for k in range(len(self.child_views)))
        return weight * z, child_values

    def _model_mean(self) -> np.ndarray:
        w = self._pair_weights
        means = self._component_means()
        return np.concatenate([w] + [w[k] * means[k] for k in range(len(means))])

    def _model_fisher(self) -> np.ndarray:
        w = self._pair_weights
        means, infos = self._component_moments()
        k_count = len(means)
        dims = [len(mu) for mu in means]
        offsets = []
        pos = k_count
        for dim in dims:
            offsets.append(pos)
            pos += dim

        out = np.zeros((pos, pos), dtype=np.float64)
        out[:k_count, :k_count] = np.diag(w) - np.outer(w, w)

        for i in range(k_count):
            for k in range(k_count):
                cov = ((w[k] if i == k else 0.0) - w[i] * w[k]) * means[k]
                s = offsets[k]
                e = s + dims[k]
                out[i, s:e] = cov
                out[s:e, i] = cov

        for k in range(k_count):
            sk = offsets[k]
            ek = sk + dims[k]
            muk = means[k]
            out[sk:ek, sk:ek] = w[k] * infos[k] + w[k] * (1.0 - w[k]) * np.outer(muk, muk)
            for l in range(k + 1, k_count):
                sl = offsets[l]
                el = sl + dims[l]
                block = -w[k] * w[l] * np.outer(muk, means[l])
                out[sk:ek, sl:el] = block
                out[sl:el, sk:ek] = block.T

        return out
