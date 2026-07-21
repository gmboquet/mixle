"""Semi-supervised mixture distributions with per-observation component priors.

Observations have the form ``(y, prior)``, where ``y`` is an observation for the
component distributions and ``prior`` is either ``None`` or a sequence of
``(component_index, probability)`` pairs. When prior information is present,
the mixture weights are reweighted for that observation before scoring or
responsibility updates. When ``prior`` is ``None``, the model behaves like an
ordinary mixture.

``seq_initialize()`` falls back to scalar ``initialize()`` calls on raw
observations, so initialization is not vectorized.
"""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.aliasing import MISSING, coalesce_alias

T0 = TypeVar("T0")  # Data type
T1 = TypeVar("T1")  # Prior type

E0 = TypeVar("E0")  # Encoded data type components
E1 = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # Encoded prior type
E = tuple[int, E0, tuple[E1, np.ndarray, np.ndarray], Sequence[tuple[T0, Sequence[tuple[int, T1]] | None]]]

SS0 = TypeVar("SS0")  # Suff-stat type from components


def _sum_prior_weights(prior: Sequence[tuple[int, T1]], num_components: int) -> np.ndarray:
    """Validate and sum prior mass by component for one observation."""
    prior_weights = np.zeros(num_components, dtype=np.float64)

    for idx, val in prior:
        if not (0 <= idx < num_components):
            raise ValueError("Prior component index %d is out of range [0, %d)." % (idx, num_components))
        if val < 0:
            raise ValueError("Prior value %s for component %d is negative." % (str(val), idx))
        prior_weights[idx] += val

    if not prior_weights.sum() > 0:
        raise ValueError("Prior has non-positive total mass.")

    return prior_weights


class SemiSupervisedMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """SemiSupervisedMixtureDistribution models observations (value, prior) where the optional
    prior labels re-weight the mixture weights over the listed components."""

    def compute_capabilities(self):
        """Declare generated-compute support inherited from mixture components."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(
            engine_ready=intersect_engine_ready(tuple(self.components)), kernel_status="generic_latent"
        )

    def compute_declaration(self):
        """Return the generated-compute declaration for the semi-supervised mixture."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        children = tuple(declaration_for(component) for component in self.components)
        children = tuple(child for child in children if child is not None)
        return DistributionDeclaration(
            name="semi_supervised_mixture",
            distribution_type=type(self),
            parameters=(ParameterSpec("w", constraint="simplex_vector"),),
            statistics=(
                StatisticSpec("component_counts"),
                StatisticSpec("components", kind="tuple"),
            ),
            support="labeled_mixture",
            children=children,
            child_roles=tuple("component_%d" % i for i in range(len(children))),
            differentiable=False,
        )

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        w: list[float] | np.ndarray = MISSING,
        name: str | None = None,
        weights: list[float] | np.ndarray = MISSING,
    ) -> None:
        """Create a semi-supervised mixture distribution.

        Args:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Mixture components.
            w (Union[List[float], np.ndarray]): Mixture weights. Values should sum to one.
            name (Optional[str]): Optional distribution name.

        Attributes:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Mixture components.
            num_components (int): Number of mixture components.
            zw (np.ndarray): Bool numpy array, True where weights are 0.0.
            log_w (np.ndarray): Log of weights. Set to -np.inf where weights are 0.
            w (np.ndarray): Mixture weights. Should sum to 1.0.
            name (Optional[str]): Optional distribution name.

        """
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        self.components = components
        self.num_components = len(components)
        self.w = np.asarray(w)
        self.zw = self.w == 0.0
        self.log_w = np.log(w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.name = name

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        return "SemiSupervisedMixtureDistribution([%s], [%s], name=%s)" % (
            ",".join([str(u) for u in self.components]),
            ",".join(map(str, self.w)),
            repr(self.name),
        )

    def density(self, x: tuple[T0, Sequence[tuple[int, T1]] | None]) -> float:
        """Density of the semi-supervised mixture at observation x.

        See log_density() for details.

        Args:
            x (Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]): Observation (value, prior).

        Returns:
            Density at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: tuple[T0, Sequence[tuple[int, T1]] | None]) -> float:
        """Log-density of the semi-supervised mixture at observation x = (value, prior).

        If prior is None this is the standard mixture log-density. Otherwise the mixture weights
        are restricted to the components listed in the prior, re-weighted by the prior
        probabilities, and re-normalized before mixing the component log-densities.

        Args:
            x (Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]): Observation (value, prior), where
                prior is an optional sequence of (component index, probability) pairs.

        Returns:
            Log-density at x.

        """

        datum, prior = x
        if prior is None:
            return vec.log_sum(np.asarray([u.log_density(datum) for u in self.components]) + self.log_w)
        else:
            w_loc = _sum_prior_weights(prior, self.num_components)
            h_loc = w_loc > 0.0

            w_loc[h_loc] = np.log(w_loc[h_loc])
            w_loc[h_loc] += self.log_w[h_loc]
            w_loc = vec.log_posterior(w_loc[h_loc])

            return vec.log_sum(
                np.asarray([self.components[i].log_density(datum) for i in np.flatnonzero(h_loc)]) + w_loc
            )

    def posterior(self, x: tuple[T0, Sequence[tuple[int, T1]] | None]) -> np.ndarray:
        """Posterior probability of each component for observation x = (value, prior).

        Components not listed in the prior (when a prior is present) receive posterior 0.

        Args:
            x (Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]): Observation (value, prior).

        Returns:
            Numpy array of length num_components containing the component posteriors.

        """
        datum, prior = x

        if prior is None:
            rv = vec.posterior(np.asarray([u.log_density(datum) for u in self.components]) + self.log_w)
        else:
            w_loc = _sum_prior_weights(prior, self.num_components)
            h_loc = w_loc > 0.0

            w_loc[h_loc] = np.log(w_loc[h_loc])
            w_loc[h_loc] += self.log_w[h_loc]
            for i in np.flatnonzero(h_loc):
                w_loc[i] += self.components[i].log_density(datum)

            w_loc[h_loc] = vec.posterior(w_loc[h_loc])
            rv = w_loc

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of the log-density on sequence encoded data x.

        Args:
            x (E): Sequence encoded data produced by SemiSupervisedMixtureDataEncoder.seq_encode().

        Returns:
            Numpy array of log-densities, one entry per encoded observation.

        """
        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), _ = x
        ll_mat = np.zeros((sz, self.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * self.w[enc_prior[1]]), minlength=sz)
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = self.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + self.log_w[enc_prior[1]]

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] += self.components[i].seq_log_density(enc_data)
                ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)
        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max

            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return ll_sum.flatten()

        else:
            ll_mat = ll_mat[good_rows, :]
            ll_max = ll_max[good_rows]

            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max
            rv = np.zeros(good_rows.shape, dtype=float)
            rv[good_rows] = ll_sum.flatten()
            rv[~good_rows] = -np.inf

            return rv

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral semi-supervised mixture log-density for encoded observations."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), _ = x
        prior_idx, prior_comp, prior_val, prior_log_val = enc_prior
        has_prior_idx = np.flatnonzero(enc_prior_flag)
        no_prior_idx = np.flatnonzero(~enc_prior_flag)

        ll_mat = engine.zeros((sz, self.num_components)) + engine.asarray(-np.inf)
        log_w = engine.asarray(self.log_w)

        if len(no_prior_idx):
            ll_mat[engine.asarray(no_prior_idx), :] = log_w

        if len(prior_idx):
            entry_scores = engine.asarray(prior_log_val + self.log_w[prior_comp])
            ll_mat[engine.asarray(prior_idx), engine.asarray(prior_comp)] = entry_scores
            norm_weights = engine.asarray(prior_val * self.w[prior_comp])
            norm_const = engine.log(
                engine.bincount(engine.asarray(prior_idx), weights=norm_weights, minlength=sz)[
                    engine.asarray(has_prior_idx)
                ]
            )
        else:
            norm_const = None

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] = ll_mat[:, i] + backend_seq_log_density(self.components[i], enc_data, engine)
                if norm_const is not None:
                    ll_mat[engine.asarray(has_prior_idx), i] = ll_mat[engine.asarray(has_prior_idx), i] - norm_const

        return engine.logsumexp(ll_mat, axis=1)

    def seq_posterior(self, x: E) -> np.ndarray:
        """Vectorized component posteriors on sequence encoded data x.

        Args:
            x (E): Sequence encoded data produced by SemiSupervisedMixtureDataEncoder.seq_encode().

        Returns:
            Numpy array of shape (number of observations, num_components) of posteriors.

        """
        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), _ = x
        ll_mat = np.zeros((sz, self.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * self.w[enc_prior[1]]), minlength=sz)
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = self.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + self.log_w[enc_prior[1]]

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] += self.components[i].seq_log_density(enc_data)
                ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w
        ll_max[bad_rows] = np.max(self.log_w)

        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        ll_mat /= ll_sum

        return ll_mat

    def density_semantics(self):
        """Return exact-or-approximate density semantics joined from component models."""
        from mixle.stats.compute.pdist import DensitySemantics, join_density_semantics

        children = list(self.components)
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics")]
        return join_density_semantics(sems) if sems else DensitySemantics.EXACT

    def sampler(self, seed: int | None = None) -> "SemiSupervisedMixtureSampler":
        """Creates a SemiSupervisedMixtureSampler for sampling component values.

        Args:
            seed (Optional[int]): Seed for the random number generator used in sampling.

        Returns:
            SemiSupervisedMixtureSampler object.

        """
        return SemiSupervisedMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SemiSupervisedMixtureEstimator":
        """Creates a SemiSupervisedMixtureEstimator with one child estimator per component.

        Args:
            pseudo_count (Optional[float]): Used to inflate the sufficient statistics of the
                mixture weights.

        Returns:
            SemiSupervisedMixtureEstimator object.

        """
        if pseudo_count is not None:
            return SemiSupervisedMixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count,
                name=self.name,
            )
        else:
            return SemiSupervisedMixtureEstimator([u.estimator() for u in self.components], name=self.name)

    def dist_to_encoder(self) -> "SemiSupervisedMixtureDataEncoder":
        """Creates a SemiSupervisedMixtureDataEncoder for encoding sequences of (value, prior)
        observations.

        Returns:
            SemiSupervisedMixtureDataEncoder object.

        """
        return SemiSupervisedMixtureDataEncoder(
            encoder=self.components[0].dist_to_encoder(), num_components=self.num_components
        )

    def enumerator(self) -> "DistributionEnumerator":
        """Enumeration is not well-defined for semi-supervised mixtures.

        Observations pair a component value with exogenous prior labels: the model defines no
        distribution over the prior part, so the support over (value, prior) pairs cannot be
        enumerated with consistent probabilities.

        Raises:
            EnumerationError always.

        """
        raise EnumerationError(
            self,
            reason="observations pair a value with exogenous prior labels; "
            "the model defines no distribution over the prior part of "
            "(value, prior) pairs",
        )


class SemiSupervisedMixtureSampler(DistributionSampler):
    """SemiSupervisedMixtureSampler draws component values from a SemiSupervisedMixtureDistribution."""

    def __init__(self, dist: SemiSupervisedMixtureDistribution, seed: int | None = None) -> None:
        """Create a sampler for a semi-supervised mixture distribution.

        Args:
            dist (SemiSupervisedMixtureDistribution): Distribution to draw samples from.
            seed (Optional[int]): Seed for the random number generator used in sampling.

        Attributes:
            dist (SemiSupervisedMixtureDistribution): Distribution to draw samples from.
            rng (RandomState): RandomState used to choose components.
            comp_samplers (List[DistributionSampler]): One sampler per mixture component.

        """
        rng_loc = RandomState(seed)
        self.rng = RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size: int | None = None, *, batched: bool = True) -> Sequence[Any] | Any:
        """Draw 'size' component values from the mixture (no prior labels are generated).

        Args:
            size (Optional[int]): Number of independent samples. If None a single value is
                returned.

        Returns:
            A single component value if size is None, else a list of 'size' component values.

        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.comp_samplers[comp_state].sample()
        else:
            return [self.comp_samplers[i].sample() for i in comp_state]


class SemiSupervisedMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """SemiSupervisedMixtureEstimatorAccumulator accumulates posterior-weighted sufficient
    statistics for the mixture weights and each component."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] | None = (None, None),
        name: str | None = None,
    ) -> None:
        """Create an accumulator for semi-supervised mixture sufficient statistics.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): One accumulator per
                mixture component.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for the weights and the
                component accumulators.
            name (Optional[str]): Name for the accumulator.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Component accumulators.
            num_components (int): Number of mixture components.
            comp_counts (np.ndarray): Posterior-weighted counts for each component.
            weight_key (Optional[str]): Key for merging the component counts.
            comp_key (Optional[str]): Key for merging the component accumulators.
            name (Optional[str]): Name for the accumulator.

            _init_rng (bool): True once the member RandomStates have been seeded.
            _acc_rng (Optional[List[RandomState]]): Per-component RandomStates used by initialize.
            _w_rng (Optional[RandomState]): RandomState reserved for the weights.

        """
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key, self.comp_key = keys if keys is not None else (None, None)
        self.name = name
        # Data log-likelihood accumulated as a byproduct of the E-step (the posterior normalizer),
        # only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        self._init_rng = False
        self._acc_rng = None
        self._w_rng = None

    def update(
        self,
        x: tuple[T0, Sequence[tuple[int, T1]] | None],
        weight: float,
        estimate: SemiSupervisedMixtureDistribution,
    ) -> None:
        """Update the sufficient statistics with one weighted observation x = (value, prior).

        The component posteriors are computed from the current estimate (the prior labels
        restrict and re-weight them), and each component accumulator receives the value with
        weight posterior * weight.

        Args:
            x (Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]): Observation (value, prior).
            weight (float): Weight for the observation.
            estimate (SemiSupervisedMixtureDistribution): Current mixture estimate used to
                compute the component posteriors. Required.

        Returns:
            None.

        """
        likelihood = estimate.posterior(x)
        datum, prior = x

        self.comp_counts += likelihood * weight

        for i in range(self.num_components):
            self.accumulators[i].update(datum, likelihood[i] * weight, estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        """Seed the member RandomStates for consistent initialize/seq_initialize calls."""
        if not self._init_rng:
            self._w_rng = RandomState(seed=rng.randint(maxrandint))
            self._prior_rng = RandomState(seed=rng.randint(maxrandint))

            seeds = rng.randint(maxrandint, size=self.num_components)
            self._acc_rng = [RandomState(seed=seeds[i]) for i in range(self.num_components)]

            self._init_rng = True

    def initialize(self, x: tuple[T0, Sequence[tuple[int, T1]] | None], weight: float, rng: RandomState) -> None:
        """Initialize the accumulator with one weighted observation x = (value, prior).

        If a prior is present the value is assigned to the listed components with the prior
        probabilities as weights; otherwise a random component receives almost all the weight.

        Args:
            x (Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]): Observation (value, prior).
            weight (float): Weight for the observation.
            rng (RandomState): RandomState used to seed the member RandomStates.

        Returns:
            None.

        """
        datum, prior = x

        if not self._init_rng:
            self._rng_initialize(rng)

        if prior is None:
            idx = self._prior_rng.choice(self.num_components)
            wc0 = 0.001
            wc1 = wc0 / max((float(self.num_components) - 1.0), 1.0)
            wc2 = 1.0 - wc0

            for i in range(self.num_components):
                w = weight * wc2 if i == idx else wc1
                self.accumulators[i].initialize(datum, w, self._acc_rng[i])
                self.comp_counts[i] += w

        else:
            for i, w in prior:
                ww = weight * w
                self.accumulators[i].initialize(datum, ww, self._acc_rng[i])
                self.comp_counts[i] += ww

    def seq_initialize(self, x: E, weights: np.ndarray, rng: RandomState) -> None:
        """Initialize the accumulator from sequence encoded data x.

        Note: falls back to scalar initialize() on the raw observations carried in the encoding,
        so it is not vectorized.

        Args:
            x (E): Sequence encoded data produced by SemiSupervisedMixtureDataEncoder.seq_encode().
            weights (np.ndarray): Weights for each encoded observation.
            rng (RandomState): RandomState used to seed the member RandomStates.

        Returns:
            None.

        """
        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), xx = x
        for i in range(len(xx)):
            self.initialize(xx[i], weights[i], rng=rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: SemiSupervisedMixtureDistribution) -> None:
        """Vectorized update of the sufficient statistics from sequence encoded data x.

        Computes the prior-adjusted component posteriors for all observations and passes the
        posterior-weighted encoded data to each component accumulator's seq_update.

        Args:
            x (E): Sequence encoded data produced by SemiSupervisedMixtureDataEncoder.seq_encode().
            weights (np.ndarray): Weights for each encoded observation.
            estimate (SemiSupervisedMixtureDistribution): Current mixture estimate used to
                compute the component posteriors. Required.

        Returns:
            None.

        """
        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), _ = x
        ll_mat = np.zeros((sz, estimate.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * estimate.w[enc_prior[1]]), minlength=sz)
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = estimate.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + estimate.log_w[enc_prior[1]]

        for i in range(self.num_components):
            ll_mat[:, i] += estimate.components[i].seq_log_density(enc_data)
            ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = estimate.log_w
        ll_max[bad_rows] = np.max(estimate.log_w)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)

        # Capture per-row data log-likelihood (== seq_log_density) by reusing the posterior
        # normalizer already computed here: row_ll = rowmax + log(rowsum), with -inf for invalid
        # rows seq_log_density would also return -inf for. Free except an O(n) log/dot, and only
        # when the fused-EM fast path requests it (_track_ll).
        if self._track_ll:
            with np.errstate(divide="ignore"):
                row_ll = ll_max[:, 0] + np.log(ll_sum[:, 0])
            if np.any(bad_rows):
                row_ll[bad_rows] = -np.inf
            self._seq_ll += float(np.dot(weights, row_ll))

        ll_mat /= ll_sum

        for i in range(self.num_components):
            w_loc = ll_mat[:, i] * weights
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(enc_data, w_loc, estimate.components[i])

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident E-step: component scoring and the responsibility softmax run on the active
        engine; the (low-overhead, index-based) semi-supervised prior adjustment is built host-side. Matches
        the host seq_update.
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), _ = x
        num_components = estimate.num_components
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)

        # prior-adjusted base log-matrix (host-side index ops; low-overhead)
        base = np.full((sz, num_components), -np.inf, dtype=np.float64)
        norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * estimate.w[enc_prior[1]]), minlength=sz)
        norm_const = np.log(norm_const[enc_prior_flag])
        base[~enc_prior_flag, :] = estimate.log_w
        base[enc_prior[0], enc_prior[1]] = enc_prior[3] + estimate.log_w[enc_prior[1]]
        nc = np.zeros((sz, 1), dtype=np.float64)
        nc[enc_prior_flag, 0] = norm_const

        emit = engine.stack(
            [backend_seq_log_density(estimate.components[i], enc_data, engine) for i in range(num_components)], axis=1
        )  # (sz, C)
        ll = engine.asarray(base) + emit - engine.asarray(nc)

        ll_max = engine.max(ll, axis=1, keepdims=True)
        bad = ll_max <= engine.asarray(-1.0e308)
        ll = engine.where(bad, engine.asarray(estimate.log_w)[None, :], ll)
        ll_max = engine.where(bad, engine.asarray(float(np.max(estimate.log_w))), ll_max)
        e = engine.exp(ll - ll_max)
        resp = e / engine.sum(e, axis=1, keepdims=True)
        resp = resp * engine.asarray(weights_np)[:, None]

        resp_np = np.asarray(engine.to_numpy(resp))
        self.comp_counts += resp_np.sum(axis=0)
        for i in range(num_components):
            self.accumulators[i].seq_update(enc_data, resp_np[:, i], estimate.components[i])

    def combine(self, suff_stat: tuple[np.ndarray, tuple[SS0, ...]]) -> "SemiSupervisedMixtureEstimatorAccumulator":
        """Aggregate sufficient statistics suff_stat with this accumulator's statistics.

        Args:
            suff_stat (Tuple[np.ndarray, Tuple[SS0, ...]]): Component counts and component
                sufficient statistics, as returned by value().

        Returns:
            SemiSupervisedMixtureEstimatorAccumulator with combined sufficient statistics.

        """
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Returns the sufficient statistics: (component counts, component values)."""
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: tuple[np.ndarray, tuple[SS0, ...]]) -> "SemiSupervisedMixtureEstimatorAccumulator":
        """Set the accumulator's sufficient statistics to x.

        Args:
            x (Tuple[np.ndarray, Tuple[SS0, ...]]): Component counts and component sufficient
                statistics, as returned by value().

        Returns:
            SemiSupervisedMixtureEstimatorAccumulator object.

        """
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge the weight and component sufficient statistics for matching keys.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to shared sufficient statistics.

        Returns:
            None.

        """

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace the weight and component sufficient statistics with keyed values.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to shared sufficient statistics.

        Returns:
            None.

        """

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "SemiSupervisedMixtureDataEncoder":
        """Creates a SemiSupervisedMixtureDataEncoder for encoding sequences of (value, prior)
        observations.

        Returns:
            SemiSupervisedMixtureDataEncoder object.

        """
        return SemiSupervisedMixtureDataEncoder(
            encoder=self.accumulators[0].acc_to_encoder(), num_components=self.num_components
        )


class SemiSupervisedMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """SemiSupervisedMixtureEstimatorAccumulatorFactory creates
    SemiSupervisedMixtureEstimatorAccumulator objects from the component factories."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        dim: int,
        keys: tuple[str | None, str | None] | None = (None, None),
        name: str | None = None,
    ):
        """Create a factory for semi-supervised mixture accumulators.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): One factory per mixture component.
            dim (int): Number of mixture components.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for the weights and the
                component accumulators.
            name (Optional[str]): Name for created accumulators.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): One factory per mixture component.
            dim (int): Number of mixture components.
            keys (Tuple[Optional[str], Optional[str]]): Keys for the weights and the components.
            name (Optional[str]): Name for created accumulators.

        """
        self.factories = factories
        self.dim = dim
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def make(self) -> "SemiSupervisedMixtureEstimatorAccumulator":
        """Creates a SemiSupervisedMixtureEstimatorAccumulator with one accumulator per component.

        Returns:
            SemiSupervisedMixtureEstimatorAccumulator object.

        """
        return SemiSupervisedMixtureEstimatorAccumulator(
            [self.factories[i].make() for i in range(self.dim)], self.keys, self.name
        )


class SemiSupervisedMixtureEstimator(ParameterEstimator):
    """SemiSupervisedMixtureEstimator estimates a SemiSupervisedMixtureDistribution from
    aggregated sufficient statistics."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        suff_stat: np.ndarray | None = None,
        pseudo_count: float | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
        name: str | None = None,
    ) -> None:
        """Estimate semi-supervised mixture distributions from aggregated sufficient statistics.

        Args:
            estimators (Sequence[ParameterEstimator]): Estimators for the component distributions. All must be
                compatible with data type ``T``.
            suff_stat (Optional[np.ndarray]): Mixture weights for components obtained from prev estimation or for
                regularization.
            pseudo_count (Optional[float]): Prior mass used to smooth or blend component weights.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Optional keys for weights and component statistics.
            name (Optional[str]): Optional name assigned to estimated distributions.

        Attributes:
            estimators (Sequence[ParameterEstimator]): Estimators for the component distributions.
            suff_stat (Optional[np.ndarray]): Mixture weights for components obtained from prev estimation or for
                regularization.
            pseudo_count (Optional[float]): Prior mass used to smooth or blend component weights.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Optional keys for weights and component statistics.
            name (Optional[str]): Optional name assigned to estimated distributions.

        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def accumulator_factory(self) -> "SemiSupervisedMixtureEstimatorAccumulatorFactory":
        """Create an accumulator factory from the child estimators.

        Returns:
            SemiSupervisedMixtureEstimatorAccumulatorFactory: Factory configured for this estimator.

        """
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return SemiSupervisedMixtureEstimatorAccumulatorFactory(
            est_factories, self.num_components, self.keys, self.name
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, tuple[SS0, ...]]
    ) -> "SemiSupervisedMixtureDistribution":
        """Estimate a SemiSupervisedMixtureDistribution from aggregated sufficient statistics.

        The mixture weights are the normalized component counts, optionally regularized by
        pseudo_count and the stored suff_stat weights.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency.
            suff_stat (Tuple[np.ndarray, Tuple[SS0, ...]]): Component counts and component
                sufficient statistics, as returned by
                SemiSupervisedMixtureEstimatorAccumulator.value().

        Returns:
            SemiSupervisedMixtureDistribution: Estimated distribution.

        """
        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i]) for i in range(num_components)]

        if self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / num_components
            w = counts + p
            w /= w.sum()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            w = (counts + self.suff_stat * self.pseudo_count) / (counts.sum() + self.pseudo_count)
        else:
            nobs_loc = counts.sum()

            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        return SemiSupervisedMixtureDistribution(components, w)


class SemiSupervisedMixtureDataEncoder(DataSequenceEncoder):
    """SemiSupervisedMixtureDataEncoder encodes sequences of (value, prior) observations using a
    shared component encoder for the values and flat arrays for the prior labels."""

    def __init__(self, encoder: DataSequenceEncoder, num_components: int | None = None):
        """Create an encoder for semi-supervised mixture observations.

        Args:
            encoder (DataSequenceEncoder): Encoder shared by all mixture components.
            num_components (Optional[int]): Number of mixture components, used to validate prior
                component indices when provided.

        Attributes:
            encoder (DataSequenceEncoder): Encoder shared by all mixture components.
            num_components (Optional[int]): Number of mixture components, used to validate prior
                component indices when provided.

        """
        self.encoder = encoder
        self.num_components = num_components

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "SemiSupervisedMixtureDataEncoder(encoder=" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent semi-supervised mixture encoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a SemiSupervisedMixtureDataEncoder with an equal component encoder.

        """
        if isinstance(other, SemiSupervisedMixtureDataEncoder):
            return self.encoder == other.encoder
        else:
            return False

    def seq_encode(
        self, x: Sequence[tuple[T0, Sequence[tuple[int, T1]] | None]]
    ) -> tuple[int, Any, tuple[E1, np.ndarray, np.ndarray], Sequence[tuple[T0, Sequence[tuple[int, T1]] | None]]]:
        """Encode a sequence of iid (value, prior) observations for vectorized "seq_" calls.

        The encoding is a tuple of length 4:
            rv[0] (int): Number of observations.
            rv[1]: The values encoded by the shared component encoder.
            rv[2]: Prior arrays ((row index, component index, prob, log prob), per-row prior
                sums, per-row has-prior flags).
            rv[3]: The raw observations (used by seq_initialize).

        Args:
            x (Sequence[Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]]): Observations.

        Returns:
            See description above.

        """

        prior_comp = []
        prior_idx = []
        prior_val = []
        data = []

        num_components = self.num_components

        for i, xi in enumerate(x):
            datum, prior = xi
            data.append(datum)
            if prior is not None:
                prior_total = 0.0
                for prior_entry in prior:
                    if num_components is not None and not (0 <= prior_entry[0] < num_components):
                        raise ValueError(
                            "Prior component index %d for observation %d is out of range [0, %d)."
                            % (prior_entry[0], i, num_components)
                        )
                    if prior_entry[1] < 0:
                        raise ValueError(
                            "Prior value %s for component %d of observation %d is negative."
                            % (str(prior_entry[1]), prior_entry[0], i)
                        )
                    prior_total += prior_entry[1]
                    prior_idx.append(i)
                    prior_comp.append(prior_entry[0])
                    prior_val.append(prior_entry[1])
                if not prior_total > 0:
                    raise ValueError("Prior for observation %d has non-positive total mass." % i)

        prior_comp = np.asarray(prior_comp, dtype=int)
        prior_idx = np.asarray(prior_idx, dtype=int)
        prior_val = np.asarray(prior_val, dtype=float)

        if len(prior_idx) > 0:
            width = num_components if num_components is not None else int(prior_comp.max()) + 1
            flat_idx = prior_idx * width + prior_comp
            prior_val = np.bincount(flat_idx, weights=prior_val, minlength=len(x) * width)
            flat_idx = np.flatnonzero(prior_val > 0.0)
            prior_idx = (flat_idx // width).astype(int)
            prior_comp = (flat_idx % width).astype(int)
            prior_val = prior_val[flat_idx]

        prior_mat = (prior_idx, prior_comp, prior_val, np.log(prior_val))

        prior_sum = np.bincount(prior_idx, weights=prior_val, minlength=len(x))
        has_prior = prior_sum != 0

        return len(x), self.encoder.seq_encode(data), (prior_mat, prior_sum, has_prior), x


# --- Backward-compatible API naming aliases ---
SemiSupervisedMixtureAccumulator = SemiSupervisedMixtureEstimatorAccumulator
SemiSupervisedMixtureAccumulatorFactory = SemiSupervisedMixtureEstimatorAccumulatorFactory


def _register_ss_mixture_engine_kernel():
    """Register the engine-resident semi-supervised-mixture kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class SemiSupervisedMixtureKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("SemiSupervisedMixtureKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class SemiSupervisedMixtureKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return SemiSupervisedMixtureKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(SemiSupervisedMixtureDistribution, SemiSupervisedMixtureKernelFactory())


_register_ss_mixture_engine_kernel()
