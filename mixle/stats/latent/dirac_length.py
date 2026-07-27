"""Dirac-length mixture distributions for integer observations.

The model mixes a learned length distribution ``P_1`` with a point mass at
integer value ``v``:

``P(Y) = p * P_1(Y) + (1 - p) * Delta_v(Y)``.

``P_1`` must have support on non-negative integers, or a subset of them.
"""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, best_first_union
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
from mixle.stats.multivariate._multinomial_contracts import (
    exact_integer,
    finite_weight,
    observation_weights,
)

E0 = TypeVar("E0")  # Type of encoded data.
E = tuple[int, np.ndarray, np.ndarray, E0]
SS0 = TypeVar("SS0")  # Type of component suff_stat
key_type = tuple[str, str] | tuple[None, None]


def _integer_value(value: Any, label: str) -> int:
    return exact_integer(value, label=label)


def _mixture_probability(value: Any, label: str = "p") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a real scalar" % label) from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("%s must be finite and lie in [0, 1]" % label)
    return result


class DiracLengthMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Mixture between a fixed length and a learned length distribution.

    Args:
        p (float): Probability of being drawn from length distribution. Must be between 0 and 1.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution with support on non-negative integers.
        name (Optional[str]): Optional distribution name.

    Attributes:
        p (float): Probability of being drawn from length distribution. Must be between 0 and 1.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution with support on non-negative integers.
        name (Optional[str]): Optional distribution name.

    """

    def compute_capabilities(self):
        """Declare generated-compute support inherited from the length distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, capabilities_for

        child = capabilities_for(self.len_dist)
        return DistributionCapabilities(
            engine_ready=child.engine_ready,
            kernel_status="numpy_only" if child.numpy_only_reason else "generic_latent",
            numpy_only_reason=child.numpy_only_reason,
        )

    def compute_declaration(self):
        """Return the generated-compute declaration for the Dirac-length mixture."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        length = declaration_for(self.len_dist)
        children = () if length is None else (length,)
        return DistributionDeclaration(
            name="dirac_length_mixture",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("p", constraint="unit_interval"),
                ParameterSpec("v", constraint="integer", differentiable=False),
            ),
            statistics=(
                StatisticSpec("component_counts"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="length_or_dirac",
            children=children,
            child_roles=("length",) if length is not None else (),
            differentiable=False,
        )

    def __init__(
        self, len_dist: SequenceEncodableProbabilityDistribution, p: float, v: int = 0, name: str | None = None
    ):
        checked_p = _mixture_probability(p)
        checked_v = _integer_value(v, "Dirac location")
        with np.errstate(divide="ignore"):
            self.p = checked_p
            self.v = checked_v
            self.log_p = float(np.log(checked_p))
            self.log_1p = float(np.log1p(-checked_p))
            self.len_dist = len_dist
            self.name = name

    def __str__(self) -> str:
        s1 = repr(self.len_dist)
        s2 = repr(self.p)
        s3 = repr(self.v)
        s4 = repr(self.name)

        return "LengthDiracMixtureDistribution(len_dist=%s, p=%s, v=%s, name=%s)" % (s1, s2, s3, s4)

    def density(self, x: int) -> float:
        """Evaluate density of length Dirac mixture distribution at observation x.

        See log_density() for details.

        Args:
            x (int): Integer value.

        Returns:
            Density at x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of length Dirac mixture distribution at observation x.

        log(P(x)) = log( p*P_1(x) + (1-p)*Delta_{v}(x) ),

        Args:
            x (int): Integer value.

        Returns:
            log-density at x.

        """
        checked_x = _integer_value(x, "Dirac-length observation")
        rv0 = self.log_p + self.len_dist.log_density(checked_x)

        if checked_x == self.v:
            rv = np.logaddexp(rv0, self.log_1p)
        else:
            rv = rv0

        return float(rv)

    def component_log_density(self, x: int) -> np.ndarray:
        """Log-density of each mixture component (length distribution, dirac at v) at x.

        Args:
            x (int): Integer value.

        Returns:
            Numpy array of the two component log-densities.

        """
        checked_x = _integer_value(x, "Dirac-length observation")
        rv = np.zeros(2, dtype=np.float64)
        rv[0] = self.len_dist.log_density(checked_x)
        if checked_x != self.v:
            rv[1] = -np.inf
        return rv

    def posterior(self, x: int) -> np.ndarray:
        """Posterior probability of each mixture component given observation x.

        Args:
            x (int): Integer value.

        Returns:
            Numpy array of the two component posterior probabilities (sums to one).

        """
        comp_log_density = self.component_log_density(x)
        comp_log_density[0] += self.log_p
        comp_log_density[1] += self.log_1p

        max_val = np.max(comp_log_density)
        if max_val == -np.inf:
            return np.zeros(2, dtype=np.float64)

        comp_log_density -= max_val
        np.exp(comp_log_density, out=comp_log_density)
        comp_log_density /= comp_log_density.sum()

        return comp_log_density

    def seq_component_log_density(self, x: E) -> np.ndarray:
        """Vectorized component log-densities at sequence encoded input x.

        Args:
            x (E): Sequence encoded data from DiracLengthMixtureDataEncoder.

        Returns:
            Numpy array of shape (len(x), 2) of component log-densities.

        """
        sz, idx_v, idx_nv, enc_x = x
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        ll_mat[:, 0] += self.len_dist.seq_log_density(enc_x)
        ll_mat[idx_nv, 1] = -np.inf

        return ll_mat

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of the mixture log-density at sequence encoded input x.

        Args:
            x (E): Sequence encoded data from DiracLengthMixtureDataEncoder.

        Returns:
            Numpy array of log-density (float) of len(x).

        """
        sz, idx_v, idx_nv, enc_x = x
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        ll_mat[:, 0] += self.len_dist.seq_log_density(enc_x) + self.log_p
        ll_mat[idx_nv, 1] = -np.inf
        ll_mat[idx_v, 1] += self.log_1p

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

    def backend_seq_component_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral component log densities for encoded length/dirac mixtures."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx_v, idx_nv, enc_x = x
        rv = engine.zeros((sz, 2))
        rv[:, 0] = backend_seq_log_density(self.len_dist, enc_x, engine)
        if len(idx_nv):
            rv[engine.asarray(idx_nv), 1] = engine.asarray(-np.inf)
        return rv

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral mixture log-density for encoded length/dirac observations."""
        ll_mat = self.backend_seq_component_log_density(x, engine)
        return engine.logsumexp(ll_mat + engine.asarray([self.log_p, self.log_1p]), axis=1)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["DiracLengthMixtureDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for shared-dirac length mixtures."""
        from mixle.stats.compute.stacked import stacked_component_params

        v = int(dists[0].v)
        if any(int(dist.v) != v for dist in dists):
            raise ValueError("Stacked DiracLengthMixtureDistribution components require shared dirac value.")
        try:
            length_route = stacked_component_params([dist.len_dist for dist in dists], engine)
        except ValueError as exc:
            raise ValueError(
                "DiracLengthMixture length child %s is not stackable: %s" % (type(dists[0].len_dist).__name__, exc)
            )
        return {
            "__pysp_component_axis__": {"log_p": 0, "log_1p": 0},
            "v": v,
            "length_route": length_route,
            "log_p": engine.asarray(np.asarray([dist.log_p for dist in dists], dtype=np.float64)),
            "log_1p": engine.asarray(np.asarray([dist.log_1p for dist in dists], dtype=np.float64)),
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: E, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of length/dirac mixture log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        sz, idx_v, idx_nv, enc_x = x
        num_components = int(params["num_components"])
        length_scores = stacked_component_log_density(enc_x, params["length_route"], engine)
        dirac_scores = engine.zeros((sz, num_components))
        if len(idx_nv) > 0:
            impossible = engine.zeros((len(idx_nv), num_components)) + engine.asarray(-np.inf)
            dirac_scores = engine.index_add(dirac_scores, engine.asarray(idx_nv), impossible)
        stacked = engine.stack(
            (
                length_scores + params["log_p"][None, :],
                dirac_scores + params["log_1p"][None, :],
            ),
            axis=2,
        )
        return engine.logsumexp(stacked, axis=2)

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: E, weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy ``(component_counts, length_stat)`` statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_log_density,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        sz, idx_v, idx_nv, enc_x = x
        ww = engine.asarray(weights)
        num_components = int(params["num_components"])
        length_scores = stacked_component_log_density(enc_x, params["length_route"], engine)
        dirac_base = np.full(sz, -np.inf, dtype=np.float64)
        dirac_base[idx_v] = 0.0
        scores = engine.stack(
            (
                length_scores + params["log_p"][None, :],
                engine.asarray(dirac_base)[:, None] + params["log_1p"][None, :],
            ),
            axis=2,
        )
        denom = engine.logsumexp(scores, axis=2)
        bad_rows = engine.isinf(denom) & (denom < engine.asarray(0.0))
        safe_denom = engine.where(bad_rows, engine.asarray(0.0), denom)
        posterior = engine.where(
            bad_rows[:, :, None],
            engine.zeros((sz, num_components, 2)),
            engine.exp(scores - safe_denom[:, :, None]),
        )
        weighted = ww[:, :, None] * posterior
        length_weights = weighted[:, :, 0]
        dirac_weights = weighted[:, :, 1]

        component_counts = engine.stack(
            (
                engine.sum(length_weights, axis=0),
                engine.sum(dirac_weights, axis=0),
            ),
            axis=1,
        )

        outer_estimators = tuple(getattr(estimator, "estimators", ()))
        length_estimators = tuple(getattr(component_est, "estimator", None) for component_est in outer_estimators)
        length_estimator = StackedEstimatorView(length_estimators) if len(length_estimators) == num_components else None
        length_stats = stacked_component_sufficient_statistics(
            enc_x, length_weights, params["length_route"], engine, length_estimator
        )
        length_by_component = unstack_component_stats(length_stats, num_components)

        return tuple((component_counts[i], length_by_component[i]) for i in range(num_components))

    def seq_posterior(self, x: E) -> np.ndarray:
        """Vectorized component posterior probabilities at sequence encoded input x.

        Args:
            x (E): Sequence encoded data from DiracLengthMixtureDataEncoder.

        Returns:
            Numpy array of shape (len(x), 2) of component posteriors.

        """
        sz, idx_v, _idx_nv, enc_x = x
        scores = np.full((sz, 2), -np.inf, dtype=np.float64)
        scores[:, 0] = self.len_dist.seq_log_density(enc_x) + self.log_p
        scores[idx_v, 1] = self.log_1p
        row_ll = np.logaddexp(scores[:, 0], scores[:, 1])
        rv = np.zeros_like(scores)
        good = np.isfinite(row_ll)
        if np.any(good):
            rv[good, :] = np.exp(scores[good, :] - row_ll[good, None])
        return rv

    def sampler(self, seed: int | None = None) -> "DiracLengthMixtureSampler":
        """Create a DiracLengthMixtureSampler from parameters of this distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            DiracLengthMixtureSampler object.

        """
        return DiracLengthMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "DiracLengthMixtureEstimator":
        """Create a DiracLengthMixtureEstimator with matching dirac value v.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            DiracLengthMixtureEstimator object.

        """

        if pseudo_count is not None:
            est = self.len_dist.estimator(pseudo_count)
            return DiracLengthMixtureEstimator(
                estimator=est, v=self.v, pseudo_count=pseudo_count, suff_stat=self.p, name=self.name
            )
        else:
            est = self.len_dist.estimator()
            return DiracLengthMixtureEstimator(estimator=est, v=self.v, name=self.name)

    def dist_to_encoder(self) -> "DiracLengthMixtureDataEncoder":
        """Returns a DiracLengthMixtureDataEncoder for encoding sequences of iid integer observations."""
        len_dist_encoder = self.len_dist.dist_to_encoder()
        return DiracLengthMixtureDataEncoder(encoder=len_dist_encoder, v=self.v)

    def enumerator(self) -> "DiracLengthMixtureEnumerator":
        """Returns a DiracLengthMixtureEnumerator iterating the union of the length-distribution
        support and the dirac point v in descending probability order."""
        return DiracLengthMixtureEnumerator(self)


class DiracLengthMixtureEnumerator(DistributionEnumerator):
    """Enumerates the union of the length-distribution support and the dirac point v.

    The model is a two-component mixture: the length distribution with weight p and a dirac
    delta at v with weight 1-p. The dirac component contributes the trivial single-point
    stream [(v, 0.0)]. Supports may overlap (the length distribution can also emit v), so
    candidates are de-duplicated and re-scored exactly with the mixture log-density.
    """

    def __init__(self, dist: DiracLengthMixtureDistribution) -> None:
        """Create an enumerator for Dirac-length mixture values.

        Args:
            dist (DiracLengthMixtureDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        streams = []
        log_offsets = []
        if dist.p > 0.0:
            streams.append(BufferedStream(child_enumerator(dist.len_dist, "DiracLengthMixtureDistribution.len_dist")))
            log_offsets.append(float(dist.log_p))
        if dist.p < 1.0:
            streams.append(BufferedStream(iter([(dist.v, 0.0)])))
            log_offsets.append(float(dist.log_1p))

        def exact_log_density(x):
            return float(dist.log_density(x))

        self._union = best_first_union(streams, log_offsets, exact_log_density)

    def __next__(self) -> tuple[int, float]:
        return next(self._union)


class DiracLengthMixtureSampler(DistributionSampler):
    """Sampler for a Dirac-length mixture distribution."""

    def __init__(self, dist: DiracLengthMixtureDistribution, seed: int | None = None) -> None:
        """DiracLengthMixtureSampler used to generate samples.

                Args:
                    dist (DiracMixtureDistribution): Assign DiracLengthMixtureDistribution to draw samples from.
                    seed (Optional[int]): Seed to set for sampling with RandomState.

                Attributes:
                    rng (RandomState): Seeded RandomState for sampling.
                    p (np.ndarray): Prob of drawing from length distribution.
                    len_dist_sampler (DistributionSampler): Sampler for the length distribution.
                    v (int): Dirac location.
        .
        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.p = np.exp(dist.log_p)
        self.len_dist_sampler = dist.len_dist.sampler(seed=self.rng.randint(maxrandint))
        self.v = dist.v

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | int:
        """Draw iid samples from a DiracLengthMixture distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Int or List[int] depending on size = None or size (int).

        """
        checked_size = None if size is None else exact_integer(size, label="Dirac-length sample size", nonnegative=True)
        comp_state = self.rng.binomial(n=1, size=checked_size, p=self.p)

        if checked_size is None:
            if comp_state == 0:
                return self.v
            else:
                return _integer_value(
                    self.len_dist_sampler.sample(),
                    "sampled Dirac length",
                )
        else:
            rv = np.zeros(checked_size, dtype=np.int64)
            rv.fill(self.v)

            idx = np.flatnonzero(comp_state == 1)
            if len(idx) > 0:
                sampled = [
                    _integer_value(value, "sampled Dirac length")
                    for value in self.len_dist_sampler.sample(size=len(idx))
                ]
                rv[idx] = np.asarray(sampled, dtype=np.int64)
            return [int(value) for value in rv]


class DiracLengthMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate component counts and length-distribution statistics.

    Args:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the length distribution.
        v (int): Dirac location.
        keys (Tuple[Optional[str], Optional[str]]): Keys for the mixture weights and component statistics.
        name (Optional[str]): Optional accumulator name.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the length distribution.
        comp_counts (np.ndarray): Posterior-weighted counts for the two components.
        weight_key (Optional[str]): Key for merging mixture weight counts.
        comp_key (Optional[str]): Key for merging component sufficient statistics.
        v (int): Dirac location.
        name (Optional[str]): Optional accumulator name.

    """

    def __init__(
        self,
        accumulator: SequenceEncodableStatisticAccumulator,
        v: int = 0,
        keys: tuple[str | None, str | None] = (None, None),
        name: str | None = None,
    ):
        self.accumulator = accumulator
        self.comp_counts = np.zeros(2, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.v = _integer_value(v, "Dirac location")
        self.name = name
        # Data log-likelihood accumulated as a byproduct of the E-step (the posterior normalizer),
        # only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        ### Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: RandomState | None = None
        self._w_rng: RandomState | None = None

    def seq_update(self, x: E, weights: np.ndarray, estimate: "DiracLengthMixtureDistribution"):
        """Vectorized accumulation of posterior-weighted statistics from encoded observations x.

        Args:
            x (E): Sequence encoded data from DiracLengthMixtureDataEncoder.
            weights (np.ndarray): Weights on the observations.
            estimate (DiracLengthMixtureDistribution): Previous estimate used for posteriors.

        """
        if not isinstance(estimate, DiracLengthMixtureDistribution):
            raise TypeError("Dirac-length accumulator estimate has the wrong model type")
        sz, idx_v, _idx_nv, enc_x = x
        checked_weights = observation_weights(
            weights,
            sz,
            label="Dirac-length observation weights",
        )
        len_ll = np.asarray(estimate.len_dist.seq_log_density(enc_x), dtype=np.float64)
        scores = np.full((sz, 2), -np.inf, dtype=np.float64)
        scores[:, 0] = len_ll + estimate.log_p
        scores[idx_v, 1] = estimate.log_1p
        row_ll = np.logaddexp(scores[:, 0], scores[:, 1])
        impossible = np.isneginf(row_ll)
        if np.any(impossible & (checked_weights > 0.0)):
            raise ValueError("Dirac-length observation is impossible under the estimate")
        posterior = np.zeros_like(scores)
        good = ~impossible
        if np.any(good):
            posterior[good, :] = np.exp(scores[good, :] - row_ll[good, None])
        weighted = posterior * checked_weights[:, None]

        if self._track_ll:
            positive = checked_weights > 0.0
            self._seq_ll += float(np.dot(checked_weights[positive], row_ll[positive]))

        self.comp_counts += weighted.sum(axis=0)
        self.accumulator.seq_update(enc_x, weighted[:, 0], estimate.len_dist)

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident E-step: the length-distribution scoring (the heavy term) runs through the
        active engine; the low-overhead two-component (length vs. Dirac) responsibility bookkeeping mirrors
        the host seq_update exactly.
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        if not isinstance(estimate, DiracLengthMixtureDistribution):
            raise TypeError("Dirac-length accumulator estimate has the wrong model type")
        sz, idx_v, _idx_nv, enc_x = x
        checked_weights = observation_weights(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            sz,
            label="Dirac-length observation weights",
        )
        len_score = np.asarray(
            engine.to_numpy(backend_seq_log_density(estimate.len_dist, enc_x, engine)),
            dtype=np.float64,
        )
        scores = np.full((sz, 2), -np.inf, dtype=np.float64)
        scores[:, 0] = len_score + estimate.log_p
        scores[idx_v, 1] = estimate.log_1p
        row_ll = np.logaddexp(scores[:, 0], scores[:, 1])
        impossible = np.isneginf(row_ll)
        if np.any(impossible & (checked_weights > 0.0)):
            raise ValueError("Dirac-length observation is impossible under the estimate")
        posterior = np.zeros_like(scores)
        good = ~impossible
        if np.any(good):
            posterior[good, :] = np.exp(scores[good, :] - row_ll[good, None])
        weighted = posterior * checked_weights[:, None]
        self.comp_counts += weighted.sum(axis=0)
        self.accumulator.seq_update(enc_x, weighted[:, 0], estimate.len_dist)

    def update(self, x: int, weight: float, estimate: "DiracLengthMixtureDistribution") -> None:
        """Add one observation's posterior-weighted contribution to the sufficient statistics.

        Args:
            x (int): Integer observation.
            weight (float): Weight on the observation.
            estimate (DiracLengthMixtureDistribution): Previous estimate used for posteriors.

        """
        if not isinstance(estimate, DiracLengthMixtureDistribution):
            raise TypeError("Dirac-length accumulator estimate has the wrong model type")
        checked_x = _integer_value(x, "Dirac-length observation")
        checked_weight = finite_weight(weight, label="Dirac-length observation weight")
        if checked_weight == 0.0:
            return
        posterior = estimate.posterior(checked_x)
        if posterior.sum() == 0.0:
            raise ValueError("Dirac-length observation is impossible under the estimate")
        posterior *= checked_weight
        self.comp_counts += posterior

        self.accumulator.update(checked_x, posterior[0], estimate.len_dist)

    def _rng_initialize(self, rng: RandomState):
        seeds = rng.randint(2**31, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: int, weight: float, rng: np.random.RandomState):
        """Initialize the accumulator with observation x, randomly splitting weight at the dirac point.

        Args:
            x (int): Integer observation.
            weight (float): Weight on the observation.
            rng (RandomState): Random number generator for initialization.

        """
        checked_x = _integer_value(x, "Dirac-length observation")
        checked_weight = finite_weight(weight, label="Dirac-length observation weight")
        if checked_weight == 0.0:
            return
        if not self._init_rng:
            self._rng_initialize(rng)

        if checked_x == self.v:
            ww = self._w_rng.dirichlet(np.ones(2) / 4)
            self.accumulator.initialize(checked_x, checked_weight * ww[0], rng=self._acc_rng)
            self.comp_counts += checked_weight * ww
        else:
            self.accumulator.initialize(checked_x, checked_weight, rng=self._acc_rng)
            self.comp_counts[0] += checked_weight

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization from encoded observations x with random splits at the dirac point.

        Args:
            x (E): Sequence encoded data from DiracLengthMixtureDataEncoder.
            weights (np.ndarray): Weights on the observations.
            rng (RandomState): Random number generator for initialization.

        """

        sz, xi_v, xi_nv, enc_x = x
        weights = observation_weights(
            weights,
            sz,
            label="Dirac-length observation weights",
        )

        if not self._init_rng:
            self._rng_initialize(rng)

        keep_len = len(xi_v)
        ww = np.ones((sz, 2))

        if keep_len > 0:
            ww[xi_v, :] = self._w_rng.dirichlet(alpha=np.ones(2) / 4, size=keep_len)

        ww *= np.reshape(weights, (sz, 1))

        self.accumulator.seq_initialize(enc_x, weights=ww[:, 0], rng=self._acc_rng)
        self.comp_counts[0] += np.sum(ww[:, 0])
        self.comp_counts[1] += np.sum(ww[xi_v, 1])

    def combine(self, suff_stat: tuple[np.ndarray, SS0]) -> "DiracLengthMixtureAccumulator":
        """Combine sufficient statistics (component counts, length-dist stats) with this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, SS0]): Component counts and length-distribution statistics.

        Returns:
            This DiracLengthMixtureAccumulator.

        """
        self.comp_counts += suff_stat[0]
        self.accumulator.combine(suff_stat[1])

        return self

    def value(self) -> tuple[np.ndarray, Any]:
        """Returns sufficient statistics as a tuple (component counts, length-distribution statistics)."""
        return self.comp_counts, self.accumulator.value()

    def from_value(self, x: tuple[np.ndarray, SS0]) -> "DiracLengthMixtureAccumulator":
        """Set sufficient statistics from a (component counts, length-distribution statistics) tuple.

        Args:
            x (Tuple[np.ndarray, SS0]): Component counts and length-distribution statistics.

        Returns:
            This DiracLengthMixtureAccumulator.

        """
        self.comp_counts = x[0]
        self.accumulator.from_value(x[1])

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed sufficient statistics into stats_dict under the weight and component keys."""
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                # Copy on adoption: stats_dict must never alias this accumulator's own live
                # array, or a later tied accumulator's in-place += above would silently mutate
                # this accumulator's private comp_counts as a side effect of merging.
                stats_dict[self.weight_key] = self.comp_counts.copy()

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                stats_dict[self.comp_key].combine(self.accumulator.value())
            else:
                stats_dict[self.comp_key] = self.accumulator

        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace keyed sufficient statistics from stats_dict under the weight and component keys."""
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                # Copy on replace too: without it, every tied accumulator ends up pointing at
                # the SAME array object, so any one of them later accumulating new local data
                # would silently corrupt every other tied accumulator's counts.
                self.comp_counts = np.asarray(stats_dict[self.weight_key]).copy()

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulator = acc

        self.accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "DiracLengthMixtureDataEncoder":
        """Returns a DiracLengthMixtureDataEncoder for encoding sequences of iid integer observations."""
        acc_encoder = self.accumulator.acc_to_encoder()
        return DiracLengthMixtureDataEncoder(encoder=acc_encoder, v=self.v)


class DiracLengthMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for Dirac-length mixture accumulators.

    Args:
        factory (StatisticAccumulatorFactory): Accumulator factory for the length distribution.
        v (int): Dirac location.
        keys (Tuple[Optional[str], Optional[str]]): Keys for the mixture weights and component statistics.
        name (Optional[str]): Optional accumulator name.

    Attributes:
        factory (StatisticAccumulatorFactory): Accumulator factory for the length distribution.
        v (int): Dirac location.
        keys (Tuple[Optional[str], Optional[str]]): Keys for the mixture weights and component statistics.
        name (Optional[str]): Optional accumulator name.

    """

    def __init__(
        self,
        factory: StatisticAccumulatorFactory,
        v: int = 0,
        keys: tuple[str | None, str | None] = (None, None),
        name: str | None = None,
    ) -> None:
        self.factory = factory
        self.v = _integer_value(v, "Dirac location")
        self.keys = keys
        self.name = name

    def make(self) -> "DiracLengthMixtureAccumulator":
        """Returns a new DiracLengthMixtureAccumulator wrapping a fresh length-distribution accumulator."""
        return DiracLengthMixtureAccumulator(accumulator=self.factory.make(), v=self.v, keys=self.keys, name=self.name)


class DiracLengthMixtureEstimator(ParameterEstimator):
    """Estimate Dirac-length mixture distributions.

    Args:
        estimator (ParameterEstimator): Estimator for the length distribution.
        v (int): Dirac location.
        fixed_p (Optional[float]): Hold the length-distribution weight p fixed at this value.
        suff_stat (Optional[float]): Prior value of p used with pseudo_count for regularization.
        pseudo_count (Optional[float]): Used to inflate the component count statistics.
        name (Optional[str]): Optional name assigned to estimated distributions.
        keys (Tuple[Optional[str], Optional[str]]): Keys for the mixture weights and component statistics.

    Attributes:
        estimator (ParameterEstimator): Estimator for the length distribution.
        v (int): Dirac location.
        pseudo_count (Optional[float]): Used to inflate the component count statistics.
        suff_stat (Optional[float]): Prior value of p used with pseudo_count for regularization.
        keys (Tuple[Optional[str], Optional[str]]): Keys for the mixture weights and component statistics.
        name (Optional[str]): Optional name assigned to estimated distributions.
        fixed_p_vec (Optional[np.ndarray]): Fixed component weights [p, 1-p] when fixed_p is given.

    """

    def __init__(
        self,
        estimator: ParameterEstimator,
        v: int = 0,
        fixed_p: float | None = None,
        suff_stat: float | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: tuple[str | None, str | None] = (None, None),
    ):
        self.estimator = estimator
        self.v = _integer_value(v, "Dirac location")
        self.pseudo_count = (
            None
            if pseudo_count is None
            else finite_weight(
                pseudo_count,
                label="Dirac-length pseudo-count",
            )
        )
        self.suff_stat = (
            None
            if suff_stat is None
            else _mixture_probability(
                suff_stat,
                "Dirac-length prior probability",
            )
        )
        self.keys = keys
        self.name = name
        if fixed_p is None:
            self.fixed_p_vec = None
        else:
            checked_fixed = _mixture_probability(fixed_p, "fixed_p")
            self.fixed_p_vec = np.asarray([checked_fixed, 1.0 - checked_fixed])

    def accumulator_factory(self) -> "DiracLengthMixtureAccumulatorFactory":
        """Returns a DiracLengthMixtureAccumulatorFactory consistent with this estimator."""
        factory = self.estimator.accumulator_factory()
        return DiracLengthMixtureAccumulatorFactory(factory=factory, v=self.v, keys=self.keys, name=self.name)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, SS0]) -> "DiracLengthMixtureDistribution":
        """Estimate a DiracLengthMixtureDistribution from accumulated sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations.
            suff_stat (Tuple[np.ndarray, SS0]): Component counts and length-distribution statistics.

        Returns:
            DiracLengthMixtureDistribution object.

        """
        counts, comp_suff_stats = suff_stat
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != (2,) or np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
            raise ValueError("Dirac-length component counts must be finite and non-negative with shape (2,)")

        len_dist = self.estimator.estimate(counts[0], comp_suff_stats)

        if self.fixed_p_vec is not None:
            p = self.fixed_p_vec[0]

        elif self.pseudo_count is not None and self.pseudo_count > 0.0 and self.suff_stat is None:
            w = counts + self.pseudo_count / 2
            w /= w.sum()
            p = w[0]

        elif self.pseudo_count is not None and self.pseudo_count > 0.0 and self.suff_stat is not None:
            ss = np.array([self.suff_stat, 1 - self.suff_stat])
            w = (counts + ss * self.pseudo_count) / (counts.sum() + self.pseudo_count)
            p = w[0]

        else:
            nobs_loc = counts.sum()

            if nobs_loc == 0:
                p = 0.5
            else:
                w = counts / counts.sum()
                p = w[0]

        return DiracLengthMixtureDistribution(len_dist=len_dist, p=p, v=self.v, name=self.name)


class DiracLengthMixtureDataEncoder(DataSequenceEncoder):
    """Data encoder for iid integer observations under a Dirac-length mixture.

    Args:
        encoder (DataSequenceEncoder): Encoder for the length distribution.
        v (int): Dirac location.

    Attributes:
        encoder (DataSequenceEncoder): Encoder for the length distribution.
        v (int): Dirac location.

    """

    def __init__(self, encoder: DataSequenceEncoder, v: int = 0) -> None:
        self.encoder = encoder
        self.v = _integer_value(v, "Dirac location")

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "DiracMixtureDataEncoder(encoder=%s, v=%s)" % (repr(self.encoder), repr(self.v))

    def __eq__(self, other: object) -> bool:
        """Return True if other is a DiracLengthMixtureDataEncoder with equal base encoder and v."""
        if isinstance(other, DiracLengthMixtureDataEncoder):
            if other.encoder == self.encoder:
                return other.v == self.v
            else:
                return False
        else:
            return False

    def row_count(self, x: tuple[int, np.ndarray, np.ndarray, Any]) -> int:
        """Return the explicit observation count stored in the encoded payload."""
        count = exact_integer(x[0], label="Dirac-length encoded observation count", nonnegative=True)
        if count < 0:
            raise ValueError("dirac-length encoded observation count must be non-negative.")
        return count

    def seq_encode(self, x: Sequence[int]) -> tuple[int, np.ndarray, np.ndarray, Any]:
        """Encode a sequence of iid integer observations for vectorized use.

        Args:
            x (Sequence[int]): Sequence of iid integer observations.

        Returns:
            Tuple of (sequence length, indices equal to v, indices not equal to v, base-encoded data).

        """
        checked = [_integer_value(value, "Dirac-length observation") for value in x]
        try:
            values = np.asarray(checked, dtype=np.int64)
        except OverflowError as exc:
            raise ValueError("Dirac-length observations must fit signed 64-bit integers") from exc
        xi_v = np.flatnonzero(values == self.v).astype(np.int64)
        xi_nv = np.flatnonzero(values != self.v).astype(np.int64)

        return len(values), xi_v, xi_nv, self.encoder.seq_encode(values)


def _register_dirac_length_engine_kernel():
    """Register the engine-resident dirac-length-mixture kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class DiracLengthMixtureKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("DiracLengthMixtureKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class DiracLengthMixtureKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return DiracLengthMixtureKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(DiracLengthMixtureDistribution, DiracLengthMixtureKernelFactory())


_register_dirac_length_engine_kernel()
