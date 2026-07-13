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

E0 = TypeVar("E0")  # Type of encoded data.
E = tuple[int, np.ndarray, np.ndarray, E0]
SS0 = TypeVar("SS0")  # Type of component suff_stat
key_type = tuple[str, str] | tuple[None, None]


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
            engine_ready=child.engine_ready, kernel_status="generic_latent", numpy_only_reason=child.numpy_only_reason
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
        if not 0 < p <= 1:
            raise ValueError("p must be between (0,1].")
        with np.errstate(divide="ignore"):
            self.p = p
            self.v = v
            self.log_p = np.log(p)
            self.log_1p = np.log1p(-p)
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
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of length Dirac mixture distribution at observation x.

        log(P(x)) = log( p*P_1(x) + (1-p)*Delta_{v}(x) ),

        Args:
            x (int): Integer value.

        Returns:
            log-density at x.

        """
        rv0 = self.log_p + self.len_dist.log_density(x)

        if x == self.v:
            c1 = self.log_1p
            if c1 > rv0:
                rv = np.log1p(np.exp(rv0 - c1)) + c1
            else:
                rv = np.log1p(np.exp(c1 - rv0)) + rv0
        else:
            rv = rv0

        return rv

    def component_log_density(self, x: int) -> np.ndarray:
        """Log-density of each mixture component (length distribution, dirac at v) at x.

        Args:
            x (int): Integer value.

        Returns:
            Numpy array of the two component log-densities.

        """
        rv = np.zeros(2, dtype=np.float64)
        rv[0] = self.len_dist.log_density(x)
        if x != self.v:
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
            rv = np.array([np.exp(self.log_p), np.exp(self.log_1p)], dtype=np.float64)
            rv /= rv.sum()
            return rv

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
        length_weights = ww
        dirac_weights = engine.zeros((sz, num_components))

        if len(idx_v) > 0:
            eidx_v = engine.asarray(idx_v)
            local_length = length_scores[eidx_v, :] + params["log_p"][None, :]
            local_dirac = engine.zeros((len(idx_v), num_components)) + params["log_1p"][None, :]
            local_scores = engine.stack((local_length, local_dirac), axis=2)
            denom = engine.logsumexp(local_scores, axis=2)
            bad_rows = engine.isinf(denom) & (denom < engine.asarray(0.0))
            fallback = engine.stack(
                (
                    engine.zeros((len(idx_v), num_components)) + params["log_p"][None, :],
                    engine.zeros((len(idx_v), num_components)) + params["log_1p"][None, :],
                ),
                axis=2,
            )
            local_scores = engine.where(bad_rows[:, :, None], fallback, local_scores)
            denom = engine.where(bad_rows, engine.asarray(0.0), denom)
            local_post = engine.exp(local_scores - denom[:, :, None])
            local_weights = ww[eidx_v, :, None] * local_post

            length_at_v = engine.zeros((sz, num_components))
            dirac_at_v = engine.zeros((sz, num_components))
            length_at_v = engine.index_add(length_at_v, eidx_v, local_weights[:, :, 0])
            dirac_at_v = engine.index_add(dirac_at_v, eidx_v, local_weights[:, :, 1])
            non_v = np.ones(sz, dtype=bool)
            non_v[idx_v] = False
            length_weights = engine.where(engine.asarray(non_v)[:, None], ww, length_at_v)
            dirac_weights = dirac_at_v

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
        sz, idx_v, idx_nv, enc_x = x
        rv = np.zeros((sz, 2), dtype=np.float64)

        if len(idx_v) == 0:
            rv[:, 0] += 1.0

        else:
            rv[idx_nv, 0] += 1.0
            ll_mat = rv[idx_v, :]

            ll_mat[:, 1] += self.log_1p
            ll_mat[:, 0] += self.len_dist.seq_log_density(enc_x)[idx_v] + self.log_p

            ll_max = ll_mat.max(axis=1, keepdims=True)
            bad_rows = np.isinf(ll_max.flatten())

            ll_mat[bad_rows, :] = np.array([self.log_p, self.log_1p], dtype=np.float64)
            ll_max[bad_rows] = np.max(np.asarray([self.log_p, self.log_1p]))
            ll_mat -= ll_max

            np.exp(ll_mat, out=ll_mat)
            np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
            ll_mat /= ll_max

            rv[idx_v, :] = ll_mat

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
        streams = [
            BufferedStream(child_enumerator(dist.len_dist, "DiracLengthMixtureDistribution.len_dist")),
            BufferedStream(iter([(dist.v, 0.0)])),
        ]
        log_offsets = [float(dist.log_p), float(dist.log_1p)]

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

    def sample(self, size: int | None = None) -> list[int] | int:
        """Draw iid samples from a DiracLengthMixture distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Int or List[int] depending on size = None or size (int).

        """
        comp_state = self.rng.binomial(n=1, size=size, p=self.p)

        if size is None:
            if comp_state == 0:
                return self.v
            else:
                return self.len_dist_sampler.sample()
        else:
            rv = np.zeros(size, dtype=np.int32)
            rv.fill(self.v)

            idx = np.flatnonzero(comp_state == 1)
            if len(idx) > 0:
                rv[idx] = np.asarray(self.len_dist_sampler.sample(size=len(idx)), dtype=np.int32)
            return list(rv)


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
        self.v = v
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
        sz, idx_v, idx_nv, enc_x = x
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        # The fused-EM fast path wants the per-row data log-likelihood (== seq_log_density). The
        # length-distribution score is the only emission term; compute it once here and reuse it for
        # both the posteriors below and the tracked likelihood. Only when _track_ll is requested do
        # we score it in the len(idx_v)==0 branch (which otherwise takes a weights-only shortcut).
        len_ll = None
        if self._track_ll:
            len_ll = np.asarray(estimate.len_dist.seq_log_density(enc_x), dtype=np.float64)

        if len(idx_v) == 0:
            ll_mat[:, 0] += weights

        else:
            if len_ll is not None:
                ll_mat[:, 0] += len_ll + estimate.log_p
            else:
                ll_mat[:, 0] += estimate.len_dist.seq_log_density(enc_x) + estimate.log_p
            ll_mat[idx_nv, 0] = weights[idx_nv].copy()

            rv = ll_mat[idx_v, :]
            rv[:, 1] += estimate.log_1p

            rv_max = rv.max(axis=1, keepdims=True)
            bad_rows = np.isinf(rv_max.flatten())

            if np.any(bad_rows):
                rv[bad_rows, :] = np.array([estimate.log_p, estimate.log_1p], dtype=np.float64)
                rv_max[bad_rows] = np.max(np.asarray([estimate.log_p, estimate.log_1p]))
            rv -= rv_max

            np.exp(rv, out=rv)
            np.sum(rv, axis=1, keepdims=True, out=rv_max)
            np.divide(weights[idx_v, None], rv_max, out=rv_max)
            rv *= rv_max

            ll_mat[idx_v, :] = rv

        # Reconstruct the per-row log-likelihood exactly as seq_log_density: comp0 = len_ll + log_p
        # for every row, comp1 = log_1p on Dirac-valued rows and -inf elsewhere, then row logsumexp
        # (with -inf for the rows seq_log_density also reports as -inf).
        if self._track_ll:
            row_mat = np.empty((sz, 2), dtype=np.float64)
            row_mat[:, 0] = len_ll + estimate.log_p
            row_mat[:, 1] = -np.inf
            row_mat[idx_v, 1] = estimate.log_1p
            r_max = row_mat.max(axis=1)
            r_bad = np.isinf(r_max)
            row_ll = np.full(sz, -np.inf, dtype=np.float64)
            good = ~r_bad
            if np.any(good):
                with np.errstate(divide="ignore"):
                    row_ll[good] = r_max[good] + np.log(np.exp(row_mat[good, :] - r_max[good, None]).sum(axis=1))
            self._seq_ll += float(np.dot(weights, row_ll))

        self.comp_counts += ll_mat.sum(axis=0)
        self.accumulator.seq_update(enc_x, ll_mat[:, 0], estimate.len_dist)

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident E-step: the length-distribution scoring (the heavy term) runs through the
        active engine; the low-overhead two-component (length vs. Dirac) responsibility bookkeeping mirrors
        the host seq_update exactly.
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx_v, idx_nv, enc_x = x
        weights = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        if len(idx_v) == 0:
            ll_mat[:, 0] += weights
        else:
            len_score = np.asarray(
                engine.to_numpy(backend_seq_log_density(estimate.len_dist, enc_x, engine)), dtype=np.float64
            )
            ll_mat[:, 0] += len_score + estimate.log_p
            ll_mat[idx_nv, 0] = weights[idx_nv].copy()

            rv = ll_mat[idx_v, :]
            rv[:, 1] += estimate.log_1p
            rv_max = rv.max(axis=1, keepdims=True)
            bad_rows = np.isinf(rv_max.flatten())
            if np.any(bad_rows):
                rv[bad_rows, :] = np.array([estimate.log_p, estimate.log_1p], dtype=np.float64)
                rv_max[bad_rows] = np.max(np.asarray([estimate.log_p, estimate.log_1p]))
            rv -= rv_max
            np.exp(rv, out=rv)
            np.sum(rv, axis=1, keepdims=True, out=rv_max)
            np.divide(weights[idx_v, None], rv_max, out=rv_max)
            rv *= rv_max
            ll_mat[idx_v, :] = rv

        self.comp_counts += ll_mat.sum(axis=0)
        self.accumulator.seq_update(enc_x, ll_mat[:, 0], estimate.len_dist)

    def update(self, x: int, weight: float, estimate: "DiracLengthMixtureDistribution") -> None:
        """Add one observation's posterior-weighted contribution to the sufficient statistics.

        Args:
            x (int): Integer observation.
            weight (float): Weight on the observation.
            estimate (DiracLengthMixtureDistribution): Previous estimate used for posteriors.

        """
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior

        self.accumulator.update(x, posterior[0], estimate.len_dist)

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
        if not self._init_rng:
            self._rng_initialize(rng)

        if x == self.v:
            ww = self._w_rng.dirichlet(np.ones(2) / 4)
            self.accumulator.initialize(x, weight * ww[0], rng=self._acc_rng)
            self.comp_counts += ww
        else:
            self.accumulator.initialize(x, weight, rng=self._acc_rng)
            self.comp_counts[0] += weight

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization from encoded observations x with random splits at the dirac point.

        Args:
            x (E): Sequence encoded data from DiracLengthMixtureDataEncoder.
            weights (np.ndarray): Weights on the observations.
            rng (RandomState): Random number generator for initialization.

        """

        sz, xi_v, xi_nv, enc_x = x

        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
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
                stats_dict[self.weight_key] = self.comp_counts

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
                self.comp_counts = stats_dict[self.weight_key]

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
        self.v = v
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
        fixed_p: int | None = None,
        suff_stat: float | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: tuple[str | None, str | None] = (None, None),
    ):
        self.estimator = estimator
        self.v = v
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_p_vec = np.asarray([fixed_p, 1 - fixed_p]) if fixed_p is not None and 0 < fixed_p <= 1 else None

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

        len_dist = self.estimator.estimate(counts[0], comp_suff_stats)

        if self.fixed_p_vec is not None:
            p = self.fixed_p_vec[0]

        elif self.pseudo_count is not None and self.suff_stat is None:
            w = counts + self.pseudo_count / 2
            w /= w.sum()
            p = w[0]

        elif self.pseudo_count is not None and self.suff_stat is not None:
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
        self.v = v

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

    def seq_encode(self, x: Sequence[int]) -> tuple[int, np.ndarray, np.ndarray, Any]:
        """Encode a sequence of iid integer observations for vectorized use.

        Args:
            x (Sequence[int]): Sequence of iid integer observations.

        Returns:
            Tuple of (sequence length, indices equal to v, indices not equal to v, base-encoded data).

        """
        x = np.asarray(x, dtype=np.int32)
        xi_v = np.flatnonzero(x == self.v).astype(np.int32)
        xi_nv = np.flatnonzero(x != self.v).astype(np.int32)

        return len(x), xi_v, xi_nv, self.encoder.seq_encode(x)


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
