"""Create, estimate, and sample from a mixture of multivariate Gaussian distributions.

Defines the GaussianMixtureDistribution, GaussianMixtureSampler, GaussianMixtureAccumulator,
GaussianMixtureEstimatorAccumulatorFactory, GaussianMixtureEstimator, and the GaussianMixtureDataEncoder
classes for use with pysparkplug.

Data type: Union[List[float], np.ndarray]. Each observation is a length-d real vector.

A K-component multivariate Gaussian mixture has density

    P(x) = sum_{k=1}^{K} w_k * N(x; mu_k, sigma_k),

where w is a vector of K mixture weights summing to 1.0, mu_k is the length-d mean of component k, and
sigma_k is the d-by-d positive-definite covariance matrix of component k. For convenience the covariance
argument 'sig2' may also be given as a (K, d) array of per-component diagonal variances, which is expanded
to full diagonal covariance matrices.

"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import maxrandint
from pysp.stats.mvn import (
    MultivariateGaussianDataEncoder,
    MultivariateGaussianDistribution,
)
from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.utils.aliasing import MISSING, coalesce_alias


class GaussianMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """GaussianMixtureDistribution object for a mixture of full-covariance multivariate Gaussians.

    Args:
        mu (Union[Sequence[Sequence[float]], np.ndarray]): Component means with shape (K, d).
        sig2 (Union[Sequence[Any], np.ndarray]): Component covariances. Either a (K, d, d) array of full
            covariance matrices or a (K, d) array of per-component diagonal variances.
        w (Union[Sequence[float], np.ndarray]): Mixture weights of length K, must sum to 1.0.
        name (Optional[str]): Assign string name to GaussianMixtureDistribution object.

    Attributes:
        dim (int): Dimension d of the component Gaussians.
        num_components (int): Number of mixture components K.
        mu (np.ndarray): Component means with shape (K, d).
        sig2 (np.ndarray): Component covariance matrices with shape (K, d, d).
        w (np.ndarray): Mixture weights of length K.
        zw (np.ndarray): Boolean array, True where a mixture weight is exactly 0.0.
        log_w (np.ndarray): Log of mixture weights, -np.inf where zw is True.
        components (List[MultivariateGaussianDistribution]): Component distributions.
        name (Optional[str]): Name of object instance.

    """

    def __init__(
        self,
        mu: Sequence[Sequence[float]] | np.ndarray,
        sig2: Sequence[Any] | np.ndarray,
        w: Sequence[float] | np.ndarray = MISSING,
        name: str | None = None,
        weights: Sequence[float] | np.ndarray = MISSING,
    ) -> None:
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        self.mu = np.asarray(mu, dtype=float)

        num_comp = self.mu.shape[0]
        dim = self.mu.shape[1]

        sig2_loc = np.asarray(sig2, dtype=float)
        if sig2_loc.ndim == 2:
            sig2_loc = np.asarray([np.diag(u) for u in sig2_loc])
        sig2_loc = np.reshape(sig2_loc, (num_comp, dim, dim))

        self.sig2 = sig2_loc
        self.w = np.asarray(w, dtype=float)
        self.zw = self.w == 0.0
        self.log_w = np.log(self.w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.name = name
        self.dim = dim
        self.num_components = num_comp
        self.components = [MultivariateGaussianDistribution(self.mu[k], self.sig2[k]) for k in range(num_comp)]

    def __str__(self) -> str:
        """Return string representation of GaussianMixtureDistribution object instance."""
        s1 = repr([list(u) for u in self.mu])
        s2 = repr([[list(v) for v in u] for u in self.sig2])
        s3 = repr(list(self.w))
        s4 = repr(self.name)
        return "GaussianMixtureDistribution(%s, %s, %s, name=%s)" % (s1, s2, s3, s4)

    def density(self, x: Sequence[float] | np.ndarray) -> float:
        """Evaluate the density of the Gaussian mixture at observation x.

        See log_density() for details.

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-d observation vector.

        Returns:
            Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[float] | np.ndarray) -> float:
        """Evaluate the log-density of the Gaussian mixture at observation x.

        The log-density is given by

            log(P(x)) = log(sum_{k=1}^{K} w_k * N(x; mu_k, sigma_k)),

        evaluated with a log-sum-exp over the component log-densities for numerical stability.

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-d observation vector.

        Returns:
            Log-density at x.

        """
        return vec.log_sum(self.component_log_density(x) + self.log_w)

    def component_log_density(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate the component-wise log-densities log(N(x; mu_k, sigma_k)) at observation x.

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-d observation vector.

        Returns:
            Numpy array of length K containing the component log-densities at x.

        """
        return np.asarray([m.log_density(x) for m in self.components], dtype=np.float64)

    def posterior(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """Obtain the posterior distribution over mixture components at observation x.

        The posterior for component k is

            P(Z=k|x) = w_k * N(x; mu_k, sigma_k) / sum_{j=1}^{K} w_j * N(x; mu_j, sigma_j).

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-d observation vector.

        Returns:
            Numpy array of length K containing the component posterior at x.

        """
        comp_log_density = self.component_log_density(x)
        comp_log_density += self.log_w
        comp_log_density[self.zw] = -np.inf

        max_val = np.max(comp_log_density)

        if max_val == -np.inf:
            return self.w.copy()
        else:
            comp_log_density -= max_val
            np.exp(comp_log_density, out=comp_log_density)
            comp_log_density /= comp_log_density.sum()
            return comp_log_density

    def seq_component_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of component-wise log-densities for encoded sequence x.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, d) produced by
                GaussianMixtureDataEncoder.seq_encode().

        Returns:
            2-d numpy array of floats with shape (sz, K) containing component log-densities.

        """
        enc_data = x
        ll_mat = np.zeros((len(enc_data), self.num_components))
        ll_mat.fill(-np.inf)

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] = self.components[i].seq_log_density(enc_data)

        return ll_mat

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of the mixture log-density for encoded sequence x.

        Evaluates log_density() for each row of the encoded data matrix using a row-wise
        log-sum-exp for numerical stability. Rows for which every positive-weight component
        has log-density -np.inf evaluate to -np.inf.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, d) produced by
                GaussianMixtureDataEncoder.seq_encode().

        Returns:
            Numpy array of length sz containing the log-density of each encoded observation.

        """
        enc_data = x
        ll_mat = np.zeros((len(enc_data), self.num_components))
        ll_mat.fill(-np.inf)

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] = self.components[i].seq_log_density(enc_data)
                ll_mat[:, i] += self.log_w[i]

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

    def seq_posterior(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of the component posterior for encoded sequence x.

        Evaluates posterior() for each row of the encoded data matrix (see posterior() for
        details). Rows with no finite component log-density fall back to the mixture weights.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, d) produced by
                GaussianMixtureDataEncoder.seq_encode().

        Returns:
            2-d numpy array of floats with shape (sz, K) containing row-wise posteriors.

        """
        enc_data = x
        ll_mat = np.zeros((len(enc_data), self.num_components))
        ll_mat.fill(-np.inf)

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] = self.components[i].seq_log_density(enc_data)
                ll_mat[:, i] += self.log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w.copy()
        ll_max[bad_rows] = np.max(self.log_w)
        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        ll_mat /= ll_max

        return ll_mat

    def seq_encode(self, x: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
        """Encode a sequence of iid mixture observations for vectorized 'seq_' calls.

        Deprecated: delegates to dist_to_encoder().seq_encode(x). Use the
        GaussianMixtureDataEncoder directly in new code.

        Args:
            x (Sequence[Union[Sequence[float], np.ndarray]]): Sequence of length-d observations.

        Returns:
            Encoded data matrix with shape (len(x), d).

        """
        return self.dist_to_encoder().seq_encode(x)

    def sampler(self, seed: int | None = None) -> "GaussianMixtureSampler":
        """Create GaussianMixtureSampler for sampling from GaussianMixtureDistribution instance.

        Args:
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Returns:
            GaussianMixtureSampler object.

        """
        return GaussianMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GaussianMixtureEstimator":
        """Create GaussianMixtureEstimator for estimating GaussianMixtureDistribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics in estimation.

        Returns:
            GaussianMixtureEstimator object.

        """
        if pseudo_count is not None:
            return GaussianMixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count,
                name=self.name,
            )
        else:
            return GaussianMixtureEstimator([u.estimator() for u in self.components], name=self.name)

    def dist_to_encoder(self) -> "GaussianMixtureDataEncoder":
        """Returns a GaussianMixtureDataEncoder object for encoding sequences of iid observations."""
        return GaussianMixtureDataEncoder(encoder=self.components[0].dist_to_encoder())


class GaussianMixtureSampler(DistributionSampler):
    """GaussianMixtureSampler object used to generate samples from a GaussianMixtureDistribution.

    Args:
        dist (GaussianMixtureDistribution): GaussianMixtureDistribution to draw samples from.
        seed (Optional[int]): Seed to set for sampling with RandomState.

    Attributes:
        dist (GaussianMixtureDistribution): GaussianMixtureDistribution to draw samples from.
        rng (RandomState): Seeded RandomState for sampling component labels.
        compSamplers (List[MultivariateGaussianSampler]): Samplers for each mixture component.

    """

    def __init__(self, dist: GaussianMixtureDistribution, seed: int | None = None) -> None:
        rng_loc = RandomState(seed)
        self.rng = RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.compSamplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size: int | None = None) -> np.ndarray:
        """Draw iid samples from the Gaussian mixture.

        If size is None, a single length-d numpy array is returned. Otherwise a numpy array
        with shape (size, d) containing 'size' iid samples is returned.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Numpy array with shape (d,) if size is None, else with shape (size, d).

        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.compSamplers[comp_state].sample()
        else:
            return np.asarray([self.compSamplers[i].sample() for i in comp_state])


class GaussianMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """GaussianMixtureAccumulator object used to aggregate sufficient statistics of observed data.

    Args:
        accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture
            components (one MultivariateGaussianAccumulator per component).
        keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and mixture components.

    Attributes:
        accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Component accumulators.
        num_components (int): Number of mixture components K.
        comp_counts (np.ndarray): Numpy array of floats accumulating component weights.
        weight_key (Optional[str]): Key for the mixture weights.
        comp_key (Optional[str]): Key for the mixture components.
        _init_rng (bool): True once RandomState objects for initialization have been created.
        _w_rng (Optional[RandomState]): RandomState for generating initialization weights.
        _acc_rng (Optional[List[RandomState]]): RandomState objects for component initialization.

    """

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        # Data log-likelihood accumulated as a byproduct of the E-step (the posterior normalizer),
        # only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        ### Initializer seeds
        self._init_rng: bool = False
        self._w_rng: RandomState | None = None
        self._acc_rng: list[RandomState] | None = None

    def update(self, x: Sequence[float] | np.ndarray, weight: float, estimate: GaussianMixtureDistribution) -> None:
        """Update sufficient statistics with a single weighted observation.

        The posterior of 'estimate' at x is scaled by weight and added to comp_counts; each
        component accumulator is then updated with its share of the posterior mass.

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-d observation vector.
            weight (float): Weight for the observation.
            estimate (GaussianMixtureDistribution): Previous EM estimate of the mixture.

        Returns:
            None.

        """
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior

        for i in range(self.num_components):
            self.accumulators[i].update(x, posterior[i], estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        """Create the RandomState objects used by initialize() and seq_initialize().

        Args:
            rng (RandomState): Used to generate seeds for the member RandomState objects.

        Returns:
            None.

        """
        seeds = rng.randint(2**31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: Sequence[float] | np.ndarray, weight: float, rng: RandomState) -> None:
        """Initialize the accumulator with a single weighted observation.

        Component responsibilities are drawn from a sparse Dirichlet distribution so that the
        components start from randomly perturbed assignments of the data.

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-d observation vector.
            weight (float): Weight for the observation.
            rng (RandomState): Used to seed the member RandomState objects if not already set.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if weight != 0:
            ww = self._w_rng.dirichlet(np.ones(self.num_components) / (self.num_components * self.num_components))
        else:
            ww = np.zeros(self.num_components)

        for i in range(self.num_components):
            w = weight * ww[i]
            self.accumulators[i].initialize(x, w, self._acc_rng[i])
            self.comp_counts[i] += w

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of the accumulator with an encoded sequence of observations.

        Vectorized implementation of initialize() for an encoded data matrix.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, d).
            weights (np.ndarray): Numpy array of sz non-negative observation weights.
            rng (RandomState): Used to seed the member RandomState objects if not already set.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        keep_idx = weights > 0
        keep_len = np.count_nonzero(keep_idx)
        ww = np.zeros((sz, self.num_components))

        if keep_len > 0:
            ww[keep_idx, :] = self._w_rng.dirichlet(
                alpha=np.ones(self.num_components) / (self.num_components**2), size=keep_len
            )
        ww *= np.reshape(weights, (sz, 1))

        for i in range(self.num_components):
            self.accumulators[i].seq_initialize(x, ww[:, i], self._acc_rng[i])
            self.comp_counts[i] += np.sum(ww[:, i])

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: GaussianMixtureDistribution) -> None:
        """Vectorized update of sufficient statistics with an encoded sequence of observations.

        Computes the posterior responsibility matrix for the encoded data under 'estimate'
        (see GaussianMixtureDistribution.seq_posterior()), scales it by weights, and passes the
        per-component responsibilities to the component accumulators.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, d).
            weights (np.ndarray): Numpy array of sz non-negative observation weights.
            estimate (GaussianMixtureDistribution): Previous EM estimate of the mixture.

        Returns:
            None.

        """
        enc_data = x
        ll_mat = np.zeros((len(enc_data), self.num_components))
        ll_mat.fill(-np.inf)

        for i in range(estimate.num_components):
            if not estimate.zw[i]:
                ll_mat[:, i] = estimate.components[i].seq_log_density(enc_data)
                ll_mat[:, i] += estimate.log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = estimate.log_w.copy()
        ll_max[bad_rows] = np.max(estimate.log_w)

        # Capture per-row data log-likelihood (== seq_log_density) by reusing the rowmax and rowsum
        # already computed for normalization: row_ll = rowmax + log(rowsum), with -inf for the bad
        # rows seq_log_density also reports as -inf. Free except an O(n) log/dot, and only when the
        # fused-EM fast path requests it (_track_ll).
        rowmax = ll_max[:, 0].copy() if self._track_ll else None

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)

        if self._track_ll:
            with np.errstate(divide="ignore"):
                row_ll = rowmax + np.log(ll_max[:, 0])
            if np.any(bad_rows):
                row_ll[bad_rows] = -np.inf
            self._seq_ll += float(np.dot(weights, row_ll))

        np.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max

        for i in range(self.num_components):
            w_loc = ll_mat[:, i]
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(enc_data, w_loc, estimate.components[i])

    def combine(self, suff_stat: tuple[np.ndarray, tuple[Any, ...]]) -> "GaussianMixtureAccumulator":
        """Merge the sufficient statistics of suff_stat into this accumulator.

        Arg suff_stat is a Tuple of length two containing,
            suff_stat[0] (np.ndarray): Aggregated component counts,
            suff_stat[1] (Tuple[Any, ...]): Tuple of K component sufficient statistics.

        Args:
            suff_stat: See above for details.

        Returns:
            GaussianMixtureAccumulator object.

        """
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Returns the sufficient statistics of the accumulator.

        Returns:
            Tuple of (component counts, tuple of K component sufficient statistics).

        """
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: tuple[np.ndarray, tuple[Any, ...]]) -> "GaussianMixtureAccumulator":
        """Set the sufficient statistics of the accumulator to x.

        Args:
            x: Tuple of (component counts, tuple of K component sufficient statistics).

        Returns:
            GaussianMixtureAccumulator object.

        """
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Combine sufficient statistics with other accumulators sharing matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to aggregated statistics.

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
        """Replace sufficient statistics with values from stats_dict for matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to aggregated statistics.

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

    def acc_to_encoder(self) -> "GaussianMixtureDataEncoder":
        """Returns a GaussianMixtureDataEncoder object for encoding sequences of iid observations."""
        return GaussianMixtureDataEncoder(encoder=self.accumulators[0].acc_to_encoder())


class GaussianMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """GaussianMixtureEstimatorAccumulatorFactory object for creating GaussianMixtureAccumulator objects.

    Args:
        factories (Sequence[StatisticAccumulatorFactory]): Factories for the component accumulators.
        dim (int): Number of mixture components K. Must equal the length of factories.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and component aggregations.

    Attributes:
        factories (Sequence[StatisticAccumulatorFactory]): Factories for the component accumulators.
        dim (int): Number of mixture components K.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and component aggregations.

    """

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        dim: int,
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        self.factories = factories
        self.dim = dim
        self.keys = keys

    def make(self) -> "GaussianMixtureAccumulator":
        """Returns a GaussianMixtureAccumulator with freshly made component accumulators."""
        return GaussianMixtureAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class GaussianMixtureEstimator(ParameterEstimator):
    """GaussianMixtureEstimator object for estimating GaussianMixtureDistribution from sufficient statistics.

    Args:
        estimators (Sequence[ParameterEstimator]): ParameterEstimator objects for the mixture
            components (typically MultivariateGaussianEstimator objects).
        name (Optional[str]): Set name for object instance.
        conj_prior_params (Optional[Any]): Reserved for conjugate prior parameters (currently unused).
        suff_stat (Optional[np.ndarray]): Prior mixture weights used with pseudo_count to regularize
            the weight estimate. Must have length equal to the number of components.
        pseudo_count (Optional[float]): Used to re-weight the mixture weight sufficient statistics
            in estimation.
        keys (Tuple[Optional[str], Optional[str]]): Set keys for the weights and component statistics.

    Attributes:
        num_components (int): Number of mixture components K.
        estimators (Sequence[ParameterEstimator]): Component estimators.
        pseudo_count (Optional[float]): Weight regularization constant.
        suff_stat (Optional[np.ndarray]): Prior mixture weights.
        conj_prior_params (Optional[Any]): Reserved for conjugate prior parameters (currently unused).
        keys (Tuple[Optional[str], Optional[str]]): Keys for the weights and component statistics.
        name (Optional[str]): Name of object instance.

    """

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        name: str | None = None,
        conj_prior_params: Any | None = None,
        suff_stat: np.ndarray | None = None,
        pseudo_count: float | None = None,
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.conj_prior_params = conj_prior_params
        self.name = name

    def accumulator_factory(self) -> "GaussianMixtureEstimatorAccumulatorFactory":
        """Returns a GaussianMixtureEstimatorAccumulatorFactory built from the component estimators."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return GaussianMixtureEstimatorAccumulatorFactory(est_factories, self.num_components, self.keys)

    def accumulatorFactory(self) -> "GaussianMixtureEstimatorAccumulatorFactory":
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, tuple[Any, ...]]
    ) -> "GaussianMixtureDistribution":
        """Estimate a GaussianMixtureDistribution from aggregated sufficient statistics.

        Arg suff_stat is a Tuple of length two containing:
            suff_stat[0] (np.ndarray): Aggregated component counts (weight sufficient statistics).
            suff_stat[1] (Tuple[Any, ...]): Tuple of K component sufficient statistics, each
                consumed by the corresponding component estimator's estimate() (M-step).

        If pseudo_count is set, the weight estimate is regularized towards uniform weights, or
        towards the member suff_stat weights when those are also set.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: See above for details.

        Returns:
            GaussianMixtureDistribution object.

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
                w = counts / nobs_loc

        mu = np.asarray([comp.mu for comp in components])
        sig2 = np.asarray([comp.covar for comp in components])

        return GaussianMixtureDistribution(mu, sig2, w, name=self.name)


class GaussianMixtureDataEncoder(DataSequenceEncoder):
    """GaussianMixtureDataEncoder object for encoding sequences of iid Gaussian mixture observations.

    Args:
        encoder (Optional[DataSequenceEncoder]): DataSequenceEncoder for the component distributions.
            Defaults to a MultivariateGaussianDataEncoder.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for the component distributions.

    """

    def __init__(self, encoder: DataSequenceEncoder | None = None) -> None:
        self.encoder = encoder if encoder is not None else MultivariateGaussianDataEncoder()

    def __str__(self) -> str:
        """Returns string representation of GaussianMixtureDataEncoder object."""
        return "GaussianMixtureDataEncoder(" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Checks if other object is an equivalent GaussianMixtureDataEncoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a GaussianMixtureDataEncoder with an equal component encoder.

        """
        if isinstance(other, GaussianMixtureDataEncoder):
            return self.encoder == other.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
        """Encode a sequence of iid length-d observations for vectorized 'seq_' calls.

        Args:
            x (Sequence[Union[Sequence[float], np.ndarray]]): Sequence of length-d observation vectors.

        Returns:
            Encoded data matrix with shape (len(x), d).

        """
        return self.encoder.seq_encode(x)


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
GaussianMixtureAccumulatorFactory = GaussianMixtureEstimatorAccumulatorFactory
