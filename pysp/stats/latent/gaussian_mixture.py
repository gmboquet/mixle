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

Implementation note: GaussianMixtureDistribution and friends are thin specializations of the generic
mixture machinery in :mod:`pysp.stats.latent.mixture`. The vectorized E-step (seq_log_density,
seq_posterior, Accumulator.seq_update, the fused-EM _track_ll path), scalar update/posterior, weight
M-step, and key merge/replace are all inherited unchanged -- the only Gaussian-specific pieces are
constructing the multivariate-Gaussian components from (mu, sig2), repacking mu/covar in estimate(),
the (n, d) ndarray sampler output, and the Gaussian data encoder.

"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.engines.arithmetic import maxrandint
from pysp.stats.compute._sampling import scatter_component_draws
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.stats.latent.mixture import (
    MixtureAccumulator,
    MixtureAccumulatorFactory,
    MixtureDataEncoder,
    MixtureDistribution,
    MixtureEstimator,
)
from pysp.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDataEncoder,
    MultivariateGaussianDistribution,
)
from pysp.utils.aliasing import MISSING, coalesce_alias


def _pack_sig2(sig2: Sequence[Any] | np.ndarray, num_comp: int, dim: int) -> np.ndarray:
    """Expand a (K, d) diagonal-variance arg to full (K, d, d) covariances; pass (K, d, d) through."""
    sig2_loc = np.asarray(sig2, dtype=float)
    if sig2_loc.ndim == 2:
        sig2_loc = np.asarray([np.diag(u) for u in sig2_loc])
    return np.reshape(sig2_loc, (num_comp, dim, dim))


class GaussianMixtureDistribution(MixtureDistribution):
    """GaussianMixtureDistribution object for a mixture of full-covariance multivariate Gaussians.

    A thin specialization of :class:`~pysp.stats.latent.mixture.MixtureDistribution` whose components
    are :class:`~pysp.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution`
    objects built from ``(mu, sig2)``. All density/posterior machinery is inherited from the base
    mixture; only the (mu, sig2) constructor packing, ``__str__``, and the Gaussian sampler/encoder
    wiring are Gaussian-specific.

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

        self.sig2 = _pack_sig2(sig2, num_comp, dim)
        self.dim = dim

        components = [MultivariateGaussianDistribution(self.mu[k], self.sig2[k]) for k in range(num_comp)]
        super().__init__(components, np.asarray(w, dtype=float), name=name)

    def __str__(self) -> str:
        """Return string representation of GaussianMixtureDistribution object instance."""
        s1 = repr([list(u) for u in self.mu])
        s2 = repr([[list(v) for v in u] for u in self.sig2])
        s3 = repr(list(self.w))
        s4 = repr(self.name)
        return "GaussianMixtureDistribution(%s, %s, %s, name=%s)" % (s1, s2, s3, s4)

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

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw iid samples from the Gaussian mixture.

        If size is None, a single length-d numpy array is returned. Otherwise a numpy array
        with shape (size, d) containing 'size' iid samples is returned. With ``batched=True``
        (default) component draws are grouped and scattered -- bit-identical to the per-draw loop
        (``batched=False``) but far faster, since each component sampler owns an independent RNG.

        Args:
            size (Optional[int]): Number of iid samples to draw.
            batched (bool): Vectorize component draws (default); set False for the per-draw loop.

        Returns:
            Numpy array with shape (d,) if size is None, else with shape (size, d).

        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.compSamplers[comp_state].sample()
        if not batched:
            return np.asarray([self.compSamplers[i].sample() for i in comp_state])
        return np.asarray(scatter_component_draws(comp_state, self.compSamplers, int(size)))


class GaussianMixtureAccumulator(MixtureAccumulator):
    """GaussianMixtureAccumulator object used to aggregate sufficient statistics of observed data.

    A thin specialization of :class:`~pysp.stats.latent.mixture.MixtureAccumulator`: the vectorized
    E-step (seq_update / seq_initialize, the fused-EM ``_track_ll`` path), scalar update/initialize,
    combine/value/from_value, and key merge/replace are all inherited unchanged. Only the Gaussian
    data encoder is Gaussian-specific.

    Args:
        accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture
            components (one MultivariateGaussianAccumulator per component).
        keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and mixture components.

    """

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        super().__init__(accumulators, keys=keys)

    def acc_to_encoder(self) -> "GaussianMixtureDataEncoder":
        """Returns a GaussianMixtureDataEncoder object for encoding sequences of iid observations."""
        return GaussianMixtureDataEncoder(encoder=self.accumulators[0].acc_to_encoder())


class GaussianMixtureEstimatorAccumulatorFactory(MixtureAccumulatorFactory):
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
        super().__init__(factories, keys=keys)
        self.dim = dim

    def make(self) -> "GaussianMixtureAccumulator":
        """Returns a GaussianMixtureAccumulator with freshly made component accumulators."""
        return GaussianMixtureAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class GaussianMixtureEstimator(MixtureEstimator):
    """GaussianMixtureEstimator object for estimating GaussianMixtureDistribution from sufficient statistics.

    A thin specialization of :class:`~pysp.stats.latent.mixture.MixtureEstimator`: the weight M-step
    (incl. pseudo_count / suff_stat regularization) is inherited unchanged; only the final repacking of
    the estimated components into a :class:`GaussianMixtureDistribution` (reading each component's
    ``mu`` / ``covar``) is Gaussian-specific.

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
        super().__init__(estimators, suff_stat=suff_stat, pseudo_count=pseudo_count, name=name, keys=keys)
        self.conj_prior_params = conj_prior_params

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

        The component M-step and weight estimate (incl. pseudo_count / suff_stat regularization) reuse
        the generic MixtureEstimator path; the estimated components are then repacked into a
        GaussianMixtureDistribution via each component's ``mu`` / ``covar``.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: Tuple of (component counts, tuple of K component sufficient statistics).

        Returns:
            GaussianMixtureDistribution object.

        """
        base = super().estimate(nobs, suff_stat)
        mu = np.asarray([comp.mu for comp in base.components])
        sig2 = np.asarray([comp.covar for comp in base.components])
        return GaussianMixtureDistribution(mu, sig2, base.w, name=self.name)


class GaussianMixtureDataEncoder(MixtureDataEncoder):
    """GaussianMixtureDataEncoder object for encoding sequences of iid Gaussian mixture observations.

    Args:
        encoder (Optional[DataSequenceEncoder]): DataSequenceEncoder for the component distributions.
            Defaults to a MultivariateGaussianDataEncoder.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for the component distributions.

    """

    def __init__(self, encoder: DataSequenceEncoder | None = None) -> None:
        super().__init__(encoder if encoder is not None else MultivariateGaussianDataEncoder())

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


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
GaussianMixtureAccumulatorFactory = GaussianMixtureEstimatorAccumulatorFactory
