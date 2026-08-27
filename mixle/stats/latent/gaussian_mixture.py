"""Multivariate Gaussian mixtures built on the generic mixture machinery.

The Gaussian mixture classes are thin specializations of
``mixle.stats.latent.mixture``. They construct multivariate-Gaussian
components from packed ``(mu, sig2)`` arrays and reuse the generic mixture
scoring, posterior, EM, fused-likelihood, keying, and stability behavior.

A K-component multivariate Gaussian mixture has density

    P(x) = sum_{k=1}^{K} w_k * N(x; mu_k, sigma_k),

where each observation is a length-d real vector. ``sig2`` may be supplied as
full ``(K, d, d)`` covariance matrices or as per-component diagonal variances
with shape ``(K, d)``.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import maxrandint
from mixle.stats.compute._sampling import scatter_component_draws
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.latent.mixture import (
    MixtureAccumulator,
    MixtureAccumulatorFactory,
    MixtureDataEncoder,
    MixtureDistribution,
    MixtureEstimator,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDataEncoder,
    MultivariateGaussianDistribution,
)
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.deprecation import deprecated_alias


def _pack_sig2(sig2: Sequence[Any] | np.ndarray, num_comp: int, dim: int) -> np.ndarray:
    """Expand a (K, d) diagonal-variance arg to full (K, d, d) covariances; pass (K, d, d) through."""
    sig2_loc = np.asarray(sig2, dtype=float)
    if sig2_loc.ndim == 2:
        sig2_loc = np.asarray([np.diag(u) for u in sig2_loc])
    return np.reshape(sig2_loc, (num_comp, dim, dim))


class GaussianMixtureDistribution(MixtureDistribution):
    """Finite mixture of full-covariance multivariate Gaussian components.

    The constructor packs each ``(mu[k], sig2[k])`` pair into a
    :class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution`.
    Density, posterior responsibility, and vectorized scoring behavior are
    inherited from :class:`~mixle.stats.latent.mixture.MixtureDistribution`.

    Args:
        mu: Component means with shape ``(K, d)``.
        sig2: Component covariance matrices with shape ``(K, d, d)`` or
            diagonal variances with shape ``(K, d)``.
        w: Mixture weights with length ``K``.
        name: Optional diagnostic name.
        weights: Alias for ``w``.

    Attributes:
        mu: Component means as a ``(K, d)`` NumPy array.
        sig2: Component covariance matrices as a ``(K, d, d)`` NumPy array.
        dim: Observation dimension.
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
        if self.mu.ndim != 2:
            # A 1-D mu is the natural first attempt at a univariate mixture; it used to die as a
            # bare ``IndexError: tuple index out of range`` on the shape lookup below.
            raise ValueError(
                f"component means must have shape (K, d) -- one length-d mean vector per component; "
                f"got a {self.mu.ndim}-D array of shape {self.mu.shape}. For a univariate mixture "
                f"write mu=[[m1], [m2], ...] (each mean is a length-1 vector)."
            )

        num_comp = self.mu.shape[0]
        dim = self.mu.shape[1]

        self.sig2 = _pack_sig2(sig2, num_comp, dim)
        self.dim = dim

        components = [MultivariateGaussianDistribution(self.mu[k], self.sig2[k]) for k in range(num_comp)]
        super().__init__(components, np.asarray(w, dtype=float), name=name)

    def __str__(self) -> str:
        """Return a constructor-style representation of the Gaussian mixture."""
        s1 = repr([list(u) for u in self.mu])
        s2 = repr([[list(v) for v in u] for u in self.sig2])
        s3 = repr(list(self.w))
        s4 = repr(self.name)
        return "GaussianMixtureDistribution(%s, %s, %s, name=%s)" % (s1, s2, s3, s4)

    def density_semantics(self):
        """Return exact-or-approximate density semantics joined from Gaussian components."""
        from mixle.stats.compute.pdist import DensitySemantics, join_density_semantics

        children = list(self.components)
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics")]
        return join_density_semantics(sems) if sems else DensitySemantics.EXACT

    def sampler(self, seed: int | None = None) -> "GaussianMixtureSampler":
        """Return a sampler that draws vectors from this Gaussian mixture.

        Args:
            seed: Optional seed for reproducible component and Gaussian draws.

        Returns:
            ``GaussianMixtureSampler`` bound to this distribution.
        """
        return GaussianMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GaussianMixtureEstimator":
        """Return an estimator with matching Gaussian component structure.

        Args:
            pseudo_count: Optional smoothing mass for mixture weights.

        Returns:
            ``GaussianMixtureEstimator`` suitable for fitting length-d vectors.
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
        """Return the Gaussian mixture encoder for iid vector observations."""
        return GaussianMixtureDataEncoder(encoder=self.components[0].dist_to_encoder())


class GaussianMixtureSampler(DistributionSampler):
    """Sampler for iid draws from a Gaussian mixture.

    Args:
        dist: Gaussian mixture distribution to sample from.
        seed: Optional seed for reproducible component and Gaussian draws.

    Attributes:
        rng: Random state used for component labels.
        compSamplers: Per-component Gaussian samplers.
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
    """EM accumulator for Gaussian mixture responsibilities and component stats.

    Vectorized E-step, fused-likelihood tracking, scalar update/initialize,
    combine/value/from_value, and key merge/replace behavior are inherited from
    :class:`~mixle.stats.latent.mixture.MixtureAccumulator`. The specialization
    only supplies the Gaussian mixture encoder.

    Args:
        accumulators: Component accumulators, typically one multivariate
            Gaussian accumulator per component.
        keys: Optional shared-statistic keys for weights and components.
        init: Initialization strategy, one of
            :data:`~mixle.stats.latent.mixture.MIXTURE_INIT_STRATEGIES`.
    """

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] = (None, None),
        init: str = "kmeans++",
    ) -> None:
        super().__init__(accumulators, keys=keys, init=init)

    def acc_to_encoder(self) -> "GaussianMixtureDataEncoder":
        """Return an encoder compatible with the component accumulators."""
        return GaussianMixtureDataEncoder(encoder=self.accumulators[0].acc_to_encoder())


class GaussianMixtureEstimatorAccumulatorFactory(MixtureAccumulatorFactory):
    """Factory for Gaussian mixture EM accumulators.

    Args:
        factories: Component accumulator factories.
        dim: Number of mixture components.
        keys: Optional shared-statistic keys for weights and components.
        init: Initialization strategy, one of
            :data:`~mixle.stats.latent.mixture.MIXTURE_INIT_STRATEGIES`.
    """

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        dim: int,
        keys: tuple[str | None, str | None] = (None, None),
        init: str = "kmeans++",
    ) -> None:
        super().__init__(factories, keys=keys, init=init)
        self.dim = dim

    def make(self) -> "GaussianMixtureAccumulator":
        """Return a fresh Gaussian mixture accumulator."""
        # `init` has to travel with the factory: dropping it here pinned every GaussianMixture fit
        # to the accumulator default, so an estimator configured with a different strategy (or with
        # robust=True) silently got the default one.
        return GaussianMixtureAccumulator(
            [self.factories[i].make() for i in range(self.dim)], self.keys, init=self.init
        )


class GaussianMixtureEstimator(MixtureEstimator):
    """EM estimator for a multivariate Gaussian mixture.

    The weight M-step and robustness options are inherited from
    :class:`~mixle.stats.latent.mixture.MixtureEstimator`. The specialization
    repacks fitted Gaussian components into ``(mu, sig2, w)`` arrays for a
    :class:`GaussianMixtureDistribution`.

    Args:
        estimators: Component estimators whose fitted components carry a mean
            vector and a covariance (``mu`` and ``covar``) -- typically
            :class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianEstimator`
            or :class:`~mixle.stats.multivariate.diagonal_gaussian.DiagonalGaussianEstimator`.
            The univariate ``GaussianEstimator`` is refused at construction (its
            fitted component has no covariance to pack); spell a univariate
            Gaussian mixture ``MixtureEstimator([GaussianEstimator(), ...])``.
        name: Optional diagnostic name.
        conj_prior_params: Reserved compatibility slot.
        suff_stat: Optional prior component-count vector used with
            ``pseudo_count``.
        pseudo_count: Optional smoothing mass for mixture weights.
        keys: Optional shared-statistic keys for weights and components.
        robust: Enable the robust weight floor, as on
            :class:`~mixle.stats.latent.mixture.MixtureEstimator`.
        init: Initialization strategy, one of
            :data:`~mixle.stats.latent.mixture.MIXTURE_INIT_STRATEGIES`; ``None`` selects the
            default ``"kmeans++"``.
    """

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        name: str | None = None,
        conj_prior_params: Any | None = None,
        suff_stat: np.ndarray | None = None,
        pseudo_count: float | None = None,
        keys: tuple[str | None, str | None] = (None, None),
        robust: bool = False,
        init: str | None = None,
    ) -> None:
        super().__init__(
            estimators,
            suff_stat=suff_stat,
            pseudo_count=pseudo_count,
            name=name,
            keys=keys,
            robust=robust,
            init=init,
        )
        # T4-9 boundary check for THE common first-attempt mistake: univariate GaussianEstimator
        # components. They run a whole E-step fine and only die at the end of the first M-step with
        # a bare ``AttributeError: 'GaussianDistribution' object has no attribute 'covar'`` from the
        # (mu, covar) repack below. No legitimate use is refused: a scalar Gaussian component can
        # never survive that repack, and a univariate Gaussian mixture is spelled
        # ``MixtureEstimator([GaussianEstimator(), ...])``. Deliberately a blocklist of exactly this
        # class -- duck-typed vector components (anything whose fitted distribution carries ``mu``
        # and ``covar``, like DiagonalGaussianEstimator) must keep working, so unknown estimator
        # types pass here and are checked against the actual repack contract in :meth:`estimate`.
        from mixle.stats.univariate.continuous.gaussian import GaussianEstimator as _ScalarGaussian

        for i, est in enumerate(self.estimators):
            if isinstance(est, _ScalarGaussian):
                raise TypeError(
                    "GaussianMixtureEstimator: estimators[%d] is the univariate GaussianEstimator, "
                    "whose fitted component has a scalar mean and variance, but "
                    "GaussianMixtureEstimator packs vector components with a mean vector and a "
                    "covariance matrix (e.g. MultivariateGaussianEstimator(dim=d) or "
                    "DiagonalGaussianEstimator(dim=d)). For a mixture of univariate Gaussians use "
                    "MixtureEstimator([GaussianEstimator(), ...]) instead." % i
                )
        self.conj_prior_params = conj_prior_params

    def accumulator_factory(self) -> "GaussianMixtureEstimatorAccumulatorFactory":
        """Return a Gaussian mixture accumulator factory."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return GaussianMixtureEstimatorAccumulatorFactory(est_factories, self.num_components, self.keys, init=self.init)

    @deprecated_alias("accumulator_factory", since="0.8.0", removed_in="0.10.0")
    def accumulatorFactory(self) -> "GaussianMixtureEstimatorAccumulatorFactory":
        """Deprecated alias for :meth:`accumulator_factory`.

        New code should call ``accumulator_factory``. The camelCase name remains
        available for older callers and returns the same factory, but now emits a
        ``DeprecationWarning`` (see ``docs/support-policy.rst``).
        """
        return self.accumulator_factory()

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, tuple[Any, ...]]
    ) -> "GaussianMixtureDistribution":
        """Estimate a Gaussian mixture from aggregated EM sufficient statistics.

        The component M-step and weight estimate reuse the generic mixture
        estimator path. The fitted component means and covariances are then
        repacked into a ``GaussianMixtureDistribution``.

        Args:
            nobs: Unused compatibility argument from ``ParameterEstimator``.
            suff_stat: ``(component_counts, component_suff_stats)`` tuple.

        Returns:
            Fitted Gaussian mixture distribution.
        """
        base = super().estimate(nobs, suff_stat)
        try:
            mu = np.asarray([comp.mu for comp in base.components])
            sig2 = np.asarray([comp.covar for comp in base.components])
        except AttributeError as exc:
            # The repack contract is ``mu`` + ``covar`` on every fitted component. Any other
            # component family (say, ExponentialEstimator) sails through EM and used to die right
            # here as a bare AttributeError from library internals (T4-9). This is the one place
            # the incompatibility is proven, so it is where non-Gaussian duck types are refused --
            # the constructor blocks only the provably wrong univariate GaussianEstimator.
            bad = next(
                (c for c in base.components if not (hasattr(c, "mu") and hasattr(c, "covar"))),
                base.components[0] if base.components else None,
            )
            raise TypeError(
                "GaussianMixtureEstimator component estimators must fit Gaussian components with "
                "a mean vector and a covariance (`mu` and `covar`), but a component fit produced "
                "%s, which does not carry both; use MultivariateGaussianEstimator / "
                "DiagonalGaussianEstimator components here, or MixtureEstimator for a mixture of "
                "non-Gaussian components." % type(bad).__name__
            ) from exc
        packed = GaussianMixtureDistribution(mu, sig2, base.w, name=self.name)
        # The repack builds a FRESH distribution from bare parameter arrays, which discarded the
        # component objects -- and with them any covariance-ridge/variance-floor repair records the
        # component estimators had attached, so a ridged mixture fit read as repair-free
        # (wave-3 adversarial check, B5). Carry the records across, component-indexed.
        repairs = tuple(
            f"component[{i}]: {note}"
            for i, comp in enumerate(base.components)
            for note in getattr(comp, "_numerical_repairs", ())
        )
        if repairs:
            packed._numerical_repairs = repairs + tuple(getattr(packed, "_numerical_repairs", ()))
        return packed


class GaussianMixtureDataEncoder(MixtureDataEncoder):
    """Encoder for iid vector observations in a Gaussian mixture.

    Args:
        encoder: Optional component encoder. Defaults to
            ``MultivariateGaussianDataEncoder``.
    """

    def __init__(self, encoder: DataSequenceEncoder | None = None) -> None:
        super().__init__(encoder if encoder is not None else MultivariateGaussianDataEncoder())

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "GaussianMixtureDataEncoder(" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is an equivalent Gaussian mixture encoder.

        Args:
            other: Object to compare against.

        Returns:
            True if other is a GaussianMixtureDataEncoder with an equal component encoder.
        """
        if isinstance(other, GaussianMixtureDataEncoder):
            return self.encoder == other.encoder
        else:
            return False


# --- Backward-compatible API naming aliases ---
GaussianMixtureAccumulatorFactory = GaussianMixtureEstimatorAccumulatorFactory
