"""Finite mixtures whose components may use different distribution families.

HeterogeneousMixtureDistribution with data type ``T`` is defined by the density

p_mat(Y) = sum_{k=1}^{K} p_mat(Y|Z=k)*p_mat(Z=k),

where p_mat(Z=k) is a mixture weight and p_mat(Y|Z=k) is the kth component distribution. Component distributions must
accept the same observation type ``T`` but do not need to share parameters or encoders.

Example: A heterogeneous mixture with weights [0.5, 0.5] and component distribution Exponential(beta) and Gamma(k,theta),
has form
    p_mat(x_mat) = 0.5*P_0(x; beta) + 0.5*P_1(x; k, theta), for x > 0.0,
where
    P_0(x;beta) is an exponential density and P_1(x; k, theta) is a Gamma density.
"""

from collections.abc import Sequence
from math import exp
from typing import Any, TypeVar

import numpy as np
from numpy import ndarray
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, best_first_union
from mixle.stats.compute._sampling import scatter_component_draws
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
from mixle.utils.aliasing import MISSING, coalesce_alias

T = TypeVar("T")


class HeterogeneousMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Mixture distribution with component-specific observation encoders."""

    def compute_capabilities(self):
        """Return compute-backend metadata shared by heterogeneous mixture components."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(
            engine_ready=intersect_engine_ready(tuple(self.components)), kernel_status="generic_latent"
        )

    def compute_declaration(self):
        """Return the symbolic declaration for heterogeneous mixture weights and components."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        children = tuple(declaration_for(component) for component in self.components)
        children = tuple(child for child in children if child is not None)
        return DistributionDeclaration(
            name="heterogeneous_mixture",
            distribution_type=type(self),
            parameters=(ParameterSpec("w", constraint="simplex_vector"),),
            statistics=(
                StatisticSpec("component_counts"),
                StatisticSpec("components", kind="tuple"),
            ),
            support="mixture",
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
        """Create a heterogeneous mixture from component distributions and weights.

        The component distributions define both the mixture branches and the accepted observation type ``T``. Components
        may use different internal parameterizations and encoders as long as each can score observations of type ``T``.

        Args:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Component distributions.
            w (ndarray[float]): Mixture weights. Values should be non-negative and sum to one.
            name (Optional[str]): Optional distribution name.

        Attributes:
            components (List[SequenceEncodableProbabilityDistribution]): Component distributions over type ``T``.
            w (ndarray[float]): Mixture weights.
            name (Optional[str]): Optional distribution name.
            zw (ndarray[bool]): Mask for zero-weight components.
            log_w (ndarray[float]): Log mixture weights, with zero-weight components set to ``-np.inf``.
            num_components (int): Number of mixture components.

        """
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        self.w = np.asarray(w, dtype=float)
        self.zw = self.w == 0.0
        self.log_w = np.log(self.w + self.zw)
        self.log_w[self.zw] = -np.inf

        self.components = components
        self.num_components = len(components)
        self.name = name

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = ",".join([str(u) for u in self.components])
        s2 = repr(list(self.w))
        s3 = repr(self.name)

        return "HeterogeneousMixtureDistribution(components=[%s], w=%s, name=%s)" % (s1, s2, s3)

    def density(self, x: T) -> float:
        """Evaluate the density of the heterogeneous mixture at one observation.

        See log_density() for details.

        Args:
            x: (T): Single observation from heterogeneous mixture distribution. T is data type of components.

        Returns:
            Density at x.

        """
        return exp(self.log_density(x))

    def density_semantics(self):
        """Return joined density semantics over heterogeneous mixture components."""
        from mixle.stats.compute.pdist import join_density_semantics

        return join_density_semantics(c.density_semantics() for c in self.components)

    def log_density(self, x: T) -> float:
        """Evaluate log-density of heterogeneous mixture distribution at observation x.

        A K-component heterogeneous mixture has log-density,

            log(p_mat(x)) = log(sum_{z=k}^{K} p_mat(x|z=k)*p_mat(z=k)),

        where p_mat(x|z=k) is component-k log-density at x, and p_mat(z=k) = w[k]. A log-sum-exp is used to evaluate the
        sum inside the log of the right-hand side above. (See mixle.utils.vector.log_sum() for details).

        Recall: p_mat(x|z=k) need only be compatible with same data type T. They are need not be the same distribution.

        Args:
            x: (T): Single observation from heterogeneous mixture distribution. T is data type of components.

        Returns:
            Log-density at x.

        """
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def component_log_density(self, x: T) -> np.ndarray:
        """Evaluate component-wise log-density of heterogeneous mixture distribution at observation x.

        A K-component heterogeneous mixture has log-density,

            log(p_mat(x)) = log(sum_{z=k}^{K} p_mat(x|z=k)*p_mat(z=k)),

        where p_mat(x|z=k) is component-k log-density at x, and p_mat(z=k) = w[k].

        This function returns an ndarray[float] of length K, containing log(p_mat(x|z=k)) as its k^{th} entry.

        Args:
            x: (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            Numpy array of floats containing component-wise log-density at x.

        """
        return np.asarray([m.log_density(x) for m in self.components], dtype=np.float64)

    def posterior(self, x: T) -> np.ndarray:
        """Obtain the posterior distribution for each heterogeneous mixture component at observation x.

        The posterior distribution of component 'k' at observation x is given by,

            (1) p_mat(Z=k|x) = p_mat(x|Z=k)*p_mat(z=k) / p_mat(x),

        where

            (2) p_mat(x) = sum_{k=1}^{K} p_mat(x|Z=k)*p_mat(z=k) = sum_{k=1}^{K} p_mat(x|Z=k)*w[k].


        This function returns an ndarray[float] of length K, containing p_mat(Z=k|x) as its k^{th} entry.

        Args:
            x: (T): Single observation from heterogeneous mixture distribution. T is data type of components.

        Returns:
            Numpy array of floats containing posterior distribution at observation x.

        """
        comp_log_density = np.asarray([m.log_density(x) for m in self.components])
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -np.inf

        max_val = np.max(comp_log_density)

        if max_val == -np.inf:
            return self.w.copy()
        else:
            comp_log_density -= max_val
            np.exp(comp_log_density, out=comp_log_density)
            comp_log_density /= comp_log_density.sum()

            return comp_log_density

    def seq_log_density(self, x: tuple[list[np.ndarray], list[Any]]) -> np.ndarray:
        """Vectorized evaluation of component-wise log-density for encoded sequence x.

        Evaluates the log-density of each observation in the encoded sequence x (see log_density() for details).

        Arg x must be a Tuple of length two containing and encoded from
            HeterogeneousMixtureDataEncoder.seq_encode(data) with data type Sequence[T] for data.

        x[0] (List[np.ndarray[int]]): The component ids for each distinct SequenceEncodableProbabilityDistribution
            subclass.
        x[1] (List[T1,T2,..Tk]): A list of sequence encodings of iid an iid observation sequence for each
            'k' distinct SequenceEncodableProbabilityDistribution subclasses. The data type for each encoding is assumed
            to be of type Ti.

        The returned value is an ndarray[float] with shape (sz,K), where K is the number of mixture components, and
        sz is the number of iid observations in the encoded sequence x.

        Note: A row-wise log-sum-exp is performed for numerical stability. If a row contains a log-density value of,
         -np.inf is returned for the corresponding observation value in the encoded sequence x.

        Args:
            x: See above for details.

        Returns:
            Numpy array of floats containing the log_density of each observation in encoded sequence.

        """
        tag_list, enc_data = x

        comp_to_tag = {}
        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                comp_to_tag[int(i)] = tag

        base_tag = comp_to_tag.get(0, 0)
        sz = len(self.components[0].seq_log_density(enc_data[base_tag]))
        ll_mat = np.zeros((sz, self.num_components))
        ll_mat.fill(-np.inf)

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    ll_mat[:, i] = temp
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

    def seq_component_log_density(self, x: tuple[list[np.ndarray], list[Any]]) -> np.ndarray:
        """Vectorized evaluation of component-wise log-density for encoded sequence x.

        Arg x must be a Tuple of length two containing and encoded from
            HeterogeneousMixtureDataEncoder.seq_encode(data) with data type Sequence[T] for data.

        x[0] (List[np.ndarray[int]]): The component ids for each distinct SequenceEncodableProbabilityDistribution
            subclass.
        x[1] (List[T1,T2,..Tk]): A list of sequence encodings of iid an iid observation sequence for each
            'k' distinct SequenceEncodableProbabilityDistribution subclasses. The data type for each encoding is assumed
            to be of type Ti.

        Creates a 2-d numpy array of floats with vectorized evaluations of component_log_density() stored in the rows
        corresponding to an observation in encoded sequence x.

        The returned value is an ndarray[float] with shape (sz,K), where K is the number of mixture components, and
        sz is the number of iid observations in the encoded sequence x.

        Args:
            x: See above for details.

        Returns:
            2-d numpy array of floats having shape (sz,K), where sz is the number of iid obs in encoded sequence x, and
            K is the number of mixture components.

        """
        tag_list, enc_data = x
        ll_mat_init = False

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    if not ll_mat_init:
                        ll_mat = np.zeros((len(temp), self.num_components))
                        ll_mat.fill(-np.inf)
                        ll_mat_init = True
                    ll_mat[:, i] = temp

        return ll_mat

    def backend_seq_component_log_density(self, x: tuple[list[np.ndarray], list[Any]], engine: Any) -> Any:
        """Engine-neutral component log densities for heterogeneous encoded data."""
        from mixle.stats.compute.backend import backend_seq_log_density

        tag_list, enc_data = x
        comp_to_tag = {}
        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                comp_to_tag[int(i)] = tag

        base_tag = comp_to_tag.get(0, 0)
        base = backend_seq_log_density(self.components[0], enc_data[base_tag], engine)
        ll_mat = engine.zeros((len(base), self.num_components)) + engine.asarray(-np.inf)

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                i = int(i)
                if not self.zw[i]:
                    ll_mat[:, i] = backend_seq_log_density(self.components[i], enc_data[tag], engine)

        return ll_mat

    def backend_seq_log_density(self, x: tuple[list[np.ndarray], list[Any]], engine: Any) -> Any:
        """Engine-neutral heterogeneous-mixture log-density for encoded data."""
        ll_mat = self.backend_seq_component_log_density(x, engine)
        return engine.logsumexp(ll_mat + engine.asarray(self.log_w), axis=1)

    def seq_posterior(self, x: tuple[list[np.ndarray], list[Any]]) -> np.ndarray:
        """Vectorized evaluation of posterior of HeterogeneousMixtureDistribution for encoded sequence x.

        Arg x must be a Tuple of length two containing and encoded from
            HeterogeneousMixtureDataEncoder.seq_encode(data) with data type Sequence[T] for data.

        x[0] (List[np.ndarray[int]]): The component ids for each distinct SequenceEncodableProbabilityDistribution
            subclass.
        x[1] (List[T1,T2,..Tk]): A list of sequence encodings of iid an iid observation sequence for each
            'k' distinct SequenceEncodableProbabilityDistribution subclasses. The data type for each encoding is assumed
            to be of type Ti.

        Vectorized evaluation the posterior of each observation in the encoded sequence x (see posterior() for details).

        The returned value is an ndarray[float] with shape (sz,K), where K is the number of mixture components, and
        sz is the number of iid observations in the encoded sequence x. Each row contains the posterior of the
        corresponding encoded observation.

        Note: A row-wise log-sum-exp is performed for numerical stability. If a row contains a log-density value of,
         -np.inf is returned for the corresponding observation value in the encoded sequence x.

        Args:
            x: See above for details.

        Returns:
            Numpy array of floats containing the posterior of each observation in encoded sequence.

        """
        tag_list, enc_data = x
        ll_mat_init = False

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    if not ll_mat_init:
                        ll_mat = np.zeros((len(temp), self.num_components))
                        ll_mat.fill(-np.inf)
                        ll_mat_init = True
                    ll_mat[:, i] = temp
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

    def sampler(self, seed: int | None = None) -> "HeterogeneousMixtureSampler":
        """Return a sampler for this heterogeneous mixture.

        Args:
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Returns:
            HeterogeneousMixtureSampler object.

        """
        return HeterogeneousMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HeterogeneousMixtureEstimator":
        """Return an estimator with one child estimator per component.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics in estimation.

        Returns:
            HeterogeneousMixtureEstimator object.

        """
        if pseudo_count is not None:
            return HeterogeneousMixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count,
                name=self.name,
            )
        else:
            return HeterogeneousMixtureEstimator([u.estimator() for u in self.components], name=self.name)

    def decomposition(self):
        """Heterogeneous mixture components split along the component axis (logsumexp responsibilities
        inside a shard; per-component stats SUM-reduce). Components are NOT homogeneous, so there is no
        stacked-parameter tensor to DTensor-shard (engine_axis=None -> host-shard executor mode)."""
        from mixle.stats.compute.decomposition import DecompAxis, Decomposition, ReductionOp

        return Decomposition(
            axis=DecompAxis.COMPONENT,
            num_units=self.num_components,
            reduction=ReductionOp.LOGSUMEXP_RESPONSIBILITY,
            exact=True,
            child_roles=("component",) * self.num_components,
            engine_axis=None,
            key_pooling=getattr(self, "keys", None) is not None,
        )

    def dist_to_encoder(self) -> "HeterogeneousMixtureDataEncoder":
        """Return a data encoder assembled from the component distribution encoders."""
        encoders = [comp.dist_to_encoder() for comp in self.components]

        return HeterogeneousMixtureDataEncoder(encoders=encoders)

    def enumerator(self) -> "HeterogeneousMixtureEnumerator":
        """Returns a HeterogeneousMixtureEnumerator iterating the union of component supports in
        descending mixture probability order."""
        return HeterogeneousMixtureEnumerator(self)


class HeterogeneousMixtureEnumerator(DistributionEnumerator):
    """Enumerator over the deduplicated union of heterogeneous component supports."""

    def __init__(self, dist: HeterogeneousMixtureDistribution) -> None:
        """Enumerates the union of component supports in descending mixture probability order.

        Same algorithm as MixtureEnumerator: candidates from the component enumerations are
        re-scored exactly with the mixture log-density and emitted once they beat the upper
        bound on unseen values. Zero-weight components are never asked to enumerate.

        Args:
            dist (HeterogeneousMixtureDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        streams = []
        log_offsets = []
        comps = []
        for k, comp in enumerate(dist.components):
            if dist.w[k] <= 0.0:
                continue
            streams.append(
                BufferedStream(child_enumerator(comp, "HeterogeneousMixtureDistribution.components[%d]" % k))
            )
            log_offsets.append(dist.log_w[k])
            comps.append(comp)
        log_w_arr = np.asarray(log_offsets, dtype=np.float64)

        # Equivalent to dist.log_density but restricted to positive-weight components, so a
        # zero-weight component never sees (possibly type-incompatible) candidate values.
        def exact_log_density(x):
            with np.errstate(divide="ignore"):
                return vec.log_sum(np.asarray([c.log_density(x) for c in comps]) + log_w_arr)

        self._union = best_first_union(streams, log_offsets, exact_log_density)

    def __next__(self) -> tuple[Any, float]:
        return next(self._union)


class HeterogeneousMixtureSampler(DistributionSampler):
    """Sampler that draws a component id and delegates to that component sampler."""

    def __init__(self, dist: HeterogeneousMixtureDistribution, seed: int | None = None):
        """HeterogeneousMixtureSampler used to generate samples from instance of HeterogeneousMixtureDistribution.

        Args:
            dist (HeterogeneousMixtureDistribution): Assign HeterogeneousMixtureDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Attributes:
            dist (CompositeDistribution): CompositeDistribution to draw samples from.
            rng (RandomState): Seeded RandomState for sampling.
            comp_samplers (List[DistributionSamplers]): List of DistributionSampler objects for each mixture component.

        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any | list[Any]:
        """Draw iid samples from a heterogeneous mixture distribution.

        The data type drawn from 'comp_samplers' is type T, corresponding to the data type of the mixture components.

        If size is None, a single sample (of data type T) is drawn and returned. If size is not None, 'size'-iid
        heterogeneous mixture samples are drawn and returned as a List with data type List[T]. With ``batched=True``
        (default) component draws are grouped and scattered -- bit-identical to the per-draw loop (``batched=False``)
        but far faster, since each component sampler owns an independent RNG.

        Args:
            size (Optional[int]): Number of iid samples to draw.
            batched (bool): Vectorize component draws (default); set False for the per-draw loop.

        Returns:
            Data type T or List[T].

        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.comp_samplers[comp_state].sample()
        if not batched:
            return [self.comp_samplers[i].sample() for i in comp_state]
        return scatter_component_draws(comp_state, self.comp_samplers, int(size))


class HeterogeneousMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """EM accumulator for heterogeneous mixture weights and per-component statistics."""

    def __init__(
        self,
        accumulators: list[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        """Create an accumulator for heterogeneous-mixture EM statistics.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Sequence of
                SequenceEncodableStatisticAccumulator instances for the mixture components.
            keys (Tuple[Optional[str], Optional[str]]): Optional keys for mixture weights and component statistics.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Sequence of
                SequenceEncodableStatisticAccumulator instances for the mixture components.
            num_components (int): Number of mixture components.
            comp_counts (np.ndarray[float]): Accumulated posterior component mass.
            weight_key (Optional[str]): Key for sharing mixture-weight statistics.
            comp_key (Optional[str]): Key for sharing component sufficient statistics.
            _init_rng (bool): Whether initialization random states have been assigned.
            _acc_rng (Optional[List[RandomState]]): Random states used to initialize component accumulators.
        """
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

        ### Initializer seeds.
        self._init_rng: bool = False
        self._acc_rng: list[RandomState] | None = None

    def update(self, x: T, weight: float, estimate: HeterogeneousMixtureDistribution) -> None:
        """Update sufficient statistics of HeterogeneousMixtureAccumulator with weighted observation.

        Requires previous estimate of HeterogeneousMixtureDistribution.

        Weights posterior of 'estimate' at x. Adds sum to comp_counts, then passes posterior[i] as weight for x
        into update() call of accumulator[i].

        Args:
            x (T): Observation of heterogeneous mixture distribution.
            weight (float): Weight for observation.
            estimate (HeterogeneousMixtureDistribution): Previous iteration of EM estimate for
                HeterogeneousMixtureDistribution.

        Returns:
            None.

        """
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior

        for i in range(self.num_components):
            self.accumulators[i].update(x, posterior[i], estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize accumulator random states from ``rng``.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _rng_acc member variable.

        Returns:
            None.

        """
        seeds = rng.randint(2**31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._init_rng = True

    def initialize(self, x: T, weight: float, rng: np.random.RandomState) -> None:
        """Initialize heterogeneous-mixture sufficient statistics from one weighted observation.

        If _init_rng is False, _acc_rng is set with rng. This is done for consistency in initialize and seq_initialize
        functions.

        Initialize heterogeneous mixture weights with a sample from Dirichlet distribution. Each
        SequenceEncodableStatisticAccumulator is for the mixture components is initialized with a call to
        accumulator[i].initialize.

        Args:
            x (T): Observation of heterogeneous mixture distribution.
            weight (float): Weight for observation.
            rng (RandomState): Used to set _acc_rng if not previously set.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if weight != 0:
            ww = rng.dirichlet(np.ones(self.num_components) / (self.num_components * self.num_components))
        else:
            ww = np.zeros(self.num_components)

        for i in range(self.num_components):
            w = weight * ww[i]
            self.accumulators[i].initialize(x, w, self._acc_rng[i])
            self.comp_counts[i] += w

    def seq_initialize(self, x: tuple[list[np.ndarray], list[Any]], weights: np.ndarray, rng: RandomState) -> None:
        """Initialize heterogeneous-mixture sufficient statistics from encoded observations.

        If _init_rng is False, _acc_rng is set with rng. This is done for consistency in initialize and seq_initialize
        functions.

        Arg x must be a Tuple of length two containing and encoded from
            HeterogeneousMixtureDataEncoder.seq_encode(data) with data type Sequence[T] for data.

        x[0] (List[np.ndarray[int]]): The component ids for each distinct SequenceEncodableProbabilityDistribution
            subclass.
        x[1] (List[T1,T2,..Tk]): A list of sequence encodings of iid an iid observation sequence for each
            'k' distinct SequenceEncodableProbabilityDistribution subclasses. The data type for each encoding is assumed
            to be of type Ti.

        Vectorized implementation of initialize(), for sequence encoded x.

        Args:
            x: See above for details.
            weights (ndarray[float]): Numpy array of positive valued floats.
            rng (RandomState): Used to set _acc_rng if not previously set.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        tag_list, enc_data = x
        sz = len(weights)

        keep_idx = weights > 0.0
        keep_len = np.sum(keep_idx)
        ww = np.zeros((sz, self.num_components))

        if keep_len > 0:
            ww[keep_idx, :] = rng.dirichlet(
                alpha=np.ones(self.num_components) / (self.num_components**2), size=keep_len
            )
        ww *= np.reshape(weights, (sz, 1))

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                self.accumulators[i].seq_initialize(enc_data[tag], ww[:, i], self._acc_rng[i])
                self.comp_counts[i] += np.sum(ww[:, i])

    def seq_update(
        self, x: tuple[list[np.ndarray], list[Any]], weights: np.ndarray, estimate: "HeterogeneousMixtureDistribution"
    ) -> None:
        """Vectorized update of sufficient statistics from encoded sequence of observations x.

        Arg x must be a Tuple of length two containing and encoded from
            HeterogeneousMixtureDataEncoder.seq_encode(data) with data type Sequence[T] for data.

        x[0] (List[np.ndarray[int]]): The component ids for each distinct SequenceEncodableProbabilityDistribution
            subclass.
        x[1] (List[T1,T2,..Tk]): A list of sequence encodings of iid an iid observation sequence for each
            'k' distinct SequenceEncodableProbabilityDistribution subclasses. The data type for each encoding is assumed
            to be of type Ti.

        Note: Requires a previous estimate of HeterogeneousMixtureDistribution be passed. This may require
        seq_initialize() to be invoked prior to performing seq_update() calls.

        Seq_update is similar to HeterogeneousMixtureDistribution.seq_posterior(). Results are aggregated to
        comp_counts and accumulators.

        Args:
            x: See above for details.
            weights (np.ndarray[float]): Numpy array of positive floats.
            estimate (MixtureDistribution): Previous heterogeneous-mixture estimate from EM.

        Returns:
            None.

        """
        tag_list, enc_data = x
        ll_mat_init = False

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not estimate.zw[i]:
                    temp = estimate.components[i].seq_log_density(enc_data[tag])
                    if not ll_mat_init:
                        ll_mat = np.zeros((len(temp), self.num_components), dtype=np.float64)
                        ll_mat.fill(-np.inf)
                        ll_mat_init = True
                    ll_mat[:, i] = temp
                    ll_mat[:, i] += estimate.log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = estimate.log_w.copy()
        ll_max[bad_rows] = np.max(estimate.log_w)

        # Capture per-row data log-likelihood (== seq_log_density) by reusing the rowmax and rowsum
        # already computed for normalization: row_ll = rowmax + log(rowsum), with -inf for invalid
        # rows seq_log_density also reports as -inf. Free except an O(n) log/dot, and only when the
        # fused-EM fast path requests it (_track_ll).
        track = self._track_ll and ll_mat_init
        rowmax = ll_max[:, 0].copy() if track else None

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)

        if track:
            with np.errstate(divide="ignore"):
                row_ll = rowmax + np.log(ll_max[:, 0])
            if np.any(bad_rows):
                row_ll[bad_rows] = -np.inf
            self._seq_ll += float(np.dot(weights, row_ll))

        np.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                w_loc = ll_mat[:, i]
                self.comp_counts[i] += w_loc.sum()
                self.accumulators[i].seq_update(enc_data[tag], w_loc, estimate.components[i])

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident mixture E-step (numpy or torch).

        Component log-densities are scored on the active engine and combined into the posterior
        responsibilities with an engine softmax; per-component responsibilities are fed to the child
        accumulators. Mirrors seq_update (heterogeneous components keep per-type encodings).
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        tag_list, enc_data = x
        num_comp = self.num_components
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)

        cols = [None] * num_comp
        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not estimate.zw[i]:
                    cols[i] = backend_seq_log_density(estimate.components[i], enc_data[tag], engine) + engine.asarray(
                        float(estimate.log_w[i])
                    )
        n = None
        for c in cols:
            if c is not None:
                n = int(np.asarray(engine.to_numpy(c)).shape[0])
                break
        if n is None:
            return

        neg = engine.asarray(-np.inf)
        col_list = [cols[i] if cols[i] is not None else (engine.zeros(n) + neg) for i in range(num_comp)]
        ll_mat = engine.stack(col_list, axis=1)

        log_w_e = engine.asarray(np.asarray(estimate.log_w, dtype=np.float64))
        ll_max = engine.max(ll_mat, axis=1)
        bad = engine.isinf(ll_max)
        ll_mat = engine.where(bad[:, None], log_w_e[None, :], ll_mat)
        ll_max = engine.where(bad, engine.asarray(float(np.max(estimate.log_w))), ll_max)
        ll_mat = engine.exp(ll_mat - ll_max[:, None])
        denom = engine.sum(ll_mat, axis=1)
        ll_mat = ll_mat * (engine.asarray(weights_np) / denom)[:, None]

        ll_np = np.asarray(engine.to_numpy(ll_mat))
        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                w_loc = ll_np[:, i]
                self.comp_counts[i] += w_loc.sum()
                self.accumulators[i].seq_update(enc_data[tag], w_loc, estimate.components[i])

    def combine(self, suff_stat: tuple[np.ndarray, tuple[Any, ...]]) -> "HeterogeneousMixtureAccumulator":
        """Merge the sufficient statistics of suff_stat with HeterogeneousMixtureAccumulator instance.

        Arg suff_stat is a Tuple of length two containing,
            suff_stat[0] (ndarray[float]): Aggregated component counts,
            suff_stat[1] (Tuple[T1,...,Tk]): Tuple of K sufficient statistics for the heterogeneous mixture components.

        Note: The components k^{th} heterogeneous mixture component is assumed to have sufficient statistics of type Tk.

        Args:
            suff_stat: See above for details.

        Returns:
            HeterogeneousMixtureAccumulator object.

        """
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Returns sufficient statistics of MixtureAccumulator instance.

        The sufficient statistics value returned (suff_stat) is a Tuple of length two containing,
            suff_stat[0] (ndarray[float]): Aggregated component counts,
            suff_stat[1] (Tuple[T1,...,Tk]): Tuple of K sufficient statistics for the heterogeneous mixture components.

        Note: The components k^{th} heterogeneous mixture component is assumed to have sufficient statistics of type Tk.

        Returns:
            Tuple[np.ndarray[float], Tuple[T1,...,Tk]] as described above.

        """
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: tuple[np.ndarray, tuple[Any, ...]]) -> "HeterogeneousMixtureAccumulator":
        """Set sufficient statistics of HeterogeneousMixtureAccumulator instance to x.

        The sufficient statistics value 'x' is a Tuple of length two containing,
            x[0] (ndarray[float]): Aggregated component counts,
            x[1] (Tuple[T1,...,Tk]): Tuple of K sufficient statistics for the heterogeneous mixture components.

        Note: The components k^{th} heterogeneous mixture component is assumed to have sufficient statistics of type Tk.

        Args:
            x: See above for details.

        Returns:
            HeterogeneousMixtureAccumulator object.

        """
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into keyed sufficient statistics.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

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
        """Replace this accumulator's statistics from matching keyed values.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

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

    def acc_to_encoder(self) -> "HeterogeneousMixtureDataEncoder":
        """Return a data encoder assembled from the component accumulator encoders."""
        encoders = [comp.acc_to_encoder() for comp in self.accumulators]

        return HeterogeneousMixtureDataEncoder(encoders=encoders)


class HeterogeneousMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for heterogeneous-mixture accumulators."""

    def __init__(
        self,
        factories: list[StatisticAccumulatorFactory],
        dim: int,
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        """Create a factory for heterogeneous-mixture accumulators.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Factories for the component accumulators.
            dim (int): Number of mixture components.
            keys (Tuple[Optional[str], Optional[str]]): Optional keys for weights and component statistics.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): Factories for the component accumulators.
            dim (int): Number of mixture components. Must equal length of factories.
            keys (Tuple[Optional[str], Optional[str]]): Optional keys for weights and component statistics.

        """
        self.factories = factories
        self.dim = dim
        self.keys = keys

    def make(self) -> "HeterogeneousMixtureAccumulator":
        """Return a new heterogeneous-mixture accumulator."""
        return HeterogeneousMixtureAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class HeterogeneousMixtureEstimator(ParameterEstimator):
    """Estimator for heterogeneous mixture weights and component distributions."""

    def __init__(
        self,
        estimators: list[ParameterEstimator],
        fixed_weights: np.ndarray | None = None,
        suff_stat: np.ndarray | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        """Estimate heterogeneous-mixture weights and component distributions from sufficient statistics.

        Args:
            estimators (Sequence[ParameterEstimator]): Estimators for the component distributions.
            fixed_weights (Optional[np.ndarray]): Fixed mixture weights. When supplied, observed component counts do not
                update the weights.
            suff_stat (Optional[np.ndarray]): Prior mixture-weight statistics, one value per component.
            pseudo_count (Optional[float]): Prior mass used to smooth or blend mixture weights.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Tuple[Optional[str], Optional[str]]): Optional keys for weights and component statistics.

        Attributes:
            estimators (Sequence[ParameterEstimator]): Estimators for the component distributions.
            fixed_weights (Optional[np.ndarray]): Fixed mixture weights.
            suff_stat (Optional[np.ndarray]): Prior mixture-weight statistics.
            pseudo_count (Optional[float]): Prior mass used to smooth or blend mixture weights.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Tuple[Optional[str], Optional[str]]): Optional keys for weights and component statistics.

        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_weights = fixed_weights

    def accumulator_factory(self) -> "HeterogeneousMixtureAccumulatorFactory":
        """Return an accumulator factory built from the component estimator factories."""
        est_factories = [u.accumulator_factory() for u in self.estimators]

        return HeterogeneousMixtureAccumulatorFactory(est_factories, self.num_components, self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, tuple[Any, ...]]
    ) -> "HeterogeneousMixtureDistribution":
        """Estimate a heterogeneous mixture from aggregated sufficient statistics.

        ``suff_stat`` is a two-item tuple containing:
            suff_stat[0] (np.ndarray): Sufficient statistics for mixture weights.
            suff_stat[1] (Tuple[T1,...,Tk]): Tuple of K sufficient statistics for the heterogeneous mixture components.

        ``suff_stat[1]`` is passed to the corresponding component estimator.

        If ``fixed_weights`` is not ``None``, ``suff_stat[0]`` is ignored and the estimated distribution uses the fixed
        weights.

        If ``pseudo_count`` is set, observed counts are blended with either the estimator's prior statistics or a uniform
        pseudo-count allocation.

        If ``pseudo_count`` is ``None``, observed component counts determine the weights.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator super class.
            suff_stat: See above for details.

        Returns:
            HeterogeneousMixtureDistribution: Estimated mixture distribution.

        """
        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i]) for i in range(num_components)]

        if self.fixed_weights is not None:
            w = np.asarray(self.fixed_weights)
        elif self.pseudo_count is not None and self.suff_stat is None:
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

        return HeterogeneousMixtureDistribution(components, w, name=self.name)


class HeterogeneousMixtureDataEncoder(DataSequenceEncoder):
    """Encoder that groups observations by compatible component encoder."""

    def __init__(self, encoders: list[DataSequenceEncoder]) -> None:
        """Create an encoder for vectorized heterogeneous-mixture scoring.

        Data must have the observation type accepted by every heterogeneous-mixture component.

        Args:
            encoders (List[DataSequenceEncoder]): Encoders for each heterogeneous-mixture component.

        Attributes:
            encoder_dict (Dict[DataSequenceEncoder, List[int]]): Dictionary of distinct DataSequenceEncoder objects
                found in encoders list. Value of encoder_dict is a list of ids for the components that are encoded by
                the corresponding encoder key.

        """
        encoder_dict: dict[str, DataSequenceEncoder] = dict()
        idx_dict: dict[str, list[int]] = dict()

        for encoder_idx, encoder in enumerate(encoders):
            enc_str = str(encoder)
            if enc_str not in encoder_dict:
                encoder_dict[enc_str] = encoder
                idx_dict[enc_str] = []
            idx_dict[enc_str].append(encoder_idx)

        self.encoder_dict: dict[str, DataSequenceEncoder] = encoder_dict
        self.idx_dict: dict[str, list[int]] = idx_dict

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        s = "HeterogeneousMixtureDataEncoder(["
        item_list = list(self.idx_dict.items())
        for enc_str, comp_list in item_list[:-1]:
            s += enc_str + ",comps=" + str(comp_list) + ","

        s += item_list[-1][0] + ",comps=" + str(item_list[-1][1]) + "])"

        return s

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder is equivalent to this encoder.

        Returns true when the grouped component indices for each distinct data encoder are equal.

        Args:
            other (Object): Object to compare.

        Returns:

        """
        if not isinstance(other, HeterogeneousMixtureDataEncoder):
            return False
        else:
            for encoder, comp_list in self.encoder_dict.items():
                if other.idx_dict[encoder] != comp_list:
                    return False
            return True

    def seq_encode(self, x: Sequence[T]) -> tuple[list[ndarray], list[Any]]:
        """Encode a sequence of iid heterogeneous mixture observations.

        Note: The data type for every encoder in the keys of HeterogeneousMixtureDataEncoder attribute
        self.encoder_dict.keys() is T.

        The returned tuple contains:
            tag_list (List[ndarray[int]): Heterogeneous mixture component ids for encoded sequences in enc_data list.
            enc_data (List[S1,...,Sm]): A list of 'm' encoded sequences of type Sm, corresponding to component ids
                in tag_list.

        Args:
            x (Sequence[T]): A Sequence of iid observations drawn from a heterogeneous mixture distribution.

        Returns:
            Tuple[List[ndarray[int], List[S1,...,Sm]] as defined above.

        """
        enc_data = []
        tag_list = []

        for enc_str, encoder_idx in self.idx_dict.items():
            tag_list.append(np.asarray(encoder_idx, dtype=int))
            enc_data.append(self.encoder_dict[enc_str].seq_encode(x))

        return tag_list, enc_data
