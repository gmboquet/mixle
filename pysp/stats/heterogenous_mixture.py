"""Create, estimate, and sample from a heterogeneous mixture distribution.

Defines the HeterogeneousMixtureDistribution, HeterogeneousMixtureSampler, HeterogeneousMixtureAccumulatorFactory,
HeterogeneousMixtureAccumulator, HeterogeneousMixtureEstimator, and the HeterogeneousMixtureDataEncoder classes for use
with pysparkplug.

HeterogeneousMixtureDistribution with data type T, is defined by the density of the form,

p_mat(Y) = sum_{k=1}^{K} p_mat(Y|Z=k)*p_mat(Z=k),

where p_mat(Z=k) is a mixture weight, and p_mat(Y|Z=k) is defined as a the k^{th} component distribution. Note that
the component distributions p_mat(Y|Z=k) must only be compatible in data type T.

Example: A heterogeneous mixture with weights [0.5, 0.5] and component distribution Exponential(beta) and Gamma(k,theta),
has form
    p_mat(x_mat) = 0.5*P_0(x; beta) + 0.5*P_1(x; k, theta), for x > 0.0,
where
    P_0(x;beta) is an exponential density and P_1(x; k, theta) is a Gamma density.

"""
import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from math import exp
from pysp.arithmetic import maxrandint
import pysp.utils.vector as vec
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, StatisticAccumulatorFactory, \
    SequenceEncodableStatisticAccumulator, DataSequenceEncoder, DistributionSampler, ParameterEstimator

from typing import Optional, Union, Tuple, Any, TypeVar, List, Dict, Sequence

T = TypeVar('T')


class HeterogeneousMixtureDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self,
                 components: Sequence[SequenceEncodableProbabilityDistribution],
                 w: Union[List[float], np.ndarray],
                 name: Optional[str] = None) -> None:
        """HeterogeneousMixtureDistribution object defined by component distributions and weights.

        The args components (Sequence[SequenceEncodableProbabilityDistribution]) define the component distributions
        of the heterogenous mixture distribution as well as the data type T. The data type of the
        HeterogeneousMixtureDistribution object is taken to be the data type (T) of the component distributions
        (all must components must be compatible with data type T).

        Args:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Set component distributions.
                Must all be compatible with type T.
            w (ndarray[float]): Mixture weights, must sum to 1.0.
            name (Optional[str]): Assign string name to HeterogeneousMixtureDistribution object.

        Attributes:
            components (List[SequenceEncodableProbabilityDistribution]): List of component distributions (data type T).
            w (ndarray[float]): Mixture weights assigned from args (w).
            name (Optional[str]): String name to HeterogeneousMixtureDistribution object.
            zw (ndarray[bool]): True if a weight is 0.0, else False.
            log_w (ndarray[float]): Log of weights (w). set to -np.inf, where zw is True.
            num_components (int): Number of components in HeterogeneousMixtureDistribution instance.

        """
        self.w = np.asarray(w, dtype=float)
        self.zw = (self.w == 0.0)
        self.log_w = np.log(w + self.zw)
        self.log_w[self.zw] = -np.inf

        self.components = components
        self.num_components = len(components)
        self.name = name

    def __str__(self) -> str:
        """Return string representation of HeterogeneousMixtureDistribution object instance."""
        s1 = ','.join([str(u) for u in self.components])
        s2 = repr(list(self.w))
        s3 = repr(self.name)

        return 'HeterogeneousMixtureDistribution(components=[%s], w=%s, name=%s)' % (s1, s2, s3)

    def density(self, x: T) -> float:
        """Evaluate density of heterogeneous mMixture distribution at observation x.

        See log_density() for details.

        Args:
            x: (T): Single observation from heterogeneous mixture distribution. T is data type of components.

        Returns:
            Density at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        """Evaluate log-density of heterogeneous mixture distribution at observation x.

        A K-component heterogeneous mixture has log-density,

            log(p_mat(x)) = log(sum_{z=k}^{K} p_mat(x|z=k)*p_mat(z=k)),

        where p_mat(x|z=k) is component-k log-density at x, and p_mat(z=k) = w[k]. A log-sum-exp is used to evaluate the
        sum inside the log of the right-hand side above. (See pysp.utils.vector.log_sum() for details).

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

    def seq_log_density(self, x: Tuple[List[np.ndarray], List[Any]]) -> np.ndarray:
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

    def seq_component_log_density(self, x: Tuple[List[np.ndarray], List[Any]]) -> np.ndarray:
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

    def seq_posterior(self, x: Tuple[List[np.ndarray], List[Any]]) -> np.ndarray:
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

    def sampler(self, seed: Optional[int] = None) -> 'HeterogeneousMixtureSampler':
        """Create HeterogeneousMixtureSampler for sampling from HeterogeneousMixtureDistribution instance.

        Args:
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Returns:
            HeterogeneousMixtureSampler object.

        """
        return HeterogeneousMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'HeterogeneousMixtureEstimator':
        """Create HeterogeneousMixtureEstimator for estimating HeterogeneousMixtureDistribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics in estimation.

        Returns:
            HeterogeneousMixtureEstimator object.

        """
        if pseudo_count is not None:
            return HeterogeneousMixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count, name=self.name)
        else:
            return HeterogeneousMixtureEstimator([u.estimator() for u in self.components], name=self.name)

    def dist_to_encoder(self) -> 'HeterogeneousMixtureDataEncoder':
        """Returns a HeterogeneousMixtureDataEncoder object for encoding sequences of iid observations from
            HeterogeneousMixtureDistribution."""
        encoders = [comp.dist_to_encoder() for comp in self.components]

        return HeterogeneousMixtureDataEncoder(encoders=encoders)


class HeterogeneousMixtureSampler(DistributionSampler):

    def __init__(self, dist: HeterogeneousMixtureDistribution, seed: Optional[int] = None):
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

    def sample(self, size: Optional[int] = None) -> Union[Any, List[Any]]:
        """Draw iid samples from a heterogeneous mixture distribution.

        The data type drawn from 'comp_samplers' is type T, corresponding to the data type of the mixture components.

        If size is None, a single sample (of data type T) is drawn and returned. If size is not None, 'size'-iid
        heterogeneous mixture samples are drawn and returned as a List with data type List[T].

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Data type T or List[T].

        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.comp_samplers[comp_state].sample()
        else:
            return [self.comp_samplers[i].sample() for i in comp_state]


class HeterogeneousMixtureAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators: List[SequenceEncodableStatisticAccumulator],
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        """HeterogeneousMixtureAccumulator object used to aggregate the sufficient statistics of observed data.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Sequence of
                SequenceEncodableStatisticAccumulator objects for the components of the heterogeneous mixture.
            keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and heterogeneous mixture components.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Sequence of
                SequenceEncodableStatisticAccumulator objects for the components of the heterogeneous mixture.
            num_components (int): Total number of mixture components (length of accumulators).
            comp_counts (np.ndarray[float]): Numpy array of floats for accumulating component weights.
            weight_key (Optional[str]): Key for weights of mixture.
            comp_key (Optional[str]): Key for components of mixture.
            _init_rng (bool): False if rng for accumulators has not been set.
            _acc_rng (Optional[List[RandomState]]): List of RandomState obejcts for setting seed on accumulator
                initialization.
        """
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]

        ### Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: Optional[List[RandomState]] = None

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
        """Initialize RandomState objects for accumulators from rng.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _rng_acc member variable.

        Returns:
            None.

        """
        seeds = rng.randint(2 ** 31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._init_rng = True

    def initialize(self, x: T, weight: float, rng: np.random.RandomState) -> None:
        """Initialize HeterogeneousMixtureAccumulator object with weighted observation x.

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

    def seq_initialize(self, x: Tuple[List[np.ndarray], List[Any]], weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of HeterogeneousMixtureAccumulator object for sequence encoded observations x.

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
            ww[keep_idx, :] = rng.dirichlet(alpha=np.ones(self.num_components) / (self.num_components ** 2),
                                            size=keep_len)
        ww *= np.reshape(weights, (sz, 1))

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                self.accumulators[i].seq_initialize(enc_data[tag], ww[:, i], self._acc_rng[i])
                self.comp_counts[i] += np.sum(ww[:, i])

    def seq_update(self,
                   x: Tuple[List[np.ndarray], List[Any]],
                   weights: np.ndarray,
                   estimate: 'HeterogeneousMixtureDistribution') -> None:
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
            estimate (MixtureDistribution): HeterogeneousMixtureDistribution object for previous estimate from EM.

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

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        np.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                w_loc = ll_mat[:, i]
                self.comp_counts[i] += w_loc.sum()
                self.accumulators[i].seq_update(enc_data[tag], w_loc, estimate.components[i])

    def combine(self, suff_stat: Tuple[np.ndarray, Tuple[Any, ...]]) -> 'HeterogeneousMixtureAccumulator':
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

    def value(self) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        """Returns sufficient statistics of MixtureAccumulator instance.

        The sufficient statistics value returned (suff_stat) is a Tuple of length two containing,
            suff_stat[0] (ndarray[float]): Aggregated component counts,
            suff_stat[1] (Tuple[T1,...,Tk]): Tuple of K sufficient statistics for the heterogeneous mixture components.

        Note: The components k^{th} heterogeneous mixture component is assumed to have sufficient statistics of type Tk.

        Returns:
            Tuple[np.ndarray[float], Tuple[T1,...,Tk]] as described above.

        """
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: Tuple[np.ndarray, Tuple[Any, ...]]) -> 'HeterogeneousMixtureAccumulator':
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

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with suff stats containing matching keys.

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

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set sufficient statistics of object instance to suff_stats with matching keys.

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

    def acc_to_encoder(self) -> 'HeterogeneousMixtureDataEncoder':
        """Returns a HeterogeneousMixtureDataEncoder object for encoding sequences of iid observations from
            HeterogeneousMixtureDistribution."""
        encoders = [comp.acc_to_encoder() for comp in self.accumulators]

        return HeterogeneousMixtureDataEncoder(encoders=encoders)


class HeterogeneousMixtureAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self,
                 factories: List[StatisticAccumulatorFactory],
                 dim: int,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        """HeterogeneousMixtureAccumulatorFactory object for creating HeterogeneousMixtureAccumulator objects.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Sequence of StatisticAccumulatorFactory objects for the
                mixture components.
            dim (int): Number of mixture components.
            keys (Tuple[Optional[str], Optional[str]]): Assign keys for weights and component aggregations.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): Sequence of StatisticAccumulatorFactory obejcts for the
                mixture components.
            dim (int): Number of mixture components. Must equal length of factories.
            keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components.

        """
        self.factories = factories
        self.dim = dim
        self.keys = keys

    def make(self) -> 'HeterogeneousMixtureAccumulator':
        """"Return HeterogeneousMixtureAccumulator object with SequenceEncodableStatisticAccumulator objects for the
            components and keys passed."""
        return HeterogeneousMixtureAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class HeterogeneousMixtureEstimator(ParameterEstimator):

    def __init__(self, estimators: List[ParameterEstimator], fixed_weights: Optional[np.ndarray] = None,
                 suff_stat: Optional[np.ndarray] = None, pseudo_count: Optional[float] = None,
                 name: Optional[str] = None, keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        """HeterogeneousMixtureEstimator object used to estimate HeterogeneousMixtureDistribution from sufficient
            statistics aggregated from HeterogeneousMixtureAccumulator.

        Args:
            estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimator objects for the heterogeneous
                mixture components.
            fixed_weights (Optional[np.ndarray]): Set fixed values for heterogeneous mixture weights.
            suff_stat (Optional[np.ndarray]): Numpy array of floats length equal to length of estimators.
            pseudo_count (Optional[float]): Used to re-weight the member variable sufficient statistics in estimation.
            name (Optional[str]): Set a name to the HeterogeneousMixtureEstimator object.
            keys (Tuple[Optional[str], Optional[str]]): Set keys for the weights and component distributions.

        Attributes:
            estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimator objects for the heterogeneous
                mixture components.
            fixed_weights (Optional[np.ndarray]): Treat heterogeneous mixture weights as fixed values. Must sum to 1.0.
            suff_stat (Optional[np.ndarray]): Weights of the heterogeneous mixture. Must sum to 1.0.
            pseudo_count (Optional[float]): Used to re-weight the member variable sufficient statistics in estimation.
            name (Optional[str]): Name to the HeterogeneousMixtureEstimator object.
            keys (Tuple[Optional[str], Optional[str]]): Keys for the weights and component distributions.

        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_weights = fixed_weights

    def accumulator_factory(self) -> 'HeterogeneousMixtureAccumulatorFactory':
        """Returns HeterogeneousMixtureAccumulatorFactory object passing component StatisticAccumulatorFactory
            objects and keys."""
        est_factories = [u.accumulator_factory() for u in self.estimators]

        return HeterogeneousMixtureAccumulatorFactory(est_factories, self.num_components, self.keys)

    def estimate(self, nobs: Optional[float],
                 suff_stat: Tuple[np.ndarray, Tuple[Any, ...]]) -> 'HeterogeneousMixtureDistribution':
        """Estimate HeterogeneousMixtureDistribution from aggregated sufficient statistics.

        Args suff_stat is a Tuple length two containing:
            suff_stat[0] (np.ndarray): Sufficient statistic for the weights of the mixture components.
            suff_stat[1] (Tuple[T1,...,Tk]): Tuple of K sufficient statistics for the heterogeneous mixture components.

        suff_stat[1] is passed to estimate() function of each corresponding entry in member variable 'estimators'.

        If fixed_weights is not None, suff_stat[0] is not used and the weights of the HeterogeneousMixtureDistribution
            are set to fixed_weights.

        If pseudo_count is passed, arg suff_stat[0] is aggregated with re-weighted member variable suff_stat. If member
        variable suff_stat is None, then the arg suff_stat[0] is re-weighted with pseudo_count to estimate the weights.

        If pseudo_count is None, ar suff_stat[0] is used to estimate the weights.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator super class.
            suff_stat: See above for details.

        Returns:
            HeterogeneousMixtureDistribution object.

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

    def __init__(self, encoders: List[DataSequenceEncoder]) -> None:
        """HeterogeneousMixtureDataEncoder used for sequence encoding data for use with vectorized 'seq_' functions.

        Data type: Data must be type T, that matches the data type of each heterogeneous mixture component.

        Args:
            encoders (List[DataSequenceEncoder]): List of DataSequenceEncoder objects for each heterogeneous mixture
                component.

        Attributes:
            encoder_dict (Dict[DataSequenceEncoder, List[int]]): Dictionary of distinct DataSequenceEncoder objects
                found in encoders list. Value of encoder_dict is a list of ids for the components that are encoded by
                'encoder_dict key.

        """
        encoder_dict: Dict[str, DataSequenceEncoder] = dict()
        idx_dict: Dict[str, List[int]] = dict()

        for encoder_idx, encoder in enumerate(encoders):
            enc_str = str(encoder)
            if enc_str not in encoder_dict:
                encoder_dict[enc_str] = encoder
                idx_dict[enc_str] = []
            idx_dict[enc_str].append(encoder_idx)

        self.encoder_dict: Dict[str, DataSequenceEncoder] = encoder_dict
        self.idx_dict: Dict[str, List[int]] = idx_dict

    def __str__(self) -> str:
        """Return string representation of HeterogeneousMixtureDataEncoder instance."""
        s = 'HeterogeneousMixtureDataEncoder(['
        item_list = list(self.idx_dict.items())
        for enc_str, comp_list in item_list[:-1]:
            s += enc_str + ',comps=' + str(comp_list) + ','

        s += item_list[-1][0] + ',comps=' + str(item_list[-1][1]) + '])'

        return s

    def __eq__(self, other: object) -> bool:
        """Checks for if other object is equivalent to HeterogeneousMixtureDataEncoder instance.

        Returns true if component indices of distinct DataSequenceEncoder are equal. Else returns false.

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

    def seq_encode(self, x: Sequence[T]) -> Tuple[List[ndarray], List[Any]]:
        """Encode a sequence of iid heterogeneous mixture observations.

        Note: The data type for every encoder in the keys of HeterogeneousMixtureDataEncoder attribute
        self.encoder_dict.keys() is T.

        Returns a Tuple of length two containing:
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
