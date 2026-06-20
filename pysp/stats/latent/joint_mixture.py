"""Create, estimate, and sample from a Joint mixture distribution.

Defines the JointMixtureDistribution, JointMixtureSampler, JointMixtureAccumulatorFactory, JointMixtureAccumulator,
JointMixtureEstimator, and the JointMixtureDataEncoder classes for use with pysparkplug.

Data type: Tuple[T0, T1].

Consider a random variable X = (X_1, X_2). A joint mixture with N components for X_1, and M components for X_2 is
given by

    P(X) = sum_{i=1}^{N} w_i * f_i(X_1) * sum_{j=1}^{M} tau_{ij}*g_j(X_2),

where w_i is the probability of sampling X_1 from distribution f_i() (data type T0), tau_{ij} is the probability of
sampling X_2 from g_j() (data type T1) given X_1 was sampled from f_i().


"""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.arithmetic import maxrandint
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)
from pysp.utils.enumeration import BufferedStream, ProductEnumerator, best_first_union

T0 = TypeVar("T0")
T1 = TypeVar("T1")
E0 = TypeVar("E0")
E1 = TypeVar("E1")
SS0 = TypeVar("SS0")
SS1 = TypeVar("SS1")


class JointMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """JointMixtureDistribution object defining a joint mixture over paired observations.

    Data type: Tuple[T0, T1], where T0 and T1 are the data types of the components for X1 and X2.

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
        """JointMixtureDistribution object for defining a joint mixture distribution.

        Note: Data type is Tuple[T0, T1] where all components1 entries and component2 entries are compatible with
        T0 and T1 respectively.

        Args:
            components1(Sequence[SequenceEncodableProbabilityDistribution]): Mixture components for mixture of X1.
            components2 (Sequence[SequenceEncodableProbabilityDistribution]): Mixture components for mixture X2.
            w1 (np.ndarray): Probability of drawing X1 from component i.
            w2 (np.ndarray): Probability of drawing X2 from component j.
            taus12 (np.ndarray): 2-d Numpy array with probabilities of drawing X2 from comp j given X1 was drawn from
                comp i. Rows are component X1 state.
            taus21 (np.ndarray): 2-d Numpy array with probabilities of drawing X1 from comp i given X2 was drawn from
                comp j. Rows are component X1 state.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        Attributes:
            components1(Sequence[SequenceEncodableProbabilityDistribution]): Mixture components for mixture of X1.
            components2 (Sequence[SequenceEncodableProbabilityDistribution]): Mixture components for mixture X2.
            w1 (np.ndarray): Probability of drawing X1 from component i.
            w2 (np.ndarray): Probability of drawing X2 from component j.
            num_components1 (int): Number of mixture components for X1.
            num_components2 (int): Number of mixture components for X2.
            taus12 (np.ndarray): 2-d Numpy array with probabilities of drawing X2 from comp j given X1 was drawn from
                comp i. Rows are component X1 state.
            taus21 (np.ndarray): 2-d Numpy array with probabilities of drawing X1 from comp i given X2 was drawn from
                comp j. Rows are component X1 state.
            log_w1 (np.ndarray): Log-probability of drawing X1 from component i.
            log_w2 (np.ndarray): Log-probability of drawing X2 from component j.
            log_taus12 (np.ndarray): 2-d Numpy array with log-probabilities of drawing X2 from comp j given X1 was
                drawn from comp i. Rows are component X1 state.
            log_taus21 (np.ndarray): 2-d Numpy array with log-probabilities of drawing X1 from comp i given X2 was
                drawn from comp j. Rows are component X1 state.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

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
        """Return string representation of JointMixtureDistribution object."""
        s1 = ",".join([str(u) for u in self.components1])
        s2 = ",".join([str(u) for u in self.components2])
        s3 = ",".join(map(str, self.w1))
        s4 = ",".join(map(str, self.w2))
        s5 = ",".join(map(str, self.taus12.flatten()))
        s6 = ",".join(map(str, self.taus21.flatten()))
        s7 = repr(self.name)

        return "JointMixtureDistribution([%s], [%s], [%s], [%s], [%s], [%s], name=%s)" % (s1, s2, s3, s4, s5, s6, s7)

    def compute_capabilities(self):
        from pysp.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = tuple(self.components1) + tuple(self.components2)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def compute_declaration(self):
        from pysp.stats.compute.declarations import (
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
        from pysp.stats.compute.backend import backend_seq_log_density

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
            from pysp.utils.fisher import JointMixtureFisherView

            return JointMixtureFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> "JointMixtureSampler":
        """Create a JointMixtureSampler object for sampling from this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator used in sampling.

        Returns:
            JointMixtureSampler object.

        """
        return JointMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "JointMixtureEstimator":
        """Create a JointMixtureEstimator object from the components of this distribution.

        Args:
            pseudo_count (Optional[float]): If passed, used to re-weight the state counts in estimation.

        Returns:
            JointMixtureEstimator object.

        """
        estimators1 = [comp1.estimator() for comp1 in self.components1]
        estimators2 = [comp2.estimator() for comp2 in self.components2]

        return JointMixtureEstimator(
            estimators1=estimators1, estimators2=estimators2, pseudo_count=pseudo_count, keys=self.keys, name=self.name
        )

    def dist_to_encoder(self) -> "DataSequenceEncoder":
        """Return a JointMixtureDataEncoder object for encoding sequences of iid observations."""
        encoder1 = self.components1[0].dist_to_encoder()
        encoder2 = self.components2[0].dist_to_encoder()
        return JointMixtureDataEncoder(encoder1=encoder1, encoder2=encoder2)

    def enumerator(self) -> "JointMixtureEnumerator":
        """Returns a JointMixtureEnumerator iterating (X1, X2) pairs in descending probability order."""
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
    """JointMixtureSampler object for sampling (X1, X2) pairs from a JointMixtureDistribution."""

    def __init__(self, dist: JointMixtureDistribution, seed: int | None = None) -> None:
        """JointMixtureSampler object.

        Args:
            dist (JointMixtureDistribution): JointMixtureDistribution instance to sample from.
            seed (Optional[int]): Seed for the random number generator used in sampling.

        Attributes:
            rng (RandomState): RandomState object with seed set if passed as arg.
            dist (JointMixtureDistribution): JointMixtureDistribution instance to sample from.
            comp_sampler1 (List[DistributionSampler]): Samplers for the X1 mixture components.
            comp_sampler2 (List[DistributionSampler]): Samplers for the X2 mixture components.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.comp_sampler1 = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in self.dist.components1]
        self.comp_sampler2 = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in self.dist.components2]

    def sample(self, size: int | None = None) -> tuple[Any, Any] | Sequence[tuple[Any, Any]]:
        """Draw one or 'size' iid (X1, X2) samples from the joint mixture.

        The X1 component state is drawn from w1, X1 is sampled from that component, the X2
        component state is drawn from taus12 given the X1 state, and X2 is sampled from the
        corresponding X2 component.

        Args:
            size (Optional[int]): Number of samples to draw. If None, a single (X1, X2) tuple is returned.

        Returns:
            A Tuple (X1, X2) if size is None, else a list of 'size' such tuples.

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
    """JointMixtureEstimatorAccumulator object for aggregating sufficient statistics of observed data."""

    def __init__(
        self,
        accumulators1: Sequence[SequenceEncodableStatisticAccumulator],
        accumulators2: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """JointMixtureEstimatorAccumulator object.

        Args:
            accumulators1 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture components
                of X1.
            accumulators2 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture components
                of X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        Attributes:
            accumulators1 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture components
                of X1.
            accumulators2 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture components
                of X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            num_components1 (int): Number of X1 mixture components.
            num_components2 (int): Number of X2 mixture components.
            comp_counts1 (np.ndarray): Weighted observation counts for states of mixture on X1.
            comp_counts2 (np.ndarray): Weighted observation counts for states of mixture on X2.
            joint_counts (np.ndarray): 2-d Numpy array for counts of state-given-state weights. Row indexed by states
                of X1, cols indexed by states of X2.
            name (Optional[str]): Set name to object.

            _rng_init (bool): Set to True once _rng_ members have been set.
            _idx1_rng (Optional[RandomState]): RandomState for generating states for X1 in initializer.
            _idx2_rng (Optional[RandomState]): RandomState for generating states for X2 in initializer.
            _acc1_rng (Optional[List[RandomState]]): List of RandomStates for initializing each accumulator for
                mixture components of X1.
            _acc2_rng (Optional[List[RandomState]]): List of RandomStates for initializing each accumulator for
                mixture components of X2.


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
        """Set member RandomState objects from rng for initialize()/seq_initialize() consistency.

        Args:
            rng (RandomState): Used to generate seeds for member RandomState objects.

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
            rng (RandomState): RandomState object used to seed member RandomState objects.

        Returns:
            None.

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
            rng (RandomState): RandomState object used to seed member RandomState objects.

        Returns:
            None.

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
        from pysp.stats.compute.backend import backend_seq_log_density

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
    ) -> "JointMixtureEstimatorAccumulator":
        """Combine the sufficient statistics of suff_stat with this accumulator.

        Arg suff_stat is a Tuple of length 5 containing:
            suff_stat[0] (np.ndarray): Component counts for the X1 mixture.
            suff_stat[1] (np.ndarray): Component counts for the X2 mixture.
            suff_stat[2] (np.ndarray): Joint counts of (X1 state, X2 state) pairs.
            suff_stat[3] (Tuple[SS0, ...]): Sufficient statistics for the X1 components.
            suff_stat[4] (Tuple[SS1, ...]): Sufficient statistics for the X2 components.

        Args:
            suff_stat: See above for details.

        Returns:
            JointMixtureEstimatorAccumulator object.

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
        """Returns sufficient statistics as a Tuple (see combine() for entry details)."""
        return (
            self.comp_counts1,
            self.comp_counts2,
            self.joint_counts,
            tuple([u.value() for u in self.accumulators1]),
            tuple([u.value() for u in self.accumulators2]),
        )

    def from_value(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[E0, ...], tuple[E1, ...]]
    ) -> "JointMixtureEstimatorAccumulator":
        """Set the sufficient statistics of this accumulator to x.

        Args:
            x: Sufficient statistic Tuple (see combine() for entry details).

        Returns:
            JointMixtureEstimatorAccumulator object.

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
        """Merge sufficient statistics of object instance with matching keys in stats_dict.

        Merges the count statistics if the weight key is set, and the X1/X2 component
        sufficient statistics if the corresponding accumulator keys are set.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

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
        """Replace sufficient statistics of object instance with those of matching keys in stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

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

    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Return a JointMixtureDataEncoder object for encoding sequences of iid observations."""
        encoder1 = self.accumulators1[0].acc_to_encoder()
        encoder2 = self.accumulators2[0].acc_to_encoder()
        return JointMixtureDataEncoder(encoder1=encoder1, encoder2=encoder2)


class JointMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """JointMixtureEstimatorAccumulatorFactory object for creating JointMixtureEstimatorAccumulator objects."""

    def __init__(
        self,
        factories1: Sequence[StatisticAccumulatorFactory],
        factories2: Sequence[StatisticAccumulatorFactory],
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """JointMixtureEstimatorAccumulatorFactory object for creating JointMixtureEstimatorAccumulator objects.

        Args:
            factories1 (Sequence[StatisticAccumulatorFactory]): List of mixture component factories for X1.
            factories2 (Sequence[StatisticAccumulatorFactory]): List of mixture component factories for X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        Attributes:
            factories1 (Sequence[StatisticAccumulatorFactory]): List of mixture component factories for X1.
            factories2 (Sequence[StatisticAccumulatorFactory]): List of mixture component factories for X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        """
        self.factories1 = factories1
        self.factories2 = factories2
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def make(self) -> "JointMixtureEstimatorAccumulator":
        """Returns a JointMixtureEstimatorAccumulator object from attribute variables."""
        f1 = [self.factories1[i].make() for i in range(len(self.factories1))]
        f2 = [self.factories2[i].make() for i in range(len(self.factories2))]
        return JointMixtureEstimatorAccumulator(f1, f2, keys=self.keys, name=self.name)


class JointMixtureEstimator(ParameterEstimator):
    """JointMixtureEstimator object for estimating a JointMixtureDistribution from sufficient statistics."""

    def __init__(
        self,
        estimators1: Sequence[ParameterEstimator],
        estimators2: Sequence[ParameterEstimator],
        suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[E0, ...], tuple[E1, ...]] | None = None,
        pseudo_count: tuple[float, float, float] | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """JointMixtureEstimator object for estimating joint mixture distribution from aggregated sufficient stats.

        Args:
            estimators1 (Sequence[ParameterEstimator]): Estimators for mixture component of X1.
            estimators2 (Sequence[ParameterEstimator]): Estimators for mixture component of X2.
            suff_stat:
            pseudo_count (Optional[Tuple[float, float, float]]): Used to re-weight the state counts in estimation.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        Attributes:
            estimators1 (Sequence[ParameterEstimator]): Estimators for mixture component of X1.
            estimators2 (Sequence[ParameterEstimator]): Estimators for mixture component of X2.
            suff_stat:
            pseudo_count (Optional[Tuple[float, float, float]]): Used to re-weight the state counts in estimation.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        """
        self.num_components1 = len(estimators1)
        self.num_components2 = len(estimators2)
        self.estimators1 = estimators1
        self.estimators2 = estimators2
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def accumulator_factory(self) -> "JointMixtureEstimatorAccumulatorFactory":
        """Returns a JointMixtureEstimatorAccumulatorFactory object from attribute variables."""
        est_factories1 = [u.accumulator_factory() for u in self.estimators1]
        est_factories2 = [u.accumulator_factory() for u in self.estimators2]
        return JointMixtureEstimatorAccumulatorFactory(est_factories1, est_factories2, self.keys)

    def estimate(
        self, nobs, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[E0, ...], tuple[E1, ...]]
    ) -> "JointMixtureDistribution":
        """Estimate a Joint mixture distribution from aggregated sufficient statistics.

        suff_stat is a Tuple containing:
            suff_stat[0] (np.ndarray): Component counts for outer mixture.
            suff_stat[1] (np.ndarray): Component counts for the inner mixture.
            suff_stat[2] (np.ndarray): Component counts for the comps of inner mix given an outer mix component.
            suff_stat[3] (Tuple[E0,...]): Suff-stats for outer comps
            suff_stat[4] (Tuple[E1,...]): Suff-stats for the inner comps.

        Args:
            nobs (Optional[float]): Weighted number of observations used in aggregation of suff_stats.
            suff_stat: See above for details.

        Returns:
            JointMixtureDistribution object.

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
            w1 = counts1 / counts1.sum()
            w2 = counts2 / counts2.sum()
            taus = joint_counts

            taus12_sum = np.sum(taus, axis=1, keepdims=True)
            taus12_sum[taus12_sum == 0] = 1.0
            taus12 = taus / taus12_sum

            taus21_sum = np.sum(taus, axis=0, keepdims=True)
            taus21_sum[taus21_sum == 0] = 1.0
            taus21 = taus / taus21_sum

        return JointMixtureDistribution(components1, components2, w1, w2, taus12, taus21, name=self.name)


class JointMixtureDataEncoder(DataSequenceEncoder):
    """JointMixtureDataEncoder object for encoding sequences of iid joint mixture observations."""

    def __init__(self, encoder1: DataSequenceEncoder, encoder2: DataSequenceEncoder) -> None:
        """JointMixtureDataEncoder object for encoding sequences of iid joint mixture observations.

        Args:
            encoder1 (DataSequenceEncoder): DataSequenceEncoder for the components of X1.
            encoder2 (DataSequenceEncoder): DataSequenceEncoder for the components of X2.

        Attributes:
            encoder1 (DataSequenceEncoder): DataSequenceEncoder for the components of X1.
            encoder2 (DataSequenceEncoder): DataSequenceEncoder for the components of X2.

        """
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def __str__(self) -> str:
        """Return string representation of JointMixtureDataEncoder object."""
        return "JointMixtureDataEncoder(encoder0=" + str(self.encoder1) + ",encoder1=" + str(self.encoder2) + ")"

    def __eq__(self, other: object) -> bool:
        """Check if other is an equivalent JointMixtureDataEncoder (both component encoders must match).

        Args:
            other (object): Object to compare to object instance.

        Returns:
            True if other is equivalent.

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


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
JointMixtureAccumulator = JointMixtureEstimatorAccumulator
JointMixtureAccumulatorFactory = JointMixtureEstimatorAccumulatorFactory


def _register_joint_mixture_engine_kernel():
    """Register the engine-resident joint-mixture kernel (idempotent; called at import)."""
    from pysp.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

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
