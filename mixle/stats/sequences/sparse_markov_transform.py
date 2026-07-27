"""Sparse Markov hidden-association models over integer word-count bags.

Data type:  Tuple[List[Tuple[int, float]], List[Tuple[int, float]]].

The SparseMarkovAssociation model describes two bags of words S_1 ={w_{1,1},...,w_{1,n}} and
S_2 ={w_{2,1},...,w_{2,m}} over W possible words. It is generative when supplied a proper joint length law and
otherwise is a fixed-length likelihood factor. The model assumes a hidden set of assignments
A_2 = {a_{2,1},...,a_{2,m}} where a_{2,j} takes on values in {1,2,...,m}. The observed likelihood function is
computed from P(S_1, S_2) = P(S_2 | S_1) P(S_1), where

    (1) log(P(S_2|S_1)) = sum_{i=1}^{m} log(P(w_{2,i}|w_{1,1},...,w_{1,n})
                        = sum_{i=1}^{m} log( (1/m)*sum_{j=1}^{n} (1-alpha)*P(w_{2,i} | w_{1,j}) + alpha/W).
    (2) log(P(S_1)) = sum_{j=1}^{n} log( (1-alpha)*P(w_{1,j} + alpha/W ).

Both bag terms include their multinomial coefficients so the returned observations are normalized count-bag
probabilities rather than probabilities of one discarded token ordering.

"""

import itertools
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse, lil_matrix

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DensitySemantics,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.sequences._keyed_accumulator import InitTransKeyedAccumulator
from mixle.stats.sequences.markov_transform import (
    _canonicalize_bag,
    _exact_nonnegative_count,
    _finite_nonnegative,
    _is_nonstring_sequence,
    _multinomial_log_coefficient,
    _positive_integer,
    _unit_interval,
    _validate_simplex_vector,
    _validate_weight_vector,
)
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.optsutil import count_by_value

T = tuple[list[tuple[int, float]], list[tuple[int, float]]]
SS1 = TypeVar("SS1")


def _validate_conditional_matrix(values, num_vals):
    """Return an owned immutable ``num_vals``-by-``num_vals`` row-simplex matrix."""
    try:
        matrix = csr_matrix(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("cond_prob_mat must be a numeric matrix.") from exc
    if matrix.shape != (num_vals, num_vals):
        raise ValueError("cond_prob_mat must have shape (%d, %d)." % (num_vals, num_vals))
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0.0):
        raise ValueError("cond_prob_mat must contain finite nonnegative probabilities.")
    matrix.sum_duplicates()
    row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    if not np.allclose(row_sums, 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("every cond_prob_mat row must sum to one.")
    matrix.eliminate_zeros()
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def _canonicalize_observation(observation, num_vals, *, label="observation"):
    """Validate one source/output bag pair while preserving impossible events for scoring."""
    if not _is_nonstring_sequence(observation) or len(observation) != 2:
        raise TypeError("%s must contain exactly two count bags." % label)
    return tuple(
        _canonicalize_bag(bag, num_vals, label="%s bag %d" % (label, index + 1))
        for index, bag in enumerate(observation)
    )


def _validate_encoded_observations(encoded, num_vals):
    """Validate the public four-item ragged encoding and canonicalize every row."""
    if not _is_nonstring_sequence(encoded) or len(encoded) != 4:
        raise TypeError("encoded sparse Markov-association data must be a four-item tuple.")
    try:
        raw_entries = list(encoded[0])
    except TypeError as exc:
        raise TypeError("encoded sparse Markov-association observations must be a sequence.") from exc
    entries = []
    for index, entry in enumerate(raw_entries):
        if not _is_nonstring_sequence(entry) or len(entry) != 4:
            raise TypeError("encoded observation %d must contain four label/count arrays." % index)
        bags = []
        for bag_index in range(2):
            values = np.asarray(entry[2 * bag_index])
            counts = np.asarray(entry[2 * bag_index + 1])
            if values.ndim != 1 or counts.ndim != 1 or len(values) != len(counts):
                raise ValueError(
                    "encoded observation %d bag %d label/count arrays must be one-dimensional and equally sized."
                    % (index, bag_index + 1)
                )
            bags.append(
                _canonicalize_bag(
                    list(zip(values.tolist(), counts.tolist())),
                    num_vals,
                    label="encoded observation %d bag %d" % (index, bag_index + 1),
                )
            )
        entries.append(tuple(bags))
    pairs = np.asarray(encoded[2])
    if pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise ValueError("encoded sparse Markov-association pair table must have shape (n, 2).")
    if pairs.size:
        if not np.issubdtype(pairs.dtype, np.integer):
            raise TypeError("encoded sparse Markov-association pair labels must be integers.")
        if np.any(pairs < 0) or np.any(pairs >= num_vals):
            raise ValueError("encoded sparse Markov-association pair labels are outside the declared support.")
    return entries


def _validate_statistic_value(value, num_vals, *, label):
    """Validate and copy one sparse Markov-association statistic tuple."""
    if not _is_nonstring_sequence(value) or len(value) != 3:
        raise TypeError("%s must be a three-item sufficient-statistic tuple." % label)
    init_count = np.asarray(value[0], dtype=np.float64)
    if init_count.shape != (num_vals,):
        raise ValueError("%s initial counts must have shape (%d,)." % (label, num_vals))
    if not np.all(np.isfinite(init_count)) or np.any(init_count < 0.0):
        raise ValueError("%s initial counts must be finite and nonnegative." % label)
    if value[1] is None:
        trans_count = None
    else:
        if not issparse(value[1]):
            try:
                trans_count = csr_matrix(value[1], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError("%s transition counts must be a numeric matrix." % label) from exc
        else:
            trans_count = csr_matrix(value[1], dtype=np.float64, copy=True)
        if trans_count.shape != (num_vals, num_vals):
            raise ValueError("%s transition counts have an incompatible shape." % label)
        trans_count.sum_duplicates()
        if not np.all(np.isfinite(trans_count.data)) or np.any(trans_count.data < 0.0):
            raise ValueError("%s transition counts must be finite and nonnegative." % label)
    return init_count.copy(), trans_count, value[2]


def _sparse_contribution(rows, columns, values, num_vals):
    """Build a sparse update that aggregates repeated coordinates."""
    if values.size == 0:
        return csr_matrix((num_vals, num_vals), dtype=np.float64)
    row_index = np.repeat(np.asarray(rows, dtype=np.int64), len(columns))
    column_index = np.tile(np.asarray(columns, dtype=np.int64), len(rows))
    return coo_matrix(
        (np.asarray(values).reshape(-1), (row_index, column_index)),
        shape=(num_vals, num_vals),
    ).tocsr()


class SparseMarkovAssociationDistribution(SequenceEncodableProbabilityDistribution):
    """Distribution for a sparse count set ``S2`` generated from a count set ``S1``."""

    def __init__(
        self,
        init_prob_vec: Sequence[float] | np.ndarray,
        cond_prob_mat: csr_matrix,
        alpha: float = 0.0,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        low_memory: bool = False,
    ) -> None:
        """Create a sparse Markov-association distribution.

        Args:
            init_prob_vec (Union[Sequence[float], np.ndarray]): Probabilities for the first set of words S1.
            cond_prob_mat (csr_matrix): Sparse matrix defining the probabilities for mapping words in S1 to S2. Dim is
                (|S2| by |S1|).
            alpha (float): Regularization parameter (should be between 0 and 1).
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for length of words. Must be
                compatible with Tuple[int, int].
            low_memory (bool): If True, uses low_memory function calls.

        Attributes:
            init_prob_vec (np.ndarray): Probabilities for the first set of words S1.
            cond_prob_mat (csr_matrix): Sparse matrix defining the probabilities for mapping words in S1 to S2. Dim is
                (|S2| by |S1|).
            alpha (float): Regularization parameter (should be between 0 and 1).
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for length of words. Must be
                compatible with Tuple[int, int]
            low_memory (bool): If True, uses low_memory function calls.

        """
        self.init_prob_vec = _validate_simplex_vector(init_prob_vec, label="init_prob_vec")
        self.num_vals = len(self.init_prob_vec)
        self.cond_prob_mat = _validate_conditional_matrix(cond_prob_mat, self.num_vals)
        if len_dist is not None and not isinstance(len_dist, SequenceEncodableProbabilityDistribution):
            raise TypeError("len_dist must be a sequence-encodable probability distribution or None.")
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.alpha = _unit_interval(alpha, label="alpha")
        self.low_memory = bool(low_memory)
        source_prob = (1.0 - self.alpha) * self.init_prob_vec + self.alpha / self.num_vals
        source_prob.setflags(write=False)
        self._source_prob_vec = source_prob

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = ",".join(map(str, self.init_prob_vec))
        temp = self.cond_prob_mat.nonzero()
        tt = np.asarray(self.cond_prob_mat[temp[0], temp[1]]).flatten()
        s20 = ",".join(map(str, tt))
        s21 = ",".join(map(str, temp[0]))
        s22 = ",".join(map(str, temp[1]))
        s2 = "([%s], ([%s],[%s]))" % (s20, s21, s22)
        s3 = str(self.alpha)
        s4 = str(self.len_dist)
        return "SparseMarkovAssociationDistribution([%s], %s, alpha=%s, len_dist=%s)" % (s1, s2, s3, s4)

    def density(self, x: tuple[list[tuple[int, float]], list[tuple[int, float]]]) -> float:
        """Density of the sparse Markov association model at observation x.

        See log_density() for details.

        Args:
            x: Observation tuple (S1, S2), each a list of (value, count) pairs.

        Returns:
            Density at observation x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: tuple[list[tuple[int, float]], list[tuple[int, float]]]) -> float:
        """Log-density of the sparse Markov association model at observation x.

        Computes log(P(S2 | S1)) (see module docstring, eq. (1)) plus the log-density of the total counts
        [n1, n2] under len_dist.

        Args:
            x: Observation tuple (S1, S2), each a list of (value, count) pairs.

        Returns:
            Log-density at observation x.

        """
        bags = _canonicalize_observation(x, self.num_vals)
        rv = self._log_density_bags(bags)
        nx, ny = (int(counts.sum()) for _, counts in bags)
        if not supports(self.len_dist, Neutral):
            rv += self.len_dist.log_density((nx, ny))
        return float(rv)

    def _log_density_bags(self, bags) -> float:
        """Score two canonical bags without the optional length-law term."""
        (vx, cx), (vy, cy) = bags
        nx = int(cx.sum())
        ny = int(cy.sum())
        with np.errstate(divide="ignore"):
            ll1 = float(np.dot(np.log(self._source_prob_vec[vx]), cx))
            if ny and not nx:
                ll2 = -np.inf
            elif ny:
                conditional = self.cond_prob_mat[vx, :][:, vy].toarray().T
                output_prob = np.dot(
                    conditional * (1.0 - self.alpha) + self.alpha / self.num_vals,
                    cx / nx,
                )
                ll2 = float(np.dot(np.log(output_prob), cy))
            else:
                ll2 = 0.0
        return float(ll1 + ll2 + _multinomial_log_coefficient(cx) + _multinomial_log_coefficient(cy))

    def seq_log_density(self, x) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x: Encoded sequence (from SparseMarkovAssociationDataEncoder.seq_encode).

        Returns:
            Numpy array of log-densities, one per encoded observation.

        """
        entries = _validate_encoded_observations(x, self.num_vals)
        rv = np.asarray([self._log_density_bags(bags) for bags in entries], dtype=np.float64)

        if not supports(self.len_dist, Neutral):
            lln = self.len_dist.seq_log_density(x[1])
            rv += lln

        return rv

    def compute_capabilities(self):
        """Engine readiness for the dense scoring tail (numpy + torch).

        The large word-by-word transition matrix is sliced/gathered host-side with SciPy sparse ops,
        but the per-pair smoothing, log, and segment reductions run on the active engine (see
        ``backend_seq_log_density``), so the model composes on numpy and torch.
        """
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        ready = ("numpy", "torch")
        if not supports(self.len_dist, Neutral):
            ready = intersect_engine_ready((self.len_dist,))
            if "numpy" not in ready:
                ready = ("numpy",)
        return DistributionCapabilities(engine_ready=ready, kernel_status="generic_object")

    def backend_seq_log_density(self, x, engine) -> Any:
        """Validate and score the sparse encoding once, then lift the result to ``engine``."""
        return engine.asarray(self.seq_log_density(x))

    def density_semantics(self):
        """Classify the neutral-length form as a conditional fixed-length factor."""
        if supports(self.len_dist, Neutral):
            return DensitySemantics.LIKELIHOOD_FACTOR
        return self.len_dist.density_semantics()

    def sampler(self, seed: int | None = None) -> "SparseMarkovAssociationSampler":
        """Create a sampler for this sparse Markov association distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            SparseMarkovAssociationSampler: Sampler bound to this distribution.

        """
        if supports(self.len_dist, Neutral):
            raise TypeError("SparseMarkovAssociationDistribution requires a generative length law for sampling.")
        return SparseMarkovAssociationSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SparseMarkovAssociationEstimator":
        """Create an estimator initialized from this sparse Markov association distribution.

        Args:
            pseudo_count (Optional[float]): Prior mass for the initial and conditional simplexes.

        Returns:
            SparseMarkovAssociationEstimator: Estimator configured with matching size and sparsity settings.

        """
        return SparseMarkovAssociationEstimator(
            num_vals=self.num_vals,
            alpha=self.alpha,
            len_estimator=self.len_dist.estimator(pseudo_count=pseudo_count),
            suff_stat=(self.init_prob_vec, self.cond_prob_mat, None) if pseudo_count is not None else None,
            pseudo_count=pseudo_count,
            low_memory=self.low_memory,
        )

    def dist_to_encoder(self) -> "SparseMarkovAssociationDataEncoder":
        """Return a data encoder for sparse Markov association observations."""
        return SparseMarkovAssociationDataEncoder(
            len_encoder=self.len_dist.dist_to_encoder(),
            low_memory=self.low_memory,
            num_vals=self.num_vals,
        )


class SparseMarkovAssociationSampler(DistributionSampler):
    """Sampler for a sparse Markov-association distribution."""

    def __init__(self, dist: SparseMarkovAssociationDistribution, seed: int | None = None) -> None:
        """Create a sparse Markov-association sampler.

        Args:
            dist (SparseMarkovAssociationDistribution): Distribution to sample from. Its len_dist must support
                sampling the total counts [n1, n2].
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (SparseMarkovAssociationDistribution): Distribution to sample from.
            rng (RandomState): RandomState with seed set if passed in args.
            size_sampler (DistributionSampler): Sampler for the total counts [n1, n2].

        """
        if not isinstance(dist, SparseMarkovAssociationDistribution):
            raise TypeError("dist must be a SparseMarkovAssociationDistribution.")
        if supports(dist.len_dist, Neutral):
            raise TypeError("SparseMarkovAssociationSampler requires a generative length law.")
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.size_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: int | None = None, *, batched: bool = True) -> T | Sequence[T]:
        """Draw 'size' iid observations from the sparse Markov association model.

        Each observation is a tuple (S1, S2) of lists of (value, count) pairs. If size is None a single
        observation is returned, else a list of 'size' observations is returned.

        Args:
            size (Optional[int]): Number of observations to draw. Treated as a single draw if None.

        Returns:
            A single observation tuple, or a list of observation tuples when size is not None.

        """
        if size is None:
            slens = self.size_sampler.sample()
            if not _is_nonstring_sequence(slens) or len(slens) != 2:
                raise ValueError("length distribution must draw exactly two nonnegative integer counts.")
            slens = tuple(
                _exact_nonnegative_count(value, label="sampled length %d" % (index + 1))
                for index, value in enumerate(slens)
            )
            if slens[1] > 0 and slens[0] == 0:
                raise ValueError("cannot sample output tokens when the sampled parent length is zero.")
            rng = np.random.RandomState(self.rng.randint(0, maxrandint))

            v1 = list(
                rng.choice(
                    self.dist.num_vals,
                    p=self.dist._source_prob_vec,
                    replace=True,
                    size=slens[0],
                )
            )
            v2 = []

            z1 = list(rng.choice(len(v1), replace=True, size=slens[1])) if slens[1] else []
            nw = self.dist.num_vals

            for zz1 in z1:
                p = (1.0 - self.dist.alpha) * self.dist.cond_prob_mat[
                    v1[zz1], :
                ].toarray().flatten() + self.dist.alpha / nw
                v2.append(rng.choice(nw, p=p))

            return list(count_by_value(v1).items()), list(count_by_value(v2).items())

        size = _exact_nonnegative_count(size, label="sample size")
        return [self.sample() for _ in range(size)]


class SparseMarkovAssociationAccumulator(InitTransKeyedAccumulator, SequenceEncodableStatisticAccumulator):
    """Accumulator for sparse Markov-association sufficient statistics."""

    def __init__(
        self,
        num_vals: int,
        size_acc: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: tuple[str | None, str | None] = (None, None),
        low_memory: bool = True,
    ) -> None:
        """Create an accumulator for sparse Markov-association sufficient statistics.

        Args:
            num_vals (int): Number of possible values W.
            size_acc (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the total counts.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial and transition statistics.
            low_memory (bool): If True, use low_memory function calls.

        Attributes:
            init_count (np.ndarray): Weighted counts for the initial probability vector.
            trans_count (Optional[Union[lil_matrix, csr_matrix]]): Weighted (W by W) transition counts.
            size_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the total counts.
            num_vals (int): Number of possible values W.
            init_key (Optional[str]): Key for the initial-count statistics.
            trans_key (Optional[str]): Key for the transition-count statistics.
            low_memory (bool): If True, use low_memory function calls.

        """
        self.num_vals = _positive_integer(num_vals, label="num_vals")
        self.init_count = np.zeros(self.num_vals)
        self.trans_count: lil_matrix | csr_matrix | None = None
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.low_memory = bool(low_memory)
        # Data log-likelihood accumulated as a byproduct of the E-step (the per-observation
        # log_density), only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing. Both the flat (non-low-memory) and per-observation branches report it.
        self._track_ll = False
        self._seq_ll = 0.0

        self._init_rng = False
        self._size_rng = None

    def _add_transition(self, contribution):
        """Add one owned sparse transition contribution."""
        if contribution is None:
            return
        if self.trans_count is None:
            self.trans_count = contribution.copy()
        else:
            self.trans_count += contribution

    def _update_canonical(self, bags, weight, estimate):
        """Apply one validated E-step update and return the two bag lengths."""
        if not isinstance(estimate, SparseMarkovAssociationDistribution):
            raise TypeError("estimate must be a SparseMarkovAssociationDistribution.")
        if estimate.num_vals != self.num_vals:
            raise ValueError("estimate support does not match accumulator support.")
        (vx, cx), (vy, cy) = bags
        nx = int(cx.sum())
        ny = int(cy.sum())
        if ny and not nx:
            raise ValueError("cannot accumulate output evidence for an empty parent bag.")

        contribution = None
        if weight > 0.0 and ny:
            conditional = estimate.cond_prob_mat[vx, :][:, vy].toarray()
            weighted = conditional * cx[:, None]
            conditional_mass = weighted.sum(axis=0)
            denominator = (1.0 - estimate.alpha) * conditional_mass + (estimate.alpha / self.num_vals) * nx
            if np.any(denominator <= 0.0):
                raise ValueError("observation has zero probability under the transition estimate.")
            responsibility = weighted * (cy * (1.0 - estimate.alpha) / denominator)[None, :] * weight
            contribution = _sparse_contribution(vx, vy, responsibility, self.num_vals)

        if len(vx):
            source_responsibility = np.divide(
                (1.0 - estimate.alpha) * estimate.init_prob_vec[vx],
                estimate._source_prob_vec[vx],
                out=np.zeros(len(vx), dtype=np.float64),
                where=estimate._source_prob_vec[vx] > 0.0,
            )
            np.add.at(self.init_count, vx, cx * source_responsibility * weight)
        self._add_transition(contribution)
        return nx, ny

    def _initialize_canonical(self, bags, weight):
        """Apply one validated uniform-responsibility initialization."""
        (vx, cx), (vy, cy) = bags
        nx = int(cx.sum())
        ny = int(cy.sum())
        if ny and not nx:
            raise ValueError("cannot initialize output evidence for an empty parent bag.")
        contribution = None
        if weight > 0.0 and ny:
            responsibility = np.outer(cx / nx, cy * weight)
            contribution = _sparse_contribution(vx, vy, responsibility, self.num_vals)
        np.add.at(self.init_count, vx, cx * weight)
        self._add_transition(contribution)
        return nx, ny

    def update(self, x: T, weight: float, estimate: SparseMarkovAssociationDistribution) -> None:
        """Update sufficient statistics with a single weighted observation.

        Args:
            x: Observation tuple (S1, S2), each a list of (value, count) pairs.
            weight (float): Weight of the observation.
            estimate (SparseMarkovAssociationDistribution): Previous estimate used to assign responsibility.

        Returns:
            None.

        """
        weight = _finite_nonnegative(weight, label="weight")
        bags = _canonicalize_observation(x, self.num_vals)
        lengths = self._update_canonical(bags, weight, estimate)
        self.size_accumulator.update(lengths, weight, estimate.len_dist)

    def initialize_rng(self, rng: np.random.RandomState) -> None:
        """Seed the internal RandomState for the size accumulator from rng (idempotent).

        Args:
            rng (RandomState): Source of the seed.

        Returns:
            None.

        """
        if not self._init_rng:
            self._size_rng = np.random.RandomState(seed=rng.randint(2**31))
            self._init_rng = True

    def initialize(self, x: T, weight: float, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics with a single weighted observation (no previous estimate).

        Args:
            x: Observation tuple (S1, S2), each a list of (value, count) pairs.
            weight (float): Weight of the observation.
            rng (RandomState): Used to seed the size accumulator initialization.

        Returns:
            None.

        """
        weight = _finite_nonnegative(weight, label="weight")
        bags = _canonicalize_observation(x, self.num_vals)
        lengths = self._initialize_canonical(bags, weight)
        if not self._init_rng:
            self.initialize_rng(rng)
        self.size_accumulator.initialize(lengths, weight, self._size_rng)

    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics with a sequence of weighted encoded observations.

        Args:
            x: Encoded sequence (from SparseMarkovAssociationDataEncoder.seq_encode).
            weights (np.ndarray): Weights, one per encoded observation.
            rng (RandomState): Used to seed the size accumulator initialization.

        Returns:
            None.

        """
        entries = _validate_encoded_observations(x, self.num_vals)
        weights = _validate_weight_vector(weights, len(entries))
        for bags, weight in zip(entries, weights):
            self._initialize_canonical(bags, float(weight))
        if not self._init_rng:
            self.initialize_rng(rng)
        self.size_accumulator.seq_initialize(x[1], weights, self._size_rng)

    def seq_update(self, x, weights: np.ndarray, estimate: SparseMarkovAssociationDistribution) -> None:
        """Update sufficient statistics with a sequence of weighted encoded observations.

        Args:
            x: Encoded sequence (from SparseMarkovAssociationDataEncoder.seq_encode).
            weights (np.ndarray): Weights, one per encoded observation.
            estimate (SparseMarkovAssociationDistribution): Previous estimate used to assign responsibility.

        Returns:
            None.

        """
        entries = _validate_encoded_observations(x, self.num_vals)
        weights = _validate_weight_vector(weights, len(entries))
        track = self._track_ll
        obs_ll = np.zeros(len(entries), dtype=np.float64) if track else None
        for index, (bags, weight) in enumerate(zip(entries, weights)):
            self._update_canonical(bags, float(weight), estimate)
            if track:
                observation = tuple(
                    [(int(value), int(count)) for value, count in zip(values, counts)] for values, counts in bags
                )
                obs_ll[index] = estimate.log_density(observation)
        self.size_accumulator.seq_update(x[1], weights, estimate.len_dist)
        if track:
            self._seq_ll += float(np.dot(weights, obs_ll))

    def combine(
        self, suff_stat: tuple[np.ndarray, lil_matrix | csr_matrix | None, SS1]
    ) -> "SparseMarkovAssociationAccumulator":
        """Merge the sufficient statistics of another accumulator into this one.

        Args:
            suff_stat: Tuple (init_count, trans_count, size_value) from another accumulator's value().

        Returns:
            This SparseMarkovAssociationAccumulator object.

        """
        init_count, trans_count, size_acc = _validate_statistic_value(
            suff_stat,
            self.num_vals,
            label="combined statistics",
        )

        self.size_accumulator.combine(size_acc)
        self.init_count += init_count
        # trans_count is lazily created on the first update()/seq_update() call (its sparse matrix type
        # depends on which one), so a freshly-made accumulator populated only via combine() -- the
        # reduce pattern parallel/Spark/engine drivers use -- has self.trans_count still None here,
        # unlike init_count (eagerly a zero array). Take the incoming matrix directly on a first combine
        # (copied, so this accumulator doesn't alias and later mutate the source's); a None incoming
        # side (the other accumulator was also never updated) contributes nothing.
        if trans_count is not None:
            if self.trans_count is None:
                self.trans_count = trans_count.copy()
            else:
                self.trans_count += trans_count

        return self

    def value(self) -> tuple[np.ndarray, lil_matrix | csr_matrix | None, Any]:
        """Returns the sufficient statistic tuple (init_count, trans_count, size_value)."""
        return (
            self.init_count.copy(),
            None if self.trans_count is None else self.trans_count.copy(),
            self.size_accumulator.value(),
        )

    def from_value(
        self, x: tuple[np.ndarray, lil_matrix | csr_matrix | None, SS1]
    ) -> "SparseMarkovAssociationAccumulator":
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple (init_count, trans_count, size_value).

        Returns:
            This SparseMarkovAssociationAccumulator object.

        """
        init_count, trans_count, size_acc = _validate_statistic_value(
            x,
            self.num_vals,
            label="restored statistics",
        )

        self.init_count = init_count
        self.trans_count = trans_count
        self.size_accumulator.from_value(size_acc)

        return self

    # key_merge / key_replace: provided by InitTransKeyedAccumulator (shared two-key plumbing).
    # The size_accumulator is a NullAccumulator (never None) here, so the mixin's
    # ``is not None`` guard delegates to it identically to the prior inline implementation.

    def acc_to_encoder(self) -> "SparseMarkovAssociationDataEncoder":
        """Return a data encoder built from the size accumulator."""
        return SparseMarkovAssociationDataEncoder(
            len_encoder=self.size_accumulator.acc_to_encoder(),
            low_memory=self.low_memory,
            num_vals=self.num_vals,
        )


class SparseMarkovAssociationAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for sparse Markov association accumulators."""

    def __init__(
        self,
        num_vals: int,
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        low_memory: bool = True,
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        """Create a factory for sparse Markov association accumulators.

        Args:
            num_vals (int): Number of possible values W.
            len_factory (Optional[StatisticAccumulatorFactory]): Factory for the total-count accumulator.
            low_memory (bool): If True, use low_memory function calls.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial and transition statistics.

        Attributes:
            num_vals (int): Number of possible values W.
            len_factory (StatisticAccumulatorFactory): Factory for the total-count accumulator.
            low_memory (bool): If True, use low_memory function calls.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial and transition statistics.

        """
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.low_memory = bool(low_memory)
        self.keys = keys
        self.num_vals = _positive_integer(num_vals, label="num_vals")

    def make(self) -> "SparseMarkovAssociationAccumulator":
        """Return a new sparse Markov association accumulator."""
        return SparseMarkovAssociationAccumulator(
            self.num_vals, size_acc=self.len_factory.make(), keys=self.keys, low_memory=self.low_memory
        )


class SparseMarkovAssociationEstimator(ParameterEstimator):
    """Estimate sparse Markov association distributions from sufficient statistics."""

    def __init__(
        self,
        num_vals: int = MISSING,
        alpha: float = 0.0,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        suff_stat: Any | None = None,
        pseudo_count: float | None = None,
        low_memory: bool = True,
        keys: tuple[str | None, str | None] = (None, None),
        num_values: int = MISSING,
    ) -> None:
        """Create an estimator for a sparse Markov-association distribution from aggregated sufficient statistics.

        Args:
            num_vals (int): Number of values in S1.
            alpha (float): Regularization parameter (should be between 0 and 1).
            len_estimator (Optional[ParameterEstimator]): Estimator for observation lengths.
            suff_stat (Optional[Any]): Optional initial/transition prior center.
            pseudo_count (Optional[float]): Total prior mass added to each estimated simplex.
            low_memory (bool): If True, use low_memory options.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial distribution and state transition stats.

        Attributes:
            num_vals (int): Number of values in S1.
            alpha (float): Regularization parameter (should be between 0 and 1).
            len_estimator (ParameterEstimator): Estimator for observation lengths.
            suff_stat (Optional[Any]): Optional initial/transition prior center.
            pseudo_count (Optional[float]): Total prior mass added to each estimated simplex.
            low_memory (bool): If True, use low_memory options.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial distribution and state transition stats.

        """
        self.keys = keys
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.num_vals = _positive_integer(
            coalesce_alias("num_vals", num_vals, "num_values", num_values, default=MISSING),
            label="num_vals",
        )
        self.pseudo_count = None if pseudo_count is None else _finite_nonnegative(pseudo_count, label="pseudo_count")
        self.suff_stat = (
            None if suff_stat is None else _validate_statistic_value(suff_stat, self.num_vals, label="prior statistics")
        )
        self.alpha = _unit_interval(alpha, label="alpha")
        self.low_memory = bool(low_memory)

    def accumulator_factory(self) -> "SparseMarkovAssociationAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        return SparseMarkovAssociationAccumulatorFactory(
            self.num_vals, self.len_estimator.accumulator_factory(), self.low_memory, self.keys
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, lil_matrix | csr_matrix | None, SS1]
    ) -> "SparseMarkovAssociationDistribution":
        """Estimate a sparse Markov association distribution from aggregated sufficient statistics.

        Arg suff_stat is a Tuple of length 3 containing:
            suff_stat[0] (np.ndarray): Weighted counts for the initial states P(S1).
            suff_stat[1] (Optional[Union[lil_matrix, csr_matrix]]): Counts for transitions used to estimate P(S2|S1).
            suff_stat[2] (SS1): Sufficient statistics from the accumulator of the size/len distribution.

        Args:
            nobs (Optional[float]): Weighted number of observations.
            suff_stat: See above for details.

        Returns:
            SparseMarkovAssociationDistribution.

        """
        if nobs is not None:
            nobs = _finite_nonnegative(nobs, label="nobs")
        init_count, trans_count, size_stats = _validate_statistic_value(
            suff_stat,
            self.num_vals,
            label="estimated statistics",
        )
        len_dist = self.len_estimator.estimate(nobs, size_stats)

        dense_trans = (
            np.zeros((self.num_vals, self.num_vals), dtype=np.float64) if trans_count is None else trans_count.toarray()
        )
        if self.pseudo_count is not None and self.pseudo_count > 0.0:
            if self.suff_stat is None:
                init_prior = np.full(self.num_vals, 1.0 / self.num_vals)
                trans_prior = np.full((self.num_vals, self.num_vals), 1.0 / self.num_vals)
            else:
                init_prior = self.suff_stat[0]
                init_prior_total = float(init_prior.sum())
                if init_prior_total <= 0.0:
                    raise ValueError("prior initial statistics must have positive total mass.")
                init_prior = init_prior / init_prior_total
                trans_prior = (
                    np.zeros((self.num_vals, self.num_vals), dtype=np.float64)
                    if self.suff_stat[1] is None
                    else self.suff_stat[1].toarray()
                )
                prior_row_sums = trans_prior.sum(axis=1, keepdims=True)
                empty_prior_rows = prior_row_sums[:, 0] == 0.0
                trans_prior[empty_prior_rows, :] = 1.0 / self.num_vals
                prior_row_sums[empty_prior_rows, :] = 1.0
                trans_prior /= prior_row_sums
            init_count += self.pseudo_count * init_prior
            dense_trans += self.pseudo_count * trans_prior

        init_total = float(init_count.sum())
        if init_total <= 0.0:
            init_prob = np.full(self.num_vals, 1.0 / self.num_vals)
        else:
            init_prob = init_count / init_total

        row_sums = dense_trans.sum(axis=1, keepdims=True)
        empty_rows = row_sums[:, 0] == 0.0
        dense_trans[empty_rows, :] = 1.0 / self.num_vals
        row_sums[empty_rows, :] = 1.0
        dense_trans /= row_sums
        trans_prob = csr_matrix(dense_trans)

        return SparseMarkovAssociationDistribution(init_prob, trans_prob, self.alpha, len_dist, self.low_memory)


class SparseMarkovAssociationDataEncoder(DataSequenceEncoder):
    """Encode sparse Markov association observations for vectorized scoring."""

    def __init__(self, len_encoder: DataSequenceEncoder, low_memory: bool, num_vals: int | None = None) -> None:
        """Create an encoder for sparse Markov association observations.

        Args:
            len_encoder (DataSequenceEncoder): Encoder for the total counts [n1, n2].
            low_memory (bool): If True, produce the compact encoding (no flattened pair-index arrays).
            num_vals (Optional[int]): Declared alphabet size used for label validation.

        Attributes:
            len_encoder (DataSequenceEncoder): Encoder for the total counts [n1, n2].
            low_memory (bool): If True, produce the compact encoding.
            num_vals (Optional[int]): Declared alphabet size, or None for an unbound compatibility encoder.

        """
        self.len_encoder = len_encoder
        self.low_memory = bool(low_memory)
        self.num_vals = None if num_vals is None else _positive_integer(num_vals, label="num_vals")

    def __eq__(self, other: object) -> bool:
        """Encoders are interchangeable iff other is a SparseMarkovAssociationDataEncoder with equal members.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is an equivalent SparseMarkovAssociationDataEncoder instance.

        """
        if isinstance(other, SparseMarkovAssociationDataEncoder):
            return (
                other.len_encoder == self.len_encoder
                and self.low_memory == other.low_memory
                and self.num_vals == other.num_vals
            )
        else:
            return False

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return (
            "SparseMarkovAssociationDataEncoder(len_encoder="
            + str(self.len_encoder)
            + ",low_memory="
            + str(self.low_memory)
            + ",num_vals="
            + str(self.num_vals)
            + ")"
        )

    def row_count(self, x: Any) -> int:
        """Return the number of top-level observations in this ragged encoding."""
        if not _is_nonstring_sequence(x) or len(x) != 4:
            raise TypeError("encoded sparse Markov-association data must be a four-item tuple.")
        try:
            return len(x[0])
        except TypeError as exc:
            raise TypeError("encoded sparse Markov-association observations must be a sequence.") from exc

    def seq_encode(self, x: Sequence[tuple[list[tuple[int, float]], list[tuple[int, float]]]]):
        """Encode a sequence of observations for vectorized calls.

        Args:
            x: Sequence of observation tuples (S1, S2), each a list of (value, count) pairs.

        Returns:
            Tuple (rv, nn, vv, qq) where rv holds per-observation (values, counts) arrays, nn is the encoded
            length data, vv is the array of distinct (u, v) pairs, and qq holds flattened pair-index arrays for
            the vectorized path (None when low_memory is True).

        """
        if self.low_memory:
            rv = []
            nn = []
            vset = set()

            for k, observation in enumerate(x):
                (vx, cx), (vy, cy) = _canonicalize_observation(
                    observation,
                    self.num_vals,
                    label="observation %d" % k,
                )

                vset.update(itertools.product(vx, vy))
                rv.append((vx, cx, vy, cy))
                nn.append((int(cx.sum()), int(cy.sum())))

            nn = self.len_encoder.seq_encode(nn)

            vv = np.zeros((len(vset), 2), dtype=int)
            for i, vvv in enumerate(vset):
                vv[i, :] = vvv[:]

            qq = None

        else:
            rv = []
            nn = []
            vmap = dict()

            obsidx = []
            pairidx = []
            seqidx = []
            cxvec = []
            cyvec = []

            fcyvec = []
            fcxvec = []
            fvxvec = []
            fsqxvec = []
            fsqyvec = []

            ridx = -1
            for k, observation in enumerate(x):
                (vx, cx), (vy, cy) = _canonicalize_observation(
                    observation,
                    self.num_vals,
                    label="observation %d" % k,
                )
                nx = int(np.sum(cx))

                fcyvec.extend(cy)
                fcxvec.extend(cx)
                fvxvec.extend(vx)
                fsqxvec.extend([k] * len(vx))
                fsqyvec.extend([k] * len(vy))

                for i, vvy in enumerate(vy):
                    ridx += 1
                    for j, vvx in enumerate(vx):
                        if (vvx, vvy) not in vmap:
                            vmap[(vvx, vvy)] = len(vmap)
                        widx = vmap[(vvx, vvy)]
                        obsidx.append(k)
                        seqidx.append(ridx)
                        pairidx.append(widx)
                        cxvec.append(cx[j] / nx)
                        cyvec.append(cy[i])

                rv.append((vx, cx, vy, cy))
                nn.append((nx, int(cy.sum())))

            nn = self.len_encoder.seq_encode(nn)

            vv = np.zeros((len(vmap), 2), dtype=int)
            for vvv, i in vmap.items():
                vv[i, :] = vvv[:]

            obsidx = np.asarray(obsidx, dtype=int)
            seqidx = np.asarray(seqidx, dtype=int)
            cxvec = np.asarray(cxvec, dtype=float)
            cyvec = np.asarray(cyvec, dtype=float)
            pairidx = np.asarray(pairidx, dtype=int)

            fcxvec = np.asarray(fcxvec, dtype=float)
            fcyvec = np.asarray(fcyvec, dtype=float)
            fvxvec = np.asarray(fvxvec, dtype=int)
            fsqxvec = np.asarray(fsqxvec, dtype=int)
            fsqyvec = np.asarray(fsqyvec, dtype=int)

            qq = (obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec)

        return rv, nn, vv, qq
