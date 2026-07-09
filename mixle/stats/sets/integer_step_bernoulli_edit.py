"""Integer step Bernoulli edit distributions over pairs of finite sets.

Data type: Tuple[Sequence[int], Sequence[int]]: An observation x = (x1, x2) is a pair of integer sets
(prev set, next set), each a subset of S = {0,1,2,...N-1}.

The density has the same form as the integer Bernoulli edit set distribution (see
mixle.stats.sets.integer_bernoulli_edit): each integer k independently transitions in or out of the set with
probabilities p(k in x2 | k in x1), p(k in x2 | k not in x1), etc., and the previous set x1 follows an
init distribution,

    p(x1, x2) = P_init(x1) * prod_{k=0}^{N-1} p(k in/not-in x2 | k in/not-in x1).

The "step" variant differs only in estimation: after the per-element edit probabilities are computed, the
estimator fits a two-level step function to the addition probabilities p(present | missing) and the
removal probabilities p(missing | present), so that each element receives one of just two probability
levels (a high level for the top-ranked elements and a low level for the rest), chosen to maximize the
Bernoulli likelihood of the per-element estimates.

Every class here subclasses its non-step counterpart in mixle.stats.sets.integer_bernoulli_edit and
overrides only what genuinely differs: the estimator's step-fit, the constructor signatures (the step
distribution/estimator do not carry the non-step ``keys`` plumbing), and the class-name strings used in
``__str__`` and in the types returned by the distribution's factory methods.

"""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from mixle.engines.arithmetic import *
from mixle.stats.combinator.null_dist import (
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.sets.integer_bernoulli_edit import (
    IntegerBernoulliEditAccumulator,
    IntegerBernoulliEditAccumulatorFactory,
    IntegerBernoulliEditDataEncoder,
    IntegerBernoulliEditDistribution,
    IntegerBernoulliEditEnumerator,
    IntegerBernoulliEditEstimator,
    IntegerBernoulliEditSampler,
)
from mixle.utils.aliasing import MISSING, coalesce_alias

T = tuple[Sequence[int] | np.ndarray, Sequence[int] | np.ndarray]
E1 = TypeVar("E1")  ## encoded type for init
E = tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], E1 | None]
SS1 = TypeVar("SS1")  ## suff-stat of init_dist


class IntegerStepBernoulliEditDistribution(IntegerBernoulliEditDistribution):
    """Step Bernoulli edit set distribution: each integer independently transitions in/out between two sets.

    Identical in form to :class:`IntegerBernoulliEditDistribution`; only the estimator (a two-level step
    fit) differs. The step distribution does not carry the non-step ``keys`` plumbing.
    """

    def __init__(
        self,
        log_edit_pmat: Sequence[tuple[float, float]] | np.ndarray,
        init_dist: SequenceEncodableProbabilityDistribution | None = None,
        name: str | None = None,
    ) -> None:
        """Create a stepwise Bernoulli-edit distribution over integer sets.

        Args:
            log_edit_pmat (Union[Sequence[Tuple[float, float]], np.ndarray]): num_vals by 2 (or 4) matrix of
                log-probabilities. With 2 columns, column 0 is log p(present | missing) and column 1 is
                log p(present | present); the missing-state columns are filled in by complement. With 4 columns,
                the columns are log p(missing | missing), log p(missing | present), log p(present | missing),
                log p(present | present).
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the previous set x[0].
                Should be compatible with Sequence[int] observations (e.g. IntegerBernoulliSetDistribution).
            name (Optional[str]): Optional distribution name.

        """
        super().__init__(log_edit_pmat, init_dist=init_dist, name=name)

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = repr(list(map(list, self.orig_log_edit_pmat)))
        s2 = repr(self.init_dist)
        s3 = repr(self.name)
        return "IntegerStepBernoulliEditDistribution(%s, init_dist=%s, name=%s)" % (s1, s2, s3)

    def sampler(self, seed: int | None = None) -> "IntegerStepBernoulliEditSampler":
        """Create a sampler for this integer step Bernoulli edit distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            IntegerStepBernoulliEditSampler: Sampler bound to this distribution.

        """
        return IntegerStepBernoulliEditSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerStepBernoulliEditEstimator":
        """Create an IntegerStepBernoulliEditEstimator with matching num_vals.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics in estimation.

        Returns:
            IntegerStepBernoulliEditEstimator: Estimator configured with matching support size.

        """
        return IntegerStepBernoulliEditEstimator(self.num_vals, pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> "IntegerStepBernoulliEditDataEncoder":
        """Return a data encoder for integer step Bernoulli edit observations."""
        return IntegerStepBernoulliEditDataEncoder(init_encoder=self.init_dist.dist_to_encoder())

    def enumerator(self) -> "IntegerStepBernoulliEditEnumerator":
        """Returns IntegerStepBernoulliEditEnumerator iterating set-pairs in descending probability order."""
        return IntegerStepBernoulliEditEnumerator(self)


class IntegerStepBernoulliEditEnumerator(IntegerBernoulliEditEnumerator):
    """Enumerates finite previous/next integer-set pairs for the step edit-set distribution."""


class IntegerStepBernoulliEditSampler(IntegerBernoulliEditSampler):
    """Sampler for ``(previous set, next set)`` pairs from a stepwise integer Bernoulli-edit distribution.

    Identical to :class:`IntegerBernoulliEditSampler`; only the bound distribution type differs.
    """


class IntegerStepBernoulliEditAccumulator(IntegerBernoulliEditAccumulator):
    """Accumulator for removed, added, and kept counts from stepwise integer set pairs.

    Identical to :class:`IntegerBernoulliEditAccumulator`; only the encoder type returned by
    :meth:`acc_to_encoder` differs.
    """

    def acc_to_encoder(self) -> "IntegerStepBernoulliEditDataEncoder":
        """Return a data encoder built from the previous-set accumulator."""
        return IntegerStepBernoulliEditDataEncoder(init_encoder=self.init_acc.acc_to_encoder())


class IntegerStepBernoulliEditAccumulatorFactory(IntegerBernoulliEditAccumulatorFactory):
    """Factory for integer step Bernoulli edit accumulators."""

    def make(self) -> "IntegerStepBernoulliEditAccumulator":
        """Return a new integer step Bernoulli edit accumulator."""
        return IntegerStepBernoulliEditAccumulator(self.num_vals, init_acc=self.init_factory.make(), keys=self.keys)


class IntegerStepBernoulliEditEstimator(IntegerBernoulliEditEstimator):
    """Estimate integer step Bernoulli edit distributions with a two-level edit-probability fit."""

    def __init__(
        self,
        num_vals: int = MISSING,
        init_estimator: ParameterEstimator | None = NullEstimator(),
        min_prob: float = 1.0e-128,
        pseudo_count: float | None = None,
        suff_stat: np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
        num_values: int = MISSING,
    ) -> None:
        """Create an estimator for integer step Bernoulli edit set distributions.

        Args:
            num_vals (int): Number of integer values N in the set range.
            init_estimator (Optional[ParameterEstimator]): Estimator for the previous set x[0].
            min_prob (float): Minimum probability for an edit transition.
            pseudo_count (Optional[float]): Prior mass used to smooth edit probabilities during estimation.
            suff_stat (Optional[np.ndarray]): num_vals by 4 matrix of edit probabilities.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.

        Attributes:
            num_vals (int): Number of integer values N in the set range.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.
            pseudo_count (Optional[float]): Prior mass used to smooth edit probabilities during estimation.
            suff_stat (Optional[np.ndarray]): num_vals by 4 matrix of edit probabilities.
            name (Optional[str]): Optional name assigned to estimated distributions.
            min_prob (float): Minimum probability for an edit transition.
            init_est (ParameterEstimator): Estimator for the previous set x[0].

        """
        self.num_vals = coalesce_alias("num_vals", num_vals, "num_values", num_values, default=MISSING)
        if pseudo_count is not None and pseudo_count < 0.0:
            raise ValueError("IntegerStepBernoulliEditEstimator requires a non-negative pseudo_count.")
        if min_prob is not None and (min_prob < 0.0 or min_prob > 0.5):
            raise ValueError("IntegerStepBernoulliEditEstimator requires 0 <= min_prob <= 0.5.")
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_prob = min_prob
        self.init_est = init_estimator if init_estimator is not None else NullEstimator()

    def accumulator_factory(self) -> "IntegerStepBernoulliEditAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        init_factory = self.init_est.accumulator_factory()
        return IntegerStepBernoulliEditAccumulatorFactory(self.num_vals, init_factory, self.keys)

    def __clip_prob(self, value: float) -> float:
        if self.min_prob is None or self.min_prob <= 0.0:
            return float(value)
        return float(np.clip(value, self.min_prob, 1.0 - self.min_prob))

    def __effective_step_counts(
        self,
        count_mat: np.ndarray,
        tot_sum: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s1 = count_mat[:, 0] + count_mat[:, 2]
        s0 = tot_sum - s1

        rem_success = count_mat[:, 0].astype(np.float64).copy()
        rem_trials = s1.astype(np.float64).copy()
        add_success = count_mat[:, 1].astype(np.float64).copy()
        add_trials = s0.astype(np.float64).copy()

        if self.pseudo_count is not None and self.pseudo_count > 0.0:
            p = self.pseudo_count
            if self.suff_stat is not None:
                s = np.asarray(self.suff_stat, dtype=np.float64)
                rem_success += p * s[:, 1]
                rem_trials += p * (s[:, 1] + s[:, 3])
                add_success += p * s[:, 2]
                add_trials += p * (s[:, 0] + s[:, 2])
            else:
                rem_success += p / 4.0
                rem_trials += p / 2.0
                add_success += p / 4.0
                add_trials += p / 2.0

        return rem_success, rem_trials, add_success, add_trials

    def __get_pqk(self, successes: np.ndarray, trials: np.ndarray) -> np.ndarray:
        """Fit a two-level (p, q) step function to per-element binomial counts.

        Sorts the elements by empirical rate, then for each split rank k assigns probability p to
        the top-k elements and q to the rest (estimated by pooled successes over pooled trials on
        each side), and keeps the split maximizing the binomial log-likelihood. Elements with no
        trials are excluded from the split search and assigned the overall pooled rate (0.5 when
        no element has trials).

        Args:
            successes (np.ndarray): Per-element success counts.
            trials (np.ndarray): Per-element trial counts.

        Returns:
            Numpy array of per-element probabilities taking at most two distinct values, clipped to
            the estimator's probability floor when one is declared.

        """
        N = len(successes)
        obs = np.flatnonzero(trials > 0)
        M = len(obs)

        if M == 0:
            return np.full(N, 0.5)

        sidx = obs[np.argsort(-(successes[obs] / trials[obs]))]
        cs = np.cumsum(successes[sidx])
        ct = np.cumsum(trials[sidx])
        tot_s = cs[-1]
        tot_t = ct[-1]

        max_ll = -np.inf
        max_params = None
        for i in range(M):
            sh, th = cs[i], ct[i]
            p = self.__clip_prob(sh / th)
            v1 = (sh * np.log(p) if sh > 0 else 0.0) + ((th - sh) * np.log1p(-p) if th > sh else 0.0)
            if i + 1 < M:
                sl, tl = tot_s - sh, tot_t - th
                q = self.__clip_prob(sl / tl)
                v2 = (sl * np.log(q) if sl > 0 else 0.0) + ((tl - sl) * np.log1p(-q) if tl > sl else 0.0)
            else:
                q = 0.0
                v2 = 0.0
            ll = v1 + v2
            if ll > max_ll:
                max_params = (p, q, i)
                max_ll = ll

        p, q, k = max_params

        arr = np.full(N, self.__clip_prob(tot_s / tot_t))
        arr[sidx[: k + 1]] = p
        arr[sidx[k + 1 :]] = q
        return arr

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, float, SS1 | None]
    ) -> "IntegerStepBernoulliEditDistribution":
        """Estimate an IntegerStepBernoulliEditDistribution from aggregated sufficient statistics.

        Per-element edit probabilities are estimated as in the non-step edit estimator, then the
        addition and removal probabilities are each replaced by a two-level step-function fit.

        Args:
            nobs (Optional[float]): Unused (kept for protocol consistency).
            suff_stat (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            IntegerStepBernoulliEditDistribution object.

        """
        init_dist = self.init_est.estimate(None, suff_stat[2])
        count_mat, tot_sum, _ = suff_stat
        rem_success, rem_trials, add_success, add_trials = self.__effective_step_counts(count_mat, tot_sum)
        arr1 = self.__get_pqk(rem_success, rem_trials)
        arr2 = self.__get_pqk(add_success, add_trials)

        log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_pmat[:, 2] = np.log(arr2)
            log_pmat[:, 0] = np.log(1 - arr2)
            log_pmat[:, 1] = np.log(arr1)
            log_pmat[:, 3] = np.log(1 - arr1)

        return IntegerStepBernoulliEditDistribution(log_pmat, init_dist=init_dist, name=self.name)


class IntegerStepBernoulliEditDataEncoder(IntegerBernoulliEditDataEncoder):
    """Encode iid ``(previous set, next set)`` observations for vectorized scoring.

    Identical to :class:`IntegerBernoulliEditDataEncoder`; only the reported class name differs.
    """

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "IntegerStepBernoulliEditDataEncoder(init_encoder=" + str(self.init_encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent integer step Bernoulli-edit encoder."""
        if isinstance(other, IntegerStepBernoulliEditDataEncoder):
            return other.init_encoder == self.init_encoder
        else:
            return False
