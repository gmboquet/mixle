"""Sparse Markov hidden-association models over integer word-count bags.

Data type:  Tuple[List[Tuple[int, float]], List[Tuple[int, float]]].

The SparseMarkovAssociation model is a generative model for two sets of words S_1 ={w_{1,1},...,w_{1,n}} and
S_2 ={w_{2,1},...,w_{2,m}} over W possible words. The model assumes a hidden set of assignments
A_2 = {a_{2,1},...,a_{2,m}} where a_{2,j} takes on values in {1,2,...,m}. The observed likelihood function is
computed from P(S_1, S_2) = P(S_2 | S_1) P(S_1), where

    (1) log(P(S_2|S_1)) = sum_{i=1}^{m} log(P(w_{2,i}|w_{1,1},...,w_{1,n})
                        = sum_{i=1}^{m} log( (1/m)*sum_{j=1}^{n} (1-alpha)*P(w_{2,i} | w_{1,j}) + alpha/W).
    (2) log(P(S_1)) = sum_{j=1}^{n} log( (1-alpha)*P(w_{1,j} + alpha/W ).

This model is great for problems where one set is given like translations.

"""

import itertools
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

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
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.sequences._keyed_accumulator import InitTransKeyedAccumulator
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.optsutil import count_by_value

T = tuple[list[tuple[int, float]], list[tuple[int, float]]]
SS1 = TypeVar("SS1")


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
        self.init_prob_vec = np.asarray(init_prob_vec, dtype=np.float64)
        self.cond_prob_mat = csr_matrix(cond_prob_mat, dtype=np.float64)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.num_vals = len(init_prob_vec)
        self.alpha = alpha
        self.low_memory = low_memory

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
        nw = self.num_vals
        a = self.alpha / nw
        b = 1 - self.alpha

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        nx = np.sum(cx)
        ny = np.sum(cy)

        temp = self.cond_prob_mat[vx[:, None], vy].toarray()
        ll2 = np.dot(np.log(np.dot((temp * b + a).T, cx / nx)), cy)
        ll1 = np.dot(np.log(self.init_prob_vec[vx] * b + a), cx)
        rv = ll1 + ll2
        rv += self.len_dist.log_density([nx, ny])

        return float(rv)

    def seq_log_density(self, x) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x: Encoded sequence (from SparseMarkovAssociationDataEncoder.seq_encode).

        Returns:
            Numpy array of log-densities, one per encoded observation.

        """
        nw = self.num_vals
        a = self.alpha / nw
        b = 1 - self.alpha

        xlen = len(x[0])

        if x[3] is not None:
            obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec = x[3]

            vv = x[2]

            p = np.asarray(self.cond_prob_mat[vv[:, 0], vv[:, 1]]).flatten()
            p = p * b + a
            sval = np.bincount(seqidx, weights=p[pairidx] * cxvec)
            np.log(sval, out=sval)
            sval *= fcyvec
            rv = np.bincount(fsqyvec, weights=sval, minlength=xlen)
            rv += np.bincount(fsqxvec, weights=np.log(self.init_prob_vec[fvxvec] * b + a) * fcxvec, minlength=xlen)

        else:
            rv = np.zeros(len(x[0]), dtype=np.float64)

            for i, entry in enumerate(x[0]):
                xx, cx, yy, cy = entry
                nx = np.sum(cx)

                temp = self.cond_prob_mat[xx[:, None], yy].toarray()
                ll2 = np.dot(np.log(np.dot((temp * b + a).T, cx / nx)), cy)
                ll1 = np.dot(np.log(self.init_prob_vec[xx] * b + a), cx)

                rv[i] = ll1 + ll2

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
        """Engine-routed sparse-association scoring.

        The conditional probabilities for the observed word pairs are gathered host-side from the
        SciPy sparse matrix; the smoothing ``p*b + a``, the logs, and the segment-sum reductions
        (initial-state and association terms) run on the active engine via ``index_add``. Falls back
        to the engine-lifted NumPy path for the low-memory encoding that lacks the flat pair index.
        """
        if x[3] is None:
            return engine.asarray(self.seq_log_density(x))

        from mixle.stats.compute.backend import backend_seq_log_density as _backend_sld

        nw = self.num_vals
        a = self.alpha / nw
        b = 1.0 - self.alpha
        xlen = len(x[0])
        (obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec) = x[3]
        vv = x[2]

        p_host = np.asarray(self.cond_prob_mat[vv[:, 0], vv[:, 1]]).flatten()
        p = engine.asarray(p_host) * engine.asarray(b) + engine.asarray(a)

        n_seq = int(np.asarray(fcyvec).shape[0])
        contrib = p[engine.asarray(np.asarray(pairidx, dtype=np.int64))] * engine.asarray(cxvec)
        sval = engine.index_add(engine.zeros(n_seq), engine.asarray(np.asarray(seqidx, dtype=np.int64)), contrib)
        sval = engine.log(sval) * engine.asarray(fcyvec)
        rv = engine.index_add(engine.zeros(xlen), engine.asarray(np.asarray(fsqyvec, dtype=np.int64)), sval)

        init_term = engine.log(
            engine.asarray(self.init_prob_vec[fvxvec]) * engine.asarray(b) + engine.asarray(a)
        ) * engine.asarray(fcxvec)
        rv = rv + engine.index_add(engine.zeros(xlen), engine.asarray(np.asarray(fsqxvec, dtype=np.int64)), init_term)

        if not supports(self.len_dist, Neutral):
            rv = rv + _backend_sld(self.len_dist, x[1], engine)
        return rv

    def sampler(self, seed: int | None = None) -> "SparseMarkovAssociationSampler":
        """Create a sampler for this sparse Markov association distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            SparseMarkovAssociationSampler: Sampler bound to this distribution.

        """
        return SparseMarkovAssociationSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SparseMarkovAssociationEstimator":
        """Create an estimator initialized from this sparse Markov association distribution.

        Args:
            pseudo_count (Optional[float]): Kept for protocol compatibility (unused).

        Returns:
            SparseMarkovAssociationEstimator: Estimator configured with matching size and sparsity settings.

        """
        return SparseMarkovAssociationEstimator(
            num_vals=self.num_vals,
            alpha=self.alpha,
            len_estimator=self.len_dist.estimator(),
            low_memory=self.low_memory,
        )

    def dist_to_encoder(self) -> "SparseMarkovAssociationDataEncoder":
        """Return a data encoder for sparse Markov association observations."""
        return SparseMarkovAssociationDataEncoder(
            len_encoder=self.len_dist.dist_to_encoder(), low_memory=self.low_memory
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
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.size_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: int | None = None) -> T | Sequence[T]:
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
            rng = np.random.RandomState(self.rng.randint(0, maxrandint))

            v1 = list(rng.choice(len(self.dist.init_prob_vec), p=self.dist.init_prob_vec, replace=True, size=slens[0]))
            v2 = []

            z1 = list(rng.choice(len(v1), replace=True, size=slens[1]))
            nw = self.dist.num_vals

            for zz1 in z1:
                if rng.rand() > self.dist.alpha:
                    p = self.dist.cond_prob_mat[v1[zz1], :].toarray().flatten()
                    v2.append(rng.choice(nw, p=p))
                else:
                    v2.append(rng.choice(nw))

            return list(count_by_value(v1).items()), list(count_by_value(v2).items())

        else:
            return [self.sample() for i in range(size)]


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
        self.init_count = np.zeros(num_vals)
        self.trans_count: lil_matrix | csr_matrix | None = None
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        self.num_vals = num_vals
        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.low_memory = low_memory
        # Data log-likelihood accumulated as a byproduct of the E-step (the per-observation
        # log_density), only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing. Both the flat (non-low-memory) and per-observation branches report it.
        self._track_ll = False
        self._seq_ll = 0.0

        self._init_rng = False
        self._size_rng = None

    def update(self, x: T, weight: float, estimate: SparseMarkovAssociationDistribution) -> None:
        """Update sufficient statistics with a single weighted observation.

        Args:
            x: Observation tuple (S1, S2), each a list of (value, count) pairs.
            weight (float): Weight of the observation.
            estimate (SparseMarkovAssociationDistribution): Previous estimate used to assign responsibility.

        Returns:
            None.

        """
        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = lil_matrix((num_vals, num_vals))

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        a = estimate.alpha / self.num_vals
        b = 1 - estimate.alpha

        temp = estimate.cond_prob_mat[vx[:, None], vy].toarray()

        loc_cprob = temp * cx[:, None]
        w = loc_cprob.sum(axis=0)
        loc_cprob *= (cy * b / (w * b + a * np.sum(cx))) * weight

        self.trans_count[vx[:, None], vy] += loc_cprob
        self.init_count[vx] += cx * weight

        self.size_accumulator.update((cx.sum(), cy.sum()), weight, estimate.len_dist)

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
        if not self._init_rng:
            self.initialize_rng(rng)

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = lil_matrix((num_vals, num_vals))

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        self.trans_count[vx[:, None], vy] += np.outer(cx / np.sum(cx), cy) * weight
        self.init_count[vx] += cx * weight

        self.size_accumulator.initialize((cx.sum(), cy.sum()), weight, self._size_rng)

    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics with a sequence of weighted encoded observations.

        Args:
            x: Encoded sequence (from SparseMarkovAssociationDataEncoder.seq_encode).
            weights (np.ndarray): Weights, one per encoded observation.
            rng (RandomState): Used to seed the size accumulator initialization.

        Returns:
            None.

        """
        if not self._init_rng:
            self.initialize_rng(rng)

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = csr_matrix((num_vals, num_vals))

        nw = self.num_vals

        if x[3] is not None:
            obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec = x[3]

            vv = x[2]

            # No estimate exists yet, so allocate transition mass uniformly
            # ((cx/sum(cx)) outer cy, as in the low-memory branch) instead of
            # reading the all-zero trans_count.
            pp = cxvec * cyvec * weights[obsidx]
            pp = np.bincount(pairidx, weights=pp, minlength=vv.shape[0])

            umat = csr_matrix((pp, (vv[:, 0], vv[:, 1])), shape=(nw, nw))
            self.trans_count += umat
            self.init_count += np.bincount(fvxvec, weights=fcxvec * weights[fsqxvec], minlength=nw)

        else:
            rows = []
            cols = []
            vals = []

            for i, (entry, weight) in enumerate(zip(x[0], weights)):
                vx, cx, vy, cy = entry
                loc_counts = np.outer(cx / np.sum(cx), cy) * weight

                rows.append(np.repeat(vx, len(vy)))
                cols.append(np.tile(vy, len(vx)))
                vals.append(loc_counts.ravel())
                self.init_count[vx] += cx * weight

            if vals:
                umat = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nw, nw))
                self.trans_count += umat

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
        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = csr_matrix((num_vals, num_vals))

        nw = self.num_vals
        a = estimate.alpha / nw
        b = 1 - estimate.alpha

        if x[3] is not None:
            obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec = x[3]

            vv = x[2]

            p = np.asarray(estimate.cond_prob_mat[vv[:, 0], vv[:, 1]]).flatten()
            pp = p[pairidx] * cxvec
            sval = np.bincount(seqidx, weights=pp)
            sval *= b
            sval += a
            if self._track_ll:
                # Per-emitted-word inner mass (== seq_log_density's b*w + a); aggregate
                # log(sval)*count by observation, then add the smoothed initial-state term, exactly
                # mirroring SparseMarkovAssociationDistribution.seq_log_density. Captured before the
                # responsibility normalization overwrites ``sval``.
                with np.errstate(divide="ignore"):
                    ll2_terms = np.log(sval) * fcyvec
                obs_ll = np.bincount(fsqyvec, weights=ll2_terms, minlength=len(x[0]))
                init_terms = np.log(estimate.init_prob_vec[fvxvec] * b + a) * fcxvec
                obs_ll += np.bincount(fsqxvec, weights=init_terms, minlength=len(x[0]))
                if not supports(estimate.len_dist, Neutral):
                    obs_ll += estimate.len_dist.seq_log_density(x[1])
                self._seq_ll += float(np.dot(np.asarray(weights, dtype=np.float64), obs_ll))
            np.divide(weights[fsqyvec] * b, sval, out=sval)
            sval *= fcyvec
            pp *= sval[seqidx]
            pp = np.bincount(pairidx, weights=pp)

            umat = csr_matrix((pp, (vv[:, 0], vv[:, 1])), shape=(nw, nw))
            self.trans_count += umat
            self.init_count += np.bincount(fvxvec, weights=fcxvec * weights[fsqxvec], minlength=nw)

        else:
            nzv = x[2]
            track = self._track_ll
            obs_ll = np.zeros(len(x[0]), dtype=np.float64) if track else None
            log_init = estimate.init_prob_vec if track else None
            rows = []
            cols = []
            vals = []

            for i, (entry, weight) in enumerate(zip(x[0], weights)):
                vx, cx, vy, cy = entry
                nx = np.sum(cx)

                temp = estimate.cond_prob_mat[vx[:, None], vy].toarray()

                loc_cprob = temp * cx[:, None]
                w = loc_cprob.sum(axis=0)
                if track:
                    # Per-observation log-density (== seq_log_density low-memory branch). The dense
                    # path normalizes counts by nx, so inner = (w*b + a*nx)/nx; reuse ``w`` before the
                    # responsibility scaling below.
                    with np.errstate(divide="ignore"):
                        ll2 = float(np.dot(np.log((w * b + a * nx) / nx), cy))
                        ll1 = float(np.dot(np.log(log_init[vx] * b + a), cx))
                    obs_ll[i] = ll1 + ll2
                loc_cprob *= (cy * b / (w * b + a * nx)) * weight

                rows.append(np.repeat(vx, len(vy)))
                cols.append(np.tile(vy, len(vx)))
                vals.append(loc_cprob.ravel())
                self.init_count[vx] += cx * weight

            if vals:
                umat = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nw, nw))
                self.trans_count += umat

            if track:
                if not supports(estimate.len_dist, Neutral):
                    obs_ll += estimate.len_dist.seq_log_density(x[1])
                self._seq_ll += float(np.dot(np.asarray(weights, dtype=np.float64), obs_ll))

        self.size_accumulator.seq_update(x[1], weights, estimate.len_dist)

    def combine(
        self, suff_stat: tuple[np.ndarray, lil_matrix | csr_matrix | None, SS1]
    ) -> "SparseMarkovAssociationAccumulator":
        """Merge the sufficient statistics of another accumulator into this one.

        Args:
            suff_stat: Tuple (init_count, trans_count, size_value) from another accumulator's value().

        Returns:
            This SparseMarkovAssociationAccumulator object.

        """
        init_count, trans_count, size_acc = suff_stat

        self.size_accumulator.combine(size_acc)
        self.init_count += init_count
        self.trans_count += trans_count

        return self

    def value(self) -> tuple[np.ndarray, lil_matrix | csr_matrix | None, Any]:
        """Returns the sufficient statistic tuple (init_count, trans_count, size_value)."""
        return self.init_count, self.trans_count, self.size_accumulator.value()

    def from_value(
        self, x: tuple[np.ndarray, lil_matrix | csr_matrix | None, SS1]
    ) -> "SparseMarkovAssociationAccumulator":
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple (init_count, trans_count, size_value).

        Returns:
            This SparseMarkovAssociationAccumulator object.

        """
        init_count, trans_count, size_acc = x

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
            len_encoder=self.size_accumulator.acc_to_encoder(), low_memory=self.low_memory
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
        self.low_memory = low_memory
        self.keys = keys
        self.num_vals = num_vals

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
            suff_stat (Optional[Any]): Kept for consistency with estimate function.
            pseudo_count (Optional[float]): Regularize sufficient statistics.
            low_memory (bool): If True, use low_memory options.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial distribution and state transition stats.

        Attributes:
            num_vals (int): Number of values in S1.
            alpha (float): Regularization parameter (should be between 0 and 1).
            len_estimator (ParameterEstimator): Estimator for observation lengths.
            suff_stat (Optional[Any]): Kept for consistency with estimate function.
            pseudo_count (Optional[float]): Regularize sufficient statistics.
            low_memory (bool): If True, use low_memory options.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial distribution and state transition stats.

        """
        self.keys = keys
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.num_vals = coalesce_alias("num_vals", num_vals, "num_values", num_values, default=MISSING)
        self.alpha = alpha
        self.low_memory = low_memory

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
        init_count, trans_count, size_stats = suff_stat
        len_dist = self.len_estimator.estimate(nobs, size_stats)

        trans_count = trans_count.tocsr()
        row_sum = trans_count.sum(axis=1)
        row_sum = csr_matrix(row_sum)
        row_sum.eliminate_zeros()
        row_sum.data = 1.0 / row_sum.data

        init_prob = init_count / np.sum(init_count)
        trans_prob = trans_count.multiply(row_sum)

        return SparseMarkovAssociationDistribution(init_prob, trans_prob, self.alpha, len_dist, self.low_memory)


class SparseMarkovAssociationDataEncoder(DataSequenceEncoder):
    """Encode sparse Markov association observations for vectorized scoring."""

    def __init__(self, len_encoder: DataSequenceEncoder, low_memory: bool) -> None:
        """Create an encoder for sparse Markov association observations.

        Args:
            len_encoder (DataSequenceEncoder): Encoder for the total counts [n1, n2].
            low_memory (bool): If True, produce the compact encoding (no flattened pair-index arrays).

        Attributes:
            len_encoder (DataSequenceEncoder): Encoder for the total counts [n1, n2].
            low_memory (bool): If True, produce the compact encoding.

        """
        self.len_encoder = len_encoder
        self.low_memory = low_memory

    def __eq__(self, other: object) -> bool:
        """Encoders are interchangeable iff other is a SparseMarkovAssociationDataEncoder with equal members.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is an equivalent SparseMarkovAssociationDataEncoder instance.

        """
        if isinstance(other, SparseMarkovAssociationDataEncoder):
            return other.len_encoder == self.len_encoder and self.low_memory == other.low_memory
        else:
            return False

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return (
            "SparseMarkovAssociationDataEncoder(len_encoder="
            + str(self.len_encoder)
            + ",low_memory="
            + str(self.low_memory)
            + ")"
        )

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

            for k, xx in enumerate(x):
                vx = np.asarray([u[0] for u in xx[0]], dtype=int)
                cx = np.asarray([u[1] for u in xx[0]], dtype=float)
                vy = np.asarray([u[0] for u in xx[1]], dtype=int)
                cy = np.asarray([u[1] for u in xx[1]], dtype=float)
                nx = np.sum(cx)

                vset.update(itertools.product(vx, vy))
                rv.append((vx, cx, vy, cy))
                nn.append((cx.sum(), cy.sum()))

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
            for k, xx in enumerate(x):
                vx = np.asarray([u[0] for u in xx[0]], dtype=int)
                cx = np.asarray([u[1] for u in xx[0]], dtype=float)
                vy = np.asarray([u[0] for u in xx[1]], dtype=int)
                cy = np.asarray([u[1] for u in xx[1]], dtype=float)
                nx = np.sum(cx)

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
                nn.append((cx.sum(), cy.sum()))

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
