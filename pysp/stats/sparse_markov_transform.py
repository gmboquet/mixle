"""Create, estimate, and sample from an integer sparse Markov hidden association model.

Defines the SparseMarkovAssociationDistribution, SparseMarkovAssociationSampler,
SparseMarkovAssociationAccumulatorFactory, SparseMarkovAssociationAccumulator, SparseMarkovAssociationEstimator, and
the SparseMarkovAssociationDataEncoder classes for use with pysparkplug.

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
import random
from typing import Optional, List, Tuple, Union, Sequence, Any, TypeVar, Dict
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from pysp.utils.optsutil import count_by_value
import itertools
from pysp.arithmetic import maxrandint

T = Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]
SS1 = TypeVar('SS1')


class SparseMarkovAssociationDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, init_prob_vec: Union[Sequence[float], np.ndarray], cond_prob_mat: csr_matrix,
                 alpha: float = 0.0, len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 low_memory: bool = False) -> None:
        """SparseMarkovAssociationDistribution object for creating a sparse Markov association model.

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
        s1 = ','.join(map(str, self.init_prob_vec))
        temp = self.cond_prob_mat.nonzero()
        tt = np.asarray(self.cond_prob_mat[temp[0], temp[1]]).flatten()
        s20 = ','.join(map(str, tt))
        s21 = ','.join(map(str, temp[0]))
        s22 = ','.join(map(str, temp[1]))
        s2 = '([%s], ([%s],[%s]))' % (s20, s21, s22)
        s3 = str(self.alpha)
        s4 = str(self.len_dist)
        return 'SparseMarkovAssociationDistribution([%s], %s, alpha=%s, len_dist=%s)' % (s1, s2, s3, s4)

    def density(self, x: Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]) -> float:

        nw = self.num_vals
        a  = self.alpha / nw
        b  = 1 - self.alpha

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        nx = np.sum(cx)
        ny = np.sum(cy)

        temp = self.cond_prob_mat[vx[:, None], vy].toarray()
        ll2 = np.dot(np.log(np.dot((temp * b + a).T, cx / nx)), cy)
        ll1 = np.dot(np.log(self.init_prob_vec[vx] * b + a), cx)
        rv  = ll2# + ll2
        rv += self.len_dist.log_density([nx, ny])

        return float(rv)

    def seq_log_density(self, x) -> np.ndarray:

        nw = self.num_vals
        a = self.alpha / nw
        b = 1 - self.alpha

        xlen = len(x[0])

        if x[3] is not None:

            obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec = x[3]

            vv = x[2]

            p = np.asarray(self.cond_prob_mat[vv[:,0], vv[:,1]]).flatten()
            p = (p*b + a)
            sval = np.bincount(seqidx, weights=p[pairidx]*cxvec)
            np.log(sval, out=sval)
            sval *= fcyvec
            rv = np.bincount(fsqyvec, weights=sval, minlength=xlen)

        else:
            rv = np.zeros(len(x[0]), dtype=np.float64)

            for i, entry in enumerate(x[0]):
                xx, cx, yy, cy = entry
                nx = np.sum(cx)

                temp = self.cond_prob_mat[xx[:,None], yy].toarray()
                ll2 = np.dot(np.log(np.dot((temp*b + a).T, cx/nx)),cy)
                ll1 = np.dot(np.log(self.init_prob_vec[xx]*b + a), cx)

                rv[i] = ll2# + ll2

        if not isinstance(self.len_dist, NullDistribution):
            lln = self.len_dist.seq_log_density(x[1])
            rv += lln

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'SparseMarkovAssociationSampler':
        return SparseMarkovAssociationSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SparseMarkovAssociationEstimator':
        return SparseMarkovAssociationEstimator(num_vals=self.num_vals,alpha=self.alpha,
                                                len_estimator=self.len_dist.estimator(), low_memory=self.low_memory)

    def dist_to_encoder(self) -> 'SparseMarkovAssociationDataEncoder':
        return SparseMarkovAssociationDataEncoder(len_encoder=self.len_dist.dist_to_encoder(),
                                                  low_memory=self.low_memory)


class SparseMarkovAssociationSampler(DistributionSampler):

    def __init__(self, dist: SparseMarkovAssociationDistribution, seed: Optional[int] = None) -> None:
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.size_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> Union[T, Sequence[T]]:

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


class SparseMarkovAssociationAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals: int, size_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 keys: Tuple[Optional[str], Optional[str]] = (None, None), low_memory: bool = True) -> None:
        self.init_count = np.zeros(num_vals)
        self.trans_count: Optional[Union[lil_matrix, csr_matrix]] = None
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        self.num_vals = num_vals
        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.low_memory = low_memory

        self._init_rng = False
        self._size_rng = None

    def update(self, x: T, weight: float, estimate: SparseMarkovAssociationDistribution) -> None:

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = lil_matrix((num_vals, num_vals))

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        temp = estimate.cond_prob_mat[vx[:, None], vy].toarray()

        loc_cprob = temp * cx[:, None]
        w = loc_cprob.sum(axis=0)
        loc_cprob *= (cy / w) * weight

        self.trans_count[vx[:, None], vy] += loc_cprob
        self.init_count[vx] += cx

        self.size_accumulator.update((cx.sum(), cy.sum()), weight, estimate.len_dist)

    def initialize_rng(self, rng: np.random.RandomState) -> None:
        if not self._init_rng:
            self._size_rng = np.random.RandomState(seed=rng.randint(2**31))
            self._init_rng = True

    def initialize(self, x: T, weight: float, rng: np.random.RandomState) -> None:

        if not self._init_rng:
            self.initialize_rng(rng)

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = lil_matrix((num_vals, num_vals))

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        self.trans_count[vx[:, None], vy] += np.outer(cx/np.sum(cx), cy)*weight
        self.init_count[vx] += cx*weight

        self.size_accumulator.initialize((cx.sum(), cy.sum()), weight, self._size_rng)

    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None:

        if not self._init_rng:
            self.initialize_rng(rng)

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = csr_matrix((num_vals, num_vals))

        nw = self.num_vals

        if x[3] is not None:

            obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec = x[3]

            vv = x[2]

            p = np.asarray(self.trans_count[vv[:,0], vv[:,1]]).flatten()
            pp = p[pairidx]*cxvec
            sval = np.bincount(seqidx, weights=pp)
            np.divide(weights[fsqyvec], sval, out=sval)
            sval *= fcyvec
            pp   *= sval[seqidx]
            pp = np.bincount(pairidx, weights=pp)

            umat = csr_matrix((pp, (vv[:,0], vv[:,1])), shape=(nw, nw))
            self.trans_count += umat
            self.init_count += np.bincount(fvxvec, weights=fcxvec, minlength=nw)

        else:

            for i, (entry, weight) in enumerate(zip(x[0], weights)):

                vx, cx, vy, cy = entry

                self.trans_count[vx[:, None], vy] += np.outer(cx / np.sum(cx), cy) * weight
                self.init_count[vx] += cx*weight

        self.size_accumulator.seq_initialize(x[1], weights, self._size_rng)

    def seq_update(self, x, weights: np.ndarray, estimate: SparseMarkovAssociationDistribution) -> None:

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = csr_matrix((num_vals, num_vals))

        nw = self.num_vals

        if x[3] is not None:

            obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec = x[3]

            vv = x[2]

            p = np.asarray(estimate.cond_prob_mat[vv[:,0], vv[:,1]]).flatten()
            pp = p[pairidx]*cxvec
            sval = np.bincount(seqidx, weights=pp)
            np.divide(weights[fsqyvec], sval, out=sval)
            sval *= fcyvec
            pp   *= sval[seqidx]
            pp = np.bincount(pairidx, weights=pp)

            umat = csr_matrix((pp, (vv[:,0], vv[:,1])), shape=(nw, nw))
            self.trans_count += umat
            self.init_count += np.bincount(fvxvec, weights=fcxvec, minlength=nw)

        else:

            nzv = x[2]
            umat = csr_matrix((np.zeros(nzv.shape[0]), (nzv[:, 0], nzv[:, 1])), shape=(nw, nw))

            for i, (entry, weight) in enumerate(zip(x[0], weights)):

                vx, cx,vy, cy = entry

                temp = estimate.cond_prob_mat[vx[:, None], vy].toarray()

                loc_cprob = temp * cx[:, None]
                w = loc_cprob.sum(axis=0)
                loc_cprob *= (cy / w) * weight

                umat[vx[:, None], vy] += loc_cprob
                self.init_count[vx] += cx

            self.trans_count += umat

        self.size_accumulator.seq_update(x[1], weights, estimate.len_dist)

    def combine(self, suff_stat: Tuple[np.ndarray, Optional[Union[lil_matrix, csr_matrix]], SS1]) \
            -> 'SparseMarkovAssociationAccumulator':
        init_count, trans_count, size_acc = suff_stat

        self.size_accumulator.combine(size_acc)
        self.init_count += init_count
        self.trans_count += trans_count

        return self

    def value(self) -> Tuple[np.ndarray, Optional[Union[lil_matrix, csr_matrix]], Any]:
        return self.init_count, self.trans_count, self.size_accumulator.value()

    def from_value(self, x: Tuple[np.ndarray, Optional[Union[lil_matrix, csr_matrix]], SS1]) \
            -> 'SparseMarkovAssociationAccumulator':
        init_count, trans_count, size_acc = x

        self.init_count = init_count
        self.trans_count = trans_count
        self.size_accumulator.from_value(size_acc)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:

        if self.init_key is not None:
            if self.init_key in stats_dict:
                stats_dict[self.init_key] += self.init_count
            else:
                stats_dict[self.init_key] = self.init_count

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                stats_dict[self.trans_key] += self.trans_count
            else:
                stats_dict[self.trans_key] = self.trans_count

        self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_count = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_count = stats_dict[self.trans_key]

        self.size_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'SparseMarkovAssociationDataEncoder':
        return SparseMarkovAssociationDataEncoder(len_encoder=self.size_accumulator.acc_to_encoder(),
                                                  low_memory=self.low_memory)


class SparseMarkovAssociationAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, num_vals: int, len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 low_memory: bool = True,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.low_memory = low_memory
        self.keys = keys
        self.num_vals = num_vals

    def make(self) -> 'SparseMarkovAssociationAccumulator':
        return SparseMarkovAssociationAccumulator(self.num_vals, size_acc=self.len_factory.make(), keys=self.keys,
                                                  low_memory=self.low_memory)


class SparseMarkovAssociationEstimator(ParameterEstimator):

    def __init__(self, num_vals: int, alpha: float = 0.0, len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 suff_stat: Optional[Any] = None, pseudo_count: Optional[float] = None,
                 low_memory: bool = True,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        """SparseMarkovAssociationEstimator object for estimating SparseMarkovAssociationModel objects from aggregated
            sufficient statistics.

        Args:
            num_vals (int): Number of values in S1.
            alpha (float): Regularization parameter (should be between 0 and 1).
            len_estimator (Optional[ParameterEstimator]): ParameterEstimator object for the length of observations.
            suff_stat (Optional[Any]): Kept for consistency with estimate function.
            pseudo_count (Optional[float]): Regularize sufficient statistics.
            low_memory (bool): If True, use low_memory options.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial distribution and state transition stats.

        Attributes:
            num_vals (int): Number of values in S1.
            alpha (float): Regularization parameter (should be between 0 and 1).
            len_estimator (ParameterEstimator): ParameterEstimator object for the length of observations.
            suff_stat (Optional[Any]): Kept for consistency with estimate function.
            pseudo_count (Optional[float]): Regularize sufficient statistics.
            low_memory (bool): If True, use low_memory options.
            keys (Tuple[Optional[str], Optional[str]]): Keys for initial distribution and state transition stats.

        """
        self.keys = keys
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.num_vals = num_vals
        self.alpha = alpha
        self.low_memory = low_memory

    def accumulator_factory(self) -> 'SparseMarkovAssociationAccumulatorFactory':
        return SparseMarkovAssociationAccumulatorFactory(self.num_vals, self.len_estimator.accumulator_factory(),
                                                         self.low_memory,
                                                         self.keys)

    def estimate(self, nobs: Optional[float],
                 suff_stat: Tuple[np.ndarray, Optional[Union[lil_matrix, csr_matrix]], SS1]) \
            -> 'SparseMarkovAssociationDistribution':
        """Estimate SparseMarkovAssociationDistribution objects from aggregated sufficient statistics.

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
        row_sum.data = 1.0/row_sum.data

        init_prob = init_count / np.sum(init_count)
        trans_prob = trans_count.multiply(row_sum)

        return SparseMarkovAssociationDistribution(init_prob, trans_prob, self.alpha, len_dist, self.low_memory)


class SparseMarkovAssociationDataEncoder(DataSequenceEncoder):

    def __init__(self, len_encoder: DataSequenceEncoder, low_memory: bool) -> None:
        self.len_encoder = len_encoder
        self.low_memory = low_memory

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SparseMarkovAssociationDataEncoder):
            return other.len_encoder == self.len_encoder and self.low_memory == other.low_memory
        else:
            return False

    def __str__(self) -> str:
        return 'SparseMarkovAssociationDataEncoder(len_encoder=' + \
               str(self.len_encoder)+',low_memory=' + str(self.low_memory) + ')'

    def seq_encode(self, x: Sequence[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]):

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

            obsidx  = []
            pairidx = []
            seqidx  = []
            cxvec   = []
            cyvec   = []

            fcyvec  = []
            fcxvec  = []
            fvxvec  = []
            fsqxvec  = []
            fsqyvec  = []

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
                fsqxvec.extend([k]*len(vx))
                fsqyvec.extend([k]*len(vy))

                for i, vvy in enumerate(vy):
                    ridx += 1
                    for j, vvx in enumerate(vx):

                        if (vvx, vvy) not in vmap:
                            vmap[(vvx, vvy)] = len(vmap)
                        widx = vmap[(vvx, vvy)]
                        obsidx.append(k)
                        seqidx.append(ridx)
                        pairidx.append(widx)
                        cxvec.append(cx[j]/nx)
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

