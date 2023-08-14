"""Create, estimate, and sample from an integer latent Dirichlet allocation model (LDA).


Defines the LDADistribution, LDASampler, LDAAccumulatorFactory, LDAEstimatorAccumulator, LDAEstimator, and the
LDADataEncoder classes for use with pysparkplug.

LDA is a generative model for producing draws from multinomial distribution. The process for generating a document of
length N from an LDA with L topics is given as follows:

    (1) Draw theta ~ Dirichlet(alpha) (alpha is L dimensional)
    (2) Draw topic-counts z_1,....,z_L ~ Multinomial(N, theta)
    (3) From each topic l = 1,2,...,L draw z_l words w_{i,l}, w_{i+1,l},...,w_{z_l,l} ~ Categorical(beta_l),
        where each topic has its own Categorical distribution parameterized by beta_l.

A document is then given by the bag of words produced from this sampling process. Note that a length distribtion is
used to sample the number of words in a given document.

"""
import sys
import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, gammaln

from pysp.arithmetic import maxrandint
from pysp.stats.dirichlet import DirichletDistribution
from pysp.utils.special import digammainv
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory
from pysp.utils.optsutil import count_by_value
from pysp.utils.vector import row_choice

from typing import TypeVar, Dict, List, Sequence, Any, Optional, Tuple, Union, Callable

E0 = TypeVar('E0')
SS0 = TypeVar('SS0')

# import pysp.c_ext

class LDADistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, topics: Sequence[SequenceEncodableProbabilityDistribution],
                 alpha: Union[Sequence[float], np.ndarray],
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 gamma_threshold: float = 1.0e-8) -> None:
        """LDADistribution object for defining a Latent Dirichlet allocation model.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions for the LDA.
            alpha (Union[Sequence[float], np.ndarray]): Parameter to the prior Dirichlet for which topics are drawn.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for length of documents.
                Must be set to non-negative support distribution for sampling.
            gamma_threshold (float): For numerical stability in estimation.

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions for the LDA.
            alpha (np.ndarray): Parameter to the prior Dirichlet for which topics are drawn.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for length of documents.
                Must be set to non-negative support distribution for sampling. Default to NullDistribution.
            gamma_threshold (float): For numerical stability in estimation.

        """
        self.topics = topics
        self.n_topics = len(topics)
        self.alpha = np.asarray(alpha)
        self.len_dist = len_dist
        self.gamma_threshold = gamma_threshold

    def __str__(self) -> str:
        return 'LDADistribution([%s], [%s])' % (','.join([str(u) for u in self.topics]), ','.join(map(str, self.alpha)))

    def density(self, x: Sequence[Tuple[int, float]]) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[Tuple[int, float]]) -> float:
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_log_density(enc_x)[0]

    def seq_log_density(self, x: Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray], E0]) -> np.ndarray:
        num_topics = self.n_topics
        alpha = self.alpha
        num_documents, idx, counts, _, enc_data = x

        idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
        idx_full *= num_topics
        idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

        log_density_gamma, document_gammas, per_topic_log_densities = seq_posterior(self, x)

        # This block keeps the gammas positive
        log_density_gamma[np.bitwise_or(np.isnan(log_density_gamma), np.isinf(log_density_gamma))] = sys.float_info.min
        log_density_gamma[log_density_gamma <= 0] = sys.float_info.min
        document_gammas[np.bitwise_or(np.isnan(document_gammas), np.isinf(document_gammas))] = sys.float_info.min

        elob0 = digamma(document_gammas) - digamma(np.sum(document_gammas, axis=1, keepdims=True))
        elob1 = elob0[idx, :]
        elob2 = log_density_gamma * (elob1 + per_topic_log_densities - np.log(log_density_gamma))
        elob3 = np.sum(elob0 * ((alpha - 1.0) - (document_gammas - 1.0)), axis=1)
        elob4 = np.bincount(idx_full.flat, weights=elob2.flat)
        elob5 = np.sum(np.reshape(elob4, (-1, num_topics)), axis=1)
        elob6 = np.sum(gammaln(document_gammas), axis=1) - gammaln(document_gammas.sum(axis=1))
        elob7 = gammaln(alpha.sum()) - gammaln(alpha).sum()

        elob = elob3 + elob5 + elob6 + elob7

        return elob

    def seq_component_log_density(self, x: Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray], E0]) -> np.ndarray:

        num_topics = self.n_topics
        alpha = self.alpha
        num_documents, idx, counts, _, enc_data = x

        ll_mat = np.zeros((len(idx), self.n_topics))
        ll_mat.fill(-np.inf)

        rv = np.zeros((num_documents, self.n_topics))
        rv.fill(-np.inf)

        for i in range(num_topics):
            ll_mat[:, i] = self.topics[i].seq_log_density(enc_data)
            rv[:, i] = np.bincount(idx, weights=ll_mat[:, i] * counts, minlength=num_documents)

        return rv

    def seq_posterior(self, x: Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray], E0]) -> np.ndarray:
        num_topics = self.n_topics
        alpha = self.alpha
        num_documents, idx, counts, _, enc_data = x

        log_density_gamma, document_gammas, per_topic_log_densities = seq_posterior(self, x)

        document_gammas /= document_gammas.sum(axis=1, keepdims=True)

        return document_gammas

    def sampler(self, seed: Optional[int] = None) -> 'LDASampler':
        return LDASampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'LDAEstimator':
        if pseudo_count is None:
            return LDAEstimator(estimators=[d.estimator() for d in self.topics])
        else:
            return LDAEstimator(estimators=[d.estimator() for d in self.topics],
                                pseudo_count=(pseudo_count,pseudo_count))

    def dist_to_encoder(self) -> 'LDADataEncoder':
        return LDADataEncoder(encoder=self.topics[0].dist_to_encoder())

class LDASampler(DistributionSampler):

    def __init__(self, dist: LDADistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.n_topics = dist.n_topics
        self.comp_samplers = [self.dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in
                              range(dist.n_topics)]
        self.dirichlet_sampler = DirichletDistribution(dist.alpha).sampler(self.rng.randint(0, maxrandint))
        self.len_dist = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> Union[Sequence[Any], Any]:
        """Sample return value is not counted by value!"""
        if size is None:
            n = self.len_dist.sample()
            weights = self.dirichlet_sampler.sample()
            topic_counts = self.rng.multinomial(n, pvals=weights)
            topics = []
            rv = []
            for i in np.flatnonzero(topic_counts):
                topics.extend([i] * topic_counts[i])
                rv.extend(self.comp_samplers[i].sample(size=topic_counts[i]))

            return rv

        else:
            return [self.sample() for i in range(size)]


class LDAEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator],
                 keys: Optional[Tuple[Optional[str],Optional[str]]] = (None, None),
                 prev_alpha: Optional[np.ndarray] = None) -> None:
        self.accumulators = accumulators
        self.num_topics = len(accumulators)
        self.sum_of_logs = np.zeros(self.num_topics)
        self.doc_counts = 0.0
        self.topic_counts = np.zeros(self.num_topics)
        self.prev_alpha = prev_alpha
        self.alpha_key, self.topics_key = keys if keys is not None else (None, None)

        self._init_rng = False
        self._rng_theta = None
        self._rng_idx = None
        self._rng_topics = None

    def update(self, x, weight, estimate):
        pass

    def _rng_initialize(self, rng: RandomState) -> None:
        if not self._init_rng:
            seeds = rng.randint(maxrandint, size=3+self.num_topics)
            self._rng_theta = RandomState(seed=seeds[0])
            self._rng_idx = RandomState(seed=seeds[1])
            self._rng_w = RandomState(seed=seeds[2])
            self._rng_topics = [RandomState(seed=seeds[3+j]) for j in range(self.num_topics)]
            self._init_rng = True

    def seq_initialize(self, x: Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray], E0], weights: np.ndarray,
                       rng: np.random.RandomState) -> None:

        num_documents, idx, counts, old_gammas, enc_data = x

        if not self._init_rng:
            self._rng_initialize(rng)

        if self.prev_alpha is None:
            self.prev_alpha = np.ones(self.num_topics)

        theta     = self._rng_theta.dirichlet(self.prev_alpha, size=num_documents)
        theta_rep = theta[idx, :]

        idx_list  = row_choice(p_mat=np.reshape(theta_rep, (-1, self.num_topics)), rng=self._rng_idx)

        self.sum_of_logs += np.sum(np.log(theta), axis=0, keepdims=False)
        self.doc_counts += np.sum(weights)

        ww_v = -np.log(self._rng_w.rand(self.num_topics*len(idx)))
        ww_v[idx_list + np.arange(0, len(ww_v), self.num_topics)] += 1
        ww_v = np.reshape(ww_v, (-1, self.num_topics))
        ww_v /= ww_v.sum(axis=1, keepdims=True)

        temp = np.reshape(weights[idx]*counts, (len(idx), 1))
        ww_v *= temp

        for j in range(self.num_topics):
            w = ww_v[:, j]
            self.topic_counts[j] += np.sum(w)
            self.accumulators[j].seq_initialize(enc_data, w, self._rng_topics[j])

    def initialize(self, x, weight: float, rng: np.random.RandomState) -> None:

        if self.prev_alpha is None:
            self.prev_alpha = np.ones(self.num_topics)

        if not self._init_rng:
            self._rng_initialize(rng)

        counts = np.reshape([x[i][1] for i in range(len(x))], (len(x), 1))

        theta = self._rng_theta.dirichlet(self.prev_alpha)
        print(theta)

        theta_rep = theta[np.arange(0, self.num_topics*len(x)) % self.num_topics]
        idx_list = row_choice(p_mat=np.reshape(theta_rep, (-1, self.num_topics)), rng=self._rng_idx)
        print('\n')
        print(idx_list)
        self.sum_of_logs += np.log(theta)
        self.doc_counts += weight

        ww_v = -np.log(self._rng_w.rand(self.num_topics*len(x)))
        ww_v[idx_list + np.arange(0, self.num_topics*len(x), self.num_topics)] += 1
        ww_v = np.reshape(ww_v, (-1, self.num_topics))
        ww_v /= np.sum(ww_v, axis=1, keepdims=True)
        ww_v *= counts*weight

        for j in range(self.num_topics):
            w = ww_v[:, j]
            for i in range(len(x)):
                self.accumulators[j].initialize(x[i][0], w[i], self._rng_topics[j])
                self.topic_counts[j] += w[i]

    def seq_update(self, x: Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray], E0], weights: np.ndarray,
                   estimate: LDADistribution) -> None:

        num_documents, idx, counts, old_gammas, enc_data = x
        log_density_gamma, final_gammas, per_topic_log_densities = seq_posterior(estimate, x)

        for i in range(self.num_topics):
            self.accumulators[i].seq_update(enc_data, log_density_gamma[:, i] * weights[idx] * counts,
                                            estimate.topics[i])

        mlpf = digamma(final_gammas) - digamma(np.sum(final_gammas, axis=1, keepdims=True))

        self.sum_of_logs += np.dot(weights, mlpf)
        self.doc_counts += weights.sum()
        self.topic_counts += np.sum(log_density_gamma, axis=0)
        self.prev_alpha = estimate.alpha

    # return num_documents, idx, counts, final_gammas, enc_data

    def combine(self, suff_stat: Tuple[Optional[np.ndarray], np.ndarray, float, np.ndarray, Sequence[SS0]]) \
            -> 'LDAEstimatorAccumulator':

        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = suff_stat

        if self.prev_alpha is None:
            self.prev_alpha = prev_alpha

        self.sum_of_logs += sum_of_logs
        self.doc_counts += doc_counts
        self.topic_counts += topic_counts

        for i in range(self.num_topics):
            self.accumulators[i].combine(topic_suff_stats[i])

        return self

    def value(self) -> Tuple[Optional[np.ndarray], np.ndarray, float, np.ndarray, Sequence[Any]]:
        return self.prev_alpha, self.sum_of_logs, self.doc_counts, self.topic_counts, [u.value() for u in
                                                                                       self.accumulators]

    def from_value(self, x: Tuple[Optional[np.ndarray], np.ndarray, float, np.ndarray, Sequence[SS0]]) \
            -> 'LDAEstimatorAccumulator':

        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = x

        self.prev_alpha = prev_alpha
        self.sum_of_logs = sum_of_logs
        self.doc_counts = doc_counts
        self.topic_counts = topic_counts
        self.accumulators = [self.accumulators[i].from_value(topic_suff_stats[i]) for i in range(self.num_topics)]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:

        if self.alpha_key is not None:
            if self.alpha_key in stats_dict:

                p_sol, p_doc, p_pa = stats_dict[self.alpha_key]

                prev_alpha = self.prev_alpha if self.prev_alpha is not None else p_pa
                stats_dict[self.alpha_key] = (self.sum_of_logs + p_sol, self.doc_counts + p_doc, prev_alpha)

            else:
                stats_dict[self.alpha_key] = (self.sum_of_logs, self.doc_counts, self.prev_alpha)

        if self.topics_key is not None:
            if self.topics_key in stats_dict:
                acc = stats_dict[self.topics_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.topics_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.alpha_key is not None:
            if self.alpha_key in stats_dict:
                p_sol, p_doc, p_pa = stats_dict[self.alpha_key]
                self.prev_alpha = p_pa
                self.sum_of_logs = p_sol
                self.doc_counts = p_doc

        if self.topics_key is not None:
            if self.topics_key in stats_dict:
                acc = stats_dict[self.topics_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'LDADataEncoder':
        return LDADataEncoder(encoder=self.accumulators[0].acc_to_encoder())


class LDAEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, factories: Sequence[StatisticAccumulatorFactory], dim: int,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
                 prev_alpha: Optional[np.ndarray] = None) -> None:
        self.factories = factories
        self.dim = dim
        self.keys = keys if keys is None else (None, None)
        self.prev_alpha = prev_alpha

    def make(self) -> 'LDAEstimatorAccumulator':
        return LDAEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys, self.prev_alpha)


class LDAEstimator(ParameterEstimator):

    def __init__(self, estimators: Sequence[ParameterEstimator], suff_stat: Optional[Any] = None,
                 pseudo_count: Optional[Tuple[float, float]] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
                 fixed_alpha: Optional[np.ndarray] = None,
                 gamma_threshold: float = 1.0e-8,
                 alpha_threshold: float = 1.0e-8) -> None:
        self.num_topics = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None)
        self.gamma_threshold = gamma_threshold
        self.alpha_threshold = alpha_threshold
        self.fixed_alpha = fixed_alpha

    def accumulator_factory(self) -> 'LDAEstimatorAccumulatorFactory':
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return LDAEstimatorAccumulatorFactory(est_factories, self.num_topics, self.keys, self.fixed_alpha)

    def estimate(self, nobs: Optional[float], suff_stat):

        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = suff_stat

        num_topics = self.num_topics
        topics = [self.estimators[i].estimate(topic_counts[i], topic_suff_stats[i]) for i in range(num_topics)]

        if doc_counts == 0:
            sys.stderr.write('Warning: LDA Estimation performed with zero documents.\n')
            LDADistribution(topics, prev_alpha, gamma_threshold=self.gamma_threshold)

        if self.fixed_alpha is None:

            if self.pseudo_count is not None:
                mean_of_logs = (sum_of_logs + np.log(self.pseudo_count[1])) / (doc_counts + self.pseudo_count[0])

            # new_alpha, _ = find_alpha(prev_alpha, sum_of_logs/doc_counts, gamma_threshold*np.sqrt(float(doc_counts)))
            new_alpha, _ = update_alpha(prev_alpha, sum_of_logs / doc_counts, self.alpha_threshold)
        else:
            new_alpha = np.asarray(self.fixed_alpha).copy()

        return LDADistribution(topics, new_alpha, gamma_threshold=self.gamma_threshold)

class LDADataEncoder(DataSequenceEncoder):

    def __init__(self, encoder: DataSequenceEncoder):
        self.encoder = encoder

    def __str__(self) -> str:
        return 'LDADataEncoder(encoder=' + str(self.encoder) + ')'

    def __eq__(self, other) -> bool:
        if isinstance(other, LDADataEncoder):
            return self.encoder == other.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Sequence[Tuple[int, float]]]) \
            -> Tuple[int, np.ndarray, np.ndarray, Optional[Any], Any]:
        """Encode a sequence of iid LDA observations for vectorized functions.

        Return value 'rv' is a Tuple containing:
            rv[0] (int): Number of documents in corpus.
            rv[1] (np.ndarray): Document id for flattened array of values.
            rv[2] (np.ndarray): Flattened array of counts for each value in each document.
            rv[3] (Optional[np.ndarray]): Currently default to None
            rv[4] (E0): Sequence encoded flattened values.

        Args:
            x (Sequence[Sequence[Tuple[int, float]]]): Sequence of LDA documents.

        Returns:
            See above for details.

        """
        num_documents = len(x)

        tx = []
        ctx = []
        nx = []
        tidx = []
        for i in range(len(x)):
            nx.append(len(x[i]))
            for j in range(len(x[i])):
                tidx.append(i)
                tx.append(x[i][j][0])
                ctx.append(x[i][j][1])

        idx = np.asarray(tidx)
        counts = np.asarray(ctx)
        gammas = None
        enc_data = self.encoder.seq_encode(tx)

        return num_documents, idx, counts, gammas, enc_data


def update_alpha(alpha_curr, mean_log_p, alpha_threshold) -> Tuple[np.ndarray, int]:
    alpha = alpha_curr.copy()
    asum = alpha.sum()
    res = np.inf
    its_cnt = 0
    while res > alpha_threshold:
        dasum = digamma(asum)
        alpha_old = alpha
        alpha = digammainv(mean_log_p + dasum)
        asum = alpha.sum()
        res = np.abs(alpha - alpha_old).sum() / asum
        its_cnt += 1

    return alpha, its_cnt


def mpe_update(x_mat: Optional[np.ndarray], y: np.ndarray, min_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    if x_mat is None:
        x_mat = np.reshape(y, (1, -1))
        return x_mat, y
    elif x_mat.shape[0] < min_size:
        x_mat = np.concatenate((x_mat, np.reshape(y, (1, -1))), axis=0)
        return x_mat, y

    dy = y - x_mat[-1, :]
    u_mat = (x_mat[1:, :] - x_mat[:-1, :]).T
    x2_mat = x_mat[1:, :].T
    c = np.dot(np.linalg.pinv(u_mat), dy)
    c *= -1
    s = (np.dot(x2_mat, c) + y) / (c.sum() + 1)

    x_mat = np.concatenate((x_mat, np.reshape(y, (1, -1))), axis=0)

    return x_mat, s


def mpe(x0, f, eps: float) -> Tuple[np.ndarray, int]:
    x1 = f(x0)
    x2 = f(x1)
    x3 = f(x2)
    x_mat = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 2

    while res > eps:
        y = f(x_mat[-1, :])
        dy = y - x_mat[-1, :]
        u_mat = (x_mat[1:, :] - x_mat[:-1, :]).T
        x2_mat = x_mat[1:, :].T
        c = np.dot(np.linalg.pinv(u_mat), dy)
        c *= -1
        s = (np.dot(x2_mat, c) + y) / (c.sum() + 1)

        res = np.abs(s - s0).sum()
        s0 = s
        x_mat = np.concatenate((x_mat, np.reshape(y, (1, -1))), axis=0)
        its_cnt += 1

    return s, its_cnt


def alpha_seq_lambda(mean_log_p: float) -> Callable[[np.ndarray], float]:
    def next_alpha(alpha_current: np.ndarray):
        return digammainv(mean_log_p + digamma(alpha_current.sum()))

    return next_alpha


def find_alpha(current_alpha: np.ndarray, mlp: float, thresh: float):
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh)


def seq_posterior2(estimate: LDADistribution, x: Tuple[int, np.ndarray, np.ndarray, Optional[Any], E0]):
    alpha = estimate.alpha
    topics = estimate.topics
    gamma_threshold = estimate.gamma_threshold

    num_documents, idx, counts, gammas, enc_data = x

    num_topics = len(topics)
    num_samples = len(idx)

    per_topic_log_densities0 = np.asarray([topics[i].seq_log_density(enc_data) for i in range(num_topics)]).transpose()

    per_topic_log_densities = per_topic_log_densities0.copy()
    max_val = per_topic_log_densities.max(axis=1, keepdims=True)
    per_topic_log_densities -= max_val
    per_topic_log_densities = np.exp(per_topic_log_densities)

    idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
    idx_full *= num_topics
    idx_full += np.reshape(np.arange(num_topics), (1, num_topics))
    alpha_loc = np.repeat(np.reshape(alpha, (1, num_topics)), num_documents, axis=0)

    if gammas is None:
        document_gammas = alpha_loc + np.reshape(np.bincount(idx_full.flat), (num_documents, num_topics)) / float(
            num_topics)
    else:
        document_gammas = gammas.copy()

    document_gammas = document_gammas.astype(np.float64)
    idx = idx.astype(np.intp)
    alpha_loc = alpha_loc.astype(np.float64)
    per_topic_log_densities = per_topic_log_densities.astype(np.float64)
    ccc = counts.astype(np.float64)

    rv0 = np.zeros(num_documents, dtype=bool)
    rv1 = np.zeros(document_gammas.shape, dtype=np.float64)
    rv2 = np.zeros(document_gammas.shape, dtype=np.float64)
    rv3 = np.zeros(per_topic_log_densities.shape, dtype=np.float64)
    rv4 = np.arange(0, num_samples, dtype=np.intp)
    rv5 = np.zeros(num_documents, dtype=np.float64)

    aa, bb = pysp.c_ext.lda_update(idx, document_gammas, rv1, rv2, alpha_loc, per_topic_log_densities, rv3, ccc, rv0,
                                   rv4, rv5, -1, gamma_threshold)

    final_gammas = bb + alpha_loc
    log_density_gamma = aa

    return log_density_gamma, final_gammas, per_topic_log_densities0


def seq_posterior(estimate: LDADistribution, x: Tuple[int, np.ndarray, np.ndarray, Optional[Any], E0]):
    alpha = estimate.alpha
    topics = estimate.topics
    gamma_threshold = estimate.gamma_threshold

    num_documents, idx, counts, gammas, enc_data = x

    num_topics = len(topics)
    num_samples = len(idx)

    per_topic_log_densities = np.asarray([topics[i].seq_log_density(enc_data) for i in range(num_topics)]).transpose()
    per_topic_log_densities2 = per_topic_log_densities.copy()
    per_topic_log_densities2 -= np.max(per_topic_log_densities2, axis=1, keepdims=True)
    np.exp(per_topic_log_densities2, out=per_topic_log_densities2)
    per_topic_log_densities3 = per_topic_log_densities2.copy()

    idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
    idx_full *= num_topics
    idx_full += np.reshape(np.arange(num_topics), (1, num_topics))
    alpha_loc = np.reshape(alpha, (1, num_topics))

    if gammas is None:
        document_gammas = alpha_loc + np.reshape(np.bincount(idx_full.flat), (num_documents, num_topics)) / float(
            num_topics)
    else:
        document_gammas = gammas.copy()

    document_gammas2 = np.zeros((num_documents, num_topics), dtype=float)
    document_gammas3 = np.zeros((num_documents, num_topics), dtype=float)

    gamma_sum = np.zeros((num_documents, 1), dtype=float)
    gamma_asum = np.zeros((num_documents, 1), dtype=float)

    posterior_sum_ll = np.zeros((num_samples, 1), dtype=float)

    log_density_gamma = np.zeros(per_topic_log_densities.shape, dtype=float)
    document_gamma_diff_loc = np.zeros((num_documents, num_topics), dtype=float)
    log_density_gamma_loc = log_density_gamma.view()
    posterior_sum_ll_loc = posterior_sum_ll.view()
    gamma_asum_loc = gamma_asum.view()
    gamma_sum_loc = gamma_sum.view()

    ndoc = num_documents

    rel_idx = idx.copy()
    rel_counts = counts.copy()
    rel_counts = np.reshape(rel_counts, (-1, 1))

    rem_gammas_idx = np.arange(num_documents, dtype=int)
    final_gammas = np.zeros((num_documents, num_topics), dtype=float)
    final_gammas_idx = np.zeros(num_documents, dtype=int)
    finished_count = 0
    itr_cnt = 0
    gamma_itr_cnt = np.zeros(num_documents, dtype=int)

    #

    digamma(document_gammas, out=document_gammas2)
    temp = np.max(document_gammas2, axis=1, keepdims=True)
    np.exp(document_gammas2 - temp, out=document_gammas3)

    np.multiply(per_topic_log_densities2, document_gammas3[rel_idx, :], out=log_density_gamma_loc)
    np.sum(log_density_gamma_loc, axis=1, keepdims=True, out=posterior_sum_ll_loc)
    log_density_gamma_loc /= posterior_sum_ll_loc

    old_stuff = None

    while ndoc > 0:

        itr_cnt += 1

        digamma(document_gammas, out=document_gammas2)
        temp = np.max(document_gammas2, axis=1, keepdims=True)
        document_gammas2 -= temp
        np.exp(document_gammas2, out=document_gammas3)

        np.multiply(per_topic_log_densities2, document_gammas3[rel_idx, :], out=log_density_gamma_loc)
        np.sum(log_density_gamma_loc, axis=1, keepdims=True, out=posterior_sum_ll_loc)
        posterior_sum_ll_loc /= rel_counts
        log_density_gamma_loc /= posterior_sum_ll_loc

        gamma_updates = np.bincount(idx_full.flat, weights=log_density_gamma_loc.flat)
        gamma_updates = np.reshape(gamma_updates, (-1, num_topics))
        gamma_updates += alpha_loc

        np.subtract(document_gammas, gamma_updates, out=document_gamma_diff_loc)
        np.abs(document_gamma_diff_loc, out=document_gamma_diff_loc)
        np.sum(document_gamma_diff_loc, axis=1, keepdims=True, out=gamma_asum_loc)
        np.sum(gamma_updates, axis=1, keepdims=True, out=gamma_sum_loc)
        gamma_asum_loc /= gamma_sum_loc

        document_gammas = gamma_updates

        has_finished = np.nonzero(gamma_asum_loc.flat <= gamma_threshold)[0]

        if has_finished.size != 0:
            final_gammas[finished_count:(finished_count + len(has_finished)), :] = document_gammas[has_finished, :]
            final_gammas_idx[finished_count:(finished_count + len(has_finished))] = rem_gammas_idx[has_finished]
            gamma_itr_cnt[finished_count:(finished_count + len(has_finished))] = itr_cnt

            is_rem_bool = gamma_asum_loc.flat > gamma_threshold

            is_rem_idx = np.nonzero(is_rem_bool)[0]
            rem_gammas_idx = rem_gammas_idx[is_rem_bool]
            finished_count += has_finished.size

            temp = np.zeros(ndoc, dtype=bool)
            temp[is_rem_bool] = True
            temp2 = np.arange(ndoc, dtype=int)
            temp2[temp] = np.arange(is_rem_idx.size, dtype=int)

            keep = temp[rel_idx]
            rel_idx = temp2[rel_idx[temp[rel_idx]]]

            idx_full = np.repeat(np.reshape(rel_idx, (-1, 1)), num_topics, axis=1)
            idx_full *= num_topics
            idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

            per_topic_log_densities2 = per_topic_log_densities2[keep, :]
            rel_counts = rel_counts[keep]
            nrec = per_topic_log_densities2.shape[0]
            ndoc = is_rem_idx.size

            log_density_gamma_loc = log_density_gamma[:nrec, :]
            posterior_sum_ll_loc = posterior_sum_ll[:nrec, :]
            gamma_sum_loc = gamma_sum[:ndoc, :]
            gamma_asum_loc = gamma_asum[:ndoc, :]
            document_gamma_diff_loc = document_gamma_diff_loc[:ndoc, :]

            document_gammas = document_gammas[is_rem_idx, :]
            document_gammas2 = document_gammas2[:ndoc, :]
            document_gammas3 = document_gammas3[:ndoc, :]

    #
    # Accumulate per-bag-sample
    #

    sidx = np.argsort(final_gammas_idx)
    final_gammas = final_gammas[sidx, :]
    gamma_itr_cnt = gamma_itr_cnt[sidx]

    digamma_gammas = digamma(final_gammas)
    temp2 = np.max(digamma_gammas, axis=1, keepdims=True)
    temp3 = np.exp(digamma_gammas - temp2)

    # per_topic_log_densities2  = per_topic_log_densities.copy()
    # per_topic_log_densities2 -= np.max(per_topic_log_densities2, axis=1, keepdims=True)
    # np.exp(per_topic_log_densities2, out=per_topic_log_densities2)

    np.multiply(per_topic_log_densities3, temp3[idx, :], out=log_density_gamma)
    np.sum(log_density_gamma, axis=1, keepdims=True, out=posterior_sum_ll)
    posterior_sum_ll /= np.reshape(counts, (-1, 1))
    log_density_gamma /= posterior_sum_ll

    effNs = log_density_gamma.sum(axis=0)

    idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
    idx_full *= num_topics
    idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

    gamma_updates = np.bincount(idx_full.flat, weights=log_density_gamma.flat)
    gamma_updates = np.reshape(gamma_updates, (-1, num_topics))
    gamma_updates += alpha_loc
    final_gammas = gamma_updates

    mlpf = digamma(final_gammas) - digamma(np.sum(final_gammas, axis=1, keepdims=True))

    return log_density_gamma, final_gammas, per_topic_log_densities
