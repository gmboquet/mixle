"""Labeled latent Dirichlet allocation for documents with observed label sets.

Data type: Tuple[Sequence[Tuple[T, float]], Sequence[int]]. Each observation is a document given as a bag of
(value, count) pairs together with a list of label (neighborhood) indices selecting rows of the 'alphas' matrix.

LabeledLDA extends latent Dirichlet allocation (see mixle.stats.latent.lda) by attaching a set of labels to each document.
The model keeps one Dirichlet parameter row alpha_a per label a (the 'alphas' matrix is num_alpha by nTopics).
A document with labels {a_1,...,a_m} draws its topic weights from a Dirichlet whose parameter is formed from
the alpha rows of its labels. Generation of a document of length N with L topics proceeds as:

        (1) Draw theta ~ Dirichlet(alpha_bar), where alpha_bar combines the alpha rows of the document labels.
        (2) Draw topic-counts z_1,...,z_L ~ Multinomial(N, theta).
        (3) For each topic l = 1,2,...,L draw z_l values from the topic distribution P_l() (data type T).

If included, 'len_dist' models the number of values N in a document, and 'set_dist' models the label sets
(both are used for sampling).

Estimation uses a mean-field variational EM (per-document gamma updates). The expected log topic weights
are aggregated per distinct label set, and the alpha rows are updated jointly by maximizing the coupled
objective in which each document's Dirichlet parameter is the average of its label rows (see
'update_alpha_coupled()'). When every document carries exactly one label the objective decouples and the
classic per-row fixed-point update ('update_alpha()') is used.

"""

import sys

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, gammaln

from mixle.engines.arithmetic import *
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.latent.lda import _lda_elbo_from_gamma, _lda_vi_fixed_point
from mixle.utils.deprecation import deprecated_alias
from mixle.utils.special import digammainv
from mixle.utils.vector import row_choice


class LabeledLDADistribution(SequenceEncodableProbabilityDistribution):
    """Labeled LDA model for documents with label sets.

    Compatible with data type Tuple[Sequence[Tuple[T, float]], Sequence[int]], where T is the data type of
    the topic distributions.
    """

    def __init__(self, topics, alphas, set_dist=None, len_dist=None, gamma_threshold=1.0e-8, max_gamma_iter=100):
        """Create a labeled LDA distribution.

        Args:
                topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions, all having
                        data type T.
                alphas (Union[Sequence[float], np.ndarray]): Per-label Dirichlet parameters, reshaped to a
                        2-d array with one row per label and one column per topic.
                set_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the label sets
                        of documents. Required for sampling.
                len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the number of
                        values in a document. Required for sampling.
                gamma_threshold (float): Convergence threshold for the per-document variational gamma updates.

        Attributes:
                topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions.
                nTopics (int): Number of topic distributions.
                alphas (np.ndarray): 2-d array of per-label Dirichlet parameters (num_alpha by nTopics).
                num_alpha (int): Number of label rows in 'alphas'.
                len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for document lengths.
                set_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for label sets.
                gamma_threshold (float): Convergence threshold for the variational gamma updates.

        """
        self.topics = topics
        self.nTopics = len(topics)
        self.alphas = np.reshape(np.asarray(alphas), (-1, self.nTopics))
        self.num_alpha = self.alphas.shape[0]
        self.len_dist = len_dist
        self.set_dist = set_dist
        self.gamma_threshold = gamma_threshold
        self.max_gamma_iter = int(max_gamma_iter)

    def __str__(self):
        """Return a constructor-style representation of the distribution."""
        return "LabeledLDADistribution([%s], [%s])" % (
            ",".join([str(u) for u in self.topics]),
            ",".join(map(str, self.alphas.flatten())),
        )

    def density(self, x):
        """Returns the density (exp of the variational lower bound) for a labeled document x.

        Args:
                x (Tuple[Sequence[Tuple[T, float]], Sequence[int]]): Document as (value, count) pairs and a list
                        of label indices.

        Returns:
                Density value for the document x.

        """
        return exp(self.log_density(x))

    def density_semantics(self):
        """Return density semantics for the labeled-LDA variational bound."""
        from mixle.stats.compute.pdist import DensitySemantics

        return DensitySemantics.LOWER_BOUND  # per-document variational ELBO, not the exact marginal

    def log_density(self, x):
        """Returns the variational lower bound (ELBO) on the log-density for a labeled document x.

        Args:
                x (Tuple[Sequence[Tuple[T, float]], Sequence[int]]): Document as (value, count) pairs and a list
                        of label indices.

        Returns:
                Lower bound on the log-density of the document x.

        """
        return self.seq_log_density(self.dist_to_encoder().seq_encode([x]))[0]

    def seq_log_density(self, x):
        """Vectorized evaluation of the variational lower bound (ELBO) for encoded documents.

        Arg 'x' is the output of 'LabeledLDADataEncoder.seq_encode()'.

        Args:
                x: Encoded sequence of iid LabeledLDA observations (see LabeledLDADataEncoder.seq_encode()).

        Returns:
                Numpy array with one lower-bound value per encoded document.

        """

        num_topics = self.nTopics
        num_documents, idx, counts, _, enc_data, _, _, _ = x

        log_density_gamma, document_gammas, document_alphas, per_topic_log_densities = seq_posterior(self, x)

        # LabeledLDA's per-document prior 'document_alphas' is the 2-d coupled mean of label rows;
        # the shared host ELBO handles both the 1-d (plain LDA) and 2-d (labeled) alpha. LabeledLDA
        # has no length or label-set term to add.
        return _lda_elbo_from_gamma(
            document_alphas, idx, counts, num_topics, log_density_gamma, document_gammas, per_topic_log_densities
        )

    @deprecated_alias("dist_to_encoder().seq_encode()", since="0.8.0", removed_in="0.10.0")
    def seq_encode(self, x):
        """Deprecated alias for ``dist_to_encoder().seq_encode()``: encode iid LabeledLDA observations.

        Use ``dist_to_encoder()`` and ``LabeledLDADataEncoder.seq_encode()`` instead.

        Args:
                x (Sequence[Tuple[Sequence[Tuple[T, float]], Sequence[int]]]): Sequence of labeled documents.

        Returns:
                Encoded sequence (see LabeledLDADataEncoder.seq_encode()).

        """
        return self.dist_to_encoder().seq_encode(x)

    def seq_component_log_density(self, x):
        """Vectorized per-topic log-density evaluation for encoded documents.

        Args:
                x: Encoded sequence of iid LabeledLDA observations (see LabeledLDADataEncoder.seq_encode()).

        Returns:
                2-d numpy array (num_documents by nTopics) of per-topic document log-densities.

        """

        num_topics = self.nTopics
        num_documents, idx, counts, _, enc_data, _, _, _ = x

        ll_mat = np.zeros((len(idx), self.nTopics))
        ll_mat.fill(-np.inf)

        rv = np.zeros((num_documents, self.nTopics))
        rv.fill(-np.inf)

        for i in range(num_topics):
            ll_mat[:, i] = self.topics[i].seq_log_density(enc_data)
            rv[:, i] = np.bincount(idx, weights=ll_mat[:, i] * counts, minlength=num_documents)

        return rv

    def seq_posterior(self, x):
        """Vectorized posterior topic weights for encoded documents.

        Args:
                x: Encoded sequence of iid LabeledLDA observations (see LabeledLDADataEncoder.seq_encode()).

        Returns:
                2-d numpy array (num_documents by nTopics) of normalized posterior topic weights.

        """
        log_density_gamma, document_gammas, document_alphas, per_topic_log_densities = seq_posterior(self, x)

        document_gammas /= document_gammas.sum(axis=1, keepdims=True)

        return document_gammas

    def compute_capabilities(self):
        """Return backend capability metadata for this concrete LabeledLDA instance."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(
            engine_ready=intersect_engine_ready(tuple(self.topics)), kernel_status="generic_latent"
        )

    def _backend_seq_posterior(self, x, engine):
        """Engine-resident LabeledLDA variational posterior (numpy or torch).

        Returns (log_density_gamma, document_gammas, alphas_loc, per_topic_log_densities), mirroring
        the host module-level seq_posterior but with a plain fixed-point loop on the active engine.
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        num_documents, idx, counts, gammas, enc_data, nbx, nbcnt, nbidx = x
        num_topics = self.nTopics
        alphas = engine.asarray(self.alphas)

        idx_np = np.asarray(idx, dtype=np.int64)
        idx_e = engine.asarray(idx_np)
        counts_e = engine.asarray(np.asarray(counts, dtype=np.float64))
        idx_full_np = (idx_np[:, None] * num_topics + np.arange(num_topics, dtype=np.int64)).reshape(-1)
        idx_full = engine.asarray(idx_full_np)

        nbx_e = engine.asarray(np.asarray(nbx, dtype=np.int64))
        nbidx_e = engine.asarray(np.asarray(nbidx, dtype=np.int64))
        # per-document mean of the label alphas (coupled prior)
        ddd = engine.index_add(
            engine.zeros(num_documents), nbidx_e, engine.asarray(np.ones(len(np.asarray(nbidx)), dtype=np.float64))
        )
        alphas_loc = engine.index_add(engine.zeros((num_documents, num_topics)), nbidx_e, alphas[nbx_e, :])
        alphas_loc = alphas_loc / ddd.reshape((-1, 1))

        per_topic_log_densities = engine.stack(
            [backend_seq_log_density(self.topics[i], enc_data, engine) for i in range(num_topics)], axis=1
        )
        centered = per_topic_log_densities - engine.max(per_topic_log_densities, axis=1).reshape((-1, 1))
        per_topic_weights = engine.exp(centered)

        if gammas is None:
            init_counts = engine.index_add(
                engine.zeros(num_documents * num_topics),
                idx_full,
                engine.asarray(np.ones(len(idx_full_np), dtype=np.float64)),
            ).reshape((num_documents, num_topics))
            document_gammas = alphas_loc + init_counts / float(num_topics)
        else:
            document_gammas = engine.asarray(gammas)

        for _ in range(self.max_gamma_iter):
            dg = engine.digamma(document_gammas)
            gw = engine.exp(dg - engine.max(dg, axis=1).reshape((-1, 1)))
            row_weights = per_topic_weights * gw[idx_e, :]
            row_sum = engine.sum(row_weights, axis=1).reshape((-1, 1))
            log_density_gamma = row_weights / row_sum * counts_e.reshape((-1, 1))
            gamma_updates = engine.index_add(
                engine.zeros(num_documents * num_topics), idx_full, log_density_gamma.reshape((-1,))
            ).reshape((num_documents, num_topics))
            gamma_updates = gamma_updates + alphas_loc
            rel_diff = engine.sum(engine.abs(document_gammas - gamma_updates), axis=1) / engine.sum(
                gamma_updates, axis=1
            )
            document_gammas = gamma_updates
            if float(np.max(engine.to_numpy(rel_diff))) <= self.gamma_threshold:
                break

        # final responsibilities consistent with the converged gammas
        dg = engine.digamma(document_gammas)
        gw = engine.exp(dg - engine.max(dg, axis=1).reshape((-1, 1)))
        row_weights = per_topic_weights * gw[idx_e, :]
        row_sum = engine.sum(row_weights, axis=1).reshape((-1, 1))
        log_density_gamma = row_weights / row_sum * counts_e.reshape((-1, 1))

        return log_density_gamma, document_gammas, alphas_loc, per_topic_log_densities

    def backend_seq_log_density(self, x, engine):
        """Backend-neutral LabeledLDA variational lower-bound (ELBO) scoring."""
        num_documents, idx, counts, _, enc_data, nbx, nbcnt, nbidx = x
        num_topics = self.nTopics
        idx_np = np.asarray(idx, dtype=np.int64)
        idx_e = engine.asarray(idx_np)
        counts_e = engine.asarray(np.asarray(counts, dtype=np.float64))
        idx_full = engine.asarray((idx_np[:, None] * num_topics + np.arange(num_topics, dtype=np.int64)).reshape(-1))

        log_density_gamma, document_gammas, document_alphas, per_topic_log_densities = self._backend_seq_posterior(
            x, engine
        )

        tiny = sys.float_info.min
        bad = engine.isnan(log_density_gamma) | engine.isinf(log_density_gamma) | (log_density_gamma <= 0)
        log_density_gamma = engine.where(bad, engine.asarray(tiny), log_density_gamma)
        bad_d = engine.isnan(document_gammas) | engine.isinf(document_gammas)
        document_gammas = engine.where(bad_d, engine.asarray(tiny), document_gammas)

        gamma_sum = engine.sum(document_gammas, axis=1).reshape((-1, 1))
        elob0 = engine.digamma(document_gammas) - engine.digamma(gamma_sum)
        elob1 = elob0[idx_e, :]
        elob2 = log_density_gamma * (
            elob1 + per_topic_log_densities - engine.log(log_density_gamma) + engine.log(counts_e.reshape((-1, 1)))
        )
        elob3 = engine.sum(elob0 * ((document_alphas - 1.0) - (document_gammas - 1.0)), axis=1)
        elob4 = engine.index_add(engine.zeros(num_documents * num_topics), idx_full, elob2.reshape((-1,))).reshape(
            (num_documents, num_topics)
        )
        elob5 = engine.sum(elob4, axis=1)
        elob6 = engine.sum(engine.gammaln(document_gammas), axis=1) - engine.gammaln(
            engine.sum(document_gammas, axis=1)
        )
        alpha_sum = engine.sum(document_alphas, axis=1)
        elob7 = engine.gammaln(alpha_sum) - engine.sum(engine.gammaln(document_alphas), axis=1)
        return elob3 + elob5 + elob6 + elob7

    def enumerator(self) -> DistributionEnumerator:
        """Not supported: LabeledLDA's ``log_density`` is a variational lower bound, not the true marginal.

        Each document's score is an ELBO obtained by running per-document variational inference
        (``seq_posterior``), not the exact marginal probability, so enumerating "in descending
        probability order" is ill-defined -- the ELBO ranking need not match the true-probability
        ranking, and the per-document optimum isn't a closed-form density over a countable support.
        Use :meth:`sampler` and the (approximate) ``log_density`` directly.
        """
        raise EnumerationError(
            self,
            reason="LabeledLDA log_density is a per-document variational lower bound (ELBO), not an exact "
            "marginal probability, so there is no well-defined descending-probability enumeration",
        )

    def sampler(self, seed=None):
        """Create a sampler for labeled LDA documents.

        Note: Requires 'set_dist' and 'len_dist' to be set.

        Args:
                seed (Optional[int]): Set seed for random sampling.

        Returns:
                LabeledLDASampler: Sampler bound to this distribution.

        """
        return LabeledLDASampler(self, seed)

    def estimator(self, pseudo_count=None):
        """Create an estimator initialized from this labeled LDA distribution.

        Args:
                pseudo_count (Optional[float]): Used to re-weight sufficient statistics in estimation.

        Returns:
                LabeledLDAEstimator: Estimator configured with matching topic estimators.

        """
        estimators = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return LabeledLDAEstimator(
            estimators,
            num_alphas=self.num_alpha,
            gamma_threshold=self.gamma_threshold,
            max_gamma_iter=self.max_gamma_iter,
        )

    def dist_to_encoder(self):
        """Return a data encoder for iid labeled LDA observations."""
        return LabeledLDADataEncoder(encoder=self.topics[0].dist_to_encoder())


class LabeledLDASampler(DistributionSampler):
    """Sample labeled documents from a LabeledLDA distribution.

    Requires 'dist.set_dist' (label sets) and 'dist.len_dist' (document lengths) to be set.
    """

    def __init__(self, dist, seed=None):
        """Create a sampler for a labeled LDA distribution.

        Args:
                dist (LabeledLDADistribution): LabeledLDADistribution instance to sample from.
                seed (Optional[int]): Set seed on random number generator for sampling.

        Attributes:
                rng (RandomState): Random state initialized from ``seed`` when supplied.
                dist (LabeledLDADistribution): LabeledLDADistribution instance to sample from.
                nTopics (int): Number of topic distributions.
                compSamplers (List[DistributionSampler]): Samplers for the topic distributions.
                len_dist (DistributionSampler): Sampler for document lengths.
                set_dist (DistributionSampler): Sampler for label sets.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.nTopics = dist.nTopics
        self.compSamplers = [self.dist.topics[i].sampler(seed=self.rng.randint(maxint)) for i in range(dist.nTopics)]
        # self.dirichletSampler = DirichletDistribution(dist.alpha).sampler(self.rng.randint(maxint))
        self.len_dist = self.dist.len_dist.sampler(seed=self.rng.randint(maxint))
        self.set_dist = self.dist.set_dist.sampler(seed=self.rng.randint(maxint))

    def sample(self, size=None):
        """Draw iid labeled documents from the LabeledLDA model.

        If size is None, a single Tuple[List[T], List[int]] is returned containing the sampled document
        values and its labels. If size > 0, a list of 'size' such Tuples is returned.

        Args:
                size (Optional[int]): Number of iid labeled documents to sample.

        Returns:
                Tuple[List[T], List[int]] or a List of such Tuples depending on arg size.

        """

        if size is None:
            nodes = []
            while len(nodes) == 0:
                nodes = self.set_dist.sample()
            n = self.len_dist.sample()
            nTopics = self.nTopics
            alpha_loc = self.dist.alphas[np.asarray(nodes), :].mean(axis=0)
            weights = self.rng.dirichlet(alpha_loc)
            # topics    = self.rng.choice(range(0, nTopics), size=n, replace=True, p=weights)
            # rv        = [None]*n
            # for i in range(n):
            # rv[i] = self.compSamplers[topics[i]].sample()
            #
            topic_counts = self.rng.multinomial(n, pvals=weights)
            topics = []
            rv = []
            for i in np.flatnonzero(topic_counts):
                topics.extend([i] * topic_counts[i])
                rv.extend(self.compSamplers[i].sample(size=topic_counts[i]))

            return (rv, nodes)

        else:
            return [self.sample() for i in range(size)]


class LabeledLDALabelSetStats:
    """Sufficient statistics for the coupled alpha update, grouped by distinct document label set.

    Maps each distinct label set S (a sorted tuple of label indices, duplicates preserved) to a pair
    [n_S, m_S], where n_S is the total weight of the documents carrying label set S and m_S is the
    weighted sum of the per-document expected log topic weights E[log theta] (a vector with one entry
    per topic) over those documents.
    """

    def __init__(self, stats=None):
        """Create grouped label-set statistics for labeled LDA.

        Args:
                stats (Optional[Dict[Tuple[int, ...], List]]): Optional mapping from label-set tuples to
                        [n_S, m_S] pairs. Defaults to an empty mapping.

        """
        self.stats = dict() if stats is None else stats

    def add(self, label_set, weight, sum_log_p):
        """Accumulate document weight and summed expected log topic weights for one label set.

        Args:
                label_set (Tuple[int, ...]): Sorted tuple of label indices.
                weight (float): Total document weight to add for the label set.
                sum_log_p (np.ndarray): Weighted sum of per-document E[log theta] vectors to add.

        Returns:
                None.

        """
        entry = self.stats.get(label_set)
        if entry is None:
            self.stats[label_set] = [float(weight), np.array(sum_log_p, dtype=float)]
        else:
            entry[0] += float(weight)
            entry[1] += sum_log_p

    def combine(self, other):
        """Merge the statistics of another LabeledLDALabelSetStats instance into this instance.

        Args:
                other (LabeledLDALabelSetStats): Statistics to merge in (left unmodified).

        Returns:
                LabeledLDALabelSetStats object (self).

        """
        for label_set, entry in other.stats.items():
            self.add(label_set, entry[0], entry[1])
        return self

    def copy(self):
        """Returns a deep copy of the LabeledLDALabelSetStats instance."""
        return LabeledLDALabelSetStats({k: [v[0], v[1].copy()] for k, v in self.stats.items()})

    def arrays(self):
        """Returns the statistics as parallel arrays in sorted label-set order.

        Returns:
                Tuple of the sorted label-set tuples (List[Tuple[int, ...]]), the per-set weights n_S
                (1-d numpy array), and the per-set summed expected log topic weights m_S (2-d numpy array
                with one row per label set).

        """
        label_sets = sorted(self.stats.keys())
        if len(label_sets) == 0:
            return label_sets, np.zeros(0), np.zeros((0, 0))
        n = np.asarray([self.stats[k][0] for k in label_sets], dtype=float)
        m = np.asarray([self.stats[k][1] for k in label_sets], dtype=float)
        return label_sets, n, m

    def __array__(self, dtype=None, copy=None):
        """Canonical array form: one row [n_S, m_S] per label set in sorted order (for comparisons)."""
        label_sets, n, m = self.arrays()
        if len(label_sets) == 0:
            rv = np.zeros((0, 0))
        else:
            rv = np.concatenate((np.reshape(n, (-1, 1)), m), axis=1)
        if dtype is not None:
            rv = rv.astype(dtype)
        return rv

    def __str__(self):
        """Return a compact representation of the label-set statistics."""
        return "LabeledLDALabelSetStats(%s)" % (str(self.stats))


def doc_label_sets(nbx, nbcnt):
    """Returns the sorted label-set tuple of each encoded document.

    Args:
            nbx (np.ndarray): Flattened array of label indices over all documents (document-contiguous).
            nbcnt (np.ndarray): Number of labels for each document.

    Returns:
            List of sorted label-index tuples, one per document.

    """
    rv = []
    pos = 0
    for c in nbcnt:
        rv.append(tuple(sorted(int(u) for u in nbx[pos : (pos + c)])))
        pos += c
    return rv


class LabeledLDAEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Aggregate sufficient statistics from labeled LDA documents.

    Tracks per-label-set expected log topic weights and document counts ('set_stats'), per-label
    weighted document counts ('doc_counts'), label-allocated topic counts ('topic_counts'), and the topic
    distribution accumulators.
    """

    def __init__(self, accumulators, num_alphas, keys=(None, None), prev_alpha=None):
        """Create an accumulator for labeled LDA variational EM statistics.

        Args:
                accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the topic
                        distributions.
                num_alphas (int): Number of label rows in the alphas matrix.
                keys (Tuple[Optional[str], Optional[str]]): Optional keys for alpha statistics and topic accumulators.
                prev_alpha (Optional[np.ndarray]): Optional previous alphas matrix (num_alphas by num_topics).

        Attributes:
                accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the topic
                        distributions.
                num_topics (int): Number of topic distributions.
                num_alphas (int): Number of label rows in the alphas matrix.
                set_stats (LabeledLDALabelSetStats): Per-label-set aggregated expected log topic weights and counts.
                doc_counts (Union[float, np.ndarray]): Per-label weighted document counts.
                topic_counts (np.ndarray): Label-allocated weighted topic counts.
                prev_alpha (Optional[np.ndarray]): Previous alphas matrix.
                alpha_key (Optional[str]): Key for alpha statistics.
                topics_key (Optional[str]): Key for topic accumulators.

                _init_rng (bool): True if random states have been initialized for seq_initialize.
                _rng_theta (Optional[RandomState]): RandomState for topic weight draws.
                _rng_idx (Optional[RandomState]): RandomState for per-value topic assignment draws.
                _rng_w (Optional[RandomState]): RandomState for per-value weight smoothing draws.
                _rng_topics (Optional[List[RandomState]]): Random states for the topic accumulators.

        """

        num_topics = len(accumulators)

        self.accumulators = accumulators
        self.num_topics = len(accumulators)
        self.num_alphas = num_alphas
        self.set_stats = LabeledLDALabelSetStats()
        self.doc_counts = 0.0
        self.topic_counts = np.zeros((num_alphas, num_topics))
        self.prev_alpha = prev_alpha

        self.alpha_key = keys[0]
        self.topics_key = keys[1]

        # Per-document variational lower bound (ELBO) accumulated as a byproduct of the E-step,
        # only when _track_ll is enabled. Equals seq_log_density_sum(enc, dist)[1] and is consumed
        # by the fused-EM fast path in optimize(reuse_estep_ll=True); not part of value(). Off by
        # default so the standard path pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        # Initialized lazily for seq_initialize consistency.
        self._init_rng = False
        self._rng_theta = None
        self._rng_idx = None
        self._rng_w = None
        self._rng_topics = None

    def update(self, x, weight, estimate):
        """Update sufficient statistics of the accumulator with one labeled document.

        Note: Not efficient. Encodes a singleton batch and delegates to 'seq_update()'.

        Args:
                x (Tuple[Sequence[Tuple[T, float]], Sequence[int]]): Document as (value, count) pairs and a list
                        of label indices.
                weight (float): Weight for observation.
                estimate (LabeledLDADistribution): Previous estimate of the LabeledLDA model.

        Returns:
                None.

        """
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng):
        """Set RandomState member variables used by seq_initialize.

        Args:
                rng (RandomState): Random state used to seed the accumulator initialization streams.

        Returns:
                None.

        """
        seeds = rng.randint(maxrandint, size=3 + self.num_topics)
        self._rng_theta = RandomState(seed=seeds[0])
        self._rng_idx = RandomState(seed=seeds[1])
        self._rng_w = RandomState(seed=seeds[2])
        self._rng_topics = [RandomState(seed=seeds[3 + j]) for j in range(self.num_topics)]
        self._init_rng = True

    def _accumulate_set_stats(self, doc_log_p, weights, nbx, nbcnt):
        """Accumulate per-label-set statistics from per-document expected log topic weights.

        Groups the documents by their (sorted) label set and adds the weighted document counts and the
        weighted sums of 'doc_log_p' rows to 'set_stats'.

        Args:
                doc_log_p (np.ndarray): Per-document expected log topic weights (num_documents by num_topics).
                weights (np.ndarray): Numpy array of weights for the documents.
                nbx (np.ndarray): Flattened array of label indices over all documents.
                nbcnt (np.ndarray): Number of labels for each document.

        Returns:
                None.

        """

        doc_sets = doc_label_sets(nbx, nbcnt)

        set_index = dict()
        set_ids = np.zeros(len(doc_sets), dtype=int)
        for d, label_set in enumerate(doc_sets):
            set_ids[d] = set_index.setdefault(label_set, len(set_index))

        num_sets = len(set_index)
        set_n = np.bincount(set_ids, weights=weights, minlength=num_sets)
        set_m = np.zeros((num_sets, self.num_topics))
        for i in range(self.num_topics):
            set_m[:, i] = np.bincount(set_ids, weights=doc_log_p[:, i] * weights, minlength=num_sets)

        for label_set, j in set_index.items():
            self.set_stats.add(label_set, set_n[j], set_m[j, :])

    def initialize(self, x, weight, rng):
        """Initialize the accumulator with a single labeled document.

        Draws document topic weights from a Dirichlet formed from the label rows of 'prev_alpha', randomly
        assigns each document value to a topic, and initializes the topic accumulators accordingly.

        Args:
                x (Tuple[Sequence[Tuple[T, float]], Sequence[int]]): Document as (value, count) pairs and a list
                        of label indices.
                weight (float): Weight for observation.
                rng (RandomState): RandomState for random topic assignments.

        Returns:
                None.

        """

        if self.prev_alpha is None:
            self.prev_alpha = np.ones((self.num_alphas, self.num_topics))

        xdoc = x[0]
        xnbh = x[1]
        aloc = self.prev_alpha[xnbh, :].mean(axis=0)

        theta = rng.dirichlet(aloc)

        idx_list = rng.choice(self.num_topics, size=len(xdoc), replace=True, p=theta)

        self.set_stats.add(tuple(sorted(int(u) for u in xnbh)), weight, weight * np.log(theta))
        self.doc_counts += weight

        for i in range(len(xdoc)):
            idx = idx_list[i]
            ww_v = -np.log(rng.rand(self.num_topics))
            ww_v[idx] += 1
            ww_v *= weight * xdoc[i][1] / ww_v.sum()
            for j in range(self.num_topics):
                # w = weight*x[i][1] if idx == j else 0.0
                w = ww_v[j]
                self.topic_counts[xnbh, j] += w / len(xnbh)
                self.accumulators[j].initialize(xdoc[i][0], w, rng)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialization of the accumulator from an encoded sequence of labeled documents.

        Mirrors 'initialize()': per-document topic weights are drawn from a Dirichlet formed from the label
        rows of 'prev_alpha', each document value is randomly assigned to a topic, and the topic accumulators
        are initialized with smoothed per-value weights.

        Args:
                x: Encoded sequence of iid LabeledLDA observations (see LabeledLDADataEncoder.seq_encode()).
                weights (np.ndarray): Numpy array of weights for the documents.
                rng (RandomState): Random state used to seed member random states on first call.

        Returns:
                None.

        """

        num_documents, idx, counts, old_gammas, enc_data, nbx, nbcnt, nbidx = x

        if not self._init_rng:
            self._rng_initialize(rng)

        if self.prev_alpha is None:
            self.prev_alpha = np.ones((self.num_alphas, self.num_topics))

        # Per-document Dirichlet parameter: average of prev_alpha rows over the document labels.
        aloc = np.zeros((num_documents, self.num_topics))
        for j in range(self.num_topics):
            aloc[:, j] = np.bincount(nbidx, weights=self.prev_alpha[nbx, j], minlength=num_documents)
        nbcnt_loc = np.maximum(nbcnt.astype(float), 1.0)
        aloc /= np.reshape(nbcnt_loc, (-1, 1))

        # Per-document topic weights theta ~ Dirichlet(aloc) via normalized gamma draws.
        theta = self._rng_theta.gamma(shape=aloc)
        theta_sum = theta.sum(axis=1, keepdims=True)
        theta_sum[theta_sum == 0] = 1.0
        theta /= theta_sum

        idx_list = row_choice(p_mat=theta[idx, :], rng=self._rng_idx)

        self._accumulate_set_stats(np.log(theta), weights, nbx, nbcnt)
        self.doc_counts += np.sum(weights)

        ww_v = -np.log(self._rng_w.rand(len(idx) * self.num_topics))
        ww_v[idx_list + np.arange(0, len(ww_v), self.num_topics)] += 1
        ww_v = np.reshape(ww_v, (-1, self.num_topics))
        ww_v /= ww_v.sum(axis=1, keepdims=True)
        ww_v *= np.reshape(weights[idx] * counts, (-1, 1))

        for j in range(self.num_topics):
            doc_w = np.bincount(idx, weights=ww_v[:, j], minlength=num_documents)
            label_weight = doc_w[nbidx] / np.maximum(nbcnt[nbidx].astype(float), 1.0)
            self.topic_counts[:, j] += np.bincount(nbx, weights=label_weight, minlength=self.num_alphas)
            self.accumulators[j].seq_initialize(enc_data, ww_v[:, j], self._rng_topics[j])

    def seq_update(self, x, weights, estimate):
        """Vectorized update of the accumulator from an encoded sequence of labeled documents.

        Computes the variational posterior for each document under 'estimate' and aggregates per-label-set
        expected log topic weights, per-label document counts, label-allocated topic counts, and the
        topic accumulator statistics.

        Args:
                x: Encoded sequence of iid LabeledLDA observations (see LabeledLDADataEncoder.seq_encode()).
                weights (np.ndarray): Numpy array of weights for the documents.
                estimate (LabeledLDADistribution): Previous EM estimate of the LabeledLDA model.

        Returns:
                None.

        """

        num_alphas = self.num_alphas
        num_topics = self.num_topics

        num_documents, idx, counts, old_gammas, enc_data, nbx, nbcnt, nbidx = x
        # num_documents, idx, counts, old_gammas, enc_data = x
        log_density_gamma, final_gammas, doc_alphas, per_topic_log_densities = seq_posterior(estimate, x)
        weighted_topic_counts = log_density_gamma * np.reshape(weights[idx], (-1, 1))

        mlpf = digamma(final_gammas) - digamma(np.sum(final_gammas, axis=1, keepdims=True))

        nbh_cnt = np.reshape(np.bincount(nbx, weights=weights[nbidx], minlength=num_alphas), (-1, 1))
        nbh_tcnt = np.zeros((num_alphas, num_topics))

        for i in range(num_topics):
            self.accumulators[i].seq_update(enc_data, weighted_topic_counts[:, i], estimate.topics[i])

            doc_tcnt = np.bincount(idx, weights=log_density_gamma[:, i], minlength=num_documents)
            label_weight = doc_tcnt[nbidx] * weights[nbidx] / np.maximum(nbcnt[nbidx].astype(float), 1.0)
            nbh_tcnt[:, i] = np.bincount(nbx, weights=label_weight, minlength=num_alphas)

        self._accumulate_set_stats(mlpf, weights, nbx, nbcnt)
        self.doc_counts += nbh_cnt
        self.topic_counts += nbh_tcnt
        self.prev_alpha = estimate.alphas

        # Fused-EM fast path: recover the per-document ELBO that estimate.seq_log_density would
        # return, reusing the variational quantities the E-step already produced -- no second
        # variational loop and no re-scoring of topics. Mirrors LabeledLDADistribution.seq_log_density
        # exactly (LabeledLDA's ELBO has no length/label-set term). Gated; standard path untouched.
        if self._track_ll:
            elob = _lda_elbo_from_gamma(
                doc_alphas, idx, counts, num_topics, log_density_gamma, final_gammas, per_topic_log_densities
            )
            self._seq_ll += float(np.dot(weights, elob))

        # return num_documents, idx, counts, final_gammas, enc_data

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident LabeledLDA E-step (numpy or torch).

        Runs the variational posterior and the per-label-set / topic-count aggregations on the active
        engine, feeding engine-computed responsibilities to the topic accumulators. Mirrors seq_update.
        """
        num_alphas = self.num_alphas
        num_topics = self.num_topics
        num_documents, idx, counts, old_gammas, enc_data, nbx, nbcnt, nbidx = x

        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        idx_np = np.asarray(idx, dtype=np.int64)
        nbx_np = np.asarray(nbx, dtype=np.int64)
        nbidx_np = np.asarray(nbidx, dtype=np.int64)

        log_density_gamma, final_gammas, doc_alphas, per_topic_log_densities = estimate._backend_seq_posterior(
            x, engine
        )

        idx_e = engine.asarray(idx_np)
        nbx_e = engine.asarray(nbx_np)
        nbidx_e = engine.asarray(nbidx_np)
        w_idx = engine.asarray(weights_np[idx_np]).reshape((-1, 1))
        weighted_topic_counts = log_density_gamma * w_idx

        gamma_sum = engine.sum(final_gammas, axis=1).reshape((-1, 1))
        mlpf = engine.digamma(final_gammas) - engine.digamma(gamma_sum)

        nbh_cnt = engine.index_add(engine.zeros(num_alphas), nbx_e, engine.asarray(weights_np[nbidx_np]))
        nbcnt_doc_e = engine.asarray(np.maximum(np.asarray(nbcnt)[nbidx_np].astype(np.float64), 1.0))
        w_nbidx_e = engine.asarray(weights_np[nbidx_np])
        nbh_tcols = []
        for i in range(num_topics):
            doc_tcnt = engine.index_add(engine.zeros(num_documents), idx_e, log_density_gamma[:, i])
            label_weight = doc_tcnt[nbidx_e] * w_nbidx_e / nbcnt_doc_e
            nbh_tcols.append(engine.index_add(engine.zeros(num_alphas), nbx_e, label_weight))
        nbh_tcnt = engine.stack(nbh_tcols, axis=1)

        wtc_np = np.asarray(engine.to_numpy(weighted_topic_counts))
        for i in range(num_topics):
            self.accumulators[i].seq_update(enc_data, wtc_np[:, i], estimate.topics[i])

        self._accumulate_set_stats(np.asarray(engine.to_numpy(mlpf)), weights_np, nbx, nbcnt)
        self.doc_counts += np.asarray(engine.to_numpy(nbh_cnt)).reshape((-1, 1))
        self.topic_counts += np.asarray(engine.to_numpy(nbh_tcnt))
        self.prev_alpha = estimate.alphas

    def combine(self, suff_stat):
        """Combine the sufficient statistics of the accumulator with the suff_stat arg.

        Sufficient statistics in suff_stat are a Tuple containing:
                suff_stat[0] (Optional[np.ndarray]): Previous alphas matrix.
                suff_stat[1] (LabeledLDALabelSetStats): Per-label-set expected log topic weights and counts.
                suff_stat[2] (Union[float, np.ndarray]): Per-label weighted document counts.
                suff_stat[3] (np.ndarray): Label-allocated weighted topic counts.
                suff_stat[4] (Sequence): Topic distribution accumulator values.

        Args:
                suff_stat: See above for details.

        Returns:
                LabeledLDAEstimatorAccumulator object.

        """

        prev_alpha, set_stats, doc_counts, topic_counts, topic_suff_stats = suff_stat

        if self.prev_alpha is None:
            self.prev_alpha = prev_alpha

        self.set_stats.combine(set_stats)
        self.doc_counts += doc_counts
        self.topic_counts += topic_counts

        for i in range(self.num_topics):
            self.accumulators[i].combine(topic_suff_stats[i])

        return self

    def value(self):
        """Returns sufficient statistics of the accumulator instance.

        Returns:
                Tuple of previous alphas matrix, per-label-set statistics (LabeledLDALabelSetStats), per-label document
                counts, label-allocated topic counts, and the topic accumulator values.

        """
        return (
            self.prev_alpha,
            self.set_stats,
            self.doc_counts,
            self.topic_counts,
            [u.value() for u in self.accumulators],
        )

    def from_value(self, x):
        """Restore accumulator state from a sufficient-statistics tuple.

        Args:
                x: Tuple of sufficient statistics (see 'value()' for details).

        Returns:
                LabeledLDAEstimatorAccumulator: This accumulator after restoration.

        """

        prev_alpha, set_stats, doc_counts, topic_counts, topic_suff_stats = x

        self.prev_alpha = prev_alpha
        self.set_stats = set_stats
        self.doc_counts = doc_counts
        self.topic_counts = topic_counts
        self.accumulators = [self.accumulators[i].from_value(topic_suff_stats[i]) for i in range(self.num_topics)]

        return self

    def key_merge(self, stats_dict):
        """Merge this accumulator into keyed sufficient statistics.

        Args:
                stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
                None.

        """

        if self.alpha_key is not None:
            if self.alpha_key in stats_dict:
                p_sol, p_doc, p_pa = stats_dict[self.alpha_key]

                prev_alpha = self.prev_alpha if self.prev_alpha is not None else p_pa
                stats_dict[self.alpha_key] = (self.set_stats.copy().combine(p_sol), self.doc_counts + p_doc, prev_alpha)

            else:
                stats_dict[self.alpha_key] = (self.set_stats, self.doc_counts, self.prev_alpha)

        if self.topics_key is not None:
            if self.topics_key in stats_dict:
                acc = stats_dict[self.topics_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.topics_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics from matching keyed values.

        Args:
                stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
                None.

        """

        if self.alpha_key is not None:
            if self.alpha_key in stats_dict:
                p_sol, p_doc, p_pa = stats_dict[self.alpha_key]
                self.prev_alpha = p_pa
                self.set_stats = p_sol
                self.doc_counts = p_doc

        if self.topics_key is not None:
            if self.topics_key in stats_dict:
                acc = stats_dict[self.topics_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self):
        """Return a data encoder built from the topic accumulators."""
        return LabeledLDADataEncoder(encoder=self.accumulators[0].acc_to_encoder())


class LabeledLDAEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for labeled LDA estimator accumulators."""

    def __init__(self, factories, dim, num_alphas, keys, prev_alpha):
        """Create a factory for labeled LDA estimator accumulators.

        Args:
                factories (Sequence[StatisticAccumulatorFactory]): Factories for the topic accumulators.
                dim (int): Number of topic distributions.
                num_alphas (int): Number of label rows in the alphas matrix.
                keys (Tuple[Optional[str], Optional[str]]): Optional keys for alpha statistics and topic accumulators.
                prev_alpha (Optional[np.ndarray]): Optional previous alphas matrix.

        """
        self.factories = factories
        self.dim = dim
        self.keys = keys
        self.num_alphas = num_alphas
        self.prev_alpha = prev_alpha

    def make(self):
        """Return a new labeled LDA estimator accumulator."""
        return LabeledLDAEstimatorAccumulator(
            [self.factories[i].make() for i in range(self.dim)], self.num_alphas, self.keys, self.prev_alpha
        )


class LabeledLDAEstimator(ParameterEstimator):
    """Estimate labeled LDA distributions from aggregated sufficient statistics."""

    def __init__(
        self,
        estimators,
        num_alphas,
        suff_stat=None,
        pseudo_count=None,
        keys=(None, None),
        fixed_alpha=None,
        gamma_threshold=1.0e-8,
        alpha_threshold=1.0e-8,
        max_gamma_iter=100,
    ):
        """Create an estimator for labeled LDA distributions.

        Args:
                estimators (Sequence[ParameterEstimator]): Estimators for the topic distributions.
                num_alphas (int): Number of label rows in the alphas matrix.
                suff_stat (Optional[Any]): Kept for consistency with ParameterEstimator.
                pseudo_count (Optional[Tuple[float, float]]): Optional pseudo counts for the alpha updates.
                keys (Tuple[Optional[str], Optional[str]]): Optional keys for alpha statistics and topic accumulators.
                fixed_alpha (Optional[np.ndarray]): If passed, the alphas matrix is fixed to this value.
                gamma_threshold (float): Convergence threshold for the per-document variational gamma updates.
                alpha_threshold (float): Convergence threshold for the alpha fixed-point updates.

        Attributes:
                num_topics (int): Number of topic distributions.
                estimators (Sequence[ParameterEstimator]): Estimators for the topic distributions.
                pseudo_count (Optional[Tuple[float, float]]): Optional pseudo counts for the alpha updates.
                num_alphas (int): Number of label rows in the alphas matrix.
                suff_stat (Optional[Any]): Kept for consistency with ParameterEstimator.
                keys (Tuple[Optional[str], Optional[str]]): Keys for alpha statistics and topic accumulators.
                gamma_threshold (float): Convergence threshold for the variational gamma updates.
                alpha_threshold (float): Convergence threshold for the alpha fixed-point updates.
                fixed_alpha (Optional[np.ndarray]): If passed, the alphas matrix is fixed to this value.

        """

        self.num_topics = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.num_alphas = num_alphas
        self.suff_stat = suff_stat
        self.keys = keys
        self.gamma_threshold = gamma_threshold
        self.alpha_threshold = alpha_threshold
        self.fixed_alpha = fixed_alpha
        self.max_gamma_iter = int(max_gamma_iter)

    def accumulator_factory(self):
        """Return an accumulator factory configured from this estimator."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return LabeledLDAEstimatorAccumulatorFactory(
            est_factories, self.num_topics, self.num_alphas, self.keys, self.fixed_alpha
        )

    @deprecated_alias("accumulator_factory", since="0.8.0", removed_in="0.10.0")
    def accumulatorFactory(self):
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def estimate(self, nobs, suff_stat):
        """Estimate a labeled LDA distribution from aggregated sufficient statistics.

        ``suff_stat`` is a tuple containing:
                suff_stat[0] (Optional[np.ndarray]): Previous alphas matrix.
                suff_stat[1] (LabeledLDALabelSetStats): Per-label-set expected log topic weights and counts.
                suff_stat[2] (Union[float, np.ndarray]): Per-label weighted document counts.
                suff_stat[3] (np.ndarray): Label-allocated weighted topic counts.
                suff_stat[4] (Sequence): Sufficient statistics for the topic distribution accumulators.

        If 'fixed_alpha' is None, the alphas matrix is re-estimated by maximizing the coupled objective
        over all label rows (see 'update_alpha_coupled()'). When every document carries exactly one label
        the objective decouples and the per-row fixed-point updates are used (see 'update_alpha()').
        Otherwise the alphas matrix is set to 'fixed_alpha'.

        Args:
                nobs (Optional[float]): Number of observations used in estimation.
                suff_stat: See above for details.

        Returns:
                LabeledLDADistribution: Estimated distribution.

        """

        prev_alpha, set_stats, doc_counts, topic_counts, topic_suff_stats = suff_stat

        num_topics = self.num_topics
        topics = [self.estimators[i].estimate(topic_counts[:, i].sum(), topic_suff_stats[i]) for i in range(num_topics)]

        # if doc_counts == 0:
        # sys.stderr.write('Warning: LDA Estimation performed with zero documents.\n')
        # LabeledLDADistribution(topics, prev_alpha, gamma_threshold=self.gamma_threshold)

        if self.fixed_alpha is None:
            if prev_alpha is None:
                prev_alpha = np.ones((self.num_alphas, num_topics))

            label_sets, set_n, set_m = set_stats.arrays()

            if len(label_sets) == 0:
                new_alpha = np.asarray(prev_alpha, dtype=float).copy()
            else:
                if self.pseudo_count is not None:
                    set_n_eff = set_n + self.pseudo_count[0]
                    mean_of_logs = (set_m + np.log(self.pseudo_count[1])) / np.reshape(set_n_eff, (-1, 1))
                else:
                    set_n_eff = set_n
                    mean_of_logs = set_m / np.reshape(set_n, (-1, 1))

                if all(len(u) == 1 for u in label_sets):
                    # Single-label documents: the coupled objective decouples per label row into the
                    # classic fixed-point objective, so update the observed rows independently.
                    rows = np.asarray([u[0] for u in label_sets], dtype=int)
                    new_alpha = np.asarray(prev_alpha, dtype=float).copy()
                    new_alpha[rows, :] = update_alpha(new_alpha[rows, :], mean_of_logs, self.alpha_threshold)
                else:
                    new_alpha = update_alpha_coupled(
                        prev_alpha, label_sets, set_n_eff, mean_of_logs, self.alpha_threshold
                    )
        else:
            new_alpha = np.asarray(self.fixed_alpha).copy()

        return LabeledLDADistribution(
            topics, new_alpha, gamma_threshold=self.gamma_threshold, max_gamma_iter=self.max_gamma_iter
        )


class LabeledLDADataEncoder(DataSequenceEncoder):
    """Encode iid labeled LDA observations for vectorized scoring."""

    def __init__(self, encoder):
        """Create an encoder for labeled LDA documents.

        Args:
                encoder (DataSequenceEncoder): Encoder of type ``T`` for document values.

        Attributes:
                encoder (DataSequenceEncoder): Encoder of type ``T`` for document values.

        """
        self.encoder = encoder

    def __str__(self):
        """Return a constructor-style representation of the encoder."""
        return "LabeledLDADataEncoder(encoder=" + str(self.encoder) + ")"

    def __eq__(self, other):
        """Return whether another encoder is equivalent to this encoder.

        Args:
                other (object): Object to compare.

        Returns:
                True if other is a LabeledLDADataEncoder with an equivalent value encoder, else False.

        """
        if isinstance(other, LabeledLDADataEncoder):
            return self.encoder == other.encoder
        else:
            return False

    def seq_encode(self, x):
        """Encode a sequence of iid LabeledLDA observations (labeled documents) for vectorized functions.

        Return value 'rv' is a Tuple containing:
                rv[0] (int): Number of documents.
                rv[1] (np.ndarray): Document id for each flattened document value.
                rv[2] (np.ndarray): Flattened array of counts for each value in each document.
                rv[3] (Optional[np.ndarray]): Document gammas (defaults to None).
                rv[4]: Sequence encoded flattened document values.
                rv[5] (np.ndarray): Flattened array of label indices over all documents.
                rv[6] (np.ndarray): Number of labels for each document.
                rv[7] (np.ndarray): Document id for each flattened label index.

        Args:
                x (Sequence[Tuple[Sequence[Tuple[T, float]], Sequence[int]]]): Sequence of labeled documents.

        Returns:
                See above for details.

        """

        num_documents = len(x)

        tx = []
        ctx = []
        nx = []
        tidx = []

        nbx = []
        nbcnt = []
        nbidx = []

        for i in range(num_documents):
            tokens_with_context = x[i][0]
            nxx = x[i][1]

            nx.append(len(tokens_with_context))
            nbcnt.append(len(nxx))

            for j in range(len(tokens_with_context)):
                tidx.append(i)
                tx.append(tokens_with_context[j][0])
                ctx.append(tokens_with_context[j][1])

            for j in range(len(nxx)):
                nbidx.append(i)
                nbx.append(nxx[j])

        idx = np.asarray(tidx)
        counts = np.asarray(ctx)
        gammas = None
        enc_data = self.encoder.seq_encode(tx)

        nbx = np.asarray(nbx, dtype=int)
        nbcnt = np.asarray(nbcnt, dtype=int)
        nbidx = np.asarray(nbidx, dtype=int)

        return num_documents, idx, counts, gammas, enc_data, nbx, nbcnt, nbidx


def update_alpha(current_alpha, mean_log_p, alpha_threshold):
    """Fixed-point update of the per-label alpha rows from mean expected log topic weights.

    Iterates alpha <- digammainv(mean_log_p + digamma(sum(alpha))) row-wise until the relative change of
    each row falls below alpha_threshold.

    Args:
            current_alpha (np.ndarray): Current alphas matrix (num_alphas by num_topics).
            mean_log_p (np.ndarray): Per-label mean expected log topic weights (num_alphas by num_topics).
            alpha_threshold (float): Convergence threshold for the row-wise updates.

    Returns:
            Numpy 2-d array of updated alphas (num_alphas by num_topics).

    """

    alpha = current_alpha.copy()
    asum = alpha.sum(axis=1, keepdims=True)
    mlp = mean_log_p

    its_cnt = 0
    rv = np.zeros(alpha.shape)
    not_done = np.arange(alpha.shape[0], dtype=int)

    while len(not_done) > 0:
        dasum = digamma(asum)
        oldAlpha = alpha
        alpha = digammainv(mlp + dasum)
        asum = alpha.sum(axis=1, keepdims=True)
        res = np.abs(alpha - oldAlpha).sum(axis=1, keepdims=True) / asum

        is_done = (res <= alpha_threshold).flatten()

        if np.any(is_done):
            nis_done = ~is_done
            rv[not_done[is_done], :] = alpha[is_done, :]
            not_done = not_done[nis_done]
            mlp = mlp[nis_done, :]
            asum = asum[nis_done]
            alpha = alpha[nis_done, :]

        its_cnt += 1

    return rv


@deprecated_alias("update_alpha", since="0.8.0", removed_in="0.10.0")
def updateAlpha(current_alpha, mean_log_p, alpha_threshold):
    """Deprecated alias for update_alpha()."""
    return update_alpha(current_alpha, mean_log_p, alpha_threshold)


def label_set_membership(label_sets):
    """Returns flattened membership arrays for a sequence of label sets.

    Args:
            label_sets (Sequence[Tuple[int, ...]]): Label-set tuples (one per distinct document label set).

    Returns:
            Tuple of the flattened label indices (member_label), the label-set index of each flattened entry
            (member_set), and the per-set sizes |S| as floats.

    """
    set_sizes = np.asarray([len(u) for u in label_sets], dtype=int)
    member_label = np.asarray([l for u in label_sets for l in u], dtype=int)
    member_set = np.repeat(np.arange(len(label_sets), dtype=int), set_sizes)
    return member_label, member_set, set_sizes.astype(float)


def coupled_alpha_doc_params(alpha, label_sets):
    """Returns the per-label-set Dirichlet parameters a_S = mean_{l in S} alpha[l].

    Args:
            alpha (np.ndarray): Alphas matrix (num_alphas by num_topics).
            label_sets (Sequence[Tuple[int, ...]]): Label-set tuples.

    Returns:
            Numpy 2-d array with one Dirichlet parameter row per label set.

    """
    member_label, member_set, set_sizes = label_set_membership(label_sets)
    a = np.zeros((len(label_sets), alpha.shape[1]))
    np.add.at(a, member_set, alpha[member_label, :])
    a /= np.reshape(set_sizes, (-1, 1))
    return a


def coupled_alpha_objective(alpha, label_sets, set_counts, set_mean_logs):
    """Coupled multi-label alpha objective (terms independent of alpha dropped).

    F(alpha) = sum_S n_S * [ log Gamma(sum_k a_Sk) - sum_k log Gamma(a_Sk) + sum_k a_Sk * mbar_Sk ],
    where a_S = mean_{l in S} alpha[l], n_S = set_counts[S], and mbar_S = set_mean_logs[S] are the
    per-set mean expected log topic weights.

    Args:
            alpha (np.ndarray): Alphas matrix (num_alphas by num_topics).
            label_sets (Sequence[Tuple[int, ...]]): Label-set tuples.
            set_counts (np.ndarray): Per-set document weights n_S.
            set_mean_logs (np.ndarray): Per-set mean expected log topic weights mbar_S (one row per set).

    Returns:
            Objective value F(alpha).

    """
    a = coupled_alpha_doc_params(alpha, label_sets)
    return np.dot(set_counts, gammaln(a.sum(axis=1)) - gammaln(a).sum(axis=1) + (a * set_mean_logs).sum(axis=1))


def coupled_alpha_gradient(alpha, label_sets, set_counts, set_mean_logs):
    """Gradient of the coupled multi-label alpha objective with respect to alpha.

    dF/d alpha[l,k] = sum_{S contains l} (n_S/|S|) * [ psi(sum_j a_Sj) - psi(a_Sk) + mbar_Sk ], with one
    term per occurrence of l in S.

    Args:
            alpha (np.ndarray): Alphas matrix (num_alphas by num_topics).
            label_sets (Sequence[Tuple[int, ...]]): Label-set tuples.
            set_counts (np.ndarray): Per-set document weights n_S.
            set_mean_logs (np.ndarray): Per-set mean expected log topic weights mbar_S (one row per set).

    Returns:
            Numpy 2-d array with the same shape as alpha.

    """
    member_label, member_set, set_sizes = label_set_membership(label_sets)
    a = coupled_alpha_doc_params(alpha, label_sets)
    g_set = digamma(a.sum(axis=1, keepdims=True)) - digamma(a) + set_mean_logs
    g_set *= np.reshape(set_counts / set_sizes, (-1, 1))
    g = np.zeros(alpha.shape)
    np.add.at(g, member_label, g_set[member_set, :])
    return g


def update_alpha_coupled(current_alpha, label_sets, set_counts, set_mean_logs, alpha_threshold, max_its=2000):
    """Coupled update of the full alphas matrix for documents with multi-label sets.

    Maximizes 'coupled_alpha_objective()' over all positive alpha entries. Since each document Dirichlet
    parameter a_S averages several alpha rows, the rows do not decouple; the objective is concave in
    alpha (a_S is linear in alpha and the Dirichlet log-partition is convex), so ascent converges to the
    global maximum. The ascent is run on beta = log(alpha) (keeping alpha positive) with backtracking
    line search and an adaptive step size, warm-started from 'current_alpha', until the row-wise relative
    change of alpha falls below alpha_threshold (matching the 'update_alpha()' convergence semantics).

    Label rows that appear in no label set have zero gradient and are returned unchanged.

    Args:
            current_alpha (np.ndarray): Current alphas matrix (num_alphas by num_topics), used as warm start.
            label_sets (Sequence[Tuple[int, ...]]): Distinct document label sets (sorted tuples).
            set_counts (np.ndarray): Per-set document weights n_S.
            set_mean_logs (np.ndarray): Per-set mean expected log topic weights mbar_S (one row per set).
            alpha_threshold (float): Convergence threshold for the row-wise relative alpha changes.
            max_its (int): Maximum number of accepted ascent steps.

    Returns:
            Numpy 2-d array of updated alphas (num_alphas by num_topics).

    """

    alpha = np.maximum(np.asarray(current_alpha, dtype=float), 1.0e-10)
    beta = np.log(alpha)
    f_cur = coupled_alpha_objective(alpha, label_sets, set_counts, set_mean_logs)
    step = 1.0

    for its_cnt in range(max_its):
        g_beta = coupled_alpha_gradient(alpha, label_sets, set_counts, set_mean_logs)
        g_beta *= alpha
        g_sq = np.sum(g_beta * g_beta)

        if not np.isfinite(g_sq) or g_sq == 0.0:
            break

        # Backtracking line search on F along the beta-space gradient direction (Armijo condition).
        t = step
        accepted = False
        while t >= 1.0e-16:
            beta_new = np.clip(beta + t * g_beta, -300.0, 300.0)
            alpha_new = np.exp(beta_new)
            f_new = coupled_alpha_objective(alpha_new, label_sets, set_counts, set_mean_logs)
            if np.isfinite(f_new) and f_new >= f_cur + 1.0e-4 * t * g_sq:
                accepted = True
                break
            t *= 0.5

        if not accepted:
            break

        res = np.max(np.abs(alpha_new - alpha).sum(axis=1) / alpha_new.sum(axis=1))
        alpha = alpha_new
        beta = beta_new
        f_cur = f_new
        step = min(t * 2.0, 1.0e8)

        if res <= alpha_threshold:
            break

    return alpha


def mpe_update(X, y, min_size=2):
    """Single minimal polynomial extrapolation (MPE) update step for a fixed-point iterate y.

    Args:
            X (Optional[np.ndarray]): Matrix of previous iterates (one per row), or None to start.
            y (np.ndarray): New fixed-point iterate.
            min_size (int): Minimum number of stored iterates before extrapolating.

    Returns:
            Tuple of the updated iterate matrix and the extrapolated estimate.

    """

    if X is None:
        X = np.reshape(y, (1, -1))
        return X, y
    elif X.shape[0] < min_size:
        X = np.concatenate((X, np.reshape(y, (1, -1))), axis=0)
        return X, y

    dy = y - X[-1, :]
    U = (X[1:, :] - X[:-1, :]).T
    X2 = X[1:, :].T
    c = np.dot(np.linalg.pinv(U), dy)
    c *= -1
    s = (np.dot(X2, c) + y) / (c.sum() + 1)

    X = np.concatenate((X, np.reshape(y, (1, -1))), axis=0)

    return X, s


def mpe(x0, f, eps):
    """Minimal polynomial extrapolation (MPE) of the fixed point of f starting from x0.

    Args:
            x0 (np.ndarray): Starting point of the fixed-point iteration.
            f (Callable[[np.ndarray], np.ndarray]): Fixed-point map.
            eps (float): Convergence threshold on the absolute change of the extrapolated estimate.

    Returns:
            Tuple of the extrapolated fixed point and the iteration count.

    """

    x1 = f(x0)
    x2 = f(x1)
    x3 = f(x2)
    X = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 2

    while res > eps:
        y = f(X[-1, :])
        dy = y - X[-1, :]
        U = (X[1:, :] - X[:-1, :]).T
        X2 = X[1:, :].T
        c = np.dot(np.linalg.pinv(U), dy)
        c *= -1
        s = (np.dot(X2, c) + y) / (c.sum() + 1)

        res = np.abs(s - s0).sum()
        s0 = s
        X = np.concatenate((X, np.reshape(y, (1, -1))), axis=0)
        its_cnt += 1

    return s, its_cnt


def alpha_seq_lambda(meanLogP):
    """Returns the alpha fixed-point map for mean expected log topic weights meanLogP."""

    def next_alpha(currentAlpha):
        return digammainv(meanLogP + digamma(currentAlpha.sum()))

    return next_alpha


def find_alpha(current_alpha, mlp, thresh):
    """Find the alpha fixed point for mean expected log topic weights mlp via MPE.

    Args:
            current_alpha (np.ndarray): Starting alpha value.
            mlp (np.ndarray): Mean expected log topic weights.
            thresh (float): Convergence threshold.

    Returns:
            Tuple of the extrapolated alpha and the iteration count.

    """
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh)


def seq_posterior(estimate, x):
    """Compute the variational posterior quantities for encoded labeled documents under 'estimate'.

    Runs the shared per-document mean-field gamma fixed point (see lda._lda_vi_fixed_point), passing
    each document's coupled Dirichlet prior 'alphas_loc' -- the mean of the alpha rows of the
    document's label set. This is the only model difference from plain LDA, which uses a single
    shared alpha; the fixed-point loop is otherwise identical.

    Args:
            estimate (LabeledLDADistribution): LabeledLDA model used to evaluate the posterior.
            x: Encoded sequence of iid LabeledLDA observations (see LabeledLDADataEncoder.seq_encode()).

    Returns:
            Tuple of per-value topic responsibilities (log_density_gamma), per-document gammas (final_gammas),
            per-document Dirichlet parameters (alphas_loc), and per-value per-topic log-densities.

    """

    alphas = estimate.alphas
    topics = estimate.topics
    gamma_threshold = estimate.gamma_threshold

    num_documents, idx, counts, gammas, enc_data, nbx, nbcnt, nbidx = x

    num_topics = len(topics)

    per_topic_log_densities = np.asarray([topics[i].seq_log_density(enc_data) for i in range(num_topics)]).transpose()

    # Per-document coupled prior: mean of the alpha rows over the document's labels.
    ddd = np.reshape(np.bincount(nbidx, minlength=num_documents), (-1, 1)).astype(float)
    alphas_loc = np.zeros((num_documents, num_topics))
    for i in range(num_topics):
        alphas_loc[:, i] = np.bincount(nbidx, weights=alphas[nbx, i], minlength=num_documents)
    alphas_loc /= ddd

    log_density_gamma, final_gammas = _lda_vi_fixed_point(
        alphas_loc,
        idx,
        counts,
        gammas,
        num_topics,
        per_topic_log_densities,
        gamma_threshold,
        getattr(estimate, "max_gamma_iter", 100),
    )

    return log_density_gamma, final_gammas, alphas_loc, per_topic_log_densities


# --- Backward-compatible API naming aliases ---
LabeledLDAAccumulator = LabeledLDAEstimatorAccumulator
LabeledLDAAccumulatorFactory = LabeledLDAEstimatorAccumulatorFactory
