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

If included, 'len_dist' models the number of values N in a document, and 'set_dist' models the label sets;
both factors are included in joint scoring and sampling and remain fixed during topic/alpha estimation.

Estimation uses a mean-field variational EM (per-document gamma updates). The expected log topic weights
are aggregated per distinct label set, and the alpha rows are updated jointly by maximizing the coupled
objective in which each document's Dirichlet parameter is the average of its label rows (see
'update_alpha_coupled()'). When every document carries exactly one label the objective decouples and the
classic per-row fixed-point update ('update_alpha()') is used.

"""

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, gammaln

from mixle.capability import Neutral, supports
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
from mixle.stats.latent.effective_sample import validate_effective_sample_mass, validated_statistic_tuple
from mixle.stats.latent.lda import (
    LDAConvergenceError,
    LDADataEncoder,
    LDAOptimizationDiagnostics,
    _lda_elbo_from_gamma,
    _lda_vi_fixed_point,
    _positive_finite_threshold,
    _positive_iteration_budget,
    _validate_lda_encoded,
    _validated_document_weights,
    mpe,
)
from mixle.stats.latent.lda import (
    update_alpha as _update_lda_alpha,
)
from mixle.utils.deprecation import deprecated_alias
from mixle.utils.special import digammainv
from mixle.utils.vector import ImpossibleEvidenceError, row_choice


def _canonical_label_set(labels, num_alphas=None, *, context="labeled LDA label set"):
    """Return a nonempty sorted unique tuple of exact label-row indices."""
    if isinstance(labels, (str, bytes)):
        raise TypeError(f"{context} must be a sequence of exact integer indices")
    try:
        raw_labels = list(labels)
    except TypeError as exc:
        raise TypeError(f"{context} must be a sequence of exact integer indices") from exc
    if not raw_labels:
        raise ValueError(f"{context} must be nonempty")
    canonical = []
    for position, raw_label in enumerate(raw_labels):
        if isinstance(raw_label, (str, bytes, bool, np.bool_, float, np.floating)):
            raise TypeError(f"{context} entry {position} must be an exact integer")
        try:
            label = int(raw_label)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{context} entry {position} must be an exact integer") from exc
        if label < 0 or (num_alphas is not None and label >= num_alphas):
            raise ValueError(f"{context} entry {position} is outside the alpha-row range")
        canonical.append(label)
    return tuple(sorted(set(canonical)))


def _validate_labeled_lda_encoded(x, num_topics, num_alphas):
    """Validate flattened token data and canonical label geometry."""
    if not isinstance(x, tuple) or len(x) != 8:
        raise TypeError("encoded labeled-LDA data must be an eight-item tuple")
    num_documents, idx, counts, gammas, enc_data = _validate_lda_encoded(x[:5], num_topics)

    def exact_vector(values, name):
        raw = np.asarray(values)
        if raw.ndim != 1 or raw.dtype.kind == "b":
            raise TypeError(f"encoded labeled-LDA {name} must be a one-dimensional exact-integer array")
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"encoded labeled-LDA {name} must contain exact integers") from exc
        if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
            raise ValueError(f"encoded labeled-LDA {name} must contain finite exact integers")
        return numeric.astype(np.intp)

    nbx = exact_vector(x[5], "label IDs")
    nbcnt = exact_vector(x[6], "label counts")
    nbidx = exact_vector(x[7], "label document IDs")
    if nbcnt.shape != (num_documents,) or np.any(nbcnt <= 0):
        raise ValueError("every encoded labeled-LDA document must have a positive label count")
    if nbx.shape != nbidx.shape or nbx.size != int(nbcnt.sum()):
        raise ValueError("encoded labeled-LDA label arrays do not match the declared label counts")
    if not np.array_equal(nbidx, np.repeat(np.arange(num_documents, dtype=np.intp), nbcnt)):
        raise ValueError("encoded labeled-LDA label document IDs do not match the declared label counts")
    if np.any(nbx < 0) or np.any(nbx >= num_alphas):
        raise ValueError("encoded labeled-LDA label IDs are outside the alpha-row range")
    offset = 0
    for document_index, count in enumerate(nbcnt):
        labels = nbx[offset : offset + count]
        if labels.size > 1 and np.any(labels[1:] <= labels[:-1]):
            raise ValueError(f"encoded labeled-LDA labels for document {document_index} must be sorted and unique")
        offset += count
    return num_documents, idx, counts, gammas, enc_data, nbx, nbcnt, nbidx


def _structural_log_scores(distribution, num_documents, idx, counts, nbx, nbcnt):
    """Return label-set and length log factors for the modeled joint law."""
    result = np.zeros(num_documents, dtype=np.float64)
    if distribution.set_dist is not None and not supports(distribution.set_dist, Neutral):
        result += np.asarray(
            [distribution.set_dist.log_density(label_set) for label_set in doc_label_sets(nbx, nbcnt)],
            dtype=np.float64,
        )
    if distribution.len_dist is not None and not supports(distribution.len_dist, Neutral):
        lengths = np.bincount(idx, weights=counts, minlength=num_documents)
        len_enc = distribution.len_dist.dist_to_encoder().seq_encode(lengths)
        result += np.asarray(distribution.len_dist.seq_log_density(len_enc), dtype=np.float64)
    if result.shape != (num_documents,) or np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise ValueError("labeled-LDA structural laws produced invalid log densities")
    return result


class LabeledLDADistribution(SequenceEncodableProbabilityDistribution):
    """Labeled LDA model for documents with label sets.

    Compatible with data type Tuple[Sequence[Tuple[T, float]], Sequence[int]], where T is the data type of
    the topic distributions.
    """

    def __init__(
        self,
        topics,
        alphas,
        set_dist=None,
        len_dist=None,
        gamma_threshold=1.0e-8,
        max_gamma_iter=1000,
        fit_diagnostics=None,
    ):
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
        if isinstance(topics, (str, bytes)) or len(topics) == 0:
            raise ValueError("labeled LDA requires at least one topic distribution")
        alpha_array = np.asarray(alphas, dtype=np.float64)
        if alpha_array.ndim != 2 or alpha_array.shape[0] == 0 or alpha_array.shape[1] != len(topics):
            raise ValueError("labeled-LDA alphas must be a nonempty matrix with one column per topic")
        if np.any(~np.isfinite(alpha_array)) or np.any(alpha_array <= 0.0):
            raise ValueError("labeled-LDA alphas must be positive and finite")
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, LDAOptimizationDiagnostics):
            raise TypeError("fit_diagnostics must be an LDAOptimizationDiagnostics record or None")
        self.topics = tuple(topics)
        self.nTopics = len(topics)
        self.alphas = alpha_array.copy()
        self.num_alpha = self.alphas.shape[0]
        self.len_dist = len_dist
        self.set_dist = set_dist
        self.gamma_threshold = _positive_finite_threshold(gamma_threshold, "gamma_threshold")
        self.max_gamma_iter = _positive_iteration_budget(max_gamma_iter, "max_gamma_iter")
        self.fit_diagnostics = fit_diagnostics

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
        num_documents, idx, counts, _, _, nbx, nbcnt, _ = _validate_labeled_lda_encoded(x, self.nTopics, self.num_alpha)

        log_density_gamma, document_gammas, document_alphas, per_topic_log_densities = seq_posterior(self, x)

        elbo = _lda_elbo_from_gamma(
            document_alphas, idx, counts, num_topics, log_density_gamma, document_gammas, per_topic_log_densities
        )
        elbo += _structural_log_scores(self, num_documents, idx, counts, nbx, nbcnt)
        return elbo

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
        num_documents, idx, counts, _, enc_data, _, _, _ = _validate_labeled_lda_encoded(
            x, self.nTopics, self.num_alpha
        )

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

        children = tuple(self.topics)
        if self.set_dist is not None and not supports(self.set_dist, Neutral):
            children += (self.set_dist,)
        if self.len_dist is not None and not supports(self.len_dist, Neutral):
            children += (self.len_dist,)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def _backend_seq_posterior(self, x, engine):
        """Evaluate backend topic scores under the validated labeled-LDA VI contract."""
        from mixle.stats.compute.backend import backend_seq_log_density

        num_documents, idx, counts, gammas, enc_data, nbx, nbcnt, nbidx = _validate_labeled_lda_encoded(
            x, self.nTopics, self.num_alpha
        )
        document_alphas = np.zeros((num_documents, self.nTopics), dtype=np.float64)
        for topic_index in range(self.nTopics):
            document_alphas[:, topic_index] = np.bincount(
                nbidx, weights=self.alphas[nbx, topic_index], minlength=num_documents
            )
        document_alphas /= nbcnt[:, None]
        topic_scores = engine.stack([backend_seq_log_density(topic, enc_data, engine) for topic in self.topics], axis=1)
        responsibilities, document_gammas = _lda_vi_fixed_point(
            document_alphas,
            idx,
            counts,
            gammas,
            self.nTopics,
            np.asarray(engine.to_numpy(topic_scores), dtype=np.float64),
            self.gamma_threshold,
            self.max_gamma_iter,
        )
        return (
            engine.asarray(responsibilities),
            engine.asarray(document_gammas),
            engine.asarray(document_alphas),
            topic_scores,
        )

    def backend_seq_log_density(self, x, engine):
        """Backend-neutral labeled-LDA joint variational lower-bound scoring."""
        num_documents, idx, counts, _, _, nbx, nbcnt, _ = _validate_labeled_lda_encoded(x, self.nTopics, self.num_alpha)
        responsibilities, gammas, document_alphas, topic_scores = self._backend_seq_posterior(x, engine)
        result = engine.asarray(
            _lda_elbo_from_gamma(
                np.asarray(engine.to_numpy(document_alphas), dtype=np.float64),
                idx,
                counts,
                self.nTopics,
                np.asarray(engine.to_numpy(responsibilities), dtype=np.float64),
                np.asarray(engine.to_numpy(gammas), dtype=np.float64),
                np.asarray(engine.to_numpy(topic_scores), dtype=np.float64),
            )
        )
        result = result + engine.asarray(_structural_log_scores(self, num_documents, idx, counts, nbx, nbcnt))
        return result

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
            set_dist=self.set_dist,
            len_dist=self.len_dist,
            gamma_threshold=self.gamma_threshold,
            max_gamma_iter=self.max_gamma_iter,
        )

    def dist_to_encoder(self):
        """Return a data encoder for iid labeled LDA observations."""
        return LabeledLDADataEncoder(encoder=self.topics[0].dist_to_encoder(), num_alphas=self.num_alpha)


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
        if dist.set_dist is None or supports(dist.set_dist, Neutral):
            raise ValueError("labeled-LDA sampling requires an explicit label-set distribution")
        if dist.len_dist is None or supports(dist.len_dist, Neutral):
            raise ValueError("labeled-LDA sampling requires an explicit document-length distribution")
        self.rng = RandomState(seed)
        self.dist = dist
        self.nTopics = dist.nTopics
        self.compSamplers = [self.dist.topics[i].sampler(seed=self.rng.randint(maxint)) for i in range(dist.nTopics)]
        # self.dirichletSampler = DirichletDistribution(dist.alpha).sampler(self.rng.randint(maxint))
        self.len_dist = self.dist.len_dist.sampler(seed=self.rng.randint(maxint))
        self.set_dist = self.dist.set_dist.sampler(seed=self.rng.randint(maxint))

    def sample(self, size=None, *, batched: bool = True):
        """Draw iid labeled documents from the LabeledLDA model.

        If size is None, a single Tuple[List[T], List[int]] is returned containing the sampled document
        values and its labels. If size > 0, a list of 'size' such Tuples is returned.

        Args:
                size (Optional[int]): Number of iid labeled documents to sample.

        Returns:
                Tuple[List[T], List[int]] or a List of such Tuples depending on arg size.

        """

        if size is None:
            nodes = _canonical_label_set(
                self.set_dist.sample(), self.dist.num_alpha, context="sampled labeled-LDA label set"
            )
            raw_length = self.len_dist.sample()
            if isinstance(raw_length, (bool, np.bool_, float, np.floating)):
                raise TypeError("sampled labeled-LDA document length must be an exact non-negative integer")
            try:
                n = int(raw_length)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("sampled labeled-LDA document length must be an exact non-negative integer") from exc
            if n < 0:
                raise ValueError("sampled labeled-LDA document length must be non-negative")
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

            return (rv, list(nodes))

        else:
            return [self.sample() for i in range(size)]


class LabeledLDALabelSetStats:
    """Sufficient statistics for the coupled alpha update, grouped by distinct document label set.

    Maps each distinct label set S (a sorted tuple of unique label indices) to a pair
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
        self.stats = {}
        if stats is not None:
            for label_set, entry in stats.items():
                try:
                    weight, sum_log_p = entry
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "labeled-LDA label-set statistic entries must contain weight and mean logs"
                    ) from exc
                self.add(label_set, weight, sum_log_p)

    def add(self, label_set, weight, sum_log_p):
        """Accumulate document weight and summed expected log topic weights for one label set.

        Args:
                label_set (Tuple[int, ...]): Sorted tuple of label indices.
                weight (float): Total document weight to add for the label set.
                sum_log_p (np.ndarray): Weighted sum of per-document E[log theta] vectors to add.

        Returns:
                None.

        """
        canonical = _canonical_label_set(label_set, context="labeled-LDA statistic label set")
        if isinstance(weight, (bool, np.bool_)):
            raise TypeError("labeled-LDA label-set statistic weight must be real-valued")
        numeric_weight = float(weight)
        logs = np.asarray(sum_log_p, dtype=np.float64)
        if not np.isfinite(numeric_weight) or numeric_weight < 0.0:
            raise ValueError("labeled-LDA label-set statistic weight must be finite and non-negative")
        if logs.ndim != 1 or np.any(~np.isfinite(logs)):
            raise ValueError("labeled-LDA label-set mean-log statistic must be a finite vector")
        entry = self.stats.get(canonical)
        if entry is None:
            self.stats[canonical] = [numeric_weight, logs.copy()]
        else:
            if entry[1].shape != logs.shape:
                raise ValueError("labeled-LDA label-set mean-log statistics have inconsistent geometry")
            entry[0] += numeric_weight
            entry[1] += logs

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
            if set_n[j] > 0.0:
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

        encoded = self.acc_to_encoder().seq_encode([x])
        self.seq_initialize(encoded, _validated_document_weights([weight], 1), rng)

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

        num_documents, idx, counts, old_gammas, enc_data, nbx, nbcnt, nbidx = _validate_labeled_lda_encoded(
            x, self.num_topics, self.num_alphas
        )
        weights = _validated_document_weights(weights, num_documents)

        if not self._init_rng:
            self._rng_initialize(rng)

        if self.prev_alpha is None:
            self.prev_alpha = np.ones((self.num_alphas, self.num_topics))

        # Per-document Dirichlet parameter: average of prev_alpha rows over the document labels.
        aloc = np.zeros((num_documents, self.num_topics))
        for j in range(self.num_topics):
            aloc[:, j] = np.bincount(nbidx, weights=self.prev_alpha[nbx, j], minlength=num_documents)
        aloc /= np.reshape(nbcnt.astype(float), (-1, 1))

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
            label_weight = doc_w[nbidx] / nbcnt[nbidx].astype(float)
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

        num_documents, idx, counts, old_gammas, enc_data, nbx, nbcnt, nbidx = _validate_labeled_lda_encoded(
            x, self.num_topics, self.num_alphas
        )
        weights = _validated_document_weights(weights, num_documents)
        (
            log_density_gamma,
            final_gammas,
            doc_alphas,
            per_topic_log_densities,
            diagnostics,
        ) = seq_posterior_with_diagnostics(estimate, x)
        structural_scores = _structural_log_scores(estimate, num_documents, idx, counts, nbx, nbcnt)
        impossible = np.unique(
            np.concatenate(
                (
                    np.asarray(diagnostics.impossible_documents, dtype=np.intp),
                    np.flatnonzero(np.isneginf(structural_scores)),
                )
            )
        )
        if impossible.size and np.any(weights[impossible] > 0.0):
            raise ImpossibleEvidenceError(
                "labeled-LDA E-step encountered zero-probability evidence at document rows %s"
                % impossible[weights[impossible] > 0.0].tolist()
            )
        weighted_topic_counts = log_density_gamma * np.reshape(weights[idx], (-1, 1))

        mlpf = digamma(final_gammas) - digamma(np.sum(final_gammas, axis=1, keepdims=True))

        nbh_cnt = np.reshape(np.bincount(nbx, weights=weights[nbidx], minlength=num_alphas), (-1, 1))
        nbh_tcnt = np.zeros((num_alphas, num_topics))

        for i in range(num_topics):
            self.accumulators[i].seq_update(enc_data, weighted_topic_counts[:, i], estimate.topics[i])

            doc_tcnt = np.bincount(idx, weights=log_density_gamma[:, i], minlength=num_documents)
            label_weight = doc_tcnt[nbidx] * weights[nbidx] / nbcnt[nbidx].astype(float)
            nbh_tcnt[:, i] = np.bincount(nbx, weights=label_weight, minlength=num_alphas)

        self._accumulate_set_stats(mlpf, weights, nbx, nbcnt)
        self.doc_counts += nbh_cnt
        self.topic_counts += nbh_tcnt
        self.prev_alpha = estimate.alphas

        # Fused-EM fast path: recover the per-document ELBO that estimate.seq_log_density would
        # return, reusing the variational quantities the E-step already produced -- no second
        # variational loop and no re-scoring of topics. Mirrors LabeledLDADistribution.seq_log_density
        # exactly, including any fixed length and label-set terms. Gated; standard path untouched.
        if self._track_ll:
            elob = _lda_elbo_from_gamma(
                doc_alphas, idx, counts, num_topics, log_density_gamma, final_gammas, per_topic_log_densities
            )
            elob += structural_scores
            positive_weight = weights > 0.0
            self._seq_ll += float(np.dot(weights[positive_weight], elob[positive_weight]))

        # return num_documents, idx, counts, final_gammas, enc_data

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident LabeledLDA E-step (numpy or torch).

        Runs the variational posterior and the per-label-set / topic-count aggregations on the active
        engine, feeding engine-computed responsibilities to the topic accumulators. Mirrors seq_update.
        """
        num_alphas = self.num_alphas
        num_topics = self.num_topics
        num_documents, idx, counts, old_gammas, enc_data, nbx, nbcnt, nbidx = _validate_labeled_lda_encoded(
            x, self.num_topics, self.num_alphas
        )
        weights_np = _validated_document_weights(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, num_documents
        )
        idx_np = np.asarray(idx, dtype=np.int64)
        nbx_np = np.asarray(nbx, dtype=np.int64)
        nbidx_np = np.asarray(nbidx, dtype=np.int64)

        log_density_gamma, final_gammas, doc_alphas, per_topic_log_densities = estimate._backend_seq_posterior(
            x, engine
        )
        score_array = np.asarray(engine.to_numpy(per_topic_log_densities), dtype=np.float64)
        structural_scores = _structural_log_scores(estimate, num_documents, idx_np, counts, nbx_np, nbcnt)
        impossible = np.unique(
            np.concatenate(
                (
                    idx_np[~np.any(np.isfinite(score_array), axis=1)],
                    np.flatnonzero(np.isneginf(structural_scores)),
                )
            )
        )
        if impossible.size and np.any(weights_np[impossible] > 0.0):
            raise ImpossibleEvidenceError(
                "labeled-LDA engine E-step encountered zero-probability evidence at document rows %s"
                % impossible[weights_np[impossible] > 0.0].tolist()
            )

        idx_e = engine.asarray(idx_np)
        nbx_e = engine.asarray(nbx_np)
        nbidx_e = engine.asarray(nbidx_np)
        w_idx = engine.asarray(weights_np[idx_np]).reshape((-1, 1))
        weighted_topic_counts = log_density_gamma * w_idx

        gamma_sum = engine.sum(final_gammas, axis=1).reshape((-1, 1))
        mlpf = engine.digamma(final_gammas) - engine.digamma(gamma_sum)

        nbh_cnt = engine.index_add(engine.zeros(num_alphas), nbx_e, engine.asarray(weights_np[nbidx_np]))
        nbcnt_doc_e = engine.asarray(np.asarray(nbcnt)[nbidx_np].astype(np.float64))
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

        prev_alpha, set_stats, doc_counts, topic_counts, topic_suff_stats = validated_statistic_tuple(
            suff_stat, 5, "labeled-LDA sufficient statistics"
        )

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

                # Copy this accumulator's own prev_alpha before it can be adopted into
                # stats_dict below -- otherwise a later key_replace would leave this
                # accumulator's private prev_alpha aliased into another tied accumulator.
                prev_alpha = self.prev_alpha.copy() if self.prev_alpha is not None else p_pa
                stats_dict[self.alpha_key] = (self.set_stats.copy().combine(p_sol), self.doc_counts + p_doc, prev_alpha)

            else:
                # Copy on adoption: stats_dict must never alias this accumulator's own live
                # set_stats/prev_alpha. doc_counts is a plain float (immutable, safe as-is).
                stats_dict[self.alpha_key] = (
                    self.set_stats.copy(),
                    self.doc_counts,
                    self.prev_alpha.copy() if self.prev_alpha is not None else None,
                )

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
                # Copy on replace too: without it, every tied accumulator ends up pointing at
                # the SAME set_stats/prev_alpha objects, so any one of them later accumulating
                # new local data would silently corrupt every other tied accumulator's counts.
                p_sol, p_doc, p_pa = stats_dict[self.alpha_key]
                self.prev_alpha = np.asarray(p_pa).copy() if p_pa is not None else None
                self.set_stats = p_sol.copy()
                self.doc_counts = p_doc

        if self.topics_key is not None:
            if self.topics_key in stats_dict:
                acc = stats_dict[self.topics_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self):
        """Return a data encoder built from the topic accumulators."""
        return LabeledLDADataEncoder(encoder=self.accumulators[0].acc_to_encoder(), num_alphas=self.num_alphas)


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
        max_gamma_iter=1000,
        max_alpha_iter=2000,
        set_dist=None,
        len_dist=None,
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

        if isinstance(estimators, (str, bytes)) or len(estimators) == 0:
            raise ValueError("LabeledLDAEstimator requires at least one topic estimator")
        self.num_topics = len(estimators)
        self.estimators = tuple(estimators)
        if pseudo_count is None:
            self.pseudo_count = None
        else:
            values = np.asarray(pseudo_count, dtype=np.float64)
            if values.shape != (2,) or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError("labeled-LDA pseudo_count must contain two positive finite values")
            self.pseudo_count = tuple(values.tolist())
        self.num_alphas = _positive_iteration_budget(num_alphas, "num_alphas")
        self.suff_stat = suff_stat
        if not isinstance(keys, tuple) or len(keys) != 2:
            raise ValueError("labeled-LDA keys must be a two-item tuple")
        self.keys = keys
        self.gamma_threshold = _positive_finite_threshold(gamma_threshold, "gamma_threshold")
        self.alpha_threshold = _positive_finite_threshold(alpha_threshold, "alpha_threshold")
        self.max_gamma_iter = _positive_iteration_budget(max_gamma_iter, "max_gamma_iter")
        self.max_alpha_iter = _positive_iteration_budget(max_alpha_iter, "max_alpha_iter")
        if fixed_alpha is None:
            self.fixed_alpha = None
        else:
            fixed = np.asarray(fixed_alpha, dtype=np.float64)
            if fixed.shape != (self.num_alphas, self.num_topics) or np.any(~np.isfinite(fixed)) or np.any(fixed <= 0.0):
                raise ValueError("fixed_alpha must be a positive finite num_alphas by num_topics matrix")
            self.fixed_alpha = fixed.copy()
        self.set_dist = set_dist
        self.len_dist = len_dist

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
        if not isinstance(set_stats, LabeledLDALabelSetStats):
            raise TypeError("labeled-LDA set statistics must be LabeledLDALabelSetStats")
        _, label_set_weights, _ = set_stats.arrays()
        validate_effective_sample_mass(
            nobs,
            label_set_weights.sum(),
            label="labeled-LDA effective sample",
        )
        document_counts = np.asarray(doc_counts, dtype=np.float64)
        if document_counts.ndim == 0:
            valid_document_counts = True
        else:
            valid_document_counts = document_counts.shape in {
                (self.num_alphas,),
                (self.num_alphas, 1),
            }
        if not valid_document_counts or np.any(~np.isfinite(document_counts)) or np.any(document_counts < 0.0):
            raise ValueError("labeled-LDA document counts have invalid geometry or values")
        topic_counts = np.asarray(topic_counts, dtype=np.float64)
        if (
            topic_counts.shape != (self.num_alphas, num_topics)
            or np.any(~np.isfinite(topic_counts))
            or np.any(topic_counts < 0.0)
        ):
            raise ValueError("labeled-LDA topic counts have invalid geometry or values")
        if len(topic_suff_stats) != num_topics:
            raise ValueError("labeled-LDA topic statistics must contain one entry per topic")
        if prev_alpha is None:
            prev_alpha = self.fixed_alpha if self.fixed_alpha is not None else np.ones((self.num_alphas, num_topics))
        prev_alpha = np.asarray(prev_alpha, dtype=np.float64)
        if (
            prev_alpha.shape != (self.num_alphas, num_topics)
            or np.any(~np.isfinite(prev_alpha))
            or np.any(prev_alpha <= 0.0)
        ):
            raise ValueError("previous labeled-LDA alphas have invalid geometry or values")
        topics = [self.estimators[i].estimate(topic_counts[:, i].sum(), topic_suff_stats[i]) for i in range(num_topics)]

        if self.fixed_alpha is None:
            label_sets, set_n, set_m = set_stats.arrays()
            positive_sets = set_n > 0.0
            label_sets = [label_set for label_set, keep in zip(label_sets, positive_sets) if keep]
            set_n = set_n[positive_sets]
            set_m = set_m[positive_sets, :]

            if len(label_sets) == 0:
                new_alpha = np.asarray(prev_alpha, dtype=float).copy()
                diagnostics = LDAOptimizationDiagnostics(
                    algorithm="labeled_lda_alpha",
                    converged=True,
                    iterations=0,
                    max_iterations=self.max_alpha_iter,
                    termination_reason="no_weighted_documents",
                    final_residual=0.0,
                )
            else:
                if self.pseudo_count is not None:
                    set_n_eff = set_n + self.pseudo_count[0]
                    mean_of_logs = (set_m + np.log(self.pseudo_count[1])) / np.reshape(set_n_eff, (-1, 1))
                else:
                    set_n_eff = set_n
                    mean_of_logs = set_m / np.reshape(set_n, (-1, 1))

                if float(set_n_eff.sum()) < 2.0 * num_topics:
                    new_alpha = np.asarray(prev_alpha, dtype=float).copy()
                    diagnostics = LDAOptimizationDiagnostics(
                        algorithm="labeled_lda_alpha",
                        converged=False,
                        iterations=0,
                        max_iterations=self.max_alpha_iter,
                        termination_reason="retained_previous_insufficient_effective_documents",
                        final_residual=float("inf"),
                    )
                else:
                    try:
                        if all(len(u) == 1 for u in label_sets):
                            # Single-label documents: the coupled objective decouples per label row into the
                            # classic fixed-point objective, so update the observed rows independently.
                            rows = np.asarray([u[0] for u in label_sets], dtype=int)
                            new_alpha = np.asarray(prev_alpha, dtype=float).copy()
                            new_alpha[rows, :], diagnostics = update_alpha(
                                new_alpha[rows, :],
                                mean_of_logs,
                                self.alpha_threshold,
                                max_iter=self.max_alpha_iter,
                                return_diagnostics=True,
                            )
                        else:
                            new_alpha, diagnostics = update_alpha_coupled(
                                prev_alpha,
                                label_sets,
                                set_n_eff,
                                mean_of_logs,
                                self.alpha_threshold,
                                max_its=self.max_alpha_iter,
                                return_diagnostics=True,
                            )
                    except LDAConvergenceError as exc:
                        # Preserve the last accepted parameter, never a failed iterate, and carry
                        # the exact finite-optimization failure receipt on the returned model.
                        new_alpha = np.asarray(prev_alpha, dtype=float).copy()
                        failed = exc.diagnostics
                        diagnostics = LDAOptimizationDiagnostics(
                            algorithm=failed.algorithm,
                            converged=False,
                            iterations=failed.iterations,
                            max_iterations=failed.max_iterations,
                            termination_reason="retained_previous_after_" + failed.termination_reason,
                            final_residual=failed.final_residual,
                            objective_trace=failed.objective_trace,
                            residual_trace=failed.residual_trace,
                        )
        else:
            new_alpha = np.asarray(self.fixed_alpha).copy()
            diagnostics = LDAOptimizationDiagnostics(
                algorithm="fixed_labeled_lda_alpha",
                converged=True,
                iterations=0,
                max_iterations=0,
                termination_reason="fixed_parameter",
                final_residual=0.0,
            )

        return LabeledLDADistribution(
            topics,
            new_alpha,
            set_dist=self.set_dist,
            len_dist=self.len_dist,
            gamma_threshold=self.gamma_threshold,
            max_gamma_iter=self.max_gamma_iter,
            fit_diagnostics=diagnostics,
        )


class LabeledLDADataEncoder(DataSequenceEncoder):
    """Encode iid labeled LDA observations for vectorized scoring."""

    def __init__(self, encoder, num_alphas=None):
        """Create an encoder for labeled LDA documents.

        Args:
                encoder (DataSequenceEncoder): Encoder of type ``T`` for document values.

        Attributes:
                encoder (DataSequenceEncoder): Encoder of type ``T`` for document values.

        """
        self.encoder = encoder
        self.num_alphas = None if num_alphas is None else _positive_iteration_budget(num_alphas, "num_alphas")

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
            return self.encoder == other.encoder and self.num_alphas == other.num_alphas
        else:
            return False

    def seq_encode(self, x):
        """Encode labeled documents under the canonical nonempty-label schema."""
        if isinstance(x, (str, bytes)):
            raise TypeError("labeled-LDA data must be a sequence of (document, labels) observations")
        documents = []
        label_sets = []
        for document_index, observation in enumerate(x):
            if isinstance(observation, (str, bytes)):
                raise TypeError(f"labeled-LDA observation {document_index} must contain a document and labels")
            try:
                document, labels = observation
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"labeled-LDA observation {document_index} must contain exactly a document and labels"
                ) from exc
            documents.append(document)
            label_sets.append(
                _canonical_label_set(
                    labels,
                    self.num_alphas,
                    context=f"labeled-LDA observation {document_index} label set",
                )
            )
        num_documents, idx, counts, gammas, enc_data = LDADataEncoder(self.encoder).seq_encode(documents)
        nbcnt = np.asarray([len(labels) for labels in label_sets], dtype=np.intp)
        nbx = np.asarray([label for labels in label_sets for label in labels], dtype=np.intp)
        nbidx = np.repeat(np.arange(num_documents, dtype=np.intp), nbcnt)
        return num_documents, idx, counts, gammas, enc_data, nbx, nbcnt, nbidx

    def row_count(self, x):
        """Return the validated number of document rows in an encoded payload."""
        if not isinstance(x, tuple) or len(x) != 8:
            raise ValueError("encoded labeled-LDA data must be an eight-item tuple")
        value = x[0]
        if isinstance(value, (bool, np.bool_, float, np.floating)):
            raise TypeError("encoded labeled-LDA document count must be an integer")
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("encoded labeled-LDA document count must be an integer") from exc
        if result < 0:
            raise ValueError("encoded labeled-LDA document count must be non-negative")
        return result


def update_alpha(
    current_alpha,
    mean_log_p,
    alpha_threshold,
    *,
    max_iter=1000,
    return_diagnostics=False,
):
    """Run bounded independent Dirichlet updates for single-label alpha rows."""
    alpha = np.asarray(current_alpha, dtype=np.float64)
    mean_logs = np.asarray(mean_log_p, dtype=np.float64)
    if alpha.ndim != 2 or alpha.shape[0] == 0 or alpha.shape[1] == 0:
        raise ValueError("current_alpha must be a nonempty two-dimensional matrix")
    if mean_logs.shape != alpha.shape:
        raise ValueError("mean_log_p must have the same geometry as current_alpha")
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("current_alpha must be positive and finite")
    if np.any(~np.isfinite(mean_logs)):
        raise ValueError("mean_log_p must be finite")
    threshold = _positive_finite_threshold(alpha_threshold, "alpha_threshold")
    budget = _positive_iteration_budget(max_iter, "max_iter")

    result = np.empty_like(alpha)
    row_diagnostics = []
    for row_index in range(alpha.shape[0]):
        result[row_index], _, diagnostics = _update_lda_alpha(
            alpha[row_index],
            mean_logs[row_index],
            threshold,
            max_iter=budget,
            return_diagnostics=True,
        )
        row_diagnostics.append(diagnostics)
    diagnostics = LDAOptimizationDiagnostics(
        algorithm="labeled_lda_independent_alpha_fixed_point",
        converged=True,
        iterations=max(item.iterations for item in row_diagnostics),
        max_iterations=budget,
        termination_reason="converged",
        final_residual=max(item.final_residual for item in row_diagnostics),
    )
    if return_diagnostics:
        return result, diagnostics
    return result


@deprecated_alias("update_alpha", since="0.8.0", removed_in="0.10.0")
def updateAlpha(current_alpha, mean_log_p, alpha_threshold):
    """Deprecated alias for update_alpha()."""
    return update_alpha(current_alpha, mean_log_p, alpha_threshold)


def label_set_membership(label_sets, num_alphas=None):
    """Returns flattened membership arrays for a sequence of label sets.

    Args:
            label_sets (Sequence[Tuple[int, ...]]): Label-set tuples (one per distinct document label set).

    Returns:
            Tuple of the flattened label indices (member_label), the label-set index of each flattened entry
            (member_set), and the per-set sizes ``|S|`` as floats.

    """
    canonical = [
        _canonical_label_set(labels, num_alphas, context=f"coupled alpha label set {index}")
        for index, labels in enumerate(label_sets)
    ]
    set_sizes = np.asarray([len(labels) for labels in canonical], dtype=np.intp)
    member_label = np.asarray([label for labels in canonical for label in labels], dtype=np.intp)
    member_set = np.repeat(np.arange(len(canonical), dtype=np.intp), set_sizes)
    return member_label, member_set, set_sizes.astype(float)


def coupled_alpha_doc_params(alpha, label_sets):
    """Returns the per-label-set Dirichlet parameters a_S = mean_{l in S} alpha[l].

    Args:
            alpha (np.ndarray): Alphas matrix (num_alphas by num_topics).
            label_sets (Sequence[Tuple[int, ...]]): Label-set tuples.

    Returns:
            Numpy 2-d array with one Dirichlet parameter row per label set.

    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 2 or alpha.shape[0] == 0 or alpha.shape[1] == 0:
        raise ValueError("coupled alpha must be a nonempty two-dimensional matrix")
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("coupled alpha must be positive and finite")
    member_label, member_set, set_sizes = label_set_membership(label_sets, alpha.shape[0])
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
    alpha = np.asarray(alpha, dtype=np.float64)
    counts = np.asarray(set_counts, dtype=np.float64)
    mean_logs = np.asarray(set_mean_logs, dtype=np.float64)
    if counts.shape != (len(label_sets),) or np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("coupled alpha set counts must be a finite non-negative vector")
    if mean_logs.shape != (len(label_sets), alpha.shape[1]) or np.any(~np.isfinite(mean_logs)):
        raise ValueError("coupled alpha mean-log statistics have invalid geometry or values")
    a = coupled_alpha_doc_params(alpha, label_sets)
    return float(np.dot(counts, gammaln(a.sum(axis=1)) - gammaln(a).sum(axis=1) + (a * mean_logs).sum(axis=1)))


def coupled_alpha_gradient(alpha, label_sets, set_counts, set_mean_logs):
    """Gradient of the coupled multi-label alpha objective with respect to alpha.

    ``dF/d alpha[l,k] = sum_{S contains l} (n_S/|S|) * [ psi(sum_j a_Sj) - psi(a_Sk) + mbar_Sk ]``, with
    one term per occurrence of l in S.

    Args:
            alpha (np.ndarray): Alphas matrix (num_alphas by num_topics).
            label_sets (Sequence[Tuple[int, ...]]): Label-set tuples.
            set_counts (np.ndarray): Per-set document weights n_S.
            set_mean_logs (np.ndarray): Per-set mean expected log topic weights mbar_S (one row per set).

    Returns:
            Numpy 2-d array with the same shape as alpha.

    """
    alpha = np.asarray(alpha, dtype=np.float64)
    counts = np.asarray(set_counts, dtype=np.float64)
    mean_logs = np.asarray(set_mean_logs, dtype=np.float64)
    if counts.shape != (len(label_sets),) or np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("coupled alpha set counts must be a finite non-negative vector")
    if mean_logs.shape != (len(label_sets), alpha.shape[1]) or np.any(~np.isfinite(mean_logs)):
        raise ValueError("coupled alpha mean-log statistics have invalid geometry or values")
    member_label, member_set, set_sizes = label_set_membership(label_sets, alpha.shape[0])
    a = coupled_alpha_doc_params(alpha, label_sets)
    g_set = digamma(a.sum(axis=1, keepdims=True)) - digamma(a) + mean_logs
    g_set *= np.reshape(counts / set_sizes, (-1, 1))
    g = np.zeros(alpha.shape)
    np.add.at(g, member_label, g_set[member_set, :])
    return g


def update_alpha_coupled(
    current_alpha,
    label_sets,
    set_counts,
    set_mean_logs,
    alpha_threshold,
    max_its=2000,
    *,
    return_diagnostics=False,
):
    """Maximize the validated coupled label-row objective with bounded ascent."""
    alpha = np.asarray(current_alpha, dtype=np.float64)
    if alpha.ndim != 2 or alpha.shape[0] == 0 or alpha.shape[1] == 0:
        raise ValueError("current_alpha must be a nonempty two-dimensional matrix")
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("current_alpha must be positive and finite")
    canonical_sets = [
        _canonical_label_set(labels, alpha.shape[0], context=f"coupled alpha label set {index}")
        for index, labels in enumerate(label_sets)
    ]
    if not canonical_sets:
        raise ValueError("coupled alpha optimization requires at least one observed label set")
    counts = np.asarray(set_counts, dtype=np.float64)
    mean_logs = np.asarray(set_mean_logs, dtype=np.float64)
    threshold = _positive_finite_threshold(alpha_threshold, "alpha_threshold")
    budget = _positive_iteration_budget(max_its, "max_its")
    f_cur = coupled_alpha_objective(alpha, canonical_sets, counts, mean_logs)
    beta = np.log(alpha)
    step = 1.0
    residual = float("inf")
    objective_trace = [f_cur]
    termination_reason = "iteration_budget_exhausted"
    converged = False
    iterations = 0

    for iterations in range(1, budget + 1):
        g_beta = coupled_alpha_gradient(alpha, canonical_sets, counts, mean_logs) * alpha
        g_sq = float(np.sum(g_beta * g_beta))
        if not np.isfinite(g_sq):
            termination_reason = "invalid_gradient"
            break
        if g_sq <= threshold * threshold:
            residual = float(np.sqrt(g_sq))
            termination_reason = "stationary"
            converged = True
            break
        trial_step = step
        accepted = False
        while trial_step >= 1.0e-16:
            beta_new = np.clip(beta + trial_step * g_beta, -300.0, 300.0)
            alpha_new = np.exp(beta_new)
            f_new = coupled_alpha_objective(alpha_new, canonical_sets, counts, mean_logs)
            if np.isfinite(f_new) and f_new >= f_cur + 1.0e-4 * trial_step * g_sq:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            residual = float(np.sqrt(g_sq))
            termination_reason = "line_search_failed"
            break
        residual = float(np.max(np.abs(alpha_new - alpha).sum(axis=1) / alpha_new.sum(axis=1)))
        alpha = alpha_new
        beta = beta_new
        f_cur = f_new
        objective_trace.append(float(f_new))
        step = min(trial_step * 2.0, 1.0e8)
        if residual <= threshold:
            termination_reason = "converged"
            converged = True
            break

    diagnostics = LDAOptimizationDiagnostics(
        algorithm="labeled_lda_coupled_alpha_ascent",
        converged=converged,
        iterations=iterations,
        max_iterations=budget,
        termination_reason=termination_reason,
        final_residual=residual,
        objective_trace=tuple(objective_trace),
    )
    if not converged:
        raise LDAConvergenceError(diagnostics)
    if return_diagnostics:
        return alpha, diagnostics
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


def alpha_seq_lambda(meanLogP):
    """Returns the alpha fixed-point map for mean expected log topic weights meanLogP."""

    def next_alpha(currentAlpha):
        return digammainv(meanLogP + digamma(currentAlpha.sum()))

    return next_alpha


def find_alpha(current_alpha, mlp, thresh, *, max_iter=1000):
    """Find the alpha fixed point for mean expected log topic weights mlp via MPE.

    Args:
            current_alpha (np.ndarray): Starting alpha value.
            mlp (np.ndarray): Mean expected log topic weights.
            thresh (float): Convergence threshold.

    Returns:
            Tuple of the extrapolated alpha and the iteration count.

    """
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh, max_iter=max_iter)


def seq_posterior_with_diagnostics(estimate, x):
    """Return labeled-LDA variational quantities and their termination record."""
    num_documents, idx, counts, gammas, enc_data, nbx, nbcnt, nbidx = _validate_labeled_lda_encoded(
        x, estimate.nTopics, estimate.num_alpha
    )
    topic_scores = np.asarray(
        [topic.seq_log_density(enc_data) for topic in estimate.topics], dtype=np.float64
    ).transpose()
    if topic_scores.shape != (idx.size, estimate.nTopics):
        raise ValueError("labeled-LDA topic scorers returned arrays with invalid geometry")
    document_alphas = np.zeros((num_documents, estimate.nTopics), dtype=np.float64)
    for topic_index in range(estimate.nTopics):
        document_alphas[:, topic_index] = np.bincount(
            nbidx, weights=estimate.alphas[nbx, topic_index], minlength=num_documents
        )
    document_alphas /= nbcnt[:, None]
    responsibilities, final_gammas, diagnostics = _lda_vi_fixed_point(
        document_alphas,
        idx,
        counts,
        gammas,
        estimate.nTopics,
        topic_scores,
        estimate.gamma_threshold,
        estimate.max_gamma_iter,
        return_diagnostics=True,
    )
    return responsibilities, final_gammas, document_alphas, topic_scores, diagnostics


def seq_posterior(estimate, x):
    """Compute the validated variational posterior for encoded labeled documents."""
    responsibilities, final_gammas, document_alphas, topic_scores, _ = seq_posterior_with_diagnostics(estimate, x)
    return responsibilities, final_gammas, document_alphas, topic_scores


# --- Backward-compatible API naming aliases ---
LabeledLDAAccumulator = LabeledLDAEstimatorAccumulator
LabeledLDAAccumulatorFactory = LabeledLDAEstimatorAccumulatorFactory
