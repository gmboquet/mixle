"""Latent Dirichlet allocation for grouped-count documents.

LDA is a generative model for producing draws from multinomial topic mixtures. The process for generating a document of
length N from an LDA with L topics is given as follows:

    (1) Draw theta ~ Dirichlet(alpha) (alpha is L dimensional)
    (2) Draw topic-counts z_1,....,z_L ~ Multinomial(N, theta)
    (3) From each topic l = 1,2,...,L draw z_l words w_{i,l}, w_{i+1,l},...,w_{z_l,l} ~ Categorical(beta_l),
        where each topic has its own Categorical distribution parameterized by beta_l.

A document is then given by the bag of words produced from this sampling process. Note that a length distribution is
used to sample the number of words in a given document.

"""

import sys
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, gammaln, logsumexp

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.compute.posterior import MeanFieldLDAPosterior
from mixle.utils.special import digammainv
from mixle.utils.vector import row_choice

E0 = TypeVar("E0")
SS0 = TypeVar("SS0")

# import mixle.c_ext


class LDADistribution(SequenceEncodableProbabilityDistribution):
    """Latent Dirichlet allocation model for documents given as bags of weighted values.

    Data type: Sequence[Tuple[T, float]], where T is the data type of the topic distributions and each
    (value, count) pair gives the count of a value in the document.

    """

    def __init__(
        self,
        topics: Sequence[SequenceEncodableProbabilityDistribution],
        alpha: Sequence[float] | np.ndarray,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        gamma_threshold: float = 1.0e-8,
        max_gamma_iter: int = 100,
    ) -> None:
        """Create a latent Dirichlet allocation distribution.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions for the LDA.
            alpha (Union[Sequence[float], np.ndarray]): Dirichlet prior concentration for document-topic proportions.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for length of documents.
                Must be set to non-negative support distribution for sampling.
            gamma_threshold (float): Convergence threshold for the per-document variational gamma fixed point.
            max_gamma_iter (int): Hard cap on per-document variational iterations. The fixed point converges
                geometrically, so a few straggler documents would otherwise chase ``gamma_threshold`` for
                thousands of iterations at negligible gain; capping bounds the worst case (default 100).

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions for the LDA.
            alpha (np.ndarray): Dirichlet prior concentration for document-topic proportions.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for length of documents.
                Must be set to non-negative support distribution for sampling. Default to NullDistribution.
            gamma_threshold (float): Convergence threshold for the per-document variational gamma fixed point.
            max_gamma_iter (int): Hard cap on per-document variational iterations.

        """
        self.topics = topics
        self.n_topics = len(topics)
        self.alpha = np.asarray(alpha)
        self.len_dist = len_dist
        self.gamma_threshold = gamma_threshold
        self.max_gamma_iter = int(max_gamma_iter)

    def compute_capabilities(self):
        """Return backend capability metadata for this concrete LDA instance."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = tuple(self.topics)
        if self.len_dist is not None and not supports(self.len_dist, Neutral):
            children = children + (self.len_dist,)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def compute_declaration(self):
        """Return the generated-compute declaration for latent Dirichlet allocation."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        topic_children = tuple(declaration_for(topic) for topic in self.topics)
        length = None if self.len_dist is None or supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = tuple(
            child for child in topic_children + ((length,) if length is not None else ()) if child is not None
        )
        roles = tuple("topic_%d" % i for i, child in enumerate(topic_children) if child is not None)
        if length is not None:
            roles += ("length",)
        return DistributionDeclaration(
            name="lda",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("alpha", constraint="positive_vector"),
                ParameterSpec("gamma_threshold", constraint="positive", differentiable=False),
            ),
            statistics=(
                StatisticSpec("previous_alpha", kind="metadata", additive=False, scales=False),
                StatisticSpec("sum_of_logs"),
                StatisticSpec("document_count"),
                StatisticSpec("topic_counts"),
                StatisticSpec("topics", kind="tuple"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="lda_document_bag",
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        return "LDADistribution([%s], [%s])" % (",".join([str(u) for u in self.topics]), ",".join(map(str, self.alpha)))

    def density(self, x: Sequence[tuple[int, float]]) -> float:
        """Evaluate the density of a single LDA document.

        See log_density() for details.

        Args:
            x (Sequence[Tuple[int, float]]): A document given as (value, count) pairs.

        Returns:
            Density evaluated at x.

        """
        return np.exp(self.log_density(x))

    def density_semantics(self):
        """Return density semantics for the variational LDA document bound."""
        from mixle.stats.compute.pdist import DensitySemantics

        return DensitySemantics.LOWER_BOUND  # per-document variational ELBO, not the exact marginal

    def log_density(self, x: Sequence[tuple[int, float]]) -> float:
        """Evaluate the log-density of a single LDA document.

        Note: The returned value is the variational lower bound (ELBO) on the marginal document
        log-likelihood obtained from the standard LDA mean-field approximation, not the exact
        (intractable) marginal log-likelihood.

        Args:
            x (Sequence[Tuple[int, float]]): A document given as (value, count) pairs.

        Returns:
            Variational lower bound on the log-density evaluated at x.

        """
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_log_density(enc_x)[0]

    def seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0]) -> np.ndarray:
        """Vectorized evaluation of the document log-densities for an encoded corpus x.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            x[0] (int): Number of documents in corpus.
            x[1] (np.ndarray): Document id for flattened array of values.
            x[2] (np.ndarray): Flattened array of counts for each value in each document.
            x[3] (Optional[np.ndarray]): Optional warm-start gammas (defaults to None).
            x[4] (E0): Sequence encoded flattened values.

        Note: Returns the per-document variational lower bound (ELBO); see log_density(). If a
        document-length distribution 'len_dist' is set, its log-density of the total token count
        of each document is added to the returned values.

        Args:
            x: Encoded corpus of LDA documents (see LDADataEncoder.seq_encode()).

        Returns:
            Numpy array of log-density (ELBO) values, one entry per document.

        """
        num_topics = self.n_topics
        alpha = self.alpha
        num_documents, idx, counts, _, enc_data = x

        log_density_gamma, document_gammas, per_topic_log_densities = seq_posterior(self, x)

        elob = _lda_elbo_from_gamma(
            alpha, idx, counts, num_topics, log_density_gamma, document_gammas, per_topic_log_densities
        )

        if self.len_dist is not None and not supports(self.len_dist, Neutral):
            doc_lens = np.bincount(idx, weights=counts, minlength=num_documents)
            len_enc = self.len_dist.dist_to_encoder().seq_encode(doc_lens)
            elob += self.len_dist.seq_log_density(len_enc)

        return elob

    def _backend_seq_posterior(
        self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Run LDA's variational posterior update using the active backend."""
        from mixle.stats.compute.backend import backend_seq_log_density

        alpha = engine.asarray(self.alpha)
        num_documents, idx, counts, gammas, enc_data = x
        num_topics = self.n_topics
        idx_np = np.asarray(idx, dtype=np.int64)
        idx_full_np = (idx_np[:, None] * num_topics + np.arange(num_topics, dtype=np.int64)).reshape(-1)
        idx_full = engine.asarray(idx_full_np)
        idx_backend = engine.asarray(idx_np)
        counts_backend = engine.asarray(counts)

        per_topic_scores = [backend_seq_log_density(topic, enc_data, engine) for topic in self.topics]
        per_topic_log_densities = engine.stack(per_topic_scores, axis=1)
        centered_topic_log_densities = per_topic_log_densities - engine.max(per_topic_log_densities, axis=1).reshape(
            (-1, 1)
        )
        per_topic_weights = engine.exp(centered_topic_log_densities)

        if gammas is None:
            init_counts = engine.bincount(
                idx_full,
                weights=engine.asarray(np.ones(len(idx_full_np), dtype=np.float64)),
                minlength=num_documents * num_topics,
            ).reshape((num_documents, num_topics))
            document_gammas = init_counts / float(num_topics) + alpha
        else:
            document_gammas = engine.asarray(gammas)

        for _ in range(self.max_gamma_iter):
            digamma_gammas = engine.digamma(document_gammas)
            centered_gammas = digamma_gammas - engine.max(digamma_gammas, axis=1).reshape((-1, 1))
            gamma_weights = engine.exp(centered_gammas)
            row_weights = per_topic_weights * gamma_weights[idx_backend, :]
            row_weight_sum = engine.sum(row_weights, axis=1).reshape((-1, 1))
            log_density_gamma = row_weights / row_weight_sum * counts_backend.reshape((-1, 1))
            gamma_updates = engine.bincount(
                idx_full,
                weights=log_density_gamma.reshape((-1,)),
                minlength=num_documents * num_topics,
            ).reshape((num_documents, num_topics))
            gamma_updates = gamma_updates + alpha
            rel_diff = engine.sum(engine.abs(document_gammas - gamma_updates), axis=1) / engine.sum(
                gamma_updates, axis=1
            )
            document_gammas = gamma_updates
            if float(np.max(engine.to_numpy(rel_diff))) <= self.gamma_threshold:
                break

        return log_density_gamma, document_gammas, per_topic_log_densities

    def backend_seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0], engine: Any) -> Any:
        """Backend-neutral LDA variational lower-bound scoring."""
        from mixle.stats.compute.backend import backend_seq_log_density

        alpha = engine.asarray(self.alpha)
        num_topics = self.n_topics
        num_documents, idx, counts, _, _ = x
        idx_backend = engine.asarray(idx)
        counts_backend = engine.asarray(counts)
        idx_full_np = (
            np.asarray(idx, dtype=np.int64)[:, None] * num_topics + np.arange(num_topics, dtype=np.int64)
        ).reshape(-1)

        log_density_gamma, document_gammas, per_topic_log_densities = self._backend_seq_posterior(x, engine)

        tiny = (
            1.0e-30
            if getattr(engine, "precision", "default") in ("float16", "float32", "bfloat16")
            else sys.float_info.min
        )
        bad_gamma = engine.isnan(log_density_gamma) | engine.isinf(log_density_gamma) | (log_density_gamma <= 0)
        log_density_gamma = engine.where(bad_gamma, engine.asarray(tiny), log_density_gamma)
        bad_docs = engine.isnan(document_gammas) | engine.isinf(document_gammas)
        document_gammas = engine.where(bad_docs, engine.asarray(tiny), document_gammas)

        gamma_sum = engine.sum(document_gammas, axis=1).reshape((-1, 1))
        elob0 = engine.digamma(document_gammas) - engine.digamma(gamma_sum)
        elob1 = elob0[idx_backend, :]
        elob2 = log_density_gamma * (
            elob1
            + per_topic_log_densities
            - engine.log(log_density_gamma)
            + engine.log(counts_backend.reshape((-1, 1)))
        )
        elob3 = engine.sum(elob0 * ((alpha - 1.0) - (document_gammas - 1.0)), axis=1)
        elob4 = engine.bincount(
            engine.asarray(idx_full_np),
            weights=elob2.reshape((-1,)),
            minlength=num_documents * num_topics,
        ).reshape((num_documents, num_topics))
        elob5 = engine.sum(elob4, axis=1)
        elob6 = engine.sum(engine.gammaln(document_gammas), axis=1) - engine.gammaln(
            engine.sum(document_gammas, axis=1)
        )
        elob7 = engine.gammaln(engine.sum(alpha)) - engine.sum(engine.gammaln(alpha))

        elob = elob3 + elob5 + elob6 + elob7

        if self.len_dist is not None and not supports(self.len_dist, Neutral):
            doc_lens = engine.bincount(idx_backend, weights=counts_backend, minlength=num_documents)
            len_enc = self.len_dist.dist_to_encoder().seq_encode(engine.to_numpy(doc_lens))
            elob = elob + backend_seq_log_density(self.len_dist, len_enc, engine)

        return elob

    def seq_component_log_density(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0]) -> np.ndarray:
        """Vectorized evaluation of the per-topic log-density of each document in encoded corpus x.

        Args:
            x: Encoded corpus of LDA documents (see LDADataEncoder.seq_encode()).

        Returns:
            2-d numpy array with shape (number of documents, n_topics), where entry (i, l) is the
            log-density of document i evaluated entirely under topic l.

        """
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

    def backend_seq_component_log_density(
        self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0], engine: Any
    ) -> Any:
        """Backend-neutral per-topic document scores."""
        from mixle.stats.compute.backend import backend_seq_log_density

        num_documents, idx, counts, _, enc_data = x
        idx_backend = engine.asarray(idx)
        counts_backend = engine.asarray(counts)
        topic_scores = []
        for topic in self.topics:
            row_scores = backend_seq_log_density(topic, enc_data, engine) * counts_backend
            topic_scores.append(engine.bincount(idx_backend, weights=row_scores, minlength=num_documents))
        return engine.stack(topic_scores, axis=1)

    def seq_posterior(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0]) -> np.ndarray:
        """Vectorized evaluation of the posterior topic proportions for each document in encoded corpus x.

        The variational gammas are computed for each document and normalized to sum to one.

        Args:
            x: Encoded corpus of LDA documents (see LDADataEncoder.seq_encode()).

        Returns:
            2-d numpy array with shape (number of documents, n_topics) containing posterior topic
            proportions for each document.

        """
        num_topics = self.n_topics
        alpha = self.alpha
        num_documents, idx, counts, _, enc_data = x

        log_density_gamma, document_gammas, per_topic_log_densities = seq_posterior(self, x)

        document_gammas /= document_gammas.sum(axis=1, keepdims=True)

        return document_gammas

    def latent_posterior(self, doc: Sequence[tuple[int, float]]) -> "MeanFieldLDAPosterior":
        """Return the mean-field variational posterior ``q(theta, z)`` for a single document.

        Runs the per-document Blei-Ng-Jordan variational fixed point and returns a
        :class:`~mixle.stats.compute.posterior.MeanFieldLDAPosterior`: ``.topic_proportions()`` (the
        document-topic mix ``E[theta]``), ``.marginals()`` (per-word topic responsibilities ``phi``),
        ``.sample(rng)`` ``(theta, z)``, ``.mode()`` (MAP topic per word), or ``.entropy()``.
        """
        enc = self.dist_to_encoder().seq_encode([list(doc)])
        _, gammas, per_topic_log_densities = seq_posterior(self, enc)
        _, _, counts, _, _ = enc
        gamma = gammas[0]
        # phi at the variational fixed point: phi_wk prop. exp(E_q[log theta_k]) * p(word_w | topic_k)
        log_phi = (digamma(gamma) - digamma(gamma.sum()))[None, :] + per_topic_log_densities
        log_phi -= logsumexp(log_phi, axis=1, keepdims=True)
        return MeanFieldLDAPosterior(gamma, np.exp(log_phi), counts)

    def posterior_predictive(
        self, doc: Sequence[tuple[int, float]], n_words: int, seed: int | None = None
    ) -> list[Any]:
        """Draw ``n_words`` new words conditioned on the document ``doc``.

        Sample the document-topic mix ``theta ~ q(theta) = Dir(gamma)`` from the variational posterior,
        then generate each new word by drawing a topic ``~ theta`` and a word from that topic -- "given
        this document, generate more words from its inferred topic mixture".
        """
        rng = RandomState(seed)
        theta = rng.dirichlet(self.latent_posterior(doc).gamma)
        topic_samplers = [t.sampler(seed=rng.randint(maxrandint)) for t in self.topics]
        topics = rng.choice(self.n_topics, size=int(n_words), p=theta)
        return [topic_samplers[k].sample() for k in topics]

    def sampler(self, seed: int | None = None) -> "LDASampler":
        """Create a sampler for documents from this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator used in sampling.

        Returns:
            LDASampler: Sampler bound to this distribution.

        """
        return LDASampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LDAEstimator":
        """Create an estimator initialized from this distribution's topics.

        Args:
            pseudo_count (Optional[float]): If passed, used to re-weight sufficient statistics
                during estimation.

        Returns:
            LDAEstimator: Estimator configured with matching topic and length estimators.

        """
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)

        if pseudo_count is None:
            return LDAEstimator(
                estimators=[d.estimator() for d in self.topics],
                len_estimator=len_est,
                max_gamma_iter=self.max_gamma_iter,
            )
        else:
            return LDAEstimator(
                estimators=[d.estimator() for d in self.topics],
                len_estimator=len_est,
                pseudo_count=(pseudo_count, pseudo_count),
                max_gamma_iter=self.max_gamma_iter,
            )

    def dist_to_encoder(self) -> "LDADataEncoder":
        """Return a data encoder for iid LDA documents."""
        return LDADataEncoder(encoder=self.topics[0].dist_to_encoder())

    def enumerator(self) -> "DistributionEnumerator":  # noqa: F821  -- forward ref; LDA raises on enumerate
        """LDA does not support enumeration.

        The document log-density is a variational lower bound (ELBO) over latent topic
        assignments rather than an exact density, so an enumeration satisfying
        log_prob == log_density over a well-defined support cannot be constructed.

        Raises:
            EnumerationError: Always.

        """
        raise EnumerationError(
            self,
            reason="the LDA document log-density is a variational lower bound (ELBO) "
            "over latent topic assignments, not an exact density, so support "
            "enumeration is not well-defined",
        )


class LDASampler(DistributionSampler):
    """Sample documents from an LDA distribution."""

    def __init__(self, dist: LDADistribution, seed: int | None = None) -> None:
        """Create a sampler for an LDA distribution.

        Args:
            dist (LDADistribution): LDADistribution instance to sample from.
            seed (Optional[int]): Seed for the random number generator used in sampling.

        Attributes:
            rng (RandomState): Random number generator initialized from ``seed``.
            dist (LDADistribution): LDADistribution instance to sample from.
            n_topics (int): Number of topics in dist.
            comp_samplers (List[DistributionSampler]): Samplers for each topic distribution.
            dirichlet_sampler (DistributionSampler): Sampler for the topic-proportion Dirichlet prior.
            len_dist (DistributionSampler): Sampler for the document length distribution.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.n_topics = dist.n_topics
        self.comp_samplers = [
            self.dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in range(dist.n_topics)
        ]
        self.dirichlet_sampler = DirichletDistribution(dist.alpha).sampler(self.rng.randint(0, maxrandint))
        self.len_dist = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: int | None = None, *, batched: bool = True) -> Sequence[Any] | Any:
        """Draw one or 'size' documents from the LDA model.

        Note: Sample return value is not counted by value! Each document is returned as a flat list
        of sampled topic values (use mixle.utils.optsutil.count_by_value to obtain (value, count) pairs).

        With ``batched=True`` (default), when ``size`` is not None the per-document lengths, Dirichlet
        proportions and topic-count multinomials are drawn first, then every token across the whole
        batch is grouped by topic and each topic sampler is invoked once. Because the topic samplers
        are consumed in topic order rather than per-document order, the token draws are statistically
        equivalent but NOT byte-identical to ``batched=False``. The length, Dirichlet and multinomial
        draws are byte-identical (same order). Set ``batched=False`` to reproduce the exact legacy
        per-document output for a given seed.

        Args:
            size (Optional[int]): Number of documents to sample. If None, a single document is returned.
            batched (bool): Vectorize token draws across documents (default); set False for the legacy
                per-document loop.

        Returns:
            A single document (list of values) if size is None, else a list of 'size' documents.

        """
        if size is None:
            n = self.len_dist.sample()
            weights = self.dirichlet_sampler.sample()
            topic_counts = self.rng.multinomial(n, pvals=weights)
            rv = []
            for i in np.flatnonzero(topic_counts):
                rv.extend(self.comp_samplers[i].sample(size=int(topic_counts[i])))

            return rv

        if not batched:
            return [self.sample(batched=False) for i in range(size)]

        # Draw the structural variates per document (byte-identical order to the loop), then group
        # every token across all documents by topic and draw each topic sampler once.
        lengths = np.asarray(self.len_dist.sample(size=size)).astype(int).reshape(-1)
        per_doc_counts = []
        for n in lengths:
            weights = self.dirichlet_sampler.sample()
            per_doc_counts.append(self.rng.multinomial(int(n), pvals=weights))
        per_doc_counts = np.asarray(per_doc_counts).reshape(size, self.n_topics)

        # Number of tokens per topic across the whole batch, and a stable doc-major slot layout.
        docs: list[list[Any]] = [[] for _ in range(size)]
        for topic in range(self.n_topics):
            total = int(per_doc_counts[:, topic].sum())
            if total == 0:
                continue
            drawn = self.comp_samplers[topic].sample(size=total)
            offset = 0
            for d in range(size):
                c = int(per_doc_counts[d, topic])
                if c:
                    docs[d].extend(drawn[offset : offset + c])
                    offset += c
        return docs


class LDAEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics from observed LDA documents."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: tuple[str | None, str | None] | None = (None, None),
        prev_alpha: np.ndarray | None = None,
    ) -> None:
        """Create an accumulator for LDA sufficient statistics.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the topic
                distributions.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the
                document-length distribution (fed the total token count of each document).
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for merging the alpha sufficient
                statistics and the topic accumulators with matching objects.
            prev_alpha (Optional[np.ndarray]): Previous (or fixed) Dirichlet parameter estimate.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the topics.
            num_topics (int): Number of topic distributions.
            sum_of_logs (np.ndarray): Aggregated expected log topic proportions (length num_topics).
            doc_counts (float): Aggregated weighted document count.
            topic_counts (np.ndarray): Aggregated weighted per-topic value counts.
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the document-length
                distribution. Set to NullAccumulator if None is passed.
            prev_alpha (Optional[np.ndarray]): Previous Dirichlet parameter estimate.
            alpha_key (Optional[str]): Key for merging alpha sufficient statistics.
            topics_key (Optional[str]): Key for merging topic accumulators.

        """
        self.accumulators = accumulators
        self.num_topics = len(accumulators)
        self.sum_of_logs = np.zeros(self.num_topics)
        self.doc_counts = 0.0
        self.topic_counts = np.zeros(self.num_topics)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.prev_alpha = prev_alpha
        self.alpha_key, self.topics_key = keys if keys is not None else (None, None)

        # Per-document variational lower bound (ELBO) accumulated as a byproduct of the E-step,
        # only when _track_ll is enabled. Equals seq_log_density_sum(enc, dist)[1] and is consumed
        # by the fused-EM fast path in optimize(reuse_estep_ll=True); not part of value(). Off by
        # default so the standard path pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        self._init_rng = False
        self._rng_theta = None
        self._rng_idx = None
        self._rng_topics = None
        self._rng_len = None

    def update(self, x: Sequence[tuple[Any, float]], weight: float, estimate: LDADistribution) -> None:
        """Update sufficient statistics with a single weighted LDA document.

        Encodes the single observation and delegates to seq_update() so that the scalar and
        vectorized estimation paths agree.

        Args:
            x (Sequence[Tuple[Any, float]]): A document given as (value, count) pairs.
            weight (float): Weight for the observation.
            estimate (LDADistribution): Previous estimate of the LDA model.

        Returns:
            None.

        """
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize member random states for ``initialize`` and ``seq_initialize`` consistency.

        Args:
            rng (RandomState): Random state used to generate member seeds.

        Returns:
            None.

        """
        if not self._init_rng:
            seeds = rng.randint(maxrandint, size=3 + self.num_topics)
            self._rng_theta = RandomState(seed=seeds[0])
            self._rng_idx = RandomState(seed=seeds[1])
            self._rng_w = RandomState(seed=seeds[2])
            self._rng_topics = [RandomState(seed=seeds[3 + j]) for j in range(self.num_topics)]
            if not supports(self.len_accumulator, Neutral):
                self._rng_len = RandomState(seed=rng.randint(maxrandint))
            self._init_rng = True

    def seq_initialize(
        self,
        x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0],
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Vectorized initialization of sufficient statistics from an encoded corpus x.

        Topic assignments are drawn at random from a Dirichlet draw of topic proportions for
        each document.

        Args:
            x: Encoded corpus of LDA documents (see LDADataEncoder.seq_encode()).
            weights (np.ndarray): Weights for each document.
            rng (np.random.RandomState): Random state used to seed the accumulator initialization streams.

        Returns:
            None.

        """
        num_documents, idx, counts, old_gammas, enc_data = x

        if not self._init_rng:
            self._rng_initialize(rng)

        if self.prev_alpha is None:
            self.prev_alpha = np.ones(self.num_topics)

        theta = self._rng_theta.dirichlet(self.prev_alpha, size=num_documents)
        theta_rep = theta[idx, :]

        idx_list = row_choice(p_mat=np.reshape(theta_rep, (-1, self.num_topics)), rng=self._rng_idx)

        self.sum_of_logs += np.sum(np.log(theta), axis=0, keepdims=False)
        self.doc_counts += np.sum(weights)

        ww_v = -np.log(self._rng_w.rand(self.num_topics * len(idx)))
        ww_v[idx_list + np.arange(0, len(ww_v), self.num_topics)] += 1
        ww_v = np.reshape(ww_v, (-1, self.num_topics))
        ww_v /= ww_v.sum(axis=1, keepdims=True)

        temp = np.reshape(weights[idx] * counts, (len(idx), 1))
        ww_v *= temp

        for j in range(self.num_topics):
            w = ww_v[:, j]
            self.topic_counts[j] += np.sum(w)
            self.accumulators[j].seq_initialize(enc_data, w, self._rng_topics[j])

        if not supports(self.len_accumulator, Neutral):
            doc_lens = np.bincount(idx, weights=counts, minlength=num_documents)
            len_enc = self.len_accumulator.acc_to_encoder().seq_encode(doc_lens)
            self.len_accumulator.seq_initialize(len_enc, weights, self._rng_len)

    def initialize(self, x: Sequence[tuple[Any, float]], weight: float, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics with a single weighted LDA document.

        Args:
            x (Sequence[Tuple[Any, float]]): A document given as (value, count) pairs.
            weight (float): Weight for the observation.
            rng (np.random.RandomState): Random state used to seed the accumulator initialization streams.

        Returns:
            None.

        """
        if self.prev_alpha is None:
            self.prev_alpha = np.ones(self.num_topics)

        if not self._init_rng:
            self._rng_initialize(rng)

        counts = np.reshape([x[i][1] for i in range(len(x))], (len(x), 1))

        theta = self._rng_theta.dirichlet(self.prev_alpha)

        theta_rep = theta[np.arange(0, self.num_topics * len(x)) % self.num_topics]
        idx_list = row_choice(p_mat=np.reshape(theta_rep, (-1, self.num_topics)), rng=self._rng_idx)
        self.sum_of_logs += np.log(theta)
        self.doc_counts += weight

        ww_v = -np.log(self._rng_w.rand(self.num_topics * len(x)))
        ww_v[idx_list + np.arange(0, self.num_topics * len(x), self.num_topics)] += 1
        ww_v = np.reshape(ww_v, (-1, self.num_topics))
        ww_v /= np.sum(ww_v, axis=1, keepdims=True)
        ww_v *= counts * weight

        for j in range(self.num_topics):
            w = ww_v[:, j]
            for i in range(len(x)):
                self.accumulators[j].initialize(x[i][0], w[i], self._rng_topics[j])
                self.topic_counts[j] += w[i]

        if not supports(self.len_accumulator, Neutral):
            self.len_accumulator.initialize(np.sum(counts), weight, self._rng_len)

    def seq_update(
        self,
        x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0],
        weights: np.ndarray,
        estimate: LDADistribution,
    ) -> None:
        """Vectorized update of sufficient statistics from an encoded corpus x.

        Computes the variational posterior over topic assignments for each document under the
        previous estimate, then aggregates per-topic statistics, expected log topic proportions,
        and document counts.

        Args:
            x: Encoded corpus of LDA documents (see LDADataEncoder.seq_encode()).
            weights (np.ndarray): Weights for each document.
            estimate (LDADistribution): Previous estimate of the LDA model.

        Returns:
            None.

        """
        num_documents, idx, counts, old_gammas, enc_data = x
        log_density_gamma, final_gammas, per_topic_log_densities = seq_posterior(estimate, x)
        weighted_topic_counts = log_density_gamma * np.reshape(weights[idx], (-1, 1))

        for i in range(self.num_topics):
            self.accumulators[i].seq_update(enc_data, weighted_topic_counts[:, i], estimate.topics[i])

        mlpf = digamma(final_gammas) - digamma(np.sum(final_gammas, axis=1, keepdims=True))

        self.sum_of_logs += np.dot(weights, mlpf)
        self.doc_counts += weights.sum()
        self.topic_counts += np.sum(weighted_topic_counts, axis=0)
        self.prev_alpha = estimate.alpha

        # Fused-EM fast path: recover the per-document ELBO that estimate.seq_log_density would
        # return, reusing the variational quantities (gammas/responsibilities/per-topic densities)
        # the E-step already produced -- no second variational loop and no re-scoring of topics.
        # Mirrors LDADistribution.seq_log_density exactly (including the gamma-positivity cleanup
        # and the optional length-distribution term). Gated; standard path untouched.
        if self._track_ll:
            elob = _lda_elbo_from_gamma(
                estimate.alpha,
                idx,
                counts,
                self.num_topics,
                log_density_gamma,
                final_gammas,
                per_topic_log_densities,
            )

            if estimate.len_dist is not None and not supports(estimate.len_dist, Neutral):
                doc_lens = np.bincount(idx, weights=counts, minlength=num_documents)
                len_enc = estimate.len_dist.dist_to_encoder().seq_encode(doc_lens)
                elob = elob + estimate.len_dist.seq_log_density(len_enc)

            self._seq_ll += float(np.dot(weights, elob))

        if not supports(self.len_accumulator, Neutral):
            doc_lens = np.bincount(idx, weights=counts, minlength=num_documents)
            len_enc = self.len_accumulator.acc_to_encoder().seq_encode(doc_lens)
            self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def seq_update_engine(
        self,
        x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0],
        weights: np.ndarray,
        estimate: LDADistribution,
        engine: Any,
    ) -> None:
        """Engine-resident LDA E-step.

        The variational gamma loop (``_backend_seq_posterior``), the expected log topic
        proportions, and the topic-count aggregations all run on the active engine (numpy or
        torch); per-item topic responsibilities are produced on the engine and fed to the child
        topic accumulators. Matches host ``seq_update``.
        """
        num_documents, idx, counts, old_gammas, enc_data = x
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        idx_np = np.asarray(idx, dtype=np.int64)

        log_density_gamma, final_gammas, per_topic_log_densities = estimate._backend_seq_posterior(x, engine)

        w_idx = engine.asarray(weights_np[idx_np]).reshape((-1, 1))
        weighted_topic_counts = log_density_gamma * w_idx

        gamma_sum = engine.sum(final_gammas, axis=1).reshape((-1, 1))
        mlpf = engine.digamma(final_gammas) - engine.digamma(gamma_sum)
        w_doc = engine.asarray(weights_np).reshape((-1, 1))
        sum_of_logs = engine.sum(mlpf * w_doc, axis=0)
        topic_counts = engine.sum(weighted_topic_counts, axis=0)

        self.sum_of_logs += np.asarray(engine.to_numpy(sum_of_logs))
        self.doc_counts += float(weights_np.sum())
        self.topic_counts += np.asarray(engine.to_numpy(topic_counts))
        self.prev_alpha = estimate.alpha

        wtc_np = np.asarray(engine.to_numpy(weighted_topic_counts))
        for i in range(self.num_topics):
            self.accumulators[i].seq_update(enc_data, wtc_np[:, i], estimate.topics[i])

        if not supports(self.len_accumulator, Neutral):
            doc_lens = np.bincount(idx_np, weights=counts, minlength=num_documents)
            len_enc = self.len_accumulator.acc_to_encoder().seq_encode(doc_lens)
            self.len_accumulator.seq_update(len_enc, weights_np, estimate.len_dist)

    # return num_documents, idx, counts, final_gammas, enc_data

    def combine(
        self, suff_stat: tuple[np.ndarray | None, np.ndarray, float, np.ndarray, Sequence[SS0], Any | None]
    ) -> "LDAEstimatorAccumulator":
        """Combine the sufficient statistics of suff_stat with this accumulator.

        Arg suff_stat is a Tuple of length 6 containing:
            suff_stat[0] (Optional[np.ndarray]): Previous Dirichlet parameter estimate.
            suff_stat[1] (np.ndarray): Aggregated expected log topic proportions.
            suff_stat[2] (float): Aggregated weighted document count.
            suff_stat[3] (np.ndarray): Aggregated weighted per-topic value counts.
            suff_stat[4] (Sequence[SS0]): Sufficient statistics for each topic.
            suff_stat[5] (Optional[Any]): Sufficient statistics for the document-length distribution.

        Args:
            suff_stat: See above for details.

        Returns:
            LDAEstimatorAccumulator object.

        """
        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats, len_suff_stat = suff_stat

        if self.prev_alpha is None:
            self.prev_alpha = prev_alpha

        self.sum_of_logs += sum_of_logs
        self.doc_counts += doc_counts
        self.topic_counts += topic_counts

        for i in range(self.num_topics):
            self.accumulators[i].combine(topic_suff_stats[i])

        if len_suff_stat is not None:
            self.len_accumulator.combine(len_suff_stat)

        return self

    def value(self) -> tuple[np.ndarray | None, np.ndarray, float, np.ndarray, Sequence[Any], Any | None]:
        """Returns sufficient statistics as a Tuple (see combine() for entry details)."""
        return (
            self.prev_alpha,
            self.sum_of_logs,
            self.doc_counts,
            self.topic_counts,
            [u.value() for u in self.accumulators],
            self.len_accumulator.value(),
        )

    def from_value(
        self, x: tuple[np.ndarray | None, np.ndarray, float, np.ndarray, Sequence[SS0], Any | None]
    ) -> "LDAEstimatorAccumulator":
        """Set the sufficient statistics of this accumulator to x.

        Args:
            x: Sufficient statistic Tuple (see combine() for entry details).

        Returns:
            LDAEstimatorAccumulator object.

        """
        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats, len_suff_stat = x

        self.prev_alpha = prev_alpha
        self.sum_of_logs = sum_of_logs
        self.doc_counts = doc_counts
        self.topic_counts = topic_counts
        self.accumulators = [self.accumulators[i].from_value(topic_suff_stats[i]) for i in range(self.num_topics)]

        if len_suff_stat is not None:
            self.len_accumulator.from_value(len_suff_stat)

        return self

    def scale(self, c: float) -> "LDAEstimatorAccumulator":
        """Scale linear variational sufficient statistics while preserving previous alpha metadata."""
        self.sum_of_logs *= c
        self.doc_counts *= c
        self.topic_counts *= c
        for acc in self.accumulators:
            acc.scale(c)
        self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into keyed sufficient statistics.

        Merges alpha sufficient statistics when ``alpha_key`` is set, and topic accumulators when ``topics_key`` is set.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
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

        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics from matching keyed values.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
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

        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "LDADataEncoder":
        """Return a data encoder built from the topic accumulators."""
        return LDADataEncoder(encoder=self.accumulators[0].acc_to_encoder())


class LDAEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LDA estimator accumulators."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        dim: int,
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: tuple[str | None, str | None] | None = (None, None),
        prev_alpha: np.ndarray | None = None,
    ) -> None:
        """Create a factory for LDA estimator accumulators.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Factories for the topic accumulators.
            dim (int): Number of topics.
            len_factory (StatisticAccumulatorFactory): Factory for the document-length accumulator.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for the alpha sufficient
                statistics and the topic accumulators.
            prev_alpha (Optional[np.ndarray]): Previous (or fixed) Dirichlet parameter estimate.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): Factories for the topic accumulators.
            dim (int): Number of topics.
            len_factory (StatisticAccumulatorFactory): Factory for the document-length accumulator.
            keys (Tuple[Optional[str], Optional[str]]): Keys for the alpha sufficient statistics and
                the topic accumulators.
            prev_alpha (Optional[np.ndarray]): Previous (or fixed) Dirichlet parameter estimate.

        """
        self.factories = factories
        self.dim = dim
        self.len_factory = len_factory
        self.keys = keys if keys is not None else (None, None)
        self.prev_alpha = prev_alpha

    def make(self) -> "LDAEstimatorAccumulator":
        """Return a new LDA estimator accumulator."""
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return LDAEstimatorAccumulator(
            [self.factories[i].make() for i in range(self.dim)], len_acc, self.keys, self.prev_alpha
        )


class LDAEstimator(ParameterEstimator):
    """Estimate LDA distributions from aggregated variational sufficient statistics."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        len_estimator: ParameterEstimator | None = NullEstimator(),
        suff_stat: Any | None = None,
        pseudo_count: tuple[float, float] | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
        fixed_alpha: np.ndarray | None = None,
        gamma_threshold: float = 1.0e-8,
        alpha_threshold: float = 1.0e-8,
        max_gamma_iter: int = 100,
    ) -> None:
        """Create an estimator for LDA distributions.

        Args:
            estimators (Sequence[ParameterEstimator]): Estimators for the topic distributions.
            len_estimator (Optional[ParameterEstimator]): Estimator for the document-length distribution.
            suff_stat (Optional[Any]): Kept for consistency with ParameterEstimator interface.
            pseudo_count (Optional[Tuple[float, float]]): Prior mass used to smooth the alpha sufficient statistics.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for the alpha sufficient
                statistics and the topic accumulators.
            fixed_alpha (Optional[np.ndarray]): If passed, alpha is fixed to this value in estimation.
            gamma_threshold (float): Convergence threshold for the per-document gamma updates.
            alpha_threshold (float): Convergence threshold for the alpha update iteration.

        Attributes:
            num_topics (int): Number of topics.
            estimators (Sequence[ParameterEstimator]): Estimators for the topic distributions.
            len_estimator (ParameterEstimator): Estimator for the document-length distribution.
            pseudo_count (Optional[Tuple[float, float]]): Prior mass used to smooth alpha sufficient statistics.
            suff_stat (Optional[Any]): Kept for consistency with ParameterEstimator interface.
            keys (Tuple[Optional[str], Optional[str]]): Keys for the alpha sufficient statistics and
                the topic accumulators.
            gamma_threshold (float): Convergence threshold for the per-document gamma updates.
            alpha_threshold (float): Convergence threshold for the alpha update iteration.
            fixed_alpha (Optional[np.ndarray]): If set, alpha is fixed to this value in estimation.

        """
        self.num_topics = len(estimators)
        self.estimators = estimators
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None)
        self.gamma_threshold = gamma_threshold
        self.alpha_threshold = alpha_threshold
        self.fixed_alpha = fixed_alpha
        self.max_gamma_iter = int(max_gamma_iter)

    def accumulator_factory(self) -> "LDAEstimatorAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return LDAEstimatorAccumulatorFactory(est_factories, self.num_topics, len_factory, self.keys, self.fixed_alpha)

    def estimate(self, nobs: float | None, suff_stat) -> "LDADistribution":
        """Estimate an LDA distribution from aggregated sufficient statistics.

        ``suff_stat`` is a six-item tuple containing:
            suff_stat[0] (Optional[np.ndarray]): Previous Dirichlet parameter estimate.
            suff_stat[1] (np.ndarray): Aggregated expected log topic proportions.
            suff_stat[2] (float): Aggregated weighted document count.
            suff_stat[3] (np.ndarray): Aggregated weighted per-topic value counts.
            suff_stat[4] (Sequence[SS0]): Sufficient statistics for each topic.
            suff_stat[5] (Optional[Any]): Sufficient statistics for the document-length distribution.

        Args:
            nobs (Optional[float]): Weighted number of observations used in aggregation of suff_stat.
            suff_stat: See above for details.

        Returns:
            LDADistribution: Estimated distribution.

        """
        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats, len_suff_stat = suff_stat

        num_topics = self.num_topics
        topics = [self.estimators[i].estimate(topic_counts[i], topic_suff_stats[i]) for i in range(num_topics)]
        len_dist = self.len_estimator.estimate(nobs, len_suff_stat)

        if doc_counts == 0:
            sys.stderr.write("Warning: LDA Estimation performed with zero documents.\n")
            return LDADistribution(
                topics,
                prev_alpha,
                len_dist=len_dist,
                gamma_threshold=self.gamma_threshold,
                max_gamma_iter=self.max_gamma_iter,
            )

        if self.fixed_alpha is None:
            if self.pseudo_count is not None:
                mean_of_logs = (sum_of_logs + np.log(self.pseudo_count[1])) / (doc_counts + self.pseudo_count[0])
            else:
                mean_of_logs = sum_of_logs / doc_counts

            # new_alpha, _ = find_alpha(prev_alpha, sum_of_logs/doc_counts, gamma_threshold*np.sqrt(float(doc_counts)))
            new_alpha, _ = update_alpha(prev_alpha, mean_of_logs, self.alpha_threshold)
        else:
            new_alpha = np.asarray(self.fixed_alpha).copy()

        return LDADistribution(
            topics,
            new_alpha,
            len_dist=len_dist,
            gamma_threshold=self.gamma_threshold,
            max_gamma_iter=self.max_gamma_iter,
        )


class LDADataEncoder(DataSequenceEncoder):
    """Encode iid LDA documents for vectorized scoring."""

    def __init__(self, encoder: DataSequenceEncoder):
        """Create an encoder for LDA documents.

        Args:
            encoder (DataSequenceEncoder): Encoder for topic-distribution observations.

        Attributes:
            encoder (DataSequenceEncoder): Encoder for topic-distribution observations.

        """
        self.encoder = encoder

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "LDADataEncoder(encoder=" + str(self.encoder) + ")"

    def __eq__(self, other) -> bool:
        """Check if other is an equivalent LDADataEncoder (topic encoders must match).

        Args:
            other (object): Object to compare.

        Returns:
            True if other is equivalent.

        """
        if isinstance(other, LDADataEncoder):
            return self.encoder == other.encoder
        else:
            return False

    def seq_encode(
        self, x: Sequence[Sequence[tuple[int, float]]]
    ) -> tuple[int, np.ndarray, np.ndarray, Any | None, Any]:
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

        nx = np.fromiter((len(doc) for doc in x), dtype=np.intp, count=num_documents)
        tx = [pair[0] for doc in x for pair in doc]
        ctx = [pair[1] for doc in x for pair in doc]

        idx = np.repeat(np.arange(num_documents), nx)
        counts = np.asarray(ctx)
        gammas = None
        enc_data = self.encoder.seq_encode(tx)

        return num_documents, idx, counts, gammas, enc_data


def update_alpha(alpha_curr, mean_log_p, alpha_threshold) -> tuple[np.ndarray, int]:
    """Fixed-point update of the Dirichlet parameter alpha given mean expected log proportions.

    Args:
        alpha_curr (np.ndarray): Current alpha estimate.
        mean_log_p (np.ndarray): Mean expected log topic proportions across documents.
        alpha_threshold (float): Convergence threshold for the fixed-point iteration.

    Returns:
        Tuple of (updated alpha, number of iterations performed).

    """
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


def mpe_update(x_mat: np.ndarray | None, y: np.ndarray, min_size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Single minimal polynomial extrapolation (MPE) step for fixed-point sequence acceleration."""
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


def mpe(x0, f, eps: float) -> tuple[np.ndarray, int]:
    """Minimal polynomial extrapolation of the fixed point of f starting from x0.

    Args:
        x0: Initial point of the fixed-point iteration.
        f: Fixed-point map.
        eps (float): Convergence threshold on successive extrapolants.

    Returns:
        Tuple of (extrapolated fixed point, number of iterations performed).

    """
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
    """Returns the alpha fixed-point map for a given mean expected log topic proportion."""

    def next_alpha(alpha_current: np.ndarray):
        return digammainv(mean_log_p + digamma(alpha_current.sum()))

    return next_alpha


def find_alpha(current_alpha: np.ndarray, mlp: float, thresh: float):
    """Find the alpha fixed point via MPE acceleration (see update_alpha for the plain iteration)."""
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh)


def seq_posterior2(estimate: LDADistribution, x: tuple[int, np.ndarray, np.ndarray, Any | None, E0]):
    """C-extension variant of seq_posterior(). Requires the optional mixle.c_ext module."""
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
            num_topics
        )
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

    aa, bb = mixle.c_ext.lda_update(  # noqa: F821  -- optional mixle.c_ext module, imported by caller when present
        idx, document_gammas, rv1, rv2, alpha_loc, per_topic_log_densities, rv3, ccc, rv0, rv4, rv5, -1, gamma_threshold
    )

    final_gammas = bb + alpha_loc
    log_density_gamma = aa

    return log_density_gamma, final_gammas, per_topic_log_densities0


def _lda_vi_fixed_point(
    per_doc_alpha,
    idx,
    counts,
    gammas,
    num_topics,
    per_topic_log_densities,
    gamma_threshold,
    max_gamma_iter,
):
    """Shared per-document mean-field variational (gamma) fixed point for (Labeled)LDA.

    This is the Blei-Ng-Jordan per-document coordinate-ascent loop that both ``LDADistribution``
    and ``LabeledLDADistribution`` run; the only model difference is the per-document Dirichlet
    prior, supplied here as ``per_doc_alpha``:

        * Plain LDA passes the single shared ``alpha`` broadcast to one identical row per document.
        * Labeled LDA passes ``alphas_loc`` -- the per-document mean of its label rows.

    Because the prior enters only additively (``gamma <- alpha + sum phi``), a 2-d per-document
    array subsumes LDA's broadcast ``(1, num_topics)`` case exactly (identical rows stay identical
    under the document-subsetting the loop performs as documents converge).

    Args:
        per_doc_alpha (np.ndarray): Per-document Dirichlet parameter (num_documents by num_topics).
        idx (np.ndarray): Document id for each flattened value.
        counts (np.ndarray): Flattened per-value counts.
        gammas (Optional[np.ndarray]): Optional warm-start gammas (num_documents by num_topics).
        num_topics (int): Number of topics.
        per_topic_log_densities (np.ndarray): Per-value per-topic log-densities (num_samples by num_topics).
        gamma_threshold (float): Relative-change convergence threshold for the gamma updates.
        max_gamma_iter (int): Hard cap on the number of fixed-point iterations.

    Returns:
        Tuple of (log_density_gamma, final_gammas), where log_density_gamma has a row per flattened
        value with the count-scaled expected topic responsibilities and final_gammas has a row per
        document with the converged variational Dirichlet parameters.

    """
    num_documents = per_doc_alpha.shape[0]
    num_samples = len(idx)

    alphas_loc = per_doc_alpha
    alphas_loc2 = alphas_loc.copy()

    per_topic_log_densities2 = per_topic_log_densities.copy()
    per_topic_log_densities2 -= np.max(per_topic_log_densities2, axis=1, keepdims=True)
    np.exp(per_topic_log_densities2, out=per_topic_log_densities2)
    per_topic_log_densities3 = per_topic_log_densities2.copy()

    idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
    idx_full *= num_topics
    idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

    if gammas is None:
        document_gammas = alphas_loc + np.reshape(np.bincount(idx_full.flat), (num_documents, num_topics)) / float(
            num_topics
        )
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

    digamma(document_gammas, out=document_gammas2)
    temp = np.max(document_gammas2, axis=1, keepdims=True)
    np.exp(document_gammas2 - temp, out=document_gammas3)

    np.multiply(per_topic_log_densities2, document_gammas3[rel_idx, :], out=log_density_gamma_loc)
    np.sum(log_density_gamma_loc, axis=1, keepdims=True, out=posterior_sum_ll_loc)
    log_density_gamma_loc /= posterior_sum_ll_loc

    while ndoc > 0 and itr_cnt < max_gamma_iter:
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
        gamma_updates += alphas_loc2

        np.subtract(document_gammas, gamma_updates, out=document_gamma_diff_loc)
        np.abs(document_gamma_diff_loc, out=document_gamma_diff_loc)
        np.sum(document_gamma_diff_loc, axis=1, keepdims=True, out=gamma_asum_loc)
        np.sum(gamma_updates, axis=1, keepdims=True, out=gamma_sum_loc)
        gamma_asum_loc /= gamma_sum_loc

        document_gammas = gamma_updates

        has_finished = np.flatnonzero(gamma_asum_loc.flat <= gamma_threshold)

        if has_finished.size != 0:
            final_gammas[finished_count : (finished_count + len(has_finished)), :] = document_gammas[has_finished, :]
            final_gammas_idx[finished_count : (finished_count + len(has_finished))] = rem_gammas_idx[has_finished]
            gamma_itr_cnt[finished_count : (finished_count + len(has_finished))] = itr_cnt

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
            alphas_loc2 = alphas_loc2[is_rem_idx, :]

    # Cap reached while some documents were still iterating: their gammas are already converged to
    # far below what EM needs (geometric convergence), so flush the current values as the result.
    if ndoc > 0:
        final_gammas[finished_count : finished_count + ndoc, :] = document_gammas
        final_gammas_idx[finished_count : finished_count + ndoc] = rem_gammas_idx
        gamma_itr_cnt[finished_count : finished_count + ndoc] = itr_cnt
        finished_count += ndoc

    sidx = np.argsort(final_gammas_idx)
    final_gammas = final_gammas[sidx, :]
    gamma_itr_cnt = gamma_itr_cnt[sidx]

    digamma_gammas = digamma(final_gammas)
    temp2 = np.max(digamma_gammas, axis=1, keepdims=True)
    temp3 = np.exp(digamma_gammas - temp2)

    np.multiply(per_topic_log_densities3, temp3[idx, :], out=log_density_gamma)
    np.sum(log_density_gamma, axis=1, keepdims=True, out=posterior_sum_ll)
    posterior_sum_ll /= np.reshape(counts, (-1, 1))
    log_density_gamma /= posterior_sum_ll

    idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
    idx_full *= num_topics
    idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

    gamma_updates = np.bincount(idx_full.flat, weights=log_density_gamma.flat)
    gamma_updates = np.reshape(gamma_updates, (-1, num_topics))
    gamma_updates += alphas_loc
    final_gammas = gamma_updates

    return log_density_gamma, final_gammas


def _lda_elbo_from_gamma(
    per_doc_alpha,
    idx,
    counts,
    num_topics,
    log_density_gamma,
    document_gammas,
    per_topic_log_densities,
):
    """Per-document variational lower bound (ELBO) from converged variational quantities.

    Shared by ``LDADistribution`` and ``LabeledLDADistribution`` (host paths). ``per_doc_alpha`` is
    the per-document Dirichlet prior: a 1-d ``alpha`` (shape ``num_topics``) for plain LDA -- which
    broadcasts across documents -- or a 2-d ``(num_documents, num_topics)`` array of per-document
    label-row means for labeled LDA. Inputs are copied before the gamma-positivity cleanup so the
    caller's arrays are left untouched.

    Returns:
        Numpy array with one ELBO value per document (length num_documents). Any document-length or
        label-set terms are model-specific and added by the caller.

    """
    idx = np.asarray(idx)

    idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
    idx_full *= num_topics
    idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

    ldg = log_density_gamma.copy()
    dg = document_gammas.copy()
    ldg[np.bitwise_or(np.isnan(ldg), np.isinf(ldg))] = sys.float_info.min
    ldg[ldg <= 0] = sys.float_info.min
    dg[np.bitwise_or(np.isnan(dg), np.isinf(dg))] = sys.float_info.min

    elob0 = digamma(dg) - digamma(np.sum(dg, axis=1, keepdims=True))
    elob1 = elob0[idx, :]
    elob2 = ldg * (elob1 + per_topic_log_densities - np.log(ldg) + np.log(np.reshape(counts, (-1, 1))))
    elob3 = np.sum(elob0 * ((per_doc_alpha - 1.0) - (dg - 1.0)), axis=1)
    elob4 = np.bincount(idx_full.flat, weights=elob2.flat)
    elob5 = np.sum(np.reshape(elob4, (-1, num_topics)), axis=1)
    elob6 = np.sum(gammaln(dg), axis=1) - gammaln(dg.sum(axis=1))
    if per_doc_alpha.ndim == 1:
        elob7 = gammaln(per_doc_alpha.sum()) - gammaln(per_doc_alpha).sum()
    else:
        elob7 = gammaln(per_doc_alpha.sum(axis=1)) - gammaln(per_doc_alpha).sum(axis=1)

    return elob3 + elob5 + elob6 + elob7


def seq_posterior(estimate: LDADistribution, x: tuple[int, np.ndarray, np.ndarray, Any | None, E0]):
    """Run the per-document variational (gamma) fixed-point iteration for an encoded corpus.

    Args:
        estimate (LDADistribution): LDA model under which the posterior is computed.
        x: Encoded corpus of LDA documents (see LDADataEncoder.seq_encode()).

    Returns:
        Tuple of (log_density_gamma, final_gammas, per_topic_log_densities), where log_density_gamma
        has a row per flattened value with the expected topic-assignment weights (scaled by counts),
        final_gammas has a row per document with the converged variational Dirichlet parameters, and
        per_topic_log_densities has a row per flattened value with each topic's log-density.

    """
    alpha = estimate.alpha
    topics = estimate.topics
    gamma_threshold = estimate.gamma_threshold

    num_documents, idx, counts, gammas, enc_data = x

    num_topics = len(topics)

    per_topic_log_densities = np.asarray([topics[i].seq_log_density(enc_data) for i in range(num_topics)]).transpose()

    # Plain LDA's single shared alpha as one identical row per document (the degenerate case of the
    # labeled-LDA per-document prior); the shared fixed point subsets these rows as documents converge.
    per_doc_alpha = np.repeat(np.reshape(alpha, (1, num_topics)), num_documents, axis=0)

    log_density_gamma, final_gammas = _lda_vi_fixed_point(
        per_doc_alpha,
        idx,
        counts,
        gammas,
        num_topics,
        per_topic_log_densities,
        gamma_threshold,
        getattr(estimate, "max_gamma_iter", 100),
    )

    return log_density_gamma, final_gammas, per_topic_log_densities


def _register_lda_engine_kernel():
    """Register the engine-resident LDA kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class LDAKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("LDAKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class LDAKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return LDAKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(LDADistribution, LDAKernelFactory())


_register_lda_engine_kernel()


# --- Backward-compatible API naming aliases ---
LDAAccumulator = LDAEstimatorAccumulator
LDAAccumulatorFactory = LDAEstimatorAccumulatorFactory
