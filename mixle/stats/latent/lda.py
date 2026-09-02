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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from mixle.stats.latent.effective_sample import (
    validate_effective_sample_mass,
    validated_statistic_tuple,
)
from mixle.utils.aliasing import broadcast_pseudo_count
from mixle.utils.special import digammainv
from mixle.utils.vector import ImpossibleEvidenceError, row_choice

E0 = TypeVar("E0")
SS0 = TypeVar("SS0")

# Absolute tolerance for update_alpha()'s alpha-non-existence boundary check
# (logsumexp(mean_log_p) >= -_ALPHA_BOUNDARY_TOL). See the comment at its use for why this is
# needed: the idealized boundary is exactly 0, but the E-step's floating-point mean_log_p lands a
# few ULPs (~1e-16) to either side of it for a genuinely-degenerate corpus, and a bare ">= 0.0"
# check is not robust to which side it lands on.
_ALPHA_BOUNDARY_TOL = 1.0e-9

# import mixle.c_ext


@dataclass(frozen=True)
class LDAOptimizationDiagnostics:
    """Serializable termination record for an LDA inner optimization."""

    algorithm: str
    converged: bool
    iterations: int
    max_iterations: int
    termination_reason: str
    final_residual: float
    objective_trace: tuple[float, ...] = ()
    residual_trace: tuple[float, ...] = ()
    impossible_documents: tuple[int, ...] = ()

    # Every LDAEstimator.estimate() fit (and LabeledLDAEstimator.estimate(), which reuses this
    # same class) attaches this unconditionally, so without this flag
    # to_serializable()/to_json()/model_hash() raised an unhandled SerializationError for EVERY
    # fitted instance of either family (campaign nine, D-0209). Unannotated on purpose: an
    # annotated name would become a dataclass field.
    __pysp_serializable__ = True


class LDAConvergenceError(RuntimeError):
    """Raised when an LDA inner optimization cannot meet its declared contract."""

    def __init__(self, diagnostics: LDAOptimizationDiagnostics):
        self.diagnostics = diagnostics
        message = "%s did not converge after %d/%d iterations (%s; residual=%g)" % (
            diagnostics.algorithm,
            diagnostics.iterations,
            diagnostics.max_iterations,
            diagnostics.termination_reason,
            diagnostics.final_residual,
        )
        if diagnostics.termination_reason == "alpha_diverging":
            # This is not slow convergence -- alpha.sum() is provably unbounded for this corpus (see
            # the divergence check in update_alpha()), so no iteration budget will ever satisfy
            # alpha_threshold; the residual above only shrinks because it shares alpha's growing sum
            # as its denominator. Name the two escape hatches that actually work instead of leaving
            # the reader to conclude (wrongly) that raising max_alpha_iter is the fix.
            message += (
                ". alpha is diverging to infinity, not converging slowly: this corpus's mean "
                "expected log-topic-proportions cannot be matched by any finite Dirichlet alpha, so "
                "raising max_alpha_iter will not help. Pass fixed_alpha=<array> to LDAEstimator to "
                "skip the alpha solve entirely, or loosen alpha_threshold (e.g. 1e-3) to accept a "
                "softer convergence criterion."
            )
        super().__init__(message)


def _positive_finite_threshold(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite real scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_iteration_budget(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_, float, np.floating)):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_document_weights(weights: Any, num_documents: int) -> np.ndarray:
    raw = np.asarray(weights)
    if raw.dtype.kind == "b":
        raise TypeError("LDA document weights must be real-valued, not boolean")
    try:
        result = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("LDA document weights must be real-valued") from exc
    if result.ndim != 1 or result.shape[0] != num_documents:
        raise ValueError(f"LDA document weights must have shape ({num_documents},)")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("LDA document weights must be finite and non-negative")
    return result


def _validate_lda_encoded(x: Any, num_topics: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray | None, Any]:
    if not isinstance(x, tuple) or len(x) != 5:
        raise TypeError("encoded LDA data must be a five-item tuple")
    num_documents, raw_idx, raw_counts, raw_gammas, enc_data = x
    if isinstance(num_documents, (bool, np.bool_, float, np.floating)):
        raise TypeError("encoded LDA document count must be a non-negative integer")
    try:
        num_documents = int(num_documents)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("encoded LDA document count must be a non-negative integer") from exc
    if num_documents < 0:
        raise ValueError("encoded LDA document count must be non-negative")

    raw_idx_array = np.asarray(raw_idx)
    if raw_idx_array.dtype.kind == "b" or raw_idx_array.ndim != 1:
        raise TypeError("encoded LDA document IDs must be a one-dimensional exact-integer array")
    try:
        idx_numeric = np.asarray(raw_idx, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("encoded LDA document IDs must be exact integers") from exc
    if np.any(~np.isfinite(idx_numeric)) or np.any(idx_numeric != np.floor(idx_numeric)):
        raise ValueError("encoded LDA document IDs must be finite exact integers")
    idx = idx_numeric.astype(np.intp)
    if np.any(idx < 0) or np.any(idx >= num_documents):
        raise ValueError("encoded LDA document IDs are outside the declared corpus")
    if idx.size > 1 and np.any(idx[1:] < idx[:-1]):
        raise ValueError("encoded LDA document IDs must be in nondecreasing document order")

    raw_counts_array = np.asarray(raw_counts)
    if raw_counts_array.dtype.kind == "b":
        raise TypeError("encoded LDA counts must be real-valued, not boolean")
    try:
        counts = np.asarray(raw_counts, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("encoded LDA counts must be real-valued") from exc
    if counts.ndim != 1 or counts.shape != idx.shape:
        raise ValueError("encoded LDA counts and document IDs must be one-dimensional arrays of equal length")
    if np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
        raise ValueError("encoded LDA counts must be positive and finite")

    gammas = None
    if raw_gammas is not None:
        try:
            gammas = np.asarray(raw_gammas, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("LDA warm-start gammas must be real-valued") from exc
        if gammas.shape != (num_documents, num_topics):
            raise ValueError(f"LDA warm-start gammas must have shape ({num_documents}, {num_topics})")
        if np.any(~np.isfinite(gammas)) or np.any(gammas <= 0.0):
            raise ValueError("LDA warm-start gammas must be positive and finite")
        gammas = gammas.copy()
    return num_documents, idx, counts, gammas, enc_data


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
        fit_diagnostics: LDAOptimizationDiagnostics | None = None,
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
        if isinstance(topics, (str, bytes)) or len(topics) == 0:
            raise ValueError("LDA requires at least one topic distribution")
        alpha_array = np.asarray(alpha, dtype=np.float64)
        if alpha_array.ndim != 1 or alpha_array.shape[0] != len(topics):
            raise ValueError("LDA alpha must be a one-dimensional vector with one entry per topic")
        if np.any(~np.isfinite(alpha_array)) or np.any(alpha_array <= 0.0):
            raise ValueError("LDA alpha entries must be positive and finite")
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, LDAOptimizationDiagnostics):
            raise TypeError("fit_diagnostics must be an LDAOptimizationDiagnostics record or None")

        self.topics = tuple(topics)
        self.n_topics = len(topics)
        self.alpha = alpha_array.copy()
        self.len_dist = len_dist
        self.gamma_threshold = _positive_finite_threshold(gamma_threshold, "gamma_threshold")
        self.max_gamma_iter = _positive_iteration_budget(max_gamma_iter, "max_gamma_iter")
        self.fit_diagnostics = fit_diagnostics

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
        num_documents, idx, counts, _, enc_data = _validate_lda_encoded(x, self.n_topics)

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
        """Evaluate topic scores on the backend and run the validated VI contract."""
        from mixle.stats.compute.backend import backend_seq_log_density

        num_documents, idx, counts, gammas, enc_data = _validate_lda_encoded(x, self.n_topics)
        per_topic_scores = [backend_seq_log_density(topic, enc_data, engine) for topic in self.topics]
        per_topic_log_densities = engine.stack(per_topic_scores, axis=1)
        score_array = np.asarray(engine.to_numpy(per_topic_log_densities), dtype=np.float64)
        per_doc_alpha = np.repeat(self.alpha[None, :], num_documents, axis=0)
        responsibilities, document_gammas = _lda_vi_fixed_point(
            per_doc_alpha,
            idx,
            counts,
            gammas,
            self.n_topics,
            score_array,
            self.gamma_threshold,
            self.max_gamma_iter,
        )
        return engine.asarray(responsibilities), engine.asarray(document_gammas), per_topic_log_densities

    def backend_seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0], engine: Any) -> Any:
        """Backend-neutral LDA variational lower-bound scoring."""
        from mixle.stats.compute.backend import backend_seq_log_density

        num_documents, idx, counts, _, _ = _validate_lda_encoded(x, self.n_topics)
        responsibilities, document_gammas, topic_scores = self._backend_seq_posterior(x, engine)
        elbo = _lda_elbo_from_gamma(
            self.alpha,
            idx,
            counts,
            self.n_topics,
            np.asarray(engine.to_numpy(responsibilities), dtype=np.float64),
            np.asarray(engine.to_numpy(document_gammas), dtype=np.float64),
            np.asarray(engine.to_numpy(topic_scores), dtype=np.float64),
        )
        result = engine.asarray(elbo)
        if self.len_dist is not None and not supports(self.len_dist, Neutral):
            doc_lens = np.bincount(idx, weights=counts, minlength=num_documents)
            len_enc = self.len_dist.dist_to_encoder().seq_encode(doc_lens)
            result = result + backend_seq_log_density(self.len_dist, len_enc, engine)
        return result

    def seq_component_log_density(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray | None, E0]) -> np.ndarray:
        """Vectorized evaluation of the per-topic log-density of each document in encoded corpus x.

        Args:
            x: Encoded corpus of LDA documents (see LDADataEncoder.seq_encode()).

        Returns:
            2-d numpy array with shape (number of documents, n_topics), where entry (i, l) is the
            log-density of document i evaluated entirely under topic l.

        """
        num_topics = self.n_topics
        num_documents, idx, counts, _, enc_data = _validate_lda_encoded(x, self.n_topics)

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

        num_documents, idx, counts, _, enc_data = _validate_lda_encoded(x, self.n_topics)
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
        _, _, _, _, _ = _validate_lda_encoded(x, self.n_topics)
        _, document_gammas, _ = seq_posterior(self, x)

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
                gamma_threshold=self.gamma_threshold,
                max_gamma_iter=self.max_gamma_iter,
            )
        else:
            return LDAEstimator(
                estimators=[d.estimator() for d in self.topics],
                len_estimator=len_est,
                pseudo_count=(pseudo_count, pseudo_count),
                gamma_threshold=self.gamma_threshold,
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
        num_documents, idx, counts, old_gammas, enc_data = _validate_lda_encoded(x, self.num_topics)
        weights = _validated_document_weights(weights, num_documents)

        if not self._init_rng:
            self._rng_initialize(rng)

        if self.prev_alpha is None:
            self.prev_alpha = np.ones(self.num_topics)

        theta = self._rng_theta.dirichlet(self.prev_alpha, size=num_documents)
        theta_rep = theta[idx, :]

        idx_list = row_choice(p_mat=np.reshape(theta_rep, (-1, self.num_topics)), rng=self._rng_idx)

        self.sum_of_logs += np.dot(weights, np.log(theta))
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
        encoded = self.acc_to_encoder().seq_encode([x])
        self.seq_initialize(encoded, _validated_document_weights([weight], 1), rng)

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
        num_documents, idx, counts, old_gammas, enc_data = _validate_lda_encoded(x, self.num_topics)
        weights = _validated_document_weights(weights, num_documents)
        log_density_gamma, final_gammas, per_topic_log_densities, diagnostics = seq_posterior_with_diagnostics(
            estimate, x
        )
        impossible = np.asarray(diagnostics.impossible_documents, dtype=np.intp)
        if impossible.size and np.any(weights[impossible] > 0.0):
            raise ImpossibleEvidenceError(
                "LDA E-step encountered zero-probability evidence at document rows %s"
                % impossible[weights[impossible] > 0.0].tolist()
            )
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

            positive_weight = weights > 0.0
            self._seq_ll += float(np.dot(weights[positive_weight], elob[positive_weight]))

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
        num_documents, idx, counts, old_gammas, enc_data = _validate_lda_encoded(x, self.num_topics)
        weights_np = _validated_document_weights(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, num_documents
        )
        idx_np = np.asarray(idx, dtype=np.int64)

        log_density_gamma, final_gammas, per_topic_log_densities = estimate._backend_seq_posterior(x, engine)
        score_array = np.asarray(engine.to_numpy(per_topic_log_densities), dtype=np.float64)
        impossible = np.unique(idx_np[~np.any(np.isfinite(score_array), axis=1)])
        if impossible.size and np.any(weights_np[impossible] > 0.0):
            raise ImpossibleEvidenceError(
                "LDA engine E-step encountered zero-probability evidence at document rows %s"
                % impossible[weights_np[impossible] > 0.0].tolist()
            )

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
        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats, len_suff_stat = validated_statistic_tuple(
            suff_stat, 6, "LDA sufficient statistics"
        )

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

                # Copy this accumulator's own prev_alpha before it can be adopted into
                # stats_dict below -- otherwise a later key_replace would leave this
                # accumulator's private prev_alpha aliased into another tied accumulator.
                prev_alpha = self.prev_alpha.copy() if self.prev_alpha is not None else p_pa
                stats_dict[self.alpha_key] = (self.sum_of_logs + p_sol, self.doc_counts + p_doc, prev_alpha)

            else:
                # Copy on adoption: stats_dict must never alias this accumulator's own live
                # arrays. doc_counts is a plain float (immutable, safe to store as-is).
                stats_dict[self.alpha_key] = (
                    self.sum_of_logs.copy(),
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
                # Copy on replace too: without it, every tied accumulator ends up pointing at
                # the SAME array objects, so any one of them later accumulating new local data
                # would silently corrupt every other tied accumulator's counts.
                p_sol, p_doc, p_pa = stats_dict[self.alpha_key]
                self.prev_alpha = np.asarray(p_pa).copy() if p_pa is not None else None
                self.sum_of_logs = np.asarray(p_sol).copy()
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
        pseudo_count: float | tuple[float, float] | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
        fixed_alpha: np.ndarray | None = None,
        gamma_threshold: float = 1.0e-8,
        alpha_threshold: float = 1.0e-8,
        max_gamma_iter: int = 100,
        max_alpha_iter: int = 1000,
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
        if isinstance(estimators, (str, bytes)) or len(estimators) == 0:
            raise ValueError("LDAEstimator requires at least one topic estimator")
        self.num_topics = len(estimators)
        self.estimators = tuple(estimators)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        pseudo_count = broadcast_pseudo_count(pseudo_count, 2)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None)
        if not isinstance(self.keys, tuple) or len(self.keys) != 2:
            raise ValueError("LDAEstimator keys must be a two-item tuple")
        self.gamma_threshold = _positive_finite_threshold(gamma_threshold, "gamma_threshold")
        self.alpha_threshold = _positive_finite_threshold(alpha_threshold, "alpha_threshold")
        self.max_gamma_iter = _positive_iteration_budget(max_gamma_iter, "max_gamma_iter")
        self.max_alpha_iter = _positive_iteration_budget(max_alpha_iter, "max_alpha_iter")
        if fixed_alpha is None:
            self.fixed_alpha = None
        else:
            fixed = np.asarray(fixed_alpha, dtype=np.float64)
            if fixed.shape != (self.num_topics,) or np.any(~np.isfinite(fixed)) or np.any(fixed <= 0.0):
                raise ValueError("fixed_alpha must contain one positive finite entry per topic")
            self.fixed_alpha = fixed.copy()

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
        if isinstance(doc_counts, (bool, np.bool_)):
            raise TypeError("LDA weighted document count must be a real scalar")
        doc_counts = float(doc_counts)
        if not np.isfinite(doc_counts) or doc_counts < 0.0:
            raise ValueError("LDA weighted document count must be finite and non-negative")
        validate_effective_sample_mass(nobs, doc_counts, label="LDA effective sample")
        sum_of_logs = np.asarray(sum_of_logs, dtype=np.float64)
        topic_counts = np.asarray(topic_counts, dtype=np.float64)
        if sum_of_logs.shape != (num_topics,) or np.any(~np.isfinite(sum_of_logs)):
            raise ValueError("LDA expected-log statistics must be a finite vector with one entry per topic")
        if topic_counts.shape != (num_topics,) or np.any(~np.isfinite(topic_counts)) or np.any(topic_counts < 0.0):
            raise ValueError("LDA topic counts must be a finite non-negative vector with one entry per topic")
        if len(topic_suff_stats) != num_topics:
            raise ValueError("LDA topic sufficient statistics must contain one entry per topic")
        if prev_alpha is None:
            prev_alpha = self.fixed_alpha if self.fixed_alpha is not None else np.ones(num_topics)
        prev_alpha = np.asarray(prev_alpha, dtype=np.float64)
        if prev_alpha.shape != (num_topics,) or np.any(~np.isfinite(prev_alpha)) or np.any(prev_alpha <= 0.0):
            raise ValueError("previous LDA alpha must contain one positive finite entry per topic")
        prev_alpha = prev_alpha.copy()

        topics = [self.estimators[i].estimate(topic_counts[i], topic_suff_stats[i]) for i in range(num_topics)]
        len_dist = self.len_estimator.estimate(nobs, len_suff_stat)

        if doc_counts == 0:
            diagnostics = LDAOptimizationDiagnostics(
                algorithm="lda_alpha_fixed_point",
                converged=True,
                iterations=0,
                max_iterations=self.max_alpha_iter,
                termination_reason="no_weighted_documents",
                final_residual=0.0,
            )
            return LDADistribution(
                topics,
                prev_alpha,
                len_dist=len_dist,
                gamma_threshold=self.gamma_threshold,
                max_gamma_iter=self.max_gamma_iter,
                fit_diagnostics=diagnostics,
            )

        if self.fixed_alpha is None:
            if self.pseudo_count is not None:
                mean_of_logs = (sum_of_logs + np.log(self.pseudo_count[1])) / (doc_counts + self.pseudo_count[0])
            else:
                mean_of_logs = sum_of_logs / doc_counts

            # new_alpha, _ = find_alpha(prev_alpha, sum_of_logs/doc_counts, gamma_threshold*np.sqrt(float(doc_counts)))
            new_alpha, _, alpha_diagnostics = update_alpha(
                prev_alpha,
                mean_of_logs,
                self.alpha_threshold,
                max_iter=self.max_alpha_iter,
                return_diagnostics=True,
            )
        else:
            new_alpha = np.asarray(self.fixed_alpha).copy()
            alpha_diagnostics = LDAOptimizationDiagnostics(
                algorithm="fixed_alpha",
                converged=True,
                iterations=0,
                max_iterations=0,
                termination_reason="fixed_parameter",
                final_residual=0.0,
            )

        return LDADistribution(
            topics,
            new_alpha,
            len_dist=len_dist,
            gamma_threshold=self.gamma_threshold,
            max_gamma_iter=self.max_gamma_iter,
            fit_diagnostics=alpha_diagnostics,
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
        lengths: list[int] = []
        tx: list[Any] = []
        ctx: list[float] = []
        for doc_index, document in enumerate(x):
            if isinstance(document, (str, bytes)):
                raise TypeError(f"LDA document {doc_index} must be a sequence of (value, count) pairs")
            try:
                pairs = list(document)
            except TypeError as exc:
                raise TypeError(f"LDA document {doc_index} must be a sequence of (value, count) pairs") from exc
            lengths.append(len(pairs))
            for pair_index, pair in enumerate(pairs):
                if isinstance(pair, (str, bytes)):
                    raise TypeError(f"LDA document {doc_index} pair {pair_index} must have exactly two entries")
                try:
                    value, raw_count = pair
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"LDA document {doc_index} pair {pair_index} must have exactly two entries"
                    ) from exc
                if isinstance(raw_count, (bool, np.bool_)):
                    raise TypeError(f"LDA document {doc_index} pair {pair_index} count must be real-valued")
                try:
                    count = float(raw_count)
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"LDA document {doc_index} pair {pair_index} count must be real-valued") from exc
                if not np.isfinite(count) or count <= 0.0:
                    raise ValueError(f"LDA document {doc_index} pair {pair_index} count must be positive and finite")
                tx.append(value)
                ctx.append(count)

        nx = np.asarray(lengths, dtype=np.intp)
        idx = np.repeat(np.arange(num_documents), nx)
        counts = np.asarray(ctx, dtype=np.float64)
        gammas = None
        enc_data = self.encoder.seq_encode(tx)

        return num_documents, idx, counts, gammas, enc_data

    def row_count(self, x: Any) -> int:
        """Return the validated number of document rows in an encoded corpus."""
        if not isinstance(x, tuple) or len(x) != 5:
            raise ValueError("encoded LDA data must be a five-item tuple")
        num_documents = x[0]
        if isinstance(num_documents, (bool, np.bool_, float, np.floating)):
            raise TypeError("encoded LDA document count must be an integer")
        try:
            result = int(num_documents)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("encoded LDA document count must be an integer") from exc
        if result < 0:
            raise ValueError("encoded LDA document count must be non-negative")
        return result


def update_alpha(
    alpha_curr,
    mean_log_p,
    alpha_threshold,
    *,
    max_iter: int = 1000,
    return_diagnostics: bool = False,
):
    """Fixed-point update of the Dirichlet parameter alpha given mean expected log proportions.

    Args:
        alpha_curr (np.ndarray): Current alpha estimate.
        mean_log_p (np.ndarray): Mean expected log topic proportions across documents.
        alpha_threshold (float): Convergence threshold for the fixed-point iteration.

    Returns:
        Tuple of (updated alpha, number of iterations performed).

    """
    alpha = np.asarray(alpha_curr, dtype=np.float64)
    mean_log_p = np.asarray(mean_log_p, dtype=np.float64)
    if alpha.ndim != 1 or alpha.size == 0:
        raise ValueError("alpha_curr must be a non-empty one-dimensional vector")
    if mean_log_p.shape != alpha.shape:
        raise ValueError("mean_log_p must have the same shape as alpha_curr")
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("alpha_curr must contain only positive finite values")
    if np.any(~np.isfinite(mean_log_p)):
        raise ValueError("mean_log_p must contain only finite values")
    threshold = _positive_finite_threshold(alpha_threshold, "alpha_threshold")
    budget = _positive_iteration_budget(max_iter, "max_iter")
    alpha = alpha.copy()
    res = np.inf
    its_cnt = 0
    objective_trace = [_dirichlet_alpha_objective(alpha, mean_log_p)]

    # For ANY finite positive alpha, Jensen's inequality gives E[log theta_k] < log(alpha_k /
    # alpha.sum()) for every topic k (strict, since theta_k is a non-degenerate Beta marginal),
    # and exponentiating and summing over k telescopes the right-hand side to exactly
    # sum_k(alpha_k / alpha.sum()) == 1. So sum_k exp(mean_log_p_k) < 1 -- equivalently
    # logsumexp(mean_log_p) < 0 -- is a *necessary* condition for mean_log_p to be the expected-log
    # statistic of any finite-alpha Dirichlet at all. When the corpus's mean_log_p violates it (the
    # posterior is concentrated enough that the target sits on or past that boundary), this fixed
    # point is chasing a target no finite alpha can reach: alpha.sum() is mathematically guaranteed
    # to grow without bound rather than plateau, however many iterations it is given. This is
    # exactly the Dirichlet-MLE analogue of GammaEstimator's CV -> 0 shape-ceiling case (see
    # estimate_shape() in stats/univariate/continuous/gamma.py): a genuinely unreachable moment
    # target, not a slow-converging one.
    #
    # The idealized boundary is logsumexp(mean_log_p) == 0 exactly (e.g. the symmetric
    # mean_log_p = [-ln(K)] * K produced by a corpus with no distinguishing evidence at all).
    # mean_log_p reaching this function has already passed through a full E-step (digamma calls,
    # weighted averaging, possibly a pseudo-count blend), so at that exact boundary it carries a
    # few ULPs of floating-point roundoff that land on either side of 0 depending on incidental
    # details like the topic count -- a bare ">= 0.0" is not robust to that noise (see T4-01: on
    # an all-empty-document corpus, k in {3, 5, 6} landed a few 1e-16-to-1e-17 below 0 and were
    # misclassified as "iteration_budget_exhausted" while k in {2, 4, 8} landed at 0 or just above
    # and were correctly classified as "alpha_diverging", even though all six are the same
    # degenerate non-existence case). _ALPHA_BOUNDARY_TOL absorbs that roundoff -- it is many
    # orders of magnitude larger than the observed noise (~1e-16) but many orders of magnitude
    # smaller than the smallest margin an ordinary, genuinely-convergent corpus has from the
    # boundary in practice (e.g. mean_log_p = [-1, -1] sits ~0.31 away in logsumexp terms), so it
    # cannot swallow real convergent fits.
    alpha_target_unreachable = bool(logsumexp(mean_log_p) >= -_ALPHA_BOUNDARY_TOL)

    while res > threshold and its_cnt < budget:
        alpha_old = alpha
        candidate = np.asarray(digammainv(mean_log_p + digamma(alpha.sum())), dtype=np.float64)
        if np.any(~np.isfinite(candidate)) or np.any(candidate <= 0.0):
            diagnostics = LDAOptimizationDiagnostics(
                algorithm="lda_alpha_fixed_point",
                converged=False,
                iterations=its_cnt + 1,
                max_iterations=budget,
                termination_reason="invalid_iterate",
                final_residual=float("inf"),
                objective_trace=tuple(objective_trace),
            )
            raise LDAConvergenceError(diagnostics)
        candidate_objective = _dirichlet_alpha_objective(candidate, mean_log_p)
        previous_objective = objective_trace[-1]
        if candidate_objective < previous_objective - 1.0e-12 * max(1.0, abs(previous_objective)):
            diagnostics = LDAOptimizationDiagnostics(
                algorithm="lda_alpha_fixed_point",
                converged=False,
                iterations=its_cnt + 1,
                max_iterations=budget,
                termination_reason="objective_decreased",
                final_residual=float(np.abs(candidate - alpha_old).sum() / candidate.sum()),
                objective_trace=tuple(objective_trace + [candidate_objective]),
            )
            raise LDAConvergenceError(diagnostics)
        alpha = candidate
        objective_trace.append(candidate_objective)
        res = float(np.abs(alpha - alpha_old).sum() / alpha.sum())
        its_cnt += 1

    converged = res <= threshold
    if converged:
        termination_reason = "converged"
    elif alpha_target_unreachable:
        termination_reason = "alpha_diverging"
    else:
        termination_reason = "iteration_budget_exhausted"

    diagnostics = LDAOptimizationDiagnostics(
        algorithm="lda_alpha_fixed_point",
        converged=converged,
        iterations=its_cnt,
        max_iterations=budget,
        termination_reason=termination_reason,
        final_residual=res,
        objective_trace=tuple(objective_trace),
    )
    if not diagnostics.converged:
        raise LDAConvergenceError(diagnostics)
    if return_diagnostics:
        return alpha, its_cnt, diagnostics
    return alpha, its_cnt


def _dirichlet_alpha_objective(alpha: np.ndarray, mean_log_p: np.ndarray) -> float:
    """Per-document Dirichlet alpha objective, excluding alpha-independent terms."""
    value = gammaln(alpha.sum()) - gammaln(alpha).sum() + np.dot(alpha - 1.0, mean_log_p)
    return float(value)


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


def mpe(x0, f, eps: float, *, max_iter: int = 1000) -> tuple[np.ndarray, int]:
    """Minimal polynomial extrapolation of the fixed point of f starting from x0.

    Args:
        x0: Initial point of the fixed-point iteration.
        f: Fixed-point map.
        eps (float): Convergence threshold on successive extrapolants.

    Returns:
        Tuple of (extrapolated fixed point, number of iterations performed).

    """
    threshold = _positive_finite_threshold(eps, "eps")
    budget = _positive_iteration_budget(max_iter, "max_iter")
    if budget < 3:
        raise ValueError("max_iter must permit the three initial MPE iterates")
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim != 1 or x0.size == 0 or np.any(~np.isfinite(x0)):
        raise ValueError("x0 must be a non-empty finite one-dimensional vector")
    x1 = np.asarray(f(x0), dtype=np.float64)
    x2 = f(x1)
    x3 = f(x2)
    if x1.shape != x0.shape or np.shape(x2) != x0.shape or np.shape(x3) != x0.shape:
        raise ValueError("fixed-point map must preserve x0 geometry")
    if np.any(~np.isfinite(x1)) or np.any(~np.isfinite(x2)) or np.any(~np.isfinite(x3)):
        raise ValueError("fixed-point map produced a non-finite iterate")
    x_mat = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 3

    while res > threshold and its_cnt < budget:
        y = f(x_mat[-1, :])
        if np.shape(y) != x0.shape or np.any(~np.isfinite(y)):
            raise ValueError("fixed-point map produced an invalid iterate")
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

    if res > threshold:
        raise LDAConvergenceError(
            LDAOptimizationDiagnostics(
                algorithm="lda_minimal_polynomial_extrapolation",
                converged=False,
                iterations=its_cnt,
                max_iterations=budget,
                termination_reason="iteration_budget_exhausted",
                final_residual=float(res),
            )
        )
    return s, its_cnt


def alpha_seq_lambda(mean_log_p: float) -> Callable[[np.ndarray], float]:
    """Returns the alpha fixed-point map for a given mean expected log topic proportion."""

    def next_alpha(alpha_current: np.ndarray):
        return digammainv(mean_log_p + digamma(alpha_current.sum()))

    return next_alpha


def find_alpha(current_alpha: np.ndarray, mlp: float, thresh: float, *, max_iter: int = 1000):
    """Find the alpha fixed point via MPE acceleration (see update_alpha for the plain iteration)."""
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh, max_iter=max_iter)


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
        document_gammas = alpha_loc + np.reshape(
            np.bincount(idx_full.flat, minlength=num_documents * num_topics), (num_documents, num_topics)
        ) / float(num_topics)
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
    *,
    return_diagnostics=False,
):
    """Run the validated per-document mean-field coordinate-ascent update."""
    alpha = np.asarray(per_doc_alpha, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.intp)
    counts = np.asarray(counts, dtype=np.float64)
    scores = np.asarray(per_topic_log_densities, dtype=np.float64)
    num_documents = alpha.shape[0]
    if alpha.shape != (num_documents, num_topics):
        raise ValueError("per-document LDA alpha has invalid geometry")
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("per-document LDA alpha must be positive and finite")
    if scores.shape != (idx.size, num_topics):
        raise ValueError("per-topic LDA log densities have invalid geometry")
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("LDA topics produced invalid log densities")

    possible_rows = np.any(np.isfinite(scores), axis=1)
    impossible_documents = np.unique(idx[~possible_rows]).astype(int)
    safe_scores = scores.copy()
    safe_scores[~possible_rows, :] = 0.0

    if gammas is None:
        document_lengths = np.bincount(idx, weights=counts, minlength=num_documents)
        document_gammas = alpha + document_lengths[:, None] / float(num_topics)
    else:
        document_gammas = np.asarray(gammas, dtype=np.float64).copy()

    responsibilities = np.zeros((idx.size, num_topics), dtype=np.float64)
    residual = 0.0
    residual_trace: list[float] = []
    converged = num_documents == 0
    iterations = 0
    for iterations in range(1, max_gamma_iter + 1):
        log_weights = safe_scores + digamma(document_gammas)[idx, :]
        if np.any(possible_rows):
            normalizers = logsumexp(log_weights[possible_rows, :], axis=1, keepdims=True)
            responsibilities[possible_rows, :] = (
                np.exp(log_weights[possible_rows, :] - normalizers) * counts[possible_rows, None]
            )
        gamma_updates = alpha.copy()
        for topic_index in range(num_topics):
            gamma_updates[:, topic_index] += np.bincount(
                idx,
                weights=responsibilities[:, topic_index],
                minlength=num_documents,
            )
        if num_documents:
            relative = np.sum(np.abs(document_gammas - gamma_updates), axis=1) / np.sum(gamma_updates, axis=1)
            residual = float(np.max(relative))
        else:
            residual = 0.0
        residual_trace.append(residual)
        document_gammas = gamma_updates
        if residual <= gamma_threshold:
            converged = True
            break

    diagnostics = LDAOptimizationDiagnostics(
        algorithm="lda_document_coordinate_ascent",
        converged=converged,
        iterations=iterations,
        max_iterations=max_gamma_iter,
        termination_reason="converged" if converged else "iteration_budget_exhausted",
        final_residual=residual,
        residual_trace=tuple(residual_trace),
        impossible_documents=tuple(impossible_documents.tolist()),
    )
    # Exhausting max_gamma_iter is normal termination, not a failure. That cap exists precisely so
    # that "a few straggler documents would otherwise chase gamma_threshold for thousands of
    # iterations at negligible gain" (see LatentDirichletAllocation's parameter docs) -- raising
    # here made the documented worst-case bound fatal, and aborted fits whose residual had already
    # fallen to ~1e-8 against a 1e-8 threshold. The diagnostics above already report converged=False
    # and the final residual honestly, which is what callers should consult.
    #
    # A residual that is non-finite, or that never improved on its starting value, is a different
    # thing: the fixed point is diverging or stuck rather than sitting in its geometric tail, and no
    # amount of extra budget helps. That still raises.
    if not converged and residual_trace:
        stalled = not np.isfinite(residual) or residual >= residual_trace[0]
        if stalled:
            raise LDAConvergenceError(diagnostics)
    if return_diagnostics:
        return responsibilities, document_gammas, diagnostics
    return responsibilities, document_gammas


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

    ldg = np.asarray(log_density_gamma, dtype=np.float64)
    dg = np.asarray(document_gammas, dtype=np.float64)
    topic_scores = np.asarray(per_topic_log_densities, dtype=np.float64)
    if np.any(np.isnan(topic_scores)) or np.any(np.isposinf(topic_scores)):
        raise ValueError("LDA topics produced invalid log densities")
    if np.any(~np.isfinite(dg)) or np.any(dg <= 0.0):
        raise ValueError("LDA posterior produced invalid document gammas")
    impossible_rows = ~np.any(np.isfinite(topic_scores), axis=1)
    impossible_documents = np.unique(idx[impossible_rows])
    safe_topic_scores = topic_scores.copy()
    safe_topic_scores[impossible_rows, :] = 0.0

    elob0 = digamma(dg) - digamma(np.sum(dg, axis=1, keepdims=True))
    elob1 = elob0[idx, :]
    elob2 = np.zeros_like(ldg)
    positive = ldg > 0.0
    elob2[positive] = ldg[positive] * (
        (elob1 + safe_topic_scores)[positive]
        - np.log(ldg[positive])
        + np.broadcast_to(np.log(np.reshape(counts, (-1, 1))), ldg.shape)[positive]
    )
    elob3 = np.sum(elob0 * ((per_doc_alpha - 1.0) - (dg - 1.0)), axis=1)
    elob4 = np.bincount(idx_full.flat, weights=elob2.flat, minlength=document_gammas.size)
    elob5 = np.sum(np.reshape(elob4, (-1, num_topics)), axis=1)
    elob6 = np.sum(gammaln(dg), axis=1) - gammaln(dg.sum(axis=1))
    if per_doc_alpha.ndim == 1:
        elob7 = gammaln(per_doc_alpha.sum()) - gammaln(per_doc_alpha).sum()
    else:
        elob7 = gammaln(per_doc_alpha.sum(axis=1)) - gammaln(per_doc_alpha).sum(axis=1)

    result = np.asarray(elob3 + elob5 + elob6 + elob7, dtype=np.float64)
    result[impossible_documents] = -np.inf
    return result


def seq_posterior_with_diagnostics(estimate: LDADistribution, x: tuple[int, np.ndarray, np.ndarray, Any | None, E0]):
    """Return the LDA variational posterior and its convergence record."""
    num_documents, idx, counts, gammas, enc_data = _validate_lda_encoded(x, estimate.n_topics)
    per_topic_log_densities = np.asarray(
        [topic.seq_log_density(enc_data) for topic in estimate.topics], dtype=np.float64
    ).transpose()
    if per_topic_log_densities.shape != (idx.size, estimate.n_topics):
        raise ValueError("LDA topic scorers returned arrays with invalid geometry")
    per_doc_alpha = np.repeat(estimate.alpha[None, :], num_documents, axis=0)
    responsibilities, final_gammas, diagnostics = _lda_vi_fixed_point(
        per_doc_alpha,
        idx,
        counts,
        gammas,
        estimate.n_topics,
        per_topic_log_densities,
        estimate.gamma_threshold,
        estimate.max_gamma_iter,
        return_diagnostics=True,
    )
    return responsibilities, final_gammas, per_topic_log_densities, diagnostics


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
    responsibilities, final_gammas, per_topic_log_densities, _ = seq_posterior_with_diagnostics(estimate, x)
    return responsibilities, final_gammas, per_topic_log_densities


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
