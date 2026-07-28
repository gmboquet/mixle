"""Integer probabilistic latent semantic indexing models.

An observation is a document id paired with a sparse integer bag of word/value
counts:

    ``(doc_id, [(value_id, count), ...])``

For ``S`` latent topics, ``V`` word values, and ``D`` document ids, the model
uses:

* ``state_word_mat[v, s] = p(value=v | topic=s)``;
* ``doc_state_mat[d, s] = p(topic=s | document=d)``;
* ``doc_vec[d] = p(document=d)``; and
* an optional length model for total bag count.

The log-density combines the document prior, the optional length density, and
the topic-marginalized word probabilities for each sparse count entry. Caller
data should use stable integer ids for documents and word values.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, LengthFrontierMerge, merge_enumerators
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)
from mixle.stats.latent.effective_sample import (
    validate_effective_sample_mass,
    validated_count_array,
    validated_observation_weight,
    validated_statistic_tuple,
)
from mixle.stats.multivariate._multinomial_contracts import (
    canonical_integer_bag,
    exact_integer,
    finite_weight,
    log_coefficient,
    observation_weights,
)
from mixle.stats.multivariate.categorical_multinomial import MultisetProductEnumerator
from mixle.utils.aliasing import broadcast_pseudo_count
from mixle.utils.optional_deps import numba
from mixle.utils.optsutil import count_by_value

T1 = TypeVar("T1")  ## type for encoded sequence of lengths.
SS1 = TypeVar("SS1")  ### type for value of length dist sufficient statistics.


def _positive_dimension(value: Any, label: str) -> int:
    result = exact_integer(value, label=label)
    if result <= 0:
        raise ValueError("%s must be positive" % label)
    return result


def _validated_integer_plsi_statistics(values, num_vals, num_states, num_docs):
    """Validate integer PLSI counts, geometry, and token-mass conservation."""
    word_count, comp_count, doc_count, length_stats = validated_statistic_tuple(
        values, 4, "integer PLSI sufficient statistics"
    )
    word_count = validated_count_array(
        word_count,
        (num_states, num_vals),
        "integer PLSI word counts",
    )
    comp_count = validated_count_array(
        comp_count,
        (num_docs, num_states),
        "integer PLSI state counts",
    )
    doc_count = validated_count_array(doc_count, (num_docs,), "integer PLSI document counts")
    validate_effective_sample_mass(
        word_count.sum(),
        comp_count.sum(),
        label="integer PLSI token mass",
    )
    return word_count, comp_count, doc_count, length_stats


def _simplex_vector(value: Any, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite nonempty simplex vector" % label) from exc
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s must be a finite nonempty simplex vector" % label)
    total = float(result.sum())
    if not np.isclose(total, 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("%s must sum to one" % label)
    # Not dividing by total, for the reason given in _simplex_matrix: the check above already holds it
    # to 1.0 within 1e-12, so the division only costs an ulp per entry on every rebuild.
    return result.copy()


def _simplex_matrix(value: Any, label: str, axis: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite nonempty simplex matrix" % label) from exc
    if result.ndim != 2 or 0 in result.shape or np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s must be a finite nonempty simplex matrix" % label)
    totals = result.sum(axis=axis, keepdims=True)
    if not np.allclose(totals, 1.0, rtol=1.0e-12, atol=1.0e-12):
        direction = "columns" if axis == 0 else "rows"
        raise ValueError("%s %s must each sum to one" % (label, direction))
    # Deliberately NOT dividing by totals. The check above already established every total is 1.0 to
    # within 1e-12, so the division cannot improve normalization beyond the tolerance this function
    # declares -- but it does shift each entry by an ulp, and it does so on every construction. That
    # made the distribution's __str__ round trip non-convergent: str -> eval -> str never reached a
    # fixed point, because each rebuild renormalized by a total that was 1.0 only to within rounding.
    return result.copy()


def _canonical_observation(value: Any, num_docs: int, num_vals: int) -> tuple[int, list[tuple[int, int]], int]:
    if isinstance(value, (str, bytes)):
        raise TypeError("integer PLSI observation must be a (document_id, bag) pair")
    try:
        parts = list(value)
    except TypeError as exc:
        raise TypeError("integer PLSI observation must be a (document_id, bag) pair") from exc
    if len(parts) != 2:
        raise ValueError("integer PLSI observation must be a (document_id, bag) pair")
    doc_id = exact_integer(parts[0], label="integer PLSI document id", nonnegative=True)
    if doc_id >= num_docs:
        raise ValueError("integer PLSI document id is outside the configured support")
    bag, total, _ = canonical_integer_bag(
        parts[1],
        min_val=0,
        max_val=num_vals - 1,
        reject_outside=True,
    )
    return doc_id, bag, total


def _encoded_coefficient(xc: np.ndarray, xi: np.ndarray, xn: np.ndarray) -> np.ndarray:
    result = gammaln(np.asarray(xn, dtype=np.float64) + 1.0)
    if len(xc):
        result -= np.bincount(
            np.asarray(xi, dtype=np.int64),
            weights=gammaln(np.asarray(xc, dtype=np.float64) + 1.0),
            minlength=len(xn),
        )
    return result


@dataclass(frozen=True)
class IntegerPLSIEncodedData:
    """Owned, validated flattened representation of canonical integer PLSI bags."""

    value_ids: np.ndarray
    counts: np.ndarray
    entry_doc_ids: np.ndarray
    row_ids: np.ndarray
    totals: np.ndarray
    doc_ids: np.ndarray
    num_vals: int
    num_docs: int

    def __post_init__(self) -> None:
        num_vals = _positive_dimension(self.num_vals, "integer PLSI encoded num_vals")
        num_docs = _positive_dimension(self.num_docs, "integer PLSI encoded num_docs")
        arrays = tuple(
            np.asarray(value)
            for value in (
                self.value_ids,
                self.counts,
                self.entry_doc_ids,
                self.row_ids,
                self.totals,
                self.doc_ids,
            )
        )
        xv, xc, xd, xi, xn, xm = arrays
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("integer PLSI encoded arrays must be one-dimensional")
        if not (len(xv) == len(xc) == len(xd) == len(xi)):
            raise ValueError("integer PLSI encoded entry arrays must have equal length")
        if len(xn) != len(xm):
            raise ValueError("integer PLSI encoded row arrays must have equal length")
        for array, label in ((xv, "value IDs"), (xd, "entry document IDs"), (xi, "row IDs"), (xm, "document IDs")):
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError("integer PLSI encoded %s must be exact integer arrays" % label)
        if np.any(~np.isfinite(np.asarray(xc, dtype=np.float64))) or np.any(xc <= 0):
            raise ValueError("integer PLSI encoded counts must be finite positive exact integers")
        if np.any(np.asarray(xc, dtype=np.float64) != np.floor(np.asarray(xc, dtype=np.float64))):
            raise ValueError("integer PLSI encoded counts must be finite positive exact integers")
        if np.any(~np.isfinite(np.asarray(xn, dtype=np.float64))) or np.any(xn < 0):
            raise ValueError("integer PLSI encoded totals must be finite non-negative exact integers")
        if np.any(np.asarray(xn, dtype=np.float64) != np.floor(np.asarray(xn, dtype=np.float64))):
            raise ValueError("integer PLSI encoded totals must be finite non-negative exact integers")
        rows = len(xn)
        if np.any(xv < 0) or np.any(xv >= num_vals):
            raise ValueError("integer PLSI encoded value ID is outside the configured support")
        if np.any(xm < 0) or np.any(xm >= num_docs) or np.any(xd < 0) or np.any(xd >= num_docs):
            raise ValueError("integer PLSI encoded document ID is outside the configured support")
        if np.any(xi < 0) or np.any(xi >= rows):
            raise ValueError("integer PLSI encoded row ID is outside the batch")
        if len(xi) > 1 and np.any(xi[1:] < xi[:-1]):
            raise ValueError("integer PLSI encoded row IDs must be non-decreasing")
        if len(xi) and not np.array_equal(xd, xm[xi]):
            raise ValueError("integer PLSI encoded entry document IDs do not match their rows")
        computed = np.bincount(
            np.asarray(xi, dtype=np.int64),
            weights=np.asarray(xc, dtype=np.float64),
            minlength=rows,
        )
        if not np.array_equal(computed, np.asarray(xn, dtype=np.float64)):
            raise ValueError("integer PLSI encoded totals do not equal their row counts")
        owned = (
            np.asarray(xv, dtype=np.int32).copy(),
            np.asarray(xc, dtype=np.float64).copy(),
            np.asarray(xd, dtype=np.int32).copy(),
            np.asarray(xi, dtype=np.int32).copy(),
            np.asarray(xn, dtype=np.float64).copy(),
            np.asarray(xm, dtype=np.int32).copy(),
        )
        for field, array in zip(
            ("value_ids", "counts", "entry_doc_ids", "row_ids", "totals", "doc_ids"),
            owned,
        ):
            object.__setattr__(self, field, array)
        object.__setattr__(self, "num_vals", num_vals)
        object.__setattr__(self, "num_docs", num_docs)

    def __iter__(self):
        return iter((self.value_ids, self.counts, self.entry_doc_ids, self.row_ids, self.totals, self.doc_ids))


def _validated_encoded(value: Any, num_vals: int, num_docs: int) -> tuple[Any, IntegerPLSIEncodedData]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("integer PLSI encoded data must be produced by its data encoder")
    length_data, rows = value
    if not isinstance(rows, IntegerPLSIEncodedData):
        raise ValueError("integer PLSI encoded data must be produced by its data encoder")
    if rows.num_vals != num_vals or rows.num_docs != num_docs:
        raise ValueError("integer PLSI encoded dimensions do not match the model")
    return length_data, rows


class IntegerProbabilisticLatentSemanticIndexingDistribution(SequenceEncodableProbabilityDistribution):
    """Integer-valued probabilistic latent semantic indexing distribution."""

    def __init__(
        self,
        state_word_mat: list[list[float]] | np.ndarray,
        doc_state_mat: list[list[float]] | np.ndarray,
        doc_vec: list[float] | np.ndarray,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
    ) -> None:
        """Create an integer PLSI distribution.

        Args:
            state_word_mat: Word/value probabilities by latent topic. Columns
                correspond to topics and should each sum to one.
            doc_state_mat: Topic probabilities by document id. Rows correspond
                to documents and should each sum to one.
            doc_vec: Document prior probabilities. Entries should sum to one.
            len_dist: Optional distribution for total bag length. ``None`` uses
                the neutral null distribution.
            name: Optional diagnostic name.

        Attributes:
            prob_mat: Word/value probabilities by topic.
            state_mat: Topic probabilities by document id.
            doc_vec: Document prior probabilities.
            log_doc_vec: Log of ``doc_vec``.
            num_vals: Number of word/value ids.
            num_states: Number of latent topics.
            num_docs: Number of document ids.
            name: Optional diagnostic name.
            len_dist: Distribution for total bag length, or a null distribution.

        """
        self.prob_mat = _simplex_matrix(state_word_mat, "state_word_mat", axis=0)
        self.state_mat = _simplex_matrix(doc_state_mat, "doc_state_mat", axis=1)
        self.doc_vec = _simplex_vector(doc_vec, "doc_vec")
        self.num_vals = self.prob_mat.shape[0]
        self.num_states = self.prob_mat.shape[1]
        self.num_docs = self.state_mat.shape[0]
        if self.state_mat.shape[1] != self.num_states:
            raise ValueError("state_word_mat and doc_state_mat must use the same number of latent states")
        if self.doc_vec.shape != (self.num_docs,):
            raise ValueError("doc_vec length must equal the number of doc_state_mat rows")
        with np.errstate(divide="ignore"):
            self.log_doc_vec = np.log(self.doc_vec)
        self.name = name
        self.len_dist = len_dist if len_dist is not None else NullDistribution()

    def compute_capabilities(self):
        """Return backend capability metadata for this concrete PLSI instance."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, capabilities_for

        child = capabilities_for(self.len_dist)
        return DistributionCapabilities(
            engine_ready=child.engine_ready,
            kernel_status="numpy_only" if child.numpy_only_reason else "generic_latent",
            numpy_only_reason=child.numpy_only_reason,
        )

    def compute_declaration(self):
        """Return the symbolic distribution declaration for code generation and PPL introspection."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = () if length is None else (length,)
        return DistributionDeclaration(
            name="integer_plsi",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("prob_mat", constraint="column_simplex_matrix"),
                ParameterSpec("state_mat", constraint="row_simplex_matrix"),
                ParameterSpec("doc_vec", constraint="simplex_vector"),
            ),
            statistics=(
                StatisticSpec("word_counts"),
                StatisticSpec("state_counts"),
                StatisticSpec("document_counts"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="integer_document_bag",
            children=children,
            child_roles=("length",) if length is not None else (),
            differentiable=False,
        )

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        s1 = ",".join(["[" + ",".join(map(str, self.prob_mat[i, :])) + "]" for i in range(len(self.prob_mat))])
        s2 = ",".join(["[" + ",".join(map(str, self.state_mat[i, :])) + "]" for i in range(len(self.state_mat))])
        s3 = ",".join(map(str, self.doc_vec))
        s4 = repr(self.name)
        s5 = str(self.len_dist)
        return "IntegerProbabilisticLatentSemanticIndexingDistribution([%s], [%s], [%s], name=%s, len_dist=%s)" % (
            s1,
            s2,
            s3,
            s4,
            s5,
        )

    def density(self, x: tuple[int, Sequence[tuple[int, float]]]) -> float:
        """Evaluate the density of PLSI model for an observation x.

        See log_density() for details on the density evaluation.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): Single observation of integer PLSI.

        Returns:
            Density evaluated at observed value x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[int, Sequence[tuple[int, float]]]) -> float:
        """Evaluate the log-density of PLSI model for an observation of x.

        Consider an Integer PLSI model for a corpus of documents with S states, V word values, and D documents ids
        (authors).

        Let x (Tuple[int, Sequence[Tuple[int, float]]]) be an observation from a PLSI model, consisting of
        x = (d, [(v_0, c_0), (v_1, c_1), ..., (v_{k-1}, c_{k-1})]), where the 'd' is some document d_id in the corpus and
        each tuple (v_i, c_i) corresponds to a value-count couple in the corpus. The log-likelihood is given by

        log(p_mat(x)) = log(p_mat(d)) + sum_{j=0}^{k-1} c_k*log( sum_{s=0}^{S-1} p_mat(d|s)p_mat(s|v_k) ) + log(P_len(nn)),

        where P_len(nn) is the density of the length distribution for 'nn' representing the total number of words in
        the document.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): (doc_id, [(value_id, count_for_value)]). See above for details.

        Returns:
            Log-density evaluated at a single observation x.

        """

        d_id, bag, total = _canonical_observation(x, self.num_docs, self.num_vals)
        xv = np.asarray([u[0] for u in bag], dtype=np.int64)
        xc = np.asarray([u[1] for u in bag], dtype=np.float64)

        rv = float(self.log_doc_vec[d_id]) + log_coefficient(xc)
        if len(xv):
            with np.errstate(divide="ignore"):
                rv += float(np.dot(np.log(np.dot(self.prob_mat[xv, :], self.state_mat[d_id, :])), xc))
        rv += float(self.len_dist.log_density(total))
        return float(rv)

    def component_log_density(self, x: tuple[int, Sequence[tuple[int, float]]]) -> np.ndarray:
        """Evaluate the log-density for each state in the PLSI.

        Returns count*log(p_mat(W|S)) for each word-count pair in the document. Returned value is S by 1 where S is the
        number of components in the model.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): Single PLSI observation of form
                (doc_id, [(value_id, count_for_value)]).

        Returns:
            Numpy array of length S (num_states).

        """
        _, bag, _ = _canonical_observation(x, self.num_docs, self.num_vals)
        if not bag:
            return np.zeros(self.num_states, dtype=np.float64)
        xv = np.asarray([u[0] for u in bag], dtype=np.int64)
        xc = np.asarray([u[1] for u in bag], dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.dot(np.log(self.prob_mat[xv, :]).T, xc)

    def seq_log_density(
        self, x: tuple[T1 | None, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Vectorized evaluation of the log-density for an encoded sequence of iid observation from a PLSI model.

        See log_density() function for details on the log-likelihood.

        The encoded sequence 'x' is a Tuple length 2. The first component contains data type Optional[T1]
        corresponding to the sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document id's for the lengths above (len = len(x)).

        Args:
            x: Encoded sequence of iid observations of PLSI model. See above for details.

        Returns:
            Numpy array of log-density evaluated at each observation in the encoded sequence.

        """
        nn, rows = _validated_encoded(x, self.num_vals, self.num_docs)
        xv, xc, xd, xi, xn, xm = rows
        cnt = len(xn)

        w = np.zeros(len(xv), dtype=np.float64)
        index_dot(self.prob_mat, xv, self.state_mat, xd, w)
        w = np.log(w, out=w)
        w *= xc

        rv = np.zeros(cnt, dtype=np.float64)
        bincount(xi, w, rv)
        rv += _encoded_coefficient(xc, xi, xn)
        rv += self.log_doc_vec[xm]

        rv += self.len_dist.seq_log_density(nn)

        return rv

    def backend_seq_log_density(
        self,
        x: tuple[T1 | None, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        engine: Any,
    ) -> Any:
        """Evaluate encoded PLSI log densities using a backend-neutral compute engine."""
        from mixle.stats.compute.backend import backend_seq_log_density

        nn, rows = _validated_encoded(x, self.num_vals, self.num_docs)
        xv, xc, xd, xi, xn, xm = rows
        cnt = len(xn)
        value_ids = engine.asarray(xv)
        count_values = engine.asarray(xc)
        doc_ids = engine.asarray(xd)
        obs_ids = engine.asarray(xi)
        obs_doc_ids = engine.asarray(xm)

        prob_mat = engine.asarray(self.prob_mat)
        state_mat = engine.asarray(self.state_mat)
        doc_log_probs = engine.asarray(self.log_doc_vec)

        row_probs = engine.sum(prob_mat[value_ids, :] * state_mat[doc_ids, :], axis=1)
        row_log_probs = engine.log(row_probs) * count_values
        rv = engine.bincount(obs_ids, weights=row_log_probs, minlength=cnt)
        rv = rv + engine.asarray(_encoded_coefficient(xc, xi, xn))
        rv = rv + doc_log_probs[obs_doc_ids]

        rv = rv + backend_seq_log_density(self.len_dist, nn, engine)

        return rv

    def seq_component_log_density(
        self, x: tuple[T1 | None, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Vectorized evaluation of the component log-density for each observation in an encoded sequence of iid PLSI
            observations.

        See component_log_density() function for details on component log-likelihood evaluation.

        The encoded sequence 'x' is a Tuple length 2. The first component contains data type Optional[T1]
        corresponding to the sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document id's for the lengths above (len = len(x)).

        Args:
            x: Encoded sequence of iid observations of PLSI model. See above for details.

        Returns:
            2-d numpy array containing N rows of num_state sized arrays.

        """
        _, rows = _validated_encoded(x, self.num_vals, self.num_docs)
        xv, xc, xd, xi, xn, xm = rows
        rv = np.zeros((len(xn), self.num_states), dtype=np.float64)
        w_mat = self.prob_mat
        fast_seq_component_log_density(xv, xc, xd, xi, xm, w_mat, rv)
        return rv

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate PLSI observations ``(doc_id, bag)`` in descending probability order.

        A PLSI observation factors as ``P(doc) * [prod_w q_d(w)^{c_w}] * P_len(n)`` where ``q_d`` is the
        per-document word distribution ``prob_mat @ state_mat[d]`` and ``n`` the total word count, so it
        is a document-labelled mixture of trial-count multinomials: for each document the bags enumerate
        by a multiset best-first search under a length frontier driven by ``len_dist`` (the real
        trial-count distribution), and the per-document streams are merged by descending score with the
        document log-probability as offset. Requires a modelled ``len_dist`` unless every per-document
        word distribution is sub-stochastic-free; an absent length distribution leaves the bag support
        infinite and is enumerated by the multinomial term alone.
        """
        return IntegerProbabilisticLatentSemanticIndexingEnumerator(self)

    def sampler(self, seed: int | None = None) -> "IntegerProbabilisticLatentSemanticIndexingSampler":
        """Return a sampler for iid integer PLSI observations."""
        return IntegerProbabilisticLatentSemanticIndexingSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerProbabilisticLatentSemanticIndexingEstimator":
        """Return an estimator initialized from this distribution's dimensions.

        Args:
            pseudo_count: Optional smoothing count for topic, word, and document counts.

        Returns:
            A configured integer PLSI estimator.

        """
        if pseudo_count is None:
            return IntegerProbabilisticLatentSemanticIndexingEstimator(
                num_vals=self.num_vals,
                num_states=self.num_states,
                num_docs=self.num_docs,
                len_estimator=self.len_dist.estimator(),
                name=self.name,
            )
        else:
            pseudo_count = (pseudo_count, pseudo_count, pseudo_count)
            return IntegerProbabilisticLatentSemanticIndexingEstimator(
                num_vals=self.num_vals,
                num_states=self.num_states,
                num_docs=self.num_docs,
                pseudo_count=pseudo_count,
                suff_stat=(self.prob_mat.T, self.state_mat, self.doc_vec),
                len_estimator=self.len_dist.estimator(),
                name=self.name,
            )

    def dist_to_encoder(self) -> "IntegerProbabilisticLatentSemanticIndexingDataEncoder":
        """Return an encoder for integer PLSI observations."""
        return IntegerProbabilisticLatentSemanticIndexingDataEncoder(
            self.num_vals,
            self.num_docs,
            len_encoder=self.len_dist.dist_to_encoder(),
        )


def multinomial_bag_stream(log_p_vec, min_val, len_dist, combine):
    """Enumerate normalized integer count-vector bags in descending probability order.

    Each fixed-size stream includes ``log(n!) - sum_w log(c_w!)``. A real length
    law is required because a conditional multinomial law for every possible
    length has infinite total mass when joined without length probabilities.
    """
    entries = [(int(min_val + k), float(lp)) for k, lp in enumerate(log_p_vec) if lp > -np.inf]
    entries.sort(key=lambda u: -u[1])
    if supports(len_dist, Neutral):
        raise EnumerationError(
            len_dist,
            reason="normalized count-vector enumeration requires a probability law for the bag length",
        )
    len_stream = BufferedStream(child_enumerator(len_dist, "multinomial_bag_stream.len_dist"))
    return LengthFrontierMerge(
        len_stream,
        lambda n, lp_len: _FiniteMultinomialEnumerator(entries, n, combine, lp_len),
    )


class _FiniteMultinomialEnumerator:
    """Exact descending enumerator for one finite categorical multinomial size."""

    def __init__(self, entries, n, combine, offset):
        self._items = []
        count = exact_integer(n, label="multinomial enumerator length", nonnegative=True)
        score_offset = float(offset)
        if not np.isfinite(score_offset):
            raise ValueError("multinomial enumerator length score must be finite")
        if count == 0:
            self._items.append((combine(()), score_offset))
        elif entries:
            counts = [0] * len(entries)

            def visit(position: int, remaining: int) -> None:
                if position == len(entries) - 1:
                    counts[position] = remaining
                    pairs = tuple((entries[index][0], value) for index, value in enumerate(counts) if value)
                    coefficient = log_coefficient(counts)
                    token_score = sum(value * entries[index][1] for index, value in enumerate(counts))
                    self._items.append((combine(pairs), score_offset + coefficient + token_score))
                    return
                for value in range(remaining + 1):
                    counts[position] = value
                    visit(position + 1, remaining - value)

            visit(0, count)
        self._items.sort(key=lambda item: -item[1])
        self._iterator = iter(self._items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)


def bag_stream(element_stream, len_dist, combine):
    """Enumerate bags (multisets) drawn from a sorted element stream, in descending bag-score order.

    ``element_stream`` is a descending ``(value, log_prob)`` iterator over the element distribution
    (any enumerable element distribution -- a fixed categorical, or e.g. a per-document/per-given
    mixture). A bag scores by the sum of its elements' log-probs plus ``log P_len(n)`` from
    ``len_dist``; bags enumerate by the per-size multiset best-first search
    (:class:`MultisetProductEnumerator`) under a length frontier. When ``len_dist`` is Null there is no
    length term and a synthetic ``n*log p_max`` frontier orders the (countably infinite) support by the
    element term alone. ``combine`` maps the tuple of ``(value, count)`` pairs to the emitted bag.
    """
    elem_buf = BufferedStream(iter(element_stream))
    head = elem_buf.get(0)
    if head is None:
        return iter([(combine(()), 0.0)])
    if supports(len_dist, Neutral):
        lp_max = float(head[1])
        if lp_max >= 0.0:
            raise EnumerationError(
                len_dist, reason="an element has probability one and no length distribution bounds the bag size"
            )
        len_stream = BufferedStream((n, n * lp_max) for n in itertools.count())
        return LengthFrontierMerge(
            len_stream, lambda n, lp_len: MultisetProductEnumerator(elem_buf, n, combine=combine, offset=0.0)
        )
    len_stream = BufferedStream(child_enumerator(len_dist, "bag_stream.len_dist"))
    return LengthFrontierMerge(
        len_stream, lambda n, lp_len: MultisetProductEnumerator(elem_buf, n, combine=combine, offset=lp_len)
    )


class IntegerProbabilisticLatentSemanticIndexingEnumerator(DistributionEnumerator):
    """Best-first enumerator for document-labelled integer PLSI bag observations."""

    def __init__(self, dist: IntegerProbabilisticLatentSemanticIndexingDistribution) -> None:
        """Best-first enumeration of ``(doc_id, bag)`` over the document-labelled multinomial mixture.

        Args:
            dist (IntegerProbabilisticLatentSemanticIndexingDistribution): Distribution whose support is enumerated.
        """
        super().__init__(dist)
        streams = []
        offsets = []
        with np.errstate(divide="ignore"):
            for d in range(dist.num_docs):
                if dist.doc_vec[d] <= 0.0:
                    continue
                q_d = dist.prob_mat @ dist.state_mat[d]  # P(word | doc d)
                log_q = np.log(q_d)

                def combine(pairs, d=d):
                    return (d, [(int(w), int(c)) for w, c in sorted(pairs)])

                streams.append(multinomial_bag_stream(log_q, 0, dist.len_dist, combine))
                offsets.append(float(dist.log_doc_vec[d]))
        self._merge = merge_enumerators(streams, offsets)

    def __next__(self) -> tuple[tuple[int, list[tuple[int, int]]], float]:
        return next(self._merge)


class IntegerProbabilisticLatentSemanticIndexingSampler(DistributionSampler):
    """Sampler for integer PLSI document ids, word-count bags, and document lengths."""

    def __init__(self, dist: IntegerProbabilisticLatentSemanticIndexingDistribution, seed: int | None = None) -> None:
        """Create a sampler for an integer PLSI distribution.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.

        Attributes:
            rng: Random state used for document and topic draws.
            dist: Distribution to sample from.
            size_rng: Sampler for document lengths.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.size_rng = self.dist.len_dist.sampler(self.rng.randint(0, maxrandint))

    def sample(
        self, size: int | None = None, *, batched: bool = True
    ) -> tuple[int, Sequence[tuple[int, float]]] | Sequence[tuple[int, Sequence[tuple[int, float]]]]:
        """Generate iid samples from PLSI model.

        Args:
            size (Optional[int]): Number of samples to generate. Defaults to 0 if size is None.

        Returns:
            Sequence of iid PLSI samples if size is not None, else a single sample from PLSI model.

        """
        if size is None:
            d_id = self.rng.choice(self.dist.num_docs, p=self.dist.doc_vec)
            cnt = self.size_rng.sample()
            z = self.rng.multinomial(cnt, pvals=self.dist.state_mat[d_id, :])
            rv = []
            for i, n in enumerate(z):
                if n > 0:
                    rv.extend(self.rng.choice(self.dist.num_vals, p=self.dist.prob_mat[:, i], replace=True, size=n))

            return d_id, list(count_by_value(rv).items())

        else:
            return [self.sample() for i in range(size)]


class IntegerProbabilisticLatentSemanticIndexingAccumulator(SequenceEncodableStatisticAccumulator):
    """EM sufficient-statistic accumulator for integer PLSI word, state, document, and length terms."""

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_acc: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        name: str | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
    ) -> None:
        """Create an accumulator for integer PLSI sufficient statistics.

        Merge keys are ordered as ``(word_key, state_key, document_key)``.

        Args:
            num_vals: Number of word/value ids.
            num_states: Number of latent topics.
            num_docs: Number of document ids.
            len_acc: Optional accumulator for total bag length.
            name: Optional diagnostic name.
            keys: Optional merge keys for word, state, and document counts.

        Attributes:
            num_vals: Number of word/value ids.
            num_states: Number of latent topics.
            num_docs: Number of document ids.
            word_count: Topic-by-word weighted counts.
            comp_count: Document-by-topic weighted counts.
            doc_count: Weighted document counts.
            name: Optional diagnostic name.
            wc_key: Merge key for ``word_count``.
            sc_key: Merge key for ``comp_count``.
            dc_key: Merge key for ``doc_count``.
            len_acc: Accumulator for total bag length.

        """
        self.num_vals = _positive_dimension(num_vals, "num_vals")
        self.num_states = _positive_dimension(num_states, "num_states")
        self.num_docs = _positive_dimension(num_docs, "num_docs")
        self.word_count = np.zeros((num_states, num_vals), dtype=np.float64)
        self.comp_count = np.zeros((num_docs, num_states), dtype=np.float64)
        self.doc_count = np.zeros(num_docs, dtype=np.float64)
        self.name = name
        self.wc_key, self.sc_key, self.dc_key = keys if keys is not None else (None, None, None)
        self.len_acc = len_acc if len_acc is not None else NullAccumulator()

        # Per-document data log-likelihood accumulated as a byproduct of the E-step, only when
        # _track_ll is enabled. Equals seq_log_density_sum(enc, dist)[1] and is consumed by the
        # fused-EM fast path in optimize(reuse_estep_ll=True); not part of value(). Off by default
        # so the standard path pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        ### Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: RandomState | None = None
        self._len_rng: RandomState | None = None

    def update(
        self,
        x: tuple[int, Sequence[tuple[int, float]]],
        weight: float,
        estimate: IntegerProbabilisticLatentSemanticIndexingDistribution,
    ) -> None:
        """Update sufficient statistics from one weighted sparse-bag observation.

        Args:
            x: Integer PLSI observation as ``(doc_id, [(value_id, count), ...])``.
            weight: Observation weight.
            estimate: Previous integer PLSI estimate.

        """
        if not isinstance(estimate, IntegerProbabilisticLatentSemanticIndexingDistribution):
            raise TypeError("integer PLSI accumulator estimate has the wrong model type")
        if (
            estimate.num_vals != self.num_vals
            or estimate.num_states != self.num_states
            or estimate.num_docs != self.num_docs
        ):
            raise ValueError("integer PLSI accumulator estimate dimensions do not match")
        checked_weight = finite_weight(weight, label="integer PLSI observation weight")
        d_id, bag, total = _canonical_observation(x, self.num_docs, self.num_vals)
        if checked_weight == 0.0:
            return
        xv = np.asarray([u[0] for u in bag], dtype=np.int64)
        xc = np.asarray([u[1] for u in bag], dtype=np.float64)

        update = (estimate.prob_mat[xv, :] * estimate.state_mat[d_id, :]).T
        denominators = np.sum(update, axis=0)
        if checked_weight > 0.0 and np.any(denominators <= 0.0):
            raise ValueError("integer PLSI observation is impossible under the estimate")
        if len(xv):
            update *= xc * checked_weight / denominators
        self.comp_count[d_id, :] += np.sum(update, axis=1)
        np.add.at(self.word_count, (slice(None), xv), update)
        self.doc_count[d_id] += checked_weight

        self.len_acc.update(total, checked_weight, estimate.len_dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize accumulator random states from ``rng``.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _acc_rng and _len_rng.

        Returns:
            None.

        """
        seeds = rng.randint(maxrandint, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._len_rng = RandomState(seed=seeds[1])
        self._init_rng = True

    def initialize(self, x: tuple[int, Sequence[tuple[int, float]]], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics from one weighted sparse-bag observation.

        Args:
            x: Integer PLSI observation as ``(doc_id, [(value_id, count), ...])``.
            weight: Observation weight.
            rng: Random state used for seeded initialization.

        """
        checked_weight = finite_weight(weight, label="integer PLSI observation weight")
        d_id, bag, total = _canonical_observation(x, self.num_docs, self.num_vals)
        if checked_weight == 0.0:
            return
        if not self._init_rng:
            self._rng_initialize(rng)
        xv = np.asarray([u[0] for u in bag], dtype=np.int64)
        xc = np.asarray([u[1] for u in bag], dtype=np.float64)

        update = self._acc_rng.dirichlet(np.ones(self.num_states) / self.num_states, size=len(xc)).T
        update *= xc * checked_weight
        np.add.at(self.word_count, (slice(None), xv), update)
        self.comp_count[d_id, :] += np.sum(update, axis=1)
        self.doc_count[d_id] += checked_weight

        self.len_acc.update(total, checked_weight, self._len_rng)

    def seq_initialize(
        self,
        x: tuple[T1 | None, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Vectorized initialization of sufficient statistics form an encoded sequence of observations in arg 'x'.

        The encoded sequence 'x' is a Tuple length 2. The first component contains data type Optional[T1]
        corresponding to the sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document id's for the lengths above (len = len(x)).

        Args:
            x: Encoded sequence of iid observations of PLSI. See above for details.
            weights (ndarray): Weights for observations in encoded sequence.
            rng (RandomState): Used to initialize member RandomState variables.

        Returns:
            None.

        """
        nn, rows = _validated_encoded(x, self.num_vals, self.num_docs)
        xv, xc, xd, xi, xn, xm = rows
        checked_weights = observation_weights(
            weights,
            len(xn),
            label="integer PLSI observation weights",
        )

        if not self._init_rng:
            self._rng_initialize(rng)

        update = self._acc_rng.dirichlet(np.ones(self.num_states) / self.num_states, size=len(xv)).T
        update *= xc * checked_weights[xi]
        self.word_count += vec_bincount3(xv, update, out=np.zeros_like(self.word_count, dtype=np.float64))
        self.doc_count += np.bincount(xm, checked_weights, minlength=self.num_docs)
        self.comp_count += vec_bincount4(xd, update, out=np.zeros_like(self.comp_count, dtype=np.float64))

        self.len_acc.seq_initialize(nn, checked_weights, self._len_rng)

    def seq_update(
        self,
        x: tuple[T1 | None, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        weights: np.ndarray,
        estimate: IntegerProbabilisticLatentSemanticIndexingDistribution,
    ) -> None:
        """Vectorized update of sufficient statistics for encoded sequence of iid observations in x.

        The encoded sequence 'x' is a Tuple length 2. The first component contains data type Optional[T1]
        corresponding to the sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document id's for the lengths above (len = len(x)).

        Args:
            x: Encoded sequence of iid observations of PLSI model. See above for details.
            weights (np.ndarray): Numpy array of observation weights.
            estimate (IntegerProbabilisticLatentSemanticIndexingDistribution): Prior estimate of IntegerProbabilisticLatentSemanticIndexingDistribution object.

        Returns:
            None.

        """
        if not isinstance(estimate, IntegerProbabilisticLatentSemanticIndexingDistribution):
            raise TypeError("integer PLSI accumulator estimate has the wrong model type")
        if (
            estimate.num_vals != self.num_vals
            or estimate.num_states != self.num_states
            or estimate.num_docs != self.num_docs
        ):
            raise ValueError("integer PLSI accumulator estimate dimensions do not match")
        nn, rows = _validated_encoded(x, self.num_vals, self.num_docs)
        xv, xc, xd, xi, xn, xm = rows
        checked_weights = observation_weights(
            weights,
            len(xn),
            label="integer PLSI observation weights",
        )
        if len(xv):
            denominators = np.sum(estimate.prob_mat[xv, :] * estimate.state_mat[xd, :], axis=1)
            if np.any((denominators <= 0.0) & (checked_weights[xi] > 0.0)):
                raise ValueError("integer PLSI observation is impossible under the estimate")
        fast_seq_update(
            xv,
            xc,
            xd,
            xi,
            xm,
            checked_weights,
            estimate.prob_mat,
            estimate.state_mat,
            self.word_count,
            self.comp_count,
            self.doc_count,
        )

        """

        temp = xc*weights[xi]
        update  = estimate.prob_mat[xv, :] * estimate.state_mat[xd, :]

        temp /= np.sum(update, axis=1)
        update *= temp[:,None]

        #vec_bincount1(xv, update, self.word_count.T)
        #vec_bincount1(xd, update, self.comp_count)
        #bincount(xm, weights, self.num_docs)

        for i in range(self.num_states):
            self.word_count[i,:] += np.bincount(xv, weights=update[:,i], minlength=self.num_vals)
            self.comp_count[:,i] += np.bincount(xd, weights=update[:,i], minlength=self.num_docs)
        self.doc_count += np.bincount(xm, weights=weights, minlength=self.num_docs)
        """

        # Fused-EM fast path: recover the per-document data log-likelihood that
        # estimate.seq_log_density would return. PLSI's seq_log_density is the exact (non-variational)
        # marginal -- no posterior loop to reuse -- so we reproduce its per-word row probabilities,
        # log/count weighting, per-document bincount, the document-prior term, and the optional
        # length term. Gated; standard path untouched.
        if self._track_ll:
            cnt = len(xn)
            w_ll = np.zeros(len(xv), dtype=np.float64)
            index_dot(estimate.prob_mat, xv, estimate.state_mat, xd, w_ll)
            np.log(w_ll, out=w_ll)
            w_ll *= xc
            rv = np.zeros(cnt, dtype=np.float64)
            bincount(xi, w_ll, rv)
            rv += _encoded_coefficient(xc, xi, xn)
            rv += estimate.log_doc_vec[xm]
            rv = rv + estimate.len_dist.seq_log_density(nn)
            self._seq_ll += float(np.dot(checked_weights, rv))

        self.len_acc.seq_update(nn, checked_weights, estimate.len_dist)

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident E-step: the PLSI responsibility update (state-word x doc-state gather,
        per-pair normalization, and the word/doc segment sums) runs on the active engine, matching
        the host seq_update.
        """
        if not isinstance(estimate, IntegerProbabilisticLatentSemanticIndexingDistribution):
            raise TypeError("integer PLSI accumulator estimate has the wrong model type")
        if (
            estimate.num_vals != self.num_vals
            or estimate.num_states != self.num_states
            or estimate.num_docs != self.num_docs
        ):
            raise ValueError("integer PLSI accumulator estimate dimensions do not match")
        nn, rows = _validated_encoded(x, self.num_vals, self.num_docs)
        xv, xc, xd, xi, xn, xm = rows
        weights_np = observation_weights(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            len(xn),
            label="integer PLSI observation weights",
        )
        denominators = np.sum(estimate.prob_mat[xv, :] * estimate.state_mat[xd, :], axis=1)
        if len(xv):
            if np.any((denominators <= 0.0) & (weights_np[xi] > 0.0)):
                raise ValueError("integer PLSI observation is impossible under the estimate")
        xv_e = engine.asarray(np.asarray(xv, dtype=np.int64))
        xd_e = engine.asarray(np.asarray(xd, dtype=np.int64))
        xi_e = engine.asarray(np.asarray(xi, dtype=np.int64))

        prob = engine.asarray(estimate.prob_mat)  # (num_vals, S)
        state = engine.asarray(estimate.state_mat)  # (num_docs, S)
        update = prob[xv_e, :] * state[xd_e, :]  # (n_pairs, S)
        temp = engine.asarray(np.asarray(xc, dtype=np.float64)) * engine.asarray(weights_np)[xi_e]
        safe_denominators = engine.asarray(np.where(denominators > 0.0, denominators, 1.0))
        update = update * (temp / safe_denominators)[:, None]

        wc_rows = [engine.index_add(engine.zeros(self.num_vals), xv_e, update[:, i]) for i in range(self.num_states)]
        cc_cols = [engine.index_add(engine.zeros(self.num_docs), xd_e, update[:, i]) for i in range(self.num_states)]
        self.word_count += np.asarray(engine.to_numpy(engine.stack(wc_rows, axis=0)))
        self.comp_count += np.asarray(engine.to_numpy(engine.stack(cc_cols, axis=1)))
        self.doc_count += np.bincount(np.asarray(xm, dtype=np.int64), weights=weights_np, minlength=self.num_docs)
        self.len_acc.seq_update(nn, weights_np, estimate.len_dist)

    def combine(
        self, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, SS1 | None]
    ) -> "IntegerProbabilisticLatentSemanticIndexingAccumulator":
        """Merge aggregated integer PLSI sufficient statistics into this accumulator.

        The tuple is interpreted as ``(word_count, comp_count, doc_count,
        length_stats)``.

        Args:
            suff_stat: Aggregated integer PLSI sufficient statistics.

        Returns:
            This accumulator.

        """
        word_count, comp_count, doc_count, length_stats = _validated_integer_plsi_statistics(
            suff_stat,
            self.num_vals,
            self.num_states,
            self.num_docs,
        )
        self.word_count += word_count
        self.comp_count += comp_count
        self.doc_count += doc_count

        self.len_acc.combine(length_stats)

        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any | None]:
        """Return sufficient statistics as ``(word_count, comp_count, doc_count, length_stats)``."""
        return self.word_count.copy(), self.comp_count.copy(), self.doc_count.copy(), self.len_acc.value()

    def from_value(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray, SS1 | None]
    ) -> "IntegerProbabilisticLatentSemanticIndexingAccumulator":
        """Replace this accumulator's sufficient statistics.

        Args:
            x: Aggregated sufficient statistics in ``value`` format.

        Returns:
            This accumulator.

        """
        self.word_count, self.comp_count, self.doc_count, length_stats = _validated_integer_plsi_statistics(
            x,
            self.num_vals,
            self.num_states,
            self.num_docs,
        )
        self.len_acc.from_value(length_stats)

        return self

    def scale(self, c: float) -> "IntegerProbabilisticLatentSemanticIndexingAccumulator":
        """Scale linear latent counts and delegate document-length statistics."""
        c = validated_observation_weight(c, "integer PLSI scale factor")
        self.word_count *= c
        self.comp_count *= c
        self.doc_count *= c
        self.len_acc.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under configured keys.

        If wc_key is set, merge the state/word count variable.
        If sc_key is set, merge the doc/state count variable.
        If dc_key is set, merge the author count variable.

        The length accumulator receives the same merge request.

        Args:
            stats_dict: Mapping from merge keys to sufficient statistics.

        """
        if self.wc_key is not None:
            if self.wc_key in stats_dict:
                stats_dict[self.wc_key] += self.word_count
            else:
                # Copy on adoption: stats_dict must never alias this accumulator's own live
                # array, or a later tied accumulator's in-place += above would silently mutate
                # this accumulator's private word_count as a side effect of merging.
                stats_dict[self.wc_key] = self.word_count.copy()

        if self.sc_key is not None:
            if self.sc_key in stats_dict:
                stats_dict[self.sc_key] += self.comp_count
            else:
                stats_dict[self.sc_key] = self.comp_count.copy()

        if self.dc_key is not None:
            if self.dc_key in stats_dict:
                stats_dict[self.dc_key] += self.doc_count
            else:
                stats_dict[self.dc_key] = self.doc_count.copy()

        self.len_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace sufficient statistics from matching keys in ``stats_dict``.

        If wc_key is set, set the state/word count variable to matching key in stats_dict.
        If sc_key is set, set the doc/state count variable to matching key in stats_dict.
        If dc_key is set, set the author count variable to matching key in stats_dict.

        The length accumulator receives the same replace request.

        Args:
            stats_dict: Mapping from merge keys to sufficient statistics.

        """
        # Copy on replace too: without it, every tied accumulator ends up pointing at the SAME
        # array object, so any one of them later accumulating new local data would silently
        # corrupt every other tied accumulator's counts.
        if self.wc_key is not None:
            if self.wc_key in stats_dict:
                self.word_count = np.asarray(stats_dict[self.wc_key]).copy()
        if self.sc_key is not None:
            if self.sc_key in stats_dict:
                self.comp_count = np.asarray(stats_dict[self.sc_key]).copy()
        if self.dc_key is not None:
            if self.dc_key in stats_dict:
                self.doc_count = np.asarray(stats_dict[self.dc_key]).copy()

        self.len_acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerProbabilisticLatentSemanticIndexingDataEncoder":
        """Return an encoder compatible with integer PLSI observations."""
        len_encoder = self.len_acc.acc_to_encoder()
        return IntegerProbabilisticLatentSemanticIndexingDataEncoder(
            self.num_vals,
            self.num_docs,
            len_encoder=len_encoder,
        )


class IntegerProbabilisticLatentSemanticIndexingAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for integer PLSI EM sufficient-statistic accumulators."""

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """Create an accumulator factory.

        Args:
            num_vals: Number of word/value ids.
            num_states: Number of latent topics.
            num_docs: Number of document ids.
            len_factory: Optional accumulator factory for total bag length.
            keys: Optional merge keys for word, state, and document counts.
            name: Optional diagnostic name.

        Attributes:
            num_vals: Number of word/value ids.
            num_states: Number of latent topics.
            num_docs: Number of document ids.
            len_factory: Accumulator factory for total bag length.
            keys: Optional sufficient-statistic merge keys.
            name: Optional diagnostic name.

        """
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys if keys is not None else (None, None, None)
        self.num_vals = _positive_dimension(num_vals, "num_vals")
        self.num_states = _positive_dimension(num_states, "num_states")
        self.num_docs = _positive_dimension(num_docs, "num_docs")
        self.name = name

    def make(self) -> "IntegerProbabilisticLatentSemanticIndexingAccumulator":
        """Return a fresh integer PLSI accumulator."""
        return IntegerProbabilisticLatentSemanticIndexingAccumulator(
            self.num_vals,
            self.num_states,
            self.num_docs,
            len_acc=self.len_factory.make(),
            keys=self.keys,
            name=self.name,
        )


class IntegerProbabilisticLatentSemanticIndexingEstimator(ParameterEstimator):
    """Estimator for integer PLSI word/state/document probabilities and the optional length model."""

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        pseudo_count: float | tuple[float | None, float | None, float | None] | None = (None, None, None),
        suff_stat: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None = (
            None,
            None,
            None,
        ),
        name: str | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
    ) -> None:
        """Create an estimator for integer PLSI sufficient statistics.

        Args:
            num_vals: Number of word/value ids.
            num_states: Number of latent topics.
            num_docs: Number of document ids.
            len_estimator: Optional estimator for total bag length.
            pseudo_count: Optional smoothing counts for word, state, and
                document probabilities.
            suff_stat: Optional prior word, state, and document sufficient statistics.
            name: Optional diagnostic name.
            keys: Optional merge keys for word, state, and document counts.

        Attributes:
            num_vals: Number of word/value ids.
            num_states: Number of latent topics.
            num_docs: Number of document ids.
            len_estimator: Estimator for total bag length.
            pseudo_count: Smoothing counts.
            suff_stat: Optional prior sufficient statistics.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic merge keys.
        """
        self.num_vals = _positive_dimension(num_vals, "num_vals")
        self.num_states = _positive_dimension(num_states, "num_states")
        self.num_docs = _positive_dimension(num_docs, "num_docs")
        self.suff_stat = suff_stat if suff_stat is not None else (None, None, None)
        pseudo_count = broadcast_pseudo_count(pseudo_count, 3)
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None, None)
        for value in self.pseudo_count:
            if value is not None and (not np.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError("integer PLSI pseudo-counts must be finite and non-negative")
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def accumulator_factory(self) -> "IntegerProbabilisticLatentSemanticIndexingAccumulatorFactory":
        """Return an accumulator factory matching this estimator."""
        len_est = self.len_estimator.accumulator_factory()
        return IntegerProbabilisticLatentSemanticIndexingAccumulatorFactory(
            self.num_vals, self.num_states, self.num_docs, len_est, self.keys
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, SS1 | None]
    ) -> "IntegerProbabilisticLatentSemanticIndexingDistribution":
        """Estimate an integer PLSI distribution from aggregated sufficient statistics.

        Args:
            nobs: Optional observation count, accepted for the estimator interface.
            suff_stat: Aggregated word, topic, document, and length statistics.

        Returns:
            A fitted integer PLSI distribution.

        """
        word_count, comp_count, doc_count, len_suff_stats = _validated_integer_plsi_statistics(
            suff_stat,
            self.num_vals,
            self.num_states,
            self.num_docs,
        )
        observation_mass = float(doc_count.sum())
        validate_effective_sample_mass(nobs, observation_mass, label="integer PLSI effective sample")

        if self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt * self.suff_stat[0].T
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)

        elif self.pseudo_count[0] is not None and self.suff_stat[0] is None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)

        else:
            word_count_sum = np.sum(word_count, axis=1)
            zero_states = word_count_sum == 0
            word_prob_mat = word_count.T / np.where(zero_states, 1.0, word_count_sum)
            word_prob_mat[:, zero_states] = 1.0 / word_count.shape[1]

        if self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt * self.suff_stat[1]
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)

        elif self.pseudo_count[1] is not None and self.suff_stat[1] is None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)

        else:
            comp_count_sum = np.sum(comp_count, axis=1, keepdims=True)
            zero_docs = comp_count_sum[:, 0] == 0
            state_prob_mat = comp_count / np.where(comp_count_sum == 0, 1.0, comp_count_sum)
            state_prob_mat[zero_docs, :] = 1.0 / comp_count.shape[1]

        if self.pseudo_count[2] is not None and self.suff_stat[2] is not None:
            adj_cnt = self.pseudo_count[2] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt * self.suff_stat[2]
            doc_prob_vec /= np.sum(doc_prob_vec)

        elif self.pseudo_count[2] is not None and self.suff_stat[2] is None:
            adj_cnt = self.pseudo_count[2] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt
            doc_prob_vec /= np.sum(doc_prob_vec)

        else:
            doc_count_sum = np.sum(doc_count)
            if doc_count_sum > 0:
                doc_prob_vec = doc_count / doc_count_sum
            else:
                doc_prob_vec = np.ones(len(doc_count)) / len(doc_count)

        len_dist = self.len_estimator.estimate(observation_mass, len_suff_stats)

        return IntegerProbabilisticLatentSemanticIndexingDistribution(
            word_prob_mat, state_prob_mat, doc_prob_vec, name=self.name, len_dist=len_dist
        )


class IntegerProbabilisticLatentSemanticIndexingDataEncoder(DataSequenceEncoder):
    """Encode integer PLSI observations into flattened sparse bag arrays and length features."""

    def __init__(
        self,
        num_vals: int,
        num_docs: int,
        len_encoder: DataSequenceEncoder | None = NullDataEncoder(),
    ) -> None:
        """Create an encoder for integer PLSI observations.

        Args:
            len_encoder: Optional encoder for total bag length.

        Attributes:
            len_encoder: Encoder for total bag length.

        """
        self.num_vals = _positive_dimension(num_vals, "num_vals")
        self.num_docs = _positive_dimension(num_docs, "num_docs")
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return (
            "IntegerProbabilisticLatentSemanticIndexingDataEncoder("
            f"num_vals={self.num_vals}, num_docs={self.num_docs}, len_dist={self.len_encoder})"
        )

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` uses the same length encoder.

        Args:
            other: Object to compare.

        Returns:
            ``True`` when both encoders use equivalent length encoders.

        """
        if isinstance(other, IntegerProbabilisticLatentSemanticIndexingDataEncoder):
            return (
                other.num_vals == self.num_vals
                and other.num_docs == self.num_docs
                and other.len_encoder == self.len_encoder
            )
        else:
            return False

    def seq_encode(
        self, x: Sequence[tuple[int, Sequence[tuple[int, float]]]]
    ) -> tuple[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Encode iid PLSI observations for vectorized ``seq_*`` methods.

        Input arg 'x' is a sequence of iid PLSI observations having form

        x = [ (doc_id, [(value, count),...]),... ].

        The return value has two entries. The first contains the optional length encoding. The second contains:
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document d_id for each word-count pair in the arrays above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each document.
            xm (ndarray[float]): Flattened array of document d_id's for the lengths above (len = len(x)).

        Args:
            x (Sequence[Tuple[int, Sequence[Tuple[int, float]]]]): See above for details.

        Returns:
            See above for details.

        """
        xv = []
        xc = []
        counts_per_doc = np.empty(len(x), dtype=np.int32)
        xn = np.empty(len(x), dtype=np.float64)
        xm = np.empty(len(x), dtype=np.int32)

        for i, value in enumerate(x):
            d_id, bag, total = _canonical_observation(value, self.num_docs, self.num_vals)
            v = [u[0] for u in bag]
            c = [u[1] for u in bag]

            xv.extend(v)
            xc.extend(c)
            counts_per_doc[i] = len(v)
            xn[i] = total
            xm[i] = d_id

        xv = np.asarray(xv, dtype=np.int32)
        xc = np.asarray(xc, dtype=np.float64)
        xd = np.repeat(xm, counts_per_doc)
        xi = np.repeat(np.arange(len(x), dtype=np.int32), counts_per_doc)

        nn = self.len_encoder.seq_encode(xn)
        rows = IntegerPLSIEncodedData(
            xv,
            xc,
            xd,
            xi,
            xn,
            xm,
            self.num_vals,
            self.num_docs,
        )
        return nn, rows

    def row_count(self, x: Any) -> int:
        """Return the validated number of document-bag rows in an encoded payload."""
        _, rows = _validated_encoded(x, self.num_vals, self.num_docs)
        return len(rows.totals)


@numba.njit(
    "void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:,:], float64[:,:], float64[:], float64[:])",
    fastmath=True,
    cache=True,
)
def fast_seq_log_density(xv, xc, xd, xi, xm, wmat, smat, dvec, out):
    """Numba kernel for accumulating encoded integer PLSI log-density contributions."""
    n = len(xv)
    m = len(xm)
    k = smat.shape[1]
    for i in range(n):
        ll = 0.0
        cc = xc[i]
        i1 = xv[i]
        i2 = xd[i]
        i3 = xi[i]
        for j in range(k):
            ll += wmat[i1, j] * smat[i2, j]
        out[i3] += cc * np.log(ll)
    for i in range(m):
        out[i] += dvec[xm[i]]


@numba.njit(
    "void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:,:], float64[:,:])", fastmath=True, cache=True
)
def fast_seq_component_log_density(xv, xc, xd, xi, xm, wmat, out):
    """Numba kernel for accumulating per-state component log-density contributions."""
    n = len(xv)
    k = wmat.shape[1]
    for i in range(n):
        cc = xc[i]
        i1 = xv[i]
        i3 = xi[i]
        for j in range(k):
            out[i3, j] += np.log(wmat[i1, j]) * cc


@numba.njit(
    "void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:], float64[:,:], float64[:,:], "
    "float64[:,:], float64[:,:], float64[:])",
    fastmath=True,
    cache=True,
)
def fast_seq_update(xv, xc, xd, xi, xm, weights, wmat, smat, wcnt, scnt, dcnt):
    """Numba kernel for the integer PLSI EM expected-count update."""
    n = len(xv)
    m = len(xm)
    k = smat.shape[1]
    posterior = np.zeros(k, dtype=np.float64)
    for i in range(n):
        norm_const = 0.0
        cc = xc[i]
        i1 = xv[i]
        i2 = xd[i]
        ww = weights[xi[i]]
        if ww == 0.0:
            continue
        for j in range(k):
            temp = wmat[i1, j] * smat[i2, j]
            posterior[j] = temp
            norm_const += temp
        norm_const = ww * cc / norm_const
        for j in range(k):
            temp = posterior[j] * norm_const
            wcnt[j, i1] += temp
            scnt[i2, j] += temp
    for i in range(m):
        dcnt[xm[i]] += weights[i]


@numba.njit("float64[:](float64[:,:], int32[:], float64[:,:], int32[:], float64[:])", cache=True)
def index_dot(x, xi, y, yi, out):
    """Return row-wise dot products ``x[xi[i]] @ y[yi[i]]`` into ``out``."""
    n = x.shape[1]
    for i in range(len(xi)):
        i1 = xi[i]
        i2 = yi[i]
        for j in range(n):
            out[i] += x[i1, j] * y[i2, j]
    return out


@numba.njit("float64[:](int32[:], float64[:], float64[:])", cache=True)
def bincount(x, w, out):
    """Accumulate weighted one-dimensional group sums into ``out``."""
    for i in range(len(x)):
        out[x[i]] += w[i]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])", cache=True)
def vec_bincount1(x, w, out):
    """Accumulate matrix-row weights into groups indexed by ``x``."""
    n = w.shape[1]
    for i in range(len(x)):
        for j in range(n):
            out[x[i], j] += w[i, j]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], int32[:], float64[:,:])", cache=True)
def vec_bincount2(x, w, y, out):
    """Accumulate rows ``w[y[i], :]`` into groups indexed by ``x``."""
    for i in range(len(x)):
        out[x[i], :] += w[y[i], :]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])", cache=True)
def vec_bincount3(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    Used to update comp counts for word/state probabilities.

    N = len(x)
    S = number of states.
    U = unique values in x can take on (unique words in corpus).

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): S by U matrix.

    Returns:
        Numpy 2-d array.

    """
    for j in range(len(x)):
        out[:, x[j]] += w[:, j]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])", cache=True)
def vec_bincount4(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    Used to initialize doc/state counts.

    N = len(x)
    S = number of states.
    U = unique values in x can take on. (Unique number of authors).

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): U by S matrix.

    Returns:
        Numpy 2-d array.

    """
    for j in range(len(x)):
        out[x[j], :] += w[:, j]
    return out


def _register_int_plsi_engine_kernel():
    """Register the engine-resident integer-PLSI kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class IntegerProbabilisticLatentSemanticIndexingKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("IntegerProbabilisticLatentSemanticIndexingKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class IntegerProbabilisticLatentSemanticIndexingKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return IntegerProbabilisticLatentSemanticIndexingKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(
        IntegerProbabilisticLatentSemanticIndexingDistribution,
        IntegerProbabilisticLatentSemanticIndexingKernelFactory(),
    )


_register_int_plsi_engine_kernel()
