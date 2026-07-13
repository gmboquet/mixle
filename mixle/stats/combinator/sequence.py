"""Sequence distributions with iid elements and an optional length model.

Data type (T): Assume the sequence distribution has a base distribution 'dist' compatible with data type T and length
distribution compatible with positive integers len_dist with respective densities P_dist() and P_len(). The density
of the sequence distribution is given by

p_mat(x) = P_dist(x[0])*...*P_dist(x[n-1])*P_len(n),

for an observation x of data type Sequence[T] having length n.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, LengthFrontierMerge, ProductEnumerator
from mixle.inference.fisher import Path
from mixle.stats.combinator.composite import _distribute_child_prior
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    ContractError,
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
    prefix_contract_error,
)

T = TypeVar("T")  # Data type of Sequence distribution dist.
E1 = TypeVar("E1")  # Generic type of distribution encoding.
E2 = TypeVar("E2")  # Generic type of length encoding.
SS1 = TypeVar("SS1")  # Generic type for sufficient statistic of base dist.
SS2 = TypeVar("SS2")  # Generic type for sufficient statistics of length dist.

E = tuple[np.ndarray, np.ndarray, np.ndarray, E1, E2 | None]


from mixle.inference.fisher import (
    FisherView,
    FixedFisherView,
    SufficientStatisticVectorizer,
    _full_info_from_view,
    _is_null_dist,
    _length_support,
    _seq_encode_model,
    to_fisher,
)


class SequenceDistribution(SequenceEncodableProbabilityDistribution):
    """Independent sequence distribution built from a component observation distribution."""

    def compute_capabilities(self):
        """Return compute-backend metadata inherited from the element and length distributions."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = (self.dist,) if self.null_len_dist else (self.dist, self.len_dist)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="numba_adapter")

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        len_normalized: bool | None = False,
        name: str | None = None,
        prior: tuple[Any, Any] | None = None,
    ) -> None:
        """Create a sequence distribution with iid elements and an optional length model.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Set base distribution of sequence (compatible with T).
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Length distribution for modeling lengths
                of sequences of observations (compatible with type int).
            len_normalized (Optional[bool]): If True, take geometric mean density for any density evaluation.
            name (Optional[str]): Set name to instance of SequenceDistribution.
            prior (Optional): Joint parameter prior ``(entry_prior, length_prior)`` distributed to the
                base distribution and length distribution via ``set_prior``. ``None`` (default) leaves
                both children plain point models (existing behavior byte-identical).

        Attributes:
            dist (SequenceEncodableProbabilityDistribution): Base distribution of sequence (compatible with T).
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Length distribution for modeling lengths
                of sequences of observations (compatible with type int). Set to NullDistribution if None is passed.
            len_normalized (Optional[bool]): If True, take geometric mean density for any density evaluation.
            name (Optional[str]): Name to instance of SequenceDistribution.
            null_len_distribution (bool): True if 'len_dist' is set to instance of NullDistribution.

        """
        self.dist = dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.len_normalized = len_normalized
        self.name = name

        self.null_len_dist = supports(self.len_dist, Neutral)
        self.set_prior(prior)

    def get_prior(self) -> tuple[Any, Any]:
        """Return the joint prior as ``(entry_prior, length_prior)`` from the wrapped children."""
        return self.dist.get_prior(), self.len_dist.get_prior()

    def set_prior(self, prior: tuple[Any, Any] | None) -> None:
        """Distribute ``(entry_prior, length_prior)`` to the base and length distributions.

        ``prior=None`` is a no-op (children keep their existing priors, leaving the MLE path
        byte-identical); otherwise the two-element prior is pushed to the base and length children via
        their own ``set_prior``.
        """
        if prior is None:
            return
        self.dist.set_prior(prior[0])
        self.len_dist.set_prior(prior[1])

    def expected_log_density(self, x: Sequence[T]) -> float:
        """Prior-expected log-density of the sequence ``x`` (sum over entries + length term)."""
        rv = 0.0
        for i in range(len(x)):
            rv += self.dist.expected_log_density(x[i])

        if self.len_normalized and len(x) > 0:
            rv /= len(x)

        if not self.null_len_dist:
            rv += self.len_dist.expected_log_density(len(x))

        return rv

    def seq_expected_log_density(self, x: E) -> np.ndarray:
        """Vectorized prior-expected log-density over sequence-encoded input ``x``."""
        idx, icnt, inz, enc_seq, enc_nseq = x

        if np.all(icnt == 0):
            ll_sum = np.zeros(len(icnt), dtype=float)
        else:
            ll = self.dist.seq_expected_log_density(enc_seq)
            ll_sum = np.bincount(idx, weights=ll, minlength=len(icnt))

            if self.len_normalized:
                ll_sum = ll_sum * icnt

        if not self.null_len_dist and enc_nseq is not None:
            nll = self.len_dist.seq_expected_log_density(enc_nseq)
            ll_sum += nll

        return ll_sum

    def compute_declaration(self):
        """Return the symbolic declaration for sequence elements and optional length statistics."""
        from mixle.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        base = declaration_for(self.dist)
        length = None if self.null_len_dist else declaration_for(self.len_dist)
        children = tuple(d for d in (base, length) if d is not None)
        roles = []
        if base is not None:
            roles.append("element")
        if length is not None:
            roles.append("length")
        return DistributionDeclaration(
            name="sequence",
            distribution_type=type(self),
            parameters=(),
            statistics=(
                StatisticSpec("elements", kind="child_stat"),
                StatisticSpec("lengths", kind="child_stat"),
            ),
            support="sequence",
            children=children,
            child_roles=tuple(roles),
            differentiable=all(child.differentiable for child in children),
        )

    def __str__(self) -> str:
        """Return a constructor-style representation of the sequence distribution."""
        s1 = str(self.dist)
        s2 = str(self.len_dist)
        s3 = repr(self.len_normalized)
        s4 = repr(self.name)

        return "SequenceDistribution(%s, len_dist=%s, len_normalized=%s, name=%s)" % (s1, s2, s3, s4)

    def density(self, x: Sequence[T]) -> float:
        """Evaluate the density of SequenceDistribution at observed sequence x.

        Assume x is a Sequence of data type T with length n > 0. Assume P_dist() is the density for the base
        distribution with data type T of SequenceDistribution, and P_len() is the length distribution with data type
        int. Then,

        P(x) = P_dist(x[0])*...*P_dist(x[n-1])*P_len(n), if len_normalize is False,

        or,

        P(x) = (P_dist(x[0])*...*P_dist(x[n-1])*P_len(n))^(1/n) if len_normalize is True.



        Args:
            x (Sequence[T]): Sequence of iid observations from base distribution of SequenceDistribution.

        Returns:
            Density evaluated at observation x.


        """
        rv = 1.0

        for i in range(len(x)):
            rv *= self.dist.density(x[i])

        if not self.null_len_dist:
            rv *= self.len_dist.density(len(x))

        if self.len_normalized and len(x) > 0:
            rv = np.power(rv, 1.0 / len(x))

        return rv

    def density_semantics(self):
        """Return the joined density semantics of the element distribution."""
        from mixle.stats.compute.pdist import join_density_semantics

        return join_density_semantics(c.density_semantics() for c in [self.dist])

    def log_density(self, x: Sequence[T]) -> float:
        """Evaluate the log-density of SequenceDistribution at observed sequence x.

        See density() for details.

        Args:
            x (Sequence[T]): Sequence of iid observations from base distribution of SequenceDistribution.

        Returns:
            Log-density evaluated at observation x.

        """
        rv = 0.0

        for i in range(len(x)):
            rv += self.dist.log_density(x[i])

        if self.len_normalized and len(x) > 0:
            rv /= len(x)

        if not self.null_len_dist:
            rv += self.len_dist.log_density(len(x))

        return rv

    def seq_ld_lambda(self):
        """Return vectorized log-density callables for encoded data."""
        rv = self.dist.seq_ld_lambda()

        if not self.null_len_dist:
            rv.extend(self.len_dist.seq_ld_lambda())

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of SequenceDistribution.log-density evaluated on sequence encoded x.

        Args:
            x (E): Sequence encoded data observation.

        Returns:
            Numpy array of log-density evaluated at each encoded observation value x.

        """
        idx, icnt, inz, enc_seq, enc_nseq = x

        if np.all(icnt == 0):
            ll_sum = np.zeros(len(icnt), dtype=float)

        else:
            ll = self.dist.seq_log_density(enc_seq)
            ll_sum = np.bincount(idx, weights=ll, minlength=len(icnt))

            if self.len_normalized:
                ll_sum = ll_sum * icnt

        if not self.null_len_dist and enc_nseq is not None:
            nll = self.len_dist.seq_log_density(enc_nseq)
            ll_sum += nll

        return ll_sum

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded sequences."""
        from mixle.stats.compute.backend import backend_seq_log_density

        idx, icnt, inz, enc_seq, enc_nseq = x
        nseq = len(icnt)
        ll_sum = engine.zeros(nseq)

        if len(idx) > 0:
            elem_ll = backend_seq_log_density(self.dist, enc_seq, engine)
            eidx = engine.asarray(idx)
            if self.len_normalized:
                elem_ll = elem_ll * engine.asarray(icnt)[eidx]
            ll_sum = engine.index_add(ll_sum, eidx, elem_ll)

        if not self.null_len_dist and enc_nseq is not None:
            ll_sum = ll_sum + backend_seq_log_density(self.len_dist, enc_nseq, engine)

        return ll_sum

    @classmethod
    def backend_stacked_params(cls, dists: Sequence[SequenceDistribution], engine: Any) -> dict[str, Any]:
        """Return stacked child routes for homogeneous sequence mixtures."""
        from mixle.stats.compute.stacked import stacked_component_params

        len_normalized = bool(dists[0].len_normalized)
        null_len_dist = bool(dists[0].null_len_dist)
        if any(bool(d.len_normalized) != len_normalized or bool(d.null_len_dist) != null_len_dist for d in dists):
            raise ValueError("Stacked SequenceDistribution components require matching length policy.")
        try:
            element_route = stacked_component_params([d.dist for d in dists], engine)
        except ValueError as exc:
            raise ValueError("Sequence element child %s is not stackable: %s" % (type(dists[0].dist).__name__, exc))
        length_route = None
        if not null_len_dist:
            try:
                length_route = stacked_component_params([d.len_dist for d in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "Sequence length child %s is not stackable: %s" % (type(dists[0].len_dist).__name__, exc)
                )
        return {
            "element_route": element_route,
            "length_route": length_route,
            "len_normalized": len_normalized,
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: E, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of sequence log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        idx, icnt, inz, enc_seq, enc_nseq = x
        nseq = len(icnt)
        rv = engine.zeros((nseq, int(params["num_components"])))

        if len(idx) > 0:
            eidx = engine.asarray(idx)
            element_scores = stacked_component_log_density(enc_seq, params["element_route"], engine)
            if params["len_normalized"]:
                element_scores = element_scores * engine.asarray(icnt)[eidx, None]
            rv = engine.index_add(rv, eidx, element_scores)

        if params["length_route"] is not None and enc_nseq is not None:
            rv = rv + stacked_component_log_density(enc_nseq, params["length_route"], engine)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: E, weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy sequence sufficient statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        idx, icnt, inz, enc_seq, enc_nseq = x
        ww = engine.asarray(weights)
        num_components = int(tuple(getattr(ww, "shape", (0, 0)))[1])
        outer_estimators = tuple(getattr(estimator, "estimators", ()))

        element_estimators = tuple(getattr(component_est, "estimator", None) for component_est in outer_estimators)
        element_estimator = (
            StackedEstimatorView(element_estimators) if len(element_estimators) == num_components else None
        )
        if len(idx) > 0:
            eidx = engine.asarray(idx)
            element_weights = ww[eidx]
            if params["len_normalized"]:
                element_weights = element_weights * engine.asarray(icnt)[eidx, None]
        else:
            element_weights = engine.zeros((0, num_components))
        element_stats = stacked_component_sufficient_statistics(
            enc_seq, element_weights, params["element_route"], engine, element_estimator
        )
        element_by_component = unstack_component_stats(element_stats, num_components)

        if params["length_route"] is None or enc_nseq is None:
            length_by_component = tuple(None for _ in range(num_components))
        else:
            length_estimators = tuple(
                getattr(component_est, "len_estimator", None) for component_est in outer_estimators
            )
            length_estimator = (
                StackedEstimatorView(length_estimators) if len(length_estimators) == num_components else None
            )
            length_stats = stacked_component_sufficient_statistics(
                enc_nseq, ww, params["length_route"], engine, length_estimator
            )
            length_by_component = unstack_component_stats(length_stats, num_components)

        return tuple((element_by_component[i], length_by_component[i]) for i in range(num_components))

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from mixle.stats.compute.gradient import SequenceGradientFitState

        child = recurse(self.dist, engine, torch, leaves)
        len_child = None if self.null_len_dist else recurse(self.len_dist, engine, torch, leaves)
        return SequenceGradientFitState(self, child, len_child)

    def to_fisher(self, **kwargs):
        """Structural Fisher view for the sequence."""
        if hasattr(self, "dist"):
            return SequenceFisherView(self)
        return super().to_fisher(**kwargs)

    def to_exponential_family(self, engine: Any = None):
        """Return the iid exponential-family view, or ``None``.

        An iid sequence of an exponential family is itself an exponential family with
        the shared element ``eta`` and ``T(x) = sum_t T_0(x_t)``.  This holds only when
        the length is not separately modeled and not length-normalized (a length term
        or normalization breaks the single-exp-family form); otherwise returns ``None``,
        as does a non-exp-family element.
        """
        from mixle.engines import NUMPY_ENGINE
        from mixle.stats.compute.exp_family import IIDExponentialFamilyForm, to_exponential_family

        if not self.null_len_dist or self.len_normalized:
            return None
        eng = NUMPY_ENGINE if engine is None else engine
        element = to_exponential_family(self.dist, engine=eng)
        if element is None:
            return None
        return IIDExponentialFamilyForm(distribution=self, element=element, engine=eng)

    def sampler(self, seed: int | None = None) -> SequenceSampler:
        """Create a sampler for this sequence distribution.

        Note: If member len_dist (SequenceEncodableDistribution) is NullDistribution() and or not compatible with
        data type int, an error is thrown.

        Args:
            seed (Optional[int]): Used to set seed of random number generator used to sample.

        Returns:
            SequenceSampler object.

        """
        if self.null_len_dist:
            raise ValueError(
                "Error: len_dist cannot be none for SequenceDistribution.sampler(seed:Optional[int]=None)."
            )
        else:
            return SequenceSampler(self.dist, self.len_dist, seed)

    def estimator(self, pseudo_count: float | None = None) -> SequenceEstimator:
        """Return an estimator for iid sequence observations and their lengths."""
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)

        return SequenceEstimator(
            self.dist.estimator(pseudo_count=pseudo_count),
            len_estimator=len_est,
            len_normalized=self.len_normalized,
            name=self.name,
        )

    def decomposition(self):
        """Sequences are iid: this names the data (sequence) axis, sufficient stats SUM-reduce. The planner
        sizes this axis from N (the data), not a fixed model count; the base ``dist`` may itself decompose."""
        from mixle.stats.compute.decomposition import DecompAxis, Decomposition, ReductionOp

        return Decomposition(axis=DecompAxis.SEQUENCE, num_units=1, reduction=ReductionOp.SUM, exact=True)

    def dist_to_encoder(self) -> SequenceDataEncoder:
        """Return an encoder for iid sequence observations.

        Base distribution DataSequenceEncoder and length distribution DataSequenceEncoder objects are passed.

        Returns:
            SequenceDataEncoder object.

        """
        dist_encoder = self.dist.dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        encoders = (dist_encoder, len_encoder)
        return SequenceDataEncoder(encoders=encoders)

    def enumerator(self) -> SequenceEnumerator:
        """Returns SequenceEnumerator iterating sequences in descending probability order."""
        return SequenceEnumerator(self)

    def structural_fine_bucket(self, value, quantizer) -> int:
        """Sum of per-element buckets plus the length term -- mirrors the count index's per-length
        L-fold element convolution shifted by the length-term bucket."""
        total = sum(self.dist.structural_fine_bucket(x, quantizer) for x in value)
        if not self.null_len_dist:
            total += self.len_dist.structural_fine_bucket(len(value), quantizer)
        return total

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """Structural count index: per-length L-fold self-convolution of the element histogram.

        log p(x) = sum_i log p(x_i) + log p(len(x)). For each length L the count histogram of the
        L-element sum is the L-fold self-convolution of the element count histogram, shifted by the
        length term's bucket; the total is the pooled sum over lengths. Lengths come from the length
        distribution's enumerator in descending probability, so once the length term alone exceeds
        the depth bound every later length does too and we stop. Sequences are unranked by resolving
        the contributing length, then the per-position element buckets via the convolution unranker.
        """
        from mixle.enumeration.quantization.core import child_count_index
        from mixle.enumeration.quantization.semiring import CountSemiring
        from mixle.stats.compute.pdist import EnumerationError

        if self.null_len_dist:
            raise EnumerationError(self, reason="no length distribution is modeled (len_dist is Null)")
        if self.len_normalized:
            raise EnumerationError(self, reason="len_normalized densities are not enumerable")

        sr = CountSemiring()
        elem_index, elem_truncated = child_count_index(
            self.dist, "SequenceDistribution.dist", quantizer, max_fine_bucket
        )
        truncated = elem_truncated

        # Collect lengths (descending probability) whose length-term bucket is within the bound.
        lengths: list[tuple[int, float]] = []  # (L, lp_len)
        _LEN_CAP = 1 << 24
        for length, lp_len in child_enumerator(self.len_dist, "SequenceDistribution.len_dist"):
            if not isinstance(length, (int, np.integer)) or length < 0:
                continue
            if lp_len == -np.inf:
                continue
            if quantizer.fine_bucket(lp_len) > max_fine_bucket:
                truncated = True
                break
            lengths.append((int(length), float(lp_len)))
            if len(lengths) >= _LEN_CAP:
                truncated = True
                break

        if not lengths:
            return sr.zero(), truncated

        max_len = max(L for L, _ in lengths)
        # The iid-sequence reduction: total = (+)_L  scale_{p(L)}( (x)^L elem ). The k-fold products
        # share incrementally-built count histograms; each carries the flat product unranker.
        prefix = sr.power_prefix(elem_index, max_len, quantizer, max_fine_bucket)
        built = len(prefix) - 1
        if built < max_len:
            truncated = True

        total = sr.zero()
        for L, lp_len in lengths:
            if L > built:
                truncated = True
                continue
            piece = sr.scale(prefix[L], lp_len, quantizer, max_fine_bucket)
            if piece.hist.is_empty():
                continue
            total = sr.plus(total, sr.map_values(piece, list))

        return total, truncated


class SequenceEnumerator(DistributionEnumerator):
    """Best-first enumerator for finite-length iid sequences with a modeled length law."""

    def __init__(self, dist: SequenceDistribution) -> None:
        """Enumerates sequences (lists) in descending probability order.

        Lengths are pulled lazily from the length distribution's enumerator; each length L
        contributes an independent L-fold product stream over the (shared, buffered) element
        enumeration, offset by the length log-probability. Supports of different lengths are
        disjoint, so streams are merged without re-scoring; the next un-instantiated length's
        log-probability is a valid frontier bound since element log-probs are non-positive.

        Raises EnumerationError when no length distribution is modeled (the support is then
        ill-defined) or when len_normalized is set (the geometric-mean density breaks the
        additive log-density structure).

        Args:
            dist (SequenceDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        if dist.null_len_dist:
            raise EnumerationError(dist, reason="no length distribution is modeled (len_dist is Null)")
        if dist.len_normalized:
            raise EnumerationError(dist, reason="len_normalized densities are not enumerable")
        elem_buf = BufferedStream(child_enumerator(dist.dist, "SequenceDistribution.dist"))
        len_stream = BufferedStream(child_enumerator(dist.len_dist, "SequenceDistribution.len_dist"))
        self._merge = LengthFrontierMerge(
            len_stream, lambda n, lp_len: ProductEnumerator([elem_buf] * n, combine=list, offset=lp_len)
        )

    def __next__(self) -> tuple[Any, float]:
        return next(self._merge)


class SequenceSampler(DistributionSampler):
    """Sampler for iid sequences whose lengths are drawn from a length distribution."""

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        len_dist: SequenceEncodableProbabilityDistribution,
        seed: int | None = None,
    ) -> None:
        """Create a sampler for a sequence distribution.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Set the base distribution for the sequences (data type T).
            len_dist (SequenceEncodableProbabilityDistribution): Set the length distribution for the length of the
                sequences (support on positive integers).
            seed (Optional[int]): Set seed of random number generator for sampling.

        Attributes:
            dist (SequenceEncodableProbabilityDistribution): The Base distribution for the sequences (data type T).
            len_dist (SequenceEncodableProbabilityDistribution): Length distribution for the length of the
                sequences (support on positive integers).
            rng (RandomState): Random state used for sampling.
            dist_sampler (DistributionSampler): DistributionSampler instance from base distribution.
            len_sampler (DistributionSampler): DistributionSampler instance from length distribution.

        """
        self.dist = dist
        self.len_dist = len_dist
        self.rng = RandomState(seed)
        self.dist_sampler = self.dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[Any]:
        """Generate iid sequence samples.

        If ``size`` is ``None``, the length ``n`` is sampled from
        ``len_sampler`` and then ``n`` iid elements are drawn from the base
        distribution sampler.

        If ``size`` is positive, the process is repeated and a list of
        sequences is returned.

        With ``batched=True`` (default) all lengths are drawn at once and all elements across every sequence are
        drawn in a single vectorized call, then split by length. Because ``len_sampler`` and ``dist_sampler`` own
        independent ``RandomState`` streams, this yields the same draws as the legacy per-element loop
        (``batched=False``) but far faster.

        Args:
            size (Optional[int]) Number of sequences to be sampled.
            batched (bool): Vectorize element draws (default); set False for the legacy per-draw loop.

        Returns:
            List[T] or List[List[T]] with length(size).

        """
        if size is None:
            n = int(self.len_sampler.sample())
            if batched and n > 0:
                return list(self.dist_sampler.sample(size=n))
            return [self.dist_sampler.sample() for i in range(n)]
        if not batched:
            return [self.sample(batched=False) for i in range(size)]

        lengths = np.asarray(self.len_sampler.sample(size=size)).astype(int).reshape(-1)
        total = int(lengths.sum())
        flat = self.dist_sampler.sample(size=total) if total > 0 else []
        out: list[Any] = []
        offset = 0
        for n in lengths:
            n = int(n)
            out.append(list(flat[offset : offset + n]))
            offset += n
        return out


class SequenceAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator that delegates element and length sufficient statistics to child accumulators."""

    def __init__(
        self,
        accumulator: SequenceEncodableStatisticAccumulator,
        len_accumulator: SequenceEncodableStatisticAccumulator = NullAccumulator(),
        len_normalized: bool | None = False,
        keys: str | None = None,
    ) -> None:
        """Create an accumulator for sequence sufficient statistics.

        Args:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator for element sufficient statistics.
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for length sufficient statistics.
            len_normalized (Optional[bool]): Geometric mean of density taken if set to True. Else ignored.
            keys (Optional[str]): Set keys for merging sufficient statistics of SequenceAccumulator objects with
                matching keys.

        Attributes:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator for element sufficient statistics.
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for length sufficient statistics.
            len_normalized (Optional[bool]): Geometric mean of density taken if set to True. Else ignored.
            keys (Optional[str]): Set keys for merging sufficient statistics of SequenceAccumulator objects with
                matching keys.
            null_len_accumulator (bool): True if ``len_accumulator`` is neutral.
            _init_rng (bool): True if _len_rng has been initialized.
            _len_rng (Optional[RandomState]): None if not initialized. Set to a RandomState during
                initialize or seq_initialize functions.

        """
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator
        self.keys = keys
        self.len_normalized = len_normalized

        self.null_len_accumulator = supports(self.len_accumulator, Neutral)

        ### Seeds for initialize/seq_initialize consistency
        self._init_rng = False
        self._len_rng: RandomState | None = None

    def update(self, x: Sequence[T], weight: float, estimate: SequenceDistribution | None) -> None:
        """Update sequence sufficient statistics with one weighted observation.

        The element accumulator receives the sequence contents and the length
        accumulator receives the sequence length.

        Args:
            x (Sequence[T]): A sequence of iid observations of data type T.
            weight (float): Weight for sequence observation.
            estimate (Optional[SequenceDistribution]): SequenceDistribution instance to aggregate sufficient statistics
                with.

        Returns:
            None.

        """
        if estimate is None:
            w = weight / len(x) if (self.len_normalized and len(x) > 0) else weight

            for i in range(len(x)):
                self.accumulator.update(x[i], w, None)

            if not self.null_len_accumulator:
                self.len_accumulator.update(len(x), weight, None)

        else:
            w = weight / len(x) if (self.len_normalized and len(x) > 0) else weight

            for i in range(len(x)):
                self.accumulator.update(x[i], w, estimate.dist)

            if not self.null_len_accumulator:
                self.len_accumulator.update(len(x), weight, estimate.len_dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Set the _len_rng for consistency between initialize and seq_initialize methods.

        Args:
            rng (RandomState): Random state for initializing ``_len_rng``.

        Returns:
            None.

        """
        self._len_rng = RandomState(seed=rng.randint(2**31))
        self._init_rng = True

    def initialize(self, x: Sequence[T], weight: float, rng: RandomState) -> None:
        """Initialize sequence sufficient statistics from one weighted observation.

        Note: Calls _rng_initialize() method if _len_rng has not been set. This ensures consistency between initialize
        and seq_initialize calls.

        Method invokes calls to accumulator.initialize() and len_accumulator.initialize() if len_accumulator is not
        NullAccumulator.

        Args:
            x (Sequence[T]): Sequence of iid observations from base distribution.
            weight (float): Weight for sequence observation.
            rng (RandomState): Random state used during initialization.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if len(x) > 0:
            w = weight / len(x) if self.len_normalized else weight
            for xx in x:
                self.accumulator.initialize(xx, w, rng)

        if not self.null_len_accumulator:
            self.len_accumulator.initialize(len(x), weight, self._len_rng)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of SequenceAccumulator sufficient statistics from sequence encoded x.


        Args:
            x (E): Encoded data sequence.
            weights (np.ndarray[float]): Numpy array of floats for weighting observations.
            rng (RandomState): Random state used during initialization.

        Returns:
            None.

        """
        idx, icnt, inz, enc_seq, enc_nseq = x

        if not self._init_rng:
            self._rng_initialize(rng)

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]

        self.accumulator.seq_initialize(enc_seq, w, rng)

        if not self.null_len_accumulator:
            self.len_accumulator.seq_initialize(enc_nseq, weights, self._len_rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: SequenceDistribution | None) -> None:
        """Vectorized update of SequenceAccumulator sufficient statistics from sequence encoded x.

        Args:
            x (E): Encoded data sequence.
            weights (np.ndarray[float]): Numpy array of floats for weighting observations.
            estimate (Optional[SequenceDistribution]): SequenceDistribution instance to aggregate sufficient statistics
                with.

        Returns:
            None.

        """
        idx, icnt, inz, enc_seq, enc_nseq = x

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]

        self.accumulator.seq_update(enc_seq, w, estimate.dist if estimate is not None else None)

        if not self.null_len_accumulator:
            self.len_accumulator.seq_update(enc_nseq, weights, estimate.len_dist if estimate is not None else None)

    def seq_update_engine(self, x: E, weights: Any, estimate: SequenceDistribution | None, engine: Any) -> None:
        """Engine-resident E-step: per-element weights are gathered/normalized on the active engine
        and the base/length accumulators are routed through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        idx, icnt, inz, enc_seq, enc_nseq = x

        w_eng = engine.asarray(weights)
        idx_arr = np.asarray(idx, dtype=np.int64)
        if self.len_normalized:
            w = w_eng[idx_arr] * engine.asarray(np.asarray(icnt, dtype=np.float64)[idx_arr])
        else:
            w = w_eng[idx_arr]

        child_seq_update(self.accumulator, enc_seq, w, estimate.dist if estimate is not None else None, engine)

        if not self.null_len_accumulator:
            child_seq_update(
                self.len_accumulator, enc_nseq, w_eng, estimate.len_dist if estimate is not None else None, engine
            )

    def combine(self, suff_stat: tuple[SS1, SS2 | None]) -> SequenceAccumulator:
        """Combine the sufficient statistics of SequenceAccumulator instance with suff_stat arg.

        Args:
            suff_stat (Tuple[SS1, Optional[SS2]]): Tuple of sufficient statistics of base distribution and value for
                length distribution.

        Returns:
            SequenceAccumulator object.

        """
        self.accumulator.combine(suff_stat[0])

        if not self.null_len_accumulator:
            self.len_accumulator.combine(suff_stat[1])

        return self

    def value(self) -> tuple[Any, Any | None]:
        """Return Tuple[SS1, Optional[SS2]], sufficient statistics of base accumulator and length accumulator."""
        return self.accumulator.value(), self.len_accumulator.value()

    def from_value(self, x: tuple[SS1, SS2 | None]) -> SequenceAccumulator:
        """Set the SequenceAccumulator base accumulator and length accumulator to values of x.

        Args:
            x (Tuple[SS1, Optional[SS2]]): Tuple of sufficient statistics of base distribution and value for length
                distribution.

        Returns:
            SequenceAccumulator object.

        """
        self.accumulator.from_value(x[0])

        if not self.null_len_accumulator:
            self.len_accumulator.from_value(x[1])

        return self

    def scale(self, c: float) -> SequenceAccumulator:
        """Scale element and length sufficient statistics through their accumulators."""
        self.accumulator.scale(c)
        if not self.null_len_accumulator:
            self.len_accumulator.scale(c)
        return self

    def get_seq_lambda(self):
        """Return low-level sequence kernels from the element and length accumulators."""
        rv = self.accumulator.get_seq_lambda()

        if self.len_accumulator is not None:
            rv.extend(self.len_accumulator.get_seq_lambda())

        return rv

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merges member sufficient statistics with sufficient statistics that contain matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        self.accumulator.key_merge(stats_dict)

        if not self.null_len_accumulator:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Set member sufficient statistics to values of objects with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        self.accumulator.key_replace(stats_dict)

        if not self.null_len_accumulator:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> SequenceDataEncoder:
        """Return an encoder for iid sequence observations.

        Base distribution DataSequenceEncoder and length distribution DataSequenceEncoder objects are passed.

        Returns:
            SequenceDataEncoder object.

        """
        encoder = self.accumulator.acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()
        encoders = (encoder, len_encoder)
        return SequenceDataEncoder(encoders=encoders)


class SequenceAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for sequence accumulators composed from element and length accumulator factories."""

    def __init__(
        self,
        dist_factory: StatisticAccumulatorFactory,
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        len_normalized: bool | None = False,
        keys: str | None = None,
    ) -> None:
        """Factory for sequence accumulators.

        Args:
            dist_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for base distribution of sequence
                distribution.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for length distribution of sequence
                distribution.
            len_normalized (Optional[bool]): Standardize by length of sequence distribution.
            keys (Optional[str]): Set key for merging/combining sufficient statistics of SequenceAccumulator.

        Attributes:
            dist_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for base distribution of sequence
                distribution.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for length distribution of sequence
                distribution, set to NullAccumulatorFactory() if corresponding SequenceDistribution has no length
                distribution desired to be estimated.
            len_normalized (Optional[bool]): Standardize by length of sequence distribution.
            keys (Optional[str]): Key for merging/combining sufficient statistics of SequenceAccumulator.

        """
        self.dist_factory = dist_factory
        self.len_factory = len_factory
        self.len_normalized = len_normalized
        self.keys = keys

    def make(self) -> SequenceAccumulator:
        """Return a fresh sequence accumulator from the element and length factories."""
        len_acc = self.len_factory.make()
        return SequenceAccumulator(self.dist_factory.make(), len_acc, self.len_normalized, self.keys)


class SequenceEstimator(ParameterEstimator):
    """Estimator for iid sequence distributions from element and optional length sufficient statistics."""

    def __init__(
        self,
        estimator: ParameterEstimator,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        len_normalized: bool | None = False,
        name: str | None = None,
        keys: str | None = None,
        prior: tuple[Any, Any] | None = None,
    ) -> None:
        """Estimator for a sequence distribution from aggregated sufficient statistics.

        Requires arg 'estimator' to be ParameterEstimator of data type T, compatible with the observed entry values
        of SequenceDistribution.

        If arg 'len_estimator' is passed, it must be a ParameterEstimator compatible with non-negative
        integers.

        If len_estimator is NullEstimator() or None, len_dist is used as length distribution in estimation.

        Args:
            estimator (ParameterEstimator): Set ParameterEstimator for base distribution.
            len_estimator (Optional[ParameterEstimator]): Set ParameterEstimator for length distribution.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Set a fixed length distribution.
            len_normalized (Optional[bool]): Take geometric mean of density if True.
            name (Optional[str]): Set name to SequenceEstimator instance.
            keys (Optional[str]): Set key to SequenceEstimator instance for merging sufficient statistics.

        Attributes:
            estimator (ParameterEstimator): ParameterEstimator for base distribution.
            len_estimator (Optional[ParameterEstimator]): ParameterEstimator for length distribution. If None, set to
                NullEstimator.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Set a fixed length distribution.
            len_normalized (Optional[bool]): Take geometric mean of density if True.
            name (Optional[str]): Name of SequenceEstimator instance.
            keys (Optional[str]): Key for SequenceEstimator instance used in aggregating sufficient statistics.

        """
        self.estimator = estimator
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.keys = keys
        self.len_normalized = len_normalized
        self.name = name
        self.set_prior(prior)

    def get_prior(self) -> tuple[Any, Any]:
        """Return the joint prior as ``(entry_prior, length_prior)`` from the child estimators."""
        return self.estimator.get_prior(), self.len_estimator.get_prior()

    def set_prior(self, prior: tuple[Any, Any] | None) -> None:
        """Distribute ``(entry_prior, length_prior)`` to the entry and length estimators.

        ``prior=None`` is a no-op. The prior is pushed to the entry/length *estimators* (not to a
        fixed ``len_dist``), so each child performs its own conjugate update.
        """
        if prior is None:
            return
        _distribute_child_prior(self.estimator, prior[0])
        _distribute_child_prior(self.len_estimator, prior[1])

    def model_log_density(self, model: SequenceDistribution) -> float:
        """Sum the entry and length estimators' ``model_log_density`` on the corresponding children."""
        return self.estimator.model_log_density(model.dist) + self.len_estimator.model_log_density(model.len_dist)

    def accumulator_factory(self) -> SequenceAccumulatorFactory:
        """Return SequenceAccumulatorFactory from len_estimator and estimator member variables with keys passed."""
        len_factory = self.len_estimator.accumulator_factory()
        dist_factory = self.estimator.accumulator_factory()

        return SequenceAccumulatorFactory(dist_factory, len_factory, self.len_normalized, self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[Any, Any | None]) -> SequenceDistribution:
        """Estimate the element distribution and, when configured, the length distribution."""
        if not isinstance(suff_stat, (tuple, list)) or len(suff_stat) != 2:
            raise ContractError(
                "SequenceEstimator.estimate(suff_stat)",
                "a 2-tuple (entry_suff_stat, length_suff_stat)",
                "%s%s"
                % (
                    type(suff_stat).__name__,
                    " of length %d" % len(suff_stat) if isinstance(suff_stat, (tuple, list)) else "",
                ),
                "pass the 2-tuple produced by SequenceAccumulator.value(), not a bare entry sufficient statistic.",
            )

        try:
            entry_dist = self.estimator.estimate(nobs, suff_stat[0])
        except ContractError as e:
            raise prefix_contract_error("SequenceEstimator.estimator", e) from None

        if isinstance(self.len_estimator, NullEstimator):
            return SequenceDistribution(
                entry_dist,
                len_dist=self.len_dist,
                len_normalized=self.len_normalized,
                name=self.name,
            )

        else:
            try:
                len_dist = self.len_estimator.estimate(nobs, suff_stat[1])
            except ContractError as e:
                raise prefix_contract_error("SequenceEstimator.len_estimator", e) from None
            return SequenceDistribution(
                entry_dist,
                len_dist=len_dist,
                len_normalized=self.len_normalized,
                name=self.name,
            )


class SequenceDataEncoder(DataSequenceEncoder):
    """Encoder that flattens sequence elements and separately encodes sequence lengths."""

    def __init__(self, encoders: tuple[DataSequenceEncoder, DataSequenceEncoder]) -> None:
        """Create an encoder for sequence observations and their lengths.

        encoders[0] is a DataSequenceEncoder for data type T, producing encoded sequences of type T1.
        encoders[1] is a DataSequenceEncoder for data type int, production encoded sequences of type T2 or None.

        Args:
            encoders (Tuple[DataSequenceEncoder, DataSequenceEncoder]): Tuple of encoders for
                distribution and length distribution of sequence distribution.

        Attributes:
            encoder (DataSequenceEncoder): Encoder for the sequence element distribution.
            len_encoder (DataSequenceEncoder): Encoder for the length distribution. A neutral encoder means no
                intended length distribution.
            null_len_encoder (bool): True if len_encoder is a NullDataEncoder(), else False.
        """
        self.encoder = encoders[0]
        self.len_encoder = encoders[1]

        self.null_len_enc = supports(self.len_encoder, Neutral)

    def __str__(self) -> str:
        """Return a constructor-style representation of the sequence encoder."""
        s = "SequenceDataEncoder("
        s += str(self.encoder) + ",len_encoder="
        s += str(self.len_encoder) + ")"

        return s

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent sequence data encoder.

        Checks if other is a SequenceDataEncoder. If it is, the encoder and len_encoder member variables must also
        be equivalent.

        Args:
            other (object): Object to compare to SequenceDataEncoder instance.

        Returns:
            True if other is an equivalent sequence encoder.

        """
        if not isinstance(other, SequenceDataEncoder):
            return False

        else:
            if not self.encoder == other.encoder:
                return False

            if not self.len_encoder == other.len_encoder:
                return False

            return True

    def seq_encode(
        self, x: Sequence[Sequence[T]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...], Any | None]:
        """Encode iid sequence observations for vectorized ``seq_*`` methods.

        Data 'x' must be a Sequence of Sequences containing data types T consistent with the distribution encoder
        (DataSequenceEncoder) object 'encoder'. That is x: Sequence containing 'N' objects of xx: Sequence[T].

        Consider example data x = [ [0,1,2], [],[3,4]]. Then x: Sequence[Sequence[int]].

        Assume the data type returned by 'encoder.seq_encode()' is T1, and 'len_encoder.seq_encode()' is T2.

        rv1 (ndarray[int]): Index for values of positive length sequence entries. I.e. x produces -> [0,0,0,2,2]
        rv2 (ndarray[float]): Inverse of sequence lengths. I.e. x -> [1/3,1/3,1/3,0,1/2,1/2]
        rv3 (ndarray[bool]): True if length of sequence is not 0. I.e. x -> [True,True, True, False, True,True]
        rv4 (T1): Sequence encoding resulting from encoder.seq_encode() on list of all observed values.
        rv5 (Optional[T2]): Sequence encoding resulting len_encoder.seq_encode() on all sequence length values.

        Args:
            x (Sequence[Sequence[T]]): Sequence of Sequence[T], where T is compatible with base distribution of
                sequence distribution. Sequence of iid sequence observations.

        Returns:

        """
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise ContractError(
                "SequenceDistribution.seq_encode",
                "a sequence of sequences (one inner sequence per observation)",
                "%s" % type(x).__name__,
                "pass a list of sequences, e.g. [[0, 1, 2], [], [3, 4]].",
            )

        tx = []
        nx = []
        tidx = []

        for i in range(len(x)):
            row = x[i]
            try:
                row_len = len(row)
            except TypeError:
                raise ContractError(
                    "SequenceDistribution.entries (row %d)" % i,
                    "a sequence (list/tuple/etc.) of entries",
                    "%s" % type(row).__name__,
                    "each observation must itself be a sequence of entries -- row %d is a scalar, "
                    "not a sequence. Wrap it in a list, e.g. [%r]." % (i, row),
                ) from None

            nx.append(row_len)

            for j in range(row_len):
                tidx.append(i)
                tx.append(row[j])

        rv1 = np.asarray(tidx, dtype=int)
        rv2 = np.asarray(nx, dtype=float)
        rv3 = rv2 != 0

        rv2[rv3] = 1.0 / rv2[rv3]

        try:
            rv4 = self.encoder.seq_encode(tx)
        except ContractError as e:
            raise prefix_contract_error("SequenceDistribution.entries", e) from None
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise ContractError(
                "SequenceDistribution.entries",
                "every flattened entry compatible with the base distribution's data type",
                "an entry that raised %s: %s" % (type(e).__name__, e),
                "check that every element of every inner sequence matches the data type expected by "
                "the base distribution (%s)." % self.encoder,
            ) from e

        ### None if NullDataEncoder() for length
        rv5 = self.len_encoder.seq_encode(nx)

        return rv1, rv2, rv3, rv4, rv5


# --- Fisher view(s) co-located with this family ---
class SequenceFisherView(FixedFisherView):
    """Structured Fisher view for iid sequence distributions."""

    def __init__(self, dist: Any) -> None:
        self.child_view = to_fisher(dist.dist)
        self.len_view = None if _is_null_dist(getattr(dist, "len_dist", None)) else to_fisher(dist.len_dist)
        super().__init__(dist, self._labels_from_children())

    def _labels_from_children(self) -> list[Path]:
        labels = [("element",) + label for label in self.child_view.vectorizer.labels]
        if self.len_view is not None:
            labels.extend(("length",) + label for label in self.len_view.vectorizer.labels)
        return labels

    def _refresh_labels(self) -> None:
        self.labels = self._labels_from_children()
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    @staticmethod
    def _lengths_from_encoded(enc_data: Any) -> np.ndarray:
        _, inv_len, nonzero, _, _ = enc_data
        lengths = np.zeros(len(inv_len), dtype=np.int64)
        nz = np.asarray(nonzero, dtype=bool)
        lengths[nz] = np.rint(1.0 / np.asarray(inv_len, dtype=np.float64)[nz]).astype(np.int64)
        return lengths

    def _aggregate_flat(
        self, flat_stats: np.ndarray, idx: np.ndarray, n: int, inv_len: np.ndarray | None
    ) -> np.ndarray:
        out = np.zeros((n, flat_stats.shape[1]), dtype=np.float64)
        if len(idx) == 0:
            return out
        weights = np.asarray(inv_len, dtype=np.float64)[idx] if self.dist.len_normalized else 1.0
        if np.isscalar(weights):
            np.add.at(out, idx, flat_stats)
        else:
            np.add.at(out, idx, flat_stats * weights[:, None])
        return out

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        enc = _seq_encode_model(self.dist if estimate is None else estimate, list(data))
        return self._statistics_from_encoded(enc, estimate=estimate)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        idx, inv_len, _, enc_seq, enc_len = enc_data
        n = len(inv_len)
        if len(idx):
            flat = self.child_view.seq_expected_statistics(enc_seq)
            elem = self._aggregate_flat(flat, np.asarray(idx, dtype=np.int64), n, inv_len)
        else:
            elem = np.zeros((n, len(self.child_view.mean_statistics())), dtype=np.float64)
        blocks = [elem]
        if self.len_view is not None:
            blocks.append(self.len_view.seq_expected_statistics(enc_len))
        self._refresh_labels()
        return np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=np.float64)

    def _sequence_model_mean_cov(self) -> tuple[np.ndarray, np.ndarray]:
        support = _length_support(self.dist.len_dist)
        if support is None:
            raise NotImplementedError("sequence model Fisher requires a supported length distribution")
        lengths, probs = support
        child_mu = np.asarray(self.child_view.mean_statistics(), dtype=np.float64)
        child_cov = _full_info_from_view(self.child_view)
        child_outer = np.outer(child_mu, child_mu)

        elem_mean = np.zeros_like(child_mu)
        elem_second = np.zeros((len(child_mu), len(child_mu)), dtype=np.float64)
        elem_cond_means = []
        for n_float, p in zip(lengths, probs):
            n = max(int(round(n_float)), 0)
            if self.dist.len_normalized:
                if n > 0:
                    cond_mean = child_mu
                    cond_second = child_cov / float(n) + child_outer
                else:
                    cond_mean = np.zeros_like(child_mu)
                    cond_second = np.zeros_like(elem_second)
            else:
                cond_mean = float(n) * child_mu
                cond_second = float(n) * child_cov + float(n * n) * child_outer
            elem_mean += p * cond_mean
            elem_second += p * cond_second
            elem_cond_means.append(cond_mean)

        elem_cov = elem_second - np.outer(elem_mean, elem_mean)
        if self.len_view is not None:
            len_mat = self.len_view.expected_statistics_matrix(data=[int(round(v)) for v in lengths])
            len_mean = np.dot(probs, len_mat)
            len_second = np.dot((probs[:, None] * len_mat).T, len_mat)
            len_cov = len_second - np.outer(len_mean, len_mean)
            elem_len_second = np.zeros((len(child_mu), len(len_mean)), dtype=np.float64)
            for cond_mean, len_row, p in zip(elem_cond_means, len_mat, probs):
                elem_len_second += p * np.outer(cond_mean, len_row)
            cross = elem_len_second - np.outer(elem_mean, len_mean)

            mean = np.concatenate((elem_mean, len_mean))
            cov = np.zeros((len(mean), len(mean)), dtype=np.float64)
            d = len(child_mu)
            cov[:d, :d] = elem_cov
            cov[:d, d:] = cross
            cov[d:, :d] = cross.T
            cov[d:, d:] = len_cov
        else:
            mean = elem_mean
            cov = elem_cov

        cov = 0.5 * (cov + cov.T)
        diag = np.maximum(np.diag(cov), 0.0)
        cov[np.diag_indices_from(cov)] = diag
        return mean, cov

    def _model_mean(self) -> np.ndarray:
        return self._sequence_model_mean_cov()[0]

    def _model_fisher(self) -> np.ndarray:
        return self._sequence_model_mean_cov()[1]

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        """Return Fisher information, falling back to the generic Fisher view when needed."""
        try:
            return super().fisher_information(stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)
        except NotImplementedError:
            return FisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return Fisher-normalized sequence statistics, with a generic fallback."""
        try:
            return super().fisher_vectors(
                stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )
        except NotImplementedError:
            if stats is None:
                stats = self.expected_statistics_matrix(**kwargs)
            return FisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge
            )
