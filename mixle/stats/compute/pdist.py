"""Defines abstract classes for SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator,
ProbabilityDistribution, StatisticAccumulator, StatisticAccumulatorFactory, DataSequenceEncoder, ParameterEstimator,
ConditionalSampler, and DistributionSampler for classes of the mixle.stats.

"""

import itertools
import math
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

import numpy as np

from mixle.engines.arithmetic import *

SS = TypeVar("SS")


class DensitySemantics(Enum):
    """What a distribution's ``log_density`` returns relative to the true log-density.

    The default contract is :attr:`EXACT`; override ``density_semantics()`` on models whose
    ``log_density`` is a variational bound or an approximation, so callers can tell an exact likelihood
    from one (e.g. LDA's per-document ELBO). Surfaced via the ``ExactDensity`` capability and ``describe``.
    """

    EXACT = "exact"  # the true log p(x)
    LOWER_BOUND = "lower_bound"  # value <= true log p(x); e.g. a variational ELBO
    UPPER_BOUND = "upper_bound"  # value >= true log p(x)
    ESTIMATE = "estimate"  # an approximation with no guaranteed direction (plug-in / Monte Carlo)


def join_density_semantics(semantics) -> "DensitySemantics":
    """Combine child density semantics for a combinator whose log_density is monotone in its children.

    A combinator whose score rises with each child's log_density -- a mixture's ``logsumexp``, a
    composite's sum -- inherits: a lower bound if any child is a lower bound, an upper bound if any is
    an upper bound, exactness only if all children are exact, and an undirected ``ESTIMATE`` if bounds
    of both directions (or any estimate) are mixed.
    """
    kinds = set(semantics)
    has_lower = DensitySemantics.LOWER_BOUND in kinds
    has_upper = DensitySemantics.UPPER_BOUND in kinds
    if DensitySemantics.ESTIMATE in kinds or (has_lower and has_upper):
        return DensitySemantics.ESTIMATE
    if has_lower:
        return DensitySemantics.LOWER_BOUND
    if has_upper:
        return DensitySemantics.UPPER_BOUND
    return DensitySemantics.EXACT


class EnumerationError(NotImplementedError):
    """Raised when a distribution (or a child of a combinator) cannot enumerate its support.

    The path argument identifies the offending child within a combinator, e.g.
    'CompositeDistribution.dists[1]'.
    """

    def __init__(self, dist: Any, path: str = "", reason: str = "") -> None:
        self.leaf = dist
        self.path = path
        self.reason = reason
        msg = "%s does not support enumeration" % type(dist).__name__
        if path:
            msg = "%s -> %s" % (path, msg)
        if reason:
            msg += ": %s" % reason
        super().__init__(msg)


class KeyValidationError(ValueError):
    """Raised when keyed sufficient-statistic sites are incompatible.

    A key denotes an equality constraint across model sites.  Sites sharing a
    key must therefore have the same accumulator family and compatible estimator
    settings before their sufficient statistics are pooled.
    """

    pass


class ContractError(ValueError):
    """Raised when malformed data trips a combinator's ``seq_encode``/``estimate`` contract boundary.

    Mirrors :class:`EnumerationError`'s path-composition convention: ``path`` names the full field
    path through nested combinators (composed with ``" -> "``, outermost first), e.g.
    ``"CompositeDistribution.dists[2] -> SequenceDistribution.entries"``. Every ``ContractError``
    also carries what was expected, what actually arrived, and (when there is one) a concrete,
    non-generic suggestion for the likely fix -- so a caller reading only the message, not the
    traceback, knows what to change.
    """

    def __init__(self, path: str, expected: str, actual: str, fix: str = "") -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        self.fix = fix
        msg = "%s: expected %s, got %s." % (path, expected, actual)
        if fix:
            msg += " Fix: %s" % fix
        super().__init__(msg)


def prefix_contract_error(prefix: str, err: "ContractError") -> "ContractError":
    """Return a new ContractError with ``prefix`` prepended to ``err``'s field path.

    Used by a combinator to annotate a ``ContractError`` raised deep inside a child's
    ``seq_encode``/``estimate`` with the outer field position, so the final message names the
    FULL path down to where the failure actually occurred (e.g. a mixture-of-composites-of-
    sequences error names every level, not just the outermost combinator).
    """
    new_path = "%s -> %s" % (prefix, err.path) if err.path else prefix
    return ContractError(new_path, err.expected, err.actual, err.fix)


def child_enumerator(child: "ProbabilityDistribution", path: str) -> "DistributionEnumerator":
    """Construct child.enumerator(), annotating EnumerationError with the child's path.

    Combinator enumerators use this so a failure deep in a nested model reports the
    full path to the offending leaf, e.g.
    'CompositeDistribution.dists[1] -> MixtureDistribution.components[0] -> GaussianDistribution ...'.
    """
    try:
        return child.enumerator()
    except EnumerationError as e:
        new_path = path if not e.path else "%s -> %s" % (path, e.path)
        raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None


class ProbabilityDistribution(ABC):
    """Base class for all probability distributions in mixle.stats.

    A distribution evaluates the (log-)density of a single observation of its data
    type, creates a DistributionSampler for drawing observations, and creates a
    ParameterEstimator for re-estimating itself from data. Discrete distributions
    may additionally provide a DistributionEnumerator over their support.
    """

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible representation of this distribution."""
        from mixle.utils.serialization import to_serializable

        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbabilityDistribution":
        """Reconstruct a distribution from ``to_dict`` output."""
        from mixle.utils.serialization import from_serializable

        rv = from_serializable(payload)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this distribution as safe strict JSON."""
        from mixle.utils.serialization import to_json

        return to_json(self, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "ProbabilityDistribution":
        """Deserialize a distribution from ``to_json`` output."""
        from mixle.utils.serialization import from_json

        rv = from_json(text)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    def density(self, x: Any) -> float:
        """Return the probability density or mass at a single observation.

        Concrete default: exponentiate ``log_density`` (the abstract method subclasses must
        provide). Leaves with a cheaper closed form may override this.
        """
        return math.exp(self.log_density(x))

    def capabilities(self) -> frozenset[str]:
        """Return the capability names this distribution supports (see :mod:`mixle.capability`).

        Feature detection by behaviour rather than class — e.g. ``"Enumerable"``,
        ``"Conditionable"``, ``"ExponentialFamily"``, ``"RankableByIndex"``. Equivalent to
        ``mixle.capabilities(self)``; combinators report the set their children jointly preserve.
        """
        from mixle.capability import capabilities

        return capabilities(self)

    def density_semantics(self) -> DensitySemantics:
        """What ``log_density`` returns relative to the true log-density (default: exact).

        Override to declare that this distribution's ``log_density`` is a variational lower bound
        (ELBO), an upper bound, or an approximation rather than the exact ``log p(x)``. This is surfaced
        as the ``ExactDensity`` capability and noted in :func:`mixle.describe`, so code that needs an
        exact likelihood can ``require(x, ExactDensity)`` instead of silently trusting a bound.
        """
        return DensitySemantics.EXACT

    def tropical_displacement_bits(self) -> float:
        """Worst-case gap (in bits) between this law's structural *count* cost and its true log-density.

        The structural count-DP (and the ``seek`` built on it) bins each value by a cost that is
        EXACT for decomposable families -- composites/sequences/markov chains whose ``log p(x)`` is a
        sum of independent per-factor terms -- so this returns ``0.0`` by default.

        For a marginal family the latent index is summed out and the count index bins by the
        *tropical* (dominant-component/path) cost ``M(x)`` instead of the true ``log p(x)``. Because
        ``M(x) <= log p(x) <= M(x) + log N`` for an ``N``-way logsumexp, the two costs differ by at
        most ``log2(N)`` bits. :func:`mixle.enumeration.density_rank.marginal_seek` widens its rank
        bracket by exactly this many bits so the bracket is a *guaranteed* bound on the true marginal
        rank (not merely the tropical rank); override to return ``log2(N)`` for such a family.
        """
        return 0.0

    @abstractmethod
    def log_density(self, x: Any) -> float:
        """Return the log-density or log-mass at a single observation."""
        ...

    @abstractmethod
    def sampler(self, seed: int | None = None) -> "DistributionSampler":
        """Return a sampler for drawing observations from this distribution."""
        ...

    @abstractmethod
    def estimator(self, pseudo_count: float | None = None) -> "ParameterEstimator":
        """Return an estimator for fitting this distribution from data."""
        ...

    def to_fisher(self, **kwargs):
        """Return a Fisher-geometry view of this distribution.

        The default view is accumulator-backed, so distributions inherit a generic
        sufficient-statistic/Fisher-vector interface.  Each distribution owns its Fisher view by
        overriding this method in its own module; families not yet migrated to a per-file hook are
        resolved by the transitional type-name dispatch in :func:`mixle.inference.fisher._legacy_to_fisher`.
        """
        from mixle.inference.fisher import _legacy_to_fisher

        return _legacy_to_fisher(self, **kwargs)

    def to_exponential_family(self, engine: Any = None):
        """Return the canonical exponential-family view, or ``None``.

        The canonical form is ``p(x) = h(x) * exp(<eta, T(x)> - A(eta))``.  The default
        reads ``declaration_for(self).exponential_family`` (the per-family
        ``ExponentialFamilySpec``) and wraps it in an
        :class:`~mixle.stats.compute.exp_family.ExponentialFamilyForm`; it returns ``None`` when
        this family is not a (single) exponential family.  There is no type switch --
        adding a family is a matter of providing its spec.
        """
        from mixle.engines import NUMPY_ENGINE
        from mixle.stats.compute.declarations import declaration_for
        from mixle.stats.compute.exp_family import ExponentialFamilyForm

        declaration = declaration_for(self)
        if declaration is None or declaration.exponential_family is None:
            return None
        return ExponentialFamilyForm(
            distribution=self,
            spec=declaration.exponential_family,
            engine=NUMPY_ENGINE if engine is None else engine,
        )

    def get_prior(self) -> Optional["ProbabilityDistribution"]:
        """Return the conjugate/parameter prior carried by this distribution, if any.

        A distribution participates in the Bayesian (variational) protocol by
        carrying a prior over its parameters. The default returns whatever was
        stored on the ``prior`` attribute (``None`` for a plain point model),
        so frequentist distributions answer ``None`` and behave as MLE models.
        """
        return getattr(self, "prior", None)

    def set_prior(self, prior: Optional["ProbabilityDistribution"]) -> None:
        """Attach a parameter prior to this distribution.

        The default just records the prior; conjugate families override this to
        precompute the variational expected natural parameters used by
        ``expected_log_density``.
        """
        self.prior = prior

    def has_conjugate_prior(self) -> bool:
        """Return whether this family supports a closed-form conjugate Bayesian update.

        Uniform, family-level signal backed by the single ``conjugate_posterior`` registry: ``True``
        means ``mixle.stats.bayes.conjugate_posterior(self, data)`` returns an exact closed-form
        posterior; ``False`` means Bayesian inference must go through the numerical fitters
        (MAP / Laplace / MCMC / VI). This is the top tier of the inference-capability ladder
        (see :class:`mixle.capability.ConjugateUpdatable`). Distinct from the per-instance
        ``has_conj_prior`` flag, which records whether a prior is currently *attached*.
        """
        from mixle.stats.bayes.conjugate import is_conjugate_family

        return is_conjugate_family(self)

    def expected_log_density(self, x: Any) -> float:
        """Return the variational expectation ``E_q[log p(x | theta)]``.

        When the distribution carries a conjugate parameter posterior ``q`` this
        is the Bayesian E-step term; for a plain point model (no prior) it
        degenerates to the plug-in ``log_density(x)``. Conjugate families
        override this with their closed form.
        """
        return self.log_density(x)

    def enumerator(self) -> "DistributionEnumerator":
        """Return a DistributionEnumerator over this distribution's support.

        Distributions with an enumerable (discrete) support override this; the
        default raises EnumerationError.
        """
        raise EnumerationError(self)

    def support_size(self) -> int | None:
        """Return the number of distinct support points, or ``None`` if infinite/unknown.

        This is the cardinality primitive for bounding a truncated descending-probability sum: after
        enumerating the top ``k`` items (whose smallest probability is ``p_k``), every un-enumerated
        item has probability ``<= p_k``, so the remaining mass is ``<= (support_size - k) * p_k`` (see
        :func:`mixle.enumeration.density_rank.truncated_sum_bound`). Finite discrete leaves return their
        cardinality; decomposable combinators compose it structurally; an upper bound is acceptable
        (it only loosens the tail bound). Infinite or continuous supports return ``None``.
        """
        return None

    def support_is_finite(self) -> bool:
        """Return whether the support has a known finite cardinality."""
        return self.support_size() is not None

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0):
        """Build a bounded bit-quantized index over this distribution's support.

        This is a convenience wrapper around ``self.enumerator().quantized_index``.
        Non-enumerable distributions raise EnumerationError through enumerator().

        Args:
            max_bits (float): Maximum information content in bits to index.
            bin_width_bits (float): Width of each quantized probability bin in bits.

        Returns:
            mixle.enumeration.algorithms.QuantizedEnumerationIndex.

        """
        return self.enumerator().quantized_index(max_bits=max_bits, bin_width_bits=bin_width_bits)

    def density_quantile(self, q: float, n_samples: int = 20000, seed: int | None = None) -> Any:
        """Return a representative value at cumulative-density index ``q`` (descending-density order).

        The arbitrary-index / inverse of the probability-ordered cumulative that
        :func:`mixle.enumeration.density_rank.density_rank` computes (``G(x) = P(p(Y) >= p(x))``): ``q = 0``
        is the mode, ``q -> 1`` walks into the tail.  Families with a closed form override this
        (univariate continuous leaves expose the spatial ``quantile``; multivariate Gaussians and von
        Mises-Fisher override ``density_quantile`` exactly); the default here is the universal
        **Monte-Carlo representative** for any samplable family whose support is uncountable or coupled
        (parameter priors, continuous mixtures, ...): draw ``n_samples`` points, order them by
        descending density, and return the one at fractional rank ``q``.  Stochastic and approximate --
        use the exact methods (``enumerator``/``count_dp_seek`` for discrete) where available.

        Args:
            q (float): Cumulative-density index in ``[0, 1]``.
            n_samples (int): Monte-Carlo sample budget.
            seed (Optional[int]): Sampler seed (reproducible).
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1].")
        samples = self.sampler(seed).sample(int(n_samples))
        with np.errstate(divide="ignore"):
            lps = np.asarray([float(self.log_density(y)) for y in samples], dtype=np.float64)
        order = np.argsort(-lps, kind="stable")
        pick = int(round(q * (len(order) - 1))) if len(order) > 1 else 0
        return samples[int(order[pick])]

    def density_enumeration(
        self, num_points: int, n_samples: int = 20000, seed: int | None = None
    ) -> list[tuple[Any, float]]:
        """Return ``num_points`` representative ``(value, log_density)`` pairs in descending density.

        The continuous analogue of :meth:`enumerator` (which enumerates a countable support exactly):
        for an uncountable or coupled support there is no exact element enumeration, so this returns a
        **Monte-Carlo representative** sweep -- ``n_samples`` draws ordered by descending density,
        keeping the ``num_points`` most probable (distinct) representatives, i.e. "the support,
        most-probable region first".  Stochastic and approximate; prefer :meth:`enumerator` where the
        support is countable.

        Args:
            num_points (int): Number of representatives to return.
            n_samples (int): Monte-Carlo sample budget.
            seed (Optional[int]): Sampler seed (reproducible).
        """
        from mixle.enumeration.algorithms import freeze

        samples = self.sampler(seed).sample(int(n_samples))
        with np.errstate(divide="ignore"):
            scored = sorted(((float(self.log_density(y)), y) for y in samples), key=lambda t: -t[0])
        out: list[tuple[Any, float]] = []
        seen: set = set()
        for lp, y in scored:
            if lp == -np.inf:
                continue
            try:
                key = freeze(y)
            except TypeError:
                key = id(y)
            if key in seen:
                continue
            seen.add(key)
            out.append((y, lp))
            if len(out) >= int(num_points):
                break
        return out

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """Build a structural CountIndex over this distribution's support, bounded by depth.

        This is the count-semiring counterpart of ``quantized_index``: it returns per-fine-bucket
        *counts* of the complete model probability together with a structural unranker, so the
        support can be indexed without being enumerated. The default builds a leaf index from the
        exact ``enumerator()`` truncated at ``max_fine_bucket`` (efficient for closed-form/small-support
        families); exponential-support composers (Composite/Sequence/MarkovChain) override this with
        a dynamic program over the model's likelihood recursion.

        Args:
            quantizer (mixle.enumeration.quantization.Quantizer): Fine/coarse bucketing.
            max_fine_bucket (int): Inclusive depth bound on indexed fine buckets.

        Returns:
            Tuple (CountIndex, truncated) -- truncated is True when in-support values were dropped
            because they fell beyond the depth bound.

        """
        from mixle.enumeration.quantization.core import leaf_count_index

        return leaf_count_index(self.enumerator(), quantizer, max_fine_bucket)

    def count_budget_index(
        self, budget_bits: float, bin_width_bits: float = 1.0, oversample: int = 8, num_workers: int | None = None
    ):
        """Build a budget-bounded quantized seek index covering the top ``2**budget_bits`` values.

        Computes per-bin counts structurally (never enumerating the domain) and accumulates coarse
        bins in descending-probability order until the cumulative count reaches the budget. The
        returned LazyQuantizedEnumerationIndex supports arbitrary-rank seek/unranking; each unranked
        value carries its exact ``log_density``.

        Args:
            budget_bits (float): Index into the top ``2**budget_bits`` most probable values.
            bin_width_bits (float): Coarse output bin width in bits.
            oversample (int): Fine buckets per coarse bin (accumulation resolution).

        Returns:
            mixle.enumeration.algorithms.LazyQuantizedEnumerationIndex.

        """
        from mixle.enumeration.quantization.core import count_budget_index

        return count_budget_index(
            self, budget_bits, bin_width_bits=bin_width_bits, oversample=oversample, num_workers=num_workers
        )

    def count_budget_distinct(
        self,
        budget_bits: float,
        bin_width_bits: float = 1.0,
        oversample: int = 8,
        dedup: str = "canonical",
        start: int = 0,
        stop: int | None = None,
        max_entries: int = 1 << 16,
        num_workers: int | None = None,
    ):
        """Iterate DISTINCT (value, exact_log_prob) over the count-budget index, approx descending.

        For exact-count families this equals the ordered index stream. For the over-counting
        MARGINAL families (Mixture/HMM) it removes the component/path duplicates by one of two modes:

          - ``dedup='canonical'`` (default): a STATELESS predicate (``is_canonical_copy``) keeps a
            value only at its dominant copy (best-weighted component / min-cost path), via the
            value's structural fine bucket (``structural_fine_bucket`` -- the SAME sum-of-floored
            sub-buckets the count index used, so nested composite/sequence values are binned
            consistently and never dropped). O(1) memory and random-accessible: ``start``/``stop``
            select an arbitrary STRUCTURAL rank range, so you can begin anywhere and the work
            partitions across workers with no shared state. GUARANTEE: every distinct in-budget value
            is emitted at least once (completeness). It is NOT strictly once -- a value is emitted
            once per component/path that ties within its minimal 1-bit coarse bin; that residue is
            inherent to a stateless ``(value, coarse_bin)`` rule (the tied copies are
            indistinguishable to it) and is bounded by the number of components/contributing paths
            (not reduced by ``oversample``, which only refines the intermediate fine bucket). For a
            strictly-once stream, use ``dedup='window'`` or de-duplicate downstream.
          - ``dedup='window'``: a bounded ``max_entries`` LRU over the stream (catches every duplicate
            within the window regardless of dominance, but is sequential -- ``start`` must be 0).

        Note: ``start``/``stop`` index the STRUCTURAL enumeration, not the distinct rank. Jumping to
        the k-th *distinct* value in O(1) is not possible -- it needs exact distinct per-bin counts,
        which require materializing the component/path overlap structure.
        """
        from mixle.enumeration.quantization.core import distinct_budget_stream

        return distinct_budget_stream(
            self,
            budget_bits,
            bin_width_bits=bin_width_bits,
            oversample=oversample,
            dedup=dedup,
            start=start,
            stop=stop,
            max_entries=max_entries,
            num_workers=num_workers,
        )

    def is_canonical_copy(self, value, coarse_bin: int, quantizer) -> bool:
        """Return True if ``coarse_bin`` is ``value``'s dominant (canonical) bin in the count index.

        Stateless deduplication hook for the over-counting MARGINAL families: a value that the
        structural index emits once per component / state-path is kept only at the copy whose bin is
        the minimal (most probable) one. The default returns True -- exact-count families
        (Composite/Sequence/MarkovChain) never duplicate, so every copy is canonical.
        """
        return True

    def structural_fine_bucket(self, value, quantizer) -> int:
        """Minimum fine bucket where ``value`` is placed by this distribution's count index.

        Mirrors ``quantized_count_index`` exactly so that stateless canonical-copy dedup can predict
        the bucket the index actually used. The count DP bins a composite/nested value by a SUM of
        *floored* per-factor buckets (a convolution), which differs from a single floor of the exact
        joint log-density by up to the number of factors -- so a canonical check that used
        ``fine_bucket(log_density(value))`` would mispredict and silently drop nested values. The
        leaf default is that single floor (correct for atomic families); combinators
        (Composite/Sequence/Mixture) override to recurse the same way their count index composes.
        """
        return quantizer.fine_bucket(float(self.log_density(value)))

    def quantized_multi_cross_index(
        self, others: list["ProbabilityDistribution"], max_bits, bin_width_bits: float = 1.0
    ):
        """Build an aligned bounded cross-bin view against other distributions.

        The generic implementation is a bounded candidate join: it unions the bounded
        quantized indexes of all participating distributions, then evaluates every
        candidate under every distribution. Structured distributions can override this
        to build the same aligned rows from support algebra instead.
        """
        from mixle.enumeration.algorithms import QuantizedCrossIndex, freeze

        dists = [self] + list(others)
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = tuple([float(max_bits)] * len(dists))
        if len(max_bits_tuple) != len(dists):
            raise ValueError("max_bits length must match the number of distributions.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        seen = set()
        values = []
        truncated = False
        for dist, bit_bound in zip(dists, max_bits_tuple):
            if bit_bound < 0.0:
                truncated = True
                continue
            index = dist.quantized_index(max_bits=bit_bound, bin_width_bits=bin_width_bits)
            truncated = truncated or index.truncated
            for value, _ in index.iter_from():
                key = freeze(value)
                if key not in seen:
                    seen.add(key)
                    values.append(value)

        items = []
        for value in values:
            items.append((value, tuple(float(dist.log_density(value)) for dist in dists)))
        return QuantizedCrossIndex.from_items(
            items, max_bits=max_bits_tuple, bin_width_bits=bin_width_bits, truncated=truncated
        )

    def quantized_cross_index(self, other: "ProbabilityDistribution", max_bits, bin_width_bits: float = 1.0):
        """Build an aligned bounded cross-bin view against another distribution."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class SequenceEncodableProbabilityDistribution(ProbabilityDistribution):
    """ProbabilityDistribution with vectorized log-density evaluation on encoded data.

    dist_to_encoder() returns a DataSequenceEncoder whose seq_encode() output is
    consumed by seq_log_density() (and by the matching accumulator's seq_update /
    seq_initialize), enabling fast vectorized estimation over iid sequences.
    """

    engine_ready = ("numpy",)

    def supported_engines(self) -> tuple[str, ...]:
        """Return engine names this distribution can evaluate on directly."""
        from mixle.stats.compute.capabilities import capabilities_for

        return capabilities_for(self).engine_ready

    def supports_engine(self, engine: Any) -> bool:
        """Return True when the distribution can safely use ``engine``."""
        from mixle.stats.compute.capabilities import capabilities_for

        return capabilities_for(self).supports_engine(engine)

    def decomposition(self) -> "Decomposition":
        """Return how this distribution may be split across devices (model parallelism).

        Defaults to :meth:`Decomposition.atomic` -- not split, replicated. Combinators / latent
        families override this to declare their component / factor / state axis. See
        :mod:`mixle.stats.compute.decomposition`.
        """
        from mixle.stats.compute.decomposition import Decomposition

        return Decomposition.atomic()

    def seq_ld_lambda(self):
        """Return vectorized log-density callables for encoded data."""
        return [self.seq_log_density]

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        return np.asarray([self.log_density(u) for u in x])

    def seq_expected_log_density(self, x: Any) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations.

        Degenerates to ``seq_log_density`` for a plain point model; conjugate
        families override with a closed form. See ``expected_log_density``.
        """
        return self.seq_log_density(x)

    def seq_log_density_lambda(self):
        """Return vectorized log-density callables for encoded data."""
        return [self.seq_log_density]

    def kernel(self, engine=None, estimator: Optional["ParameterEstimator"] = None):
        """Return an engine-aware evaluation kernel for this distribution."""
        from mixle.stats.compute.kernel import kernel_for

        return kernel_for(self, engine=engine, estimator=estimator)

    @abstractmethod
    def dist_to_encoder(self) -> "DataSequenceEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        ...


class DistributionSampler(ABC):
    """Draws iid observations from a distribution using a seeded RandomState.

    sample(size=None) returns a single observation of the distribution's data type;
    sample(size=n) returns a length-n collection of observations.
    """

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        seed: int | None = None,
        *,
        rng: np.random.RandomState | None = None,
    ) -> None:
        self.dist = dist
        # ``rng`` (keyword-only) lets callers share one RandomState across samplers for composable,
        # reproducible streams; ``seed`` remains the default scalar-seed path when no rng is supplied.
        self.rng = rng if rng is not None else np.random.RandomState(seed)

    def new_seed(self) -> int:
        """Return a fresh random seed drawn from this sampler's RandomState."""
        return self.rng.randint(0, maxrandint)

    @abstractmethod
    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw observations.

        Combinator samplers (mixture/sequence/...) accept ``batched``. With
        ``batched=True`` (the default) each child stream is drawn in one vectorized
        call instead of a per-draw Python loop -- far faster. Because every child
        sampler owns an independent ``RandomState``, batching consumes each stream
        in the same order as the loop, so the draws are identical to the legacy
        path. ``batched=False`` forces that legacy per-draw loop as a guaranteed-
        stable reference. Leaf samplers are already vectorized and ignore the flag.
        """
        ...


class DistributionEnumerator(ABC):
    """Lazy iterator over the support of dist in non-increasing probability order.

    Yields (value, log_prob) pairs, possibly infinitely many. Contract:
      - Each support value is yielded exactly once (deduplication is the
        enumerator's responsibility).
      - log_prob equals dist.log_density(value) up to float round-off (~1e-10),
        and the sequence of log_probs is non-increasing up to the same tolerance.
      - Values with zero probability are skipped, never yielded.
      - Ties are broken deterministically by insertion order; no further guarantee.
    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        self.dist = dist

    def __iter__(self) -> "DistributionEnumerator":
        return self

    @abstractmethod
    def __next__(self) -> tuple[Any, float]: ...

    def top_k(self, k: int) -> list[tuple[Any, float]]:
        """Return the k most probable (value, log_prob) pairs (fewer if the support is smaller)."""
        return list(itertools.islice(self, k))

    def top_p(self, p: float, max_items: int | None = None) -> list[tuple[Any, float]]:
        """Return the smallest descending-probability prefix whose total probability reaches ``p``.

        The nucleus / minimal high-probability set: because values are yielded in non-increasing
        probability order, the returned prefix is a minimum-size set of outcomes whose summed mass is
        ``>= p`` (e.g. ``p=0.95`` gives a 95%-coverage support set -- the discrete analogue of nucleus
        sampling). Accumulation stops as soon as the cumulative probability reaches ``p``.

        ``max_items`` caps how many values are pulled so an infinite or heavy-tailed support cannot
        run away; if the cap is hit before the threshold, the (sub-threshold) prefix gathered so far
        is returned. ``p >= 1.0`` on an infinite support therefore requires ``max_items``.

        Args:
            p (float): Target cumulative probability; ``p <= 0`` returns the empty set.
            max_items (Optional[int]): Hard cap on the number of values pulled.

        Returns:
            List of (value, log_prob) pairs in non-increasing probability order.

        """
        if p <= 0.0:
            return []
        out: list[tuple[Any, float]] = []
        total = 0.0
        for value, log_prob in self:
            out.append((value, log_prob))
            total += math.exp(log_prob)
            if total >= p or (max_items is not None and len(out) >= max_items):
                break
        return out

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0):
        """Precompute a bounded bit-quantized index over this enumeration.

        The index groups values by floor((-log2 p(x)) / bin_width_bits), includes only
        values with -log2 p(x) <= max_bits, and returns exact log probabilities for
        indexed values. Building the index consumes this enumerator.

        Args:
            max_bits (float): Maximum information content in bits to index.
            bin_width_bits (float): Width of each quantized probability bin in bits.

        Returns:
            mixle.enumeration.algorithms.QuantizedEnumerationIndex.

        """
        from mixle.enumeration.algorithms import QuantizedEnumerationIndex

        return QuantizedEnumerationIndex.from_enumerator(self, max_bits=max_bits, bin_width_bits=bin_width_bits)

    # -- where-does-a-value-sit / what-is-at-this-index, as methods on the enumerator ---------------
    # The enumerator is the one home for "rank a value", "seek the value at an index", and "iterate
    # from an index" over the weighted structure it enumerates. These delegate to the descending-
    # probability machinery in mixle.enumeration.density_rank (imported lazily to keep the pdist
    # contract layer free of an eager dependency on the enumeration package).

    def rank(self, value: Any):
        """Rank and cumulative probability of ``value`` in the descending-probability order.

        Returns a :class:`~mixle.enumeration.density_rank.DensityRankResult` with ``.rank`` (0-based
        count of strictly-more-probable outcomes; ``None`` if only the sampling estimate was used) and
        ``.cumulative_probability`` (``G(value) = P(p(Y) >= p(value))``). Exact head enumeration where
        the support is countable, with a Monte-Carlo fallback for the deep tail.
        """
        from mixle.enumeration.density_rank import density_rank

        return density_rank(self.dist, value)

    def seek(self, index: int):
        """The value at descending-probability ``index`` (0-based) -- the inverse of :meth:`rank`.

        Returns a :class:`~mixle.enumeration.density_rank.CountDPSeekResult` carrying the value and a
        provable ``[rank_lower, rank_upper]`` bracket. Uses the structural count-DP for decomposable
        families, so arbitrarily deep indices are reachable without enumerating the prefix.
        """
        from mixle.enumeration.density_rank import count_dp_seek

        return count_dp_seek(self.dist, index)

    def seek_certified(self, index: int):
        """The value at descending ``index`` with a GUARANTEED bracket on its TRUE marginal rank.

        Unlike :meth:`seek` -- whose bracket bounds only the *tropical* rank for a marginal family
        (mixture/HMM) -- this widens the rank window by the family's ``tropical_displacement_bits`` and
        divides out the component over-count, so the returned
        :class:`~mixle.enumeration.density_rank.MarginalSeekResult` ``[true_rank_lower, true_rank_upper]``
        provably contains ``#{u : log p(u) > log p(value)}``. It pins the rank exactly (``.exact``) for
        decomposable / provably-disjoint families and for shallow indices, and otherwise returns the
        certified provable envelope. For a decomposable family it agrees with :meth:`seek`.
        """
        from mixle.enumeration.density_rank import marginal_seek

        return marginal_seek(self.dist, index)

    def cumulative(self, value: Any):
        """``G(value) = P(p(Y) >= p(value))`` -- total mass of outcomes at least as probable as ``value``."""
        from mixle.enumeration.density_rank import cumulative_probability

        return cumulative_probability(self.dist, value)

    def nucleus_size(self, p: float):
        """Size of :meth:`top_p` (the minimal ``>= p``-mass set) WITHOUT materializing it.

        Returns a :class:`~mixle.enumeration.density_rank.CountDPTopPResult` with a provable size
        bracket, from the structural count-DP -- usable when the nucleus is far too large to list.
        """
        from mixle.enumeration.density_rank import count_dp_top_p

        return count_dp_top_p(self.dist, p)

    def from_index(self, start: int, stop: int | None = None):
        """Iterate ``(value, log_prob)`` in descending-probability order starting at structural ``start``.

        Yields the same stream as iterating a fresh enumerator but beginning at index ``start`` (and
        ending before ``stop`` if given). A fresh underlying enumeration is used, so this does not
        consume ``self``. (Decomposable families admit a direct structural jump via the count-budget
        index; the current implementation skips the best-first prefix -- the structural fast path is a
        WS-3 performance follow-up.)
        """
        return itertools.islice(self.dist.enumerator(), start, stop)


class ConditionalSampler(ABC):
    """Sampler mixin for conditional draws: sample_given(x) draws from P(. | x)."""

    @abstractmethod
    def sample_given(self, x):
        """Draw a sample from the conditional distribution given ``x``."""
        ...


class StatisticAccumulator(ABC, Generic[SS]):
    """Accumulates weighted sufficient statistics of type SS from observations.

    update(x, weight, estimate) adds one observation (estimate is the previous model,
    used for E-step posteriors; it may be None during initialization). Accumulators
    merge across partitions via combine(suff_stat) / value() / from_value(), and
    key_merge / key_replace pool statistics shared across model components through
    a stats_dict keyed by the accumulator's key.
    """

    def update(self, x: Any, weight: float, estimate) -> None:
        """Accumulate one weighted observation under an optional current estimate."""
        ...

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics from one weighted observation."""
        self.update(x, weight, estimate=None)

    @abstractmethod
    def combine(self, suff_stat: SS) -> "StatisticAccumulator":
        """Merge serialized sufficient statistics into this accumulator."""
        ...

    @abstractmethod
    def value(self) -> SS:
        """Return this accumulator's serialized sufficient statistics."""
        ...

    @abstractmethod
    def from_value(self, x: SS) -> "SequenceEncodableStatisticAccumulator":
        """Restore this accumulator from serialized sufficient statistics."""
        ...

    def scale(self, c: float) -> "StatisticAccumulator":
        """Scale linear sufficient statistics in-place by ``c``.

        The structural default is correct for ordinary weighted sums, nested
        tuples/lists/dicts, and numeric arrays. Families whose ``value()``
        payload includes non-linear metadata such as support bounds must
        override this method and leave that metadata unscaled.
        """
        return self.from_value(scale_suff_stat(self.value(), c))

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Pool this accumulator's statistics into ``stats_dict`` under its merge key.

        The structural default implements the common single-key pattern: store the accumulator
        under ``self.keys`` the first time the key is seen, else ``combine`` into the one already
        there. Accumulators with several named keys (e.g. an HMM's init/trans/state keys) or a
        non-accumulator stats payload override this. A ``keys`` of ``None`` (the default) is a no-op.
        """
        keys = getattr(self, "keys", None)
        if keys is not None:
            if keys in stats_dict:
                stats_dict[keys].combine(self.value())
            else:
                stats_dict[keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics from the pooled ``stats_dict`` entry (see key_merge)."""
        keys = getattr(self, "keys", None)
        if keys is not None and keys in stats_dict:
            self.from_value(stats_dict[keys].value())


class SequenceEncodableStatisticAccumulator(StatisticAccumulator[SS]):
    """StatisticAccumulator with vectorized updates on encoded data sequences.

    seq_update / seq_initialize consume the output of the matching
    DataSequenceEncoder's seq_encode() (obtained via acc_to_encoder()) together with
    a per-observation weight vector.
    """

    def get_seq_lambda(self):
        """Return optional low-level sequence-update kernels used by generated code."""
        pass

    @abstractmethod
    def seq_update(self, x, weights: np.ndarray, estimate) -> None:
        """Accumulate weighted sufficient statistics from sequence-encoded observations."""
        ...

    @abstractmethod
    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics from sequence-encoded observations."""
        ...

    @abstractmethod
    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Return a sequence encoder compatible with this accumulator."""
        ...


def scale_suff_stat(x: Any, c: float) -> Any:
    """Return ``x`` with numeric sufficient-statistic leaves multiplied by ``c``."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.number) and not np.issubdtype(x.dtype, np.bool_):
            return x * c
        return x.copy()
    if isinstance(x, dict):
        return {k: scale_suff_stat(v, c) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(scale_suff_stat(v, c) for v in x)
    if isinstance(x, list):
        return [scale_suff_stat(v, c) for v in x]
    if isinstance(x, np.generic):
        if np.issubdtype(type(x), np.number) and not np.issubdtype(type(x), np.bool_):
            return x * c
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return x * c
    return x


class StatisticAccumulatorFactory(ABC):
    """Factory whose make() returns a fresh, zeroed accumulator for one estimator."""

    @abstractmethod
    def make(self) -> "SequenceEncodableStatisticAccumulator":
        """Create a fresh accumulator instance."""
        ...


class ParameterEstimator(ABC, Generic[SS]):
    """Estimates a distribution from accumulated sufficient statistics.

    accumulator_factory() supplies accumulators that gather sufficient statistics of
    type SS, and estimate(nobs, suff_stat) maps those statistics (plus optional
    regularization configured on the estimator) to a new distribution.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible representation of this estimator."""
        from mixle.utils.serialization import to_serializable

        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParameterEstimator":
        """Reconstruct an estimator from ``to_dict`` output."""
        from mixle.utils.serialization import from_serializable

        rv = from_serializable(payload)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this estimator as safe strict JSON."""
        from mixle.utils.serialization import to_json

        return to_json(self, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "ParameterEstimator":
        """Deserialize an estimator from ``to_json`` output."""
        from mixle.utils.serialization import from_json

        rv = from_json(text)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    @abstractmethod
    def estimate(self, nobs: float | None, suff_stat: SS) -> "SequenceEncodableProbabilityDistribution":
        """Estimate a distribution from accumulated sufficient statistics."""
        ...

    @abstractmethod
    def accumulator_factory(self) -> "StatisticAccumulatorFactory":
        """Return the accumulator factory used to collect this estimator's sufficient statistics."""
        ...

    def resident_accumulation_supported(self) -> bool:
        """Return whether engine-resident (fixed-width) sufficient statistics suffice for ``estimate``.

        Most exponential-family M-steps consume only the resident sufficient statistics,
        so the default is ``True``. Estimators whose M-step needs more than that (e.g. a
        full count histogram for the negative-binomial dispersion solve) override this to
        ``False`` so stacked/generated kernels fall back to the host accumulator, keeping
        every backend's fixed point identical.
        """
        return True

    def get_prior(self) -> Optional["ProbabilityDistribution"]:
        """Return the parameter prior configured on this estimator, if any.

        The unified estimation contract treats the prior as the single
        regularization concept: ``None`` gives maximum likelihood, a conjugate
        prior gives the Bayesian posterior update inside ``estimate``. The
        default reads the ``prior`` attribute (``None`` when unset).
        """
        return getattr(self, "prior", None)

    def model_log_density(self, model: "ProbabilityDistribution") -> float:
        """Return the prior log-density of ``model``'s parameters (the ELBO global term).

        Used by the variational/MAP objective in ``fit``. The default is ``0.0``
        (no prior); conjugate estimators override this to evaluate their prior at
        the model's parameters, mapping to the prior's parameterization first.
        """
        return 0.0


class DataSequenceEncoder(ABC):
    """Encodes an iid data sequence into the vectorized form used by seq_* methods.

    seq_encode(x) transforms a sequence of observations into the encoding consumed
    by seq_log_density / seq_update / seq_initialize. Encoders must define __eq__
    (so equivalent encoders are interchangeable when batching) and a readable
    __str__.
    """

    def __str__(self) -> str:
        return type(self).__name__

    def seq_encode(self, x: Any) -> Any:
        """Encode the iid observation sequence x for vectorized evaluation."""
        return x

    def nbytes(self, x: Any) -> int:
        """Return the approximate in-memory byte size of an encoded payload."""
        return encoded_nbytes(x)

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...


def encoded_nbytes(x: Any) -> int:
    """Return an approximate byte size for nested encoded array payloads.

    Encoders mostly return arrays or tuples/lists/dicts of arrays. The helper
    keeps accounting structural and deterministic; Python object overhead is
    included only for scalar leaves where no array-native byte count exists.
    """
    return _encoded_nbytes(x, set())


def _encoded_nbytes(x: Any, seen: set[int]) -> int:
    oid = id(x)
    if oid in seen:
        return 0

    if isinstance(x, np.ndarray):
        seen.add(oid)
        return int(x.nbytes)

    nbytes = getattr(x, "nbytes", None)
    if nbytes is not None and not isinstance(x, (bytes, bytearray, str)):
        seen.add(oid)
        return int(nbytes)

    if hasattr(x, "numel") and hasattr(x, "element_size"):
        seen.add(oid)
        return int(x.numel() * x.element_size())

    if isinstance(x, dict):
        seen.add(oid)
        return sum(_encoded_nbytes(k, seen) + _encoded_nbytes(v, seen) for k, v in x.items())

    if isinstance(x, (list, tuple)):
        seen.add(oid)
        return sum(_encoded_nbytes(v, seen) for v in x)

    if isinstance(x, (bytes, bytearray)):
        return len(x)

    if isinstance(x, str):
        return len(x.encode("utf-8"))

    return sys.getsizeof(x)


_KEY_ATTRS = ("key", "keys", "weight_key", "comp_key", "init_key", "trans_key", "state_key")


def _is_key_value(x: Any) -> bool:
    """Return True for scalar key values used by accumulator key_merge methods."""
    if x is None:
        return False
    if isinstance(x, (str, int, float, bytes)):
        return True
    return False


def _freeze_for_signature(x: Any) -> Any:
    """Convert common mutable/numpy values into a hashable compatibility shape."""
    if isinstance(x, np.ndarray):
        return ("ndarray", tuple(x.shape), str(x.dtype))
    if isinstance(x, dict):
        return ("dict", tuple(sorted((repr(k), _freeze_for_signature(v)) for k, v in x.items())))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(_freeze_for_signature(v) for v in x))
    if isinstance(x, set):
        return ("set", tuple(sorted(repr(v) for v in x)))
    if isinstance(x, (str, int, float, bool, type(None))):
        return (type(x).__name__, repr(x))
    if hasattr(x, "__dict__"):
        return _object_signature(x)
    return (type(x).__module__, type(x).__qualname__, repr(x))


def _object_signature(x: Any) -> Any:
    """Best-effort structural signature for estimator compatibility checks."""
    values = []
    for name, value in sorted(vars(x).items()):
        if name in ("name", "key", "keys", "weight_key", "comp_key", "init_key", "trans_key", "state_key"):
            continue
        if name.startswith("_"):
            continue
        values.append((name, _freeze_for_signature(value)))
    return (type(x).__module__, type(x).__qualname__, tuple(values))


def _accumulator_signature(accumulator: StatisticAccumulator, role: str) -> Any:
    try:
        value_sig = _freeze_for_signature(accumulator.value())
    except Exception as err:  # noqa: BLE001
        value_sig = ("value-error", type(err).__name__, str(err))
    return (type(accumulator).__module__, type(accumulator).__qualname__, role, value_sig)


def _register_key(registry: dict[Any, tuple[Any, str]], key: Any, signature: Any, path: str) -> None:
    old = registry.get(key)
    if old is None:
        registry[key] = (signature, path)
        return
    old_signature, old_path = old
    if old_signature != signature:
        raise KeyValidationError(
            "Incompatible keyed sufficient-statistic sites for key %r: %s has %r, "
            "but %s has %r." % (key, old_path, old_signature, path, signature)
        )


def _iter_children(x: Any) -> list[Any]:
    if isinstance(x, dict):
        return list(x.values())
    if isinstance(x, (list, tuple)):
        return list(x)
    return []


def _collect_estimator_keys(
    estimator: ParameterEstimator, registry: dict[Any, tuple[Any, str]], path: str, visited: set[int]
) -> None:
    obj_id = id(estimator)
    if obj_id in visited:
        return
    visited.add(obj_id)

    estimator_sig = _object_signature(estimator)
    for attr in _KEY_ATTRS:
        if not hasattr(estimator, attr):
            continue
        keys = getattr(estimator, attr)
        if _is_key_value(keys):
            _register_key(
                registry,
                keys,
                (type(estimator).__module__, type(estimator).__qualname__, attr, estimator_sig),
                "%s.%s" % (path, attr),
            )
        elif isinstance(keys, (list, tuple)):
            for i, key in enumerate(keys):
                if _is_key_value(key):
                    _register_key(
                        registry,
                        key,
                        (type(estimator).__module__, type(estimator).__qualname__, "%s[%d]" % (attr, i), estimator_sig),
                        "%s.%s[%d]" % (path, attr, i),
                    )

    for name, value in sorted(vars(estimator).items()):
        for i, child in enumerate(_iter_children(value)):
            if isinstance(child, ParameterEstimator):
                _collect_estimator_keys(child, registry, "%s.%s[%d]" % (path, name, i), visited)
        if isinstance(value, ParameterEstimator):
            _collect_estimator_keys(value, registry, "%s.%s" % (path, name), visited)


def _collect_accumulator_keys(
    accumulator: StatisticAccumulator, registry: dict[Any, tuple[Any, str]], path: str, visited: set[int]
) -> None:
    obj_id = id(accumulator)
    if obj_id in visited:
        return
    visited.add(obj_id)

    for attr in _KEY_ATTRS:
        if not hasattr(accumulator, attr):
            continue
        key = getattr(accumulator, attr)
        if _is_key_value(key):
            _register_key(registry, key, _accumulator_signature(accumulator, attr), "%s.%s" % (path, attr))

    for name, value in sorted(vars(accumulator).items()):
        if isinstance(value, StatisticAccumulator):
            _collect_accumulator_keys(value, registry, "%s.%s" % (path, name), visited)
        else:
            for i, child in enumerate(_iter_children(value)):
                if isinstance(child, StatisticAccumulator):
                    _collect_accumulator_keys(child, registry, "%s.%s[%d]" % (path, name, i), visited)


def validate_estimator_keys(estimator: ParameterEstimator) -> None:
    """Validate keyed estimator and accumulator sites before EM folds stats.

    The validator catches the classic keying footgun: two different families, or
    two sites with incompatible estimator settings, accidentally sharing the same
    key string.  Validation is intentionally protocol-level and best-effort; a
    family can still perform stricter checks in its own factory if needed.
    """
    estimator_registry: dict[Any, tuple[Any, str]] = {}
    _collect_estimator_keys(estimator, estimator_registry, type(estimator).__name__, set())

    accumulator_registry: dict[Any, tuple[Any, str]] = {}
    accumulator = estimator.accumulator_factory().make()
    _collect_accumulator_keys(accumulator, accumulator_registry, type(accumulator).__name__, set())


def validate_accumulator_keys(accumulator: StatisticAccumulator) -> None:
    """Validate keyed sites in an already-created accumulator tree."""
    accumulator_registry: dict[Any, tuple[Any, str]] = {}
    _collect_accumulator_keys(accumulator, accumulator_registry, type(accumulator).__name__, set())


def merge_accumulator_keys(accumulator: StatisticAccumulator) -> None:
    """Pool keyed statistics across ``accumulator``'s tree -- the parameter-tying pass.

    Runs the ``key_merge``/``key_replace`` pair every EM driver applies exactly once after
    accumulation (see :func:`mixle.stats.compute.sequence.seq_estimate`), so sites sharing a key
    estimate from the pooled statistics.  A no-op when no site in the tree carries a key.  Call it
    on the fully-combined accumulator, never per shard: pooling twice would double-count the
    shared statistics on the second ``combine``.
    """
    stats_dict: dict[Any, Any] = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
