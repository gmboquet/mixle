"""Categorical distributions over arbitrary hashable labels.

Data type: Any. The data type is taken as the categorical object and a probability is estimated.

If Data type is int, consider using mixle.stats.univariate.discrete.integer_categorical (IntegerCategoricalDistribution) instead.

Reference: Johnson, Kemp & Kotz, *Univariate Discrete Distributions* (3rd ed., Wiley, 2005).
"""

import math
from collections.abc import Sequence
from typing import Any, Optional, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.algorithms import QuantizedCrossIndex, QuantizedEnumerationIndex
from mixle.inference.fisher import FixedFisherView
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
from mixle.stats.univariate.discrete._count_contracts import nonnegative_weights
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.special import digamma

T = TypeVar("T")


def _is_normalized_support(pmap: dict[Any, float], default_value: float) -> bool:
    """Return whether ``(pmap, default_value)`` describe a proper finite probability law.

    Derived from the parameters themselves so it can never disagree with them -- in particular
    it is recomputed on deserialization rather than trusted from the persisted state, since a
    stale ``is_normalized_probability=True`` on an unnormalized map is exactly what lets a
    non-summing-to-1 probability vector reach ``RandomState.choice`` and crash there.
    """
    if not pmap or default_value != 0.0:
        return False
    return bool(
        np.isclose(float(np.sum(np.asarray(list(pmap.values()), dtype=np.float64))), 1.0, rtol=1.0e-12, atol=1.0e-12)
    )


class CategoricalFisherView(FixedFisherView):
    """Fisher view for categorical one-hot sufficient statistics."""

    def __init__(self, dist: Any, keys: Sequence[Any], probs: Sequence[float]) -> None:
        self.keys = list(keys)
        self.key_index: dict[Any, int] = {k: i for i, k in enumerate(self.keys)}
        p = np.asarray(probs, dtype=np.float64)
        total = p.sum()
        self.probs = p / total if total > 0.0 else np.ones(len(self.keys), dtype=np.float64) / max(len(self.keys), 1)
        super().__init__(dist, [(repr(k),) for k in self.keys])

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        mat = np.zeros((len(data), len(self.keys)), dtype=np.float64)
        for i, x in enumerate(data):
            j = self.key_index.get(x)
            if j is not None:
                mat[i, j] = 1.0
        return mat

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        xs, values = enc_data
        value_to_col = np.asarray([self.key_index.get(v, -1) for v in values], dtype=np.int64)
        cols = value_to_col[np.asarray(xs, dtype=np.int64)]
        mat = np.zeros((len(cols), len(self.keys)), dtype=np.float64)
        rows = np.arange(len(cols), dtype=np.int64)
        good = cols >= 0
        mat[rows[good], cols[good]] = 1.0
        return mat

    def _model_mean(self) -> np.ndarray:
        return self.probs.copy()

    def _model_fisher(self) -> np.ndarray:
        return np.diag(self.probs) - np.outer(self.probs, self.probs)


class CategoricalDistribution(SequenceEncodableProbabilityDistribution):
    """Categorical distribution over hashable labels."""

    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for categorical scoring."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the symbolic declaration for categorical probability maps."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="categorical",
            distribution_type=cls,
            parameters=(ParameterSpec("pmap", constraint="simplex_map"),),
            statistics=(StatisticSpec("count_map", kind="count_map"),),
            support="finite_or_default_hashable",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                sufficient_statistics_from_params=cls.exp_family_sufficient_statistics_from_params,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure_from_params=cls.exp_family_base_measure_from_params,
                # T(x) is the one-hot label indicator (categories in canonical sorted-by-repr key
                # order) and eta = log(pmap); A = 0, h(x) = 0 on the support (the keys of pmap). The
                # category set depends on the per-instance pmap, so fixed_base=False. eta has -inf
                # entries when a label has p = 0, which makes the generic <eta, T> dot form NaN via
                # 0*-inf for OTHER labels; runtime_scoring is therefore False so scoring keeps the
                # safe dict-indexing backend path while to_exponential_family still exposes the
                # canonical map (valid for the plain default_value=0 categorical, where p > 0).
                fixed_base=False,
                runtime_scoring=False,
            ),
        )

    @staticmethod
    def _ef_pmap(params: dict[str, Any]) -> dict[Any, float]:
        """Recover the ``pmap`` dict (parameter packing wraps it in a 0-d object array)."""
        pmap = params["pmap"]
        return pmap.item() if isinstance(pmap, np.ndarray) else pmap

    @staticmethod
    def _ef_labels(x: Any) -> np.ndarray:
        """Recover the observed labels from the exp-family input (encoded ``(idx, value_map)`` or raw)."""
        if isinstance(x, tuple) and len(x) == 2:
            idx = np.asarray(x[0]).reshape(-1).astype(np.int64)
            value_map = np.asarray(x[1], dtype=object).reshape(-1)
            return value_map[idx]
        return np.atleast_1d(np.asarray(list(x), dtype=object))

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return a shape-only fallback; category-aware statistics come from ``..._from_params``."""
        n = len(CategoricalDistribution._ef_labels(x))
        return (engine.asarray(np.zeros(n, dtype=np.float64)),)

    @staticmethod
    def exp_family_sufficient_statistics_from_params(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the one-hot label indicator ``T(x)`` of shape ``(n, K)`` (zeros for off-support labels).

        Categories are ordered canonically by ``sorted(pmap, key=repr)`` so the columns line up with
        :meth:`exp_family_natural_parameters`.
        """
        labels = CategoricalDistribution._ef_labels(x)
        keys = sorted(CategoricalDistribution._ef_pmap(params).keys(), key=repr)
        index = {key: i for i, key in enumerate(keys)}
        onehot = np.zeros((len(labels), len(keys)), dtype=np.float64)
        for row, label in enumerate(labels):
            col = index.get(label)
            if col is not None:
                onehot[row, col] = 1.0
        return (engine.asarray(onehot),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the natural parameter ``eta = log(pmap)`` over categories in canonical key order."""
        pmap = CategoricalDistribution._ef_pmap(params)
        keys = sorted(pmap.keys(), key=repr)
        probs = np.asarray([pmap[key] for key in keys], dtype=np.float64)
        return (engine.log(engine.asarray(probs)),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the log partition ``A = 0`` (normalization is carried by ``eta = log p``)."""
        return engine.asarray(0.0)

    @staticmethod
    def exp_family_base_measure_from_params(x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return ``log h(x) = 0`` on the support (a key of ``pmap``) and ``-inf`` for off-support labels."""
        labels = CategoricalDistribution._ef_labels(x)
        keys = set(CategoricalDistribution._ef_pmap(params).keys())
        h = np.array([0.0 if label in keys else -np.inf for label in labels], dtype=np.float64)
        return engine.asarray(h)

    def __init__(
        self,
        pmap: dict[Any, float] = MISSING,
        default_value: float = 0.0,
        name: str | None = None,
        keys: str | None = None,
        prob_map: dict[Any, float] = MISSING,
        prior: Optional["SequenceEncodableProbabilityDistribution"] = None,
        scoring_only: bool = False,
    ) -> None:
        """Create a categorical distribution over an explicit support map.

        Labels in ``pmap`` receive their configured probabilities. Labels outside
        the support receive ``default_value``; the default of ``0.0`` gives finite
        support and ``-inf`` log-density for unknown labels.

        Args:
            pmap: Mapping from labels to probabilities. Values must be finite and non-negative.
                ``density()``/``log_density()`` assume ``sum(pmap.values()) == 1`` for the
                result to be a properly-normalized probability (labels with p = 0 may be
                included or omitted interchangeably), but the constructor does not check or
                enforce that sum: several existing call sites intentionally build a pmap with
                an arbitrary, non-1-summing total -- e.g. as a cheap fixture for exercising
                dispatch logic that never inspects the values, or to make ``default_value``
                dominate every in-pmap probability by construction. An empty ``pmap`` is
                likewise legal: it is the (vacuous) empty-support law, which support-enumeration
                code constructs directly. Pass a normalized map when you need ``density()`` to
                be meaningful; read :attr:`is_normalized_probability` to find out whether a
                given instance actually is one.
            default_value: Probability assigned to labels outside ``pmap``. Must be finite and
                within ``[0, 1]``. When ``default_value > 0`` and ``pmap`` does not cover the
                full label space, ``density()``/``log_density()`` become a smoothed *scoring*
                function rather than a normalized probability measure: every distinct label
                outside ``pmap`` receives the same flat fallback mass, so querying several
                distinct out-of-vocabulary labels can each score positive density, and those
                densities can sum to more than 1 over the true (potentially unbounded) label
                space. This is inherent to using a single fallback constant for "everything
                else" -- the same tradeoff a Laplace/Kneser-Ney-smoothed language model makes --
                since the true unknown-label cardinality is unknowable in general, so there is
                no way to normalize it properly, and this class makes no attempt to.
                :class:`CategoricalSampler` reflects the same boundary from the sampling side:
                it can only draw labels registered in ``pmap``, never a synthesized
                "default"/unknown label, since there is no way to generatively sample from an
                unspecified label space.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prob_map: Alias for ``pmap``.
            prior (Optional): Conjugate parameter prior over the category-probability simplex. A
                :class:`~mixle.stats.bayes.dict_dirichlet.DictDirichletDistribution` enables the Bayesian /
                variational machinery (``expected_log_density`` and the conjugate posterior update);
                ``None`` (default) is a plain point model.
            scoring_only: Opt-in declaration that this object is an evaluation-only likelihood
                factor rather than a generative law. It is never *required* (see ``pmap``
                above), but declaring it makes the object report
                :attr:`~mixle.stats.compute.pdist.DensitySemantics.LIKELIHOOD_FACTOR`, which
                keeps it out of generative machinery: mixtures refuse it as a component and
                :class:`CategoricalSampler` refuses to sample it.

        Attributes:
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            pmap: Owned copy of the configured label weights (not renormalized or sum-checked
                by the constructor).
            default_value: Probability assigned to labels outside ``pmap``.
            scoring_only: Whether this object was explicitly declared evaluation-only.
            is_normalized_probability: Derived, never asserted: ``True`` only when ``pmap`` is
                non-empty, sums to 1, ``default_value`` is 0, and ``scoring_only`` is false.
            no_default: ``True`` when outside-support labels have nonzero mass.
            log_default_value: Log of ``default_value``.
            log1p_default_value: Log normalizer for ``1 + default_value``.

        Raises:
            ValueError: If ``pmap`` contains a non-finite or negative value, or if
                ``default_value`` is non-finite or outside ``[0, 1]``.
            TypeError: If ``scoring_only`` is not a boolean.

        """
        pmap = coalesce_alias("pmap", pmap, "prob_map", prob_map, default=MISSING)
        probs = np.asarray(list(pmap.values()), dtype=np.float64)
        if not np.isfinite(probs).all() or np.any(probs < 0.0):
            # A NaN or negative "probability" silently propagates into density()/log_density()
            # answers (and a NaN surfaces again, far from the mistake, as an opaque error out of
            # the sampler's RandomState.choice), so reject it at the constructor like the scalar
            # families do.
            #
            # Deliberately NOT also enforced here: sum(pmap.values()) == 1, a non-empty pmap, or
            # default_value == 0. density()'s docstring assumes that sum, but several existing,
            # intentional call sites construct a pmap that doesn't hold it -- e.g.
            # mixle/tests/quantized_index_test.py and fused_em_mixtures_test.py build a partial
            # pmap (such as ``{2: 0.5}``) purely as a cheap object to call
            # .estimator()/.quantized_index() on, never inspecting the values;
            # mixle/tests/sparse_mixture_test.py builds pmap={"a": 0.005, "b": 0.005} with
            # default_value=0.99 specifically so default_value dominates every in-pmap
            # probability, which a sum-to-1 rejection (or a silent renormalization, which would
            # rescale "a"/"b" up and defeat that domination) would break; and support-enumeration
            # code (mixle/tests/truncation_bound_test.py) constructs the empty-support
            # CategoricalDistribution({}) to assert that a zero-item support is vacuously
            # exhausted. A finite/non-negative check has no such legitimate counterexample: no
            # call site anywhere in the codebase relies on constructing a NaN or negative
            # "probability".
            #
            # This was briefly enforced (commit 7bdc3646) and had to be reverted: it broke every
            # one of those call sites, including unrelated truncation-bound code. If normalization
            # matters to a consumer, read the derived `is_normalized_probability` flag below
            # rather than re-adding a constructor-time rejection here.
            raise ValueError("CategoricalDistribution requires finite non-negative probabilities.")
        if not np.isfinite(default_value) or default_value < 0.0 or default_value > 1.0:
            # default_value used to be silently clamped into [0, 1] for self.default_value while
            # log_default_value/log1p_default_value were computed from the raw, unclamped input --
            # so an out-of-range value desynced the two: density() (which uses self.default_value)
            # and log_density() (which uses log1p_default_value) would disagree, and a NaN
            # default_value made log_density() return nan for every in-pmap label. Reject it here
            # instead, like pmap above.
            raise ValueError("CategoricalDistribution requires default_value in [0, 1], not %s." % repr(default_value))
        if not isinstance(scoring_only, (bool, np.bool_)):
            raise TypeError("scoring_only must be a boolean.")
        self.name = name
        self.keys = keys
        self.pmap = dict(pmap)
        self.scoring_only = bool(scoring_only)
        # Descriptive, not prescriptive: report whether this instance happens to be a proper
        # finite law so consumers that genuinely need one (packers, exact enumeration) can check
        # instead of assuming. Constructing a non-normalized instance stays legal -- see the
        # comment above the finite/non-negative check.
        self.is_normalized_probability = _is_normalized_support(self.pmap, default_value) and not self.scoring_only
        self.no_default = default_value != 0.0
        self.default_value = default_value
        self.log_default_value = float(-np.inf if default_value == 0 else math.log(default_value))
        self.log1p_default_value = float(math.log1p(default_value))
        self.set_prior(prior)

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Return JSON-serializable state with ``pmap`` as an order-preserving list of pairs.

        ``pmap``'s insertion order is load-bearing: :class:`CategoricalSampler` turns it into
        parallel ``levels``/``probs`` arrays and draws a positional index from them, so a fixed
        seed maps to a specific *sequence position*, not a category label. mixle's generic dict
        encoding (:func:`mixle.utils.serialization._encode_dict`) canonicalizes key order for
        diffable JSON output, which would silently reorder ``pmap`` on decode and change which
        category a fixed-seed sampler draws -- same seed, same per-category probabilities, but a
        different sampled sequence after a save/load round trip. Encoding ``pmap`` as a plain list
        of ``(key, value)`` pairs instead of a dict routes it through the list/tuple encoders,
        which never reorder, so category order survives regardless of any dict-key-sorting
        elsewhere in the serialization path.
        """
        state = dict(self.__dict__)
        state["pmap"] = list(self.pmap.items())
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Restore state, rebuilding ``pmap`` from its order-preserving pair list.

        Accepts a plain dict for ``pmap`` too, so pre-fix artifacts (serialized before this
        method existed, with ``pmap`` as a generic -- and therefore key-sorted -- dict) still
        decode without error; their original category order was already lost at write time and
        cannot be recovered, but new writes now round-trip order-faithfully.
        """
        state = dict(state)
        pmap = state["pmap"]
        state["pmap"] = pmap if isinstance(pmap, dict) else dict(pmap)
        # ``scoring_only`` is an author declaration that cannot be re-derived from the numbers (a
        # perfectly normalized map may still be declared a likelihood factor), so carry the
        # persisted value forward -- defaulting to False for artifacts written before the flag
        # existed. ``is_normalized_probability`` is the opposite: it is a *derived* property, so
        # recompute it from the restored pmap instead of believing whatever the artifact claims.
        # These were previously `setdefault` calls, which were silent no-ops because the persisted
        # state already carries both keys -- so an unnormalized distribution round-tripped still
        # asserting `is_normalized_probability=True`, handing the sampler a probability vector
        # that does not sum to 1 and reopening the RandomState.choice crash that was fixed there.
        state["scoring_only"] = bool(state.get("scoring_only", False))
        state["is_normalized_probability"] = (
            _is_normalized_support(state["pmap"], state.get("default_value", 0.0)) and not state["scoring_only"]
        )
        self.__dict__.update(state)

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        s1 = ", ".join(
            ["%s: %s" % (repr(k), repr(v)) for k, v in sorted(self.pmap.items(), key=lambda item: repr(item[0]))]
        )
        s2 = repr(self.default_value)
        s3 = repr(self.name)
        s4 = repr(self.keys)

        scoring = ", scoring_only=True" if self.scoring_only else ""
        return "CategoricalDistribution({%s}, default_value=%s, name=%s, keys=%s%s)" % (
            s1,
            s2,
            s3,
            s4,
            scoring,
        )

    def get_prior(self) -> Optional["SequenceEncodableProbabilityDistribution"]:
        """Return the conjugate parameter prior over the category-probability simplex (or None)."""
        return self.prior

    def set_prior(self, prior: Optional["SequenceEncodableProbabilityDistribution"]) -> None:
        """Attach a parameter prior and precompute conjugate-prior expectations.

        With a DictDirichlet(alpha) prior over the category probabilities this caches the variational
        expected log-probabilities E[log p_k] = digamma(alpha_k) - digamma(sum_k alpha_k) for each key
        of ``pmap`` so that ``expected_log_density(x) = E[log p_x] - log(1 + default_value)``. A scalar
        alpha is treated as a symmetric Dirichlet of dimension ``len(pmap)``. Any other prior
        (including ``None``) leaves the distribution a plain point model.
        """
        from mixle.stats.bayes.dict_dirichlet import DictDirichletDistribution

        self.prior = prior
        if isinstance(prior, DictDirichletDistribution):
            a = prior.get_parameters()
            n = len(self.pmap)
            if isinstance(a, float):
                bb = digamma(a) - digamma(n * a)
                b = {k: bb for k in self.pmap.keys()}
            else:
                asum = digamma(sum(a.values()))
                b = {k: digamma(v) - asum for k, v in a.items()}
            self.conj_prior_params = a
            self.expected_nparams = b
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def expected_log_density(self, x: Any) -> float:
        """Variational expectation E_q[log p(x)] under the DictDirichlet prior.

        Falls back to the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if not self.has_conj_prior:
            return self.log_density(x)

        if x not in self.pmap:
            return self.log_default_value - self.log1p_default_value

        return self.expected_nparams[x] - self.log1p_default_value

    def seq_expected_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if not self.has_conj_prior:
            return self.seq_log_density(x)

        xs, val_map_inv = x
        rv = np.asarray([self.expected_log_density(u) for u in val_map_inv])
        return rv[xs]

    def density(self, x: Any) -> float:
        """Density evaluation of CategoricalDistribution.

        p_mat(x) = p_i, if x in pmap.keys(), else p_mat(x) = default_value.

        With ``default_value > 0`` and a ``pmap`` that does not cover the full label space, this
        is a smoothed scoring function, not a normalized probability measure -- see the
        ``default_value`` note on :meth:`__init__`.

        Args:
            x (Any): Evaluate CategoricalDistribution density value at x.

        Returns:
            float density value at x

        """
        return self.pmap.get(x, self.default_value) / (1.0 + self.default_value)

    def density_semantics(self):
        """Report what ``log_density`` actually returns, derived from the parameters.

        This keys off :attr:`is_normalized_probability`, so an object whose numbers do not describe a
        law does not claim to be one (MXR-080-1841). Three shapes report ``LIKELIHOOD_FACTOR``:

        * ``scoring_only``, the author's declaration that this is an evaluation-only factor,
        * ``default_value != 0`` open-world smoothing, which spreads mass over every unseen label and
          so has no finite total,
        * a pmap that does not sum to one -- including the empty pmap, whose total is zero.

        An earlier revision returned ``EXACT`` for the last two, because ``_owned_generative_components``
        refuses ``LIKELIHOOD_FACTOR`` and mixtures, HMMs and segment models in this repository are all
        built from categoricals of exactly those shapes: a component that won zero responsibility this
        iteration, the explicit no-evidence state fitted on all-empty sequences, and deliberately
        unnormalized components. Labelling them honestly made them non-composable, so the label was
        bent instead. That was the wrong half to bend: composition inconvenience cannot turn a non-law
        into an exact law.

        The composition layer now decides admissibility for itself, by asking a factor whether it
        differs from a law only by scale (see ``composable_as_component`` and
        ``_owned_generative_components``), so both questions get a truthful answer.
        """
        from mixle.stats.compute.pdist import DensitySemantics

        if not self.is_normalized_probability:
            return DensitySemantics.LIKELIHOOD_FACTOR
        return super().density_semantics()

    def composable_as_component(self) -> bool:
        """Whether a latent model may use this factor as a component (MXR-080-1841).

        A mixture, HMM or segment model normalizes responsibilities across components, so a component
        that differs from a law by a positive constant scale is absorbed into its own mixing weight
        and leaves the E-step unchanged; a zero-mass component simply wins no responsibility. A
        categorical is a lookup over a discrete support, so an unnormalized or empty pmap is exactly
        such a scaled law and stays admissible.

        ``scoring_only`` is not: it is an author's declaration that the object is not generative at
        all, and a declaration is not something the composition layer may reason its way around.
        """
        return not self.scoring_only

    def log_density(self, x: Any) -> float:
        """Log-Density evaluation of CategoricalDistribution.

        log(p_mat(x)) = log(p_i), if x in pmap.keys(), else log(p_mat(x)) = log(default_value).

        With ``default_value > 0`` and a ``pmap`` that does not cover the full label space, this
        is a smoothed scoring function, not a normalized probability measure -- see the
        ``default_value`` note on :meth:`__init__`.

        Args:
            x (Any): Evaluate CategoricalDistribution density value at x.

        Returns:
            Log-density of Categorical distribution evaluated at x.

        """
        p = self.pmap.get(x, self.default_value)
        return -np.inf if p <= 0.0 else np.log(p) - self.log1p_default_value

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of log-density for sequence encoded data.

        Input value x must be obtained from a call to CategoricalDataEncoder.seq_encode(data). Returns numpy array
        of log-density evaluated at all observations contained in encoded data x.

        Args:
            x: (Tuple[np.ndarray,np.ndarray]): Tuple of numpy indices for unique categories, and numpy array unique
                objects that index xs maps to.

        Returns:
            Numpy array of log-density evaluated at all observations contained in encoded data x.

        """
        with np.errstate(divide="ignore"):
            xs, val_map_inv = x
            mapped_log_prob = np.asarray([self.pmap.get(u, self.default_value) for u in val_map_inv], dtype=np.float64)
            np.log(mapped_log_prob, out=mapped_log_prob)
            mapped_log_prob -= self.log1p_default_value
            rv = mapped_log_prob[xs]

        return rv

    def backend_seq_log_density(self, x: tuple[np.ndarray, np.ndarray], engine: Any) -> Any:
        """Engine-neutral log-density for encoded object categories.

        The object-to-index lookup remains Python-side at the encoding boundary;
        the selected log-probability vector is an engine tensor, so simplex-map
        parameters can still participate in autograd.
        """
        xs, val_map_inv = x
        if hasattr(self, "_backend_labels"):
            label_to_idx = {label: i for i, label in enumerate(self._backend_labels)}
            default = getattr(self, "_backend_log_default", -np.inf)
            mapped = [label_to_idx.get(label, -1) for label in val_map_inv]
            mapped = np.asarray(mapped, dtype=np.int64)
            good = mapped >= 0
            safe = np.clip(mapped, 0, max(0, len(self._backend_labels) - 1))
            vals = self._backend_log_probs[engine.asarray(safe)]
            vals = engine.where(engine.asarray(good), vals, engine.asarray(default))
            return vals[engine.asarray(xs)]

        probs = np.asarray([self.pmap.get(u, self.default_value) for u in val_map_inv], dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_probs = np.log(probs) - self.log1p_default_value
        return engine.asarray(log_probs)[engine.asarray(xs)]

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from mixle.stats.compute.gradient import CategoricalGradientFitState

        labels = tuple(self.pmap.keys())
        probs = [self.pmap[label] for label in labels]
        logits = tensor_param(probs, engine, torch, transform="logits")
        leaves.append(logits)
        return CategoricalGradientFitState(self, labels, logits)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["CategoricalDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked categorical probabilities for shared finite supports."""
        labels = tuple(dists[0].pmap.keys())
        if any(tuple(d.pmap.keys()) != labels or d.default_value != dists[0].default_value for d in dists):
            raise ValueError("Stacked CategoricalDistribution components require shared support/defaults.")
        with np.errstate(divide="ignore"):
            log_p = np.asarray(
                [[np.log(d.pmap[label]) - d.log1p_default_value for d in dists] for label in labels], dtype=np.float64
            )
        return {
            "__pysp_component_axis__": {"log_p": 1},
            "labels": labels,
            "log_p": engine.asarray(log_p),
            "default": dists[0].log_default_value - dists[0].log1p_default_value,
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[np.ndarray, np.ndarray], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of categorical log densities."""
        xs, val_map_inv = x
        label_to_idx = {label: i for i, label in enumerate(params["labels"])}
        mapped = np.asarray([label_to_idx.get(label, -1) for label in val_map_inv], dtype=np.int64)
        good = mapped >= 0
        safe = np.clip(mapped, 0, max(0, len(params["labels"]) - 1))
        scores = params["log_p"][engine.asarray(safe), :]
        scores = engine.where(engine.asarray(good)[:, None], scores, engine.asarray(params["default"]))
        return scores[engine.asarray(xs), :]

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[np.ndarray, np.ndarray], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[dict[Any, float], ...]:
        """Return per-component legacy count maps from engine-resident posterior weights."""
        xs, val_map_inv = x
        xx = engine.asarray(xs)
        ww = engine.asarray(weights)
        count_rows = []
        for i in range(len(val_map_inv)):
            mask = xx == engine.asarray(i)
            count_rows.append(engine.sum(ww * mask[:, None], axis=0))
        if count_rows:
            counts = np.asarray(engine.to_numpy(engine.stack(count_rows, axis=0)), dtype=np.float64)
        else:
            counts = np.zeros((0, int(np.asarray(engine.to_numpy(engine.sum(ww, axis=0))).shape[0])), dtype=np.float64)
        return tuple(
            {val_map_inv[j]: float(counts[j, k]) for j in range(len(val_map_inv))} for k in range(counts.shape[1])
        )

    def support_size(self) -> int:
        """Number of categories in the support."""
        return len(self.pmap)

    def to_fisher(self, **kwargs):
        """Return the categorical's one-hot Fisher view (generic fallback for default-augmented maps)."""
        if hasattr(self, "pmap") and not getattr(self, "no_default", False):
            keys = sorted(self.pmap.keys(), key=repr)
            probs = [self.pmap[k] / (1.0 + getattr(self, "default_value", 0.0)) for k in keys]
            return CategoricalFisherView(self, keys, probs)
        if hasattr(self, "prob_map") and getattr(self, "default_value", 0.0) == 0.0:
            keys = sorted(self.prob_map.keys(), key=repr)
            probs = [self.prob_map[k] / (1.0 + getattr(self, "default_value", 0.0)) for k in keys]
            return CategoricalFisherView(self, keys, probs)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> "CategoricalSampler":
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional random seed.

        Returns:
            A configured ``CategoricalSampler``.

        """
        return CategoricalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "CategoricalEstimator":
        """Return an estimator initialized from this distribution's support.

        Args:
            pseudo_count: Optional smoothing count applied to current probabilities.

        Returns:
            A ``CategoricalEstimator``.
        """
        if pseudo_count is None:
            return CategoricalEstimator(name=self.name, keys=self.keys, prior=self.prior)

        else:
            return CategoricalEstimator(
                pseudo_count=pseudo_count, suff_stat=self.pmap, name=self.name, keys=self.keys, prior=self.prior
            )

    def dist_to_encoder(self) -> "CategoricalDataEncoder":
        """Return an encoder for categorical observations."""
        return CategoricalDataEncoder()

    def enumerator(self) -> "CategoricalEnumerator":
        """Return an enumerator over support labels in descending probability order."""
        return CategoricalEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index directly from the finite support map."""
        if self.no_default:
            raise EnumerationError(self, reason="non-zero default_value gives an unbounded support")
        items = [(k, math.log(v) - self.log1p_default_value) for k, v in self.pmap.items() if v > 0.0]
        return QuantizedEnumerationIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view for finite categorical maps."""
        dists = [self] + list(others)
        if any(not isinstance(dist, CategoricalDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)
        if any(dist.no_default for dist in dists):
            raise EnumerationError(self, reason="non-zero default_value gives an unbounded support")

        keys = set()
        for dist in dists:
            keys.update(dist.pmap.keys())
        items = []
        for key in keys:
            lps = []
            for dist in dists:
                p = dist.pmap.get(key, 0.0)
                lps.append(math.log(p) - dist.log1p_default_value if p > 0.0 else -np.inf)
            items.append((key, tuple(lps)))
        return QuantizedCrossIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view for two finite categorical maps."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class CategoricalSampler(DistributionSampler):
    """Sampler for categorical labels according to the configured probability map."""

    def __init__(self, dist: CategoricalDistribution, seed: int | None = None) -> None:
        """Create a sampler for a categorical distribution.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.

        Attributes:
            rng: Random state used for sampling.
            levels: Category labels.
            probs: Category probabilities in ``levels`` order, normalized to sum to 1.
            num_levels: Number of categories.

        Raises:
            ValueError: If ``dist`` was declared ``scoring_only`` (a likelihood factor is not a
                generative law), or if ``dist.pmap`` is empty or every one of its values is 0.0,
                since there is then no relative proportion of registered labels to sample from.

        """
        self.rng = RandomState(seed)
        # One rule covering every way this object can fail to be a law (MXR-080-1857). The checks
        # below caught scoring_only and an all-zero pmap; they did not catch the two shapes that
        # matter most, because both looked like ordinary data:
        #
        #   * a pmap whose total is not one -- {a: .25, b: .25} scores total mass .5, and the
        #     normalization further down silently rescaled it to 1, so the sampler drew from a
        #     DIFFERENT law than log_density scored;
        #   * a positive default_value -- every unseen label carries mass the sampler can never
        #     emit, because it draws only from the registered labels.
        #
        # is_normalized_probability is derived from the parameters and is exactly this condition, so
        # one test replaces three and cannot drift from what the numbers say.
        if not getattr(dist, "is_normalized_probability", True):
            raise ValueError(
                "CategoricalSampler requires a normalized probability law, but this distribution is "
                f"not one: pmap total {float(sum(dist.pmap.values())) if dist.pmap else 0.0:.6g}, "
                f"default_value {getattr(dist, 'default_value', 0.0)!r}, "
                f"scoring_only {getattr(dist, 'scoring_only', False)!r}. Sampling it would draw from "
                "a renormalized law that log_density does not score -- the scorer and the sampler "
                "would describe different distributions. Normalize the pmap (and set default_value=0) "
                "to obtain a generative categorical; log_density remains available either way."
            )
        if getattr(dist, "scoring_only", False):
            # An explicitly declared scoring-only object is a likelihood factor, not a normalized
            # generative law; there is nothing to draw from. This is the one construction the
            # sampler refuses outright -- a merely non-1-summing pmap is handled by normalizing
            # below, because that construction is legal by design (see CategoricalDistribution).
            raise ValueError(
                "CategoricalSampler requires a generative distribution, not a scoring_only "
                "likelihood factor, which is not a normalized probability law."
            )
        temp = list(dist.pmap.items())
        self.levels = [u[0] for u in temp]
        raw_probs = np.asarray([u[1] for u in temp], dtype=np.float64)
        total = raw_probs.sum()
        if total <= 0.0:
            # dist.pmap values are already guaranteed finite and non-negative by
            # CategoricalDistribution.__init__, so total <= 0.0 here only happens when pmap is
            # empty or every entry is exactly 0.0 -- there is no relative proportion left to
            # sample from, and np.random.RandomState.choice cannot sample from an all-zero-weight
            # distribution either. Fail clearly here, at sampler construction, instead of letting
            # a 0/0 division or an opaque numpy error surface later out of sample().
            what = "is empty" if not dist.pmap else "gives every category zero probability"
            raise ValueError(
                f"CategoricalSampler requires at least one category with positive probability, but pmap {what}. "
                "A categorical fitted on no evidence is this shape (MXR-080-1220): it reports "
                "density_semantics() == LIKELIHOOD_FACTOR and is_normalized_probability False rather than "
                "posing as a law, and there is nothing to draw from. Supply a prior or a declared support "
                "(CategoricalEstimator(suff_stat=...)) if a no-evidence fit must still be sampleable."
            )
        # CategoricalDistribution.pmap is deliberately allowed to not sum to 1 (see its
        # __init__ docstring; density()/log_density() already tolerate this by dividing by
        # (1 + default_value)), but np.random.RandomState.choice requires p to sum to exactly 1.
        # Normalizing by pmap's own sum makes sampling work from the relative proportions of
        # pmap's registered labels regardless, so a construction-time-legal, non-1-summing pmap
        # no longer crashes here. This does not change what can be sampled -- draws are still
        # only ever one of pmap's registered labels, never a synthesized "default"/unknown label
        # (there is no way to generatively sample from an unspecified label space).
        self.probs = (raw_probs / total).tolist()
        self.num_levels = len(self.levels)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any | list[Any]:
        """Draw iid samples from the categorical distribution.

        Args:
            size: Number of iid samples to draw. ``None`` returns a scalar label.

        Returns:
            A scalar label when ``size`` is ``None``; otherwise a list of labels.

        """
        if size is None:
            idx = self.rng.choice(self.num_levels, p=self.probs, size=size)
            return self.levels[idx]

        else:
            levels = self.levels
            rv = self.rng.choice(self.num_levels, p=self.probs, size=size)

            return [levels[i] for i in rv]


class CategoricalEnumerator(DistributionEnumerator):
    """Enumerator over finite categorical support in descending probability order."""

    def __init__(self, dist: CategoricalDistribution) -> None:
        """Enumerates the support of a CategoricalDistribution in descending probability order.

        Raises EnumerationError if the distribution has a non-zero default_value, since the
        support is then unbounded (every value outside pmap has positive probability).

        Args:
            dist (CategoricalDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        if dist.no_default:
            raise EnumerationError(dist, reason="non-zero default_value gives an unbounded support")
        entries = [(k, v) for k, v in dist.pmap.items() if v > 0.0]
        entries.sort(key=lambda u: -u[1])
        self._entries = entries
        self._pos = 0

    def __next__(self) -> tuple[Any, float]:
        if self._pos >= len(self._entries):
            raise StopIteration
        k, v = self._entries[self._pos]
        self._pos += 1
        return (k, math.log(v) - self.dist.log1p_default_value)


class CategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for weighted categorical label counts."""

    def __init__(self, keys: str | None = None) -> None:
        """Create an accumulator for categorical sufficient statistics.

        The sufficient statistic is ``count_map``, a mapping from category label
        to weighted count.

        Args:
            keys: Optional key for merging sufficient statistics.

        Attributes:
            count_map: Weighted counts by category label.
            keys: Optional sufficient-statistic key.

        """
        self.count_map = dict()
        self.keys = keys
        # Every label this accumulator was shown, including ones that earned zero weight. ``count_map``
        # holds positive counts only, so a component that saw the data but won no responsibility used
        # to report an EMPTY statistic -- and an empty statistic estimates to a categorical with no
        # support, which scores -inf everywhere and can therefore never win responsibility again
        # (MXR-080-1220). Keeping the support lets that estimate be a uniform instead: generative,
        # and recoverable on the next E-step, which is what the dead-component case actually needs.
        #
        # A dict, not a set: ``pmap`` insertion order is load-bearing (CategoricalSampler draws a
        # positional index from it), and a set's iteration order varies with PYTHONHASHSEED.
        self._support: dict[Any, None] = {}

    def update(self, x: Any, weight: float, estimate: Optional["CategoricalDistribution"]) -> None:
        """Adds weight to the category_count for category x.

        If x is new Category label, a new key in the dict count_map is created and the count is incremented by weight.

        Args:
            x (Any): Category label.
            weight (float): Weight for the observation x.
            estimate (Optional['CategoricalDistribution']): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None, updates sufficient_stat of Accumulator, count_map.

        """
        checked = nonnegative_weights([weight], shape=(1,))
        if x in getattr(estimate, "pmap", ()):  # a label this component models, see seq_update
            self._support.setdefault(x, None)
        if checked[0] == 0.0:
            return
        self.count_map[x] = self.count_map.get(x, 0.0) + weight

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initializes the CategoricalAccumulator sufficient statistics one observation at a time.

        This delegates to :meth:`update`, since initialization has no random component.

        Args:
            x (Any): Category label.
            weight (float): Weight incrementing suff stat count_map counts for the observation x.
            rng (Optional[RandomState]): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None, initializes sufficient_stat of Accumulator, count_map.

        """
        self.update(x, weight, None)

    def get_seq_lambda(self):
        """Return the sequence-update kernel used by generated accumulation code."""
        return [self.seq_update]

    def seq_update(
        self, x: (tuple[np.ndarray, np.ndarray]), weights: np.ndarray, estimate: Optional["CategoricalDistribution"]
    ) -> None:
        """Vectorized accumulation of Categorical sufficient statistics from encoded sequence of data.

        Requires data as encoded sequence from CategoricalDataEncoder.seq_encode(data).

        Args:
            x (Tuple[np.ndarray,np.ndarray]): Tuple of numpy indices for unique categories, and numpy array unique
                objects that index xs maps to.
            weights (np.ndarray): weights for each observation in encoded data set.
            estimate (Optional['CategoricalDistribution']): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        indices = np.asarray(x[0], dtype=np.int64)
        inv_key_map = np.asarray(x[1], dtype=object)
        checked = nonnegative_weights(weights, shape=indices.shape)
        bcnt = np.bincount(indices, weights=checked, minlength=len(inv_key_map))
        modelled = getattr(estimate, "pmap", None)
        for index, count in enumerate(bcnt):
            key = inv_key_map[index]
            # Record the support only for a label the component actually MODELS. A label it assigns
            # no probability is not part of its support, and resetting a dead component to a uniform
            # over labels it never modelled would invent a law from evidence it could not explain --
            # which is what "impossible evidence has zero responsibility" means.
            if modelled is not None and key in modelled:
                self._support.setdefault(key, None)
            if count > 0.0:
                self.count_map[key] = self.count_map.get(key, 0.0) + count

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of Categorical sufficient statistics from encoded sequence of data.

        Requires data as encoded sequence from CategoricalDataEncoder.seq_encode(data).
        This delegates to :meth:`seq_update`, since initialization has no random component.

        Args:
            x (Tuple[np.ndarray,np.ndarray]): Tuple of numpy indices for unique categories, and numpy array unique
                objects that index xs maps to.
            weights (np.ndarray): weights for each observation in encoded data set.
            rng (Optional[RandomState]): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        return self.seq_update(x, weights, None)

    def combine(self, suff_stat: dict[Any, float]) -> "CategoricalAccumulator":
        """Combine the sufficient statistics of CategoricalAccumulator with suff_stat.

        Args:
            suff_stat (Dict[Any, float]): Prior data observations aggregated into dictionary with category levels
                as keys and counts as values.

        Returns:
            None, updates the count_map of CategoricalAccumulator.

        """
        for k, v in suff_stat.items():
            self._support.setdefault(k, None)
            if v == 0.0:
                # A zero entry carries support, not evidence. Adding it to count_map would enlarge
                # len(suff_stat) and so change how a pseudo_count is spread; keeping count_map to
                # positive counts leaves every ordinary fit byte-identical.
                continue
            self.count_map[k] = self.count_map.get(k, 0.0) + v

        return self

    def value(self) -> dict[Any, float]:
        """Returns sufficient statistic of CategoricalAccumulator.

        Sufficient statistic value is a dictionary with category as keys and counts of categories as values.

        Returns:
            Dict[Any, float] of sufficient statistic.

        """
        if self.count_map:
            return self.count_map.copy()
        # No positive counts, but labels were seen: report them at zero rather than reporting
        # nothing. estimate() turns a zero-count support into a uniform (its nobs_loc == 0 branch),
        # which is a generative law; an empty statistic estimated to a supportless object whose
        # sampler raised (MXR-080-1220).
        return dict.fromkeys(self._support, 0.0)

    def from_value(self, x: dict[Any, float]) -> "CategoricalAccumulator":
        """Set CategoricalAccumulator sufficient statistics and member variables from suff_stat dict defined in value().

        Takes sufficient statistic value from dictionary with category as keys and counts of categories as values. Sets
        count_map to the passed value x.

        Args:
            x (Dict[Any, float]): Dictionary with category as keys and counts of categories as values

        Returns:
            CategoricalAccumulator with member variable sufficient statistics set to x.

        """
        self.count_map = x
        for key in x:
            self._support.setdefault(key, None)

        return self

    def acc_to_encoder(self) -> "CategoricalDataEncoder":
        """Return an encoder compatible with categorical observations."""
        return CategoricalDataEncoder()


class CategoricalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for categorical count accumulators."""

    def __init__(self, keys: str | None = None) -> None:
        """Create an accumulator factory.

        Args:
            keys: Optional key for merging sufficient statistics.

        """
        self.keys = keys

    def make(self) -> "CategoricalAccumulator":
        """Return a fresh categorical accumulator."""
        return CategoricalAccumulator(keys=self.keys)


class CategoricalEstimator(ParameterEstimator):
    """Estimator for categorical probability maps from weighted label counts."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: dict[Any, float] | None = None,
        default_value: bool = False,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create an estimator for categorical sufficient statistics.

        Args:
            pseudo_count: Optional smoothing count applied to existing probabilities.
            suff_stat: Optional prior probability map used with ``pseudo_count``.
            default_value: Whether to estimate an outside-support default probability.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prior: Optional conjugate prior over the probability map.

        Attributes:
            pseudo_count: Smoothing count.
            suff_stat: Prior probability map.
            default_value: Whether to estimate an outside-support default probability.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic key.

        """
        if pseudo_count is not None and (
            isinstance(pseudo_count, (bool, np.bool_)) or not np.isfinite(pseudo_count) or pseudo_count < 0.0
        ):
            raise ValueError("categorical pseudo_count must be finite and non-negative.")
        if suff_stat is not None and (
            not suff_stat
            or any(not np.isfinite(value) or value < 0.0 for value in suff_stat.values())
            or sum(suff_stat.values()) <= 0.0
        ):
            raise ValueError("categorical prior statistics must be a non-empty map of finite non-negative values.")
        # A supplied prior count map without an explicit multiplier is one unit of prior
        # evidence. This makes the configuration meaningful instead of multiplying by None.
        self.pseudo_count = 1.0 if suff_stat is not None and pseudo_count is None else pseudo_count
        self.suff_stat = None if suff_stat is None else dict(suff_stat)
        self.default_value = default_value
        self.name = name
        self.keys = keys
        self.prior = prior
        from mixle.stats.bayes.dict_dirichlet import DictDirichletDistribution

        self.has_conj_prior = isinstance(prior, DictDirichletDistribution)

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the conjugate parameter prior over the category-probability simplex (or None)."""
        return self.prior

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Set the conjugate parameter prior over the category-probability simplex."""
        from mixle.stats.bayes.dict_dirichlet import DictDirichletDistribution

        self.prior = prior
        self.has_conj_prior = isinstance(prior, DictDirichletDistribution)

    def model_log_density(self, model: "CategoricalDistribution") -> float:
        """Log-density of the model probability map under the DictDirichlet prior (ELBO global term)."""
        if self.has_conj_prior:
            return float(self.prior.log_density(model.pmap))
        return 0.0

    def accumulator_factory(self) -> "CategoricalAccumulatorFactory":
        """Return an accumulator factory matching this estimator."""
        return CategoricalAccumulatorFactory(self.keys)

    def _estimate_conjugate(self, suff_stat: dict[Any, float]) -> "CategoricalDistribution":
        """Dirichlet MAP estimate (counts + alpha - 1, clamped at the simplex boundary, posterior
        mean when degenerate) carrying the posterior DictDirichlet forward as the new prior."""
        from mixle.stats.bayes.dict_dirichlet import DictDirichletDistribution

        count_map = suff_stat
        conj_prior_params = self.prior.get_parameters()

        if isinstance(conj_prior_params, float):
            alpha = conj_prior_params
            keys = count_map.keys()
            # Dirichlet MAP sits on the boundary when alpha_k + n_k < 1
            num = {k: max((alpha - 1) + count_map[k], 0.0) for k in keys}
            cpp = {k: (alpha + count_map[k]) for k in keys}
        else:
            keys = set(conj_prior_params.keys()).union(count_map.keys())
            num = {k: max((conj_prior_params.get(k, 0.0) - 1) + count_map.get(k, 0.0), 0.0) for k in keys}
            cpp = {k: (conj_prior_params.get(k, 0.0) + count_map.get(k, 0.0)) for k in keys}

        norm_const = sum(num.values())

        if norm_const > 0:
            p_map = {k: v / norm_const for k, v in num.items()}
        else:
            # fall back to the posterior mean when the MAP is degenerate
            cpp_sum = sum(cpp.values())
            p_map = {k: v / cpp_sum for k, v in cpp.items()}

        return CategoricalDistribution(
            pmap=p_map, default_value=0.0, name=self.name, keys=self.keys, prior=DictDirichletDistribution(cpp)
        )

    def estimate(self, nobs: float | None, suff_stat: dict[Any, float]) -> "CategoricalDistribution":
        """Estimate a CategoricalDistribution from suff_stat value.

        If default_value is True, we estimate a default value from the suff_stat counts. Else, it is set to 0.0.

        pseudo_count is used to averaged over the number of levels and added to the corresponding counts.

        If suff_stat member value is None, estimate for CategoricalDistribution is formed from the suff_stat passed.
        Otherwise, the suff_stat member value is combined with the suff_stat values passed to estimate.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.estimate.
            suff_stat (Dict[Any, float]): Dict with categories as keys and counts as values from accumulated data.

        Returns:
            CategoricalDistribution estimated from passed in suff_stat value and sufficient statistic member variable
                (if it is not None).

        """
        if not suff_stat:
            if self.has_conj_prior:
                prior_parameters = self.prior.get_parameters()
                if isinstance(prior_parameters, dict) and prior_parameters:
                    suff_stat = {key: 0.0 for key in prior_parameters}
                else:
                    raise ValueError("empty categorical fitting requires a prior with an explicit finite support.")
            elif self.suff_stat is not None:
                suff_stat = {key: 0.0 for key in self.suff_stat}
            # With no prior and no declared support there is nothing to widen an empty count map
            # into. The EM dead-component case that used to justify returning an empty categorical
            # here is now handled upstream: CategoricalAccumulator records every label it was shown,
            # so a component that won zero responsibility reports those labels at zero count and
            # estimates to a uniform below -- generative, and able to win responsibility again, which
            # an all -inf object never could (MXR-080-1220).

        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        if not suff_stat:
            # NOTHING was accumulated -- not even a label at zero weight -- so there is no set of
            # labels to place a distribution over. Raising here was tried and reverted: it is a state
            # the library legitimately reaches (a learned segment fitted on all-empty sequences, a
            # gated mixture whose evidence is impossible for a component, hidden association), and
            # refusing broke all three. The result is the empty categorical, which scores -inf
            # everywhere and reports LIKELIHOOD_FACTOR rather than posing as a law; a mixture built
            # over it is a factor too and refuses to sample (MXR-080-1857).
            #
            # The EM dead-component case that used to share this branch no longer reaches it: the
            # accumulator records every label it was shown, so a component that saw data and won no
            # weight arrives here with a zero-count support and estimates to a uniform below -- which,
            # unlike an all -inf object, can win responsibility again (MXR-080-1220).
            # Deliberately NOT scoring_only. Declaring it would be the more honest label and was
            # tried three times; each attempt broke the learned-segment, gated-mixture and
            # hidden-association paths, because scoring_only makes it non-composable and those
            # legitimately build latent models over it. What closes the sampler half of
            # MXR-080-1220 instead is CategoricalSampler's normalized-law requirement
            # (MXR-080-1857): this object has is_normalized_probability False, so sampling it is
            # refused with the reason, and density_semantics() already reports LIKELIHOOD_FACTOR.
            return CategoricalDistribution({}, default_value=0.0, name=self.name, keys=self.keys)

        stats_sum = sum(suff_stat.values())

        if self.default_value:
            if stats_sum > 0:
                default_value = 1.0 / stats_sum
                default_value *= default_value

            else:
                default_value = 0.5
        else:
            default_value = 0.0

        if self.pseudo_count is None and self.suff_stat is None:
            nobs_loc = stats_sum

            if nobs_loc == 0.0:
                p_map = {k: 1.0 / float(len(suff_stat)) for k in suff_stat.keys()}
            else:
                p_map = {k: v / nobs_loc for k, v in suff_stat.items()}

        elif self.pseudo_count is not None and self.suff_stat is None:
            nobs_loc = stats_sum
            pseudo_count_per_level = self.pseudo_count / len(suff_stat)
            adjusted_nobs = nobs_loc + self.pseudo_count

            p_map = {k: (v + pseudo_count_per_level) / adjusted_nobs for k, v in suff_stat.items()}

        else:
            suff_stat_sum = sum(self.suff_stat.values())

            levels = set(suff_stat.keys()).union(self.suff_stat.keys())
            adjusted_nobs = suff_stat_sum * self.pseudo_count + stats_sum

            p_map = {
                k: (suff_stat.get(k, 0) + self.suff_stat.get(k, 0) * self.pseudo_count) / adjusted_nobs for k in levels
            }

        # Deliberately not scoring_only: a smoothed estimate with a positive default_value is a
        # legitimate fitted model, and flagging it as a likelihood factor would silently bar every
        # smoothed categorical from being used as a mixture component or sampled from.
        return CategoricalDistribution(pmap=p_map, default_value=default_value, name=self.name, keys=self.keys)


class CategoricalDataEncoder(DataSequenceEncoder):
    """Sequence encoder for categorical observations used by vectorized ``seq_*`` methods."""

    def __str__(self) -> str:
        """Print out name of DataSequenceEncoder.

        Returns:
            (str) CategoricalDataEncoder.

        """
        return "CategoricalDataEncoder"

    def __eq__(self, other) -> bool:
        """Return whether ``other`` is an equivalent categorical encoder.

        Args:
            other (object): Object to compare.

        Returns:
            True if ``other`` is a categorical encoder, else False.

        """
        return isinstance(other, CategoricalDataEncoder)

    def seq_encode(self, x: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Encode a list of category labels for vectorized ``seq_*`` methods.

        Args:
            x (List[Any]): List of category labels.

        Returns:
            Tuple of integer category indices and the object array mapping those indices back to labels.

        """
        level_to_index: dict[Any, int] = {}
        levels: list[Any] = []
        indices = np.empty(len(x), dtype=np.int64)
        for position, value in enumerate(x):
            try:
                index = level_to_index.get(value)
            except TypeError as exc:
                raise ValueError("categorical labels must be hashable.") from exc
            if index is None:
                index = len(levels)
                try:
                    level_to_index[value] = index
                except TypeError as exc:
                    raise ValueError("categorical labels must be hashable.") from exc
                levels.append(value)
            indices[position] = index
        return indices, np.asarray(levels, dtype=object)

    def row_count(self, x: Any) -> int:
        """Return the observation count from the categorical index vector."""
        if not isinstance(x, tuple) or len(x) != 2:
            raise ValueError("categorical encoded data must be an (indices, levels) pair.")
        indices = np.asarray(x[0])
        levels = np.asarray(x[1], dtype=object)
        if indices.ndim != 1 or levels.ndim != 1:
            raise ValueError("categorical encoded indices and levels must be one-dimensional.")
        if len(indices) and (
            not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(levels))
        ):
            raise ValueError("categorical encoded indices must refer to the encoded level table.")
        return len(indices)
