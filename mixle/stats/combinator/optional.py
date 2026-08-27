"""Optional distributions for explicit missing-value mass.

This distribution assigns a probability (p) to data being missing. With probability (1-p) the data is assumed to come
from a base distribution set by the user.

The OptionalDistribution allows for potentially missing data. The value p (the probability of being missing)
must be specified to sample from the distribution.

**The two modes, and why only one of them is a law.** ``p`` is optional, and leaving it out does not
mean "p is zero" -- it selects a different object:

* ``p`` given -- a normalized mixture: mass ``p`` on the sentinel, the base family scaled by
  ``1-p`` everywhere else. It integrates to one, ``density_semantics()`` reports ``EXACT``, and it
  samples.
* ``p is None`` -- the missing outcome is MARGINALIZED, i.e. ``log_density(sentinel)`` is ``0.0``
  and the base density is passed through unscaled. This is a likelihood FACTOR conditioned on an
  externally modeled missingness mechanism, not a law: its total mass is 1 (the base) + 1 (the
  sentinel scored at probability one) = 2, sentinel rows cost exactly zero nats, and
  ``density_semantics()`` reports ``LIKELIHOOD_FACTOR`` to say so. It cannot be sampled
  (``NonGenerativeOptionalError``).

Two consequences worth knowing before choosing. First, ``.p`` reads ``0.0`` in the marginalized mode
-- ``has_p`` is the attribute that distinguishes "no rate was ever fitted" from "the rate was fitted
and came out zero", and ``repr`` prints ``p=None`` for the former. Second, the free sentinel rows
mean a marginalized model cannot be compared on likelihood against a proper one over the same data:
it wins by construction, by ``-log(p)`` nats per sentinel row, and the paired comparison helpers in
``mixle.inference`` (``vuong_test``, ``clarke_test``, ``compare_elpd``) take plain score arrays and
so cannot see the flag that would have warned them. Auto-inference accordingly fits the rate for
every sentinel it introduces (see :meth:`mixle.utils.automatic.DatumNode.get_estimator`); the
marginalized mode is for callers who ask for it deliberately.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.capability import Discrete, supports
from mixle.enumeration.algorithms import freeze, merge_enumerators
from mixle.stats.combinator.composite import _distribute_child_prior
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
from mixle.utils.special import digamma

T = TypeVar("T")
E = TypeVar("E")
SS = TypeVar("SS")


from mixle.inference.fisher import EmpiricalMetricFixedFisherView, FixedFisherView, to_fisher


class NonGenerativeOptionalError(TypeError):
    """Raised when a marginalized optional likelihood factor is used to generate data."""


def _sentinel_key(value: Any) -> Any:
    """Return a typed, total key for missing-sentinel identity."""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ("nan",)
    try:
        return ("value", type(value), freeze(value))
    except (TypeError, ValueError, OverflowError):
        return ("identity", type(value), id(value))


def _same_sentinel(left: Any, right: Any) -> bool:
    """Total equivalence used by every optional missing-value surface."""
    if left is right:
        return True
    try:
        result = _sentinel_key(left) == _sentinel_key(right)
    except Exception:  # noqa: BLE001 - equivalence must be total for arbitrary user values
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _finite_nonnegative(value: Any, *, label: str) -> float:
    try:
        checked = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a finite non-negative scalar." % label) from exc
    if not np.isfinite(checked) or checked < 0.0:
        raise ValueError("%s must be a finite non-negative scalar." % label)
    return checked


def _validated_optional_statistics(value: Any) -> tuple[tuple[float, float], Any]:
    """Validate and detach a legacy ``((missing,present), child)`` statistic."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ContractError(
            "OptionalEstimator.estimate(suff_stat)",
            "a 2-tuple ((missing_weight, present_weight), base_suff_stat)",
            type(value).__name__,
            "pass the value produced by OptionalEstimatorAccumulator.value().",
        )
    counts = value[0]
    if not isinstance(counts, (tuple, list, np.ndarray)) or np.asarray(counts).shape != (2,):
        raise ContractError(
            "OptionalEstimator.estimate(suff_stat[0])",
            "a 2-element (missing_weight, present_weight) pair",
            type(counts).__name__,
            "pass the count pair produced by OptionalEstimatorAccumulator.value().",
        )
    missing = _finite_nonnegative(counts[0], label="missing weight")
    present = _finite_nonnegative(counts[1], label="present weight")
    return (missing, present), value[1]


def _beta_posterior_point(alpha: float, beta: float) -> float:
    """Choose a valid deterministic posterior point for every Beta shape regime."""
    if alpha > 1.0 and beta > 1.0:
        return (alpha - 1.0) / (alpha + beta - 2.0)
    if alpha <= 1.0 < beta:
        return 0.0
    if beta <= 1.0 < alpha:
        return 1.0
    if alpha > beta:
        return 1.0
    if beta > alpha:
        return 0.0
    return 0.5


class OptionalDistribution(SequenceEncodableProbabilityDistribution):
    """Mixture-style wrapper that models missing observations explicitly."""

    def compute_capabilities(self):
        """Return compute capabilities inherited from the observed-data distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, capabilities_for

        child = capabilities_for(self.dist)
        return DistributionCapabilities(engine_ready=child.engine_ready, kernel_status="numba_adapter")

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        p: float | None = None,
        missing_value: Any = None,
        name: str | None = None,
        prior: tuple[Any, Any] | None = None,
    ) -> None:
        """OptionalDistribution for handling missing values in estimation.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Base distribution.
            p (Optional[float]): Probability that dist has missing_value.
            missing_value (Any): Missing value from dist.
            name (Optional[str]): Optional distribution name.
            prior (Optional): Joint parameter prior ``(p_prior, dist_prior)``. ``p_prior`` is a conjugate
                Beta prior on the missing probability ``p`` (a
                :class:`~mixle.stats.univariate.continuous.beta.BetaDistribution`); ``dist_prior`` is the underlying
                distribution's prior, distributed via ``set_prior``. ``None`` (default) leaves a plain
                point model (existing behavior byte-identical).

        Attributes:
            dist (SequenceEncodableProbabilityDistribution): Base distribution.
            p (float): Probability that dist has missing_value. Reads 0.0 when ``p`` was not
                supplied -- ``has_p``, not this, distinguishes "no rate was ever fitted" (the
                marginalized likelihood factor) from "the rate was fitted and came out zero".
            has_p (bool): True if distribution has arg p passed.
            log_p (float): log of p.
            log_pn (float): log(1-p).
            missing_value_is_nan (bool): True if the missing value is nan.
            missing_value (Any): Missing value from dist.
            name (Optional[str]): Optional distribution name.

        """
        if p is not None and (not np.isfinite(p) or p < 0.0 or p > 1.0):
            raise ValueError("OptionalDistribution requires p in [0, 1].")
        self.dist = dist
        self.p = p if p is not None else 0.0
        self.has_p = p is not None
        self.log_p = -np.inf if self.p == 0 else np.log(self.p)
        self.log_pn = -np.inf if self.p == 1 else np.log1p(-self.p)

        self.missing_value = missing_value
        self.missing_value_is_nan = _same_sentinel(missing_value, float("nan"))
        if supports(dist, Discrete):
            try:
                collision_score = float(dist.log_density(missing_value))
            except (TypeError, ValueError, IndexError, KeyError, OverflowError):
                collision_score = -np.inf
            if np.isnan(collision_score) or collision_score > -np.inf:
                raise ValueError(
                    "OptionalDistribution missing_value collides with positive-mass child support; "
                    "choose a disjoint sentinel."
                )
        self.name = name
        if not self.has_p and prior is not None and prior[0] is not None:
            raise ValueError(
                "a marginalized OptionalDistribution has no missingness parameter and cannot carry a p prior."
            )
        self.set_prior(prior)

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Return the constructor-owned state used by the safe JSON codec.

        Everything else on this instance is derived in __init__ (has_p, log_p, log_pn,
        missing_value_is_nan, the prior split) and must not be serialized: the generic object codec
        would otherwise try to encode derived values it cannot represent and reject the whole model.
        ``p`` round-trips as None when it was never supplied, which is what has_p records.
        """
        return {
            "dist": self.dist,
            "p": self.p if self.has_p else None,
            "missing_value": self.missing_value,
            "name": self.name,
            "prior": self.get_prior()[0],
        }

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild from constructor-owned state, re-deriving everything __init__ computes."""
        required = {"dist", "p", "missing_value", "name"}
        missing = required - set(state)
        if missing:
            raise ValueError("OptionalDistribution state is missing %s" % ", ".join(sorted(missing)))
        prior = state.get("prior")
        self.__init__(
            state["dist"],
            p=state["p"],
            missing_value=state["missing_value"],
            name=state["name"],
            prior=None if prior is None else (prior, None),
        )

    def _is_missing(self, value: Any) -> bool:
        return _same_sentinel(value, self.missing_value)

    def get_prior(self) -> tuple[Any, Any]:
        """Return the joint prior as ``(p_prior, dist_prior)``."""
        return self.prior, self.dist.get_prior()

    def set_prior(self, prior: tuple[Any, Any] | None) -> None:
        """Distribute the joint prior ``(p_prior, dist_prior)`` to the missing probability and base dist.

        ``prior=None`` is a no-op (point model, existing behavior byte-identical). Otherwise the first
        element is a conjugate Beta prior on ``p`` (caching the digamma expectations used by
        ``expected_log_density``) and the second is pushed to the base distribution's ``set_prior``.
        """
        if prior is None:
            self.prior = None
            self.conj_prior_params = None
            self.has_conj_prior = False
            return
        self.dist.set_prior(prior[1])
        self._set_p_prior(prior[0])

    def _set_p_prior(self, p_prior: Any) -> None:
        from mixle.stats.univariate.continuous.beta import BetaDistribution

        self.prior = p_prior
        if isinstance(p_prior, BetaDistribution):
            a, b = p_prior.get_parameters()
            self.conj_prior_params = (digamma(a), digamma(b), digamma(a + b))
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.has_conj_prior = False

    def expected_log_density(self, x: T) -> float:
        """Posterior-expected log-density ``E_q[log p(x)]`` at ``x``.

        With a conjugate Beta prior on ``p`` the expectation over ``p`` is available in closed form via
        digamma terms (missing => ``da - dab``; observed => ``db - dab + dist.expected_log_density(x)``);
        otherwise this falls back to the plug-in ``log_density``.
        """
        if not self.has_conj_prior:
            return self.log_density(x)
        da, db, dab = self.conj_prior_params
        missing = self._is_missing(x)
        if missing:
            return da - dab
        return db - dab + self.dist.expected_log_density(x)

    def seq_expected_log_density(self, x: tuple[int, np.ndarray, np.ndarray, E]) -> np.ndarray:
        """Vectorized posterior-expected log-density; falls back to ``seq_log_density`` without a prior."""
        if not self.has_conj_prior:
            return self.seq_log_density(x)
        sz, z_idx, nz_idx, enc_data = x
        da, db, dab = self.conj_prior_params
        rv = np.empty(sz, dtype=np.float64)
        rv.fill(da - dab)
        rv[nz_idx] = self.dist.seq_expected_log_density(enc_data) + (db - dab)
        return rv

    def compute_declaration(self):
        """Return a structured declaration for the optional missingness wrapper."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        child = declaration_for(self.dist)
        children = () if child is None else (child,)
        return DistributionDeclaration(
            name="optional",
            distribution_type=type(self),
            parameters=(ParameterSpec("p", constraint="unit_interval"),),
            statistics=(
                StatisticSpec("missing_observed_counts"),
                StatisticSpec("observed", kind="child_stat"),
            ),
            support="optional",
            children=children,
            child_roles=("observed",) if children else (),
            differentiable=all(child.differentiable for child in children),
        )

    def __str__(self) -> str:
        s1 = str(self.dist)
        s2 = repr(None if not self.has_p else self.p)
        if self.missing_value_is_nan:
            s3 = 'float("nan")'
        else:
            s3 = repr(self.missing_value)
        s4 = repr(self.name)
        return "OptionalDistribution(%s, p=%s, missing_value=%s, name=%s)" % (s1, s2, s3, s4)

    def density(self, x: T) -> float:
        """Evaluate the density of the Optional distribution at x.

        See log_density() for details.

        Args:
            x (T): Observation from base dist or missing value.

        Returns:
            Density at x.

        """
        return np.exp(self.log_density(x))

    def density_semantics(self):
        """Return child semantics for a proper gate or likelihood-factor semantics for MAR."""
        from mixle.stats.compute.pdist import DensitySemantics

        return self.dist.density_semantics() if self.has_p else DensitySemantics.LIKELIHOOD_FACTOR

    @property
    def marginalized_core(self):
        """Return the wrapped law when this is a pure marginalization, else None.

        Without a missingness rate this distribution is a likelihood factor: it contributes 1 where the
        value is absent rather than a normalized density over values-plus-missing. That makes it
        non-generative, but only because ``p`` was not supplied -- the wrapped law is untouched, and
        giving ``p`` makes the whole thing generative again. Callers that need to tell "unnormalized
        because we are marginalizing over missingness" apart from "intrinsically cannot generate"
        (a scoring-only leaf) read this: the former returns its core, the latter has no such attribute.
        """
        return None if self.has_p else self.dist

    def log_density(self, x: T) -> float:
        """Evalute the log density of the Optional distribution at x.

        If x is a missing value: return log(p) if p is not None, else return 0.0
        If x is not the missing_value: if p is not None, return the log_denisty(x) at base dist + log(1-p) else: return
        log_density(x).

        Args:
            x (T): Observation from base dist or missing value.

        Returns:
            Log-density at x.

        """
        not_missing = not self._is_missing(x)

        if self.has_p:
            if not_missing:
                return self.dist.log_density(x) + self.log_pn
            else:
                return self.log_p
        # p is None: MARGINALIZE the missing value (it contributes log-density 0) instead of modeling a
        # missingness probability -- the missing-at-random treatment for occasional missing entries.
        # See mixle.stats.missing (MISSING sentinel + marginalized()/composite_with_missing() builders).
        else:
            if not_missing:
                return self.dist.log_density(x)
            else:
                return 0.0

    def seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, E]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        sz, z_idx, nz_idx, enc_data = x

        rv = np.zeros(sz)

        if self.has_p:
            rv[z_idx] = self.log_p
            rv[nz_idx] = self.dist.seq_log_density(enc_data) + self.log_pn
        else:
            rv[nz_idx] = self.dist.seq_log_density(enc_data)

        return rv

    def backend_seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, E], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for optional encoded data."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, z_idx, nz_idx, enc_data = x
        rv = engine.zeros(sz)
        if self.has_p and len(z_idx):
            rv[engine.asarray(z_idx)] = engine.asarray(self.log_p)
        if len(nz_idx):
            nz_scores = backend_seq_log_density(self.dist, enc_data, engine)
            if self.has_p:
                nz_scores = nz_scores + engine.asarray(self.log_pn)
            rv[engine.asarray(nz_idx)] = nz_scores
        return rv

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from mixle.stats.compute.gradient import OptionalGradientFitState

        child = recurse(self.dist, engine, torch, leaves)
        logit_p = None
        if self.has_p:
            logit_p = tensor_param(self.p, engine, torch, transform="logit")
            leaves.append(logit_p)
        return OptionalGradientFitState(self, child, logit_p)

    @staticmethod
    def _same_missing_value(a: OptionalDistribution, b: OptionalDistribution) -> bool:
        return _same_sentinel(a.missing_value, b.missing_value)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence[OptionalDistribution], engine: Any) -> dict[str, Any]:
        """Return stacked optional-wrapper parameters for homogeneous mixture kernels."""
        from mixle.stats.compute.stacked import stacked_component_params

        if any(not cls._same_missing_value(dists[0], dist) for dist in dists[1:]):
            raise ValueError("Stacked OptionalDistribution components require a shared missing value.")
        child_dists = [dist.dist for dist in dists]
        try:
            child_route = stacked_component_params(child_dists, engine)
        except ValueError as exc:
            raise ValueError("Optional child %s is not stackable: %s" % (type(child_dists[0]).__name__, exc))
        return {
            "__pysp_component_axis__": {"has_p": 0, "log_p": 0, "log_pn": 0},
            "child_route": child_route,
            "has_p": engine.asarray([dist.has_p for dist in dists]),
            "log_p": engine.asarray([dist.log_p for dist in dists]),
            "log_pn": engine.asarray([dist.log_pn for dist in dists]),
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(
        cls, x: tuple[int, np.ndarray, np.ndarray, E], params: dict[str, Any], engine: Any
    ) -> Any:
        """Return an ``(n, k)`` matrix of optional-wrapper log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        sz, z_idx, nz_idx, enc_data = x
        num_components = params["num_components"]
        rv = engine.zeros((sz, num_components))
        has_p = params["has_p"]
        if len(z_idx):
            missing_scores = engine.where(has_p, params["log_p"], engine.asarray(0.0))
            rv[engine.asarray(z_idx), :] = missing_scores[None, :] + engine.zeros((len(z_idx), num_components))
        if len(nz_idx):
            child_scores = stacked_component_log_density(enc_data, params["child_route"], engine)
            observed_scores = engine.where(has_p[None, :], child_scores + params["log_pn"][None, :], child_scores)
            rv[engine.asarray(nz_idx), :] = observed_scores
        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: tuple[int, np.ndarray, np.ndarray, E], weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy optional-wrapper sufficient statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        _, z_idx, nz_idx, enc_data = x
        ww = engine.asarray(weights)
        num_components = int(params["num_components"])
        if len(z_idx):
            missing_counts = engine.sum(ww[engine.asarray(z_idx), :], axis=0)
        else:
            missing_counts = engine.zeros(num_components)
        if len(nz_idx):
            observed_weights = ww[engine.asarray(nz_idx), :]
            observed_counts = engine.sum(observed_weights, axis=0)
        else:
            observed_weights = engine.zeros((0, num_components))
            observed_counts = engine.zeros(num_components)
        component_estimators = tuple(getattr(est, "estimator", None) for est in getattr(estimator, "estimators", ()))
        child_estimator = (
            StackedEstimatorView(component_estimators) if len(component_estimators) == num_components else None
        )
        child_stats = stacked_component_sufficient_statistics(
            enc_data, observed_weights, params["child_route"], engine, child_estimator
        )
        child_values = unstack_component_stats(child_stats, num_components)
        return tuple(((missing_counts[i], observed_counts[i]), child_values[i]) for i in range(num_components))

    def to_fisher(self, **kwargs):
        """Fisher view for the optional/missing-gate."""
        if hasattr(self, "dist"):
            return OptionalFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> OptionalSampler:
        """Return a sampler for drawing observations from this distribution."""
        if not self.has_p:
            raise NonGenerativeOptionalError(
                "OptionalDistribution with p=None is a marginalized likelihood factor and cannot "
                "generate observations; specify p or sample from its child explicitly."
            )
        return OptionalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> OptionalEstimator:
        """Return an estimator for fitting this distribution from data."""
        prior = None if self.prior is None else (self.prior, self.dist.get_prior())
        estimator = OptionalEstimator(
            self.dist.estimator(pseudo_count=pseudo_count),
            missing_value=self.missing_value,
            pseudo_count=pseudo_count,
            est_prob=self.has_p,
            name=self.name,
            prior=prior,
        )
        if getattr(self, "_marginalized_by_helper", False):
            estimator._marginalized_by_helper = True
        return estimator

    def dist_to_encoder(self) -> OptionalDataEncoder:
        """Return the data encoder used by this distribution for vectorized methods."""
        return OptionalDataEncoder(encoder=self.dist.dist_to_encoder(), missing_value=self.missing_value)

    def enumerator(self) -> OptionalEnumerator:
        """Returns an OptionalEnumerator iterating the support (including the missing value) in
        descending probability order."""
        return OptionalEnumerator(self)


class OptionalEnumerator(DistributionEnumerator):
    """Enumerate the optional support by merging missing mass with observed support."""

    def __init__(self, dist: OptionalDistribution) -> None:
        """Enumerates the base support scaled by (1-p), merged with the missing value at p.

        Base-support entries equal to the missing value are filtered out: log_density routes
        them to the missing branch, so their base mass is unreachable. Raises EnumerationError
        when no p was given (the degenerate legacy mode where total mass exceeds one).

        Args:
            dist (OptionalDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        if not dist.has_p:
            raise EnumerationError(
                dist, reason="no missing probability p given; total mass exceeds one in this legacy mode"
            )
        if dist.p >= 1.0:
            self._merged = iter([(dist.missing_value, 0.0)])
            return
        base = child_enumerator(dist.dist, "OptionalDistribution.dist")
        base = ((v, lp) for v, lp in base if not _same_sentinel(v, dist.missing_value))
        if dist.p <= 0.0:
            self._merged = ((v, lp) for v, lp in base)
            return
        self._merged = merge_enumerators([iter([(dist.missing_value, 0.0)]), base], [dist.log_p, dist.log_pn])

    def __next__(self) -> tuple[Any, float]:
        return next(self._merged)


class OptionalSampler(DistributionSampler):
    """Sample from an optional distribution by first drawing the missingness gate."""

    def __init__(self, dist: OptionalDistribution, seed: int | None = None) -> None:
        if not dist.has_p:
            raise NonGenerativeOptionalError(
                "a marginalized OptionalDistribution is a likelihood factor, not a sampling law."
            )
        super().__init__(dist, seed)
        self.dist = dist
        self.sampler = self.dist.dist.sampler(self.new_seed())

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one observation or a list of observations from the optional mixture."""

        sampler = self.sampler

        if size is None:
            if self.rng.choice([0, 1], replace=True, p=[self.dist.p, 1.0 - self.dist.p]) == 0:
                return self.dist.missing_value
            else:
                return sampler.sample(size=size)
        else:
            states = self.rng.choice([0, 1], size=size, replace=True, p=[self.dist.p, 1.0 - self.dist.p])

            nz_count = int(np.sum(states))

            if nz_count == size:
                return sampler.sample(size=size)
            elif nz_count == 0:
                return [self.dist.missing_value for i in range(size)]
            else:
                nz_vals = sampler.sample(size=nz_count)
                nz_idx = np.flatnonzero(states)
                rv = [self.dist.missing_value for i in range(size)]

                for cnt, i in enumerate(nz_idx):
                    rv[i] = nz_vals[cnt]

                return rv


class OptionalEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate missing/observed gate weights plus observed-branch statistics."""

    def __init__(
        self,
        accumulator: SequenceEncodableStatisticAccumulator,
        missing_value: Any = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.accumulator = accumulator
        self.weights = [0.0, 0.0]
        self.missing_value = missing_value
        self.missing_value_is_nan = _same_sentinel(missing_value, float("nan"))
        self.keys = keys
        self.name = name

    def update(self, x: T, weight: float, estimate: OptionalDistribution) -> None:
        """Update from a single observation, routing observed values to the child accumulator."""
        base_estimate = estimate.dist if estimate is not None else None
        checked_weight = _finite_nonnegative(weight, label="weight")
        if _same_sentinel(x, self.missing_value):
            self.weights[0] += checked_weight
        else:
            self.accumulator.update(x, checked_weight, base_estimate)
            self.weights[1] += checked_weight

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        """Initialize from a single observation using the child initializer when observed."""
        checked_weight = _finite_nonnegative(weight, label="weight")
        if _same_sentinel(x, self.missing_value):
            self.weights[0] += checked_weight
        else:
            self.accumulator.initialize(x, checked_weight, rng)
            self.weights[1] += checked_weight

    def seq_update(
        self, x: tuple[int, np.ndarray, np.ndarray, E], weights: np.ndarray, estimate: OptionalDistribution
    ) -> None:
        """Update from encoded optional data and observation weights."""
        sz, z_idx, nz_idx, enc_data = x
        nz_weights = weights[nz_idx]
        z_weights = weights[z_idx]

        self.weights[0] += np.sum(z_weights)
        self.weights[1] += np.sum(nz_weights)
        self.accumulator.seq_update(enc_data, nz_weights, estimate.dist if estimate is not None else None)

    def seq_update_engine(
        self, x: tuple[int, np.ndarray, np.ndarray, E], weights: Any, estimate: OptionalDistribution, engine: Any
    ) -> None:
        """Engine-resident E-step: missing/observed mass is summed on the active engine and the
        observed child accumulator is routed through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        sz, z_idx, nz_idx, enc_data = x
        w_eng = engine.asarray(weights)
        nz_weights = w_eng[np.asarray(nz_idx, dtype=np.int64)]
        z_weights = w_eng[np.asarray(z_idx, dtype=np.int64)]

        self.weights[0] += float(engine.to_numpy(engine.sum(z_weights)))
        self.weights[1] += float(engine.to_numpy(engine.sum(nz_weights)))
        child_seq_update(
            self.accumulator, enc_data, nz_weights, estimate.dist if estimate is not None else None, engine
        )

    def seq_initialize(self, x: tuple[int, np.ndarray, np.ndarray, E], weights: np.ndarray, rng: RandomState) -> None:
        """Initialize from encoded optional data and weights."""
        sz, z_idx, nz_idx, enc_data = x
        nz_weights = weights[nz_idx]
        z_weights = weights[z_idx]

        self.weights[0] += np.sum(z_weights)
        self.weights[1] += np.sum(nz_weights)
        self.accumulator.seq_initialize(enc_data, nz_weights, rng)

    def combine(self, suff_stat: tuple[list[float], SS]) -> OptionalEstimatorAccumulator:
        """Merge missing/observed weights and child sufficient statistics."""
        counts, child_stat = _validated_optional_statistics(suff_stat)
        self.weights[0] += counts[0]
        self.weights[1] += counts[1]
        self.accumulator.combine(child_stat)

        return self

    def value(self) -> tuple[tuple[float, float], Any]:
        """Return gate weights together with observed-branch sufficient statistics."""
        return (float(self.weights[0]), float(self.weights[1])), self.accumulator.value()

    def from_value(self, x: tuple[list[float], SS]) -> OptionalEstimatorAccumulator:
        """Restore gate weights and observed-branch sufficient statistics."""
        counts, child_stat = _validated_optional_statistics(x)
        self.accumulator.from_value(child_stat)
        self.weights = [counts[0], counts[1]]

        return self

    def scale(self, c: float) -> OptionalEstimatorAccumulator:
        """Scale missing/observed weights and delegate observed statistics."""
        self.weights[0] *= c
        self.weights[1] *= c
        self.accumulator.scale(c)
        return self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace own + wrapped-child statistics from the pooled ``stats_dict`` (see key_merge)."""
        # The pull direction matters: this used to PUSH self.value() INTO the dict-held pool,
        # which overwrote the pooled statistics with the last site's own and left the site itself
        # untouched -- tied sites never received the pool (caught by the keyed-protocol sweep).
        super().key_replace(stats_dict)
        self.accumulator.key_replace(stats_dict)

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge own statistics under ``self.keys`` and DELEGATE to the wrapped child.

        The delegation is the point of the override: without it, a keyed estimator nested under an
        Optional never pooled at all (its key pass was unreachable -- the dead-keys shape the
        keyed-protocol sweep hunts). The own-key pooling itself is the base-class canonical
        protocol.
        """
        super().key_merge(stats_dict)
        self.accumulator.key_merge(stats_dict)

    def acc_to_encoder(self) -> OptionalDataEncoder:
        """Return the optional encoder matching the wrapped child accumulator."""
        return OptionalDataEncoder(encoder=self.accumulator.acc_to_encoder(), missing_value=self.missing_value)


class OptionalEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for optional missingness estimators."""

    def __init__(
        self,
        estimator: ParameterEstimator,
        missing_value: Any = None,
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.missing_value = missing_value
        self.keys = keys
        self.name = name

    def make(self) -> OptionalEstimatorAccumulator:
        """Create an empty optional estimator accumulator."""
        return OptionalEstimatorAccumulator(
            self.estimator.accumulator_factory().make(), self.missing_value, keys=self.keys, name=self.name
        )


class OptionalEstimator(ParameterEstimator):
    """Estimate optional missingness probability and observed-data distribution parameters."""

    def __init__(
        self,
        estimator: ParameterEstimator,
        missing_value: Any = None,
        est_prob: bool = False,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: tuple[Any, Any] | None = None,
    ) -> None:
        """OptionalEstimator for estimating OptionalDistribution from sufficient statistics.

        Args:
            estimator (ParameterEstimator): Estimator for base distribution.
            missing_value (Any): Missing_value specification.
            est_prob (bool): If true estimate the probability of a missing value.
            pseudo_count (Optional[float]): Regularize estimate of missing data.
            name (Optional[str]): Optional name assigned to the estimated distribution.
            keys (Optional[str]): Set keys for sufficient statistics.
            prior (Optional): Joint parameter prior ``(p_prior, dist_prior)``. ``p_prior`` is a conjugate
                Beta prior on ``p``; ``dist_prior`` is delegated to the base estimator. ``None`` (default)
                leaves the empirical / pseudo-count update byte-identical.

        Attributes:
            estimator (ParameterEstimator): Estimator for base distribution.
            missing_value (Any): Missing_value specification.
            est_prob (bool): If true estimate the probability of a missing value.
            pseudo_count (Optional[float]): Regularize estimate of missing data.
            name (Optional[str]): Optional name assigned to the estimated distribution.
            keys (Optional[str]): Set keys for sufficient statistics.

        """
        self.estimator = estimator
        self.est_prob = est_prob
        self.pseudo_count = None if pseudo_count is None else _finite_nonnegative(pseudo_count, label="pseudo_count")
        self.missing_value = missing_value
        self.keys = keys
        self.name = name
        self.prior = None
        self.has_conj_prior = False
        self.set_prior(prior)

    def accumulator_factory(self) -> OptionalEstimatorAccumulatorFactory:
        """Return an accumulator factory for optional sufficient statistics."""
        return OptionalEstimatorAccumulatorFactory(self.estimator, self.missing_value, keys=self.keys, name=self.name)

    def get_prior(self) -> tuple[Any, Any]:
        """Return the joint prior as ``(p_prior, dist_prior)`` from this estimator and the base estimator."""
        return self.prior, self.estimator.get_prior()

    def set_prior(self, prior: tuple[Any, Any] | None) -> None:
        """Distribute ``(p_prior, dist_prior)`` to this estimator's ``p`` prior and the base estimator.

        ``prior=None`` is a no-op (empirical/pseudo-count path stays byte-identical). The first element
        is a conjugate Beta prior on ``p``; the second is pushed to the base estimator via ``set_prior``.
        """
        from mixle.stats.univariate.continuous.beta import BetaDistribution

        if prior is None:
            return
        _distribute_child_prior(self.estimator, prior[1])
        self.prior = prior[0]
        self.has_conj_prior = isinstance(prior[0], BetaDistribution)

    def model_log_density(self, model: OptionalDistribution) -> float:
        """Sum the Beta-prior log-density at ``p`` and the base estimator's term (ELBO global term)."""
        rv = self.estimator.model_log_density(model.dist)
        if self.has_conj_prior:
            rv += float(self.prior.log_density(model.p))
        return rv

    def _preserve_marginalization_provenance(self, dist: OptionalDistribution) -> OptionalDistribution:
        """Carry the MAR-helper marker through an estimator update."""
        if getattr(self, "_marginalized_by_helper", False):
            dist._marginalized_by_helper = True
        return dist

    def _validate_suff_stat(self, suff_stat: tuple[list[float], SS] | None) -> tuple[tuple[float, float], SS]:
        return _validated_optional_statistics(suff_stat)

    def _estimate_conjugate(self, suff_stat: tuple[list[float], SS]) -> OptionalDistribution:
        """Closed-form Beta conjugate posterior update on ``p`` (carried forward as the fitted prior).

        ``psum`` is the missing weight and ``nsum`` the observed weight. The unique interior or boundary
        mode is used where one exists; symmetric/non-unique boundary regimes use the deterministic
        posterior point declared by :func:`_beta_posterior_point`.
        """
        from mixle.stats.univariate.continuous.beta import BetaDistribution

        counts, child_stat = self._validate_suff_stat(suff_stat)
        psum, nsum = counts
        try:
            dist = self.estimator.estimate(nsum, child_stat)
        except ContractError as e:
            raise prefix_contract_error("OptionalDistribution.dist", e) from None

        a, b = self.prior.get_parameters()
        new_a = a + psum
        new_b = b + nsum
        new_p = _beta_posterior_point(new_a, new_b)
        new_prior = BetaDistribution(new_a, new_b)
        return self._preserve_marginalization_provenance(
            OptionalDistribution(
                dist,
                p=new_p,
                missing_value=self.missing_value,
                name=self.name,
                prior=(new_prior, dist.get_prior()),
            )
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[list[float], SS] | None) -> OptionalDistribution:
        """Estimate an OptionalDistribution from missing/observed sufficient statistics."""
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        counts, child_stat = self._validate_suff_stat(suff_stat)
        try:
            dist = self.estimator.estimate(counts[1], child_stat)
        except ContractError as e:
            raise prefix_contract_error("OptionalDistribution.dist", e) from None

        if self.pseudo_count is not None and self.est_prob:
            return self._preserve_marginalization_provenance(
                OptionalDistribution(
                    dist,
                    (counts[0] + self.pseudo_count) / ((2 * self.pseudo_count) + counts[0] + counts[1]),
                    missing_value=self.missing_value,
                    name=self.name,
                )
            )

        elif self.est_prob:
            nobs_loc = counts[0] + counts[1]
            z_nobs = counts[0]

            if nobs_loc == 0:
                result = OptionalDistribution(dist, None, missing_value=self.missing_value, name=self.name)
            else:
                result = OptionalDistribution(
                    dist,
                    p=z_nobs / nobs_loc,
                    missing_value=self.missing_value,
                    name=self.name,
                )
            return self._preserve_marginalization_provenance(result)
        else:
            return self._preserve_marginalization_provenance(
                OptionalDistribution(
                    dist,
                    p=None,
                    missing_value=self.missing_value,
                    name=self.name,
                )
            )


class OptionalDataEncoder(DataSequenceEncoder):
    """Encode optional data as missing indices, observed indices, and child-encoded data."""

    def __init__(self, encoder: DataSequenceEncoder, missing_value: Any = None) -> None:
        self.encoder = encoder
        self.missing_value = missing_value
        self.missing_value_is_nan = _same_sentinel(missing_value, float("nan"))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OptionalDataEncoder):
            return self.encoder == other.encoder and _same_sentinel(self.missing_value, other.missing_value)
        else:
            return False

    def seq_encode(self, x: Sequence[T]) -> tuple[int, np.ndarray, np.ndarray, Any]:
        """Split a sequence into missing positions and encoded observed values."""
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise ContractError(
                "OptionalDistribution.seq_encode",
                "a sequence of observations (or the missing-value sentinel)",
                "%s" % type(x).__name__,
                "pass a list/tuple of observations, e.g. [x0, missing_value, x2, ...].",
            )

        nz_idx = []
        nz_val = []
        z_idx = []

        for i, v in enumerate(x):
            if _same_sentinel(v, self.missing_value):
                z_idx.append(i)
            else:
                nz_idx.append(i)
                nz_val.append(v)

        try:
            enc_data = self.encoder.seq_encode(nz_val)
        except ContractError as e:
            raise prefix_contract_error("OptionalDistribution.dist", e) from None
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise ContractError(
                "OptionalDistribution.dist",
                "every present (non-missing) value compatible with the base distribution's data type",
                "a value that raised %s: %s" % (type(e).__name__, e),
                "check that every present value matches the data type expected by the base "
                "distribution (%s); missing entries should equal missing_value=%r."
                % (self.encoder, self.missing_value),
            ) from e

        nz_idx = np.asarray(nz_idx, dtype=int)
        z_idx = np.asarray(z_idx, dtype=int)

        return len(x), z_idx, nz_idx, enc_data

    def row_count(self, x: Any) -> int:
        """Validate the missing/present partition and return its declared observation count."""
        if not isinstance(x, tuple) or len(x) != 4:
            raise ValueError("optional encoded data must be a (size, missing, present, child) tuple.")
        size, missing, present, child = x
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 0:
            raise ValueError("optional encoded size must be a non-negative integer.")
        missing = np.asarray(missing)
        present = np.asarray(present)
        if (
            missing.ndim != 1
            or present.ndim != 1
            or not np.issubdtype(missing.dtype, np.integer)
            or not np.issubdtype(present.dtype, np.integer)
        ):
            raise ValueError("optional missing/present indices must be one-dimensional integer arrays.")
        joined = np.concatenate((missing, present))
        if len(joined) != int(size) or not np.array_equal(np.sort(joined), np.arange(int(size))):
            raise ValueError("optional missing/present indices must partition every encoded row exactly once.")
        if self.encoder.row_count(child) != len(present):
            raise ValueError("optional child encoding row count must equal the number of present indices.")
        return int(size)


# --- Backward-compatible API naming aliases ---
OptionalAccumulator = OptionalEstimatorAccumulator
OptionalAccumulatorFactory = OptionalEstimatorAccumulatorFactory


# --- Fisher view(s) co-located with this family ---
class OptionalFisherView(EmpiricalMetricFixedFisherView):
    """Fisher view for optional distributions with gate and observed-branch statistics."""

    def __init__(self, dist: Any) -> None:
        self.child_view = to_fisher(dist.dist)
        self.has_gate = getattr(dist, "has_p", getattr(dist, "p", None) is not None)
        self._encoded_missing_first = hasattr(dist, "missing_value_is_nan")
        labels = []
        if self.has_gate:
            labels.extend([("missing",), ("present",)])
        labels.extend(("present_stat",) + label for label in self.child_view.vectorizer.labels)
        super().__init__(dist, labels)

    def _is_missing(self, x: Any) -> bool:
        return _same_sentinel(x, self.dist.missing_value)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        n = len(data)
        d = len(self.child_view.vectorizer.labels)
        child = np.zeros((n, d), dtype=np.float64)
        present_idx = []
        present_values = []
        gate = np.zeros((n, 2), dtype=np.float64) if self.has_gate else None
        for i, x in enumerate(data):
            missing = self._is_missing(x)
            if gate is not None:
                gate[i, 0 if missing else 1] = 1.0
            if not missing:
                present_idx.append(i)
                present_values.append(x)
        if present_values:
            child[np.asarray(present_idx, dtype=np.int64)] = self.child_view.expected_statistics_matrix(
                data=present_values
            )
        return np.hstack((gate, child)) if gate is not None else child

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        n, idx_a, idx_b, enc_child = enc_data
        z_idx, nz_idx = (idx_a, idx_b) if self._encoded_missing_first else (idx_b, idx_a)
        d = len(self.child_view.vectorizer.labels)
        child = np.zeros((n, d), dtype=np.float64)
        if len(nz_idx):
            child[np.asarray(nz_idx, dtype=np.int64)] = self.child_view.seq_expected_statistics(enc_child)
        if self.has_gate:
            gate = np.zeros((n, 2), dtype=np.float64)
            gate[np.asarray(z_idx, dtype=np.int64), 0] = 1.0
            gate[np.asarray(nz_idx, dtype=np.int64), 1] = 1.0
            return np.hstack((gate, child))
        return child

    def _model_mean(self) -> np.ndarray:
        if not self.has_gate:
            raise NotImplementedError
        p = float(self.dist.p)
        q = 1.0 - p
        return np.concatenate((np.asarray([p, q]), q * self.child_view.mean_statistics()))

    def _model_fisher(self) -> np.ndarray:
        if not self.has_gate:
            raise NotImplementedError
        p = float(self.dist.p)
        q = 1.0 - p
        mu = np.asarray(self.child_view.mean_statistics(), dtype=np.float64)
        info = np.asarray(self.child_view.fisher_information(ridge=0.0), dtype=np.float64)
        d = len(mu)
        out = np.zeros((2 + d, 2 + d), dtype=np.float64)
        gate_mean = np.asarray([p, q])
        out[:2, :2] = np.diag(gate_mean) - np.outer(gate_mean, gate_mean)
        out[0, 2:] = -p * q * mu
        out[2:, 0] = out[0, 2:]
        out[1, 2:] = p * q * mu
        out[2:, 1] = out[1, 2:]
        out[2:, 2:] = q * info + p * q * np.outer(mu, mu)
        return out

    def mean_statistics(self, stats: np.ndarray | None = None, model: bool = True, **kwargs: Any) -> np.ndarray:
        """Return model or empirical mean statistics for the optional Fisher view."""
        try:
            return FixedFisherView.mean_statistics(self, stats=stats, model=model, **kwargs)
        except NotImplementedError:
            return EmpiricalMetricFixedFisherView.mean_statistics(self, stats=stats, **kwargs)

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        """Return Fisher information, falling back to empirical statistics when needed."""
        try:
            return FixedFisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)
        except NotImplementedError:
            return EmpiricalMetricFixedFisherView.fisher_information(
                self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs
            )

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return Fisher-whitened statistic vectors for optional observations."""
        try:
            return FixedFisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )
        except NotImplementedError:
            return EmpiricalMetricFixedFisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )
