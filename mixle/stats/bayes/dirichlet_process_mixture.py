"""Dirichlet process mixture (truncated stick-breaking) with variational
Bayes estimation, adapted onto the mixle.stats base-class protocol.

A truncated stick-breaking representation with K components:

    alpha ~ Gamma(s1, 1/s2)                  (concentration hyper-prior)
    v_k | alpha ~ Beta(1, alpha)             (stick fractions, k < K)
    w_k = v_k * prod_{j<k} (1 - v_j)         (mixture weights)
    z_i | w ~ Categorical(w)
    x_i | z_i = k ~ components[k]

Data type: whatever the component distributions accept; a datum is a single
observation scored under the mixture log sum_k w_k p(x | theta_k).

Estimation is mean-field variational Bayes: accumulators collect the optimal
local assignments phi_ik (computed from the components' expected_log_density,
i.e. the VB E-step), and the estimator updates the variational Beta posteriors
gamma_k on the stick fractions, the Gamma hyper-posterior on alpha, and each
component's conjugate update. Components are re-sorted by expected count each
iteration, and each component's posterior (carried as its prior, i.e.
``component.get_prior()``) serves as the variational factor q(theta_k).
``seq_local_elbo`` provides the per-observation data terms of the ELBO; the
data-independent terms live in ``DirichletProcessMixtureEstimator.model_log_density``.

This is a port of ``mixle.bstats.dpm``. The variational math is preserved
exactly; only the surrounding object protocol is adapted to mixle.stats:
``SequenceEncodableProbabilityDistribution`` / ``SequenceEncodableStatisticAccumulator``
/ ``StatisticAccumulatorFactory`` / ``ParameterEstimator``, a ``DataSequenceEncoder``
(delegated to ``components[0]``), the two-argument ``estimate(nobs, suff_stat)``
signature, and ``seq_initialize`` on the accumulator.
"""

import copy
import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import maxrandint
from mixle.stats.compute._sampling import scatter_component_draws
from mixle.stats.compute.mixture_evidence import normalize_mixture_log_scores
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.compute.posterior import CategoricalLatentPosterior, ImpossiblePosteriorError
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.utils.special import betaln, digamma

default_prior = GammaDistribution(2, 1)
_SIMPLEX_RTOL = 1.0e-10
_SIMPLEX_ATOL = 1.0e-12


def _type_id(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return "%s.%s" % (cls.__module__, cls.__qualname__)


def _prior_structure(prior: Any) -> tuple[Any, ...]:
    if prior is None:
        return ("none",)
    if isinstance(prior, (tuple, list)):
        return ("sequence", len(prior), *(_prior_structure(child) for child in prior))
    return ("leaf", _type_id(prior))


def _validated_hyperprior(prior: Any) -> GammaDistribution | None:
    if prior is not None and not isinstance(prior, GammaDistribution):
        raise TypeError("Dirichlet-process concentration prior must be GammaDistribution or None.")
    return prior


def _validated_concentration(value: Any) -> float:
    if np.ndim(value) != 0:
        raise ValueError("Dirichlet-process concentration must be a finite positive scalar.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet-process concentration must be a finite positive scalar.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("Dirichlet-process concentration must be a finite positive scalar.")
    return result


def _validated_weights(value: Any, component_count: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet-process weights must be numeric.") from exc
    if weights.shape != (component_count,):
        raise ValueError(
            "Dirichlet-process weights must have exact shape (%d,)." % component_count
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Dirichlet-process weights must be finite and non-negative.")
    if not np.isclose(
        float(weights.sum()),
        1.0,
        rtol=_SIMPLEX_RTOL,
        atol=_SIMPLEX_ATOL,
    ):
        raise ValueError("Dirichlet-process weights must sum to one.")
    return weights.copy()


def _validated_sticks(value: Any, component_count: int) -> np.ndarray:
    try:
        sticks = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet-process stick parameters must be numeric.") from exc
    if sticks.shape != (component_count, 2):
        raise ValueError(
            "Dirichlet-process stick parameters must have exact shape (%d, 2)."
            % component_count
        )
    if np.any(~np.isfinite(sticks)) or np.any(sticks <= 0.0):
        raise ValueError("Dirichlet-process stick parameters must be finite and positive.")
    return sticks.copy()


def _validated_component_sequence(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("Dirichlet-process %s must be a sequence." % name)
    result = list(value)
    if not result:
        raise ValueError("Dirichlet-process mixtures require at least one component.")
    return result


def _compatible_component_encoder(components: Sequence[Any]) -> DataSequenceEncoder:
    encoders: list[DataSequenceEncoder] = []
    for component in components:
        if not callable(getattr(component, "dist_to_encoder", None)):
            raise TypeError("Every Dirichlet-process component must provide dist_to_encoder().")
        encoder = component.dist_to_encoder()
        if not isinstance(encoder, DataSequenceEncoder):
            raise TypeError("Every Dirichlet-process component encoder must satisfy DataSequenceEncoder.")
        encoders.append(encoder)
    reference = encoders[0]
    for index, encoder in enumerate(encoders[1:], start=1):
        try:
            compatible = reference == encoder and encoder == reference
        except Exception as exc:
            raise ValueError(
                "Dirichlet-process component encoder compatibility check failed at component %d."
                % index
            ) from exc
        if not compatible:
            raise ValueError(
                "Dirichlet-process components must use compatible encoders; component %d differs."
                % index
            )
    return reference


def _component_structure_fingerprint(components: Sequence[Any]) -> tuple[Any, ...]:
    encoder = _compatible_component_encoder(components)
    return (
        len(components),
        _type_id(encoder),
        str(encoder),
        tuple(_type_id(component) for component in components),
        tuple(_prior_structure(component.get_prior()) for component in components),
    )


def _validated_component_priors(components: Sequence[Any], priors: Any) -> list[Any]:
    if isinstance(priors, (str, bytes)) or not isinstance(priors, Sequence):
        raise TypeError("Dirichlet-process component_priors must be a sequence.")
    result = list(priors)
    if len(result) != len(components):
        raise ValueError(
            "Dirichlet-process component_priors must contain exactly one entry per component."
        )
    for index, (component, prior) in enumerate(zip(components, result)):
        if _prior_structure(component.get_prior()) != _prior_structure(prior):
            raise ValueError(
                "Dirichlet-process component prior structure differs at component %d." % index
            )
    return result


def _validated_observation_weight(value: Any) -> float:
    if np.ndim(value) != 0:
        raise ValueError("Dirichlet-process observation weight must be a finite non-negative scalar.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Dirichlet-process observation weight must be a finite non-negative scalar."
        ) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("Dirichlet-process observation weight must be a finite non-negative scalar.")
    return result


def _validated_observation_weights(value: Any, row_count: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet-process observation weights must be numeric.") from exc
    if result.shape != (row_count,):
        raise ValueError(
            "Dirichlet-process observation weights must have exact shape (%d,)." % row_count
        )
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("Dirichlet-process observation weights must be finite and non-negative.")
    return result


def _weighted_scores(component_scores: Any, log_weights: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.asarray(component_scores, dtype=np.float64).copy()
    positive = weights > 0.0
    scores[..., positive] += log_weights[positive]
    scores[..., ~positive] = -np.inf
    return scores


def _validated_dpm_statistics(
    value: Any,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray, float, float, tuple[Any, ...]]:
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise ValueError("Dirichlet-process sufficient statistics must be a five-item tuple.")
    try:
        component_counts = np.asarray(value[0], dtype=np.float64)
        beta_counts = np.asarray(value[1], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet-process count statistics must be numeric.") from exc
    if component_counts.shape != (component_count,):
        raise ValueError(
            "Dirichlet-process component counts must have exact shape (%d,)."
            % component_count
        )
    if beta_counts.shape != (component_count, 2):
        raise ValueError(
            "Dirichlet-process beta counts must have exact shape (%d, 2)."
            % component_count
        )
    if np.any(~np.isfinite(component_counts)) or np.any(component_counts < 0.0):
        raise ValueError("Dirichlet-process component counts must be finite and non-negative.")
    if np.any(~np.isfinite(beta_counts)) or np.any(beta_counts < 0.0):
        raise ValueError("Dirichlet-process beta counts must be finite and non-negative.")
    if not np.allclose(
        beta_counts[:, 0],
        component_counts,
        rtol=_SIMPLEX_RTOL,
        atol=_SIMPLEX_ATOL,
    ):
        raise ValueError("Dirichlet-process beta success counts must match component counts.")
    expected_remaining = component_counts.sum() - np.cumsum(component_counts)
    expected_remaining[-1] = 0.0
    if not np.allclose(
        beta_counts[:, 1],
        expected_remaining,
        rtol=_SIMPLEX_RTOL,
        atol=_SIMPLEX_ATOL,
    ):
        raise ValueError(
            "Dirichlet-process beta failure counts violate remaining-stick conservation."
        )
    concentration = _validated_concentration(value[2])
    if np.ndim(value[3]) != 0:
        raise ValueError("Dirichlet-process previous log-stick statistic must be finite.")
    try:
        previous_log_stick = float(value[3])
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet-process previous log-stick statistic must be finite.") from exc
    if not np.isfinite(previous_log_stick):
        raise ValueError("Dirichlet-process previous log-stick statistic must be finite.")
    if isinstance(value[4], (str, bytes)) or not isinstance(value[4], Sequence):
        raise TypeError("Dirichlet-process component statistics must be a sequence.")
    component_statistics = tuple(value[4])
    if len(component_statistics) != component_count:
        raise ValueError(
            "Dirichlet-process component statistics must match the component count."
        )
    return (
        component_counts.copy(),
        beta_counts.copy(),
        concentration,
        previous_log_stick,
        component_statistics,
    )


def _prior_cross_entropy(p: Any, q: Any) -> float:
    """Cross entropy between two component priors that may factor over children.

    For a leaf family ``get_prior()`` returns a single distribution and this is
    just ``p.cross_entropy(q)``. For structured families (Composite, Sequence)
    ``get_prior()`` returns a tuple/list of per-child priors; the joint prior
    factors over the children, so the cross entropy is the sum over the matching
    children. ``p`` and ``q`` always share the same nested structure here.
    """
    if p is None and q is None:
        return 0.0
    if (p is None) != (q is None):
        raise ValueError("Variational and reference prior trees must have identical structure.")
    p_sequence = isinstance(p, (tuple, list))
    q_sequence = isinstance(q, (tuple, list))
    if p_sequence != q_sequence:
        raise TypeError("Variational and reference prior trees must have identical structure.")
    if p_sequence:
        if len(p) != len(q):
            raise ValueError("Variational and reference prior trees must have identical arity.")
        return float(sum(_prior_cross_entropy(pc, qc) for pc, qc in zip(p, q)))
    return float(p.cross_entropy(q))


def _prior_entropy(p: Any) -> float:
    """Entropy of a component prior that may factor over children.

    Mirrors :func:`_prior_cross_entropy`: leaf priors expose ``entropy()``
    directly, while structured priors are tuples/lists whose joint entropy is
    the sum over the independent children.
    """
    if p is None:
        return 0.0
    if isinstance(p, (tuple, list)):
        return float(sum(_prior_entropy(pc) for pc in p))
    return float(p.entropy())


def cbg(x: float, s1: float, s2: float) -> float:
    """Log-density of a compound Beta-Gamma stick fraction: x = 1 - exp(-y)
    with y ~ Exponential(alpha) and alpha ~ Gamma(s1, 1/s2), marginalized
    over alpha.

    Args:
        x (float): Stick fraction in (0, 1).
        s1 (float): Gamma shape of the concentration hyper-prior.
        s2 (float): Gamma rate of the concentration hyper-prior.

    Returns:
        Log-density at x.

    """
    return np.log(s1) + s1 * np.log(s2) - (s1 + 1) * np.log(s2 - np.log1p(-x)) - np.log1p(-x)


def _expected_log_stick_weights(gam: np.ndarray) -> np.ndarray:
    """Return E_q[log pi_k] for a truncated stick-breaking variational state.

    The final component consumes the remaining stick, so it has no v_K term:
    log pi_K = sum_{j<K} log(1 - v_j).  ``gam`` keeps a final row for shape
    compatibility, but that row is ignored by the stick prior.
    """
    num_components = gam.shape[0]
    if num_components == 1:
        return np.zeros(1, dtype=float)

    gams = gam[:, 0] + gam[:, 1]
    exp_v = digamma(gam[:, 0]) - digamma(gams)
    exp_nv = digamma(gam[:, 1]) - digamma(gams)

    rv = np.empty(num_components, dtype=float)
    remaining_log = 0.0
    for i in range(num_components - 1):
        rv[i] = remaining_log + exp_v[i]
        remaining_log += exp_nv[i]
    rv[-1] = remaining_log
    return rv


class DirichletProcessMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Truncated Dirichlet process mixture with stick-breaking weights w over
    K component distributions, carrying the variational Beta posteriors."""

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        w: np.ndarray | list[float],
        a: float,
        g: np.ndarray,
        component_priors: Sequence[SequenceEncodableProbabilityDistribution],
        name: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = default_prior,
    ) -> None:
        """Create a finite direct-assignment Dirichlet-process mixture approximation.

        Args:
            components: List of K component distributions (each carrying its
                own posterior as its prior, i.e. ``component.get_prior()``).
            w: Length-K mixture weight vector.
            a (float): Concentration parameter alpha (point estimate).
            g (np.ndarray): (K, 2) array of variational Beta posterior
                parameters gamma_k on the stick fractions.
            component_priors: List of the K component priors used as the
                variational factors q(theta_k) in the ELBO.
            name (Optional[str]): Optional distribution name.
            prior: Gamma hyper-prior (or hyper-posterior) on alpha.

        """
        checked_components = _validated_component_sequence(components, "components")
        checked_weights = _validated_weights(w, len(checked_components))
        checked_concentration = _validated_concentration(a)
        checked_sticks = _validated_sticks(g, len(checked_components))
        checked_component_priors = copy.deepcopy(
            _validated_component_priors(
                checked_components,
                component_priors,
            )
        )
        checked_prior = _validated_hyperprior(prior)
        fingerprint = _component_structure_fingerprint(checked_components)

        self.components = checked_components
        self.max_components = len(checked_components)
        self.num_components = len(checked_components)
        self.w = checked_weights
        self.v = self.w
        self.a = checked_concentration
        with np.errstate(divide="ignore"):
            self.log_w = np.log(self.w)
        self.expected_log_nw = self.log_w[-1]
        self.g = checked_sticks
        self.component_priors = checked_component_priors
        self.prior = checked_prior
        self._structure_fingerprint = fingerprint
        self.name = name

    def __str__(self) -> str:
        return "DirichletProcessMixtureDistribution([%s], %s, %s, %s, [%s], name=%s, prior=%s)" % (
            ",".join([str(u) for u in self.components]),
            repr(list(self.v)),
            repr(self.a),
            repr(self.g.tolist()),
            ",".join(str(prior) for prior in self.component_priors),
            repr(self.name),
            str(self.prior),
        )

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the Gamma hyper-posterior on the concentration alpha."""
        return self.prior

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Set the Gamma hyper-prior (or hyper-posterior) on alpha."""
        self.prior = _validated_hyperprior(prior)

    def get_parameters(self) -> tuple[float, np.ndarray, list[Any]]:
        """Return concentration, weights, and independent component snapshots."""
        return self.a, self.v.copy(), copy.deepcopy(self.components)

    def _assert_structure(self) -> None:
        """Reject externally mutated component geometry before it reaches inference."""
        if (
            self.num_components != len(self.components)
            or self.max_components != self.num_components
        ):
            raise RuntimeError(
                "Dirichlet-process component structure changed after model construction."
            )
        current = _component_structure_fingerprint(self.components)
        if current != self._structure_fingerprint:
            raise RuntimeError(
                "Dirichlet-process component structure changed after model construction."
            )
        _validated_component_priors(self.components, self.component_priors)
        checked_weights = _validated_weights(self.w, self.num_components)
        _validated_sticks(self.g, self.num_components)
        _validated_concentration(self.a)
        _validated_hyperprior(self.prior)
        if np.asarray(self.v).shape != checked_weights.shape or not np.array_equal(
            self.v,
            checked_weights,
        ):
            raise RuntimeError("Dirichlet-process cached weights are inconsistent.")
        with np.errstate(divide="ignore"):
            expected_log_weights = np.log(checked_weights)
        if (
            np.asarray(self.log_w).shape != expected_log_weights.shape
            or not np.array_equal(self.log_w, expected_log_weights)
            or self.expected_log_nw != self.log_w[-1]
        ):
            raise RuntimeError("Dirichlet-process cached log weights are inconsistent.")

    def set_parameters(self, params: tuple[float, np.ndarray, Sequence[Any]]) -> None:
        """Atomically set concentration, weights, and component snapshots.

        Args:
            params: Tuple ``(alpha, weights, components)`` matching
                :meth:`get_parameters`.

        """
        if not isinstance(params, (tuple, list)) or len(params) != 3:
            raise ValueError("Dirichlet-process parameters must be a three-item tuple.")
        self._assert_structure()
        a, w, components = params
        checked_a = _validated_concentration(a)
        checked_w = _validated_weights(w, self.num_components)
        checked_components = _validated_component_sequence(components, "components")
        if len(checked_components) != self.num_components:
            raise ValueError(
                "Dirichlet-process replacement components must match the component count."
            )
        checked_components = copy.deepcopy(checked_components)
        if _component_structure_fingerprint(checked_components) != self._structure_fingerprint:
            raise ValueError("Dirichlet-process replacement components changed model structure.")
        _validated_component_priors(checked_components, self.component_priors)

        self.components = checked_components
        self.a = checked_a
        self.w = checked_w
        self.v = self.w
        with np.errstate(divide="ignore"):
            self.log_w = np.log(self.w)
        self.expected_log_nw = self.log_w[-1]

    def density(self, x: Any) -> float:
        """Density of the mixture at observation x; see log_density()."""
        return np.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Mixture log-density log sum_k w_k p(x | theta_k) at observation x."""
        self._assert_structure()
        scores = _weighted_scores(
            [u.log_density(x) for u in self.components],
            self.log_w,
            self.w,
        )
        return float(normalize_mixture_log_scores(scores[None, :]).log_evidence[0])

    def expected_log_density(self, x: Any) -> float:
        """Mixture log-density with each component's plug-in log-density
        replaced by its variational expectation E_q[log p(x | theta_k)]."""
        self._assert_structure()
        scores = _weighted_scores(
            [u.expected_log_density(x) for u in self.components],
            self.log_w,
            self.w,
        )
        return float(normalize_mixture_log_scores(scores[None, :]).log_evidence[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Vectorized log-density at sequence-encoded input x."""
        self._assert_structure()
        ll_mat = _weighted_scores(
            np.asarray([u.seq_log_density(x) for u in self.components]).T,
            self.log_w,
            self.w,
        )
        return normalize_mixture_log_scores(ll_mat).log_evidence

    def posterior(self, x: Any) -> np.ndarray:
        """Return the component posterior ``p(z = k | x)`` at a single observation.

        This is the plug-in mixture posterior consistent with :meth:`log_density`:
        ``softmax_k( log p(x | theta_k) + log w_k )``. An observation with no support under any
        component returns an all-zero responsibility vector. Returns a
        length-K array summing to one for possible evidence and zero otherwise.
        """
        self._assert_structure()
        comp_log_density = _weighted_scores(
            [u.log_density(x) for u in self.components],
            self.log_w,
            self.w,
        )
        return normalize_mixture_log_scores(comp_log_density[None, :]).responsibilities[0]

    def seq_posterior(self, x: Any) -> np.ndarray:
        """Vectorized component posterior over a sequence-encoded input.

        Returns an ``(sz, K)`` array whose row ``i`` is the plug-in posterior ``p(z = k | x_i)``
        (see :meth:`posterior`); rows for observations with no support under any
        component receive zero responsibility.
        """
        self._assert_structure()
        ll_mat = _weighted_scores(
            np.asarray([u.seq_log_density(x) for u in self.components]).T,
            self.log_w,
            self.w,
        )
        return normalize_mixture_log_scores(ll_mat).responsibilities

    def seq_local_elbo(self, x: Any) -> np.ndarray:
        """Per-observation local ELBO contributions.

        For each observation i this returns

            sum_k phi_ik * ( E_q[log p(z_i = k | v)] + E_q[log p(x_i | theta_k)] - log phi_ik )

        where phi_i is the optimal variational assignment for x_i. The global
        (data-independent) ELBO terms are returned by
        ``DirichletProcessMixtureEstimator.model_log_density``.
        """
        self._assert_structure()
        component_scores = np.asarray(
            [u.seq_expected_log_density(x) for u in self.components]
        ).T
        weighted = component_scores + _expected_log_stick_weights(self.g)
        return normalize_mixture_log_scores(weighted).log_evidence

    def seq_expected_log_density(self, x: Any) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x."""
        self._assert_structure()
        ll = _weighted_scores(
            np.asarray([u.seq_expected_log_density(x) for u in self.components]).T,
            self.log_w,
            self.w,
        )
        return normalize_mixture_log_scores(ll).log_evidence

    def latent_posterior(self, x: Sequence[Any]) -> CategoricalLatentPosterior:
        """Return a typed latent posterior or reject zero-probability evidence."""
        encoded = self.dist_to_encoder().seq_encode(list(x))
        responsibilities = self.seq_posterior(encoded)
        impossible = np.flatnonzero(responsibilities.sum(axis=1) == 0.0)
        if impossible.size:
            raise ImpossiblePosteriorError(
                "Dirichlet-process mixture evidence has zero probability at rows %s."
                % tuple(int(index) for index in impossible)
            )
        return CategoricalLatentPosterior(responsibilities)

    def density_semantics(self):
        """Return exact-or-approximate density semantics joined from component models."""
        from mixle.stats.compute.pdist import DensitySemantics, join_density_semantics

        children = list(self.components)
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics")]
        return join_density_semantics(sems) if sems else DensitySemantics.EXACT

    def sampler(self, seed: int | None = None) -> "DirichletProcessMixtureSampler":
        """Create a DirichletProcessMixtureSampler for this distribution."""
        return DirichletProcessMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "DirichletProcessMixtureEstimator":
        """Create a DirichletProcessMixtureEstimator from this distribution's components."""
        if pseudo_count is not None:
            raise ValueError("Dirichlet-process pseudo-count regularization is not implemented.")
        return DirichletProcessMixtureEstimator(
            [u.estimator() for u in self.components],
            name=self.name,
            prior=self.prior,
        )

    def dist_to_encoder(self) -> "DirichletProcessMixtureDataEncoder":
        """Returns a DirichletProcessMixtureDataEncoder delegating to the components."""
        self._assert_structure()
        return DirichletProcessMixtureDataEncoder(self.components[0].dist_to_encoder())


class DirichletProcessMixtureSampler(DistributionSampler):
    """Draws samples from a DirichletProcessMixtureDistribution."""

    def __init__(self, dist: DirichletProcessMixtureDistribution, seed: int | None = None) -> None:
        """Create a sampler for the finite DP-mixture approximation.

        Args:
            dist (DirichletProcessMixtureDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        rng_loc = RandomState(seed)

        self.rng = RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw size samples (a single observation when size is None).

        A component is chosen with probability w_k and an observation is drawn from that component.
        With ``batched=True`` (default) component draws are grouped and scattered -- bit-identical to
        the per-draw loop (``batched=False``) but far faster, since each component sampler owns an
        independent RNG.

        Args:
            size (Optional[int]): Number of samples to draw.
            batched (bool): Vectorize component draws (default); set False for the per-draw loop.

        Returns:
            A single observation if size is None, else a list of size observations.

        """
        self.dist._assert_structure()
        if size is not None:
            if isinstance(size, (bool, np.bool_)):
                raise TypeError("Dirichlet-process sample size must be a non-negative integer.")
            try:
                size = operator.index(size)
            except TypeError as exc:
                raise TypeError(
                    "Dirichlet-process sample size must be a non-negative integer."
                ) from exc
            if size < 0:
                raise ValueError("Dirichlet-process sample size must be non-negative.")
        comp_state = self.rng.choice(
            range(0, len(self.dist.w)),
            size=size,
            replace=True,
            p=self.dist.w,
        )

        if size is None:
            return self.comp_samplers[comp_state].sample()
        if not batched:
            return [self.comp_samplers[i].sample() for i in comp_state]
        return scatter_component_draws(comp_state, self.comp_samplers, int(size))


class DirichletProcessMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulates DPM sufficient statistics: expected component counts,
    Beta stick-fraction counts, and each component's weighted statistics."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        """Create an accumulator for DP-mixture sufficient statistics.

        Args:
            accumulators: List of K component accumulators.
            keys (Tuple[Optional[str], Optional[str]]): Keys for sharing the
            stick-fraction counts and the component accumulators.

        """
        self.accumulators = list(accumulators)
        if not self.accumulators:
            raise ValueError("Dirichlet-process accumulators require at least one component.")
        if (
            not isinstance(keys, tuple)
            or len(keys) != 2
            or any(key is not None and not isinstance(key, str) for key in keys)
        ):
            raise TypeError("Dirichlet-process accumulator keys must be a two-item string/None tuple.")
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.beta_counts = np.zeros((self.num_components, 2), dtype=float)
        self.prev_nw = np.log(0.5) * (self.num_components - 1)
        self.a = 1.0
        self.weight_key = keys[0]
        self.comp_key = keys[1]

        self._init_rng: bool = False
        self._acc_rng: list[RandomState] | None = None
        self._w_rng: RandomState | None = None

    def update(self, x: Any, weight: float, estimate: DirichletProcessMixtureDistribution) -> None:
        """Accumulate the VB E-step statistics for observation x.

        Computes the optimal variational assignment phi from the current
        estimate's expected log-densities and weights, then adds the weighted
        phi to the component and stick-fraction counts and pushes phi-weighted
        updates into the component accumulators.
        """
        checked_weight = _validated_observation_weight(weight)
        if not isinstance(estimate, DirichletProcessMixtureDistribution):
            raise TypeError("Dirichlet-process accumulation requires a matching model estimate.")
        estimate._assert_structure()
        if estimate.num_components != self.num_components:
            raise ValueError("Dirichlet-process accumulator and estimate component counts differ.")
        exp_ll = _weighted_scores(
            np.asarray(
                [
                    estimate.components[i].expected_log_density(x)
                    for i in range(self.num_components)
                ]
            ),
            estimate.log_w,
            estimate.w,
        )
        phi = normalize_mixture_log_scores(exp_ll[None, :]).responsibilities[0]
        weighted_phi = phi * checked_weight
        possible = float(phi.sum()) > 0.0
        remaining = (
            np.maximum(0.0, 1.0 - np.cumsum(phi))
            * checked_weight
            * float(possible)
        )
        remaining[-1] = 0.0

        self.comp_counts += weighted_phi
        self.beta_counts[:, 0] += weighted_phi
        self.beta_counts[:, 1] += remaining
        self.a = estimate.a
        self.prev_nw = float(_expected_log_stick_weights(estimate.g)[-1])

        for i in range(self.num_components):
            self.accumulators[i].update(x, weighted_phi[i], estimate.components[i])

    def seq_update(self, x: Any, weights: np.ndarray, estimate: DirichletProcessMixtureDistribution) -> None:
        """Vectorized update() on sequence-encoded data."""
        if not isinstance(estimate, DirichletProcessMixtureDistribution):
            raise TypeError("Dirichlet-process accumulation requires a matching model estimate.")
        estimate._assert_structure()
        if estimate.num_components != self.num_components:
            raise ValueError("Dirichlet-process accumulator and estimate component counts differ.")
        exp_ll = _weighted_scores(
            np.asarray([u.seq_expected_log_density(x) for u in estimate.components]).T,
            estimate.log_w,
            estimate.w,
        )
        normalized = normalize_mixture_log_scores(exp_ll)
        phi = normalized.responsibilities
        checked_weights = _validated_observation_weights(weights, phi.shape[0])
        weighted_phi = phi * checked_weights[:, None]
        possible = phi.sum(axis=1) > 0.0
        remaining = (
            np.maximum(0.0, 1.0 - np.cumsum(phi, axis=1))
            * possible[:, None]
        )
        remaining[:, -1] = 0.0
        weighted_remaining = remaining * checked_weights[:, None]

        self.comp_counts += weighted_phi.sum(axis=0)
        self.beta_counts[:, 0] += weighted_phi.sum(axis=0)
        self.beta_counts[:, 1] += weighted_remaining.sum(axis=0)
        self.a = estimate.a
        self.prev_nw = float(_expected_log_stick_weights(estimate.g)[-1])

        for i in range(self.num_components):
            self.accumulators[i].seq_update(
                x,
                weighted_phi[:, i],
                estimate.components[i],
            )

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize child RandomState objects for consistent (seq_)initialize."""
        seeds = rng.randint(2**31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initialize with a random Dirichlet assignment of observation x."""
        checked_weight = _validated_observation_weight(weight)
        if not isinstance(rng, RandomState):
            raise TypeError("Dirichlet-process initialization requires numpy.random.RandomState.")
        if not self._init_rng:
            self._rng_initialize(rng)

        p = self._w_rng.dirichlet(np.ones(self.num_components))
        weighted_p = p * checked_weight
        remaining = np.maximum(0.0, 1.0 - np.cumsum(p)) * checked_weight
        remaining[-1] = 0.0

        self.comp_counts += weighted_p
        self.beta_counts[:, 0] += weighted_p
        self.beta_counts[:, 1] += remaining

        for i in range(self.num_components):
            self.accumulators[i].initialize(x, weighted_p[i], self._acc_rng[i])

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialize() with random Dirichlet assignments."""
        if not isinstance(rng, RandomState):
            raise TypeError("Dirichlet-process initialization requires numpy.random.RandomState.")
        row_count = self.accumulators[0].acc_to_encoder().row_count(x)
        checked_weights = _validated_observation_weights(weights, row_count)
        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(checked_weights)
        p = self._w_rng.dirichlet(np.ones(self.num_components), size=sz)
        weighted_p = p * checked_weights[:, None]
        remaining = np.maximum(0.0, 1.0 - np.cumsum(p, axis=1))
        remaining[:, -1] = 0.0
        weighted_remaining = remaining * checked_weights[:, None]

        self.comp_counts += weighted_p.sum(axis=0)
        self.beta_counts[:, 0] += weighted_p.sum(axis=0)
        self.beta_counts[:, 1] += weighted_remaining.sum(axis=0)

        for i in range(self.num_components):
            self.accumulators[i].seq_initialize(x, weighted_p[:, i], self._acc_rng[i])

    def combine(self, suff_stat: tuple) -> "DirichletProcessMixtureAccumulator":
        """Add another accumulator's sufficient-statistic value into this one."""
        comp_counts, beta_counts, concentration, prev_nw, component_stats = (
            _validated_dpm_statistics(suff_stat, self.num_components)
        )
        if self.comp_counts.sum() > 0.0 and comp_counts.sum() > 0.0:
            if not np.isclose(self.a, concentration, rtol=_SIMPLEX_RTOL, atol=_SIMPLEX_ATOL):
                raise ValueError(
                    "Cannot merge Dirichlet-process statistics from different concentrations."
                )
            if not np.isclose(
                self.prev_nw,
                prev_nw,
                rtol=_SIMPLEX_RTOL,
                atol=_SIMPLEX_ATOL,
            ):
                raise ValueError(
                    "Cannot merge Dirichlet-process statistics from different stick states."
                )
        new_accumulators = copy.deepcopy(self.accumulators)
        for i in range(self.num_components):
            new_accumulators[i].combine(component_stats[i])
        self.comp_counts += comp_counts
        self.beta_counts += beta_counts
        self.a = concentration
        self.prev_nw = prev_nw
        self.accumulators = new_accumulators
        return self

    def scale(self, c: float) -> "DirichletProcessMixtureAccumulator":
        """Scale linear DP mixture sufficient statistics while preserving metadata."""
        # Scale only the linear count statistics (and the component accumulators); ``a`` (the alpha
        # hyper-posterior) and ``prev_nw`` are non-linear scalar metadata that must stay untouched --
        # the inherited default would multiply them and corrupt the state.
        checked_scale = _validated_observation_weight(c)
        self.comp_counts *= checked_scale
        self.beta_counts *= checked_scale
        for u in self.accumulators:
            u.scale(checked_scale)
        return self

    def value(self) -> tuple:
        """Returns (comp_counts, beta_counts, alpha, prev_nw, component values)."""
        return (
            self.comp_counts.copy(),
            self.beta_counts.copy(),
            self.a,
            self.prev_nw,
            tuple(u.value() for u in self.accumulators),
        )

    def from_value(self, x: tuple) -> "DirichletProcessMixtureAccumulator":
        """Set the sufficient statistics from a value() tuple."""
        comp_counts, beta_counts, concentration, prev_nw, component_stats = (
            _validated_dpm_statistics(x, self.num_components)
        )
        new_accumulators = copy.deepcopy(self.accumulators)
        for i in range(self.num_components):
            new_accumulators[i].from_value(component_stats[i])
        self.comp_counts = comp_counts
        self.beta_counts = beta_counts
        self.a = concentration
        self.prev_nw = prev_nw
        self.accumulators = new_accumulators
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator's keyed statistics into a shared dict."""
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.beta_counts
            else:
                stats_dict[self.weight_key] = self.beta_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics with the pooled keyed values."""
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.beta_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "DirichletProcessMixtureDataEncoder":
        """Returns a DirichletProcessMixtureDataEncoder delegating to the components."""
        return DirichletProcessMixtureDataEncoder(self.accumulators[0].acc_to_encoder())


class DirichletProcessMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for DP-mixture sufficient-statistic accumulators."""

    def __init__(
        self, factories: Sequence[StatisticAccumulatorFactory], dim: int, keys: tuple[str | None, str | None]
    ) -> None:
        """Create a DP-mixture accumulator factory.

        Args:
            factories: List of K component accumulator factories.
            dim (int): Number of components K.
            keys: Keys passed to created accumulators.

        """
        self.factories = list(factories)
        if not self.factories:
            raise ValueError("Dirichlet-process factories require at least one component.")
        if isinstance(dim, (bool, np.bool_)):
            raise TypeError("Dirichlet-process factory dimension must be a positive integer.")
        try:
            checked_dim = operator.index(dim)
        except TypeError as exc:
            raise TypeError(
                "Dirichlet-process factory dimension must be a positive integer."
            ) from exc
        if checked_dim <= 0 or checked_dim != len(self.factories):
            raise ValueError(
                "Dirichlet-process factory dimension must match its non-empty factory list."
            )
        if (
            not isinstance(keys, tuple)
            or len(keys) != 2
            or any(key is not None and not isinstance(key, str) for key in keys)
        ):
            raise TypeError("Dirichlet-process factory keys must be a two-item string/None tuple.")
        self.dim = int(checked_dim)
        self.keys = keys

    def make(self) -> "DirichletProcessMixtureAccumulator":
        """Returns a new DirichletProcessMixtureAccumulator."""
        return DirichletProcessMixtureAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class DirichletProcessMixtureEstimator(ParameterEstimator):
    """Estimates a DirichletProcessMixtureDistribution by mean-field variational
    Bayes from accumulated assignment statistics."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        name: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = default_prior,
        pseudo_count: float | None = None,
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        """Create an estimator for the finite DP-mixture approximation.

        Args:
            estimators: List of K component estimators (each carrying its own
                conjugate prior; their ``estimate`` does the conjugate update).
            name (Optional[str]): Name of the estimated distribution.
            prior: Gamma hyper-prior on the concentration alpha.
            pseudo_count (Optional[float]): Accepted for interface parity; not used.
            keys (Tuple[Optional[str], Optional[str]]): Keys for sharing the
                stick-fraction counts and the component accumulators.

        """
        if isinstance(estimators, (str, bytes)) or not isinstance(estimators, Sequence):
            raise TypeError("Dirichlet-process estimators must be a sequence.")
        checked_estimators = list(estimators)
        if not checked_estimators:
            raise ValueError("Dirichlet-process estimators require at least one component.")
        if pseudo_count is not None:
            raise ValueError("Dirichlet-process pseudo-count regularization is not implemented.")
        if (
            not isinstance(keys, tuple)
            or len(keys) != 2
            or any(key is not None and not isinstance(key, str) for key in keys)
        ):
            raise TypeError("Dirichlet-process estimator keys must be a two-item string/None tuple.")
        self.name = name
        self.num_components = len(checked_estimators)
        self.estimators = checked_estimators
        self.keys = keys
        self.prior = _validated_hyperprior(prior)
        self.pseudo_count = None

    def accumulator_factory(self) -> "DirichletProcessMixtureAccumulatorFactory":
        """Returns a DirichletProcessMixtureAccumulatorFactory for this estimator."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return DirichletProcessMixtureAccumulatorFactory(est_factories, self.num_components, self.keys)

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the Gamma hyper-prior on the concentration alpha."""
        return self.prior

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Set the Gamma hyper-prior on the concentration alpha."""
        self.prior = _validated_hyperprior(prior)

    def model_log_density(self, model: DirichletProcessMixtureDistribution) -> float:
        """Data-independent ELBO terms of the variational approximation.

        Combines the cross-entropies of the stick-fraction prior and the
        component priors against their variational posteriors with the entropies
        of those posteriors. Together with
        ``DirichletProcessMixtureDistribution.seq_local_elbo`` this forms the full
        ELBO maximized by the fit driver.
        """
        if not isinstance(model, DirichletProcessMixtureDistribution):
            raise TypeError("Dirichlet-process ELBO requires a matching mixture model.")
        model._assert_structure()
        if model.num_components != self.num_components:
            raise ValueError("Dirichlet-process estimator and model component counts differ.")
        gam = model.g[:-1, :] if model.g.shape[0] > 1 else model.g[:0, :]
        gams = gam[:, 0] + gam[:, 1]
        if self.prior is None:
            if model.prior is not None:
                raise ValueError(
                    "A point-estimated concentration model cannot carry a variational hyper-posterior."
                )
            expected_alpha = model.a
            expected_log_alpha = float(np.log(model.a))
            concentration_term = 0.0
        else:
            if not isinstance(model.prior, GammaDistribution):
                raise ValueError(
                    "Gamma concentration prior requires a Gamma variational hyper-posterior."
                )
            expected_alpha = model.prior.k * model.prior.theta
            expected_log_alpha = float(
                digamma(model.prior.k) + np.log(model.prior.theta)
            )
            concentration_term = float(
                -model.prior.cross_entropy(self.prior) + model.prior.entropy()
            )

        if gam.shape[0] == 0:
            stick_prior_term = 0.0
            stick_entropy = 0.0
        else:
            expected_log_remaining = digamma(gam[:, 1]) - digamma(gams)
            stick_prior_term = float(
                np.sum(expected_log_alpha + (expected_alpha - 1.0) * expected_log_remaining)
            )
            stick_entropy = float(
                np.sum(
                    betaln(gam[:, 0], gam[:, 1])
                    - (gam[:, 0] - 1.0) * digamma(gam[:, 0])
                    - (gam[:, 1] - 1.0) * digamma(gam[:, 1])
                    + (gams - 2.0) * digamma(gams)
                )
            )

        component_term = 0.0
        for i in range(model.max_components):
            posterior = model.components[i].get_prior()
            reference = model.component_priors[i]
            component_term += -_prior_cross_entropy(posterior, reference)
            component_term += _prior_entropy(posterior)

        result = stick_prior_term + stick_entropy + component_term + concentration_term
        if not np.isfinite(result):
            raise ValueError("Dirichlet-process global ELBO terms must be finite.")
        return float(result)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> DirichletProcessMixtureDistribution:
        """Estimate a DirichletProcessMixtureDistribution by one VB M-step.

        Re-estimates each component (whose conjugate update carries its
        posterior forward as its prior), re-sorts components by expected count,
        updates the variational Beta posteriors gamma_k on the stick fractions,
        updates the Gamma hyper-posterior on the concentration alpha (carried as
        the returned distribution's prior), and converts the expected log stick
        fractions into the mixture weights w.

        Args:
            nobs (Optional[float]): Not used. Kept for the stats
                ``ParameterEstimator.estimate(nobs, suff_stat)`` signature.
            suff_stat: Tuple (comp_counts, beta_counts, alpha, prev_nw,
                component suff stats) as returned by
                ``DirichletProcessMixtureAccumulator.value()``.

        Returns:
            Fitted Dirichlet-process mixture approximation.

        """
        num_components = self.num_components
        comp_counts, beta_counts, alpha, prev_nw, comp_suff_stats = (
            _validated_dpm_statistics(suff_stat, num_components)
        )

        component_priors = [u.get_prior() for u in self.estimators]
        components = [self.estimators[i].estimate(comp_counts[i], comp_suff_stats[i]) for i in range(num_components)]

        sidx = np.argsort(-comp_counts)
        comp_counts = comp_counts[sidx]
        beta_counts = beta_counts[sidx, :]
        components = [components[i] for i in sidx]
        component_priors = [component_priors[i] for i in sidx]

        beta_counts[:, 1] = np.sum(beta_counts[:, 0]) - np.cumsum(beta_counts[:, 0])

        if self.prior is None:
            s1 = 0.0
            s2 = 0.0
            hyper_posterior = None
        else:
            s1 = self.prior.k
            s2 = 1 / self.prior.theta

        if num_components <= 1:
            if self.prior is not None:
                new_alpha = s1 / s2
                hyper_posterior = self.prior
            else:
                new_alpha = alpha
        else:
            old_alpha = alpha
            old_gammas = np.copy(beta_counts)
            old_gammas[:, 0] += 1.0
            old_gammas[:, 1] += old_alpha
            expected_log_remaining = _expected_log_stick_weights(old_gammas)[-1]

            gw1 = s1 + num_components - 1.0
            gw2 = s2 - expected_log_remaining
            if (
                not np.isfinite(gw1)
                or not np.isfinite(gw2)
                or gw1 <= 0.0
                or gw2 <= 0.0
            ):
                raise ValueError(
                    "Dirichlet-process concentration update produced invalid Gamma parameters."
                )
            new_alpha = gw1 / gw2
            _validated_concentration(new_alpha)

            if self.prior is not None:
                hyper_posterior = GammaDistribution(gw1, 1 / gw2)

        gammas = np.copy(beta_counts)
        gammas[:, 0] += 1
        gammas[:, 1] += new_alpha

        expected_log_w = _expected_log_stick_weights(gammas)
        w = np.exp(expected_log_w - np.max(expected_log_w))
        w /= w.sum()

        result = DirichletProcessMixtureDistribution(
            components, w, new_alpha, gammas, component_priors, name=self.name, prior=hyper_posterior
        )
        result.fit_metadata = {
            "converged": True,
            "repairs": (),
            "component_order": tuple(int(index) for index in sidx),
            "previous_concentration": alpha,
            "previous_log_remaining_stick": prev_nw,
            "concentration_variational": hyper_posterior is not None,
        }
        return result


class DirichletProcessMixtureDataEncoder(DataSequenceEncoder):
    """Encodes observations with the shared component encoding."""

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        """Create a data encoder for DP-mixture observations.

        Args:
            encoder (DataSequenceEncoder): Encoder for the component distributions.

        """
        if not isinstance(encoder, DataSequenceEncoder):
            raise TypeError("Dirichlet-process data encoder requires a DataSequenceEncoder child.")
        self.encoder = encoder

    def __str__(self) -> str:
        return "DirichletProcessMixtureDataEncoder(" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirichletProcessMixtureDataEncoder):
            return False
        return self.encoder == other.encoder

    def seq_encode(self, x: Sequence[Any]) -> Any:
        """Encode a sequence of observations with the component encoder."""
        return self.encoder.seq_encode(x)

    def row_count(self, x: Any) -> int:
        """Delegate encoded-row accounting to the shared component encoder."""
        return self.encoder.row_count(x)
