"""Homogeneous finite mixtures with stable scoring and EM accumulation.

This module defines ``MixtureDistribution``, ``MixtureEstimator`` and the
sampler, accumulator, factory, and encoder types used by the standard Mixle
estimation loop.

A ``MixtureDistribution`` has density

``p(y) = sum_k p(y | z=k) p(z=k)``.

All components are expected to model the same observation type. Scoring uses
log-sum-exp over component log densities and log weights; impossible rows are
represented as ``-inf`` scores rather than ``NaN``.

"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import (
    BufferedStream,
    QuantizedEnumerationIndex,
    best_first_union,
    bounded_best_first_union_index,
    freeze,
)
from mixle.inference.fisher import Path
from mixle.stats.compute.mixture_evidence import (
    normalize_engine_mixture_log_scores,
    normalize_mixture_log_scores,
    validated_probability_vector,
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
from mixle.stats.compute.posterior import CategoricalLatentPosterior, ImpossiblePosteriorError
from mixle.stats.latent.effective_sample import (
    heal_pooled_statistics,
    require_finite_count_totals,
    restore_accumulator_statistics,
    snapshot_accumulator_statistics,
    validate_effective_sample_mass,
    validated_count_array,
    validated_observation_weight,
    validated_observation_weights,
    validated_statistic_tuple,
    validated_weighted_responsibilities,
)
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.special import digamma

if TYPE_CHECKING:
    from mixle.stats.bayes.dirichlet import DirichletDistribution


def _dirichlet_types() -> tuple[type, type]:
    """Import `(DirichletDistribution, SymmetricDirichletDistribution)` lazily.

    `mixle.stats.bayes.dirichlet` itself transitively imports this module (via
    `mixle.inference` -> `mixle.analysis` -> `mixle.reason` -> `mixle.reason.cross_modal`), so a
    module-level import here would be circular whenever `dirichlet.py` is the entry point of that
    chain (e.g. `import mixle.ppl` / `import mixle_pde` before anything else has warmed
    `mixle.stats.bayes.dirichlet`). Deferring the import to call time breaks the cycle.
    """
    from mixle.stats.bayes.dirichlet import DirichletDistribution
    from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

    return DirichletDistribution, SymmetricDirichletDistribution


T = TypeVar("T")  ### Type of Mixture component data.
T1 = TypeVar("T1")  ### Type of encoded data.
T2 = TypeVar("T2")  ### Type of component suff_stat
key_type = tuple[str, str] | tuple[None, None]


from mixle.inference.fisher import FixedFisherView, SufficientStatisticVectorizer, to_fisher
from mixle.utils.exact import require_exact_bool


def _owned_generative_components(
    components: Sequence[SequenceEncodableProbabilityDistribution],
    label: str,
    *,
    minimum: int = 1,
) -> list[SequenceEncodableProbabilityDistribution]:
    """Return an owned component list, rejecting components that carry no evidence.

    Neutral components are refused: their density does not depend on the observation, so the mixture
    weight on one is unidentifiable and it silently absorbs responsibility from components that do
    explain the data.

    Components that cannot generate are refused too, so a scoring-only leaf cannot silently become a
    generative mixture component.

    Marginalized emissions are the one exception, and they are not really an exception. ``marginalized``
    is the library's own mechanism for observations with absent fields; it reports LIKELIHOOD_FACTOR
    because it contributes 1 where a field is missing rather than a normalized density over the
    augmented space. Refusing that tag outright made the mechanism unusable inside the latent models it
    exists to serve -- marginalizing over missing data inside a mixture or HMM is precisely its purpose.
    Such a component is non-generative only because the missingness rate was left unspecified, and it
    exposes the law it wraps via ``marginalized_core``; that core is validated here in its place, so an
    intrinsically scoring-only leaf is still rejected whether or not it is wrapped.

    The mixture math is unaffected: an absent field contributes the same factor to every component, so
    responsibilities stay proportional and the E-step is unchanged. The composite does not hide the
    result -- density_semantics() already joins to LIKELIHOOD_FACTOR when any child is one.

    Scaled laws are admitted on the same reasoning, by asking rather than assuming. A component that
    differs from a normalized law by a positive constant over its whole support has that constant
    absorbed into its own mixing weight, so responsibilities are unchanged; a zero-mass component wins
    no responsibility at all. Categoricals reach both states in normal use -- a component that drew
    zero weight this iteration, an open-world smoothed emission, a deliberately unnormalized pmap --
    and previously stayed composable only because they mislabelled themselves EXACT (MXR-080-1841).
    A factor declares itself admissible via ``composable_as_component``; absent that declaration a
    likelihood factor is still refused, so an intrinsically scoring-only leaf cannot slip through.
    """
    try:
        owned = list(components)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of probability distributions.") from exc
    if len(owned) < minimum:
        raise ValueError(f"{label} requires at least {minimum} component(s).")
    from mixle.stats.compute.pdist import DensitySemantics

    neutral: list[int] = []
    non_generative: list[int] = []
    for index, component in enumerate(owned):
        if supports(component, Neutral):
            neutral.append(index)
            continue
        # A marginalization wrapper stands in for the law it wraps, which is what must be generative.
        core = getattr(component, "marginalized_core", None)
        subject = component if core is None else core
        if subject.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR:
            admits = getattr(subject, "composable_as_component", None)
            if not (callable(admits) and admits()):
                non_generative.append(index)
    if neutral or non_generative:
        invalid = sorted(set(neutral).union(non_generative))
        raise TypeError(
            f"{label} components must be generative probability laws; likelihood factors found at indices {invalid}."
        )
    return owned


def mixture_prior(
    weight_prior: SequenceEncodableProbabilityDistribution,
    component_priors: Sequence[SequenceEncodableProbabilityDistribution],
) -> tuple[SequenceEncodableProbabilityDistribution, tuple[SequenceEncodableProbabilityDistribution, ...]]:
    """Build the joint mixture prior: a weight prior plus one prior per component.

    Args:
        weight_prior: Prior on the mixture weights (a
            :class:`~mixle.stats.bayes.dirichlet.DirichletDistribution` or
            :class:`~mixle.stats.bayes.symmetric_dirichlet.SymmetricDirichletDistribution`).
        component_priors: Sequence of one conjugate prior per component.

    Returns:
        A ``(weight_prior, tuple(component_priors))`` pair consumed by
        ``MixtureDistribution``/``MixtureEstimator`` ``set_prior``.
    """
    return weight_prior, tuple(component_priors)


def _default_weight_prior(num_components: int) -> DirichletDistribution:
    """Flat (concentration-one) Dirichlet weight prior of the given dimension."""
    DirichletDistribution, _ = _dirichlet_types()
    return DirichletDistribution(np.ones(num_components))


def _component_prior_tuple(
    component_priors: Any, num_components: int
) -> tuple[SequenceEncodableProbabilityDistribution, ...] | None:
    if component_priors is None:
        return None
    if isinstance(component_priors, (list, tuple)):
        rv = tuple(component_priors)
    elif num_components == 1:
        rv = (component_priors,)
    else:
        raise TypeError("mixture component priors must be a sequence.")
    if len(rv) != num_components:
        raise ValueError("expected %d component priors, got %d." % (num_components, len(rv)))
    return rv


def _split_mixture_prior(
    prior: Any, num_components: int
) -> tuple[
    SequenceEncodableProbabilityDistribution | None, tuple[SequenceEncodableProbabilityDistribution, ...] | None
]:
    """Split a joint mixture prior into (weight_prior, component_priors).

    Accepts ``None``, a bare weight prior, a ``(weight_prior, component_priors)`` pair (as
    produced by :func:`mixture_prior`), or a mapping with ``weights``/``components`` entries.
    Returns ``(None, None)`` for ``None`` so the caller can fall back to the MLE path.
    """
    if prior is None:
        return None, None
    if isinstance(prior, Mapping) and (
        "weights" in prior or "weight_prior" in prior or "components" in prior or "component_priors" in prior
    ):
        weight_prior = prior.get("weights", prior.get("weight_prior"))
        component_priors = prior.get("components", prior.get("component_priors"))
        if weight_prior is None:
            weight_prior = _default_weight_prior(num_components)
        return weight_prior, _component_prior_tuple(component_priors, num_components)
    if (
        isinstance(prior, (list, tuple))
        and len(prior) == 2
        and isinstance(prior[1], (list, tuple))
        and not isinstance(prior[0], (list, tuple))
    ):
        return prior[0], _component_prior_tuple(prior[1], num_components)
    return prior, None


def _set_estimator_prior(estimator: ParameterEstimator, prior: Any) -> None:
    """Push a component prior onto a child estimator.

    Stats leaf estimators take their prior via the constructor rather than a ``set_prior``
    method, so this prefers ``set_prior`` when present and otherwise updates the conventional
    ``prior``/``has_conj_prior`` attributes used by the folded leaf estimators.
    """
    set_prior = getattr(estimator, "set_prior", None)
    if callable(set_prior):
        set_prior(prior)
        return
    estimator.prior = prior
    if hasattr(estimator, "has_conj_prior"):
        estimator.has_conj_prior = prior is not None


def _dirichlet_expectations(prior: Any, num_components: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(alpha, E[log w_k])`` for a (symmetric) Dirichlet weight prior, else ``(None, None)``.

    ``E[log w_k] = digamma(alpha_k) - digamma(sum_j alpha_j)`` are the variational weight
    expectations used by ``expected_log_density``.
    """
    DirichletDistribution, SymmetricDirichletDistribution = _dirichlet_types()
    if isinstance(prior, DirichletDistribution):
        alpha = np.asarray(prior.get_parameters(), dtype=float)
        if alpha.shape[0] != num_components:
            # unlike SymmetricDirichletDistribution (broadcasts a scalar to num_components by
            # construction), a full DirichletDistribution's alpha is taken as-is -- a mismatched
            # length would otherwise pass silently here and only fail later, deep inside a numpy
            # broadcast in expected_log_density or mid-optimize() in the M-step, far from the
            # actual mistake (the prior's own arity, not a mixture internals bug).
            raise ValueError(
                "mixture weight prior has %d components but the mixture has %d." % (alpha.shape[0], num_components)
            )
        return alpha, digamma(alpha) - digamma(np.sum(alpha))
    if isinstance(prior, SymmetricDirichletDistribution):
        alpha = np.ones(num_components) * prior.get_parameters()
        return alpha, digamma(alpha) - digamma(np.sum(alpha))
    return None, None


# Tolerance for the "does w sum to one" simplex check below -- numpy's own np.isclose default
# (rtol=1e-05, atol=1e-08), matching the established convention for this same kind of check
# elsewhere (SymmetricDirichletDistribution, DictDirichletDistribution), not a bespoke bound and
# not dirichlet.py's tighter 1e-10/1e-12. A mixture's fitted weights are not guaranteed float64
# precision end to end: MixtureGradientFitState.build's torch softmax renormalization (the
# gradient-fit path, e.g. mixle.inference.gradient_fit.fit_mle with precision="float32") was
# measured landing ~6e-8 from 1.0 -- comfortably inside this tolerance but outside a naive
# 1e-10/1e-12 bound, which would reject a legitimately fitted float32 mixture as off-simplex.
# Plain float64 EM MLE fits (counts / counts.sum()) measured at ~2e-16, also well inside.
class MixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Finite mixture over homogeneous component distributions.

    ``components`` define both the conditional families ``p(x | z=k)`` and the
    observation type accepted by the mixture. ``w`` contains the component
    probabilities and is cached as ``log_w`` for stable scalar and vectorized
    scoring. Zero-weight components are retained for structural compatibility
    but contribute ``-inf`` to log-density calculations.

    Args:
        components: Component distributions. Each component should support the
            same raw observation shape and sequence-encoding contract.
        w: Component weights. The values are interpreted as simplex weights and
            should sum to one.
        name: Optional display name for diagnostics and generated artifacts.
        weights: Alias for ``w``.
        prior: Optional joint mixture prior or weight prior.

    Attributes:
        components: Component distribution objects.
        w: Component weights as a NumPy array.
        zw: Boolean mask for zero-weight components.
        log_w: Log weights, with zero-weight entries represented as ``-inf``.
        num_components: Number of mixture components.
    """

    def compute_capabilities(self):
        """Return compute-backend metadata shared by all mixture components."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(
            engine_ready=intersect_engine_ready(tuple(self.components)), kernel_status="numba_adapter"
        )

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        w: np.ndarray | list[float] = MISSING,
        name: str | None = None,
        weights: np.ndarray | list[float] = MISSING,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        components = _owned_generative_components(components, "MixtureDistribution")
        self.w = validated_probability_vector(w, "MixtureDistribution weights", size=len(components))

        self.zw = self.w == 0.0
        self.log_w = np.log(self.w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.components = components.copy()
        self.num_components = len(components)
        self.name = name
        self.set_prior(prior)

    def compute_declaration(self):
        """Return the symbolic declaration for mixture weights and component statistics."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        children = tuple(declaration_for(d) for d in self.components)
        children = tuple(d for d in children if d is not None)
        return DistributionDeclaration(
            name="mixture",
            distribution_type=type(self),
            parameters=(ParameterSpec("w", constraint="simplex"),),
            statistics=(
                StatisticSpec("component_counts"),
                StatisticSpec("components", kind="tuple"),
            ),
            support="mixture",
            children=children,
            child_roles=tuple("component_%d" % i for i in range(len(children))),
            differentiable=all(child.differentiable for child in children),
        )

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = ",".join([str(u) for u in self.components])
        s2 = repr(list(self.w))
        s3 = repr(self.name)

        return "MixtureDistribution([%s], %s, name=%s)" % (s1, s2, s3)

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the joint mixture prior, or ``None`` for a plain point model.

        When a weight prior is attached the joint prior is the
        ``(weight_prior, tuple(component priors))`` pair produced by
        :func:`mixture_prior`; otherwise ``None``.
        """
        if not self.has_conj_prior:
            return None
        return self.prior, tuple(d.get_prior() for d in self.components)

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a weight prior (and optional per-component priors), caching weight expectations.

        With a (symmetric) Dirichlet weight prior this caches the variational weight
        expectations ``E[log w_k] = digamma(alpha_k) - digamma(sum_j alpha_j)`` used by
        ``expected_log_density``. Component priors, when supplied, are delegated to each
        component via ``component.set_prior``. ``prior=None`` (the default) leaves the
        mixture a plain point model (byte-identical MLE behaviour).
        """
        weight_prior, component_priors = _split_mixture_prior(prior, self.num_components)
        self.prior = weight_prior
        if component_priors is not None:
            for d, p in zip(self.components, component_priors):
                d.set_prior(p)
        self.conj_prior_params, self.expected_nparams = _dirichlet_expectations(self.prior, self.num_components)
        self.has_conj_prior = self.expected_nparams is not None

    def expected_log_density(self, x: T) -> float:
        """Return the optimized local variational log-evidence bound.

        This compatibility name does not compute the generally intractable
        ``E[log(sum_k w_k p_k(x))]``. With conjugate priors it returns
        ``logsumexp_k(E[log w_k] + E[log p_k(x)])``, the optimized local
        variational lower bound. Use :meth:`variational_log_evidence_bound`
        when the semantic distinction matters. Without a conjugate weight
        prior it falls back to the plug-in ``log_density``.
        """
        return self.variational_log_evidence_bound(x)

    def variational_log_evidence_bound(self, x: T) -> float:
        """Return the local variational lower bound for one observation."""
        if not self.has_conj_prior:
            return self.log_density(x)
        cc = self.expected_nparams
        return vec.log_sum(np.asarray([u.expected_log_density(x) for u in self.components]) + cc)

    def seq_expected_log_density(self, x: T1) -> np.ndarray:
        """Return vectorized local variational log-evidence bounds.

        This is the sequence form of :meth:`variational_log_evidence_bound`,
        retained under the framework-wide compatibility name.
        """
        return self.seq_variational_log_evidence_bound(x)

    def seq_variational_log_evidence_bound(self, x: T1) -> np.ndarray:
        """Return local variational lower bounds for encoded observations."""
        if not self.has_conj_prior:
            return self.seq_log_density(x)
        cc = self.expected_nparams
        ll = (
            np.asarray([u.seq_expected_log_density(_component_enc(x, i)) for i, u in enumerate(self.components)]).T + cc
        )
        ml = np.max(ll, axis=1, keepdims=True)
        return np.log(np.sum(np.exp(ll - ml), axis=1)) + ml.flatten()

    def density(self, x: T) -> float:
        """Return the mixture density at one raw observation.

        Args:
            x: Observation accepted by every component family.

        Returns:
            ``exp(log_density(x))``.
        """
        return np.exp(self.log_density(x))

    def density_semantics(self):
        """Return joined density semantics over all mixture components."""
        from mixle.stats.compute.pdist import join_density_semantics

        return join_density_semantics(c.density_semantics() for c in self.components)

    def log_density(self, x: T) -> float:
        """Return the mixture log-density at one raw observation.

        The calculation is ``logsumexp_k(log p_k(x) + log w_k)``. Component
        families are responsible for returning ``-inf`` for observations outside
        support; the mixture combines those values without converting them to
        ``NaN``.

        Args:
            x: Observation accepted by every component family.

        Returns:
            Finite log-density when at least one positive-weight component can
            score the observation, otherwise ``-inf``.
        """
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def conditional(self, observed: dict[int, float]) -> MixtureDistribution:
        """Return the conditional mixture over the unobserved coordinates given ``observed``.

        The conditional of a mixture is *itself a mixture*: for ``sum_k w_k f_k`` observing ``x_o``,

            P(x_u | x_o) = sum_k w'_k f_k(x_u | x_o),  w'_k proportional to w_k f_k.marginal(x_o)(x_o),

        i.e. the component responsibilities are updated by how well each component explains the observed
        coordinates and each component is replaced by its own conditional. Because the result is a full
        ``MixtureDistribution`` you can both score it and ``.sampler(seed).sample()`` from it -- the latter
        is ``given=``-style conditional sampling that first draws a component from the posterior
        responsibilities, then draws the unobserved coordinates from that component's conditional.

        Requires each component to support ``marginal(indices)`` and ``condition(observed)`` (e.g. the
        multivariate Gaussian / Student-t). ``observed`` maps coordinate index to its fixed value.
        """
        obs_idx = sorted(observed)
        if not obs_idx:
            return MixtureDistribution([c.condition({}) for c in self.components], self.w.copy())
        # numeric components (e.g. multivariate Gaussian) take the observed sub-vector as a float array;
        # heterogeneous components (CompositeDistribution of mixed-type fields) take it as a tuple.
        vals = [observed[i] for i in obs_idx]
        try:
            x_o = np.array(vals, dtype=float)
        except (ValueError, TypeError):
            x_o = tuple(vals)
        log_post = np.array(
            [self.log_w[k] + self.components[k].marginal(obs_idx).log_density(x_o) for k in range(self.num_components)]
        )
        normalized = normalize_mixture_log_scores(log_post[None, :])
        if normalized.impossible[0]:
            raise ImpossiblePosteriorError("cannot condition a mixture on zero-probability evidence.")
        new_components = [c.condition(observed) for c in self.components]
        return MixtureDistribution(new_components, normalized.responsibilities[0])

    def component_log_density(self, x: T) -> np.ndarray:
        """Return component-wise log densities for one raw observation.

        Args:
            x: Observation accepted by every component family.

        Returns:
            One log-density per component, before mixture weights are applied.
        """
        return np.asarray([m.log_density(x) for m in self.components], dtype=np.float64)

    def posterior(self, x: T) -> np.ndarray:
        """Return component responsibilities for one raw observation.

        Responsibilities are proportional to ``w[k] * p_k(x)``. An observation
        outside every component's support receives zero responsibility and
        therefore cannot fabricate component statistics.

        Args:
            x: Observation accepted by every component family.

        Returns:
            Responsibility vector over component labels. It sums to one when
            the evidence is possible and is all zero otherwise.
        """
        comp_log_density = np.asarray([m.log_density(x) for m in self.components])
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -np.inf
        return normalize_mixture_log_scores(comp_log_density[None, :]).responsibilities[0]

    def seq_component_log_density(self, x: T1) -> np.ndarray:
        """Return vectorized component log densities for encoded observations.

        ``x`` must be produced by ``MixtureDataEncoder.seq_encode`` or by an
        equivalent component encoder. The output has shape ``(n, k)`` where
        ``n`` is the number of encoded observations and ``k`` is the number of
        mixture components.

        Args:
            x: Encoded observation batch.

        Returns:
            Component log-density matrix before mixture weights are applied.
        """
        enc_data = x
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(_component_enc(enc_data, i))
                if not ll_mat_init:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                    ll_mat_init = True
                ll_mat[:, i] = temp

        return ll_mat

    def seq_log_density(self, x: T1) -> np.ndarray:
        """Return vectorized mixture log densities for encoded observations.

        Each row is evaluated with a row-wise log-sum-exp over component scores
        plus log weights. Rows for which every positive-weight component is
        impossible return ``-inf``.

        Args:
            x: Encoded observation batch.

        Returns:
            One log-density per encoded observation.
        """
        weighted = self.seq_component_log_density(x) + self.log_w
        return normalize_mixture_log_scores(weighted).log_evidence

    def backend_seq_component_log_density(self, x: T1, engine: Any) -> Any:
        """Engine-neutral component log densities for encoded data."""
        from mixle.stats.compute.backend import backend_seq_log_density

        scores = []
        for i in range(self.num_components):
            if self.zw[i]:
                base = backend_seq_log_density(self.components[0], _component_enc(x, 0), engine)
                scores.append(engine.zeros(base.shape) + engine.asarray(-np.inf))
            else:
                scores.append(backend_seq_log_density(self.components[i], _component_enc(x, i), engine))
        return engine.stack(scores, axis=1)

    def backend_seq_log_density(self, x: T1, engine: Any) -> Any:
        """Engine-neutral mixture log-density for encoded data."""
        ll_mat = self.backend_seq_component_log_density(x, engine)
        log_w = engine.asarray(self.log_w)
        return normalize_engine_mixture_log_scores(ll_mat + log_w, engine).log_evidence

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from mixle.stats.compute.gradient import MixtureGradientFitState

        components = [recurse(component, engine, torch, leaves) for component in self.components]
        w_logits = tensor_param(self.w, engine, torch, transform="logits")
        leaves.append(w_logits)
        return MixtureGradientFitState(self, components, w_logits)

    def seq_posterior(self, x: T1) -> np.ndarray:
        """Return vectorized component responsibilities for encoded observations.

        Each row is proportional to ``w[k] * p_k(x_i)``. Rows where all
        positive-weight components are impossible receive zero responsibility,
        matching :meth:`posterior` without inventing latent assignments.

        Args:
            x: Encoded observation batch.

        Returns:
            ``(n, k)`` responsibility matrix. Possible rows sum to one;
            impossible rows sum to zero.
        """
        enc_data = x
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(_component_enc(enc_data, i))
                if not ll_mat_init:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                    ll_mat_init = True

                ll_mat[:, i] = temp
                ll_mat[:, i] += self.log_w[i]

        return normalize_mixture_log_scores(ll_mat).responsibilities

    def latent_posterior(self, x: Sequence[T]) -> CategoricalLatentPosterior:
        """Return the latent posterior ``q(z | x)`` over component labels for raw observations ``x``.

        ``q(z)`` is the exact independent-categorical posterior whose marginals are the EM
        responsibilities. The returned :class:`~mixle.stats.compute.posterior.CategoricalLatentPosterior`
        can ``.marginals()`` (the responsibilities), ``.sample(rng)`` component labels, ``.mode()``
        (the MAP labels), or ``.entropy()``.
        """
        enc = self.dist_to_encoder().seq_encode(list(x))
        responsibilities = self.seq_posterior(enc)
        impossible = np.flatnonzero(responsibilities.sum(axis=1) == 0.0)
        if impossible.size:
            raise ImpossiblePosteriorError(
                "mixture evidence has zero probability at observation rows %s." % tuple(int(i) for i in impossible)
            )
        return CategoricalLatentPosterior(responsibilities)

    def posterior_predictive(self, x: Sequence[T], seed: int | None = None) -> list[Any]:
        """Draw posterior-predictive observations conditioned on ``x``.

        For each observed ``x_i`` the component is sampled from the latent posterior ``q(z_i | x_i)``
        and a *fresh* observation is emitted from that component -- i.e. "given I saw ``x_i``, draw a
        new point from the same mixture component it likely came from". Returns a list the length of
        ``x``. Draws are grouped by component and scattered (vectorized) via the shared sampling
        helper.
        """
        from mixle.stats.compute._sampling import scatter_component_draws

        rng = RandomState(seed)
        z = self.latent_posterior(x).sample(rng)
        comp_samplers = [c.sampler(seed=rng.randint(maxrandint)) for c in self.components]
        return scatter_component_draws(z, comp_samplers, len(z))

    def support_size(self) -> int | None:
        """Upper bound on distinct support points: the sum over components (union <= sum)."""
        total = 0
        for c in self.components:
            s = c.support_size()
            if s is None:
                return None
            total += s
        return total

    def tropical_displacement_bits(self) -> float:
        """``log2(#positive-weight components)`` -- the tropical-vs-marginal cost gap (in bits).

        The marginal ``log p(x) = logsumexp_k (log w_k + log p_k(x))`` is bounded by its largest term
        ``M(x) = max_k (log w_k + log p_k(x))`` via ``M(x) <= log p(x) <= M(x) + log K``, where ``K`` is
        the number of components that can contribute (positive weight). The structural seek bins by the
        tropical cost ``M(x)``; :func:`mixle.enumeration.density_rank.marginal_seek` widens its smear
        window by this many bits so the reported rank bracket provably contains the TRUE marginal rank.
        ``K <= 1`` means the marginal is a single term -> ``0.0`` (the seek is then exact). When the
        component supports are *provably disjoint* every value lands in one component, so ``M(x)`` equals
        the marginal and there is likewise no displacement -> ``0.0`` (the seek is exact and tight).
        """
        k = int(np.count_nonzero(np.asarray(self.w) > 0.0))
        if k <= 1:
            return 0.0
        if self._components_provably_disjoint():
            return 0.0
        return math.log2(k)

    def _components_provably_disjoint(self, probe_cap: int = 2048) -> bool:
        """True only if the positive-weight component supports are pairwise disjoint, by enumeration.

        Sound but conservative: it materializes each component's support into a shared ``seen`` set and
        returns False on the first collision. If any component cannot enumerate (continuous leaf) or the
        combined support exceeds ``probe_cap`` distinct points, it also returns False -- so a ``False``
        never wrongly blocks the safe bracketed seek, it only forgoes the exact disjoint fast path.
        """
        seen: set = set()
        pulled = 0
        for k, comp in enumerate(self.components):
            if self.w[k] <= 0.0:
                continue
            try:
                enumerator = comp.enumerator()
            except Exception:  # noqa: BLE001
                return False
            for value, _lp in enumerator:
                key = freeze(value)
                if key in seen:
                    return False  # shared support point -> components overlap
                seen.add(key)
                pulled += 1
                if pulled > probe_cap:
                    return False  # too large to certify cheaply; assume overlap (conservative)
        return True

    def to_fisher(self, **kwargs):
        """Structural Fisher view for the mixture."""
        if hasattr(self, "components") and hasattr(self, "w"):
            return MixtureFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> MixtureSampler:
        """Return a sampler that draws from the mixture distribution.

        Args:
            seed: Optional ``RandomState`` seed for reproducible draws.

        Returns:
            ``MixtureSampler`` bound to this distribution.

        Raises:
            ValueError: If this mixture is a likelihood factor rather than a normalized law.

        A component may be a scaled law -- an unnormalized categorical, an open-world smoothed
        emission, a component that won zero responsibility -- and ``_owned_generative_components``
        admits those deliberately, because a constant scale cancels in the E-step's responsibility
        normalization. It does NOT cancel in the composite: a mass-0.75 component under weight 0.5
        leaves the mixture with total mass 0.875, and ``density_semantics()`` reports
        ``LIKELIHOOD_FACTOR`` to say so. The sampler used to ignore that and draw as though the
        weights described a law, so the scorer and the sampler described different objects
        (MXR-080-1857).

        Refusing here rather than at construction is the point: scoring such a mixture is legitimate
        and tested (an open-world component has infinite total mass and no finite scale to absorb),
        so the object must remain constructible. What it cannot do is pretend to be a generative law.
        """
        from mixle.stats.compute.pdist import DensitySemantics

        if self.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR:
            unnormalized = [
                index
                for index, component in enumerate(self.components)
                if component.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR
            ]
            raise ValueError(
                "MixtureDistribution is a likelihood factor, not a normalized law, so it cannot be "
                f"sampled: component(s) {unnormalized} do not integrate to one, and the mixture "
                "weights therefore do not describe draw probabilities. Normalize those components "
                "(or drop a zero-evidence one) to obtain a generative mixture; log_density remains "
                "available either way."
            )
        return MixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> MixtureEstimator:
        """Return an estimator with matching component structure.

        Args:
            pseudo_count: Optional smoothing mass applied through the estimator
                path.

        Returns:
            ``MixtureEstimator`` suitable for fitting observations of the same
            type as this distribution.
        """
        if pseudo_count is not None:
            return MixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count,
                name=self.name,
                prior=self.prior,
            )
        else:
            return MixtureEstimator([u.estimator() for u in self.components], name=self.name, prior=self.prior)

    def decomposition(self):
        """Mixture components split along the component axis. Responsibilities (logsumexp) are computed
        INSIDE a shard; across shards the per-component sufficient stats SUM-reduce plus one scalar
        total-count all-reduce -- the homogeneous stacked-kernel + DTensor path (engine_axis=0)."""
        from mixle.stats.compute.decomposition import DecompAxis, Decomposition, ReductionOp

        return Decomposition(
            axis=DecompAxis.COMPONENT,
            num_units=self.num_components,
            reduction=ReductionOp.LOGSUMEXP_RESPONSIBILITY,
            exact=True,
            child_roles=tuple(f"component_{i}" for i in range(self.num_components)),
            engine_axis=0,
            key_pooling=getattr(self, "keys", None) is not None,
        )

    def dist_to_encoder(self) -> MixtureDataEncoder:
        """Return an encoder that delegates observation encoding to components."""
        dist_encoders = [c.dist_to_encoder() for c in self.components]
        return MixtureDataEncoder(encoder=dist_encoders)

    def enumerator(self) -> MixtureEnumerator:
        """Return an enumerator over the union of component supports."""
        return MixtureEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index from a global mixture frontier.

        The primary path pulls candidates from weighted component enumerator heads.
        The log-sum of those heads bounds every unseen value, so construction stops
        when the live global frontier falls below ``2**(-max_bits)``. This avoids the
        looser per-component ``log2(K)`` candidate expansion. If a component cannot
        enumerate, the method falls back to the structured cross-index path.
        """
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        active = [
            (k, comp, float(self.w[k]), float(self.log_w[k]))
            for k, comp in enumerate(self.components)
            if self.w[k] > 0.0
        ]
        if not active:
            return QuantizedEnumerationIndex.from_items(
                [], max_bits=max_bits, bin_width_bits=bin_width_bits, truncated=False
            )

        active_count = len(active)
        comps = [comp for _, comp, _, _ in active]
        log_w_arr = np.asarray([log_w for _, _, _, log_w in active], dtype=np.float64)

        def exact_log_density(x):
            with np.errstate(divide="ignore"):
                return vec.log_sum(np.asarray([c.log_density(x) for c in comps]) + log_w_arr)

        def component_log_density(k: int, x: T) -> float:
            return float(comps[k].log_density(x))

        try:
            streams = [
                BufferedStream(child_enumerator(comp, "MixtureDistribution.components[%d]" % k))
                for k, comp, _, _ in active
            ]
            log_offsets = [log_w for _, _, _, log_w in active]
            return bounded_best_first_union_index(
                streams,
                log_offsets,
                exact_log_density,
                max_bits=max_bits,
                bin_width_bits=bin_width_bits,
                component_log_density=component_log_density,
            )
        except EnumerationError:
            pass

        cross_bits = tuple(float(max_bits) + math.log(active_count * weight, 2.0) for _, _, weight, _ in active)
        try:
            cross = comps[0].quantized_multi_cross_index(comps[1:], max_bits=cross_bits, bin_width_bits=bin_width_bits)
            candidates = []
            for value, log_probs in cross.iter_items():
                mix_lp = vec.log_sum(log_w_arr + np.asarray(log_probs, dtype=np.float64))
                candidates.append((value, float(mix_lp)))
            return QuantizedEnumerationIndex.from_items(
                candidates, max_bits=max_bits, bin_width_bits=bin_width_bits, truncated=cross.truncated
            )
        except EnumerationError:
            pass

        candidates = []
        seen = set()
        truncated = False
        for k, comp, weight, _ in active:
            candidate_bits = float(max_bits) + math.log(active_count * weight, 2.0)
            if candidate_bits < 0.0:
                truncated = True
                continue
            try:
                child_index = comp.quantized_index(max_bits=candidate_bits, bin_width_bits=bin_width_bits)
            except EnumerationError as e:
                path = "MixtureDistribution.components[%d]" % k
                new_path = path if not e.path else "%s -> %s" % (path, e.path)
                raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None
            truncated = truncated or child_index.truncated
            for value, _ in child_index.iter_from():
                key = freeze(value)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((value, exact_log_density(value)))

        if truncated:
            return QuantizedEnumerationIndex.from_items(
                candidates, max_bits=max_bits, bin_width_bits=bin_width_bits, truncated=True
            )
        return QuantizedEnumerationIndex.from_items(candidates, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """BoundedCount for the MARGINAL mixture law: pool weight-scaled component count indices.

        log p(x) = logsumexp_k (log w_k + log p_k(x)) has no exact structural count -- overlapping
        component supports would need value-level deduplication. This builds the count semiring's
        ``plus``-fold over ``scale(component_index, log w_k)`` instead, which:

          * reaches a 2**M budget structurally (no enumeration), and
          * is a conservative UPPER bound -- a value shared by several components is counted once
            per component, and each value is binned by its dominant weighted component (the tropical
            cost, within log2(K) bits of the exact logsumexp).

        Every unranked value still carries its exact mixture ``log_density`` (re-evaluated by the
        budget builder). For an exact small-budget index (best-first union with dedup), use
        ``quantized_index``. Components that cannot count structurally raise EnumerationError.
        """
        from mixle.enumeration.quantization.core import child_count_index
        from mixle.enumeration.quantization.semiring import CountSemiring

        sr = CountSemiring()
        total = sr.zero()
        built = False
        truncated = False
        for k, comp in enumerate(self.components):
            if self.w[k] <= 0.0:
                continue
            child_index, child_truncated = child_count_index(
                comp, "MixtureDistribution.components[%d]" % k, quantizer, max_fine_bucket
            )
            truncated = truncated or child_truncated
            # A component individually fitting under max_fine_bucket does NOT imply its WEIGHTED
            # (shifted) contribution does: the true pre-shift top is child_index.hist.max_bucket()
            # (exact -- the deepest histogram entry is always nonzero post-normalize), so the exact
            # pre-cap top after the weight shift is that plus the weight term's own bucket. `sr.scale`
            # truncates its own output at max_fine_bucket without reporting whether it had to, so it
            # must be checked here instead (mirrors CompositeDistribution.quantized_count_index's
            # identical check for its convolution step).
            comp_top = child_index.hist.max_bucket()
            if comp_top is not None and comp_top + quantizer.fine_bucket(float(self.log_w[k])) > max_fine_bucket:
                truncated = True
            scaled = sr.scale(child_index, float(self.log_w[k]), quantizer, max_fine_bucket)
            total = scaled if not built else sr.plus(total, scaled)
            built = True

        if not built:
            return sr.zero(), truncated
        return total, truncated

    def _min_structural_fine_bucket(self, value, quantizer):
        """Minimum over components of (component structural bucket + weight-term bucket), or None.

        Uses each component's ``structural_fine_bucket`` -- the SUM-of-floored sub-buckets the count
        index actually used -- not ``fine_bucket(log p_k(value))``. For a nested component (composite/
        sequence) those differ by up to the number of sub-factors, and the old single-floor form
        mispredicted the canonical bin and silently dropped such values from the distinct stream.
        """
        best = None
        for k in range(len(self.components)):
            if self.w[k] <= 0.0:
                continue
            comp = self.components[k]
            if comp.log_density(value) == -np.inf:
                continue
            fb = comp.structural_fine_bucket(value, quantizer) + quantizer.fine_bucket(float(self.log_w[k]))
            if best is None or fb < best:
                best = fb
        return best

    def structural_fine_bucket(self, value, quantizer) -> int:
        """Dominant weighted-component structural bucket (mirrors the plus-of-scaled-children index)."""
        best = self._min_structural_fine_bucket(value, quantizer)
        return quantizer.fine_bucket(float(self.log_density(value))) if best is None else best

    def is_canonical_copy(self, value, coarse_bin: int, quantizer) -> bool:
        """Stateless dedup: keep ``value`` only at its dominant (best-weighted) component's bin.

        The canonical bin is the coarse bin of the minimum, over components, of the component's
        structural fine bucket shifted by the weight term. O(K) model evaluations, no state.
        """
        best = self._min_structural_fine_bucket(value, quantizer)
        return best is not None and coarse_bin == quantizer.coarse_bin(best)


class MixtureEnumerator(DistributionEnumerator):
    """Enumerator over the deduplicated union of weighted component supports."""

    def __init__(self, dist: MixtureDistribution) -> None:
        """Enumerates the union of component supports in descending mixture probability order.

        Component supports may overlap, so candidates pulled from the component enumerations
        are re-scored exactly with the mixture log-density and emitted only once their score
        beats the upper bound on any not-yet-seen value. Components with zero weight are
        never asked to enumerate.

        Args:
            dist (MixtureDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        streams = []
        log_offsets = []
        comps = []
        for k, comp in enumerate(dist.components):
            if dist.w[k] <= 0.0:
                continue
            streams.append(BufferedStream(child_enumerator(comp, "MixtureDistribution.components[%d]" % k)))
            log_offsets.append(dist.log_w[k])
            comps.append(comp)
        log_w_arr = np.asarray(log_offsets, dtype=np.float64)

        # Equivalent to dist.log_density but restricted to positive-weight components, so a
        # zero-weight component never sees (possibly type-incompatible) candidate values.
        def exact_log_density(x):
            with np.errstate(divide="ignore"):
                return vec.log_sum(np.asarray([c.log_density(x) for c in comps]) + log_w_arr)

        self._union = best_first_union(streams, log_offsets, exact_log_density)

    def __next__(self) -> tuple[Any, float]:
        return next(self._union)


class MixtureSampler(DistributionSampler):
    """Sampler that draws a latent component and then samples from that component."""

    def __init__(self, dist: MixtureDistribution, seed: int | None = None) -> None:
        """MixtureSampler used to generate samples from instance of MixtureDistribution.

        Args:
            dist (MixtureDistribution): Assign MixtureDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Attributes:
            dist (MixtureDistribution): MixtureDistribution to draw samples from.
            rng (RandomState): Seeded RandomState for sampling.
            comp_samplers (List[DistributionSamplers]): List of DistributionSampler objects for each mixture component.

        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[Any] | Any:
        """Draw iid samples from a mixture distribution.

        The data type drawn from 'comp_samplers' is type T, corresponding to the data type of the mixture components.

        If size is None, a single sample (of data type T) is drawn and returned. If size is not None, 'size'-iid
        mixture samples are drawn and returned as a List with data type List[T].

        With ``batched=True`` (default) each component sampler is invoked once with the number of draws assigned to
        it and the results are scattered back into draw order. Because every component sampler owns an independent
        ``RandomState``, this yields the same draws as the legacy per-draw loop (``batched=False``) but far faster.

        Args:
            size (Optional[int]): Number of iid samples to draw.
            batched (bool): Vectorize component draws (default); set False for the legacy per-draw loop.

        Returns:
            Data type T or List[T].

        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.comp_samplers[comp_state].sample()
        if not batched:
            return [self.comp_samplers[i].sample() for i in comp_state]
        from mixle.stats.compute._sampling import scatter_component_draws

        return scatter_component_draws(comp_state, self.comp_samplers, int(size))


MIXTURE_INIT_STRATEGIES = ("dirichlet", "kmeans++")
"""The closed set of ``init=`` strategies the mixture EM accumulator implements.

``"kmeans++"`` (the default) seeds near-hard responsibilities from k-means++ centers when the
encoded data yields a numeric feature matrix, and falls back to ``"dirichlet"`` when it does not.
``"dirichlet"`` draws sparse random responsibilities for every observation.
"""

_INIT_SPELLING_HINTS = {
    # Alphanumeric-normalized near misses -> the name this module uses. The reference library a
    # practitioner is most likely carrying habits over from spells the same two ideas differently:
    # sklearn.cluster.KMeans(init="k-means++") and GaussianMixture(init_params="kmeans") both mean
    # "kmeans++" here, and GaussianMixture(init_params="random") is the random-responsibility init,
    # which is what "dirichlet" spells here.
    "kmeans": "kmeans++",
    "kmeansplusplus": "kmeans++",
    "random": "dirichlet",
    "dirichlet": "dirichlet",
}


def _init_spelling_hint(value: Any) -> str:
    """Return a ``; did you mean ...`` clause when ``value`` is a recognisable misspelling."""
    if not isinstance(value, str):
        return ""
    normalized = "".join(ch for ch in value.lower() if ch.isalnum())
    suggestion = _INIT_SPELLING_HINTS.get(normalized)
    return '; did you mean "%s"?' % suggestion if suggestion is not None else ""


def _validated_mixture_init(value: Any, label: str) -> str:
    """Return ``value`` as one of :data:`MIXTURE_INIT_STRATEGIES`, or raise naming the accepted set.

    ``init`` selects an algorithm, and there is no defensible fit for one this module does not
    implement, so an unrecognised name is refused rather than absorbed. Absorbing it ran the
    ``"dirichlet"`` path under another name: a caller who spelled the option the way scikit-learn
    spells it got the fit they were trying to avoid, and passing such a value alongside
    ``robust=True`` disarmed that path's own ``"kmeans++"`` default with no signal anywhere. The
    accepted set has only ever held these two names -- ``MixtureEstimator`` (and, through it, every
    accumulator/factory in this family) is the only producer -- so no state the library itself
    creates is rejected here.
    """
    if isinstance(value, str) and value in MIXTURE_INIT_STRATEGIES:
        return value
    raise ValueError(
        "%s: init must be one of %r, got %r%s" % (label, MIXTURE_INIT_STRATEGIES, value, _init_spelling_hint(value))
    )


def _validated_component_estimators(estimators: Any, label: str) -> Sequence[ParameterEstimator]:
    """Validate ``estimators`` at the mixture-constructor boundary, naming what a mistake was.

    The three likely mistakes each used to surface as a bare error from library internals (T4-9):
    a single estimator instead of a sequence raised ``TypeError: object of type 'GaussianEstimator'
    has no len()`` here; an empty sequence constructed fine and then raised ``IndexError: list index
    out of range`` deep in the first E-step (or ``ZeroDivisionError`` at construction under
    ``robust=True``); and a non-estimator element raised ``AttributeError`` from
    ``accumulator_factory()``. Nothing the library itself builds is refused: every internal caller
    passes a non-empty list of estimators, and the element probe is the ``accumulator_factory``
    attribute EM itself requires -- an object without one never survived the E-step anyway.
    """
    if isinstance(estimators, str) or not hasattr(estimators, "__len__"):
        if hasattr(estimators, "accumulator_factory"):
            raise TypeError(
                "%s: estimators must be a sequence of component estimators, got a single %s; "
                "wrap it in a list: %s([estimator])" % (label, type(estimators).__name__, label)
            )
        raise TypeError(
            "%s: estimators must be a sequence of component estimators (one per mixture "
            "component), got %s" % (label, type(estimators).__name__)
        )
    if len(estimators) == 0:
        raise ValueError("%s: estimators is empty -- a mixture needs at least one component estimator" % label)
    for i, est in enumerate(estimators):
        if not hasattr(est, "accumulator_factory"):
            raise TypeError(
                "%s: estimators[%d] must be a component estimator (an object providing "
                "accumulator_factory(), e.g. GaussianEstimator()), got %s" % (label, i, type(est).__name__)
            )
    return estimators


def _validated_mixture_keys(keys: Any, label: str) -> Any:
    """Validate the ``keys`` pair at the mixture-constructor boundary.

    A mixture has two keyable statistic sites -- the weights and the pooled component payloads --
    so ``keys`` is a 2-entry pair, unlike the single string leaf estimators take. A single string
    used to be absorbed by indexing: ``keys="k"`` raised a bare ``IndexError: string index out of
    range`` when the accumulator asked for ``keys[1]``, and a 2-character string was silently split
    into two 1-character keys (``keys="wc"`` shared weights under ``"w"`` and components under
    ``"c"``), which is worse than an error (T4-9). ``None`` is accepted as "no sharing at all" --
    the meaning it visibly asks for. Every internal caller already passes a 2-entry tuple or list,
    which is returned unchanged.
    """
    if keys is None:
        return (None, None)
    if isinstance(keys, (tuple, list)) and len(keys) == 2 and all(k is None or isinstance(k, str) for k in keys):
        return keys
    if isinstance(keys, str):
        raise TypeError(
            "%s: keys must be a (weights_key, components_key) pair, got the single string %r; "
            "pass (%r, None) to share only the mixture weights, (None, %r) to share only the "
            "component statistics, or (%r, %r) to share both" % (label, keys, keys, keys, keys, keys)
        )
    raise TypeError(
        "%s: keys must be a (weights_key, components_key) pair of str-or-None entries, got %r" % (label, keys)
    )


def _validated_fixed_weights(fixed_weights: Any, num_components: int, label: str) -> np.ndarray | None:
    """Validate ``fixed_weights`` against the component count at the constructor boundary.

    Without this, a wrong-length vector constructed fine and failed only at the end of the first
    EM iteration, and a non-numeric value failed there with a message that never named the argument
    (T4-9). Weights are checked -- through a float view -- for shape, finiteness, non-negativity,
    and positive total mass, values for which no mixture density exists; they are deliberately NOT
    renormalized (``estimate`` uses them exactly as given, matching every internal caller), and
    what is stored is ``np.asarray(fixed_weights)`` with its dtype preserved, byte-identical to
    what this constructor stored before validation existed, so serialized 0.7-era estimator
    artifacts (whose canonical state is compared against a fresh construction on load) keep
    loading.
    """
    if fixed_weights is None:
        return None
    try:
        w = np.asarray(fixed_weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "%s: fixed_weights must be a numeric vector of %d mixture weights, got %r"
            % (label, num_components, fixed_weights)
        ) from exc
    if w.ndim != 1 or w.shape[0] != num_components:
        raise ValueError(
            "%s: fixed_weights must hold exactly %d weights (one per component estimator), "
            "got shape %s" % (label, num_components, w.shape)
        )
    if not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("%s: fixed_weights must be finite and non-negative, got %s" % (label, w))
    if not w.sum() > 0.0:
        raise ValueError("%s: fixed_weights must have positive total mass, got %s" % (label, w))
    return np.asarray(fixed_weights)


def _validated_weight_scalar(value: Any, name: str, label: str, nonnegative: bool = False) -> Any:
    """Validate a scalar weight-path argument (``pseudo_count``, ``w_min``) at the boundary.

    A non-numeric value used to construct fine and only fail mid-M-step with a bare operand
    ``TypeError`` (T4-9); a negative ``pseudo_count`` silently subtracted mass from every
    component's count, which can drive fitted weights negative -- no defensible fit exists for
    either. Anything ``float()`` accepts (ints, numpy scalars, numeric strings) is computed with
    rather than refused; bools are rejected as the near-certain mistakes they are, matching
    ``finite_scalar`` in the vector families. A value that already IS a real number is returned
    unchanged -- an int stored as an int is the canonical state 0.7-era serialized estimators
    carry, and the load path refuses an artifact whose state a fresh construction would rewrite.
    """
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s: %s must be a real scalar, got %r" % (label, name, value))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s: %s must be a real scalar, got %r" % (label, name, value)) from exc
    if not np.isfinite(result):
        raise ValueError("%s: %s must be finite, got %r" % (label, name, value))
    if nonnegative and result < 0.0:
        raise ValueError("%s: %s must be non-negative, got %r" % (label, name, value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value
    return result


def _validated_weight_counts(suff_stat: Any, num_components: int, label: str) -> Any:
    """Validate the ``suff_stat`` prior component-count vector at the constructor boundary.

    A wrong-length list used to fail mid-M-step with ``TypeError: can't multiply sequence by
    non-int of type 'float'`` -- and a TUPLE alongside an integer ``pseudo_count`` was silently
    REPEATED by sequence multiplication instead of scaled (T4-9). Representations that were
    already sound downstream -- a numeric ndarray of per-component counts, or a single scalar,
    which broadcasts to a uniform prior count -- are stored exactly as passed, keeping 0.7-era
    serialized artifacts loadable under the canonical-state check; any other container is
    converted to a float ndarray so the M-step's arithmetic operates on values, never on
    containers.
    """
    if suff_stat is None:
        return None
    if isinstance(suff_stat, np.ndarray) and (
        np.issubdtype(suff_stat.dtype, np.floating) or np.issubdtype(suff_stat.dtype, np.integer)
    ):
        counts = suff_stat
        store: Any = suff_stat
    elif not isinstance(suff_stat, (bool, np.bool_)) and isinstance(suff_stat, (int, float, np.integer, np.floating)):
        counts = np.asarray(suff_stat, dtype=float)
        store = suff_stat
    else:
        try:
            counts = np.asarray(suff_stat, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "%s: suff_stat must be a numeric component-count vector of length %d (or a single "
                "scalar), got %r" % (label, num_components, suff_stat)
            ) from exc
        store = counts
    if counts.ndim not in (0, 1) or (counts.ndim == 1 and counts.shape[0] != num_components):
        raise ValueError(
            "%s: suff_stat must hold exactly %d component counts (one per component estimator) "
            "or a single scalar, got shape %s" % (label, num_components, counts.shape)
        )
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("%s: suff_stat counts must be finite and non-negative, got %r" % (label, suff_stat))
    return store


class MixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """EM accumulator for mixture weights and component sufficient statistics."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] = (None, None),
        name: str | None = None,
        init: str = "kmeans++",
    ) -> None:
        """Create an EM accumulator for mixture responsibilities.

        Args:
            accumulators: Component sufficient-statistic accumulators.
            keys: Optional shared-statistic keys for mixture weights and
                component payloads.
            name: Optional diagnostic name.
            init: Initialization strategy, one of
                :data:`MIXTURE_INIT_STRATEGIES`. ``"kmeans++"`` (the default)
                uses numeric encoded features when possible and falls back to
                ``"dirichlet"`` otherwise; ``"dirichlet"`` draws random
                responsibilities. Any other value is rejected.

        Attributes:
            comp_counts: Accumulated expected component counts.
            accumulators: Component accumulators receiving responsibility-
                weighted observations.
        """
        self.accumulators = list(accumulators)
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.name = name
        self.init = _validated_mixture_init(init, "MixtureAccumulator()")
        # Data log-likelihood accumulated as a byproduct of the E-step (the posterior normalizer),
        # only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        ### Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: list[RandomState] | None = None

    def seq_update(self, x: T1, weights: np.ndarray, estimate: MixtureDistribution) -> None:
        """Accumulate a vectorized EM E-step from encoded observations.

        Responsibilities are computed from ``estimate`` using the same
        log-sum-exp normalization as ``MixtureDistribution.seq_posterior``.
        Rows where every component is impossible receive zero responsibility,
        so an observation with zero model probability cannot fabricate
        component sufficient statistics.

        Args:
            x: Encoded observation batch.
            weights: Non-negative observation weights.
            estimate: Previous EM iterate used to compute responsibilities.
        """
        enc_data = x
        rows = self.acc_to_encoder().row_count(enc_data)
        weights = validated_observation_weights(weights, rows, "mixture observation weights")
        if estimate.num_components != self.num_components:
            raise ValueError("mixture estimate and accumulator component counts must match")
        ll_mat_init = False

        for i in range(estimate.num_components):
            if not estimate.zw[i]:
                temp = estimate.components[i].seq_log_density(_component_enc(enc_data, i))

                if not ll_mat_init:
                    ll_mat = np.zeros((len(temp), self.num_components), dtype=np.float64)
                    ll_mat.fill(-np.inf)
                    ll_mat_init = True

                ll_mat[:, i] = temp
                ll_mat[:, i] += estimate.log_w[i]

        normalized = normalize_mixture_log_scores(ll_mat)
        if self._track_ll and ll_mat_init:
            included = np.asarray(weights) != 0
            self._seq_ll += float(np.dot(np.asarray(weights)[included], normalized.log_evidence[included]))

        ll_mat = validated_weighted_responsibilities(
            normalized.responsibilities * weights[:, None],
            weights,
            self.num_components,
            label="mixture weighted responsibilities",
            allow_unassigned=True,
        )

        for i in range(self.num_components):
            w_loc = ll_mat[:, i]
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(_component_enc(enc_data, i), w_loc, estimate.components[i])

    def update(self, x: T, weight: float, estimate: MixtureDistribution) -> None:
        """Accumulate one weighted raw observation under an EM estimate.

        The observation is routed to each component accumulator with
        ``weight * estimate.posterior(x)[k]``.

        Args:
            x: Raw observation.
            weight: Observation weight.
            estimate: Previous EM iterate used to compute responsibilities.
        """
        weight = validated_observation_weight(weight, "mixture observation weight")
        posterior = estimate.posterior(x)
        posterior *= weight
        posterior = validated_weighted_responsibilities(
            posterior[None, :],
            np.asarray([weight]),
            self.num_components,
            label="mixture weighted responsibility",
            allow_unassigned=True,
        )[0]
        self.comp_counts += posterior

        for i in range(self.num_components):
            self.accumulators[i].update(x, posterior[i], estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        """Seed per-component initializer RNGs from a caller-provided RNG.

        Args:
            rng: Source random state for reproducible mixture initialization.
        """
        seeds = rng.randint(2**31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: T, weight: float, rng: np.random.RandomState) -> None:
        """Initialize component sufficient statistics from one observation.

        This path draws a responsibility vector from a Dirichlet distribution
        and delegates responsibility-weighted initialization to every component
        accumulator. It does so for every ``init`` strategy: k-means++ seeding
        needs the whole batch to place its centers, and this method sees one
        observation at a time. The scalar and vectorized initializations
        therefore agree only under ``init="dirichlet"``; use
        :meth:`seq_initialize` to get the configured strategy.

        Args:
            x: Raw observation.
            weight: Observation weight.
            rng: Random state used to seed component initializers.
        """
        weight = validated_observation_weight(weight, "mixture initialization weight")
        if not self._init_rng:
            self._rng_initialize(rng)

        if weight != 0:
            ww = self._w_rng.dirichlet(np.ones(self.num_components) / (self.num_components * self.num_components))
        else:
            ww = np.zeros(self.num_components)

        for i in range(self.num_components):
            w = weight * ww[i]
            self.accumulators[i].initialize(x, w, self._acc_rng[i])
            self.comp_counts[i] += w

    def seq_initialize(self, x: T1, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize component sufficient statistics from encoded observations.

        With ``init="kmeans++"`` the method uses a numeric feature matrix when
        one can be extracted from the encoded data. Ragged, object, hetero, or
        non-finite encodings fall back to Dirichlet responsibilities rather than
        mutating input data or forcing an invalid numeric representation.

        Args:
            x: Encoded observation batch.
            weights: Non-negative observation weights.
            rng: Random state used to seed component initializers.
        """
        rows = self.acc_to_encoder().row_count(x)
        weights = validated_observation_weights(weights, rows, "mixture initialization weights")
        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        keep_idx = weights > 0
        keep_len = np.count_nonzero(keep_idx)
        ww = np.zeros((sz, self.num_components))

        responsibilities = None
        if self.init == "kmeans++" and keep_len > 0:
            responsibilities = self._kmeanspp_responsibilities(x, keep_idx)

        if responsibilities is not None:
            ww = responsibilities
        elif keep_len > 0:
            ww[keep_idx, :] = self._w_rng.dirichlet(
                alpha=np.ones(self.num_components) / (self.num_components**2), size=keep_len
            )
        ww *= np.reshape(weights, (sz, 1))

        for i in range(self.num_components):
            self.accumulators[i].seq_initialize(_component_enc(x, i), ww[:, i], self._acc_rng[i])
            self.comp_counts[i] += np.sum(ww[:, i])

    def _feature_matrix(self, x: Any, keep_idx: np.ndarray) -> np.ndarray | None:
        """Best-effort extraction of a dense (kept_n, d) numeric feature matrix from encoded data.

        Returns ``None`` (so we fall back to the Dirichlet path) when the encoded data is not a
        simple real-valued array — e.g. composite/tuple encodings, ragged sequences, non-numeric
        dtypes. k-means++ only makes sense for vector-space leaves (Gaussian / diagonal Gaussian).
        """
        if isinstance(x, _HeteroMixtureEncoded):
            return None
        try:
            arr = np.asarray(x)
        except (TypeError, ValueError):
            return None
        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            return None
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            return None
        if arr.shape[0] != len(keep_idx):
            return None
        arr = arr[keep_idx]
        if arr.shape[0] == 0 or not np.isfinite(arr).all():
            return None
        return np.asarray(arr, dtype=float)

    def _kmeanspp_responsibilities(self, x: Any, keep_idx: np.ndarray) -> np.ndarray | None:
        """P4 k-means++ seeding: assign near-hard responsibilities from nearest k-means++ center.

        Falls back to ``None`` (legacy Dirichlet init) when a numeric feature matrix cannot be
        extracted from the encoded data. This sidesteps the random-Dirichlet EM saddle for
        Gaussian-mixture initialization with no new dependency.
        """
        feats = self._feature_matrix(x, keep_idx)
        if feats is None:
            return None

        n, _ = feats.shape
        k = self.num_components
        rng = self._w_rng
        centers_idx = np.empty(k, dtype=int)
        centers_idx[0] = rng.randint(n)
        closest_sq = np.sum((feats - feats[centers_idx[0]]) ** 2, axis=1)

        for c in range(1, k):
            total = float(closest_sq.sum())
            if total <= 0.0 or not np.isfinite(total):
                centers_idx[c] = rng.randint(n)
            else:
                probs = closest_sq / total
                centers_idx[c] = int(rng.choice(n, p=probs))
            new_sq = np.sum((feats - feats[centers_idx[c]]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_sq)

        centers = feats[centers_idx]
        # squared distances (n, k); assign each kept point to its nearest center
        dists = np.sum((feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dists, axis=1)

        sz = len(keep_idx)
        ww = np.zeros((sz, k))
        # soft-ish responsibilities: dominant mass on nearest center, small floor on the rest so
        # no component starts byte-degenerate even if a center captures few points.
        kept_rows = np.nonzero(keep_idx)[0]
        floor = 1.0e-3 / k
        ww[kept_rows, :] = floor
        ww[kept_rows, assign] = 1.0 - floor * (k - 1)
        return ww

    def combine(self, suff_stat: tuple[np.ndarray, tuple[T2, ...]]) -> MixtureAccumulator:
        """Merge serialized mixture sufficient statistics into this accumulator.

        Args:
            suff_stat: ``(component_counts, component_suff_stats)`` tuple.

        Returns:
            ``self`` for accumulator chaining.
        """
        suff_stat = validated_statistic_tuple(suff_stat, 2, "mixture sufficient statistics")
        counts = validated_count_array(
            suff_stat[0],
            (self.num_components,),
            "mixture component counts",
        )
        if not isinstance(suff_stat[1], (tuple, list)) or len(suff_stat[1]) != self.num_components:
            raise ValueError("mixture child sufficient statistics must match the component count")
        # The ENTIRE combine is transactional: a child rejecting its part mid-loop used to leave
        # the counts and every earlier child already merged, and two individually valid count
        # vectors can sum to an infinite aggregate that per-element ingestion checks cannot see
        # (measured in the latent-family mutator audit; the chain-HMM findings STAT-RR8-1 and
        # STAT-RR9-1 established the defect classes).
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("comp_counts",), child_attrs=("accumulators",))
        self.comp_counts += counts
        try:
            require_finite_count_totals((("component counts", self.comp_counts),), label="combined mixture")
            for i in range(self.num_components):
                self.accumulators[i].combine(copy.deepcopy(suff_stat[1][i]))
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise

        return self

    def value(self) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Return serialized mixture sufficient statistics.

        Returns:
            ``(component_counts, component_suff_stats)`` where the second item
            contains one serialized child accumulator value per component.
        """
        return self.comp_counts.copy(), tuple(copy.deepcopy(u.value()) for u in self.accumulators)

    def from_value(self, x: tuple[np.ndarray, tuple[T2, ...]]) -> MixtureAccumulator:
        """Restore this accumulator from serialized sufficient statistics.

        Args:
            x: ``(component_counts, component_suff_stats)`` tuple.

        Returns:
            ``self`` after restoring child accumulator state.
        """
        x = validated_statistic_tuple(x, 2, "mixture sufficient statistics")
        candidate = validated_count_array(
            x[0],
            (self.num_components,),
            "mixture component counts",
        )
        if not isinstance(x[1], (tuple, list)) or len(x[1]) != self.num_components:
            raise ValueError("mixture child sufficient statistics must match the component count")
        # Validate before assigning, and restore transactionally: a child rejecting its part
        # mid-loop used to leave the counts already replaced and earlier children restored
        # (measured in the latent-family mutator audit; STAT-RR9-1 class).
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("comp_counts",), child_attrs=("accumulators",))
        self.comp_counts = candidate
        try:
            for i in range(self.num_components):
                self.accumulators[i].from_value(copy.deepcopy(x[1][i]))
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise
        return self

    def scale(self, c: float) -> MixtureAccumulator:
        """Scale component counts and delegate child sufficient statistics."""
        c = validated_observation_weight(c, "mixture statistic scale")
        # Parent counts and children scale as ONE transaction, with the scaled result validated
        # as a postcondition: a valid factor times a valid statistic can overflow silently, and
        # a child raising mid-loop used to leave the parent and earlier children scaled
        # (measured; STAT-RR8-1 and STAT-RR10-1 classes).
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("comp_counts",), child_attrs=("accumulators",))
        self.comp_counts *= c
        try:
            require_finite_count_totals((("component counts", self.comp_counts),), label="scaled mixture")
            for acc in self.accumulators:
                acc.scale(c)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed mixture statistics into a shared statistics dictionary.

        Args:
            stats_dict: Mutable shared sufficient-statistics mapping keyed by
                estimator key names.
        """
        # The WHOLE merge is transactional: a failure in the component pool (or a child's own
        # keyed merge) used to leave the weight pool already merged, and restoring the mapping's
        # values alone leaves a caller's alias to a pooled array holding the partial merge --
        # the failed mapping is healed IN PLACE instead (measured; STAT-RR9-1 and STAT-RR10-1
        # classes). The snapshot must cover keys this level cannot enumerate, so it deep-copies
        # the dict on entry.
        _snapshot = copy.deepcopy(stats_dict)
        try:
            if self.weight_key is not None:
                if self.weight_key in stats_dict:
                    stats_dict[self.weight_key] += self.comp_counts
                    # Pooling reaches overflow by addition exactly as combine() does.
                    require_finite_count_totals(
                        (("pooled component counts", stats_dict[self.weight_key]),),
                        label="mixture key merge",
                    )
                else:
                    # Copy on adoption: stats_dict must never alias this accumulator's own live
                    # array, or a later tied accumulator's in-place += above would silently mutate
                    # this accumulator's private comp_counts as a side effect of merging.
                    stats_dict[self.weight_key] = self.comp_counts.copy()

            if self.comp_key is not None:
                if self.comp_key in stats_dict:
                    acc = stats_dict[self.comp_key]
                    for i in range(len(acc)):
                        acc[i] = acc[i].combine(copy.deepcopy(self.accumulators[i].value()))
                else:
                    stats_dict[self.comp_key] = copy.deepcopy(self.accumulators)

            for u in self.accumulators:
                u.key_merge(stats_dict)
        except Exception:
            heal_pooled_statistics(stats_dict, _snapshot)
            raise

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace local keyed statistics from a shared statistics dictionary.

        Args:
            stats_dict: Shared sufficient-statistics mapping keyed by estimator
                key names.
        """
        # A replacement arrives from outside and used to be copied in with no shape, element, or
        # aggregate validation at all -- [inf, 0] landed directly in the accumulator -- so the
        # candidate now goes through the same count-array contract as ingestion BEFORE anything
        # is assigned, and a failure at any later point (a poisoned pooled child, a child's own
        # keyed replace) rolls the whole accumulator back (measured; STAT-RR8-1 and STAT-RR9-1
        # classes). Shape comes from what this accumulator already holds.
        candidate = None
        if self.weight_key is not None and self.weight_key in stats_dict:
            candidate = validated_count_array(
                stats_dict[self.weight_key],
                np.shape(self.comp_counts),
                "mixture replacement component counts",
            )
            require_finite_count_totals((("component counts", candidate),), label="mixture key replace")

        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("comp_counts",), child_attrs=("accumulators",))
        if candidate is not None:
            # Copy on replace too: without it, every tied accumulator ends up pointing at
            # the SAME array object, so any one of them later accumulating new local data
            # would silently corrupt every other tied accumulator's counts.
            self.comp_counts = candidate.copy()
        try:
            if self.comp_key is not None:
                if self.comp_key in stats_dict:
                    acc = stats_dict[self.comp_key]
                    if len(acc) != self.num_components:
                        raise ValueError("keyed mixture component statistics have incompatible arity.")
                    for local, pooled in zip(self.accumulators, acc):
                        local.from_value(copy.deepcopy(pooled.value()))

            for u in self.accumulators:
                u.key_replace(stats_dict)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise

    def acc_to_encoder(self) -> MixtureDataEncoder:
        """Return an encoder assembled from the component accumulators."""
        acc_encoders = [a.acc_to_encoder() for a in self.accumulators]
        return MixtureDataEncoder(encoder=acc_encoders)


class MixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for mixture accumulators built from component accumulator factories."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        keys: tuple[str | None, str | None] = (None, None),
        name: str | None = None,
        init: str = "kmeans++",
    ) -> None:
        """Create a factory for mixture EM accumulators.

        Args:
            factories: Component accumulator factories.
            keys: Optional shared-statistic keys for weights and components.
            name: Optional diagnostic name.
            init: Initialization strategy passed to ``MixtureAccumulator``,
                one of :data:`MIXTURE_INIT_STRATEGIES`.
        """
        self.factories = factories
        self.keys = keys
        self.name = name
        self.init = _validated_mixture_init(init, "MixtureAccumulatorFactory()")

    def make(self) -> MixtureAccumulator:
        """Return a fresh mixture accumulator with fresh component accumulators."""
        return MixtureAccumulator(
            [factory.make() for factory in self.factories], keys=self.keys, name=self.name, init=self.init
        )


class MixtureEstimator(ParameterEstimator):
    """Estimator for mixture weights and component distributions from EM sufficient statistics."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        fixed_weights: list[float] | np.ndarray | None = None,
        suff_stat: np.ndarray | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: tuple[str | None, str | None] = (None, None),
        prior: SequenceEncodableProbabilityDistribution | None = None,
        w_min: float = 0.0,
        robust: bool = False,
        init: str | None = None,
    ) -> None:
        """Create an EM estimator for a homogeneous finite mixture.

        Args:
            estimators: Component estimators.
            fixed_weights: Optional fixed simplex weights. When supplied, EM
                updates only the component parameters.
            suff_stat: Optional prior component-count vector used with
                ``pseudo_count``.
            pseudo_count: Smoothing mass for mixture weights.
            name: Optional diagnostic name.
            keys: Optional shared-statistic keys for weights and components.
            prior: Optional Dirichlet weight prior or joint mixture prior.
            w_min: Plain-MLE weight floor. When positive, fitted weights are
                clamped and renormalized so a component cannot be frozen at
                exact zero in later EM iterations.
            robust: Enable the robust default path: a small data-independent
                weight floor on top of the ordinary k-means++ initialization.
            init: Initialization strategy for the accumulator, one of
                :data:`MIXTURE_INIT_STRATEGIES`; ``None`` selects the default ``"kmeans++"``.
                Any other value is rejected. ``"kmeans++"`` seeds near-hard responsibilities from
                k-means++ centers whenever the encoded data yields a numeric feature matrix, and
                falls back to ``"dirichlet"`` when it does not. ``"dirichlet"`` draws
                per-observation responsibilities from a deliberately degenerate Dirichlet
                (``alpha = 1/K**2``) -- sparse random assignments, the classical responsibility
                init. It is no longer the default because random subsets of a large dataset share
                its pooled moments, so every component starts near the *global* law as ``n`` grows:
                on a well-separated location family the components then sit on top of each other in
                the empty middle and EM has no gradient to pull them apart, which returns a
                one-cluster answer for visibly multi-cluster data. Pass ``init="dirichlet"`` for
                the classical start, and see :func:`~mixle.inference.best_of` for restarts.
        """
        # Boundary validation (T4-9): each likely construction mistake used to surface as a bare
        # IndexError/TypeError/AttributeError/ZeroDivisionError from library internals, at
        # construction or -- worse -- only deep inside the first EM iteration. ``label`` is the
        # dynamic class name so GaussianMixtureEstimator (which forwards here) names itself.
        label = type(self).__name__
        estimators = _validated_component_estimators(estimators, label)
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = (
            None
            if pseudo_count is None
            else _validated_weight_scalar(pseudo_count, "pseudo_count", label, nonnegative=True)
        )
        self.suff_stat = _validated_weight_counts(suff_stat, self.num_components, label)
        self.keys = _validated_mixture_keys(keys, label)
        self.name = name
        self.fixed_weights = _validated_fixed_weights(fixed_weights, self.num_components, label)
        self.robust = require_exact_bool(robust, "robust")
        # k-means++ is the default start for every mode, not just the robust one: the Dirichlet
        # responsibility draw starts every component at the pooled law, which on a separated
        # location family leaves EM with nothing to pull the components apart -- it converges,
        # quietly, to a single cluster sitting in the empty middle. k-means++ falls back to the
        # Dirichlet draw whenever the encoded data has no numeric feature matrix, so families it
        # cannot seed are unaffected. robust=True adds the tiny data-independent weight floor.
        if init is None:
            init = "kmeans++"
        self.init = _validated_mixture_init(init, "%s()" % label)
        # w_min may be negative on purpose (any value <= 0 means "no weight floor"), so only its
        # scalar-ness is validated: a non-numeric value used to raise a bare comparison TypeError.
        w_min = _validated_weight_scalar(w_min, "w_min", label)
        if w_min <= 0.0 and self.robust:
            w_min = 1.0e-4 / self.num_components
        self.w_min = float(w_min)
        self.prior = None
        self.has_conj_prior = False
        self.set_prior(prior)

    def accumulator_factory(self) -> MixtureAccumulatorFactory:
        """Return a mixture accumulator factory matching the component estimators."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return MixtureAccumulatorFactory(est_factories, keys=self.keys, name=self.name, init=self.init)

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the joint mixture prior, or ``None`` for a plain MLE estimator.

        When a weight prior is attached the joint prior is the
        ``(weight_prior, tuple(component priors))`` pair produced by :func:`mixture_prior`.
        """
        if not self.has_conj_prior:
            return None
        return self.prior, tuple(d.get_prior() for d in self.estimators)

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a weight prior (and optional per-component priors).

        With a (symmetric) Dirichlet weight prior the estimator switches to the conjugate
        MAP weight update; component priors, when supplied, are delegated to each component
        estimator via ``estimator.set_prior`` (those carry out their own conjugate updates).
        ``prior=None`` leaves the estimator a plain MLE estimator (byte-identical behaviour).
        """
        weight_prior, component_priors = _split_mixture_prior(prior, self.num_components)
        self.prior = weight_prior
        if component_priors is not None:
            for d, p in zip(self.estimators, component_priors):
                _set_estimator_prior(d, p)
        DirichletDistribution, SymmetricDirichletDistribution = _dirichlet_types()
        self.has_conj_prior = isinstance(self.prior, (DirichletDistribution, SymmetricDirichletDistribution))

    def model_log_density(self, model: MixtureDistribution) -> float:
        """Log density of the model parameters under this estimator's prior (ELBO global term).

        Returns the Dirichlet weight-prior log-density evaluated at ``model.w`` plus the sum of
        each component estimator's ``model_log_density`` at the corresponding component model.
        Returns ``0.0`` for a plain MLE estimator with no priors anywhere.
        """
        rv = 0.0
        if self.has_conj_prior:
            rv += float(self.prior.log_density(model.w))
        for est, comp in zip(self.estimators, model.components):
            fn = getattr(est, "model_log_density", None)
            if fn is not None:
                term = fn(comp)
                if term is not None:
                    rv += float(term)
        return rv

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, tuple[Any, ...]]) -> MixtureDistribution:
        """Estimate a mixture distribution from EM sufficient statistics.

        ``suff_stat`` is ``(component_counts, component_suff_stats)``. Component
        parameters are delegated to the child estimators. Mixture weights follow
        the fixed-weight, conjugate-prior, pseudo-count, or plain-MLE path
        selected by the estimator configuration. Plain-MLE weights may be
        floored by ``w_min`` and are always renormalized.

        Args:
            nobs: Unused compatibility argument from ``ParameterEstimator``.
            suff_stat: Serialized mixture sufficient statistics.

        Returns:
            Fitted ``MixtureDistribution``.
        """
        num_components = self.num_components
        if not isinstance(suff_stat, (tuple, list)) or len(suff_stat) != 2:
            raise ContractError(
                "MixtureEstimator.estimate(suff_stat)",
                "a 2-tuple (component_weight_counts, component_suff_stats)",
                "%s%s"
                % (
                    type(suff_stat).__name__,
                    " of length %d" % len(suff_stat) if isinstance(suff_stat, (tuple, list)) else "",
                ),
                "pass the 2-tuple produced by MixtureAccumulator.value(), not a bare component sufficient statistic.",
            )
        counts, comp_suff_stats = suff_stat
        try:
            counts = validated_count_array(
                counts,
                (num_components,),
                "mixture component counts",
            )
        except (TypeError, ValueError) as exc:
            # validated_count_array is a shared leaf helper and raises a bare TypeError/ValueError
            # with no idea which field it was validating. Every other check at this boundary raises
            # a path-annotated ContractError, so a malformed count vector was the one way into
            # estimate() that produced an unannotated error -- and it is the check that fires
            # first, so it masked the annotated component-count check just below on the common
            # "wrong number of components" mistake.
            raise ContractError(
                "MixtureEstimator.estimate(suff_stat)",
                "%d finite non-negative component weight counts" % num_components,
                str(exc),
                "pass the 2-tuple produced by MixtureAccumulator.value() for this "
                "%d-component mixture." % num_components,
            ) from exc
        if not isinstance(comp_suff_stats, (tuple, list)) or len(comp_suff_stats) != num_components:
            raise ContractError(
                "MixtureEstimator.estimate(suff_stat)",
                "%d component weight counts and %d component sufficient statistics" % (num_components, num_components),
                "%d component weight counts and %s component sufficient statistics"
                % (
                    len(counts),
                    len(comp_suff_stats) if isinstance(comp_suff_stats, (tuple, list)) else "non-sequence",
                ),
                "suff_stat must carry exactly %d entries per side, matching MixtureEstimator's %d "
                "component estimators -- a mismatched MixtureAccumulator/MixtureEstimator component "
                "count is the usual cause." % (num_components, num_components),
            )
        _keys = self.keys if isinstance(self.keys, (tuple, list)) and len(self.keys) == 2 else (None, None)
        if _keys[0] is None:
            # a POOLED component-count vector carries the mass of every site sharing the
            # weight key, so comparing it against THIS site's observation count stops being a
            # corruption check -- the unconditional version rejected every weight-keyed fit's
            # M-step (measured end-to-end in the latent-family mutator audit); an unkeyed one
            # is still bounded by the observations that produced it
            validate_effective_sample_mass(
                nobs,
                float(counts.sum()),
                label="mixture effective sample",
                allow_unassigned=True,
            )

        components = []
        for i in range(num_components):
            try:
                components.append(self.estimators[i].estimate(counts[i], comp_suff_stats[i]))
            except ContractError as e:
                raise prefix_contract_error("MixtureDistribution.components[%d]" % i, e) from None

        if self.has_conj_prior and self.fixed_weights is None:
            # Conjugate Dirichlet weight update: MAP weights w_k proportional to
            # (count_k + alpha_k - 1), clamped at the simplex boundary; the posterior
            # Dirichlet(alpha + counts) is carried forward as the new weight prior.
            DirichletDistribution, SymmetricDirichletDistribution = _dirichlet_types()
            if isinstance(self.prior, SymmetricDirichletDistribution):
                alpha = np.ones(num_components) * float(self.prior.get_parameters())
            else:
                alpha = np.asarray(self.prior.get_parameters(), dtype=float)

            cpp = np.add(counts, alpha) - 1.0
            cpp = np.maximum(cpp, 0.0)

            if cpp.sum() == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = cpp / cpp.sum()

            return MixtureDistribution(
                components, w, name=self.name, prior=DirichletDistribution(np.add(counts, alpha))
            )

        if self.fixed_weights is not None:
            w = np.asarray(self.fixed_weights)

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / num_components
            w = counts + p
            w /= w.sum()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            w = (counts + self.suff_stat * self.pseudo_count) / (counts.sum() + self.pseudo_count)

        else:
            nobs_loc = counts.sum()

            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        # P3 MLE weight floor: clamp component weights at >= w_min and renormalize so a
        # collapsing component cannot reach exactly zero weight (which would permanently
        # freeze it out of subsequent EM iterations). Only applied on the plain MLE path
        # (not fixed_weights / conjugate-prior paths) and only when w_min > 0.
        if self.w_min > 0.0 and self.fixed_weights is None:
            w = np.asarray(w, dtype=float)
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.maximum(w, self.w_min)
            w = w / w.sum()

        return MixtureDistribution(components, w, name=self.name)


class _HeteroMixtureEncoded:
    """Per-component encodings for a heterogeneous mixture (components of differing families).

    Wraps a tuple ``encodings[i]`` holding the sequence encoding produced by component ``i``'s
    own encoder, so each component is fed the encoding its ``seq_*`` methods expect. Homogeneous
    mixtures bypass this wrapper entirely and encode once (bit-identical to the legacy path).
    """

    __slots__ = ("encodings",)

    def __init__(self, encodings: tuple[Any, ...]) -> None:
        self.encodings = encodings


class _SharedMixtureEncoded:
    """One shared nested-mixture encoding, retaining the current mixture depth."""

    __slots__ = ("encoding",)

    def __init__(self, encoding: Any) -> None:
        self.encoding = encoding


def _component_enc(enc_data: Any, i: int) -> Any:
    """Select the encoding destined for component ``i``.

    For a homogeneous mixture (single shared encoding) this returns ``enc_data`` unchanged; for a
    heterogeneous mixture it returns that component's own encoding from the wrapper.
    """
    if isinstance(enc_data, _SharedMixtureEncoded):
        return enc_data.encoding
    if isinstance(enc_data, _HeteroMixtureEncoded):
        return enc_data.encodings[i]
    return enc_data


def _child_encoding_signature(encoder):
    """A child encoder's column-layout identity, or the encoder itself when it declares none."""
    signature = getattr(encoder, "encoding_signature", None)
    if callable(signature):
        try:
            return signature()
        except (TypeError, ValueError):
            return encoder
    return encoder


class MixtureDataEncoder(DataSequenceEncoder):
    """Encoder for homogeneous or heterogeneous mixture component encodings."""

    def __init__(self, encoder: DataSequenceEncoder | Sequence[DataSequenceEncoder]) -> None:
        """Create an encoder for data scored by all mixture components.

        Observations must be valid for every component distribution in the mixture.

        Components may belong to different distribution families. When the per-component encoders
        are all equal the mixture encodes the data once and shares it (bit-identical to the legacy
        single-encoder behaviour); when they differ each component's data is encoded separately and
        carried in a :class:`_HeteroMixtureEncoded` wrapper.

        Args:
            encoder: A single DataSequenceEncoder (shared by all components) or a sequence of
                per-component DataSequenceEncoder objects.

        Attributes:
            encoders (list[DataSequenceEncoder]): Per-component encoders.
            encoder (DataSequenceEncoder): First component encoder (kept for backward compatibility).
            homogeneous (bool): True when all component encoders are equal.

        """
        if isinstance(encoder, DataSequenceEncoder):
            encoders: list[DataSequenceEncoder] = [encoder]
        else:
            encoders = list(encoder)
        self.encoders = encoders
        self.encoder = encoders[0]
        self.homogeneous = all(e == encoders[0] for e in encoders)

    def encoding_signature(self) -> tuple:
        """Column-layout identity of this mixture encoder: its kind plus each child's signature.

        Composes so that a leaf which distinguishes "same encoder" from "same encoding" (see
        ``BinomialDataEncoder.encoding_signature``) is not overruled by a wrapper falling back to
        ``__eq__``. A child that does not implement it contributes itself, so equality still decides
        for that branch.
        """
        return ("mixture", tuple(_child_encoding_signature(e) for e in (self.encoders or (self.encoder,))))

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        if self.homogeneous:
            return "MixtureDataEncoder(" + str(self.encoder) + ")"
        return "MixtureDataEncoder([" + ", ".join(str(e) for e in self.encoders) + "])"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent mixture data encoder.

        If 'other' object is a MixtureDataEncoder, the two must carry equivalent per-component
        encoders. If 'other' is not a MixtureDataEncoder, it is compared against the (single) shared
        encoder, preserving the legacy homogeneous-mixture behaviour.

        Args:
            other (object): Object to be compared to MixtureDataEncoder instance.

        Returns:
            bool.

        """
        if not isinstance(other, MixtureDataEncoder):
            return self.homogeneous and self.encoder == other
        if len(other.encoders) != len(self.encoders):
            return False
        return all(a == b for a, b in zip(self.encoders, other.encoders))

    def seq_encode(self, x: Sequence[T]) -> Any:
        """Sequence encode a sequence of iid observations drawn from the mixture distribution.

        For a homogeneous mixture this delegates to the single shared component encoder. For a
        heterogeneous mixture each component encoder encodes the data separately and the encodings
        are bundled in a :class:`_HeteroMixtureEncoded` wrapper.

        Args:
            x (Sequence[T]): A Sequence of iid observations drawn from a mixture distribution with
                component distributions consistent with the per-component encoders.

        Returns:
            Encoded sequence (single shared encoding, or a per-component wrapper).

        """
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise ContractError(
                "MixtureDistribution.seq_encode",
                "a sequence of observations (all components share the same observation type)",
                "%s" % type(x).__name__,
                "pass a list/tuple of observations, e.g. [x0, x1, ...].",
            )
        if self.homogeneous:
            try:
                encoded = self.encoder.seq_encode(x)
                if isinstance(encoded, (_HeteroMixtureEncoded, _SharedMixtureEncoded)):
                    return _SharedMixtureEncoded(encoded)
                return encoded
            except ContractError as e:
                raise prefix_contract_error("MixtureDistribution.components", e) from None
            except (TypeError, ValueError, IndexError, KeyError) as e:
                raise ContractError(
                    "MixtureDistribution.components",
                    "every observation compatible with the shared component data type",
                    "data that raised %s: %s" % (type(e).__name__, e),
                    "check that every observation matches the data type expected by the mixture's "
                    "components (%s)." % self.encoder,
                ) from e
        try:
            return _HeteroMixtureEncoded(tuple(e.seq_encode(x) for e in self.encoders))
        except (TypeError, ValueError) as e:
            raise TypeError(
                "MixtureDistribution could not encode the data with all of its component encoders. A "
                "finite mixture treats the component as LATENT, so every component must accept the same "
                "observation type and be able to score every observation. For data of DISJOINT types -- "
                "e.g. a mix of strings and numbers where the type already identifies the component -- the "
                "component is OBSERVED, not latent: use a weighted SelectDistribution (a dispatch "
                "mixture) whose choice function routes each observation to the matching child. "
                "Original error: %s" % e
            ) from e

    def row_count(self, x: Any) -> int:
        """Return the common observation count across shared or heterogeneous encodings."""
        if isinstance(x, _SharedMixtureEncoded):
            return self.encoder.row_count(x.encoding)
        if isinstance(x, _HeteroMixtureEncoded):
            if len(x.encodings) != len(self.encoders):
                raise ValueError("heterogeneous mixture encoded arity does not match its encoders.")
            counts = tuple(encoder.row_count(payload) for encoder, payload in zip(self.encoders, x.encodings))
            if not counts or any(count != counts[0] for count in counts[1:]):
                raise ValueError("heterogeneous mixture encodings have inconsistent row counts.")
            return counts[0]
        return self.encoder.row_count(x)


# --- Fisher view(s) co-located with this family ---
class MixtureFisherView(FixedFisherView):
    """Complete-data Fisher view for finite mixture distributions.

    Coordinates are component assignment indicators followed by each
    component's sufficient statistics gated by that assignment.  Observed data
    map to posterior-expected complete-data statistics.
    """

    def __init__(self, dist: Any) -> None:
        self.child_views = [to_fisher(d) for d in dist.components]
        labels = self._labels_from_children()
        super().__init__(dist, labels)

    def _labels_from_children(self) -> list[Path]:
        labels: list[Path] = [("component", str(k)) for k in range(len(self.child_views))]
        for k, view in enumerate(self.child_views):
            labels.extend(("component_stat", str(k)) + label for label in view.vectorizer.labels)
        return labels

    def _refresh_labels(self) -> None:
        self.labels = self._labels_from_children()
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    def _posterior_from_data(self, data: Sequence[Any]) -> np.ndarray:
        return np.asarray([self.dist.posterior(x) for x in data], dtype=np.float64)

    def _posterior_from_encoded(self, enc_data: Any) -> np.ndarray:
        return np.asarray(self.dist.seq_posterior(enc_data), dtype=np.float64)

    def _component_stats_from_data(self, data: Sequence[Any]) -> list[np.ndarray]:
        return [view.expected_statistics_matrix(data=data) for view in self.child_views]

    def _component_stats_from_encoded(self, enc_data: Any) -> list[np.ndarray]:
        return [view.seq_expected_statistics(enc_data) for view in self.child_views]

    @staticmethod
    def _join_stats(z: np.ndarray, child_stats: Sequence[np.ndarray]) -> np.ndarray:
        blocks = [z]
        for k, stats in enumerate(child_stats):
            blocks.append(z[:, [k]] * stats)
        return np.hstack(blocks)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        values = list(data)
        z = self._posterior_from_data(values)
        mats = self._component_stats_from_data(values)
        self._refresh_labels()
        return self._join_stats(z, mats)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        z = self._posterior_from_encoded(enc_data)
        mats = self._component_stats_from_encoded(enc_data)
        self._refresh_labels()
        return self._join_stats(z, mats)

    def structured_statistics(self, x: Any, estimate: Any | None = None, weight: float = 1.0) -> Any:
        """Return mixture responsibility statistics and weighted component statistics for one observation."""
        z = self.dist.posterior(x) if estimate is None else estimate.posterior(x)
        child_values = tuple(z[k] * self.child_views[k].sufficient_statistics(x) for k in range(len(self.child_views)))
        return weight * z, child_values

    def _component_means(self) -> list[np.ndarray]:
        return [np.asarray(view.mean_statistics(), dtype=np.float64) for view in self.child_views]

    def _component_moments(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        means = self._component_means()
        infos = [np.asarray(view.fisher_information(ridge=0.0), dtype=np.float64) for view in self.child_views]
        return means, infos

    def _model_mean(self) -> np.ndarray:
        w = np.asarray(self.dist.w, dtype=np.float64)
        means = self._component_means()
        return np.concatenate([w] + [w[k] * means[k] for k in range(len(means))])

    def _model_fisher(self) -> np.ndarray:
        w = np.asarray(self.dist.w, dtype=np.float64)
        means, infos = self._component_moments()
        k_count = len(means)
        dims = [len(mu) for mu in means]
        offsets = []
        pos = k_count
        for dim in dims:
            offsets.append(pos)
            pos += dim

        out = np.zeros((pos, pos), dtype=np.float64)
        out[:k_count, :k_count] = np.diag(w) - np.outer(w, w)

        for i in range(k_count):
            for k in range(k_count):
                cov = ((w[k] if i == k else 0.0) - w[i] * w[k]) * means[k]
                s = offsets[k]
                e = s + dims[k]
                out[i, s:e] = cov
                out[s:e, i] = cov

        for k in range(k_count):
            sk = offsets[k]
            ek = sk + dims[k]
            muk = means[k]
            out[sk:ek, sk:ek] = w[k] * infos[k] + w[k] * (1.0 - w[k]) * np.outer(muk, muk)
            for l in range(k + 1, k_count):
                sl = offsets[l]
                el = sl + dims[l]
                block = -w[k] * w[l] * np.outer(muk, means[l])
                out[sk:ek, sl:el] = block
                out[sl:el, sk:ek] = block.T

        return out
