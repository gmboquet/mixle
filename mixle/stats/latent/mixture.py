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

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import (
    BufferedStream,
    QuantizedEnumerationIndex,
    best_first_union,
    bounded_best_first_union_index,
    freeze,
)
from mixle.inference.fisher import Path
from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution
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
from mixle.stats.compute.posterior import CategoricalLatentPosterior
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.special import digamma

T = TypeVar("T")  ### Type of Mixture component data.
T1 = TypeVar("T1")  ### Type of encoded data.
T2 = TypeVar("T2")  ### Type of component suff_stat
key_type = tuple[str, str] | tuple[None, None]


from mixle.inference.fisher import FixedFisherView, SufficientStatisticVectorizer, to_fisher


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
        if isinstance(w, np.ndarray):
            self.w = w
        else:
            self.w = np.asarray(w, dtype=float)

        self.zw = self.w == 0.0
        self.log_w = np.log(w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.components = components
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
        """Variational expected log-density at observation x.

        Uses ``E[log w_k]`` under the (symmetric) Dirichlet weight prior together with each
        component's ``expected_log_density``. Falls back to the plug-in ``log_density(x)``
        when no conjugate weight prior is attached.
        """
        if not self.has_conj_prior:
            return self.log_density(x)
        cc = self.expected_nparams
        return vec.log_sum(np.asarray([u.expected_log_density(x) for u in self.components]) + cc)

    def seq_expected_log_density(self, x: T1) -> np.ndarray:
        """Vectorized variational expected log-density at sequence-encoded input x.

        Falls back to ``seq_log_density(x)`` when no conjugate weight prior is attached.
        """
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
        log_post -= vec.log_sum(log_post)
        new_components = [c.condition(observed) for c in self.components]
        return MixtureDistribution(new_components, np.exp(log_post))

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

        Responsibilities are proportional to ``w[k] * p_k(x)``. If every
        positive-weight component reports an impossible observation, the method
        returns a copy of the prior mixture weights so callers receive a finite
        responsibility vector rather than ``NaN``.

        Args:
            x: Observation accepted by every component family.

        Returns:
            Probability vector over component labels.
        """
        comp_log_density = np.asarray([m.log_density(x) for m in self.components])
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -np.inf

        max_val = np.max(comp_log_density)

        if max_val == -np.inf:
            return self.w.copy()
        else:
            comp_log_density -= max_val
            np.exp(comp_log_density, out=comp_log_density)
            comp_log_density /= comp_log_density.sum()

            return comp_log_density

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

        ll_max = ll_mat.max(axis=1, keepdims=True)
        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return ll_sum.flatten()

        else:
            ll_mat = ll_mat[good_rows, :]
            ll_max = ll_max[good_rows]
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)

            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            rv = np.zeros(good_rows.shape, dtype=float)
            rv[good_rows] = ll_sum.flatten()
            rv[~good_rows] = -np.inf

            return rv

    def backend_seq_component_log_density(self, x: T1, engine: Any) -> Any:
        """Engine-neutral component log densities for encoded data."""
        from mixle.stats.compute.backend import backend_seq_log_density

        scores = []
        for i in range(self.num_components):
            if self.zw[i]:
                base = backend_seq_log_density(self.components[0], _component_enc(x, 0), engine)
                scores.append(base * 0.0 + engine.asarray(-np.inf))
            else:
                scores.append(backend_seq_log_density(self.components[i], _component_enc(x, i), engine))
        return engine.stack(scores, axis=1)

    def backend_seq_log_density(self, x: T1, engine: Any) -> Any:
        """Engine-neutral mixture log-density for encoded data."""
        ll_mat = self.backend_seq_component_log_density(x, engine)
        log_w = engine.asarray(self.log_w)
        return engine.logsumexp(ll_mat + log_w, axis=1)

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
        positive-weight components are impossible fall back to the prior mixture
        weights, matching :meth:`posterior` and avoiding ``NaN`` responsibility
        rows during EM accumulation.

        Args:
            x: Encoded observation batch.

        Returns:
            ``(n, k)`` probability matrix whose rows sum to one.
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

        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w.copy()
        ll_max[bad_rows] = np.max(self.log_w)
        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        ll_mat /= ll_max

        return ll_mat

    def latent_posterior(self, x: Sequence[T]) -> CategoricalLatentPosterior:
        """Return the latent posterior ``q(z | x)`` over component labels for raw observations ``x``.

        ``q(z)`` is the exact independent-categorical posterior whose marginals are the EM
        responsibilities. The returned :class:`~mixle.stats.compute.posterior.CategoricalLatentPosterior`
        can ``.marginals()`` (the responsibilities), ``.sample(rng)`` component labels, ``.mode()``
        (the MAP labels), or ``.entropy()``.
        """
        enc = self.dist_to_encoder().seq_encode(list(x))
        return CategoricalLatentPosterior(self.seq_posterior(enc))

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
        """
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
            child_roles=("component",) * self.num_components,
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

        comp_state = np.asarray(comp_state)
        draws_by_comp = {}
        all_array = True
        for c in range(self.dist.num_components):
            count = int(np.count_nonzero(comp_state == c))
            if count:
                drawn = self.comp_samplers[c].sample(size=count)
                draws_by_comp[c] = drawn
                all_array = all_array and isinstance(drawn, np.ndarray)
        if all_array and draws_by_comp:
            sample = next(iter(draws_by_comp.values()))
            # carry any trailing sample shape (e.g. D-vectors from multivariate
            # leaves) so the scatter is not restricted to scalar draws
            out_arr = np.empty((size,) + sample.shape[1:], dtype=sample.dtype)
            for c, drawn in draws_by_comp.items():
                out_arr[comp_state == c] = drawn
            return list(out_arr)
        out: list[Any] = [None] * size
        for c, drawn in draws_by_comp.items():
            for m, pos in enumerate(np.nonzero(comp_state == c)[0]):
                out[pos] = drawn[m]
        return out


class MixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """EM accumulator for mixture weights and component sufficient statistics."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] = (None, None),
        name: str | None = None,
        init: str = "dirichlet",
    ) -> None:
        """Create an EM accumulator for mixture responsibilities.

        Args:
            accumulators: Component sufficient-statistic accumulators.
            keys: Optional shared-statistic keys for mixture weights and
                component payloads.
            name: Optional diagnostic name.
            init: Initialization strategy. ``"dirichlet"`` draws random
                responsibilities; ``"kmeans++"`` uses numeric encoded features
                when possible and falls back to ``"dirichlet"`` otherwise.

        Attributes:
            comp_counts: Accumulated expected component counts.
            accumulators: Component accumulators receiving responsibility-
                weighted observations.
        """
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.name = name
        self.init = init
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
        Rows where every component is impossible fall back to the estimate's
        mixture weights, so the accumulator receives finite responsibility
        weights rather than ``NaN``.

        Args:
            x: Encoded observation batch.
            weights: Non-negative observation weights.
            estimate: Previous EM iterate used to compute responsibilities.
        """
        enc_data = x
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

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = estimate.log_w.copy()
        ll_max[bad_rows] = np.max(estimate.log_w)

        # Capture the per-row data log-likelihood (== what seq_log_density returns) by reusing the
        # rowmax and rowsum already computed for normalization: row_ll = rowmax + log(rowsum). This
        # is the convergence likelihood, free except an O(n) copy/log, and only when the fused-EM
        # fast path requests it (_track_ll), so the standard path is unaffected.
        track = self._track_ll and ll_mat_init
        rowmax = ll_max[:, 0].copy() if track else None

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)

        if track:
            with np.errstate(divide="ignore"):
                row_ll = rowmax + np.log(ll_max[:, 0])
            if np.any(bad_rows):
                row_ll[bad_rows] = -np.inf
            self._seq_ll += float(np.dot(weights, row_ll))

        np.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max

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
        posterior = estimate.posterior(x)
        posterior *= weight
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

        The default initialization draws a responsibility vector from a
        Dirichlet distribution and delegates responsibility-weighted
        initialization to every component accumulator.

        Args:
            x: Raw observation.
            weight: Observation weight.
            rng: Random state used to seed component initializers.
        """
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
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Return serialized mixture sufficient statistics.

        Returns:
            ``(component_counts, component_suff_stats)`` where the second item
            contains one serialized child accumulator value per component.
        """
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: tuple[np.ndarray, tuple[T2, ...]]) -> MixtureAccumulator:
        """Restore this accumulator from serialized sufficient statistics.

        Args:
            x: ``(component_counts, component_suff_stats)`` tuple.

        Returns:
            ``self`` after restoring child accumulator state.
        """
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def scale(self, c: float) -> MixtureAccumulator:
        """Scale component counts and delegate child sufficient statistics."""
        self.comp_counts *= c
        for acc in self.accumulators:
            acc.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed mixture statistics into a shared statistics dictionary.

        Args:
            stats_dict: Mutable shared sufficient-statistics mapping keyed by
                estimator key names.
        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

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
        """Replace local keyed statistics from a shared statistics dictionary.

        Args:
            stats_dict: Shared sufficient-statistics mapping keyed by estimator
                key names.
        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

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
        init: str = "dirichlet",
    ) -> None:
        """Create a factory for mixture EM accumulators.

        Args:
            factories: Component accumulator factories.
            keys: Optional shared-statistic keys for weights and components.
            name: Optional diagnostic name.
            init: Initialization strategy passed to ``MixtureAccumulator``.
        """
        self.factories = factories
        self.keys = keys
        self.name = name
        self.init = init

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
            robust: Enable the robust default path: k-means++ initialization
                where applicable plus a small data-independent weight floor.
            init: Initialization strategy for the accumulator. ``None`` selects
                ``"kmeans++"`` in robust mode and ``"dirichlet"`` otherwise.
        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_weights = np.asarray(fixed_weights) if fixed_weights is not None else None
        self.robust = bool(robust)
        # In robust mode default to k-means++ init and a tiny data-independent weight floor.
        if init is None:
            init = "kmeans++" if self.robust else "dirichlet"
        self.init = init
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
        if len(counts) != num_components or len(comp_suff_stats) != num_components:
            raise ContractError(
                "MixtureEstimator.estimate(suff_stat)",
                "%d component weight counts and %d component sufficient statistics" % (num_components, num_components),
                "%d component weight counts and %d component sufficient statistics"
                % (len(counts), len(comp_suff_stats)),
                "suff_stat must carry exactly %d entries per side, matching MixtureEstimator's %d "
                "component estimators -- a mismatched MixtureAccumulator/MixtureEstimator component "
                "count is the usual cause." % (num_components, num_components),
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
