"""Create, estimate, and sample from a mixture distribution with homogenous components.

Defines the MixtureDistribution, MixtureSampler, MixtureAccumulatorFactory, MixtureAccumulator,
MixtureEstimator, and the MixtureDataEncoder classes for use with pysparkplug.

MixtureDistribution is defined by the density of the form,

P(Y) = sum_{k=1}^{K} P(Y|Z=k)*P(Z=k),

where P(Z=k) is a mixture weight for component k, and P(Y|Z=k) is defined as a the k^{th} component distribution.

If component distribution P(Y|Z=k) has data type (T), then the Mixture distribution has data type (T) as well.

"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.engines.arithmetic import maxrandint
from pysp.enumeration.algorithms import (
    BufferedStream,
    QuantizedEnumerationIndex,
    best_first_union,
    bounded_best_first_union_index,
    freeze,
)
from pysp.stats.bayes.dirichlet import DirichletDistribution
from pysp.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution
from pysp.stats.compute.pdist import (
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
from pysp.stats.compute.posterior import CategoricalLatentPosterior
from pysp.utils.aliasing import MISSING, coalesce_alias
from pysp.utils.special import digamma

T = TypeVar("T")  ### Type of Mixture component data.
T1 = TypeVar("T1")  ### Type of encoded data.
T2 = TypeVar("T2")  ### Type of component suff_stat
key_type = tuple[str, str] | tuple[None, None]


def mixture_prior(
    weight_prior: SequenceEncodableProbabilityDistribution,
    component_priors: Sequence[SequenceEncodableProbabilityDistribution],
) -> tuple[SequenceEncodableProbabilityDistribution, tuple[SequenceEncodableProbabilityDistribution, ...]]:
    """Build the joint mixture prior: a weight prior plus one prior per component.

    Args:
        weight_prior: Prior on the mixture weights (a
            :class:`~pysp.stats.bayes.dirichlet.DirichletDistribution` or
            :class:`~pysp.stats.bayes.symmetric_dirichlet.SymmetricDirichletDistribution`).
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
        return alpha, digamma(alpha) - digamma(np.sum(alpha))
    if isinstance(prior, SymmetricDirichletDistribution):
        alpha = np.ones(num_components) * prior.get_parameters()
        return alpha, digamma(alpha) - digamma(np.sum(alpha))
    return None, None


class MixtureDistribution(SequenceEncodableProbabilityDistribution):
    """MixtureDistribution object defined by component distributions and weights.

    The args components (Sequence[SequenceEncodableProbabilityDistribution]) define the component distributions
    of the mixture distribution as well as the data type. The data type of the MixtureDistribution object is taken
    to be the data type (T) of the component distributions (all must be the same subclass of
    SequenceEncodableProbabilityDistribution super class).

    Args:
        components (Sequence[SequenceEncodableProbabilityDistribution]): Set component distributions.
            Must be same subclass of SequenceEncodableProbabilityDistribution super class with type T.
        w (ndarray[float]): Mixture weights, must sum to 1.0.
        name (Optional[str]): Assign string name to MixtureDistribution object.

    Attributes:
        components (List[SequenceEncodableProbabilityDistribution]): List of component distributions (data type T).
        w (ndarray[float]): Mixture weights assigned from args (w).
        name (Optional[str]): String name to MixtureDistribution object.
        zw (ndarray[bool]): True if a weight is 0.0, else False.
        log_w (ndarray[float]): Log of weights (w). set to -np.inf, where zw is True.
        num_components (int): Number of components in MixtureDistribution instance.

    """

    def compute_capabilities(self):
        from pysp.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

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
        from pysp.stats.compute.declarations import (
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
        """Return string representation of MixtureDistribution object instance."""
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
        ll = np.asarray([u.seq_expected_log_density(x) for u in self.components]).T + cc
        ml = np.max(ll, axis=1, keepdims=True)
        return np.log(np.sum(np.exp(ll - ml), axis=1)) + ml.flatten()

    def density(self, x: T) -> float:
        """Evaluate density of Mixture distribution at observation x.

        See log_density() for details.

        Args:
            x: (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        """Evaluate log-density of Mixture distribution at observation x.

        A K-component Mixture has log-density,

            log(P(x)) = log(sum_{z=k}^{K} P(x|z=k)*P(z=k)),

        where P(x|z=k) is component-k log-density at x, and P(z=k) = w[k]. A log-sum-exp is used to evaluate the
        sum inside the log of the right-hand side above. (See pysp.utils.vector.log_sum() for details).

        Args:
            x: (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            Log-density at x.

        """
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def conditional(self, observed: dict[int, float]) -> "MixtureDistribution":
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
        x_o = np.array([observed[i] for i in obs_idx], dtype=float)
        log_post = np.array(
            [self.log_w[k] + self.components[k].marginal(obs_idx).log_density(x_o) for k in range(self.num_components)]
        )
        log_post -= vec.log_sum(log_post)
        new_components = [c.condition(observed) for c in self.components]
        return MixtureDistribution(new_components, np.exp(log_post))

    def component_log_density(self, x: T) -> np.ndarray:
        """Evaluate component-wise log-density of Mixture distribution at observation x.

        A K-component Mixture has log-density, log(P(x|z=k)) for the K-th component.

        Args:
            x: (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            Numpy array of floats containing component-wise log-density at x.

        """
        return np.asarray([m.log_density(x) for m in self.components], dtype=np.float64)

    def posterior(self, x: T) -> np.ndarray:
        """Obtain the posterior distribution for each mixture component at observation x.

        The posterior distribution of component 'k' at observation x is given by,

            (1) p_mat(Z=k|x) = p_mat(x|Z=k)*p_mat(z=k) / p_mat(x),

        where

            (2) p_mat(x) = sum_{k=1}^{K} p_mat(x|Z=k)*p_mat(z=k) = sum_{k=1}^{K} p_mat(x|Z=k)*w[k].


        This function returns an ndarray[float] of length K, containing p_mat(Z=k|x) as its k^{th} entry.

        Args:
            x: (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            Numpy array of floats containing posterior distribution at observation x.

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
        """Vectorized evaluation of component-wise log-density for encoded sequence x.

        Arg x must be a sequence encoded from MixtureDataEncoder.seq_encode(data) with data type Sequence[T] for data,
        or results from an equivalent encoding from a DataSequenceEncoder object for the components. The resulting
        encoded sequence is assumed to be data type T1.

        Creates a 2-d numpy array of floats with vectorized evaluations of component_log_density() stored in the rows
        corresponding to an observation in encoded sequence x.

        The returned value is an ndarray[float] with shape (sz,K), where K is the number of mixture components, and
        sz is the number of iid observations in the encoded sequence x.

        Args:
            x (T1): See above for details.

        Returns:
            2-d numpy array of floats having shape (sz,K), where sz is the number of iid obs in encoded sequence x, and
            K is the number of mixture components.

        """
        enc_data = x
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
                if not ll_mat_init:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                    ll_mat_init = True
                ll_mat[:, i] = temp

        return ll_mat

    def seq_log_density(self, x: T1) -> np.ndarray:
        """Vectorized evaluation of log-density for encoded sequence x.

        Arg x must be a sequence encoded from MixtureDataEncoder.seq_encode(data) with data type Sequence[T] for data,
        or results from an equivalent encoding from a DataSequenceEncoder object for the components. The resulting
        encoded sequence is assumed to be data type T1.

        Evaluates the log-density of each observation in the encoded sequence x (see log_density() for details).

        The returned value is an ndarray[float] with shape (sz,K), where K is the number of mixture components, and
        sz is the number of iid observations in the encoded sequence x.

        Note: A row-wise log-sum-exp is performed for numerical stability. If a row contains a log-density value of,
         -np.inf is returned for the corresponding observation value in the encoded sequence x.

        Args:
            x (T1): See above for details.

        Returns:
            Numpy array of floats containing the log_density of each observation in encoded sequence.

        """
        enc_data = x
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
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
        from pysp.stats.compute.backend import backend_seq_log_density

        scores = []
        for i in range(self.num_components):
            if self.zw[i]:
                base = backend_seq_log_density(self.components[0], x, engine)
                scores.append(base * 0.0 + engine.asarray(-np.inf))
            else:
                scores.append(backend_seq_log_density(self.components[i], x, engine))
        return engine.stack(scores, axis=1)

    def backend_seq_log_density(self, x: T1, engine: Any) -> Any:
        """Engine-neutral mixture log-density for encoded data."""
        ll_mat = self.backend_seq_component_log_density(x, engine)
        log_w = engine.asarray(self.log_w)
        return engine.logsumexp(ll_mat + log_w, axis=1)

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from pysp.stats.compute.gradient import MixtureGradientFitState

        components = [recurse(component, engine, torch, leaves) for component in self.components]
        w_logits = tensor_param(self.w, engine, torch, transform="logits")
        leaves.append(w_logits)
        return MixtureGradientFitState(self, components, w_logits)

    def seq_posterior(self, x: T1) -> np.ndarray:
        """Vectorized evaluation of posterior of MixtureDistribution for encoded sequence x.

        Arg x must be a sequence encoded from MixtureDataEncoder.seq_encode(data) with data type Sequence[T] for data,
        or results from an equivalent encoding from a DataSequenceEncoder object for the components. The resulting
        encoded sequence is assumed to be data type T1.

        Vectorized evaluation the posterior of each observation in the encoded sequence x (see posterior() for details).

        The returned value is an ndarray[float] with shape (sz,K), where K is the number of mixture components, and
        sz is the number of iid observations in the encoded sequence x. Each row contains the posterior of the
        corresponding encoded observation.

        Note: A row-wise log-sum-exp is performed for numerical stability. If a row contains a log-density value of,
         -np.inf is returned for the corresponding observation value in the encoded sequence x.

        Args:
            x (T1): See above for details.

        Returns:
            Numpy array of floats containing the posterior of each observation in encoded sequence.

        """
        enc_data = x
        ll_mat_init = False

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
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

    def latent_posterior(self, x: Sequence[T]) -> "CategoricalLatentPosterior":
        """Return the latent posterior ``q(z | x)`` over component labels for raw observations ``x``.

        ``q(z)`` is the exact independent-categorical posterior whose marginals are the EM
        responsibilities. The returned :class:`~pysp.stats.compute.posterior.CategoricalLatentPosterior`
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
        from pysp.stats.compute._sampling import scatter_component_draws

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

    def to_fisher(self, **kwargs):
        """Structural Fisher view for the mixture."""
        if hasattr(self, "components") and hasattr(self, "w"):
            from pysp.inference.fisher import MixtureFisherView

            return MixtureFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> "MixtureSampler":
        """Create MixtureSampler for sampling from MixtureDistribution instance.

        Args:
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Returns:
            MixtureSampler object.

        """
        return MixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "MixtureEstimator":
        """Create MixtureEstimator for estimating MixtureDistribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics in estimation.

        Returns:
            MixtureEstimator object.

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

    def dist_to_encoder(self) -> "MixtureDataEncoder":
        """Returns a MixtureDataEncoder object for encoding sequences of iid observations from MixtureDistribution."""
        dist_encoder = self.components[0].dist_to_encoder()
        return MixtureDataEncoder(encoder=dist_encoder)

    def enumerator(self) -> "MixtureEnumerator":
        """Returns a MixtureEnumerator iterating the union of component supports in descending
        mixture probability order."""
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
        from pysp.enumeration.quantization.core import child_count_index
        from pysp.enumeration.quantization.semiring import CountSemiring

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
    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: tuple[str | None, str | None] = (None, None),
        name: str | None = None,
        init: str = "dirichlet",
    ) -> None:
        """MixtureAccumulator object used to aggregate the sufficient statistics of observed data.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Sequence of
                SequenceEncodableStatisticAccumulator objects for the components of the mixture.
            keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and mixture components.
            init (str): Initialization strategy: ``"dirichlet"`` (legacy random responsibilities) or
                ``"kmeans++"`` (k-means++ seeding when the encoded data is a numeric matrix).

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Sequence of
                SequenceEncodableStatisticAccumulator objects for the components of the mixture.
            num_components (int): Total number of mixture components (length of accumulators).
            comp_counts (np.ndarray[float]): Numpy array of floats for accumulating component weights.
            weight_key (Optional[str]): Key for weights of mixture.
            comp_key (Optional[str]): Key for components of mixture.
            _init_rng (bool): False if rng for accumulators has not been set.
            _w_rng (Optional[RandomState]): RandomState for generating weights in init.
            _acc_rng (Optional[List[RandomState]]): List of RandomState obejcts for setting seed on accumulator
                initialization.
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

    def seq_update(self, x: T1, weights: np.ndarray, estimate: "MixtureDistribution") -> None:
        """Vectorized update of sufficient statistics from encoded sequence of observations x.

        Args value x is a sequence encoded sequence of mixture observations. The data type for each mixture observation
        is data type T. T1 is the data type produced by MixtureDataEncoder.seq_encode() function used to encode the
        sequence of type T observations.

        Note: Requires a previous estimate of MixtureDistribution be passed. This may require seq_initialize() to be
        invoked prior to performing seq_update() calls.

        Seq_update is similar to MixtureDistribution.seq_posterior(). Results are aggregated to comp_counts
        and accumulators.

        Args:
            x (T1): See above for details.
            weights (np.ndarray): Numpy array of positive floats.
            estimate (MixtureDistribution): MixtureDistribution object representing previous estimate from EM.

        Returns:
            None.

        """
        enc_data = x
        ll_mat_init = False

        for i in range(estimate.num_components):
            if not estimate.zw[i]:
                temp = estimate.components[i].seq_log_density(enc_data)

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
            self.accumulators[i].seq_update(enc_data, w_loc, estimate.components[i])

    def update(self, x: T, weight: float, estimate: "MixtureDistribution") -> None:
        """Update sufficient statistics of MixtureAccumulator with weighted observation.

        Requires previous estimate of MixtureDistribution.

        Weights posterior of 'estimate' at x. Adds sum to comp_counts, then passes posterior[i] as weight for x
        into update() call of accumulator[i].

        Args:
            x (T): Observation of mixture distribution.
            weight (float): Weight for observation.
            estimate (MixtureDistribution): Previous iteration of EM estimate for MixtureDistribution.

        Returns:
            None.

        """
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior

        for i in range(self.num_components):
            self.accumulators[i].update(x, posterior[i], estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize RandomState objects for accumulators from rng.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _rng_acc member variable.

        Returns:
            None.

        """
        seeds = rng.randint(2**31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: T, weight: float, rng: np.random.RandomState) -> None:
        """Initialize MixtureAccumulator object with weighted observation x.

        If _init_rng is False, _acc_rng is set with rng. This is done for consistency in initialize and seq_initialize
        functions.

        Initialize mixture weights with a sample from Dirichlet distribution. Each SequenceEncodableStatisticAccumulator
        is for the mixture components is initialized with a call to accumulator[i].initialize.

        Args:
            x (T): Observation of mixture distribution.
            weight (float): Weight for observation.
            rng (RandomState): Used to set _acc_rng if not previously set.

        Returns:
            None.

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
        """Vectorized initialization of MixtureAccumulator object for sequence encoded observations x.

        If _init_rng is False, _acc_rng is set with rng. This is done for consistency in initialize and seq_initialize
        functions.

        Vectorized implementation of initialize(), for sequence encoded x.

        Args:
            x (T1): Sequence encoded observations of mixture distribution.
            weights (ndarray[float]): Numpy array of positive valued floats.
            rng (RandomState): Used to set _acc_rng if not previously set.

        Returns:
            None.

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
            self.accumulators[i].seq_initialize(x, ww[:, i], self._acc_rng[i])
            self.comp_counts[i] += np.sum(ww[:, i])

    def _feature_matrix(self, x: Any, keep_idx: np.ndarray) -> np.ndarray | None:
        """Best-effort extraction of a dense (kept_n, d) numeric feature matrix from encoded data.

        Returns ``None`` (so we fall back to the Dirichlet path) when the encoded data is not a
        simple real-valued array — e.g. composite/tuple encodings, ragged sequences, non-numeric
        dtypes. k-means++ only makes sense for vector-space leaves (Gaussian / diagonal Gaussian).
        """
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

    def combine(self, suff_stat: tuple[np.ndarray, tuple[T2, ...]]) -> "MixtureAccumulator":
        """Merge the sufficient statistics of suff_stat with MixtureAccumulator instance.

        Arg suff_stat is a Tuple of length two containing,
            suff_stat[0] (ndarray[float]): Aggregated component counts,
            suff_stat[1] (Tuple[T2,...]): Tuple of K sufficient statistics for the mixture components.

        Note: The components of the mixture are assumed to have sufficient statistics of type T2.

        Args:
            suff_stat: See above for details.

        Returns:
            MixtureAccumulator object.

        """
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Returns sufficient statistics of MixtureAccumulator instance.

        The sufficient statistics value returned (suff_stat) is a Tuple of length two containing,
            suff_stat[0] (ndarray[float]): Aggregated component counts,
            suff_stat[1] (Tuple[T2,...]): Tuple of K sufficient statistics for the mixture components.

        Note: The components of the mixture are assumed to have sufficient statistics of type T2.

        Returns:
            Tuple[np.ndarray[float], Tuple[T2,...,]] described above.

        """
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: tuple[np.ndarray, tuple[T2, ...]]) -> "MixtureAccumulator":
        """Set sufficient statistics of MixtureAccumulator instance to x.

        The sufficient statistics value 'x' is a Tuple of length two containing,
            x[0] (ndarray[float]): Aggregated component counts,
            x[1] (Tuple[T2,...]): Tuple of K sufficient statistics for the mixture components.

        Note: The components of the mixture are assumed to have sufficient statistics of type T2.

        Args:
            x: See above for details.

        Returns:
            MixtureAccumulator object.

        """
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def scale(self, c: float) -> "MixtureAccumulator":
        """Scale component counts and delegate child sufficient statistics."""
        self.comp_counts *= c
        for acc in self.accumulators:
            acc.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Combine the sufficient statistics of MixtureAccumulator instance with other MixtureAccumulator that have
            matching weight or component keys.

        Arg passed stats_dict is Dict[str, Union[np.ndarray, Tuple[T2,...]]. If the key is weight key, stats_dict
        value is a numpy array of floats containing component counts for a Mixture. If the key is a component key,
        the value is a list of SequenceEncodableStatisticAccumulator objects corresponding to the Mixture components.

        Args:
            stats_dict: See above for details.

        Returns:
            None.

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
        """Replace the sufficient statistics of MixtureAccumulator instance with sufficient statistics of matching
            weight and/or component keys found in stats_dict.

        Arg passed stats_dict is Dict[str, Union[np.ndarray, Tuple[T2,...]]. If the key is weight key, stats_dict
        value is a numpy array of floats containing component counts for a Mixture. If the key is a component key,
        the value is a list of SequenceEncodableStatisticAccumulator objects corresponding to the Mixture components.

        Args:
            stats_dict: See above for details.

        Returns:
            None.

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

    def acc_to_encoder(self) -> "MixtureDataEncoder":
        """Returns a MixtureDataEncoder object for encoding sequences of iid observations from MixtureDistribution."""
        acc_encoder = self.accumulators[0].acc_to_encoder()
        return MixtureDataEncoder(encoder=acc_encoder)


class MixtureAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        keys: tuple[str | None, str | None] = (None, None),
        name: str | None = None,
        init: str = "dirichlet",
    ) -> None:
        """MixtureAccumulatorFactory object for creating MixtureAccumulator objects.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Sequence of StatisticAccumulatorFactory for the mixture
                components.
            dim (int): Number of mixture components.
            keys (Tuple[Optional[str], Optional[str]]): Assign keys for weights and component aggregations.
            init (str): Initialization strategy passed to the accumulator (``"dirichlet"`` or ``"kmeans++"``).

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): Sequence of StatisticAccumulatorFactory for the mixture
                components.
            dim (int): Number of mixture components. Must equal length of factories.
            keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components.

        """
        self.factories = factories
        self.keys = keys
        self.name = name
        self.init = init

    def make(self) -> "MixtureAccumulator":
        """Return MixtureAccumulator object with SequenceEncodableStatisticAccumulator objects for the components
        and keys passed."""
        return MixtureAccumulator(
            [factory.make() for factory in self.factories], keys=self.keys, name=self.name, init=self.init
        )


class MixtureEstimator(ParameterEstimator):
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
        """MixtureEstimator object used to estimate MixtureDistribution from aggregated sufficient statistics.

        Args:
            estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimator objects for the mixture
                components.
            fixed_weights (Optional[Union[List[float], np.ndarray]]): Set fixed values for mixture weights.
            suff_stat (Optional[np.ndarray]): Numpy array of floats with length equal to length of estimators.
            pseudo_count (Optional[float]): Used to re-weight the member variable sufficient statistics in estimation.
            name (Optional[str]): Set a name to the MixtureEstimator object.
            keys (Tuple[Optional[str], Optional[str]]): Set keys for the weights and component distributions.
            w_min (float): MLE weight floor (P3). Component weights are clamped at ``>= w_min`` and the
                weight vector is renormalized, so a collapsing component cannot reach exactly zero
                weight (which would freeze it out of all subsequent EM iterations). ``0.0`` (default)
                disables the floor and preserves byte-identical behaviour.
            robust (bool): Enable the bundled robust path (P1 component floors are always on; this
                additionally turns on k-means++ initialization and a small ``w_min`` weight floor).
            init (Optional[str]): Initialization strategy for the accumulator. ``"kmeans++"`` seeds
                responsibilities from k-means++ centers; ``"dirichlet"`` (default unless ``robust``)
                keeps the legacy random-Dirichlet path. ``None`` defers to ``robust``.

        Attributes:
            estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimator objects for the mixture
                components.
            fixed_weights (Optional[np.ndarray]): Treat mixture weights as fixed values. Must sum to 1.0.
            suff_stat (Optional[np.ndarray]): Weights of the mixture. Must sum to 1.0.
            pseudo_count (Optional[float]): Used to re-weight the member variable sufficient statistics in estimation.
            name (Optional[str]): Name for MixtureEstimator object.
            keys (Tuple[Optional[str], Optional[str]]): Keys for the weights and component distributions.

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

    def accumulator_factory(self) -> "MixtureAccumulatorFactory":
        """Returns MixtureAccumulatorFactory object passing component StatisticAccumulatorFactory objects and keys."""
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

    def model_log_density(self, model: "MixtureDistribution") -> float:
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

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, tuple[Any, ...]]) -> "MixtureDistribution":
        """Estimate MixtureDistribution from aggregated sufficient statistics.

        Args suff_stat is a Tuple length two containing:
            suff_stat[0] (np.ndarray): Sufficient statistic for the weights of the mixture components.
            suff_stat[1] (Tuple[T2, ...]): A tuple of length K (number of mixture components), containing the
                sufficient statistics of each mixture component of data type T2.

        If fixed_weights is not None, suff_stat[0] is not used and the weights of the MixtureDistribution are set to
            fixed_weights.

        If pseudo_count is passed, arg suff_stat[0] is aggregated with re-weighted member variable suff_stat. If member
        variable suff_stat is None, then the arg suff_stat[0] is re-weighted with pseudo_count to estimate the weights.

        If pseudo_count is None, ar suff_stat[0] is used to estimate the wieghts.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator super class.
            suff_stat: See above for details.

        Returns:
            MixtureDistribution object.

        """
        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i]) for i in range(num_components)]

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


class MixtureDataEncoder(DataSequenceEncoder):
    def __init__(self, encoder: DataSequenceEncoder) -> None:
        """MixtureDataEncoder used for sequence encoding data for use with vectorized 'seq_' functions.

        Data type: Data must be type T, that matches the data type of each Mixture component.

        Args:
            encoder (DataSequenceEncoder): DataSequenceEncoder corresponding to the component Distributions.

        Attributes:
            encoder (DataSequenceEncoder): DataSequenceEncoder for encoding sequence of iid data.

        """
        self.encoder = encoder

    def __str__(self) -> str:
        """Returns string representation of MixtureDataEncoder object."""
        return "MixtureDataEncoder(" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Checks if an object is equivalent to a MixtureDataEncoder instance.

        If 'other' object is a MixtureDataEncoder, 'other' must have member variable encoder that is equal to
        encoder member variable of MixtureDataEncoder instance.

        If 'other' object is not a MixtureDataEncoder, then 'other' must be equivalent to the encoder of
        MixtureDataEncoder instance.

        Args:
            other (object): Object to be compared to MixtureDataEncoder instance.

        Returns:
            bool.

        """
        if not isinstance(other, MixtureDataEncoder):
            return self.encoder == other
        else:
            if other.encoder == self.encoder:
                return True
            else:
                return False

    def seq_encode(self, x: Sequence[T]) -> Any:
        """Sequence encoder a sequence of iid observations that match the data type of 'encoder' member variable.

        Note: MixtureDataEncoder attribute 'encoder' is an encoder for the components of the MixtureDistribution.
        The data type for 'encoder' is T.

        Args:
            x (Sequence[T]): A Sequence of iid observations drawn from a mixture distribution with component
                distributions consistent with 'encoder'.

        Returns:
            Data encoded sequence produced from a DataSequenceEncoder 'encoder' for data type T.

        """
        return self.encoder.seq_encode(x)
