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

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import maxrandint
from mixle.stats.compute._sampling import scatter_component_draws
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.utils.special import betaln, digamma

default_prior = GammaDistribution(2, 1)


def _prior_cross_entropy(p: Any, q: Any) -> float:
    """Cross entropy between two component priors that may factor over children.

    For a leaf family ``get_prior()`` returns a single distribution and this is
    just ``p.cross_entropy(q)``. For structured families (Composite, Sequence)
    ``get_prior()`` returns a tuple/list of per-child priors; the joint prior
    factors over the children, so the cross entropy is the sum over the matching
    children. ``p`` and ``q`` always share the same nested structure here.
    """
    if p is None or q is None:
        return 0.0
    if isinstance(p, (tuple, list)):
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
        self.set_parameters((a, w, components))
        self.name = name
        self.prior = prior
        self.g = np.asarray(g, dtype=float)
        self.component_priors = list(component_priors)

    def __str__(self) -> str:
        return "DirichletProcessMixtureDistribution([%s], [%s], %s, name=%s, prior=%s)" % (
            ",".join([str(u) for u in self.components]),
            ",".join(map(str, self.v)),
            str(self.a),
            str(self.name),
            str(self.prior),
        )

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the Gamma hyper-posterior on the concentration alpha."""
        return self.prior

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Set the Gamma hyper-prior (or hyper-posterior) on alpha."""
        self.prior = prior

    def get_parameters(self) -> tuple[float, np.ndarray, list[Any]]:
        """Returns the parameter tuple (alpha, weights, component parameters)."""
        return self.a, self.v, [u.get_parameters() for u in self.components]

    def set_parameters(self, params: tuple[float, np.ndarray, Sequence[Any]]) -> None:
        """Set the parameters and refresh the cached log-weights.

        Args:
            params: Tuple (alpha, weights, components).

        """
        a, w, components = params

        self.components = list(components)
        self.max_components = len(components)
        self.num_components = len(components)
        self.w = np.asarray(w, dtype=float)
        self.a = a
        self.log_w = np.log(self.w)
        self.expected_log_nw = self.log_w[-1]
        self.v = self.w

    def density(self, x: Any) -> float:
        """Density of the mixture at observation x; see log_density()."""
        return np.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Mixture log-density log sum_k w_k p(x | theta_k) at observation x."""
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def expected_log_density(self, x: Any) -> float:
        """Mixture log-density with each component's plug-in log-density
        replaced by its variational expectation E_q[log p(x | theta_k)]."""
        return vec.log_sum(np.asarray([u.expected_log_density(x) for u in self.components]) + self.log_w)

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Vectorized log-density at sequence-encoded input x."""
        ll_mat = np.asarray([u.seq_log_density(x) for u in self.components]).T + self.log_w
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
            rv = np.zeros(len(good_rows), dtype=float)
            rv[~good_rows] = -np.inf
            if np.any(good_rows):
                ll_loc = ll_mat[good_rows, :] - ll_max[good_rows]
                rv[good_rows] = (np.log(np.sum(np.exp(ll_loc), axis=1, keepdims=True)) + ll_max[good_rows]).flatten()
            return rv

    def posterior(self, x: Any) -> np.ndarray:
        """Return the component posterior ``p(z = k | x)`` at a single observation.

        This is the plug-in mixture posterior consistent with :meth:`log_density`:
        ``softmax_k( log p(x | theta_k) + log w_k )``. An observation with no support under any
        component falls back to the mixture weights ``w``. Returns a length-K array summing to 1.
        """
        comp_log_density = np.asarray([u.log_density(x) for u in self.components]) + self.log_w
        max_val = np.max(comp_log_density)
        if max_val == -np.inf:
            return self.w.copy()
        comp_log_density -= max_val
        np.exp(comp_log_density, out=comp_log_density)
        comp_log_density /= comp_log_density.sum()
        return comp_log_density

    def seq_posterior(self, x: Any) -> np.ndarray:
        """Vectorized component posterior over a sequence-encoded input.

        Returns an ``(sz, K)`` array whose row ``i`` is the plug-in posterior ``p(z = k | x_i)``
        (see :meth:`posterior`); rows for observations with no support under any component fall back
        to the mixture weights. A row-wise log-sum-exp keeps the softmax numerically stable.
        """
        ll_mat = np.asarray([u.seq_log_density(x) for u in self.components]).T + self.log_w
        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        if np.any(bad_rows):
            ll_mat[bad_rows, :] = self.log_w
            ll_max[bad_rows] = np.max(self.log_w)

        ll_mat = ll_mat - ll_max
        np.exp(ll_mat, out=ll_mat)
        ll_mat /= np.sum(ll_mat, axis=1, keepdims=True)
        return ll_mat

    def seq_local_elbo(self, x: Any) -> np.ndarray:
        """Per-observation local ELBO contributions.

        For each observation i this returns

            sum_k phi_ik * ( E_q[log p(z_i = k | v)] + E_q[log p(x_i | theta_k)] - log phi_ik )

        where phi_i is the optimal variational assignment for x_i. The global
        (data-independent) ELBO terms are returned by
        ``DirichletProcessMixtureEstimator.model_log_density``.
        """
        exp_ll = np.asarray([u.seq_expected_log_density(x) for u in self.components]).T + self.log_w
        max_ell = exp_ll.max(axis=1, keepdims=True)

        phi = np.exp(exp_ll - max_ell)
        phi /= phi.sum(axis=1, keepdims=True)

        # E_q[log p(z_i | v)] via truncated stick-breaking expectations.
        rv = np.sum(phi * _expected_log_stick_weights(self.g), axis=1)

        # E_q[log p(x_i | theta_k)] under the variational assignments
        rv += np.sum(phi * (exp_ll - self.log_w), axis=1)

        # entropy of the local variational multinomials
        log_phi = np.log(phi, out=np.zeros_like(phi), where=phi > 0)
        rv -= np.sum(phi * log_phi, axis=1)

        return rv

    def seq_expected_log_density(self, x: Any) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x."""
        ll = np.asarray([u.seq_expected_log_density(x) for u in self.components]).T + self.log_w
        ml = np.max(ll, axis=1, keepdims=True)
        return (np.log(np.sum(np.exp(ll - ml), axis=1, keepdims=True)) + ml).flatten()

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
            return DirichletProcessMixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count,
            )
        else:
            return DirichletProcessMixtureEstimator([u.estimator() for u in self.components])

    def dist_to_encoder(self) -> "DirichletProcessMixtureDataEncoder":
        """Returns a DirichletProcessMixtureDataEncoder delegating to the components."""
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
        comp_state = self.rng.choice(range(0, len(self.dist.w)), size=size, replace=True, p=self.dist.w)

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
        exp_ll = np.asarray([estimate.components[i].expected_log_density(x) for i in range(self.num_components)])
        exp_ll += estimate.log_w
        exp_ll -= exp_ll.max()

        phi = np.exp(exp_ll)
        phi /= phi.sum()

        self.comp_counts += phi * weight
        self.beta_counts[:, 0] += phi * weight
        self.beta_counts[:, 1] += (1 - np.cumsum(phi)) * weight

        for i in range(self.num_components):
            self.accumulators[i].update(x, phi[i] * weight, estimate.components[i])

    def seq_update(self, x: Any, weights: np.ndarray, estimate: DirichletProcessMixtureDistribution) -> None:
        """Vectorized update() on sequence-encoded data."""
        exp_ll = np.asarray([u.seq_expected_log_density(x) for u in estimate.components]).T
        exp_ll += estimate.log_w
        exp_ll -= exp_ll.max(axis=1, keepdims=True)

        phi = np.exp(exp_ll.T)
        phi /= phi.sum(axis=0, keepdims=True)

        cc_loc = np.dot(phi, weights)
        cs_loc = np.dot((1 - np.cumsum(phi, axis=0)), weights)

        self.comp_counts += cc_loc
        self.beta_counts[:, 0] += cc_loc
        self.beta_counts[:, 1] += cs_loc

        for i in range(self.num_components):
            self.accumulators[i].seq_update(x, phi[i, :] * weights, estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize child RandomState objects for consistent (seq_)initialize."""
        seeds = rng.randint(2**31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initialize with a random Dirichlet assignment of observation x."""
        if not self._init_rng:
            self._rng_initialize(rng)

        p = self._w_rng.dirichlet(np.ones(self.num_components))

        self.comp_counts += p * weight
        self.beta_counts[:, 0] += p * weight
        self.beta_counts[:, 1] += (1 - np.cumsum(p)) * weight

        for i in range(self.num_components):
            self.accumulators[i].initialize(x, p[i] * weight, self._acc_rng[i])

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialize() with random Dirichlet assignments."""
        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        p = self._w_rng.dirichlet(np.ones(self.num_components), size=sz)
        p *= np.reshape(weights, (sz, 1))

        self.comp_counts += p.sum(axis=0)
        self.beta_counts[:, 0] += p.sum(axis=0)
        self.beta_counts[:, 1] += np.dot((1 - np.cumsum(p, axis=1)).T, np.ones(sz))

        for i in range(self.num_components):
            self.accumulators[i].seq_initialize(x, p[:, i], self._acc_rng[i])

    def combine(self, suff_stat: tuple) -> "DirichletProcessMixtureAccumulator":
        """Add another accumulator's sufficient-statistic value into this one."""
        self.comp_counts += suff_stat[0]
        self.beta_counts += suff_stat[1]
        self.a = suff_stat[2]
        self.prev_nw = suff_stat[3]

        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[4][i])
        return self

    def scale(self, c: float) -> "DirichletProcessMixtureAccumulator":
        """Scale linear DP mixture sufficient statistics while preserving metadata."""
        # Scale only the linear count statistics (and the component accumulators); ``a`` (the alpha
        # hyper-posterior) and ``prev_nw`` are non-linear scalar metadata that must stay untouched --
        # the inherited default would multiply them and corrupt the state.
        self.comp_counts *= c
        self.beta_counts *= c
        for u in self.accumulators:
            u.scale(c)
        return self

    def value(self) -> tuple:
        """Returns (comp_counts, beta_counts, alpha, prev_nw, component values)."""
        return self.comp_counts, self.beta_counts, self.a, self.prev_nw, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: tuple) -> "DirichletProcessMixtureAccumulator":
        """Set the sufficient statistics from a value() tuple."""
        self.comp_counts = x[0]
        self.beta_counts = x[1]
        self.a = x[2]
        self.prev_nw = x[3]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[4][i])
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
        self.dim = dim
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
        self.name = name
        self.num_components = len(estimators)
        self.estimators = list(estimators)
        self.keys = keys
        self.prior = prior
        self.pseudo_count = pseudo_count

    def accumulator_factory(self) -> "DirichletProcessMixtureAccumulatorFactory":
        """Returns a DirichletProcessMixtureAccumulatorFactory for this estimator."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return DirichletProcessMixtureAccumulatorFactory(est_factories, self.num_components, self.keys)

    def get_prior(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the Gamma hyper-prior on the concentration alpha."""
        return self.prior

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Set the Gamma hyper-prior on the concentration alpha."""
        self.prior = prior

    def model_log_density(self, model: DirichletProcessMixtureDistribution) -> float:
        """Data-independent ELBO terms of the variational approximation.

        Combines the cross-entropies of the stick-fraction prior and the
        component priors against their variational posteriors with the entropies
        of those posteriors. Together with
        ``DirichletProcessMixtureDistribution.seq_local_elbo`` this forms the full
        ELBO maximized by the fit driver.
        """
        gam = model.g[:-1, :] if model.g.shape[0] > 1 else model.g[:0, :]
        gams = gam[:, 0] + gam[:, 1]
        a = model.a

        # cross entropy of beta and variational betas
        if gam.shape[0] == 0:
            temp1 = 0.0
            temp41 = 0.0
        else:
            temp1 = np.sum(-betaln(1, a) + (digamma(gam[:, 1]) - digamma(gams)) * (a - 1))
            # entropy of variational betas
            temp41 = -(
                betaln(gam[:, 0], gam[:, 1]).sum()
                - ((gam - 1) * digamma(gam)).sum()
                + ((gams - 2) * digamma(gams)).sum()
            )

        # cross entropy of component priors and variational priors
        temp2 = 0.0
        for i in range(model.max_components):
            temp2 += -_prior_cross_entropy(model.components[i].get_prior(), model.component_priors[i])

        # entropy of the variational approximation
        # entropy of variational component priors
        temp42 = np.sum([-_prior_entropy(u.get_prior()) for u in model.components])
        temp4 = temp41 + temp42

        return temp1 + temp2 - temp4

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
        comp_counts, beta_counts, alpha, prev_nw, comp_suff_stats = suff_stat

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

        elif isinstance(self.prior, GammaDistribution):
            s1 = self.prior.k
            s2 = 1 / self.prior.theta

        else:
            s1 = 0.0
            s2 = 0.0
            hyper_posterior = None

        if num_components <= 1:
            if isinstance(self.prior, GammaDistribution):
                new_alpha = s1 / s2
                hyper_posterior = self.prior
            else:
                new_alpha = alpha
        else:
            old_alpha = max(float(alpha), 1.0e-12)
            old_gammas = np.copy(beta_counts)
            old_gammas[:, 0] += 1.0
            old_gammas[:, 1] += old_alpha
            expected_log_remaining = _expected_log_stick_weights(old_gammas)[-1]

            gw1 = s1 + num_components - 1.0
            gw2 = s2 - expected_log_remaining
            new_alpha = gw1 / max(gw2, 1.0e-300)

            if isinstance(self.prior, GammaDistribution):
                hyper_posterior = GammaDistribution(gw1, 1 / gw2)

        gammas = np.copy(beta_counts)
        gammas[:, 0] += 1
        gammas[:, 1] += new_alpha

        expected_log_w = _expected_log_stick_weights(gammas)
        w = np.exp(expected_log_w - np.max(expected_log_w))
        w /= w.sum()

        return DirichletProcessMixtureDistribution(
            components, w, new_alpha, gammas, component_priors, name=self.name, prior=hyper_posterior
        )


class DirichletProcessMixtureDataEncoder(DataSequenceEncoder):
    """Encodes observations with the shared component encoding."""

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        """Create a data encoder for DP-mixture observations.

        Args:
            encoder (DataSequenceEncoder): Encoder for the component distributions.

        """
        self.encoder = encoder

    def __str__(self) -> str:
        return "DirichletProcessMixtureDataEncoder(" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirichletProcessMixtureDataEncoder):
            return self.encoder == other
        return other.encoder == self.encoder

    def seq_encode(self, x: Sequence[Any]) -> Any:
        """Encode a sequence of observations with the component encoder."""
        return self.encoder.seq_encode(x)
