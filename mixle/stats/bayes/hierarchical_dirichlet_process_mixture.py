"""Hierarchical Dirichlet process mixture (truncated) for grouped data,
adapted onto the mixle.stats base-class protocol.

Observations arrive in groups (a datum is a sequence of observations). All
groups share K global atoms; each group mixes them with its own weights. This
implements the finite "direct-assignment" truncation of the HDP
(Teh et al. 2006):

    beta             ~ Dirichlet(gamma/K, ..., gamma/K)    (global weights)
    pi_j | beta      ~ Dirichlet(alpha * beta)             (group j weights)
    z_ji | pi_j      ~ pi_j
    x_ji | z_ji = k  ~ components[k]

Estimation alternates:
  - E-step at point estimates: responsibilities phi_jik from the group's
    current weights and the atom densities,
  - posterior-mean update for each group's weights under
    Dirichlet(alpha*beta + expected_counts), deliberately using the mean
    rather than the boundary-degenerate MAP when alpha*beta_k < 1, together
    with the atoms' estimator updates,
  - global-weight update via the standard expected-table-count approximation
    m_jk = alpha*beta_k*(psi(alpha*beta_k + n_jk) - psi(alpha*beta_k)), with
    beta set to the Dirichlet(gamma/K + m_.k) posterior mean. Applying this
    table-count formula to fractional responsibility counts is a deterministic
    approximation, not an exact collapsed-HDP CAVI step.

``seq_local_elbo`` scores training groups with their fitted weights (this is
what the fit driver maximizes); ``seq_log_density`` scores a (possibly new)
group with the global weights beta, i.e. the expected weights of an unseen
group. For multi-observation new groups this is a beta plug-in score, not the
integrated finite-HDP predictive density obtained by integrating over a new
group row pi ~ Dirichlet(alpha*beta).

Group sizes are exogenous unless len_dist is supplied (used for sampling and
added to the per-group score). The length model uses the mixle.stats
NullDistribution/NullEstimator/NullAccumulator family.

This is a port of ``mixle.bstats.hdpm`` onto the ``mixle.stats`` protocol. The
object should be read as the finite direct-assignment approximation described
above, with posterior-mean rows and an expected-table global-row heuristic.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_TINY = 1.0e-300


class HierarchicalDirichletProcessMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Truncated hierarchical DP mixture over K shared atoms with global weights
    beta and (optionally) fitted per-group weights."""

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        beta: np.ndarray | list[float],
        alpha: float,
        gamma: float,
        group_weights: np.ndarray | None = None,
        name: str | None = None,
        len_dist: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create a finite hierarchical Dirichlet-process mixture approximation.

        Args:
            components: List of K shared atom distributions (each carrying its
                own prior).
            beta: Length-K global weight vector.
            alpha (float): Group-level concentration of Dirichlet(alpha*beta).
            gamma (float): Global concentration of Dirichlet(gamma/K).
            group_weights (Optional[np.ndarray]): (J, K) fitted weights of the
                training groups (used by seq_local_elbo); None scores all groups
                with beta.
            name (Optional[str]): Optional distribution name.
            len_dist (Optional): Distribution of group sizes; a NullDistribution
                (the default) treats sizes as exogenous.

        """
        self.name = name
        self.components = list(components)
        self.num_components = len(components)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()

        self.beta = np.asarray(beta, dtype=float)
        with np.errstate(divide="ignore"):
            self.log_beta = np.log(self.beta)

        self.group_weights = None if group_weights is None else np.asarray(group_weights, dtype=float)

    def __str__(self) -> str:
        cstr = ",".join(str(u) for u in self.components)
        bstr = ",".join(map(str, self.beta.tolist()))
        return "HierarchicalDirichletProcessMixtureDistribution([%s], [%s], %f, %f, name=%s, len_dist=%s)" % (
            cstr,
            bstr,
            self.alpha,
            self.gamma,
            self.name,
            str(self.len_dist),
        )

    def get_parameters(self) -> tuple[np.ndarray, float, float, list[Any]]:
        """Returns the parameter tuple (beta, alpha, gamma, component parameters)."""
        return self.beta, self.alpha, self.gamma, [u.get_parameters() for u in self.components]

    def set_parameters(self, params: tuple[np.ndarray, float, float, Sequence[Any]]) -> None:
        """Set the parameters and refresh the cached log-weights."""
        beta, alpha, gamma, comp_params = params
        self.beta = np.asarray(beta, dtype=float)
        with np.errstate(divide="ignore"):
            self.log_beta = np.log(self.beta)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        for c, p in zip(self.components, comp_params):
            c.set_parameters(p)

    def _len_term(self, x: Any) -> float:
        if supports(self.len_dist, Neutral) or self.len_dist is None:
            return 0.0
        return self.len_dist.log_density(len(x))

    def _group_log_density(self, log_b: np.ndarray, log_w: np.ndarray) -> float:
        """Sum over observations of log sum_k w_k p(x_i | theta_k)."""
        ll = log_b + log_w
        ll_max = ll.max(axis=1, keepdims=True)
        good = np.isfinite(ll_max.flatten())
        rv = np.full(len(good), -np.inf)
        if np.any(good):
            rv[good] = (
                np.log(np.sum(np.exp(ll[good, :] - ll_max[good]), axis=1, keepdims=True)) + ll_max[good]
            ).flatten()
        return float(rv.sum())

    def density(self, x: Any) -> float:
        """Density of a group x; see log_density()."""
        return np.exp(self.log_density(x))

    def density_semantics(self):
        """Return density semantics for the expected-weight HDP mixture approximation."""
        from mixle.stats.compute.pdist import DensitySemantics

        return DensitySemantics.ESTIMATE  # plug-in with expected global weights (expected-table-count approx.)

    def log_density(self, x: Any) -> float:
        """Score a group with the global weights beta (expected weights of a
        new group)."""
        if len(x) == 0:
            return self._len_term(x)
        enc = self.components[0].dist_to_encoder().seq_encode(list(x))
        log_b = np.asarray([c.seq_log_density(enc) for c in self.components]).T
        return self._group_log_density(log_b, self.log_beta) + self._len_term(x)

    def seq_encode(self, x: Sequence[Sequence]) -> Any:
        """Encode groups into a flat component encoding with offsets.

        Args:
            x (Sequence[Sequence]): Iterable of groups (sequences of observations).

        Returns:
            Tuple ``(lengths, offsets, flat_enc, len_enc)`` consumed by vectorized ``seq_*`` methods.

        """
        lengths = np.asarray([len(u) for u in x], dtype=int)
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        flat: list[Any] = []
        for u in x:
            flat.extend(u)
        flat_enc = self.components[0].dist_to_encoder().seq_encode(flat)

        if supports(self.len_dist, Neutral) or self.len_dist is None:
            len_enc = None
        else:
            len_enc = self.len_dist.dist_to_encoder().seq_encode(lengths)

        return lengths, offsets, flat_enc, len_enc

    def _emission_log_densities(self, flat_enc: Any) -> np.ndarray:
        return np.asarray([c.seq_log_density(flat_enc) for c in self.components]).T

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Vectorized log_density() at sequence-encoded input x (each group
        scored with the global weights beta)."""
        lengths, offsets, flat_enc, len_enc = x
        log_b_all = self._emission_log_densities(flat_enc)

        rv = np.zeros(len(lengths))
        for j in range(len(lengths)):
            if lengths[j] == 0:
                continue
            rv[j] = self._group_log_density(log_b_all[offsets[j] : offsets[j + 1], :], self.log_beta)

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def seq_local_elbo(self, x: Any) -> np.ndarray:
        """Per-group data term of the penalized objective: training groups are
        scored with their own fitted weights. Falls back to beta when the group
        count does not match the fit."""
        lengths, offsets, flat_enc, len_enc = x

        if self.group_weights is None or len(self.group_weights) != len(lengths):
            return self.seq_log_density(x)

        log_b_all = self._emission_log_densities(flat_enc)
        with np.errstate(divide="ignore"):
            log_gw = np.log(np.maximum(self.group_weights, _TINY))

        rv = np.zeros(len(lengths))
        for j in range(len(lengths)):
            if lengths[j] == 0:
                continue
            rv[j] = self._group_log_density(log_b_all[offsets[j] : offsets[j + 1], :], log_gw[j, :])

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def group_posteriors(self, x: Sequence[Sequence]) -> np.ndarray:
        """Posterior atom-usage (mean responsibility) per group.

        Groups are scored with their fitted weights when available (else with the
        global weights beta).
        """
        lengths, offsets, flat_enc, len_enc = self.seq_encode(x)
        log_b_all = self._emission_log_densities(flat_enc)
        with np.errstate(divide="ignore"):
            log_gw = (
                np.log(np.maximum(self.group_weights, _TINY))
                if self.group_weights is not None and len(self.group_weights) == len(lengths)
                else np.tile(self.log_beta, (len(lengths), 1))
            )

        rv = np.zeros((len(lengths), self.num_components))
        for j in range(len(lengths)):
            if lengths[j] == 0:
                continue
            ll = log_b_all[offsets[j] : offsets[j + 1], :] + log_gw[j, :]
            ll -= ll.max(axis=1, keepdims=True)
            phi = np.exp(ll)
            phi /= phi.sum(axis=1, keepdims=True)
            rv[j, :] = phi.mean(axis=0)
        return rv

    def sampler(self, seed: int | None = None) -> "HierarchicalDirichletProcessMixtureSampler":
        """Create a HierarchicalDirichletProcessMixtureSampler for this distribution."""
        return HierarchicalDirichletProcessMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HierarchicalDirichletProcessMixtureEstimator":
        """Create a HierarchicalDirichletProcessMixtureEstimator from this
        distribution's components, concentrations, and length estimator."""
        len_est = NullEstimator() if supports(self.len_dist, Neutral) else self.len_dist.estimator()
        return HierarchicalDirichletProcessMixtureEstimator(
            [u.estimator() for u in self.components],
            gamma=self.gamma,
            alpha=self.alpha,
            name=self.name,
            len_estimator=len_est,
        )

    def dist_to_encoder(self) -> "HierarchicalDirichletProcessMixtureDataEncoder":
        """Returns a HierarchicalDirichletProcessMixtureDataEncoder for this distribution."""
        comp_encoder = self.components[0].dist_to_encoder()
        len_encoder = None if supports(self.len_dist, Neutral) else self.len_dist.dist_to_encoder()
        return HierarchicalDirichletProcessMixtureDataEncoder(comp_encoder, len_encoder)


class HierarchicalDirichletProcessMixtureSampler(DistributionSampler):
    """Draws groups from a HierarchicalDirichletProcessMixtureDistribution
    (per-group weights drawn from Dirichlet(alpha*beta))."""

    def __init__(self, dist: HierarchicalDirichletProcessMixtureDistribution, seed: int | None = None) -> None:
        """Create a sampler for the finite HDP-mixture approximation."""
        rng = RandomState(seed)
        self.rng = RandomState(rng.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [u.sampler(seed=rng.randint(0, maxrandint)) for u in dist.components]
        if supports(dist.len_dist, Neutral) or dist.len_dist is None:
            self.len_sampler = None
        else:
            self.len_sampler = dist.len_dist.sampler(seed=rng.randint(0, maxrandint))

    def sample_group(self, n: int | None = None) -> list[Any]:
        """Draw a single group of n observations.

        Group weights pi ~ Dirichlet(alpha*beta) are drawn once for the group,
        then each observation draws an atom from pi.
        """
        if n is None:
            if self.len_sampler is None:
                raise ValueError("HDP sampler requires a len_dist (or explicit n) to sample groups.")
            n = int(self.len_sampler.sample())

        pi = self.rng.dirichlet(np.maximum(self.dist.alpha * self.dist.beta, 1.0e-8))
        states = self.rng.choice(self.dist.num_components, size=n, p=pi)
        return [self.comp_samplers[k].sample() for k in states]

    def sample(self, size: int | None = None) -> Any:
        """Draw size groups (a single group when size is None)."""
        if size is None:
            return self.sample_group()
        return [self.sample_group() for _ in range(size)]


class HierarchicalDirichletProcessMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulates HDP mixture sufficient statistics: per-group expected atom
    counts (in data order) plus each atom's weighted statistics."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        len_accumulator: SequenceEncodableStatisticAccumulator | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an accumulator for HDP-mixture sufficient statistics."""
        self.accumulators = list(accumulators)
        self.num_components = len(accumulators)
        self.name = name
        self.keys = keys
        self.group_counts: list[np.ndarray] = []  # one (K,) count vector per group, in data order
        self.prev_beta: np.ndarray | None = None
        self.prev_alpha: float | None = None
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initialize with random Dirichlet assignments for group x."""
        counts = np.zeros(self.num_components)
        for u in x:
            p = rng.dirichlet(np.ones(self.num_components))
            counts += p * weight
            for k in range(self.num_components):
                self.accumulators[k].initialize(u, p[k] * weight, rng)
        self.group_counts.append(counts)

        if not supports(self.len_accumulator, Neutral):
            self.len_accumulator.update(len(x), weight, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialize() with random Dirichlet assignments."""
        lengths, offsets, flat_enc, len_enc = x
        tot = int(lengths.sum())

        phi = rng.dirichlet(np.ones(self.num_components), size=tot)
        seq_w = np.repeat(weights, lengths)

        for j in range(len(lengths)):
            sl = slice(offsets[j], offsets[j + 1])
            self.group_counts.append(
                np.dot(phi[sl, :].T, np.repeat(weights[j], lengths[j]))
                if lengths[j] > 0
                else np.zeros(self.num_components)
            )

        for k in range(self.num_components):
            self.accumulators[k].seq_initialize(flat_enc, phi[:, k] * seq_w, rng)

        if len_enc is not None and not supports(self.len_accumulator, Neutral):
            self.len_accumulator.seq_initialize(len_enc, weights, rng)

    def update(self, x: Any, weight: float, estimate: HierarchicalDirichletProcessMixtureDistribution) -> None:
        """Accumulate the E-step statistics for one group (delegates to seq_update
        on a singleton encoding)."""
        enc = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc, np.asarray([weight]), estimate)

    def seq_update(
        self, x: Any, weights: np.ndarray, estimate: HierarchicalDirichletProcessMixtureDistribution
    ) -> None:
        """E-step on sequence-encoded data at the current point estimates.

        Computes responsibilities phi from each group's current weights (the
        fitted group weights, or beta for new/unmatched groups) and the atom
        densities, recording per-group expected counts and pushing phi-weighted
        updates into the atom accumulators. Also records the estimate's beta and
        alpha for the estimator's global-weight update.
        """
        lengths, offsets, flat_enc, len_enc = x

        self.prev_beta = estimate.beta
        self.prev_alpha = estimate.alpha

        log_b_all = estimate._emission_log_densities(flat_enc)

        gw = estimate.group_weights
        if gw is None or len(gw) != len(lengths):
            gw = np.tile(estimate.beta, (len(lengths), 1))

        with np.errstate(divide="ignore"):
            log_gw = np.log(np.maximum(gw, _TINY))

        phi_all = np.zeros_like(log_b_all)
        for j in range(len(lengths)):
            sl = slice(offsets[j], offsets[j + 1])
            counts = np.zeros(self.num_components)
            if lengths[j] > 0:
                ll = log_b_all[sl, :] + log_gw[j, :]
                ll -= ll.max(axis=1, keepdims=True)
                phi = np.exp(ll)
                phi /= phi.sum(axis=1, keepdims=True)
                phi *= weights[j]
                phi_all[sl, :] = phi
                counts = phi.sum(axis=0)
            self.group_counts.append(counts)

        for k in range(self.num_components):
            self.accumulators[k].seq_update(flat_enc, phi_all[:, k], estimate.components[k])

        if len_enc is not None and not supports(self.len_accumulator, Neutral):
            self.len_accumulator.seq_update(len_enc, weights, None)

    def combine(self, suff_stat: tuple) -> "HierarchicalDirichletProcessMixtureAccumulator":
        """Add another accumulator's sufficient-statistic value into this one."""
        self.group_counts.extend(suff_stat[0])
        if suff_stat[1] is not None:
            self.prev_beta = suff_stat[1]
            self.prev_alpha = suff_stat[2]
        for k in range(self.num_components):
            self.accumulators[k].combine(suff_stat[3][k])
        if suff_stat[4] is not None and not supports(self.len_accumulator, Neutral):
            self.len_accumulator.combine(suff_stat[4])
        return self

    def scale(self, c: float) -> "HierarchicalDirichletProcessMixtureAccumulator":
        """Scale linear HDP mixture sufficient statistics while preserving metadata."""
        # Scale only the linear count statistics (per-group counts, atom accumulators, len accumulator).
        # ``prev_beta`` and ``prev_alpha`` are non-linear scalar/vector metadata carried for the
        # estimator's global-weight update; the inherited default would multiply and corrupt them.
        self.group_counts = [gc * c for gc in self.group_counts]
        for u in self.accumulators:
            u.scale(c)
        if not supports(self.len_accumulator, Neutral):
            self.len_accumulator.scale(c)
        return self

    def value(self) -> tuple:
        """Returns (group_counts, prev_beta, prev_alpha, atom values, len_value)."""
        len_val = None if supports(self.len_accumulator, Neutral) else self.len_accumulator.value()
        return (
            list(self.group_counts),
            self.prev_beta,
            self.prev_alpha,
            tuple(u.value() for u in self.accumulators),
            len_val,
        )

    def from_value(self, x: tuple) -> "HierarchicalDirichletProcessMixtureAccumulator":
        """Set the sufficient statistics from a value() tuple."""
        self.group_counts = list(x[0])
        self.prev_beta = x[1]
        self.prev_alpha = x[2]
        for k in range(self.num_components):
            self.accumulators[k].from_value(x[3][k])
        if x[4] is not None and not supports(self.len_accumulator, Neutral):
            self.len_accumulator.from_value(x[4])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator's keyed statistics into a shared dict."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self
        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics with the pooled keyed values."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())
        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "HierarchicalDirichletProcessMixtureDataEncoder":
        """Returns a HierarchicalDirichletProcessMixtureDataEncoder for this accumulator."""
        comp_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = None if supports(self.len_accumulator, Neutral) else self.len_accumulator.acc_to_encoder()
        return HierarchicalDirichletProcessMixtureDataEncoder(comp_encoder, len_encoder)


class HierarchicalDirichletProcessMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for HDP-mixture sufficient-statistic accumulators."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        len_factory: StatisticAccumulatorFactory | None,
        name: str | None,
        keys: str | None,
    ) -> None:
        """Create an HDP-mixture accumulator factory."""
        self.factories = list(factories)
        self.len_factory = len_factory
        self.name = name
        self.keys = keys

    def make(self) -> "HierarchicalDirichletProcessMixtureAccumulator":
        """Returns a new HierarchicalDirichletProcessMixtureAccumulator."""
        len_acc = NullAccumulator() if self.len_factory is None else self.len_factory.make()
        return HierarchicalDirichletProcessMixtureAccumulator(
            [f.make() for f in self.factories], len_accumulator=len_acc, name=self.name, keys=self.keys
        )


class HierarchicalDirichletProcessMixtureEstimator(ParameterEstimator):
    """Estimates a HierarchicalDirichletProcessMixtureDistribution from
    accumulated group counts via the direct-assignment truncation updates."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        gamma: float = 1.0,
        alpha: float = 1.0,
        name: str | None = None,
        keys: str | None = None,
        len_estimator: ParameterEstimator | None = None,
    ) -> None:
        """Create an estimator for the finite HDP-mixture approximation."""
        self.estimators = list(estimators)
        self.num_components = len(estimators)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.name = name
        self.keys = keys
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()

    def accumulator_factory(self) -> "HierarchicalDirichletProcessMixtureAccumulatorFactory":
        """Returns a HierarchicalDirichletProcessMixtureAccumulatorFactory for this estimator."""
        len_factory = (
            None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.accumulator_factory()
        )
        return HierarchicalDirichletProcessMixtureAccumulatorFactory(
            [u.accumulator_factory() for u in self.estimators], len_factory, self.name, self.keys
        )

    def model_log_density(self, model: HierarchicalDirichletProcessMixtureDistribution) -> float:
        """Log-density of the model parameters under the HDP priors.

        Sums the Dirichlet(gamma/K) log-density of the global weights beta, the
        Dirichlet(alpha*beta) log-density of each fitted group's weights (all
        floored at a tiny constant for boundary estimates), and each atom
        estimator's model_log_density of its atom. Together with seq_local_elbo
        this forms the penalized objective maximized by the fit driver.
        """
        k = self.num_components

        beta_prior = DirichletDistribution(np.ones(k) * self.gamma / k)
        rv = float(beta_prior.log_density(np.maximum(model.beta, _TINY)))

        if model.group_weights is not None:
            ab = np.maximum(self.alpha * model.beta, _TINY)
            group_prior = DirichletDistribution(ab)
            for j in range(len(model.group_weights)):
                rv += float(group_prior.log_density(np.maximum(model.group_weights[j, :], _TINY)))

        for est, comp in zip(self.estimators, model.components):
            fn = getattr(est, "model_log_density", None)
            if fn is not None:
                term = fn(comp)
                if term is not None:
                    rv += float(term)

        return rv

    def estimate(self, nobs: float | None, suff_stat: tuple) -> HierarchicalDirichletProcessMixtureDistribution:
        """Estimate a HierarchicalDirichletProcessMixtureDistribution.

        Re-estimates each atom (whose conjugate update carries its posterior
        forward as its prior), updates the global weights beta via the
        expected-table-count approximation followed by the Dirichlet(gamma/K +
        m_.k) posterior mean, and sets each group's weights to the
        Dirichlet(alpha*beta) posterior mean (deliberately the mean, not the MAP,
        which degenerates when alpha*beta_k < 1).

        Args:
            nobs (Optional[float]): Not used. Kept for the stats
                ``ParameterEstimator.estimate(nobs, suff_stat)`` signature.
            suff_stat: Tuple (group_counts, prev_beta, prev_alpha, atom stats,
                len_value) as returned by
                ``HierarchicalDirichletProcessMixtureAccumulator.value()``.

        Returns:
            Fitted hierarchical Dirichlet-process mixture approximation.

        """
        group_counts, prev_beta, prev_alpha, comp_stats, len_val = suff_stat
        k = self.num_components

        components = [self.estimators[i].estimate(None, comp_stats[i]) for i in range(k)]

        if isinstance(self.len_estimator, NullEstimator) or len_val is None:
            len_dist: SequenceEncodableProbabilityDistribution = NullDistribution()
        else:
            len_dist = self.len_estimator.estimate(None, len_val)

        counts = np.asarray(group_counts) if len(group_counts) > 0 else np.zeros((0, k))
        alpha = self.alpha if prev_alpha is None else float(prev_alpha)
        beta0 = np.ones(k) / k if prev_beta is None else np.asarray(prev_beta, dtype=float)

        # global weights via the expected-table-count approximation:
        # m_jk = alpha*beta_k * (psi(alpha*beta_k + n_jk) - psi(alpha*beta_k))
        ab = np.maximum(alpha * beta0, 1.0e-12)
        if counts.shape[0] > 0:
            m_mat = ab * (digamma(ab + counts) - digamma(ab))
            m_k = m_mat.sum(axis=0)
        else:
            m_k = np.zeros(k)

        beta = (m_k + self.gamma / k) / (m_k.sum() + self.gamma)

        # per-group posterior-mean weights under the Dirichlet(alpha*beta) prior.
        # The mean (not the MAP) is used deliberately: with alpha*beta_k < 1 the
        # Dirichlet density is unbounded on the simplex boundary, so MAP weights
        # degenerate to spikes; the mean is strictly interior and keeps the
        # penalized objective well-defined.
        ab_new = alpha * beta
        group_weights = np.zeros((counts.shape[0], k))
        for j in range(counts.shape[0]):
            group_weights[j, :] = (counts[j, :] + ab_new) / (counts[j, :].sum() + alpha)

        return HierarchicalDirichletProcessMixtureDistribution(
            components, beta, alpha, self.gamma, group_weights=group_weights, name=self.name, len_dist=len_dist
        )


class HierarchicalDirichletProcessMixtureDataEncoder(DataSequenceEncoder):
    """Encodes groups into a flat component encoding with per-group offsets."""

    def __init__(self, encoder: DataSequenceEncoder, len_encoder: DataSequenceEncoder | None = None) -> None:
        """Data encoder for grouped HDP-mixture observations.

        Args:
            encoder (DataSequenceEncoder): Encoder for the atom (component)
                distributions.
            len_encoder (Optional[DataSequenceEncoder]): Encoder for the group
                sizes; None treats sizes as exogenous.

        """
        self.encoder = encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        return "HierarchicalDirichletProcessMixtureDataEncoder(%s, %s)" % (str(self.encoder), str(self.len_encoder))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HierarchicalDirichletProcessMixtureDataEncoder):
            return False
        return self.encoder == other.encoder and self.len_encoder == other.len_encoder

    def seq_encode(self, x: Sequence[Sequence]) -> Any:
        """Encode groups into (lengths, offsets, flat_enc, len_enc)."""
        lengths = np.asarray([len(u) for u in x], dtype=int)
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        flat: list[Any] = []
        for u in x:
            flat.extend(u)
        flat_enc = self.encoder.seq_encode(flat)

        len_enc = None if self.len_encoder is None else self.len_encoder.seq_encode(lengths)

        return lengths, offsets, flat_enc, len_enc
