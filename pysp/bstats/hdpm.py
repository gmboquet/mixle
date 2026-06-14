"""Hierarchical Dirichlet process mixture (truncated) for grouped data.

Observations arrive in groups (a datum is a sequence of observations). All
groups share K global atoms; each group mixes them with its own weights.
This implements the finite "direct-assignment" truncation of the HDP
(Teh et al. 2006):

    beta             ~ Dirichlet(gamma/K, ..., gamma/K)    (global weights)
    pi_j | beta      ~ Dirichlet(alpha * beta)             (group j weights)
    z_ji | pi_j      ~ pi_j
    x_ji | z_ji = k  ~ components[k]

Estimation alternates:
  - E-step at point estimates: responsibilities phi_jik from the group's
    current weights and the atom densities,
  - MAP M-step for each group's weights (Dirichlet(alpha*beta) prior, clamped
    at the simplex boundary) and the atoms' conjugate updates,
  - global-weight update via the standard expected-table-count approximation
    m_jk = alpha*beta_k*(psi(alpha*beta_k + n_jk) - psi(alpha*beta_k)), with
    beta set to the Dirichlet(gamma/K + m_.k) posterior mean.

The group-weight and atom updates are exact coordinate ascent on the
penalized log-likelihood (data term + model_log_density); the beta step is
the customary approximation, and bestimation.optimize's acceptance gate
rejects any step that decreases the objective. seq_local_elbo scores
training groups with their fitted weights (this is what optimize maximizes);
seq_log_density scores a (possibly new) group with the global weights beta,
i.e. the expected weights of an unseen group.

Group sizes are exogenous unless len_dist is supplied (used for sampling and
added to the per-group score).
"""

from collections.abc import Sequence

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma

from pysp.arithmetic import maxint
from pysp.bstats.dirichlet import DirichletDistribution
from pysp.bstats.nulldist import NullAccumulator, NullDistribution, NullEstimator, null_dist
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator

_TINY = 1.0e-300


class HierarchicalDirichletProcessMixtureDistribution(ProbabilityDistribution):
    """Truncated hierarchical DP mixture over K shared atoms with global
    weights beta and (optionally) fitted per-group weights."""

    def __init__(
        self,
        components,
        beta,
        alpha: float,
        gamma: float,
        group_weights: np.ndarray | None = None,
        name: str | None = None,
        len_dist: ProbabilityDistribution = null_dist,
    ):
        """HierarchicalDirichletProcessMixtureDistribution object.

        Args:
            components: List of K shared atom distributions (each carrying
                its own prior).
            beta: Length-K global weight vector.
            alpha (float): Group-level concentration of Dirichlet(alpha*beta).
            gamma (float): Global concentration of Dirichlet(gamma/K).
            group_weights (Optional[np.ndarray]): (J, K) fitted weights of
                the training groups (used by seq_local_elbo); None scores
                all groups with beta.
            name (Optional[str]): Name of object.
            len_dist (ProbabilityDistribution): Distribution of group sizes;
                null_dist treats sizes as exogenous.

        """
        self.name = name
        self.components = components
        self.num_components = len(components)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.len_dist = len_dist

        self.beta = np.asarray(beta, dtype=float)
        with np.errstate(divide="ignore"):
            self.log_beta = np.log(self.beta)

        self.group_weights = None if group_weights is None else np.asarray(group_weights, dtype=float)

    def __str__(self):
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

    def get_parameters(self):
        """Returns the parameter tuple (beta, alpha, gamma, component parameters)."""
        return self.beta, self.alpha, self.gamma, [u.get_parameters() for u in self.components]

    def set_parameters(self, params):
        """Set the parameters and refresh the cached log-weights.

        Args:
            params: Tuple (beta, alpha, gamma, component parameters); the
                component parameters are pushed down into the components.

        """
        beta, alpha, gamma, comp_params = params
        self.beta = np.asarray(beta, dtype=float)
        with np.errstate(divide="ignore"):
            self.log_beta = np.log(self.beta)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        for c, p in zip(self.components, comp_params):
            c.set_parameters(p)

    def _len_term(self, x) -> float:
        if isinstance(self.len_dist, NullDistribution) or self.len_dist is None:
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

    def density(self, x) -> float:
        """Density of a group x; see log_density().

        Args:
            x: Sequence of observations accepted by the components.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        """Score a group with the global weights beta (expected weights of a
        new group).

        Args:
            x: Sequence of observations accepted by the components.

        Returns:
            Log-density at observation x (plus the len_dist term when present).

        """
        if len(x) == 0:
            return self._len_term(x)
        enc = self.components[0].seq_encode(list(x))
        log_b = np.asarray([c.seq_log_density(enc) for c in self.components]).T
        return self._group_log_density(log_b, self.log_beta) + self._len_term(x)

    def seq_encode(self, x: Sequence[Sequence]):
        """Encode groups into a flat component encoding with offsets.

        Args:
            x (Sequence[Sequence]): Iterable of groups (sequences of
                observations).

        Returns:
            Tuple (lengths, offsets, flat_enc, len_enc) for use with
            seq_ methods.

        """
        lengths = np.asarray([len(u) for u in x], dtype=int)
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        flat = []
        for u in x:
            flat.extend(u)
        flat_enc = self.components[0].seq_encode(flat)

        if isinstance(self.len_dist, NullDistribution) or self.len_dist is None:
            len_enc = None
        else:
            len_enc = self.len_dist.seq_encode(lengths)

        return lengths, offsets, flat_enc, len_enc

    def _emission_log_densities(self, flat_enc) -> np.ndarray:
        return np.asarray([c.seq_log_density(flat_enc) for c in self.components]).T

    def seq_log_density(self, x) -> np.ndarray:
        """Vectorized log_density() at sequence-encoded input x (each group
        scored with the global weights beta).

        Args:
            x: Encoded groups from seq_encode().

        Returns:
            Numpy array of log-densities, one per group.

        """
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

    def seq_local_elbo(self, x) -> np.ndarray:
        """Per-group data term of the penalized objective: training groups are
        scored with their own fitted weights. Falls back to beta when the
        group count does not match the fit.

        Args:
            x: Encoded groups from seq_encode().

        Returns:
            Numpy array of per-group data terms, one per group.

        """
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

    def group_posteriors(self, x) -> np.ndarray:
        """Posterior atom-usage (mean responsibility) per group.

        Groups are scored with their fitted weights when available (else
        with the global weights beta).

        Args:
            x: Iterable of groups (unencoded).

        Returns:
            (J, K) numpy array of mean responsibilities per group.

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

    def sampler(self, seed: int | None = None):
        """Create a HierarchicalDirichletProcessMixtureSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            HierarchicalDirichletProcessMixtureSampler object.

        """
        return HierarchicalDirichletProcessMixtureSampler(self, seed)

    def estimator(self):
        """Create a HierarchicalDirichletProcessMixtureEstimator from this
        distribution's components, concentrations, and length estimator.

        Returns:
            HierarchicalDirichletProcessMixtureEstimator object.

        """
        len_est = NullEstimator() if isinstance(self.len_dist, NullDistribution) else self.len_dist.estimator()
        return HierarchicalDirichletProcessMixtureEstimator(
            [u.estimator() for u in self.components],
            gamma=self.gamma,
            alpha=self.alpha,
            name=self.name,
            len_estimator=len_est,
        )


class HierarchicalDirichletProcessMixtureSampler:
    """Draws groups from a HierarchicalDirichletProcessMixtureDistribution
    (per-group weights drawn from Dirichlet(alpha*beta))."""

    def __init__(self, dist: HierarchicalDirichletProcessMixtureDistribution, seed: int | None = None):
        """HierarchicalDirichletProcessMixtureSampler object.

        Args:
            dist (HierarchicalDirichletProcessMixtureDistribution):
                Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        rng = RandomState(seed)
        self.rng = RandomState(rng.randint(0, maxint))
        self.dist = dist
        self.comp_samplers = [u.sampler(seed=rng.randint(0, maxint)) for u in dist.components]
        if isinstance(dist.len_dist, NullDistribution) or dist.len_dist is None:
            self.len_sampler = None
        else:
            self.len_sampler = dist.len_dist.sampler(seed=rng.randint(0, maxint))

    def sample_group(self, n: int | None = None):
        """Draw a single group of n observations.

        Group weights pi ~ Dirichlet(alpha*beta) are drawn once for the
        group, then each observation draws an atom from pi.

        Args:
            n (Optional[int]): Group size; drawn from the len_dist sampler
                when None (which then must exist).

        Returns:
            List of n observations.

        """
        if n is None:
            if self.len_sampler is None:
                raise Exception("HDP sampler requires a len_dist (or explicit n) to sample groups.")
            n = int(self.len_sampler.sample())

        pi = self.rng.dirichlet(np.maximum(self.dist.alpha * self.dist.beta, 1.0e-8))
        states = self.rng.choice(self.dist.num_components, size=n, p=pi)
        return [self.comp_samplers[k].sample() for k in states]

    def sample(self, size=None):
        """Draw size groups (a single group when size is None).

        Args:
            size (Optional[int]): Number of groups to draw.

        Returns:
            A group if size is None, else a list of size groups.

        """
        if size is None:
            return self.sample_group()
        return [self.sample_group() for _ in range(size)]


class HierarchicalDirichletProcessMixtureAccumulator(StatisticAccumulator):
    """Accumulates HDP mixture sufficient statistics: per-group expected
    atom counts (in data order) plus each atom's weighted statistics."""

    def __init__(self, accumulators, len_accumulator=NullAccumulator(), name=None, keys=None):
        """HierarchicalDirichletProcessMixtureAccumulator object.

        Args:
            accumulators: List of K atom accumulators.
            len_accumulator: Accumulator for the group sizes.
            name (Optional[str]): Name of the accumulator.
            keys (Optional[str]): Key for sharing sufficient statistics.

        """
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.name = name
        self.key = keys
        self.group_counts = []  # one (K,) count vector per group, in data order
        self.prev_beta = None
        self.prev_alpha = None
        self.len_accumulator = len_accumulator

    def initialize(self, x, weight, rng):
        """Initialize with random Dirichlet assignments for group x.

        Args:
            x: Group (sequence of observations).
            weight (float): Weight of the group.
            rng (RandomState): Random number generator for the assignments.

        """
        counts = np.zeros(self.num_components)
        for u in x:
            p = rng.dirichlet(np.ones(self.num_components))
            counts += p * weight
            for k in range(self.num_components):
                self.accumulators[k].initialize(u, p[k] * weight, rng)
        self.group_counts.append(counts)

        if not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.update(len(x), weight, None)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialize() with random Dirichlet assignments.

        Args:
            x: Encoded groups from seq_encode().
            weights (np.ndarray): Weight per group.
            rng (RandomState): Random number generator for the assignments.

        """
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

        if len_enc is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.seq_initialize(len_enc, weights, rng)

    def update(self, x, weight, estimate):
        """Accumulate the E-step statistics for one group (delegates to
        seq_update on a singleton encoding).

        Args:
            x: Group (sequence of observations).
            weight (float): Weight of the group.
            estimate (HierarchicalDirichletProcessMixtureDistribution):
                Current estimate supplying weights and atom densities.

        """
        enc = estimate.seq_encode([x])
        self.seq_update(enc, np.asarray([weight]), estimate)

    def seq_update(self, x, weights, estimate):
        """E-step on sequence-encoded data at the current point estimates.

        Computes responsibilities phi from each group's current weights
        (the fitted group weights, or beta for new/unmatched groups) and
        the atom densities, recording per-group expected counts and pushing
        phi-weighted updates into the atom accumulators. Also records the
        estimate's beta and alpha for the estimator's global-weight update.

        Args:
            x: Encoded groups from seq_encode().
            weights (np.ndarray): Weight per group.
            estimate (HierarchicalDirichletProcessMixtureDistribution):
                Current estimate supplying weights and atom densities.

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
        base = len(self.group_counts)
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

        if len_enc is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.seq_update(len_enc, weights, None)

    def combine(self, suff_stat):
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.group_counts.extend(suff_stat[0])
        if suff_stat[1] is not None:
            self.prev_beta = suff_stat[1]
            self.prev_alpha = suff_stat[2]
        for k in range(self.num_components):
            self.accumulators[k].combine(suff_stat[3][k])
        if suff_stat[4] is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.combine(suff_stat[4])
        return self

    def value(self):
        """Returns (group_counts, prev_beta, prev_alpha, atom values, len_value)."""
        len_val = None if isinstance(self.len_accumulator, NullAccumulator) else self.len_accumulator.value()
        return (
            list(self.group_counts),
            self.prev_beta,
            self.prev_alpha,
            tuple(u.value() for u in self.accumulators),
            len_val,
        )

    def from_value(self, x):
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.group_counts = list(x[0])
        self.prev_beta = x[1]
        self.prev_alpha = x[2]
        for k in range(self.num_components):
            self.accumulators[k].from_value(x[3][k])
        if x[4] is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.from_value(x[4])
        return self

    def key_merge(self, stats_dict):
        """Merge this accumulator's keyed statistics into a shared dict.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self
        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics with the pooled keyed values.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())
        for u in self.accumulators:
            u.key_replace(stats_dict)


class HierarchicalDirichletProcessMixtureAccumulatorFactory:
    """Factory that creates HierarchicalDirichletProcessMixtureAccumulator objects."""

    def __init__(self, factories, len_factory, name, keys):
        """HierarchicalDirichletProcessMixtureAccumulatorFactory object.

        Args:
            factories: List of K atom accumulator factories.
            len_factory: Factory for the group-size accumulators (None for none).
            name (Optional[str]): Name passed to created accumulators.
            keys (Optional[str]): Key passed to created accumulators.

        """
        self.factories = factories
        self.len_factory = len_factory
        self.name = name
        self.keys = keys

    def make(self):
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
        estimators,
        gamma: float = 1.0,
        alpha: float = 1.0,
        name: str | None = None,
        keys: str | None = None,
        len_estimator: ParameterEstimator = NullEstimator(),
    ):
        """HierarchicalDirichletProcessMixtureEstimator object.

        Args:
            estimators: List of K atom estimators.
            gamma (float): Global concentration of Dirichlet(gamma/K).
            alpha (float): Group-level concentration of Dirichlet(alpha*beta).
            name (Optional[str]): Name of the estimated distribution.
            keys (Optional[str]): Key for sharing sufficient statistics.
            len_estimator (ParameterEstimator): Estimator for the group
                sizes (NullEstimator treats sizes as exogenous).

        """
        self.estimators = estimators
        self.num_components = len(estimators)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.name = name
        self.keys = keys
        self.len_estimator = len_estimator

    def accumulator_factory(self):
        """Returns a HierarchicalDirichletProcessMixtureAccumulatorFactory
        for this estimator."""
        len_factory = (
            None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.accumulator_factory()
        )
        return HierarchicalDirichletProcessMixtureAccumulatorFactory(
            [u.accumulator_factory() for u in self.estimators], len_factory, self.name, self.keys
        )

    def model_log_density(self, model) -> float:
        """Log-density of the model parameters under the HDP priors.

        Sums the Dirichlet(gamma/K) log-density of the global weights beta,
        the Dirichlet(alpha*beta) log-density of each fitted group's
        weights (all floored at a tiny constant for boundary estimates),
        and each atom estimator's model_log_density of its atom. Together
        with seq_local_elbo this forms the penalized objective maximized by
        bestimation.optimize.

        Args:
            model (HierarchicalDirichletProcessMixtureDistribution): Model
                to score.

        Returns:
            Prior log-density of the model parameters.

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
            rv += est.model_log_density(comp)

        return rv

    def estimate(self, suff_stat) -> HierarchicalDirichletProcessMixtureDistribution:
        """Estimate a HierarchicalDirichletProcessMixtureDistribution.

        Re-estimates each atom (whose conjugate update carries its
        posterior forward as its prior), updates the global weights beta via
        the expected-table-count approximation followed by the
        Dirichlet(gamma/K + m_.k) posterior mean, and sets each group's
        weights to the Dirichlet(alpha*beta) posterior mean (deliberately
        the mean, not the MAP, which degenerates when alpha*beta_k < 1).

        Args:
            suff_stat: Tuple (group_counts, prev_beta, prev_alpha,
                atom stats, len_value) as returned by
                HierarchicalDirichletProcessMixtureAccumulator.value().

        Returns:
            HierarchicalDirichletProcessMixtureDistribution object.

        """
        group_counts, prev_beta, prev_alpha, comp_stats, len_val = suff_stat
        k = self.num_components

        components = [self.estimators[i].estimate(comp_stats[i]) for i in range(k)]

        if isinstance(self.len_estimator, NullEstimator) or len_val is None:
            len_dist = null_dist
        else:
            len_dist = self.len_estimator.estimate(len_val)

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

        # per-group posterior-mean weights under the Dirichlet(alpha*beta)
        # prior. The mean (not the MAP) is used deliberately: with
        # alpha*beta_k < 1 the Dirichlet density is unbounded on the simplex
        # boundary, so MAP weights degenerate to spikes; the mean is strictly
        # interior and keeps the penalized objective well-defined.
        ab_new = alpha * beta
        group_weights = np.zeros((counts.shape[0], k))
        for j in range(counts.shape[0]):
            group_weights[j, :] = (counts[j, :] + ab_new) / (counts[j, :].sum() + alpha)

        return HierarchicalDirichletProcessMixtureDistribution(
            components, beta, alpha, self.gamma, group_weights=group_weights, name=self.name, len_dist=len_dist
        )
