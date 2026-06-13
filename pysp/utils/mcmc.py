"""Small MCMC utilities over pysparkplug log-density objects.

The low-level functions here deliberately operate on user-supplied log-target
callables and proposal objects.  That keeps the transition machinery orthogonal
to the distribution, estimator, and compute-engine protocols while still making
ordinary ``dist.log_density(x)`` models easy to sample from.

On top of that machinery this module also provides a high-level parameter
posterior API (:func:`sample_parameter_posterior`,
:func:`sample_conjugate_posterior`).  Given a prototype distribution that fixes
the model family, a dataset, and a prior over parameters, it samples
``p(theta | data) proportional to exp(sum_i log p(x_i | theta)) * prior(theta)``
by running Metropolis-Hastings or Hamiltonian Monte Carlo in an unconstrained
reparameterization (log for positive scales, stick-breaking for probability
simplices) and mapping the retained samples back to parameter space (or to
rebuilt distribution objects).
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


LogTarget = Callable[[Any], float]


@dataclass(frozen=True)
class MCMCResult:
    """Samples and diagnostics returned by an MCMC run."""

    samples: List[Any]
    log_probs: np.ndarray
    accepted: np.ndarray
    transition_labels: Optional[Tuple[str, ...]] = None

    @property
    def acceptance_rate(self) -> float:
        """Return the overall fraction of accepted transitions."""
        return float(np.mean(self.accepted)) if len(self.accepted) else 0.0

    @property
    def acceptance_rate_by_label(self) -> Dict[str, float]:
        """Return acceptance rates for labelled transition kernels."""
        if self.transition_labels is None:
            return {}
        if len(self.transition_labels) != len(self.accepted):
            raise ValueError('transition label count does not match accepted count.')
        rv: Dict[str, List[bool]] = {}
        for label, accepted in zip(self.transition_labels, self.accepted):
            rv.setdefault(label, []).append(bool(accepted))
        return {label: float(np.mean(values)) for label, values in rv.items()}

    def sample_array(self) -> np.ndarray:
        """Return numeric samples as an ndarray for diagnostics."""
        try:
            arr = np.asarray(self.samples, dtype=float)
        except Exception as e:
            raise ValueError('samples cannot be represented as a numeric array.') from e
        if arr.ndim == 0:
            arr = arr.reshape((1,))
        if len(arr) != len(self.samples):
            raise ValueError('samples have inconsistent numeric shape.')
        return arr

    def effective_sample_size(self, max_lag: Optional[int] = None) -> Any:
        """Estimate effective sample size using positive autocorrelation lags.

        Scalar samples return a float. Vector samples return one ESS value per
        trailing dimension.
        """
        arr = self.sample_array()
        n = int(arr.shape[0])
        if n <= 1:
            return float(n)
        flat = arr.reshape((n, -1))
        centered = flat - flat.mean(axis=0, keepdims=True)
        var = np.mean(centered * centered, axis=0)
        if np.any(var <= 0.0):
            ess = np.where(var <= 0.0, float(n), 0.0)
            return float(ess[0]) if arr.ndim == 1 else ess.reshape(arr.shape[1:])

        lag_limit = n - 1 if max_lag is None else min(int(max_lag), n - 1)
        tau = np.ones(flat.shape[1], dtype=float)
        for lag in range(1, lag_limit + 1):
            rho = np.mean(centered[:-lag] * centered[lag:], axis=0) / var
            positive = rho > 0.0
            if not np.any(positive):
                break
            tau += 2.0 * np.where(positive, rho, 0.0)
        ess = np.maximum(1.0, n / tau)
        return float(ess[0]) if arr.ndim == 1 else ess.reshape(arr.shape[1:])

    def summary(self, max_lag: Optional[int] = None) -> Dict[str, Any]:
        """Return basic numeric chain diagnostics.

        The summary intentionally stays dependency-free and returns plain
        numbers/arrays: sample count, mean, variance, Monte Carlo standard
        error estimate, ESS, and acceptance diagnostics.
        """
        arr = self.sample_array()
        n = int(arr.shape[0])
        if n == 0:
            raise ValueError('cannot summarize an empty chain.')
        mean = np.mean(arr, axis=0)
        variance = np.var(arr, axis=0)
        ess = self.effective_sample_size(max_lag=max_lag)
        ess_arr = np.asarray(ess, dtype=float)
        mcse = np.sqrt(np.asarray(variance, dtype=float) / np.maximum(ess_arr, 1.0))
        return {
            'num_samples': n,
            'mean': _scalar_if_zero_dim(mean),
            'variance': _scalar_if_zero_dim(variance),
            'ess': _scalar_if_zero_dim(ess),
            'mcse': _scalar_if_zero_dim(mcse),
            'acceptance_rate': self.acceptance_rate,
            'acceptance_rate_by_label': self.acceptance_rate_by_label,
        }


class Proposal(object):
    """Base proposal protocol for Metropolis-Hastings kernels."""

    def sample(self, current: Any, rng: np.random.RandomState) -> Any:
        """Draw a proposed state given the current state."""
        raise NotImplementedError

    def log_density(self, proposed: Any, current: Any) -> float:
        """Return ``log q(proposed | current)`` for Hastings correction."""
        return 0.0

    def adapt(self, current: Any, proposed: Any, accepted: bool, step: int, in_burn_in: bool) -> None:
        """Optional adaptation hook called after each transition."""
        return None


class RandomWalkProposal(Proposal):
    """Symmetric Gaussian random-walk proposal for scalar/vector states."""

    def __init__(self, scale: Any) -> None:
        self.scale = np.asarray(scale, dtype=float)
        if np.any(self.scale <= 0.0) or not np.all(np.isfinite(self.scale)):
            raise ValueError('scale must be finite and positive.')

    def sample(self, current: Any, rng: np.random.RandomState) -> Any:
        """Draw a Gaussian random-walk proposal centered at ``current``."""
        cur = np.asarray(current, dtype=float)
        value = cur + rng.normal(size=cur.shape) * self.scale
        return float(value) if value.ndim == 0 else value

    def log_density(self, proposed: Any, current: Any) -> float:
        """Return the Gaussian random-walk transition log density."""
        prop = np.asarray(proposed, dtype=float)
        cur = np.asarray(current, dtype=float)
        resid = prop - cur
        scale = np.broadcast_to(self.scale, resid.shape)
        dim = int(resid.size) if resid.ndim > 0 else 1
        return float(-0.5 * dim * np.log(2.0 * np.pi) - np.sum(np.log(scale))
                     - 0.5 * np.sum((resid / scale) * (resid / scale)))


class AdaptiveRandomWalkProposal(RandomWalkProposal):
    """Gaussian random walk with Robbins-Monro scale adaptation.

    By default adaptation only runs during burn-in, which preserves the
    stationary post-burn chain used for retained samples.
    """

    def __init__(self, scale: Any, target_acceptance: float = 0.44,
                 adaptation_rate: float = 0.05, adapt_during_burn_in_only: bool = True,
                 min_scale: float = 1.0e-12, max_scale: float = 1.0e12) -> None:
        super().__init__(scale)
        if not (0.0 < target_acceptance < 1.0):
            raise ValueError('target_acceptance must be in (0, 1).')
        if adaptation_rate <= 0.0:
            raise ValueError('adaptation_rate must be positive.')
        self.target_acceptance = float(target_acceptance)
        self.adaptation_rate = float(adaptation_rate)
        self.adapt_during_burn_in_only = bool(adapt_during_burn_in_only)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def adapt(self, current: Any, proposed: Any, accepted: bool, step: int, in_burn_in: bool) -> None:
        """Adjust the random-walk scale toward the target acceptance rate."""
        if self.adapt_during_burn_in_only and not in_burn_in:
            return None
        gamma = self.adaptation_rate / np.sqrt(float(step + 1))
        direction = (1.0 if accepted else 0.0) - self.target_acceptance
        log_scale = np.log(np.asarray(self.scale, dtype=float)) + gamma * direction
        self.scale = np.clip(np.exp(log_scale), self.min_scale, self.max_scale)
        return None


class AdaptiveCovarianceProposal(Proposal):
    """Full-covariance Gaussian random walk with burn-in covariance learning."""

    def __init__(self, initial_covariance: Any = 1.0,
                 scale: Optional[float] = None,
                 regularization: float = 1.0e-6,
                 adapt_after: int = 2,
                 adapt_during_burn_in_only: bool = True) -> None:
        if scale is not None and (scale <= 0.0 or not np.isfinite(scale)):
            raise ValueError('scale must be finite and positive.')
        if regularization < 0.0 or not np.isfinite(regularization):
            raise ValueError('regularization must be finite and non-negative.')
        if adapt_after < 2:
            raise ValueError('adapt_after must be at least 2.')
        self.initial_covariance = np.asarray(initial_covariance, dtype=float)
        if not np.all(np.isfinite(self.initial_covariance)):
            raise ValueError('initial_covariance must be finite.')
        self.user_scale = None if scale is None else float(scale)
        self.scale: Optional[float] = None if scale is None else float(scale)
        self.regularization = float(regularization)
        self.adapt_after = int(adapt_after)
        self.adapt_during_burn_in_only = bool(adapt_during_burn_in_only)
        self._dim: Optional[int] = None
        self._shape: Optional[Tuple[int, ...]] = None
        self.covariance: Optional[np.ndarray] = None
        self._proposal_covariance: Optional[np.ndarray] = None
        self._proposal_inv: Optional[np.ndarray] = None
        self._proposal_log_det: Optional[float] = None
        self._count = 0
        self._mean: Optional[np.ndarray] = None
        self._m2: Optional[np.ndarray] = None

    def sample(self, current: Any, rng: np.random.RandomState) -> Any:
        """Draw a full-covariance Gaussian random-walk proposal."""
        cur = self._state_vector(current)
        self._ensure_cache()
        noise = rng.multivariate_normal(np.zeros(self._dim, dtype=float), self._proposal_covariance)
        return self._restore(cur + noise)

    def log_density(self, proposed: Any, current: Any) -> float:
        """Return the current adaptive Gaussian proposal log density."""
        prop = self._state_vector(proposed)
        cur = self._state_vector(current)
        self._ensure_cache()
        resid = prop - cur
        quad = float(np.dot(resid, np.dot(self._proposal_inv, resid)))
        return float(-0.5 * (self._dim * np.log(2.0 * np.pi) + self._proposal_log_det + quad))

    def adapt(self, current: Any, proposed: Any, accepted: bool, step: int, in_burn_in: bool) -> None:
        """Update the empirical covariance estimate during adaptation."""
        if self.adapt_during_burn_in_only and not in_burn_in:
            return None
        state = proposed if accepted else current
        x = self._state_vector(state)
        self._count += 1
        if self._mean is None:
            self._mean = x.copy()
            self._m2 = np.zeros((self._dim, self._dim), dtype=float)
            return None
        delta = x - self._mean
        self._mean = self._mean + delta / float(self._count)
        delta2 = x - self._mean
        self._m2 = self._m2 + np.outer(delta, delta2)
        if self._count >= self.adapt_after:
            self.covariance = self._m2 / float(self._count - 1)
            self._refresh_cache()
        return None

    def _state_vector(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError('adaptive covariance proposal requires finite numeric states.')
        flat = arr.reshape((-1,))
        if self._dim is None:
            self._dim = int(flat.size)
            self._shape = arr.shape
            if self._dim == 0:
                raise ValueError('state must contain at least one numeric value.')
            if self.scale is None:
                self.scale = 2.38 / np.sqrt(float(self._dim))
            self.covariance = self._initial_covariance(self._dim)
            self._refresh_cache()
        elif int(flat.size) != self._dim or arr.shape != self._shape:
            raise ValueError('state shape %s does not match initial shape %s.' % (arr.shape, self._shape))
        return flat.copy()

    def _restore(self, x: np.ndarray) -> Any:
        shaped = np.asarray(x, dtype=float).reshape(self._shape)
        return float(shaped) if shaped.ndim == 0 else shaped.copy()

    def _initial_covariance(self, dim: int) -> np.ndarray:
        cov = self.initial_covariance
        if cov.shape == ():
            if float(cov) <= 0.0:
                raise ValueError('initial_covariance scalar must be positive.')
            return np.eye(dim, dtype=float) * float(cov)
        if cov.ndim == 1:
            if cov.shape != (dim,):
                raise ValueError('initial_covariance vector length must match state dimension.')
            if np.any(cov <= 0.0):
                raise ValueError('initial_covariance diagonal entries must be positive.')
            return np.diag(cov)
        if cov.shape != (dim, dim):
            raise ValueError('initial_covariance matrix shape must match state dimension.')
        return cov.copy()

    def _ensure_cache(self) -> None:
        if self._proposal_covariance is None:
            self._refresh_cache()

    def _refresh_cache(self) -> None:
        cov = np.asarray(self.covariance, dtype=float)
        cov = 0.5 * (cov + cov.T)
        proposal_cov = (self.scale * self.scale) * cov
        proposal_cov = proposal_cov + self.regularization * np.eye(self._dim, dtype=float)
        sign, log_det = np.linalg.slogdet(proposal_cov)
        if sign <= 0.0 or not np.isfinite(log_det):
            proposal_cov = proposal_cov + max(self.regularization, 1.0e-8) * np.eye(self._dim, dtype=float)
            sign, log_det = np.linalg.slogdet(proposal_cov)
        if sign <= 0.0 or not np.isfinite(log_det):
            raise ValueError('proposal covariance is not positive definite.')
        self._proposal_covariance = proposal_cov
        self._proposal_inv = np.linalg.inv(proposal_cov)
        self._proposal_log_det = float(log_det)


class IndependentProposal(Proposal):
    """Independence proposal from a sampler plus optional log density."""

    def __init__(self, sampler: Callable[[np.random.RandomState], Any],
                 log_density: Optional[Callable[[Any], float]] = None) -> None:
        self.sampler = sampler
        self._log_density = log_density

    def sample(self, current: Any, rng: np.random.RandomState) -> Any:
        """Draw a state independent of ``current`` from the supplied sampler."""
        return self.sampler(rng)

    def log_density(self, proposed: Any, current: Any) -> float:
        """Return the proposal log density when one was supplied."""
        if self._log_density is None:
            return 0.0
        return float(self._log_density(proposed))


class MixtureProposal(Proposal):
    """Mixture of proposal kernels with exact mixture proposal density."""

    def __init__(self, proposals: Sequence[Proposal],
                 weights: Optional[Sequence[float]] = None) -> None:
        if len(proposals) == 0:
            raise ValueError('MixtureProposal requires at least one proposal.')
        self.proposals = tuple(proposals)
        if weights is None:
            w = np.ones(len(self.proposals), dtype=float) / float(len(self.proposals))
        else:
            w = np.asarray(weights, dtype=float)
        if w.shape != (len(self.proposals),):
            raise ValueError('weights must match proposal count.')
        if np.any(w < 0.0) or not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0.0:
            raise ValueError('weights must be finite, non-negative, and have positive sum.')
        self.weights = w / float(np.sum(w))
        self._last_index: Optional[int] = None

    def sample(self, current: Any, rng: np.random.RandomState) -> Any:
        """Draw from one mixture component proposal and remember its index."""
        idx = int(rng.choice(len(self.proposals), p=self.weights))
        self._last_index = idx
        return self.proposals[idx].sample(current, rng)

    def log_density(self, proposed: Any, current: Any) -> float:
        """Return the log mixture density of all component proposals."""
        terms = []
        for weight, proposal in zip(self.weights, self.proposals):
            if weight > 0.0:
                terms.append(np.log(weight) + float(proposal.log_density(proposed, current)))
        return _logsumexp(terms)

    def adapt(self, current: Any, proposed: Any, accepted: bool, step: int, in_burn_in: bool) -> None:
        """Forward adaptation to the component used for the last proposal."""
        if self._last_index is not None:
            self.proposals[self._last_index].adapt(current, proposed, accepted, step, in_burn_in)


class BlockProposal(Proposal):
    """Lift a proposal on one or more mapping fields to a full-record proposal."""

    def __init__(self, keys: Any, proposal: Proposal) -> None:
        if isinstance(keys, str) or not isinstance(keys, Sequence):
            self.keys = (keys,)
        else:
            self.keys = tuple(keys)
        if len(self.keys) == 0:
            raise ValueError('BlockProposal requires at least one key.')
        self.proposal = proposal

    def _block(self, state: Mapping[Any, Any]) -> Any:
        if len(self.keys) == 1:
            return state[self.keys[0]]
        return tuple(state[key] for key in self.keys)

    def _replace(self, state: Mapping[Any, Any], value: Any) -> Dict[Any, Any]:
        rv = dict(state)
        if len(self.keys) == 1:
            rv[self.keys[0]] = value
            return rv
        if isinstance(value, Mapping):
            for key in self.keys:
                rv[key] = value[key]
            return rv
        values = tuple(value)
        if len(values) != len(self.keys):
            raise ValueError('proposed block has %d values for %d keys.' % (len(values), len(self.keys)))
        for key, item in zip(self.keys, values):
            rv[key] = item
        return rv

    def sample(self, current: Mapping[Any, Any], rng: np.random.RandomState) -> Dict[Any, Any]:
        """Propose a replacement for the configured mapping field block."""
        if not isinstance(current, Mapping):
            raise TypeError('BlockProposal requires mapping states.')
        return self._replace(current, self.proposal.sample(self._block(current), rng))

    def log_density(self, proposed: Mapping[Any, Any], current: Mapping[Any, Any]) -> float:
        """Return the block proposal density inside the full mapping state."""
        if not isinstance(proposed, Mapping) or not isinstance(current, Mapping):
            raise TypeError('BlockProposal requires mapping states.')
        return self.proposal.log_density(self._block(proposed), self._block(current))

    def adapt(self, current: Any, proposed: Any, accepted: bool, step: int, in_burn_in: bool) -> None:
        """Forward adaptation to the wrapped block proposal."""
        self.proposal.adapt(self._block(current), self._block(proposed), accepted, step, in_burn_in)


class LangevinProposal(Proposal):
    """Metropolis-adjusted Langevin proposal for scalar/vector states."""

    def __init__(self, step_size: float, grad_log_target: Callable[[Any], Any]) -> None:
        if step_size <= 0.0 or not np.isfinite(step_size):
            raise ValueError('step_size must be finite and positive.')
        self.step_size = float(step_size)
        self.grad_log_target = grad_log_target

    def _mean(self, x: Any) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        grad = np.asarray(self.grad_log_target(x), dtype=float)
        if grad.shape != xx.shape:
            raise ValueError('grad_log_target shape %s does not match state shape %s.' % (grad.shape, xx.shape))
        return xx + 0.5 * self.step_size * self.step_size * grad

    def sample(self, current: Any, rng: np.random.RandomState) -> Any:
        """Draw a MALA proposal centered at the Langevin drift mean."""
        cur = np.asarray(current, dtype=float)
        value = self._mean(current) + self.step_size * rng.normal(size=cur.shape)
        return float(value) if value.ndim == 0 else value

    def log_density(self, proposed: Any, current: Any) -> float:
        """Return the asymmetric Langevin proposal log density."""
        prop = np.asarray(proposed, dtype=float)
        mean = self._mean(current)
        resid = prop - mean
        dim = int(resid.size) if resid.ndim > 0 else 1
        var = self.step_size * self.step_size
        return float(-0.5 * dim * np.log(2.0 * np.pi * var) - 0.5 * np.sum(resid * resid) / var)


def distribution_log_target(dist: Any, evidence: Optional[Callable[[Any], float]] = None) -> LogTarget:
    """Return ``log_target(x) = dist.log_density(x) + evidence(x)``."""
    if evidence is None:
        return lambda x: float(dist.log_density(x))
    return lambda x: float(dist.log_density(x)) + float(evidence(x))


def metropolis_hastings(log_target: LogTarget,
                        initial: Any,
                        proposal: Proposal,
                        num_samples: int,
                        burn_in: int = 0,
                        thin: int = 1,
                        rng: Optional[np.random.RandomState] = None) -> MCMCResult:
    """Run a generic Metropolis-Hastings chain.

    Args:
        log_target: Callable returning an unnormalized log target.
        initial: Initial Markov-chain state.
        proposal: Proposal object with ``sample`` and optional
            ``log_density`` methods.
        num_samples: Number of post-burn/thinned states to return.
        burn_in: Number of initial transitions to discard.
        thin: Keep one sample every ``thin`` transitions.
        rng: Optional RandomState.

    Returns:
        MCMCResult with retained samples, retained log probabilities, and the
        accept/reject indicator for every transition.
    """
    if num_samples < 0:
        raise ValueError('num_samples must be non-negative.')
    if burn_in < 0:
        raise ValueError('burn_in must be non-negative.')
    if thin <= 0:
        raise ValueError('thin must be positive.')

    rng = np.random.RandomState() if rng is None else rng
    current = initial
    current_lp = float(log_target(current))
    if not np.isfinite(current_lp):
        raise ValueError('initial state has non-finite log target: %r.' % current_lp)

    total_steps = burn_in + num_samples * thin
    samples: List[Any] = []
    log_probs: List[float] = []
    accepted: List[bool] = []

    for step in range(total_steps):
        old_current = current
        proposed = proposal.sample(current, rng)
        proposed_lp = float(log_target(proposed))
        if np.isfinite(proposed_lp):
            log_alpha = proposed_lp - current_lp
            log_alpha += proposal.log_density(current, proposed)
            log_alpha -= proposal.log_density(proposed, current)
            accept = np.log(rng.rand()) < min(0.0, log_alpha)
        else:
            accept = False

        if accept:
            current = proposed
            current_lp = proposed_lp
        accepted.append(bool(accept))
        proposal.adapt(old_current, proposed, bool(accept), step, step < burn_in)

        if step >= burn_in and ((step - burn_in) % thin == 0):
            samples.append(_copy_state(current))
            log_probs.append(current_lp)

    return MCMCResult(samples=samples,
                      log_probs=np.asarray(log_probs, dtype=float),
                      accepted=np.asarray(accepted, dtype=bool))


def metropolis_within_gibbs(log_target: LogTarget,
                            initial: Any,
                            proposals: Any,
                            num_samples: int,
                            burn_in: int = 0,
                            thin: int = 1,
                            rng: Optional[np.random.RandomState] = None) -> MCMCResult:
    """Cycle labelled proposal kernels and accept/reject each against one target.

    This is useful for record/dict states where each proposal updates a field
    or a small block while the full joint log target still owns all model math.
    Retained samples are recorded after complete sweeps through all proposals.
    """
    if num_samples < 0:
        raise ValueError('num_samples must be non-negative.')
    if burn_in < 0:
        raise ValueError('burn_in must be non-negative.')
    if thin <= 0:
        raise ValueError('thin must be positive.')

    kernels = _normalize_transition_proposals(proposals)
    rng = np.random.RandomState() if rng is None else rng
    current = initial
    current_lp = float(log_target(current))
    if not np.isfinite(current_lp):
        raise ValueError('initial state has non-finite log target: %r.' % current_lp)

    total_sweeps = burn_in + num_samples * thin
    samples: List[Any] = []
    log_probs: List[float] = []
    accepted: List[bool] = []
    labels: List[str] = []

    for sweep in range(total_sweeps):
        for label, proposal in kernels:
            step = len(accepted)
            old_current = current
            proposed = proposal.sample(current, rng)
            proposed_lp = float(log_target(proposed))
            if np.isfinite(proposed_lp):
                log_alpha = proposed_lp - current_lp
                log_alpha += proposal.log_density(current, proposed)
                log_alpha -= proposal.log_density(proposed, current)
                accept = np.log(rng.rand()) < min(0.0, log_alpha)
            else:
                accept = False
            if accept:
                current = proposed
                current_lp = proposed_lp
            accepted.append(bool(accept))
            labels.append(label)
            proposal.adapt(old_current, proposed, bool(accept), step, sweep < burn_in)

        if sweep >= burn_in and ((sweep - burn_in) % thin == 0):
            samples.append(_copy_state(current))
            log_probs.append(current_lp)

    return MCMCResult(samples=samples,
                      log_probs=np.asarray(log_probs, dtype=float),
                      accepted=np.asarray(accepted, dtype=bool),
                      transition_labels=tuple(labels))


def hamiltonian_monte_carlo(log_target: LogTarget,
                            grad_log_target: Callable[[Any], Any],
                            initial: Any,
                            num_samples: int,
                            step_size: float,
                            num_steps: int,
                            mass: Any = 1.0,
                            burn_in: int = 0,
                            thin: int = 1,
                            rng: Optional[np.random.RandomState] = None) -> MCMCResult:
    """Run Hamiltonian Monte Carlo for scalar/vector numeric states.

    ``log_target`` may be unnormalized. ``grad_log_target`` must return the
    gradient of that log target with respect to the numeric state. Both
    callables stay user/model-owned; this utility only owns the transition
    mechanics.
    """
    if num_samples < 0:
        raise ValueError('num_samples must be non-negative.')
    if burn_in < 0:
        raise ValueError('burn_in must be non-negative.')
    if thin <= 0:
        raise ValueError('thin must be positive.')
    if step_size <= 0.0 or not np.isfinite(step_size):
        raise ValueError('step_size must be finite and positive.')
    if num_steps <= 0:
        raise ValueError('num_steps must be positive.')

    rng = np.random.RandomState() if rng is None else rng
    current = _numeric_state(initial)
    state_shape = current.shape
    mass_arr = _numeric_mass(mass, state_shape)
    current_external = _restore_numeric_state(current)
    current_lp = float(log_target(current_external))
    if not np.isfinite(current_lp):
        raise ValueError('initial state has non-finite log target: %r.' % current_lp)
    _numeric_gradient(grad_log_target, current, state_shape)

    total_steps = burn_in + num_samples * thin
    samples: List[Any] = []
    log_probs: List[float] = []
    accepted: List[bool] = []

    for step in range(total_steps):
        momentum0 = rng.normal(size=state_shape) * np.sqrt(mass_arr)
        proposal_state, proposal_momentum, proposed_lp = _hmc_leapfrog(
            log_target, grad_log_target, current, momentum0, mass_arr, step_size, num_steps)
        if np.isfinite(proposed_lp):
            log_alpha = proposed_lp - current_lp
            log_alpha += _kinetic_energy(momentum0, mass_arr)
            log_alpha -= _kinetic_energy(proposal_momentum, mass_arr)
            accept = np.log(rng.rand()) < min(0.0, log_alpha)
        else:
            accept = False

        if accept:
            current = proposal_state
            current_lp = proposed_lp
        accepted.append(bool(accept))

        if step >= burn_in and ((step - burn_in) % thin == 0):
            samples.append(_restore_numeric_state(current))
            log_probs.append(current_lp)

    return MCMCResult(samples=samples,
                      log_probs=np.asarray(log_probs, dtype=float),
                      accepted=np.asarray(accepted, dtype=bool),
                      transition_labels=tuple('hmc' for _ in accepted))


def sample_distribution(dist: Any,
                        initial: Any,
                        proposal: Proposal,
                        num_samples: int,
                        burn_in: int = 0,
                        thin: int = 1,
                        rng: Optional[np.random.RandomState] = None,
                        evidence: Optional[Callable[[Any], float]] = None) -> MCMCResult:
    """Sample from a distribution's log-density, optionally with evidence."""
    return metropolis_hastings(distribution_log_target(dist, evidence=evidence),
                               initial=initial, proposal=proposal,
                               num_samples=num_samples, burn_in=burn_in,
                               thin=thin, rng=rng)


def posterior_predictive(samples: Any,
                         sampler: Callable[..., Any],
                         rng: Optional[np.random.RandomState] = None,
                         size: Optional[int] = None) -> List[Any]:
    """Draw posterior predictive samples from retained MCMC states.

    ``sampler`` is called as ``sampler(state, rng)`` or
    ``sampler(state, rng, size)``. It can build a pysparkplug distribution,
    evaluate arbitrary simulation code, or call a model-specific predictive
    function; the MCMC utility only handles iteration and RNG plumbing.
    """
    chain = samples.samples if isinstance(samples, MCMCResult) else samples
    rng = np.random.RandomState() if rng is None else rng
    draws = []
    for state in chain:
        if size is None:
            draws.append(sampler(state, rng))
        else:
            draws.append(sampler(state, rng, size))
    return draws


# ---------------------------------------------------------------------------
# Parameter posterior: flatten/unflatten bridge + samplers
# ---------------------------------------------------------------------------
#
# Sampling p(theta | data) needs three things for each model family:
#   * a way to read theta off a prototype distribution as a flat vector,
#   * an unconstrained reparameterization phi = T(theta) so random-walk / HMC
#     proposals never leave the parameter domain (log for positive scales,
#     stick-breaking for probability simplices), with the log |det dtheta/dphi|
#     Jacobian so the sampled density is the correct posterior in theta-space,
#   * a constructor that rebuilds a fresh distribution from a proposed theta so
#     the data log-likelihood can be evaluated with seq_encode/seq_log_density.
#
# ``ParameterBridge`` packages those operations.  ``build_parameter_bridge``
# resolves the bridge from the prototype's type; unsupported shapes raise a
# clear NotImplementedError naming the type.


@dataclass(frozen=True)
class ParameterBridge:
    """Map between a distribution's parameters and an unconstrained vector.

    Attributes:
        dim: Length of the unconstrained vector ``phi``.
        to_unconstrained: ``theta -> phi`` for the prototype's parameters.
        from_unconstrained: ``phi -> theta`` (parameter-space value).
        log_abs_det_jacobian: ``phi -> log|det dtheta/dphi|`` so a flat prior in
            theta-space maps to the right density in phi-space.
        build: ``theta -> distribution`` rebuilds a model from parameters.
        param_names: Human-readable names for each theta block (diagnostics).
        initial_theta: The prototype's parameters in theta-space (chain start).
    """

    dim: int
    to_unconstrained: Callable[[Any], np.ndarray]
    from_unconstrained: Callable[[np.ndarray], Any]
    log_abs_det_jacobian: Callable[[np.ndarray], float]
    build: Callable[[Any], Any]
    param_names: Tuple[str, ...]
    initial_theta: Any = None


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _stick_breaking_forward(phi: np.ndarray) -> np.ndarray:
    """Map ``k-1`` unconstrained reals to a length-``k`` probability simplex.

    Uses the standard logistic stick-breaking transform (as in Stan).  The
    returned vector is strictly positive and sums to one.
    """
    k = phi.shape[0] + 1
    p = np.empty(k, dtype=float)
    remaining = 1.0
    for i in range(k - 1):
        # offset keeps a uniform-ish base measure but is irrelevant to validity
        z = 1.0 / (1.0 + np.exp(-(phi[i] - np.log(float(k - 1 - i)))))
        p[i] = remaining * z
        remaining = remaining - p[i]
    p[k - 1] = remaining
    return p


def _stick_breaking_log_det(phi: np.ndarray) -> float:
    """Return ``log|det d p / d phi|`` for the stick-breaking transform.

    Only the first ``k-1`` coordinates of ``p`` are free; the Jacobian of the
    map from ``phi`` to those coordinates is lower triangular, so the log abs
    determinant is the sum of the diagonal log terms.
    """
    k = phi.shape[0] + 1
    remaining = 1.0
    total = 0.0
    for i in range(k - 1):
        z = 1.0 / (1.0 + np.exp(-(phi[i] - np.log(float(k - 1 - i)))))
        # d p_i / d phi_i = remaining * z * (1 - z); off-diagonal terms below it
        total += np.log(remaining) + np.log(z) + np.log1p(-z)
        remaining = remaining * (1.0 - z)
    return float(total)


def _seq_log_density_sum(dist: Any, encoded: Any) -> float:
    """Return ``sum_i log p(x_i | dist)`` for a stats or bstats distribution."""
    return float(np.sum(dist.seq_log_density(encoded)))


def _encode_data(prototype: Any, data: Any) -> Tuple[Any, Callable[[Any], Any]]:
    """Encode ``data`` once and return (encoded, encode_fn) for the family.

    ``stats`` distributions encode through ``dist_to_encoder().seq_encode``;
    ``bstats`` distributions expose ``seq_encode`` directly.  The returned
    ``encode_fn`` lets re-encoding happen if a rebuilt distribution needs it
    (categorical encodings depend only on the data, so the cached encoding is
    reused across proposals).
    """
    if hasattr(prototype, 'dist_to_encoder'):
        encoder = prototype.dist_to_encoder()
        return encoder.seq_encode(data), encoder.seq_encode
    if hasattr(prototype, 'seq_encode'):
        return prototype.seq_encode(data), prototype.seq_encode
    raise NotImplementedError(
        '%s exposes neither seq_encode nor dist_to_encoder; cannot encode data.'
        % type(prototype).__name__)


def _make_builder(prototype: Any, ctor_kwargs: Dict[str, Any]) -> Callable[..., Any]:
    cls = type(prototype)

    def build(*args: Any) -> Any:
        return cls(*args, **ctor_kwargs)

    return build


def build_parameter_bridge(prototype: Any) -> ParameterBridge:
    """Build a :class:`ParameterBridge` for a prototype distribution.

    Supported families (both ``pysp.stats`` and ``pysp.bstats`` variants):

    * Gaussian ``(mu, sigma2)`` -> ``(mu, log sigma2)``
    * Gamma ``(k, theta)`` -> ``(log k, log theta)``
    * Exponential ``beta``/``lam`` (positive scalar) -> ``log``
    * Poisson ``lam`` -> ``log lam``
    * Bernoulli ``p`` -> ``logit p``
    * Beta ``(a, b)`` -> ``(log a, log b)``
    * Categorical probability map -> stick-breaking over the simplex

    Raises:
        NotImplementedError: if the family or a parameter shape is unsupported.
    """
    cls_name = type(prototype).__name__
    name = getattr(prototype, 'name', None)
    keys = getattr(prototype, 'keys', None)

    # carry name (and keys when present) so the rebuilt model matches the
    # prototype but never the prior/posterior bookkeeping that would change the
    # likelihood surface.
    kw: Dict[str, Any] = {}
    if name is not None and 'name' in _ctor_param_names(prototype):
        kw['name'] = name

    if cls_name == 'GaussianDistribution':
        def to_u(theta):
            mu, sigma2 = theta
            return np.asarray([float(mu), float(np.log(sigma2))], dtype=float)

        def from_u(phi):
            return (float(phi[0]), float(np.exp(phi[1])))

        build = _make_builder(prototype, kw)
        return ParameterBridge(
            dim=2, to_unconstrained=to_u, from_unconstrained=from_u,
            log_abs_det_jacobian=lambda phi: float(phi[1]),  # d sigma2 / d log sigma2
            build=lambda theta: build(theta[0], theta[1]),
            param_names=('mu', 'sigma2'),
            initial_theta=(float(prototype.mu), float(prototype.sigma2)))

    if cls_name in ('GammaDistribution', 'BetaDistribution'):
        a0, a1 = (('k', 'theta') if cls_name == 'GammaDistribution' else ('a', 'b'))
        build = _make_builder(prototype, kw)

        def to_u(theta):
            return np.asarray([float(np.log(theta[0])), float(np.log(theta[1]))], dtype=float)

        def from_u(phi):
            return (float(np.exp(phi[0])), float(np.exp(phi[1])))

        return ParameterBridge(
            dim=2, to_unconstrained=to_u, from_unconstrained=from_u,
            log_abs_det_jacobian=lambda phi: float(phi[0] + phi[1]),
            build=lambda theta: build(theta[0], theta[1]),
            param_names=(a0, a1),
            initial_theta=(float(getattr(prototype, a0)), float(getattr(prototype, a1))))

    if cls_name == 'ExponentialDistribution':
        positive_attr = 'beta' if hasattr(prototype, 'beta') else 'lam'
        build = _make_builder(prototype, kw)
        return ParameterBridge(
            dim=1,
            to_unconstrained=lambda theta: np.asarray([float(np.log(theta))], dtype=float),
            from_unconstrained=lambda phi: float(np.exp(phi[0])),
            log_abs_det_jacobian=lambda phi: float(phi[0]),
            build=lambda theta: build(theta),
            param_names=(positive_attr,),
            initial_theta=float(getattr(prototype, positive_attr)))

    if cls_name == 'PoissonDistribution':
        build = _make_builder(prototype, kw)
        return ParameterBridge(
            dim=1,
            to_unconstrained=lambda theta: np.asarray([float(np.log(theta))], dtype=float),
            from_unconstrained=lambda phi: float(np.exp(phi[0])),
            log_abs_det_jacobian=lambda phi: float(phi[0]),
            build=lambda theta: build(theta),
            param_names=('lam',),
            initial_theta=float(prototype.lam))

    if cls_name == 'BernoulliDistribution':
        build = _make_builder(prototype, kw)

        def from_u(phi):
            return float(1.0 / (1.0 + np.exp(-phi[0])))

        return ParameterBridge(
            dim=1,
            to_unconstrained=lambda theta: np.asarray(
                [float(np.log(theta) - np.log1p(-theta))], dtype=float),
            from_unconstrained=from_u,
            # d p / d logit = p (1 - p); log = -softplus(-phi) - softplus(phi)
            log_abs_det_jacobian=lambda phi: float(-_softplus(-phi[0]) - _softplus(phi[0])),
            build=lambda theta: build(theta),
            param_names=('p',),
            initial_theta=float(prototype.p))

    if cls_name == 'CategoricalDistribution':
        prob_map = prototype.get_parameters() if hasattr(prototype, 'get_parameters') \
            else getattr(prototype, 'pmap', getattr(prototype, 'prob_map', None))
        if not isinstance(prob_map, Mapping):
            raise NotImplementedError(
                'CategoricalDistribution parameter posterior requires a probability map.')
        labels = tuple(prob_map.keys())
        k = len(labels)
        if k < 2:
            raise NotImplementedError(
                'CategoricalDistribution parameter posterior needs at least two categories.')
        default_value = float(getattr(prototype, 'default_value', 0.0))
        # build keyword set: stats CategoricalDistribution takes default_value/name
        cat_kw = dict(kw)
        if 'default_value' in _ctor_param_names(prototype):
            cat_kw['default_value'] = default_value
        build = _make_builder(prototype, cat_kw)

        def to_u(theta):
            p = np.asarray([float(theta[label]) for label in labels], dtype=float)
            p = np.clip(p, 1.0e-12, None)
            p = p / p.sum()
            # invert stick-breaking
            phi = np.empty(k - 1, dtype=float)
            remaining = 1.0
            for i in range(k - 1):
                z = p[i] / remaining
                z = min(max(z, 1.0e-12), 1.0 - 1.0e-12)
                phi[i] = np.log(z) - np.log1p(-z) + np.log(float(k - 1 - i))
                remaining = remaining - p[i]
            return phi

        def from_u(phi):
            p = _stick_breaking_forward(np.asarray(phi, dtype=float))
            return {label: float(p[i]) for i, label in enumerate(labels)}

        return ParameterBridge(
            dim=k - 1, to_unconstrained=to_u, from_unconstrained=from_u,
            log_abs_det_jacobian=lambda phi: _stick_breaking_log_det(np.asarray(phi, dtype=float)),
            build=lambda theta: build(theta),
            param_names=tuple(str(label) for label in labels),
            initial_theta={label: float(prob_map[label]) for label in labels})

    raise NotImplementedError(
        'sample_parameter_posterior does not support %s; supported families are '
        'Gaussian, Gamma, Exponential, Poisson, Bernoulli, Beta, and Categorical.'
        % cls_name)


def _ctor_param_names(prototype: Any) -> Tuple[str, ...]:
    import inspect
    try:
        sig = inspect.signature(type(prototype).__init__)
        return tuple(sig.parameters.keys())
    except (TypeError, ValueError):
        return ()


def _coerce_prior_logpdf(prior: Any, bridge: ParameterBridge) -> Callable[[Any], float]:
    """Return ``theta -> log prior(theta)`` from a flexible ``prior`` argument.

    Accepts: ``None`` (flat / improper prior, returns 0), a callable taking the
    parameter-space ``theta``, or a pysparkplug distribution exposing
    ``log_density`` whose support matches the bridge's theta representation.
    """
    if prior is None:
        return lambda theta: 0.0
    if callable(prior) and not hasattr(prior, 'log_density'):
        return lambda theta: float(prior(theta))
    if hasattr(prior, 'log_density'):
        return lambda theta: float(prior.log_density(theta))
    raise TypeError('prior must be None, a callable, or a distribution with log_density.')


def sample_parameter_posterior(prototype_dist: Any,
                               data: Any,
                               prior: Any = None,
                               sampler: str = 'mh',
                               steps: int = 2000,
                               burn_in: int = 500,
                               thin: int = 1,
                               seed: Optional[int] = None,
                               proposal: Optional[Proposal] = None,
                               initial: Any = None,
                               step_size: float = 0.05,
                               num_steps: int = 20,
                               return_distributions: bool = False) -> MCMCResult:
    """Sample the parameter posterior ``p(theta | data)`` of a distribution.

    The model family is fixed by ``prototype_dist``; its parameters define the
    sampled space.  The unnormalized log target is the data log-likelihood
    (rebuilding the distribution per proposed ``theta`` and summing
    ``seq_log_density``) plus a prior log-density and the reparameterization
    Jacobian.  Sampling runs in an unconstrained space (see
    :func:`build_parameter_bridge`) so proposals stay in-domain, and retained
    samples are mapped back to parameter space.

    Args:
        prototype_dist: A distribution instance fixing the model family.
        data: Observations accepted by the family's encoder.
        prior: ``None`` (flat), a callable ``theta -> log p(theta)``, or a
            distribution with ``log_density`` over the parameter representation.
        sampler: ``'mh'`` (Metropolis-Hastings) or ``'hmc'`` (Hamiltonian Monte
            Carlo with finite-difference gradients).
        steps: Number of retained posterior samples.
        burn_in: Number of initial transitions to discard.
        thin: Keep one sample every ``thin`` transitions.
        seed: Seed for the RandomState.
        proposal: Optional MH proposal in the unconstrained space; defaults to a
            random walk (MH only).
        initial: Optional starting ``theta``; defaults to the prototype's
            parameters.
        step_size, num_steps: HMC leapfrog controls.
        return_distributions: If True, ``samples`` are rebuilt distribution
            objects instead of parameter-space values.

    Returns:
        MCMCResult whose ``samples`` are parameter-space values (or rebuilt
        distributions) and whose diagnostics come from the underlying driver.
    """
    bridge = build_parameter_bridge(prototype_dist)
    encoded, _ = _encode_data(prototype_dist, data)
    prior_logpdf = _coerce_prior_logpdf(prior, bridge)
    rng = np.random.RandomState(seed)

    def log_target(phi: Any) -> float:
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        if not np.all(np.isfinite(phi)):
            return -np.inf
        theta = bridge.from_unconstrained(phi)
        try:
            dist = bridge.build(theta)
            ll = _seq_log_density_sum(dist, encoded)
            lp = prior_logpdf(theta)
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            return -np.inf
        rv = ll + lp + bridge.log_abs_det_jacobian(phi)
        return rv if np.isfinite(rv) else -np.inf

    theta0 = bridge.initial_theta if initial is None else initial
    phi0 = bridge.to_unconstrained(theta0)

    sampler = sampler.lower()
    if sampler == 'mh':
        if proposal is None:
            proposal = RandomWalkProposal(scale=0.1 * np.ones(bridge.dim, dtype=float))
        raw = metropolis_hastings(
            log_target, initial=phi0 if bridge.dim > 1 else float(phi0[0]),
            proposal=proposal, num_samples=steps, burn_in=burn_in, thin=thin, rng=rng)
    elif sampler == 'hmc':
        grad = _finite_difference_gradient(log_target)
        raw = hamiltonian_monte_carlo(
            log_target, grad_log_target=grad,
            initial=phi0 if bridge.dim > 1 else float(phi0[0]),
            num_samples=steps, step_size=step_size, num_steps=num_steps,
            burn_in=burn_in, thin=thin, rng=rng)
    else:
        raise ValueError("sampler must be 'mh' or 'hmc'.")

    mapped: List[Any] = []
    for phi in raw.samples:
        phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
        theta = bridge.from_unconstrained(phi_arr)
        mapped.append(bridge.build(theta) if return_distributions else theta)

    return MCMCResult(samples=mapped, log_probs=raw.log_probs, accepted=raw.accepted,
                      transition_labels=raw.transition_labels)


def _finite_difference_gradient(log_target: LogTarget,
                                eps: float = 1.0e-5) -> Callable[[Any], Any]:
    """Return a central finite-difference gradient of ``log_target``."""

    def grad(x: Any) -> Any:
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        g = np.zeros_like(arr)
        for i in range(arr.size):
            step = eps * (1.0 + abs(arr[i]))
            up = arr.copy()
            down = arr.copy()
            up[i] += step
            down[i] -= step
            f_up = log_target(up if arr.size > 1 else float(up[0]))
            f_down = log_target(down if arr.size > 1 else float(down[0]))
            if not (np.isfinite(f_up) and np.isfinite(f_down)):
                g[i] = 0.0
            else:
                g[i] = (f_up - f_down) / (2.0 * step)
        return float(g[0]) if np.isscalar(x) or np.ndim(x) == 0 else g

    return grad


def sample_conjugate_posterior(bstats_dist: Any,
                               data: Any,
                               draws: int = 1000,
                               seed: Optional[int] = None,
                               return_distributions: bool = False) -> MCMCResult:
    """Draw exact posterior parameter samples for a conjugate bstats leaf.

    For ``bstats`` distributions carrying a closed-form conjugate prior, the
    posterior over parameters is available analytically.  This runs the
    distribution's own conjugate ``estimate`` update over ``data`` to obtain the
    posterior hyperparameters, then draws iid parameter samples from that
    posterior.  This is an exact alternative to :func:`sample_parameter_posterior`.

    Supported leaves: ``bstats`` Gaussian (NormalGamma posterior, samples
    ``(mu, sigma2)``), Poisson (Gamma posterior, samples ``lam``), and Bernoulli
    (Beta posterior, samples ``p``).

    Args:
        bstats_dist: A ``pysp.bstats`` distribution carrying a conjugate prior.
        data: Observations for the family.
        draws: Number of iid posterior samples.
        seed: Seed for the RandomState.
        return_distributions: Return rebuilt distributions instead of parameters.

    Returns:
        MCMCResult with iid samples (all accepted, no autocorrelation).
    """
    if draws < 0:
        raise ValueError('draws must be non-negative.')
    rng = np.random.RandomState(seed)
    cls_name = type(bstats_dist).__name__

    # run the family's conjugate posterior update via accumulate + estimate
    posterior_dist = _bstats_posterior(bstats_dist, data)
    prior = posterior_dist.get_prior()

    samples: List[Any] = []
    if cls_name == 'GaussianDistribution':
        from pysp.bstats.normgamma import NormalGammaDistribution
        if not isinstance(prior, NormalGammaDistribution):
            raise NotImplementedError(
                'sample_conjugate_posterior(Gaussian) requires a NormalGamma prior.')
        mu0, lam, a, b = prior.get_parameters()
        for _ in range(draws):
            tau = rng.gamma(shape=a, scale=1.0 / b)         # precision ~ Gamma(a, rate=b)
            tau = max(tau, 1.0e-300)
            mu = rng.normal(loc=mu0, scale=np.sqrt(1.0 / (lam * tau)))
            sigma2 = 1.0 / tau
            samples.append(type(bstats_dist)(mu, sigma2) if return_distributions else (float(mu), float(sigma2)))
    elif cls_name == 'PoissonDistribution':
        from pysp.bstats.gamma import GammaDistribution as BGamma
        if not isinstance(prior, BGamma):
            raise NotImplementedError(
                'sample_conjugate_posterior(Poisson) requires a Gamma prior.')
        k, theta = prior.get_parameters()
        for _ in range(draws):
            lam = rng.gamma(shape=k, scale=theta)
            samples.append(type(bstats_dist)(lam) if return_distributions else float(lam))
    elif cls_name == 'BernoulliDistribution':
        from pysp.bstats.beta import BetaDistribution as BBeta
        if not isinstance(prior, BBeta):
            raise NotImplementedError(
                'sample_conjugate_posterior(Bernoulli) requires a Beta prior.')
        a, b = prior.get_parameters()
        for _ in range(draws):
            p = rng.beta(a, b)
            samples.append(type(bstats_dist)(p) if return_distributions else float(p))
    else:
        raise NotImplementedError(
            'sample_conjugate_posterior supports bstats Gaussian, Poisson, and '
            'Bernoulli leaves; got %s.' % cls_name)

    return MCMCResult(samples=samples,
                      log_probs=np.zeros(len(samples), dtype=float),
                      accepted=np.ones(len(samples), dtype=bool))


def _bstats_posterior(bstats_dist: Any, data: Any) -> Any:
    """Run a bstats family's conjugate update over ``data`` and return the
    estimated distribution (which carries the posterior as its prior)."""
    estimator = bstats_dist.estimator()
    factory = estimator.accumulator_factory()
    acc = factory.make()
    encoded = bstats_dist.seq_encode(data)
    weights = np.ones(len(data), dtype=float)
    acc.seq_update(encoded, weights, None)
    return estimator.estimate(acc.value())


def _copy_state(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, (list, tuple)):
        return type(x)(_copy_state(u) for u in x)
    if isinstance(x, dict):
        return {key: _copy_state(value) for key, value in x.items()}
    return x


def _scalar_if_zero_dim(x: Any) -> Any:
    arr = np.asarray(x)
    return float(arr) if arr.shape == () else x


def _numeric_state(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError('numeric MCMC state must be finite.')
    return arr.copy()


def _restore_numeric_state(x: np.ndarray) -> Any:
    return float(x) if x.ndim == 0 else x.copy()


def _numeric_mass(mass: Any, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(mass, dtype=float)
    if arr.shape == ():
        arr = np.full(shape, float(arr), dtype=float)
    else:
        arr = np.broadcast_to(arr, shape).astype(float, copy=True)
    if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError('mass must be finite and positive.')
    return arr


def _numeric_gradient(grad_log_target: Callable[[Any], Any],
                      state: np.ndarray,
                      shape: Tuple[int, ...]) -> np.ndarray:
    grad = np.asarray(grad_log_target(_restore_numeric_state(state)), dtype=float)
    if grad.shape != shape:
        raise ValueError('grad_log_target shape %s does not match state shape %s.' % (grad.shape, shape))
    if not np.all(np.isfinite(grad)):
        raise ValueError('grad_log_target returned non-finite values.')
    return grad


def _hmc_leapfrog(log_target: LogTarget,
                  grad_log_target: Callable[[Any], Any],
                  state: np.ndarray,
                  momentum: np.ndarray,
                  mass: np.ndarray,
                  step_size: float,
                  num_steps: int) -> Tuple[np.ndarray, np.ndarray, float]:
    x = state.copy()
    p = momentum.copy()
    try:
        p = p + 0.5 * step_size * _numeric_gradient(grad_log_target, x, state.shape)
        for i in range(num_steps):
            x = x + step_size * p / mass
            if i != num_steps - 1:
                p = p + step_size * _numeric_gradient(grad_log_target, x, state.shape)
        p = p + 0.5 * step_size * _numeric_gradient(grad_log_target, x, state.shape)
        proposed_lp = float(log_target(_restore_numeric_state(x)))
    except (FloatingPointError, OverflowError, ValueError):
        return state.copy(), momentum.copy(), -np.inf
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return state.copy(), momentum.copy(), -np.inf
    return x, -p, proposed_lp


def _kinetic_energy(momentum: np.ndarray, mass: np.ndarray) -> float:
    return float(0.5 * np.sum(momentum * momentum / mass))


def _normalize_transition_proposals(proposals: Any) -> Tuple[Tuple[str, Proposal], ...]:
    if isinstance(proposals, Mapping):
        items = tuple((str(label), proposal) for label, proposal in proposals.items())
    else:
        items = tuple((str(label), proposal) for label, proposal in proposals)
    if len(items) == 0:
        raise ValueError('at least one proposal is required.')
    for label, proposal in items:
        if not isinstance(proposal, Proposal):
            raise TypeError('proposal %r is not a Proposal instance.' % (label,))
    return items


def _logsumexp(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return -np.inf
    m = float(np.max(arr))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(arr - m))))


__all__ = [
    'AdaptiveCovarianceProposal',
    'AdaptiveRandomWalkProposal',
    'BlockProposal',
    'IndependentProposal',
    'LangevinProposal',
    'MCMCResult',
    'MixtureProposal',
    'ParameterBridge',
    'Proposal',
    'RandomWalkProposal',
    'build_parameter_bridge',
    'distribution_log_target',
    'hamiltonian_monte_carlo',
    'metropolis_hastings',
    'metropolis_within_gibbs',
    'posterior_predictive',
    'sample_conjugate_posterior',
    'sample_distribution',
    'sample_parameter_posterior',
]
