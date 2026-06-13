"""Small MCMC utilities over pysparkplug log-density objects.

The functions here deliberately operate on user-supplied log-target callables
and proposal objects.  That keeps MCMC orthogonal to the distribution,
estimator, and compute-engine protocols while still making ordinary
``dist.log_density(x)`` models easy to sample from.
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
    'Proposal',
    'RandomWalkProposal',
    'distribution_log_target',
    'hamiltonian_monte_carlo',
    'metropolis_hastings',
    'metropolis_within_gibbs',
    'posterior_predictive',
    'sample_distribution',
]
