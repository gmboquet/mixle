"""Proposal kernels and adaptation helpers for Metropolis-Hastings samplers.

The module collects random-walk, adaptive covariance, mixture, and discrete
proposal objects with explicit transition-density methods for Hastings
corrections.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mixle.utils.special import logsumexp as _logsumexp


class Proposal:
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
            raise ValueError("scale must be finite and positive.")

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
        return float(
            -0.5 * dim * np.log(2.0 * np.pi) - np.sum(np.log(scale)) - 0.5 * np.sum((resid / scale) * (resid / scale))
        )


class AdaptiveRandomWalkProposal(RandomWalkProposal):
    """Gaussian random walk with Robbins-Monro scale adaptation.

    By default adaptation only runs during burn-in, which preserves the
    stationary post-burn chain used for retained samples.
    """

    def __init__(
        self,
        scale: Any,
        target_acceptance: float = 0.44,
        adaptation_rate: float = 0.05,
        adapt_during_burn_in_only: bool = True,
        min_scale: float = 1.0e-12,
        max_scale: float = 1.0e12,
    ) -> None:
        super().__init__(scale)
        if not (0.0 < target_acceptance < 1.0):
            raise ValueError("target_acceptance must be in (0, 1).")
        if adaptation_rate <= 0.0:
            raise ValueError("adaptation_rate must be positive.")
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

    def __init__(
        self,
        initial_covariance: Any = 1.0,
        scale: float | None = None,
        regularization: float = 1.0e-6,
        adapt_after: int = 2,
        adapt_during_burn_in_only: bool = True,
    ) -> None:
        if scale is not None and (scale <= 0.0 or not np.isfinite(scale)):
            raise ValueError("scale must be finite and positive.")
        if regularization < 0.0 or not np.isfinite(regularization):
            raise ValueError("regularization must be finite and non-negative.")
        if adapt_after < 2:
            raise ValueError("adapt_after must be at least 2.")
        self.initial_covariance = np.asarray(initial_covariance, dtype=float)
        if not np.all(np.isfinite(self.initial_covariance)):
            raise ValueError("initial_covariance must be finite.")
        self.user_scale = None if scale is None else float(scale)
        self.scale: float | None = None if scale is None else float(scale)
        self.regularization = float(regularization)
        self.adapt_after = int(adapt_after)
        self.adapt_during_burn_in_only = bool(adapt_during_burn_in_only)
        self._dim: int | None = None
        self._shape: tuple[int, ...] | None = None
        self.covariance: np.ndarray | None = None
        self._proposal_covariance: np.ndarray | None = None
        self._proposal_inv: np.ndarray | None = None
        self._proposal_log_det: float | None = None
        self._count = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None

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
            raise ValueError("adaptive covariance proposal requires finite numeric states.")
        flat = arr.reshape((-1,))
        if self._dim is None:
            self._dim = int(flat.size)
            self._shape = arr.shape
            if self._dim == 0:
                raise ValueError("state must contain at least one numeric value.")
            if self.scale is None:
                self.scale = 2.38 / np.sqrt(float(self._dim))
            self.covariance = self._initial_covariance(self._dim)
            self._refresh_cache()
        elif int(flat.size) != self._dim or arr.shape != self._shape:
            raise ValueError("state shape %s does not match initial shape %s." % (arr.shape, self._shape))
        return flat.copy()

    def _restore(self, x: np.ndarray) -> Any:
        shaped = np.asarray(x, dtype=float).reshape(self._shape)
        return float(shaped) if shaped.ndim == 0 else shaped.copy()

    def _initial_covariance(self, dim: int) -> np.ndarray:
        cov = self.initial_covariance
        if cov.shape == ():
            if float(cov) <= 0.0:
                raise ValueError("initial_covariance scalar must be positive.")
            return np.eye(dim, dtype=float) * float(cov)
        if cov.ndim == 1:
            if cov.shape != (dim,):
                raise ValueError("initial_covariance vector length must match state dimension.")
            if np.any(cov <= 0.0):
                raise ValueError("initial_covariance diagonal entries must be positive.")
            return np.diag(cov)
        if cov.shape != (dim, dim):
            raise ValueError("initial_covariance matrix shape must match state dimension.")
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
            raise ValueError("proposal covariance is not positive definite.")
        self._proposal_covariance = proposal_cov
        self._proposal_inv = np.linalg.inv(proposal_cov)
        self._proposal_log_det = float(log_det)


class IndependentProposal(Proposal):
    """Independence proposal from a sampler plus optional log density."""

    def __init__(
        self, sampler: Callable[[np.random.RandomState], Any], log_density: Callable[[Any], float] | None = None
    ) -> None:
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

    def __init__(self, proposals: Sequence[Proposal], weights: Sequence[float] | None = None) -> None:
        if len(proposals) == 0:
            raise ValueError("MixtureProposal requires at least one proposal.")
        self.proposals = tuple(proposals)
        if weights is None:
            w = np.ones(len(self.proposals), dtype=float) / float(len(self.proposals))
        else:
            w = np.asarray(weights, dtype=float)
        if w.shape != (len(self.proposals),):
            raise ValueError("weights must match proposal count.")
        if np.any(w < 0.0) or not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0.0:
            raise ValueError("weights must be finite, non-negative, and have positive sum.")
        self.weights = w / float(np.sum(w))
        self._last_index: int | None = None

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
            raise ValueError("BlockProposal requires at least one key.")
        self.proposal = proposal

    def _block(self, state: Mapping[Any, Any]) -> Any:
        if len(self.keys) == 1:
            return state[self.keys[0]]
        return tuple(state[key] for key in self.keys)

    def _replace(self, state: Mapping[Any, Any], value: Any) -> dict[Any, Any]:
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
            raise ValueError("proposed block has %d values for %d keys." % (len(values), len(self.keys)))
        for key, item in zip(self.keys, values):
            rv[key] = item
        return rv

    def sample(self, current: Mapping[Any, Any], rng: np.random.RandomState) -> dict[Any, Any]:
        """Propose a replacement for the configured mapping field block."""
        if not isinstance(current, Mapping):
            raise TypeError("BlockProposal requires mapping states.")
        return self._replace(current, self.proposal.sample(self._block(current), rng))

    def log_density(self, proposed: Mapping[Any, Any], current: Mapping[Any, Any]) -> float:
        """Return the block proposal density inside the full mapping state."""
        if not isinstance(proposed, Mapping) or not isinstance(current, Mapping):
            raise TypeError("BlockProposal requires mapping states.")
        return self.proposal.log_density(self._block(proposed), self._block(current))

    def adapt(self, current: Any, proposed: Any, accepted: bool, step: int, in_burn_in: bool) -> None:
        """Forward adaptation to the wrapped block proposal."""
        self.proposal.adapt(self._block(current), self._block(proposed), accepted, step, in_burn_in)


class LangevinProposal(Proposal):
    """Metropolis-adjusted Langevin proposal for scalar/vector states."""

    def __init__(self, step_size: float, grad_log_target: Callable[[Any], Any]) -> None:
        if step_size <= 0.0 or not np.isfinite(step_size):
            raise ValueError("step_size must be finite and positive.")
        self.step_size = float(step_size)
        self.grad_log_target = grad_log_target

    def _mean(self, x: Any) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        grad = np.asarray(self.grad_log_target(x), dtype=float)
        if grad.shape != xx.shape:
            raise ValueError("grad_log_target shape %s does not match state shape %s." % (grad.shape, xx.shape))
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


def _normalize_transition_proposals(proposals: Any) -> tuple[tuple[str, Proposal], ...]:
    if isinstance(proposals, Mapping):
        items = tuple((str(label), proposal) for label, proposal in proposals.items())
    else:
        items = tuple((str(label), proposal) for label, proposal in proposals)
    if len(items) == 0:
        raise ValueError("at least one proposal is required.")
    for label, proposal in items:
        if not isinstance(proposal, Proposal):
            raise TypeError("proposal %r is not a Proposal instance." % (label,))
    return items
