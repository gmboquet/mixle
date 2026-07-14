"""Generic MCMC drivers over user-supplied log targets and proposals.

The low-level functions here deliberately operate on user-supplied log-target
callables and proposal objects.  That keeps the transition machinery orthogonal
to the distribution, estimator, and compute-engine protocols while still making
ordinary ``dist.log_density(x)`` models easy to sample from.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.diagnostics import _geyer_tau

from .proposals import Proposal, _normalize_transition_proposals

LogTarget = Callable[[Any], float]


@dataclass(frozen=True)
class MCMCResult:
    """Samples and diagnostics returned by an MCMC run."""

    samples: list[Any]
    log_probs: np.ndarray
    accepted: np.ndarray
    transition_labels: tuple[str, ...] | None = None

    @property
    def acceptance_rate(self) -> float:
        """Return the overall fraction of accepted transitions."""
        return float(np.mean(self.accepted)) if len(self.accepted) else 0.0

    @property
    def acceptance_rate_by_label(self) -> dict[str, float]:
        """Return acceptance rates for labelled transition kernels."""
        if self.transition_labels is None:
            return {}
        if len(self.transition_labels) != len(self.accepted):
            raise ValueError("transition label count does not match accepted count.")
        rv: dict[str, list[bool]] = {}
        for label, accepted in zip(self.transition_labels, self.accepted):
            rv.setdefault(label, []).append(bool(accepted))
        return {label: float(np.mean(values)) for label, values in rv.items()}

    def sample_array(self) -> np.ndarray:
        """Return numeric samples as an ndarray for diagnostics."""
        try:
            arr = np.asarray(self.samples, dtype=float)
        except Exception as e:
            raise ValueError("samples cannot be represented as a numeric array.") from e
        if arr.ndim == 0:
            arr = arr.reshape((1,))
        if len(arr) != len(self.samples):
            raise ValueError("samples have inconsistent numeric shape.")
        return arr

    def effective_sample_size(self, max_lag: int | None = None) -> Any:
        """Estimate effective sample size via Geyer's initial monotone sequence, per component.

        Scalar samples return a float. Vector samples return one ESS value per
        trailing dimension (each component's autocorrelation sum truncates
        independently at its own first non-positive adjacent-lag pair).
        """
        arr = self.sample_array()
        n = int(arr.shape[0])
        if n <= 1:
            return float(n)
        flat = arr.reshape((n, -1))
        centered = flat - flat.mean(axis=0, keepdims=True)
        var = np.mean(centered * centered, axis=0)
        positive_var = var > 0.0
        ess = np.full(flat.shape[1], float(n), dtype=float)
        if not np.any(positive_var):
            return float(ess[0]) if arr.ndim == 1 else ess.reshape(arr.shape[1:])

        lag_limit = n - 1 if max_lag is None else min(int(max_lag), n - 1)
        tau = _geyer_tau(centered[None, :, positive_var], var[positive_var], lag_limit)
        ess[positive_var] = np.maximum(1.0, n / tau)
        return float(ess[0]) if arr.ndim == 1 else ess.reshape(arr.shape[1:])

    def summary(self, max_lag: int | None = None) -> dict[str, Any]:
        """Return basic numeric chain diagnostics.

        The summary intentionally stays dependency-free and returns plain
        numbers/arrays: sample count, mean, variance, Monte Carlo standard
        error estimate, ESS, and acceptance diagnostics.
        """
        arr = self.sample_array()
        n = int(arr.shape[0])
        if n == 0:
            raise ValueError("cannot summarize an empty chain.")
        mean = np.mean(arr, axis=0)
        variance = np.var(arr, axis=0)
        ess = self.effective_sample_size(max_lag=max_lag)
        ess_arr = np.asarray(ess, dtype=float)
        mcse = np.sqrt(np.asarray(variance, dtype=float) / np.maximum(ess_arr, 1.0))
        return {
            "num_samples": n,
            "mean": _scalar_if_zero_dim(mean),
            "variance": _scalar_if_zero_dim(variance),
            "ess": _scalar_if_zero_dim(ess),
            "mcse": _scalar_if_zero_dim(mcse),
            "acceptance_rate": self.acceptance_rate,
            "acceptance_rate_by_label": self.acceptance_rate_by_label,
        }


def distribution_log_target(dist: Any, evidence: Callable[[Any], float] | None = None) -> LogTarget:
    """Return ``log_target(x) = dist.log_density(x) + evidence(x)``."""
    if evidence is None:
        return lambda x: float(dist.log_density(x))
    return lambda x: float(dist.log_density(x)) + float(evidence(x))


# --- MCMC samplers (MH / ensemble / Gibbs / HMC / reflective / dense / NUTS) -
def metropolis_hastings(
    log_target: LogTarget,
    initial: Any,
    proposal: Proposal,
    num_samples: int,
    burn_in: int = 0,
    thin: int = 1,
    rng: np.random.RandomState | None = None,
) -> MCMCResult:
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
        raise ValueError("num_samples must be non-negative.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")

    rng = np.random.RandomState() if rng is None else rng
    current = initial
    current_lp = float(log_target(current))
    if not np.isfinite(current_lp):
        raise ValueError("initial state has non-finite log target: %r." % current_lp)

    total_steps = burn_in + num_samples * thin
    samples: list[Any] = []
    log_probs: list[float] = []
    accepted: list[bool] = []

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

    return MCMCResult(
        samples=samples, log_probs=np.asarray(log_probs, dtype=float), accepted=np.asarray(accepted, dtype=bool)
    )


def affine_invariant_ensemble(
    log_target: Callable[[np.ndarray], float],
    p0: np.ndarray,
    num_samples: int,
    burn_in: int = 0,
    thin: int = 1,
    a: float = 2.0,
    rng: np.random.RandomState | None = None,
) -> MCMCResult:
    """Goodman & Weare affine-invariant ensemble sampler (the "stretch move").

    A population of ``W`` walkers explores the target jointly; each walker is proposed along
    the line to a randomly chosen complementary walker, so the sampler is invariant to affine
    rescalings of the target and needs no per-dimension step tuning. It mixes far better than
    random-walk Metropolis on correlated/poorly-scaled posteriors and, because every proposal
    is one log-target evaluation, delivers very high ESS/sec on low/medium-dimensional models.

    Args:
        log_target: unnormalized log target for a single walker state ``(d,)``.
        p0: initial ensemble, shape ``(W, d)`` with ``W`` even and ``W >= 2*d + 2``.
        num_samples: retained *sweeps*; each sweep contributes all ``W`` walker states.
        burn_in: sweeps to discard. thin: keep one sweep in ``thin``.
        a: stretch scale (>1; 2.0 is the standard default).
        rng: optional RandomState.

    Returns:
        MCMCResult whose ``samples`` are the pooled walker states (sweep-major), so its
        diagnostics see ``W * num_samples / thin`` draws.
    """
    if a <= 1.0:
        raise ValueError("stretch scale a must be > 1.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    rng = np.random.RandomState() if rng is None else rng
    p = np.array(p0, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[0] % 2 != 0:
        raise ValueError("p0 must be (W, d) with W even and >= 2.")
    nwalkers, d = p.shape
    lp = np.array([float(log_target(p[k])) for k in range(nwalkers)])
    if not np.all(np.isfinite(lp)):
        raise ValueError("some initial walkers have non-finite log target.")

    half = nwalkers // 2
    idx = np.arange(nwalkers)
    samples: list[Any] = []
    log_probs: list[float] = []
    accepted: list[bool] = []
    total = burn_in + num_samples * thin
    for sweep in range(total):
        for first in (True, False):
            active = idx[:half] if first else idx[half:]
            other = idx[half:] if first else idx[:half]
            n = len(active)
            # z ~ g(z) ∝ 1/sqrt(z) on [1/a, a]  (inverse-CDF sample)
            z = ((a - 1.0) * rng.random_sample(n) + 1.0) ** 2 / a
            partners = other[rng.randint(0, len(other), size=n)]
            prop = p[partners] + z[:, None] * (p[active] - p[partners])
            for i in range(n):
                w = active[i]
                lpp = float(log_target(prop[i]))
                if np.isfinite(lpp):
                    log_alpha = (d - 1) * np.log(z[i]) + lpp - lp[w]
                    acc = np.log(rng.random_sample()) < log_alpha
                else:
                    acc = False
                accepted.append(bool(acc))
                if acc:
                    p[w] = prop[i]
                    lp[w] = lpp
        if sweep >= burn_in and ((sweep - burn_in) % thin == 0):
            for k in range(nwalkers):
                samples.append(p[k].copy())
                log_probs.append(lp[k])

    return MCMCResult(
        samples=samples, log_probs=np.asarray(log_probs, dtype=float), accepted=np.asarray(accepted, dtype=bool)
    )


def metropolis_within_gibbs(
    log_target: LogTarget,
    initial: Any,
    proposals: Any,
    num_samples: int,
    burn_in: int = 0,
    thin: int = 1,
    rng: np.random.RandomState | None = None,
) -> MCMCResult:
    """Cycle labelled proposal kernels and accept/reject each against one target.

    This is useful for record/dict states where each proposal updates a field
    or a small block while the full joint log target still owns all model math.
    Retained samples are recorded after complete sweeps through all proposals.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")

    kernels = _normalize_transition_proposals(proposals)
    rng = np.random.RandomState() if rng is None else rng
    current = initial
    current_lp = float(log_target(current))
    if not np.isfinite(current_lp):
        raise ValueError("initial state has non-finite log target: %r." % current_lp)

    total_sweeps = burn_in + num_samples * thin
    samples: list[Any] = []
    log_probs: list[float] = []
    accepted: list[bool] = []
    labels: list[str] = []

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

    return MCMCResult(
        samples=samples,
        log_probs=np.asarray(log_probs, dtype=float),
        accepted=np.asarray(accepted, dtype=bool),
        transition_labels=tuple(labels),
    )


def hamiltonian_monte_carlo(
    log_target: LogTarget,
    grad_log_target: Callable[[Any], Any],
    initial: Any,
    num_samples: int,
    step_size: float,
    num_steps: int,
    mass: Any = 1.0,
    burn_in: int = 0,
    thin: int = 1,
    rng: np.random.RandomState | None = None,
) -> MCMCResult:
    """Run Hamiltonian Monte Carlo for scalar/vector numeric states.

    ``log_target`` may be unnormalized. ``grad_log_target`` must return the
    gradient of that log target with respect to the numeric state. Both
    callables stay user/model-owned; this utility only owns the transition
    mechanics.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if step_size <= 0.0 or not np.isfinite(step_size):
        raise ValueError("step_size must be finite and positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    rng = np.random.RandomState() if rng is None else rng
    current = _numeric_state(initial)
    state_shape = current.shape
    mass_arr = _numeric_mass(mass, state_shape)
    current_external = _restore_numeric_state(current)
    current_lp = float(log_target(current_external))
    if not np.isfinite(current_lp):
        raise ValueError("initial state has non-finite log target: %r." % current_lp)
    _numeric_gradient(grad_log_target, current, state_shape)

    total_steps = burn_in + num_samples * thin
    samples: list[Any] = []
    log_probs: list[float] = []
    accepted: list[bool] = []

    for step in range(total_steps):
        momentum0 = rng.normal(size=state_shape) * np.sqrt(mass_arr)
        proposal_state, proposal_momentum, proposed_lp = _hmc_leapfrog(
            log_target, grad_log_target, current, momentum0, mass_arr, step_size, num_steps
        )
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

    return MCMCResult(
        samples=samples,
        log_probs=np.asarray(log_probs, dtype=float),
        accepted=np.asarray(accepted, dtype=bool),
        transition_labels=tuple("hmc" for _ in accepted),
    )


def _reflect_into_box(x: np.ndarray, p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflect any out-of-box coordinates back into ``[lo, hi]``, flipping the corresponding momentum."""
    x = x.copy()
    p = p.copy()
    for _ in range(64):  # repeat until inside (handles a step that overshoots a thin box several times)
        below = x < lo
        above = x > hi
        if not (below.any() or above.any()):
            break
        x = np.where(below, 2.0 * lo - x, x)
        x = np.where(above, 2.0 * hi - x, x)
        p = np.where(below | above, -p, p)
    return np.clip(x, lo, hi), p


def reflective_hmc(
    log_target: LogTarget,
    grad_log_target: Callable[[Any], Any],
    initial: Any,
    lower: Any,
    upper: Any,
    num_samples: int,
    step_size: float,
    num_steps: int,
    mass: Any = 1.0,
    burn_in: int = 0,
    thin: int = 1,
    rng: np.random.RandomState | None = None,
) -> MCMCResult:
    """Hamiltonian Monte Carlo on a box ``[lower, upper]`` by reflecting trajectories off the walls.

    Samples a target constrained to a hyper-rectangle without distorting it: the leapfrog trajectory
    bounces specularly off each boundary (mirror the position, flip that momentum component), which is
    volume-preserving and time-reversible, so the usual Metropolis correction leaves the box-restricted
    target invariant. ``log_target`` (may be unnormalized) and ``grad_log_target`` are defined on the box;
    ``lower``/``upper`` broadcast to the state shape. Returns an :class:`MCMCResult` whose samples all lie
    in the box -- the constrained-HMC answer for box/simplex-style bounds (WS-1 constraints).
    """
    if step_size <= 0.0 or not np.isfinite(step_size):
        raise ValueError("step_size must be finite and positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    rng = np.random.RandomState() if rng is None else rng
    current = _numeric_state(initial)
    shape = current.shape
    lo = np.broadcast_to(np.asarray(lower, dtype=float), shape).astype(float)
    hi = np.broadcast_to(np.asarray(upper, dtype=float), shape).astype(float)
    if np.any(current < lo) or np.any(current > hi):
        raise ValueError("initial state is outside the box [lower, upper].")
    mass_arr = _numeric_mass(mass, shape)
    current_lp = float(log_target(_restore_numeric_state(current)))
    if not np.isfinite(current_lp):
        raise ValueError("initial state has non-finite log target.")

    samples: list[Any] = []
    log_probs: list[float] = []
    accepted: list[bool] = []
    for step in range(burn_in + num_samples * thin):
        momentum0 = rng.normal(size=shape) * np.sqrt(mass_arr)
        x = current.copy()
        p = momentum0.copy()
        grad = _numeric_gradient(grad_log_target, x, shape)
        for _ in range(num_steps):
            p = p + 0.5 * step_size * grad
            x = x + step_size * (p / mass_arr)
            x, p = _reflect_into_box(x, p, lo, hi)
            grad = _numeric_gradient(grad_log_target, x, shape)
            p = p + 0.5 * step_size * grad
        proposed_lp = float(log_target(_restore_numeric_state(x)))
        if np.isfinite(proposed_lp):
            log_alpha = proposed_lp - current_lp
            log_alpha += _kinetic_energy(momentum0, mass_arr) - _kinetic_energy(p, mass_arr)
            accept = np.log(rng.rand()) < min(0.0, log_alpha)
        else:
            accept = False
        if accept:
            current, current_lp = x, proposed_lp
        accepted.append(bool(accept))
        if step >= burn_in and ((step - burn_in) % thin == 0):
            samples.append(_restore_numeric_state(current))
            log_probs.append(current_lp)

    return MCMCResult(
        samples=samples,
        log_probs=np.asarray(log_probs, dtype=float),
        accepted=np.asarray(accepted, dtype=bool),
        transition_labels=tuple("reflective_hmc" for _ in accepted),
    )


def particle_filter(
    observations: Any,
    propagate: Callable[[np.ndarray, np.random.RandomState], Any],
    log_likelihood: Callable[[np.ndarray, Any], Any],
    initial_particles: Any,
    *,
    resample: bool = True,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:
    """Bootstrap particle filter (sequential Monte Carlo) for a general state-space model.

    Propagates a cloud of weighted particles through a user-supplied state-space model and conditions on
    each observation in turn -- the nonlinear/non-Gaussian generalization of the Kalman filter (and a
    member of the SMC family). ``initial_particles`` is an ``(N, d)`` array from the prior; ``propagate(
    particles, rng)`` returns the particles advanced one step through the transition (including process
    noise); ``log_likelihood(particles, y)`` returns the per-particle observation log-density. Each step
    reweights by the likelihood, records the weighted-mean filtered state, and (by default) multinomially
    resamples to fight weight degeneracy; with ``resample=False`` it is sequential importance sampling,
    carrying the weight product ``w_t = w_{t-1} * p(y_t | x_t)`` across steps. Returns
    ``(filtered_means, log_likelihood)`` where the second value is the SMC estimate of the model's
    marginal log-likelihood ``log p(y_1:T)`` (an unbiased evidence estimate, usable for parameter
    inference). For a linear-Gaussian model it converges to the exact Kalman filter as ``N -> infinity``.
    """
    rng = np.random.RandomState() if rng is None else rng
    particles = np.array(initial_particles, dtype=np.float64)
    n = particles.shape[0]
    means: list[np.ndarray] = []
    log_lik = 0.0
    log_w_cum = np.zeros(n)  # carried log-weights (uniform again after every resample)
    for y in observations:
        particles = np.asarray(propagate(particles, rng), dtype=np.float64)
        log_w_cum = log_w_cum + np.asarray(log_likelihood(particles, y), dtype=np.float64)
        max_lw = float(np.max(log_w_cum))
        weights = np.exp(log_w_cum - max_lw)
        w_sum = float(weights.sum())
        weights = weights / w_sum
        means.append(weights @ particles)
        if resample:
            # weights entered this step uniform, so log mean w is the per-step evidence increment
            log_lik += max_lw + np.log(w_sum / n)
            particles = particles[rng.choice(n, size=n, p=weights)]
            log_w_cum = np.zeros(n)
        else:
            # pure SIS: the evidence estimate is the mean carried weight, log p(y_1:t) ~= log mean_i w_t^i
            log_lik = max_lw + np.log(w_sum / n)
    return np.array(means), log_lik


def _dense_hmc_run(lp, grad, x0, num, step, n_leap, m_inv, m_chol, rng):
    """Run HMC for ``num`` samples with a (dense) mass metric; returns ``(samples, n_accepted)``."""
    d = x0.size
    x = x0.copy()
    cur_lp = lp(x)
    out = np.empty((num, d))
    accepted = 0
    for it in range(num):
        p0 = m_chol @ rng.standard_normal(d)
        xn, pn = x.copy(), p0.copy()
        g = grad(xn)
        for _ in range(n_leap):
            pn = pn + 0.5 * step * g
            xn = xn + step * (m_inv @ pn)
            g = grad(xn)
            pn = pn + 0.5 * step * g
        prop_lp = lp(xn)
        ham0 = -cur_lp + 0.5 * p0 @ (m_inv @ p0)
        ham1 = -prop_lp + 0.5 * pn @ (m_inv @ pn)
        if np.isfinite(prop_lp) and np.log(rng.rand()) < ham0 - ham1:
            x, cur_lp = xn, prop_lp
            accepted += 1
        out[it] = x
    return out, accepted


def dense_mass_hmc(
    log_target: LogTarget,
    grad_log_target: Callable[[Any], Any],
    initial: Any,
    num_samples: int,
    step_size: float,
    num_steps: int,
    warmup: int = 500,
    rng: np.random.RandomState | None = None,
) -> MCMCResult:
    """HMC with a warmup-adapted dense mass matrix (the Stan-style Euclidean metric).

    Runs a warmup phase with an identity metric, estimates the posterior covariance ``Sigma`` from the
    warmup draws, and runs the sampling phase with mass matrix ``M = Sigma^{-1}`` (momentum ``~ N(0, M)``,
    position step ``M^{-1} p``). Matching the metric to the posterior shape decorrelates the parameters,
    so strongly-correlated/ill-conditioned targets mix far better than identity-mass HMC -- the standard
    fix that makes HMC competitive on realistic posteriors. ``log_target`` may be unnormalized;
    ``grad_log_target`` returns its gradient. Returns an :class:`MCMCResult` from the sampling phase.
    """
    rng = np.random.RandomState() if rng is None else rng
    x0 = _numeric_state(initial)
    shape = x0.shape
    d = x0.size

    def lp(flat: np.ndarray) -> float:
        return float(log_target(_restore_numeric_state(flat.reshape(shape))))

    def grad(flat: np.ndarray) -> np.ndarray:
        return np.asarray(_numeric_gradient(grad_log_target, flat.reshape(shape), shape), dtype=float).ravel()

    flat0 = x0.ravel().astype(float)
    eye = np.eye(d)
    warm_samples, _ = _dense_hmc_run(lp, grad, flat0, max(warmup, 0), step_size, num_steps, eye, eye, rng)
    if warm_samples.shape[0] > d + 1:  # enough warmup draws to estimate a covariance
        cov = np.cov(warm_samples[warm_samples.shape[0] // 2 :].T)
        cov = np.atleast_2d(cov) + 1.0e-8 * eye
        mass = np.linalg.inv(cov)
    else:
        mass = eye
    start = warm_samples[-1] if warm_samples.shape[0] else flat0
    m_inv = np.linalg.inv(mass)
    m_chol = np.linalg.cholesky(mass)
    samples, accepted = _dense_hmc_run(lp, grad, start, num_samples, step_size, num_steps, m_inv, m_chol, rng)
    return MCMCResult(
        samples=[_restore_numeric_state(s.reshape(shape)) for s in samples],
        log_probs=np.asarray([lp(s) for s in samples], dtype=float),
        accepted=np.concatenate([np.ones(accepted, bool), np.zeros(num_samples - accepted, bool)]),
        transition_labels=tuple("dense_mass_hmc" for _ in range(num_samples)),
    )


def nuts(
    log_target: LogTarget | None = None,
    grad_log_target: Callable[[Any], Any] | None = None,
    initial: Any = None,
    num_samples: int = 0,
    warmup: int = 1000,
    mass: Any = 1.0,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    thin: int = 1,
    rng: np.random.RandomState | None = None,
    *,
    value_and_grad: Callable[[Any], tuple[float, Any]] | None = None,
    adapt_mass: bool = False,
) -> MCMCResult:
    """No-U-Turn Sampler (Hoffman & Gelman 2014, efficient NUTS with dual-averaging step size).

    Auto-tunes the leapfrog trajectory length (recursive tree doubling, U-turn termination) and,
    during ``warmup``, the step size to hit ``target_accept`` — so unlike fixed-step HMC it needs
    no manual tuning and mixes well on correlated / higher-dimensional posteriors. ``mass`` is a
    diagonal mass matrix.

    Two equivalent target interfaces (back-compatible):

    * ``nuts(log_target, grad_log_target, initial, ...)`` — separate value/gradient callables
      (the historical signature). ``grad_log_target`` returns the gradient of the
      (unnormalized) log target.
    * ``nuts(value_and_grad=fn, initial=..., ...)`` — a *fused* callable returning
      ``(logp, grad)`` in one shot. This halves forward passes (the value NUTS already needs for
      the slice/Metropolis criterion is shared with the gradient) and lets the sampler cache
      ``(logp, grad)`` at every trajectory endpoint so shared leapfrog/tree nodes are never
      re-evaluated — typically ~2-3x fewer target evaluations than the split path.
    """
    if num_samples < 0 or warmup < 0 or thin <= 0:
        raise ValueError("require num_samples>=0, warmup>=0, thin>0.")
    if value_and_grad is not None:
        if log_target is not None or grad_log_target is not None:
            raise ValueError("pass either value_and_grad or (log_target, grad_log_target), not both.")
        fused = value_and_grad
    else:
        if log_target is None or grad_log_target is None:
            raise ValueError("nuts requires value_and_grad= or both log_target and grad_log_target.")

        def fused(theta_external):  # adapt the split callables to the fused contract
            return float(log_target(theta_external)), grad_log_target(theta_external)

    if initial is None:
        raise ValueError("nuts requires an initial state.")
    rng = np.random.RandomState() if rng is None else rng
    theta0 = _numeric_state(initial)
    shape = theta0.shape
    mass_arr = _numeric_mass(mass, shape)
    minv = 1.0 / mass_arr  # diagonal inverse mass (momentum -> velocity)
    sqrt_m = np.sqrt(mass_arr)
    delta_max = 1000.0

    # Fused (logp, grad) at a numeric state; counts every evaluation for diagnostics.
    eval_count = [0]

    def value_and_grad_at(theta) -> tuple[float, np.ndarray]:
        eval_count[0] += 1
        lp, g = fused(_restore_numeric_state(theta))
        lp = float(lp)
        grad = np.asarray(g, dtype=float)
        if grad.shape != shape:
            raise ValueError("gradient shape %s does not match state shape %s." % (grad.shape, shape))
        return lp, grad

    def kinetic(r):
        return 0.5 * float(np.sum(r * r * minv))

    # A leapfrog step from a node carrying a cached gradient ``grad`` at ``theta``.
    # Returns the new (theta, r) plus the *fresh* (logp, grad) at the endpoint so the
    # caller can cache it — no endpoint is ever evaluated twice.
    def leapfrog(theta, r, grad, eps):
        r = r + 0.5 * eps * grad
        theta = theta + eps * (minv * r)
        lp1, grad1 = value_and_grad_at(theta)
        r = r + 0.5 * eps * grad1
        return theta, r, lp1, grad1

    def no_uturn(tm, tp, rm, rp):
        d = tp - tm
        return float(np.dot(d, minv * rm)) >= 0 and float(np.dot(d, minv * rp)) >= 0

    cur = theta0
    cur_lp, cur_grad = value_and_grad_at(cur)
    if not np.isfinite(cur_lp):
        raise ValueError("initial state has non-finite log target.")
    eps = _find_reasonable_eps(cur, cur_lp, cur_grad, leapfrog, kinetic, sqrt_m, shape, rng)
    mu = math.log(10.0 * eps)
    log_eps_bar, h_bar, gamma, t0, kappa = 0.0, 0.0, 0.05, 10.0, 0.75

    samples: list[Any] = []
    log_probs: list[float] = []
    depths: list[int] = []
    divergences: list[bool] = []
    div_this = [False]  # set by build_tree when a leapfrog step diverges (energy error / non-finite)
    total = warmup + num_samples * thin

    # Optional diagonal mass-matrix adaptation: accumulate the position variance over the first half
    # of warmup (Welford), then set the inverse mass to that variance (rescaling momentum to the
    # posterior's per-coordinate scale) and restart step-size dual-averaging for the new metric.
    da_offset = 0
    adapt_at = (warmup // 2) if (adapt_mass and warmup >= 20) else -1
    mass_count = 0
    mass_mean = np.zeros(shape, dtype=float)
    mass_m2 = np.zeros(shape, dtype=float)

    # Tree nodes carry (theta, r, grad) at both endpoints so doubling re-extends a leaf
    # from its cached gradient instead of recomputing it.
    def build_tree(theta, r, grad, logu, v, j, eps, joint0):
        if j == 0:
            theta1, r1, lp1, grad1 = leapfrog(theta, r, grad, v * eps)
            joint1 = lp1 - kinetic(r1)
            n1 = 1 if logu <= joint1 else 0
            s1 = 1 if (joint1 - logu) > -delta_max and np.isfinite(joint1) else 0
            if s1 == 0:  # base-case termination == the Hamiltonian diverged (U-turns stop higher up)
                div_this[0] = True
            a = min(1.0, math.exp(min(joint1 - joint0, 0.0))) if np.isfinite(joint1) else 0.0
            return theta1, r1, grad1, theta1, r1, grad1, theta1, lp1, grad1, n1, s1, a, 1
        tm, rm, gm, tp, rp, gp, tpr, lpr, gpr, n1, s1, a1, na1 = build_tree(theta, r, grad, logu, v, j - 1, eps, joint0)
        if s1 == 1:
            if v == -1:
                tm, rm, gm, _, _, _, t2, lp2, g2, n2, s2, a2, na2 = build_tree(tm, rm, gm, logu, v, j - 1, eps, joint0)
            else:
                _, _, _, tp, rp, gp, t2, lp2, g2, n2, s2, a2, na2 = build_tree(tp, rp, gp, logu, v, j - 1, eps, joint0)
            if n2 > 0 and rng.random_sample() < n2 / max(n1 + n2, 1):
                tpr, lpr, gpr = t2, lp2, g2
            a1 += a2
            na1 += na2
            n1 += n2
            s1 = s2 if no_uturn(tm, tp, rm, rp) else 0
        return tm, rm, gm, tp, rp, gp, tpr, lpr, gpr, n1, s1, a1, na1

    for it in range(total):
        div_this[0] = False
        r0 = sqrt_m * rng.standard_normal(shape)
        joint0 = cur_lp - kinetic(r0)
        logu = joint0 - rng.exponential()  # log of a slice height u ~ Uniform(0, exp(joint0))
        tm = tp = cur
        rm = rp = r0
        gm = gp = cur_grad  # cached endpoint gradients shared with the next doubling
        theta_new, lp_new, grad_new, n, s, j = cur, cur_lp, cur_grad, 1, 1, 0
        alpha, n_alpha = 0.0, 1
        while s == 1 and j < max_tree_depth:
            v = -1 if rng.random_sample() < 0.5 else 1
            if v == -1:
                tm, rm, gm, _, _, _, tpr, lpr, gpr, n_p, s_p, alpha, n_alpha = build_tree(
                    tm, rm, gm, logu, v, j, eps, joint0
                )
            else:
                _, _, _, tp, rp, gp, tpr, lpr, gpr, n_p, s_p, alpha, n_alpha = build_tree(
                    tp, rp, gp, logu, v, j, eps, joint0
                )
            if s_p == 1 and rng.random_sample() < min(1.0, n_p / max(n, 1)):
                theta_new, lp_new, grad_new = tpr, lpr, gpr  # carry the proposal's cached gradient
            n += n_p
            s = s_p if no_uturn(tm, tp, rm, rp) else 0
            j += 1
        # The selected proposal always carries its (logp, grad) from the leapfrog that produced it,
        # so the new chain state reuses the cached gradient — never a fresh evaluation per iteration.
        cur, cur_lp, cur_grad = theta_new, lp_new, grad_new

        # Accumulate position variance over the first warmup window for mass adaptation.
        if 0 <= adapt_at and it < adapt_at:
            mass_count += 1
            delta = cur - mass_mean
            mass_mean = mass_mean + delta / mass_count
            mass_m2 = mass_m2 + delta * (cur - mass_mean)
        elif it == adapt_at and mass_count > 1:
            var = mass_m2 / (mass_count - 1)
            minv = np.maximum(var, 1.0e-8)  # inverse mass = posterior variance estimate
            mass_arr = 1.0 / minv
            sqrt_m = np.sqrt(mass_arr)
            eps = _find_reasonable_eps(cur, cur_lp, cur_grad, leapfrog, kinetic, sqrt_m, shape, rng)
            mu = math.log(10.0 * eps)
            h_bar, log_eps_bar = 0.0, 0.0  # restart dual-averaging for the new metric
            da_offset = it  # the new dual-averaging clock starts at m1 = 1 this iteration

        accept_stat = alpha / max(n_alpha, 1)
        if it < warmup:  # dual-averaging adaptation of the step size
            m1 = it - da_offset + 1
            h_bar = (1.0 - 1.0 / (m1 + t0)) * h_bar + (target_accept - accept_stat) / (m1 + t0)
            log_eps = mu - math.sqrt(m1) / gamma * h_bar
            eta = m1 ** (-kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
            eps = math.exp(log_eps)
        elif it == warmup:
            eps = math.exp(log_eps_bar)

        if it >= warmup and ((it - warmup) % thin == 0):
            samples.append(_restore_numeric_state(cur))
            log_probs.append(cur_lp)
            depths.append(j)
            divergences.append(bool(div_this[0]))

    res = MCMCResult(
        samples=samples,
        log_probs=np.asarray(log_probs, dtype=float),
        accepted=np.ones(len(samples), dtype=bool),  # NUTS always moves (multinomial over the tree)
        transition_labels=tuple("nuts" for _ in samples),
    )
    object.__setattr__(res, "tree_depth", np.asarray(depths, dtype=int))  # frozen dataclass
    object.__setattr__(res, "divergences", np.asarray(divergences, dtype=bool))  # post-warmup divergent draws
    object.__setattr__(res, "step_size", float(eps))
    object.__setattr__(res, "num_target_evals", int(eval_count[0]))
    object.__setattr__(res, "inverse_mass", np.asarray(minv, dtype=float))  # adapted (or fixed) diagonal
    return res


def _find_reasonable_eps(theta, lp0, grad0, leapfrog, kinetic, sqrt_m, shape, rng) -> float:
    """Heuristic initial step size: double/halve until one leapfrog step moves the acceptance
    probability across 0.5 (Hoffman & Gelman Algorithm 4). ``leapfrog(theta, r, grad, eps)``
    returns ``(theta1, r1, lp1, grad1)`` and ``grad0`` is the cached gradient at ``theta``."""
    eps = 1.0
    r = sqrt_m * rng.standard_normal(shape)
    joint0 = lp0 - kinetic(r)

    def joint_after(step):
        _t1, r1, lp1, _g1 = leapfrog(theta, r, grad0, step)
        return (lp1 - kinetic(r1)) if np.isfinite(lp1) else -np.inf

    j1 = joint_after(eps)
    a = 1.0 if (j1 - joint0) > math.log(0.5) else -1.0
    while np.isfinite(j1) and a * (j1 - joint0) > a * math.log(0.5):
        eps *= 2.0**a
        j1 = joint_after(eps)
        if eps < 1e-10 or eps > 1e10:
            break
    return float(eps)


# --- sampling, posterior-predictive & multi-chain diagnostics ---------------
def sample_distribution(
    dist: Any,
    initial: Any,
    proposal: Proposal,
    num_samples: int,
    burn_in: int = 0,
    thin: int = 1,
    rng: np.random.RandomState | None = None,
    evidence: Callable[[Any], float] | None = None,
) -> MCMCResult:
    """Sample from a distribution's log-density, optionally with evidence."""
    return metropolis_hastings(
        distribution_log_target(dist, evidence=evidence),
        initial=initial,
        proposal=proposal,
        num_samples=num_samples,
        burn_in=burn_in,
        thin=thin,
        rng=rng,
    )


def posterior_predictive(
    samples: Any, sampler: Callable[..., Any], rng: np.random.RandomState | None = None, size: int | None = None
) -> list[Any]:
    """Draw posterior predictive samples from retained MCMC states.

    ``sampler`` is called as ``sampler(state, rng)`` or
    ``sampler(state, rng, size)``. It can build a mixle distribution,
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


def _chain_array(chain: Any) -> np.ndarray:
    """Coerce a chain (MCMCResult or array-like of states) to a ``(n, d)`` float array."""
    arr = chain.sample_array() if isinstance(chain, MCMCResult) else np.asarray(chain, dtype=float)
    n = int(arr.shape[0])
    return arr.reshape((n, -1))


def gelman_rubin(chains: Sequence[Any]) -> Any:
    """Gelman-Rubin potential scale reduction factor (R-hat) across independent chains.

    R-hat compares the variance *between* chains to the variance *within* chains for each
    parameter. Values near 1.0 indicate the chains have mixed and are sampling a common
    target; values noticeably above 1.0 (a common threshold is 1.01-1.1) flag
    non-convergence -- chains stuck in different regions, too short a run, or poor mixing.
    This is the standard multi-chain convergence check (Gelman & Rubin 1992) and the
    multi-chain complement to :meth:`MCMCResult.effective_sample_size`.

    Args:
        chains: Two or more chains, each an :class:`MCMCResult` or an array-like of states.
            Chains may differ in length; all are truncated to the shortest common length.

    Returns:
        A float for scalar parameters, otherwise an array of R-hat values shaped like a
        single sampled state (one R-hat per parameter dimension).
    """
    arrs = [_chain_array(c) for c in chains]
    m = len(arrs)
    if m < 2:
        raise ValueError("gelman_rubin requires at least two chains.")
    d = arrs[0].shape[1]
    if any(a.shape[1] != d for a in arrs):
        raise ValueError("all chains must have the same parameter dimension.")
    n = min(a.shape[0] for a in arrs)
    if n < 2:
        raise ValueError("each chain needs at least two samples to estimate R-hat.")

    stacked = np.stack([a[:n] for a in arrs], axis=0)  # (m, n, d)
    chain_means = stacked.mean(axis=1)  # (m, d)
    grand_mean = chain_means.mean(axis=0)  # (d,)
    # Between-chain variance (per parameter).
    b = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    # Within-chain variance (per parameter): mean of per-chain sample variances.
    w = np.mean(np.var(stacked, axis=1, ddof=1), axis=0)
    # Marginal posterior variance estimate and the resulting R-hat.
    var_hat = (n - 1) / n * w + b / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(np.where(w > 0.0, var_hat / w, 1.0))
    rhat = np.where(w > 0.0, rhat, 1.0)
    sample_shape = (
        chains[0].sample_array().shape[1:] if isinstance(chains[0], MCMCResult) else np.asarray(arrs[0][0]).shape
    )
    if sample_shape == () or d == 1:
        return float(rhat.reshape(-1)[0])
    return rhat.reshape(sample_shape)


def run_chains(
    sampler: Callable[..., MCMCResult],
    num_chains: int,
    initials: Sequence[Any] | Callable[[np.random.RandomState], Any],
    rng: np.random.RandomState | None = None,
    **sampler_kwargs: Any,
) -> tuple[list[MCMCResult], Any]:
    """Run several independent chains and report their Gelman-Rubin R-hat.

    Each chain is given its own initial state (from ``initials``) and its own RNG seeded
    deterministically from ``rng`` for reproducibility. The chains are otherwise independent,
    so this is the multi-chain convergence harness: overdisperse the initials, run, and check
    the returned R-hat is near 1.0 before trusting the pooled samples.

    Args:
        sampler: Callable invoked as ``sampler(initial=..., rng=..., **sampler_kwargs)`` and
            returning an :class:`MCMCResult` (e.g. :func:`metropolis_hastings`, :func:`nuts`).
        num_chains: Number of independent chains to run (>= 2 for a meaningful R-hat).
        initials: Either a sequence of per-chain initial states (length ``num_chains``) or a
            callable ``initials(rng) -> state`` that draws an overdispersed start per chain.
        rng: Optional RandomState used to seed the per-chain RNGs.
        **sampler_kwargs: Forwarded to ``sampler`` (e.g. ``proposal``, ``num_samples``).

    Returns:
        ``(results, rhat)`` -- the list of per-chain results and their R-hat.
    """
    if num_chains < 2:
        raise ValueError("run_chains needs num_chains >= 2 for a meaningful R-hat.")
    rng = np.random.RandomState() if rng is None else rng
    if not callable(initials):
        seq = list(initials)
        if len(seq) != num_chains:
            raise ValueError("initials sequence length must equal num_chains.")

    results: list[MCMCResult] = []
    for c in range(num_chains):
        chain_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        initial = initials(chain_rng) if callable(initials) else seq[c]
        results.append(sampler(initial=initial, rng=chain_rng, **sampler_kwargs))
    return results, gelman_rubin(results)


# --- numeric state, mass & leapfrog helpers ---------------------------------
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
        raise ValueError("numeric MCMC state must be finite.")
    return arr.copy()


def _restore_numeric_state(x: np.ndarray) -> Any:
    return float(x) if x.ndim == 0 else x.copy()


def _numeric_mass(mass: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(mass, dtype=float)
    if arr.shape == ():
        arr = np.full(shape, float(arr), dtype=float)
    else:
        arr = np.broadcast_to(arr, shape).astype(float, copy=True)
    if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError("mass must be finite and positive.")
    return arr


def _numeric_gradient(grad_log_target: Callable[[Any], Any], state: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    grad = np.asarray(grad_log_target(_restore_numeric_state(state)), dtype=float)
    if grad.shape != shape:
        raise ValueError("grad_log_target shape %s does not match state shape %s." % (grad.shape, shape))
    if not np.all(np.isfinite(grad)):
        raise ValueError("grad_log_target returned non-finite values.")
    return grad


def _hmc_leapfrog(
    log_target: LogTarget,
    grad_log_target: Callable[[Any], Any],
    state: np.ndarray,
    momentum: np.ndarray,
    mass: np.ndarray,
    step_size: float,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
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
