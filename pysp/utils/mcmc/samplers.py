"""Generic MCMC drivers over user-supplied log targets and proposals.

The low-level functions here deliberately operate on user-supplied log-target
callables and proposal objects.  That keeps the transition machinery orthogonal
to the distribution, estimator, and compute-engine protocols while still making
ordinary ``dist.log_density(x)`` models easy to sample from.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

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
        positive_var = var > 0.0
        ess = np.full(flat.shape[1], float(n), dtype=float)
        if not np.any(positive_var):
            return float(ess[0]) if arr.ndim == 1 else ess.reshape(arr.shape[1:])

        lag_limit = n - 1 if max_lag is None else min(int(max_lag), n - 1)
        tau = np.ones(int(np.sum(positive_var)), dtype=float)
        centered_pos = centered[:, positive_var]
        var_pos = var[positive_var]
        for lag in range(1, lag_limit + 1):
            rho = np.mean(centered_pos[:-lag] * centered_pos[lag:], axis=0) / var_pos
            positive = rho > 0.0
            if not np.any(positive):
                break
            tau += 2.0 * np.where(positive, rho, 0.0)
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


def nuts(
    log_target: LogTarget,
    grad_log_target: Callable[[Any], Any],
    initial: Any,
    num_samples: int,
    warmup: int = 1000,
    mass: Any = 1.0,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    thin: int = 1,
    rng: np.random.RandomState | None = None,
) -> MCMCResult:
    """No-U-Turn Sampler (Hoffman & Gelman 2014, efficient NUTS with dual-averaging step size).

    Auto-tunes the leapfrog trajectory length (recursive tree doubling, U-turn termination) and,
    during ``warmup``, the step size to hit ``target_accept`` — so unlike fixed-step HMC it needs
    no manual tuning and mixes well on correlated / higher-dimensional posteriors. ``grad_log_target``
    returns the gradient of the (unnormalized) log target; ``mass`` is a diagonal mass matrix.
    """
    if num_samples < 0 or warmup < 0 or thin <= 0:
        raise ValueError("require num_samples>=0, warmup>=0, thin>0.")
    rng = np.random.RandomState() if rng is None else rng
    theta0 = _numeric_state(initial)
    shape = theta0.shape
    mass_arr = _numeric_mass(mass, shape)
    minv = 1.0 / mass_arr  # diagonal inverse mass (momentum -> velocity)
    sqrt_m = np.sqrt(mass_arr)
    delta_max = 1000.0

    def logp(theta):
        return float(log_target(_restore_numeric_state(theta)))

    def gradient(theta):
        return _numeric_gradient(grad_log_target, theta, shape)

    def kinetic(r):
        return 0.5 * float(np.sum(r * r * minv))

    def leapfrog(theta, r, eps):
        r = r + 0.5 * eps * gradient(theta)
        theta = theta + eps * (minv * r)
        r = r + 0.5 * eps * gradient(theta)
        return theta, r

    def no_uturn(tm, tp, rm, rp):
        d = tp - tm
        return float(np.dot(d, minv * rm)) >= 0 and float(np.dot(d, minv * rp)) >= 0

    cur = theta0
    cur_lp = logp(cur)
    if not np.isfinite(cur_lp):
        raise ValueError("initial state has non-finite log target.")
    eps = _find_reasonable_eps(cur, cur_lp, logp, leapfrog, kinetic, sqrt_m, shape, rng)
    mu = math.log(10.0 * eps)
    log_eps_bar, h_bar, gamma, t0, kappa = 0.0, 0.0, 0.05, 10.0, 0.75

    samples: list[Any] = []
    log_probs: list[float] = []
    depths: list[int] = []
    total = warmup + num_samples * thin

    def build_tree(theta, r, logu, v, j, eps, joint0):
        if j == 0:
            theta1, r1 = leapfrog(theta, r, v * eps)
            lp1 = logp(theta1)
            joint1 = lp1 - kinetic(r1)
            n1 = 1 if logu <= joint1 else 0
            s1 = 1 if (joint1 - logu) > -delta_max and np.isfinite(joint1) else 0
            a = min(1.0, math.exp(min(joint1 - joint0, 0.0))) if np.isfinite(joint1) else 0.0
            return theta1, r1, theta1, r1, theta1, lp1, n1, s1, a, 1
        tm, rm, tp, rp, tpr, lpr, n1, s1, a1, na1 = build_tree(theta, r, logu, v, j - 1, eps, joint0)
        if s1 == 1:
            if v == -1:
                tm, rm, _, _, t2, lp2, n2, s2, a2, na2 = build_tree(tm, rm, logu, v, j - 1, eps, joint0)
            else:
                _, _, tp, rp, t2, lp2, n2, s2, a2, na2 = build_tree(tp, rp, logu, v, j - 1, eps, joint0)
            if n2 > 0 and rng.random_sample() < n2 / max(n1 + n2, 1):
                tpr, lpr = t2, lp2
            a1 += a2
            na1 += na2
            n1 += n2
            s1 = s2 if no_uturn(tm, tp, rm, rp) else 0
        return tm, rm, tp, rp, tpr, lpr, n1, s1, a1, na1

    for it in range(total):
        r0 = sqrt_m * rng.standard_normal(shape)
        joint0 = cur_lp - kinetic(r0)
        logu = joint0 - rng.exponential()  # log of a slice height u ~ Uniform(0, exp(joint0))
        tm = tp = cur
        rm = rp = r0
        theta_new, lp_new, n, s, j = cur, cur_lp, 1, 1, 0
        alpha, n_alpha = 0.0, 0
        while s == 1 and j < max_tree_depth:
            v = -1 if rng.random_sample() < 0.5 else 1
            if v == -1:
                tm, rm, _, _, tpr, lpr, n_p, s_p, alpha_p, n_alpha_p = build_tree(
                    tm, rm, logu, v, j, eps, joint0
                )
            else:
                _, _, tp, rp, tpr, lpr, n_p, s_p, alpha_p, n_alpha_p = build_tree(
                    tp, rp, logu, v, j, eps, joint0
                )
            alpha += alpha_p
            n_alpha += n_alpha_p
            if s_p == 1 and rng.random_sample() < min(1.0, n_p / max(n, 1)):
                theta_new, lp_new = tpr, lpr
            n += n_p
            s = s_p if no_uturn(tm, tp, rm, rp) else 0
            j += 1
        cur, cur_lp = theta_new, lp_new

        accept_stat = alpha / max(n_alpha, 1)
        if it < warmup:  # dual-averaging adaptation of the step size
            m1 = it + 1
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

    res = MCMCResult(
        samples=samples,
        log_probs=np.asarray(log_probs, dtype=float),
        accepted=np.ones(len(samples), dtype=bool),  # NUTS always moves (multinomial over the tree)
        transition_labels=tuple("nuts" for _ in samples),
    )
    object.__setattr__(res, "tree_depth", np.asarray(depths, dtype=int))  # frozen dataclass
    object.__setattr__(res, "step_size", float(eps))
    return res


def _find_reasonable_eps(theta, lp0, logp, leapfrog, kinetic, sqrt_m, shape, rng) -> float:
    """Heuristic initial step size: double/halve until one leapfrog step moves the acceptance
    probability across 0.5 (Hoffman & Gelman Algorithm 4)."""
    eps = 1.0
    r = sqrt_m * rng.standard_normal(shape)
    joint0 = lp0 - kinetic(r)

    def joint_after(step):
        t1, r1 = leapfrog(theta, r, step)
        lp1 = logp(t1)
        return (lp1 - kinetic(r1)) if np.isfinite(lp1) else -np.inf

    j1 = joint_after(eps)
    a = 1.0 if (j1 - joint0) > math.log(0.5) else -1.0
    while np.isfinite(j1) and a * (j1 - joint0) > a * math.log(0.5):
        eps *= 2.0**a
        j1 = joint_after(eps)
        if eps < 1e-10 or eps > 1e10:
            break
    return float(eps)


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
