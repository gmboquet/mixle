"""``forecast`` -- horizon predictions with calibrated intervals from a fitted sequence model.

The forecasting front door for state-space families. For a fitted HMM: filter the history to the
current state posterior (the forward-backward's final step), propagate it through the transition
matrix, and at each horizon step draw from the exact predictive mixture over states — so the mean,
the central interval, and the per-step state probabilities all come from the model itself, for ANY
emission family with a sampler (Gaussian, Gamma, categorical, wrapped, neural, ...)::

    f = forecast(hmm, history, horizon=12, level=0.9)
    f.mean, f.lo, f.hi          # (H,) arrays (or lists for non-scalar emissions)
    f.state_probs               # (H, S): where the chain is expected to be at each step

Sampling-based on purpose: exact for the state marginals (``p_T A^h``), Monte Carlo only for the
emission quantiles, so the intervals reflect arbitrary (skewed / multimodal / discrete)
emission families rather than pretending everything is Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Forecast:
    """Per-step predictive summaries plus the state-marginal trajectory."""

    mean: Any
    lo: Any
    hi: Any
    level: float
    state_probs: np.ndarray  # (H, S)
    samples: Any = field(default=None, repr=False)  # (H, n) predictive draws (scalar emissions)


def _exact_positive_int(value: Any, name: str) -> int:
    """An exact positive integer, or a domain ``ValueError``.

    ``bool`` is excluded (``True`` is not a count), and so are floats: ``horizon`` used to be checked
    only with ``horizon < 1``, which NaN passes (every NaN comparison is False) and which a fractional
    value passes too -- both then failed much later inside ``range``/``np.empty`` with a bare TypeError
    naming neither the argument nor this function.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}")
    return int(value)


def _filtered_state_posterior(model: Any, history: Any) -> np.ndarray:
    """``p(state_T | y_{1:T})`` via the engine forward-backward (the final smoothed step is filtered)."""
    from mixle.engines import NUMPY_ENGINE
    from mixle.stats.latent.hidden_markov import hmm_engine_forward_backward, hmm_pad_log_emissions

    hist = list(history)
    if not hist:
        # An empty history reached the forward pass and died with
        # "IndexError: index 0 is out of bounds for axis 1 with size 0" -- an internal shape accident,
        # not a statement about the request. There is no filtered posterior to condition on with no
        # observations; forecasting from the prior is a different (and unrequested) computation.
        raise ValueError(
            "history must contain at least one observation: forecast() conditions on the filtered state "
            "posterior p(state_T | y_1..T), which is undefined with no observations. Pass the observed "
            "sequence, or sample from the model's prior directly if that is what you want."
        )
    num_states = model.n_states
    pr = np.empty((len(hist), num_states), dtype=np.float64)
    enc = model.topics[0].dist_to_encoder().seq_encode(hist)
    for i in range(num_states):
        pr[:, i] = model.topics[i].seq_log_density(enc)
    padded, mask, _ = hmm_pad_log_emissions(pr, np.array([len(hist)]))
    with np.errstate(divide="ignore"):
        log_w = np.log(model.w)
        log_a = np.log(model.transitions)
    _, gamma, _, _ = hmm_engine_forward_backward(NUMPY_ENGINE, padded, log_w, log_a, mask)
    p = np.asarray(gamma)[0, len(hist) - 1, :]
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("history has zero likelihood under the model; cannot filter a state posterior")
    return p / total


def forecast(
    model: Any,
    history: Any,
    horizon: int,
    *,
    level: float = 0.9,
    n: int = 4000,
    seed: int = 0,
    keep_samples: bool = False,
) -> Forecast:
    """Forecast ``horizon`` steps beyond ``history`` under a fitted HMM.

    Args:
        model: a fitted ``HiddenMarkovModelDistribution`` (any emission family with a sampler).
        history: the observed sequence to condition on (one sequence).
        horizon: number of future steps to predict.
        level: central-interval mass (0.9 -> the 5%..95% band).
        n: Monte-Carlo draws per step for the emission quantiles (state marginals are exact).
        seed: reproducibility.
        keep_samples: also return the raw ``(H, n)`` predictive draws (scalar emissions only).

    Raises:
        TypeError: if ``model`` is not a fitted HMM.
        ValueError: for a non-exact/non-positive ``horizon`` or ``n``, a ``level`` outside
            ``(0, 1)``, or an empty ``history``.

    Every control is checked up front, before any state is allocated or any draw is taken. Only the
    horizon used to be checked, and each remaining control failed somewhere else or not at all:
    ``level=-0.5`` produced ``lo=6.75`` above ``hi=2.25`` -- an interval whose lower endpoint exceeds
    its upper one, silently returned as an ordinary Forecast -- while a NaN level surfaced as NumPy's
    generic quantile error and ``n=0`` warned about an empty sample and then raised an opaque
    IndexError partway through the loop.
    """
    if not hasattr(model, "transitions") or not hasattr(model, "topics"):
        raise TypeError("forecast() currently supports fitted HMMs (HiddenMarkovModelDistribution)")
    horizon = _exact_positive_int(horizon, "horizon")
    n = _exact_positive_int(n, "n")
    level = float(level)
    if not np.isfinite(level) or not (0.0 < level < 1.0):
        # (1 - level) / 2 is the lower tail probability; outside (0, 1) it crosses the upper one and
        # the returned band is inverted rather than merely wide.
        raise ValueError(f"level must be a finite central-interval mass strictly between 0 and 1, got {level!r}")

    rng = np.random.RandomState(seed)
    a_mat = np.asarray(model.transitions, dtype=np.float64)
    p = _filtered_state_posterior(model, history)

    samplers = [t.sampler(seed=rng.randint(2**31 - 1)) for t in model.topics]
    alpha_lo, alpha_hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0

    state_probs = np.empty((horizon, len(p)))
    means: list[Any] = []
    los: list[Any] = []
    his: list[Any] = []
    all_draws: list[np.ndarray] = []
    all_scalar = True
    for h in range(horizon):
        p = p @ a_mat  # exact state marginal at T+h+1
        state_probs[h] = p
        counts = rng.multinomial(n, p)
        draws: list[Any] = []
        for s, c in enumerate(counts):
            if c:
                out = samplers[s].sample(int(c))
                draws.extend(list(np.asarray(out)) if np.ndim(out) else [out])
        try:
            arr = np.asarray(draws, dtype=np.float64)
            step_scalar = arr.ndim == 1
        except (TypeError, ValueError):
            step_scalar = False
            arr = None
        # `step_scalar` is THIS step's own outcome, not sticky across steps -- an earlier
        # non-scalar step must not misroute a later step that is perfectly capable of a
        # computed mean/lo/hi into raw draws instead. `all_scalar` (below) separately tracks
        # whether EVERY step was scalar, which only decides the function's overall return shape.
        if step_scalar:
            means.append(float(arr.mean()))
            los.append(float(np.quantile(arr, alpha_lo)))
            his.append(float(np.quantile(arr, alpha_hi)))
            all_draws.append(arr)
        else:
            means.append(draws)  # non-scalar emissions: hand back the draws per step
            los.append(None)
            his.append(None)
        all_scalar = all_scalar and step_scalar

    if all_scalar:
        return Forecast(
            mean=np.asarray(means),
            lo=np.asarray(los),
            hi=np.asarray(his),
            level=level,
            state_probs=state_probs,
            samples=np.asarray(all_draws) if keep_samples else None,
        )
    return Forecast(mean=means, lo=los, hi=his, level=level, state_probs=state_probs)
