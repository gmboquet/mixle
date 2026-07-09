"""Bayes-optimal decisions under a fitted mixle posterior.

Given a :class:`~mixle.inference.posterior.Posterior` (the Monte-Carlo law over an unknown -- a
parameter, a latent state, or a future outcome), a loss function ``loss(action, draw) -> float``, and a
finite set of candidate actions, :func:`bayes_action` returns the action that minimises the *posterior
expected loss* and a tail-risk profile (CVaR + loss quantiles) of the chosen action.

This is the decision half of the platform's differentiator: a point predictor returns a number; a
mixle model returns the action that is optimal under the user's own loss *and* explicit about its tail
risk.

It depends only on the public ``Posterior.samples(n, rng)`` contract
(``mixle.inference.posterior``), so it carries no serving / HTTP opinion -- a lack of capability is
reported via :class:`mixle.capability.CapabilityError`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.capability import CapabilityError

Loss = Callable[[Any, Any], float]


@dataclass
class RiskProfile:
    """The tail-risk summary of a single action's posterior loss distribution."""

    expected_loss: float
    cvar: float  # Conditional Value-at-Risk: mean loss in the worst ``alpha`` tail
    cvar_alpha: float
    var: float  # Value-at-Risk: the ``1-alpha`` loss quantile
    quantiles: dict[float, float]
    std: float

    def as_dict(self) -> dict[str, Any]:
        """Return risk metrics and quantiles as JSON-compatible data."""
        return {
            "expected_loss": self.expected_loss,
            "cvar": self.cvar,
            "cvar_alpha": self.cvar_alpha,
            "var": self.var,
            "std": self.std,
            "quantiles": {str(q): v for q, v in self.quantiles.items()},
        }


def _loss_samples(loss: Loss, action: Any, draws: Sequence[Any]) -> np.ndarray:
    """Vectorize-or-loop the loss over posterior draws.

    Tries a single vectorized ``loss(action, draws_array)`` call first (fast path when the loss is
    numpy-aware); falls back to a per-draw Python loop otherwise.
    """
    try:
        arr = np.asarray(loss(action, np.asarray(draws)), dtype=float).reshape(-1)
        if arr.size == len(draws):
            return arr
    except Exception:
        pass
    return np.asarray([float(loss(action, d)) for d in draws], dtype=float)


def _risk_profile(losses: np.ndarray, *, alpha: float, quantiles: Sequence[float]) -> RiskProfile:
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, 1.0 - alpha))  # the (1-alpha) quantile == VaR at level alpha
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var  # mean loss in the worst-alpha tail
    return RiskProfile(
        expected_loss=float(losses.mean()),
        cvar=cvar,
        cvar_alpha=float(alpha),
        var=var,
        quantiles={float(q): float(np.quantile(losses, q)) for q in quantiles},
        std=float(losses.std()),
    )


def bayes_action(
    posterior: Any,
    loss: Loss,
    actions: Sequence[Any],
    *,
    n: int = 2000,
    seed: int = 0,
    cvar_alpha: float = 0.1,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict[str, Any]:
    """Pick the Bayes action: ``argmin_a E_{draw ~ posterior}[ loss(a, draw) ]``.

    Args:
        posterior: any object exposing ``samples(n, rng)`` -- e.g.
            ``mixle.inference.posterior(model, data, over=...)`` (parameter, latent, or predictive).
        loss: ``loss(action, draw) -> float`` (or a numpy-vectorized ``loss(action, draws) -> array``).
        actions: the finite candidate-action set to minimise over.
        n: number of posterior draws for the Monte-Carlo expectation.
        seed: RNG seed for the posterior draw (reproducible).
        cvar_alpha: tail mass for the CVaR / VaR of the chosen action (0.1 -> worst 10%).
        quantiles: loss quantiles to report per action.

    Returns:
        ``{action, action_index, expected_loss, risk_profile, alternatives}`` -- the chosen action, its
        expected loss, its tail-risk profile, and the expected loss of every candidate.

    Raises:
        ValueError: if ``actions`` is empty.
        CapabilityError: if ``posterior`` does not expose the ``samples(n, rng)`` contract.
    """
    actions = list(actions)
    if not actions:
        raise ValueError("bayes_action requires at least one candidate action")

    sample_fn = getattr(posterior, "samples", None)
    if not callable(sample_fn):
        raise CapabilityError(
            f"{type(posterior).__name__} does not support samples(n, rng) (needed for bayes_action); "
            "pass a mixle.inference.posterior(...) object."
        )

    rng = np.random.RandomState(seed)
    draws = sample_fn(int(n), rng)
    # ``samples`` may return an ndarray, a list of scalars, or a list of param dicts -- normalise to a list.
    if isinstance(draws, np.ndarray):
        draw_list: list[Any] = list(draws)
    elif isinstance(draws, dict):
        # a dict of length-n arrays (conjugate parameter posterior) -> n per-draw dicts
        keys = list(draws)
        m = len(np.asarray(draws[keys[0]]))
        draw_list = [{k: np.asarray(draws[k])[j] for k in keys} for j in range(m)]
    else:
        draw_list = list(draws)

    profiles: list[RiskProfile] = []
    expected: list[float] = []
    for action in actions:
        losses = _loss_samples(loss, action, draw_list)
        prof = _risk_profile(losses, alpha=cvar_alpha, quantiles=quantiles)
        profiles.append(prof)
        expected.append(prof.expected_loss)

    best = int(np.argmin(expected))
    return {
        "action": actions[best],
        "action_index": best,
        "expected_loss": expected[best],
        "risk_profile": profiles[best].as_dict(),
        "alternatives": [{"action": a, "expected_loss": e} for a, e in zip(actions, expected)],
    }


__all__ = ["bayes_action", "RiskProfile"]
