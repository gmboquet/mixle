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


def _declared_vectorized(loss: Loss) -> bool | None:
    """Read a loss's declared calling convention: ``loss.vectorized = True/False``, else ``None``."""
    flag = getattr(loss, "vectorized", None)
    return bool(flag) if isinstance(flag, (bool, np.bool_)) else None


def _vector_losses(loss: Loss, action: Any, draws: Sequence[Any]) -> np.ndarray:
    """Evaluate a vectorized loss exactly once over the whole draw array."""
    arr = np.asarray(loss(action, np.asarray(draws)), dtype=float).reshape(-1)
    if arr.size != len(draws):
        raise ValueError(
            f"vectorized loss returned {arr.size} value(s) for {len(draws)} draws (action={action!r}); "
            "a vectorized loss must return one loss per draw."
        )
    return arr


def _scalar_losses(loss: Loss, action: Any, draws: Sequence[Any], *, context: str = "bayes_action") -> np.ndarray:
    """Evaluate a scalar loss exactly once per draw, reporting which draw failed."""
    out = np.empty(len(draws), dtype=float)
    for i, draw in enumerate(draws):
        try:
            out[i] = float(loss(action, draw))
        except Exception as exc:
            exc.add_note(f"raised by loss(action={action!r}, draw #{i} of {len(draws)}) in {context}")
            raise
    return out


def _loss_samples(
    loss: Loss, action: Any, draws: Sequence[Any], *, vectorized: bool | None, context: str = "bayes_action"
) -> tuple[np.ndarray, bool]:
    """Evaluate the loss over posterior draws; return the losses and the resolved calling convention.

    ``vectorized`` is the caller's declaration: ``True`` calls the loss once with the whole draw array,
    ``False`` calls it once per draw, and neither invokes it more times than that. ``None`` keeps the
    legacy auto-detection, which *probes* by trying the array call once -- a probe is a real invocation
    of user code, so a stateful or side-effecting loss should declare its convention rather than be
    guessed at. The resolved convention is returned so the probe happens at most once per decision, not
    once per candidate action.
    """
    if vectorized is True:
        values, mode = _vector_losses(loss, action, draws), True
    elif vectorized is False:
        values, mode = _scalar_losses(loss, action, draws, context=context), False
    else:
        try:
            values, mode = _vector_losses(loss, action, draws), True
        except Exception as probe_error:  # noqa: BLE001 - no exception type separates the two cases
            # A per-draw loss handed the whole array raises whatever its body raises -- TypeError
            # from an index, ValueError from a shape check, KeyError from a lookup. Narrowing the
            # catch to any one of those breaks the others, so the probe stays broad and the cost is
            # made addressable instead: declare vectorized=True/False (or loss.vectorized) to skip
            # discovery entirely, which is what a metered or stateful loss should do. If the scalar
            # retry fails too, the discarded probe error is attached rather than lost.
            try:
                values, mode = _scalar_losses(loss, action, draws, context=context), False
            except Exception as exc:
                exc.add_note(
                    f"the vectorized-call probe first failed with: {probe_error!r} -- pass "
                    "vectorized=True/False (or set loss.vectorized) to skip this discovery"
                )
                raise
    unit, caller = ("posterior draw", "bayes_action") if context == "bayes_action" else ("outcome", context)
    return _validated_losses(values, action, unit=unit, caller=caller), mode


def _validated_losses(
    values: np.ndarray, action: Any, *, unit: str = "posterior draw", caller: str = "bayes_action"
) -> np.ndarray:
    """Reject NaN losses, which ``argmin`` would otherwise rank ahead of every real candidate.

    A NaN means the loss could not be evaluated, not that the action is attractive: NaN fails every
    ordered comparison, so ``np.argmin`` returns its position and an unscored action is reported as
    optimal with an all-NaN risk profile. Infinite losses are deliberately left alone -- ``+inf`` is a
    legitimate way to write "inadmissible" and never wins an argmin.
    """
    bad = np.flatnonzero(np.isnan(values))
    if bad.size:
        raise ValueError(
            f"loss for action {action!r} is NaN on {bad.size} of {values.size} {unit}(s) "
            f"(first at #{int(bad[0])}); {caller} cannot use an unscored action."
        )
    return values


def _draw_count(value: Any) -> int:
    """Return the requested number of posterior draws as an exact positive integer."""
    if isinstance(value, (bool, np.bool_, float, np.floating)):
        raise TypeError(f"bayes_action n must be an exact positive integer, got {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"bayes_action n must be an exact positive integer, got {value!r}") from exc
    if count <= 0:
        raise ValueError(f"bayes_action n must be positive, got {count!r}")
    return count


def _tail_mass(value: Any) -> float:
    """Return the CVaR tail mass as a finite fraction in ``(0, 1]``."""
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"cvar_alpha must be a real number, got {value!r}") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(f"cvar_alpha must be finite and in (0, 1], got {alpha!r}")
    return alpha


def _draw_list(draws: Any, requested: int) -> list[Any]:
    """Normalise a ``samples`` return value to a nonempty, aligned list of per-draw values."""
    # ``samples`` may return an ndarray, a list of scalars, or a dict of per-parameter arrays.
    if isinstance(draws, np.ndarray):
        normalised: list[Any] = list(draws)
    elif isinstance(draws, dict):
        # a dict of length-n arrays (conjugate parameter posterior) -> n per-draw dicts
        keys = list(draws)
        if not keys:
            raise ValueError("posterior samples returned an empty parameter dictionary")
        columns = {k: np.atleast_1d(np.asarray(draws[k])) for k in keys}
        lengths = {k: len(v) for k, v in columns.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"posterior samples returned ragged parameter arrays: {lengths}")
        normalised = [{k: columns[k][j] for k in keys} for j in range(next(iter(lengths.values())))]
    else:
        normalised = list(draws)
    if not normalised:
        raise ValueError(f"posterior samples returned no draws for the requested n={requested}")
    return normalised


def _tail_mass_mean(ordered: np.ndarray, alpha: float) -> float:
    """Mean of exactly the worst ``alpha`` probability mass of an ascending-sorted loss sample.

    Averaging every loss ``>= VaR`` is wrong whenever the VaR falls on an atom: all the ties at that
    value join the tail, so a much larger mass than ``alpha`` is averaged and the reported risk is
    diluted. Here the tail carries ``alpha * n`` observations' worth of mass -- whole observations from
    the top, plus only the fractional remainder of the observation straddling the VaR boundary.
    """
    n = ordered.size
    mass = alpha * n
    whole = min(int(np.floor(mass)), n)
    total = float(ordered[n - whole :].sum()) if whole else 0.0
    remainder = mass - whole
    if remainder > 0.0 and whole < n:
        total += remainder * float(ordered[n - whole - 1])
    return total / mass


def _risk_profile(losses: np.ndarray, *, alpha: float, quantiles: Sequence[float]) -> RiskProfile:
    losses = np.asarray(losses, dtype=float)
    ordered = np.sort(losses)
    var = float(np.quantile(losses, 1.0 - alpha))  # the (1-alpha) quantile == VaR at level alpha
    cvar = _tail_mass_mean(ordered, alpha)  # mean loss over exactly the worst-alpha tail mass
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
    vectorized: bool | None = None,
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
        vectorized: the loss's calling convention. ``True`` -> called once per action with the whole
            draw array; ``False`` -> called once per draw; ``None`` (default) -> read ``loss.vectorized``
            if the loss declares it, else auto-detect by *probing* the array call once. A probe is a real
            invocation of the loss, so a loss that keeps state, records calls, or has side effects should
            declare its convention (argument or attribute) to be evaluated exactly the required number of
            times.

    Returns:
        ``{action, action_index, expected_loss, risk_profile, alternatives}`` -- the chosen action, its
        expected loss, its tail-risk profile, and the expected loss of every candidate.

    Raises:
        ValueError: if ``actions`` is empty, ``n`` is not positive, ``cvar_alpha`` is outside ``(0, 1]``,
            the posterior returns an empty or ragged sample batch, or a loss evaluates to NaN.
        TypeError: if ``n`` is not an exact integer.
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

    requested = _draw_count(n)
    alpha = _tail_mass(cvar_alpha)
    rng = np.random.RandomState(seed)
    draw_list = _draw_list(sample_fn(requested, rng), requested)

    profiles: list[RiskProfile] = []
    expected: list[float] = []
    mode = vectorized if vectorized is not None else _declared_vectorized(loss)
    for action in actions:
        losses, mode = _loss_samples(loss, action, draw_list, vectorized=mode)
        prof = _risk_profile(losses, alpha=alpha, quantiles=quantiles)
        profiles.append(prof)
        expected.append(prof.expected_loss)

    best = int(np.argmin(expected))
    return {
        "action": actions[best],
        "action_index": best,
        "expected_loss": expected[best],
        "risk_profile": profiles[best].as_dict(),
        "alternatives": [{"action": a, "expected_loss": e} for a, e in zip(actions, expected)],
        # The convention this call resolved, so a caller that goes on to evaluate the same loss --
        # decision_regret_objective scoring the chosen action against real data, say -- can pass it
        # straight back instead of probing the loss a second time. A probe is a real invocation of
        # user code; discovering the same fact twice is one wasted call per decision, forever.
        "vectorized": mode,
    }


__all__ = ["bayes_action", "RiskProfile"]
