"""The *measure* contract for self-improvement: a model-agnostic fitness.

An :class:`Objective` turns ``(model, data)`` into a single comparable scalar **and**, when it can, a
per-observation paired vector. The paired vector is what lets the verify gate
(:mod:`mixle.evolve.verify`) run a *paired* significance test instead of comparing two bare score
totals. Objectives that cannot produce a paired vector (a pure summary like a calibration-error
scalar) set ``pointwise`` to return ``None``; the gate then falls back to a bare scalar-delta-vs-
``min_effect`` comparison, with ``p_value``/``ci`` set to the explicit ``nan`` "not applicable"
sentinel -- no bootstrap, replication, or other resampling evidence backs a raw scalar delta, so a
scalar-only verdict is reported for a human to review but can never auto-promote on its own (see
:mod:`mixle.evolve.verify` module docstring point 8 and
:attr:`~mixle.evolve.verify.Verdict.has_statistical_evidence`).

Every builder here is a thin adapter over an existing, verified scorer:

* ``nll_objective``        -> per-obs ``-log p(y_i)`` from ``model.seq_log_density``.
* ``log_score_objective``  -> per-obs ``-log p(y_i)`` computed directly from the log-density (the same
  quantity as ``nll_objective``, kept as a separate name for the decision-theoretic framing). Computed
  straight from the log-density, never via ``exp`` then re-``log`` through
  :func:`mixle.inference.scoring.log_score` -- that round trip underflows to ``0.0`` for any log-density
  past about ``-745``, after which the clip-and-relog step can no longer tell two bad fits apart.
* ``crps_objective``       -> :func:`mixle.inference.scoring.crps_ensemble` on a sampled ensemble.
* ``interval_objective``   -> :func:`mixle.inference.scoring.interval_score` (Winkler) on ensemble quantiles.
* ``calibration_objective``-> PIT-based calibration error (:func:`mixle.inference.calibration`), scalar-only.
* ``decision_regret_objective`` -> :func:`mixle.inference.decision.bayes_action`'s chosen action, scored
  by realized loss against the actual ``data`` -- not the model's own posterior draws.

The model-to-array bridges (encoding, per-obs log density, ensemble sampling) live in this module so
the scorers stay pure array functions and the objectives stay five-line adapters.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.inference import calibration as _cal
from mixle.inference import scoring as _scoring
from mixle.inference.decision import _declared_vectorized, _loss_samples, bayes_action


@runtime_checkable
class Objective(Protocol):
    """A lower-is-better-or-higher-with-a-flag scalar fitness with an optional paired vector."""

    name: str
    lower_is_better: bool

    def pointwise(self, model: Any, data: Any) -> np.ndarray | None:
        """The ``(n,)`` per-observation score, or ``None`` if the objective is scalar-only."""
        ...

    def scalar(self, model: Any, data: Any) -> float:
        """The single comparable fitness (mean of ``pointwise`` by default)."""
        ...


# ---------------------------------------------------------------------------
# model -> array bridges (kept here so the scorers stay pure array functions)
# ---------------------------------------------------------------------------
def _as_array(data: Any) -> np.ndarray:
    """Best-effort 1-D float view of held-out responses for the array scorers."""
    return np.asarray(data, dtype=float).reshape(-1)


def _positive_int(value: Any, name: str) -> int:
    """Validate an exact positive integer control (an ensemble size, a bin count, a row count).

    ``ensemble``/``bins`` are advertised budgets: the number of predictive draws or histogram bins
    the score is actually computed from. A zero, negative, or fractional value is not a count, and
    letting one through means the objective silently runs a different experiment than the one its
    signature describes.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}") from None
    if count != value or count < 1:
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}")
    return count


def _exact_int(value: Any, name: str, *, minimum: int) -> int:
    """Validate an exact, non-Boolean integer control that must be at least ``minimum``.

    The ``minimum=0`` generalization of :func:`_positive_int`, shared with the population/loop size
    and acquisition controls in :mod:`mixle.evolve.population` and :mod:`mixle.evolve.closed_loop`
    (MXR-080-1902). Those read their controls with a bare ``int(value)``, which is not validation but
    TRUNCATION: ``size=7.9`` became 7, ``acquire_k=7.9`` became 7, and -- because ``bool`` is an
    ``int`` subclass -- ``size=True`` became 1 and ``acquire_k=True`` became 1. Each of those runs a
    genuinely different, smaller experiment than the one the caller wrote, with nothing in the result
    to say the control was reinterpreted. An integral float (``7.0``) is still accepted, exactly as
    :func:`_positive_int` accepts one: it names the same count unambiguously.

    Deliberately NOT checked here: an upper bound. These are search-budget controls whose sensible
    ceiling depends on the caller's data and compute, not on anything this function can see.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an exact integer >= {minimum}, got {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an exact integer >= {minimum}, got {value!r}") from None
    if count != value or count < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}, got {value!r}")
    return count


def pointwise_log_density(model: Any, data: Sequence[Any]) -> np.ndarray:
    """Per-observation ``log p(y_i)`` under ``model`` via the vectorized ``seq_log_density`` path.

    Models exposing a ``seq_log_density_raw(rows)`` (e.g. an affine recalibration whose
    change-of-variables needs the raw values) are scored through that split-safe path instead of the
    encode-then-score path, which cannot recover the raw responses from an encoded handle.
    """
    raw_path = getattr(model, "seq_log_density_raw", None)
    if callable(raw_path):
        return np.asarray(raw_path(list(data)), dtype=float).reshape(-1)
    enc = model.dist_to_encoder().seq_encode(list(data))
    return np.asarray(model.seq_log_density(enc), dtype=float).reshape(-1)


def sample_ensemble(model: Any, n: int, m: int, *, seed: int) -> np.ndarray:
    """Draw an ``(n, m)`` predictive ensemble: ``m`` iid draws repeated for ``n`` observations.

    The plug-in predictive is exchangeable across observations, so one ``(m,)`` draw row is broadcast
    to all ``n`` rows -- the per-observation CRPS/interval scores then differ only through ``y_i``.

    The realized draw count is checked against ``m`` rather than accepted and broadcast. Nothing
    verified it before, so a sampler that always returned two draws produced shape ``(n, 2)`` for
    ``m=10`` -- and ``m=0`` still produced two. The resulting CRPS/interval score is then a genuinely
    different (far noisier) estimator than the one the objective's ``ensemble`` argument advertises,
    with nothing in the score or the verdict to say so.
    """
    n = _positive_int(n, "n (observations to score)")
    m = _positive_int(m, "m (ensemble draws)")
    sampler = model.sampler(seed)
    row = np.asarray(sampler.sample(m), dtype=float).reshape(-1)
    if row.shape[0] != m:
        raise ValueError(
            f"{type(model).__name__}'s sampler returned {row.shape[0]} draw(s) for a requested ensemble of "
            f"{m}; the realized ensemble must match the requested budget the score is reported against."
        )
    return np.broadcast_to(row, (n, m)).copy()


@dataclass(frozen=True)
class _ScalarObjective:
    """An :class:`Objective` carrying its scalar function and optional pointwise function."""

    name: str
    lower_is_better: bool
    _pointwise: Callable[[Any, Any], np.ndarray | None]
    _scalar: Callable[[Any, Any], float] | None = None

    def pointwise(self, model: Any, data: Any) -> np.ndarray | None:
        return self._pointwise(model, data)

    def scalar(self, model: Any, data: Any) -> float:
        if self._scalar is not None:
            return float(self._scalar(model, data))
        vec = self._pointwise(model, data)
        if vec is None:
            raise ValueError(f"objective {self.name!r} is scalar-only but no scalar function was provided.")
        return float(np.mean(vec))


def nll_objective() -> Objective:
    """Negative log-likelihood: per-obs ``-log p(y_i)`` (strictly proper, lower is better)."""

    def pw(model: Any, data: Any) -> np.ndarray:
        return -pointwise_log_density(model, data)

    return _ScalarObjective("nll", True, pw)


def log_score_objective() -> Objective:
    """Logarithmic score (log loss) of the predictive density at the realised outcomes.

    Computed directly from the log-density (``-log p(y_i)``), never by exponentiating the log-density to
    a plain probability and then re-deriving a log-scale score from it. That round trip is destructive:
    ``exp(x)`` underflows to exactly ``0.0`` for any ``x`` below about ``-745`` (the float64 underflow
    boundary), so every log-density past that point becomes the identical clipped probability and hence
    the identical score, even though the underlying fits are genuinely -- and very differently -- bad.
    Working from the log-density keeps that information: float64 has ample range in log-space, just not
    in probability-space.
    """

    def pw(model: Any, data: Any) -> np.ndarray:
        return -pointwise_log_density(model, data)

    return _ScalarObjective("log_score", True, pw)


def crps_objective(*, ensemble: int = 256, seed: int = 0) -> Objective:
    """Continuous Ranked Probability Score from a sampled predictive ensemble (lower is better).

    Args:
        ensemble: number of predictive draws per observation (an exact positive integer).
        seed: RNG seed for the ensemble (reproducible).
    """
    ensemble = _positive_int(ensemble, "ensemble")

    def pw(model: Any, data: Any) -> np.ndarray:
        y = _as_array(data)
        f = sample_ensemble(model, y.shape[0], ensemble, seed=seed)
        return np.asarray(_scoring.crps_ensemble(f, y, mean=False), dtype=float)

    return _ScalarObjective("crps", True, pw)


def interval_objective(level: float = 0.9, *, ensemble: int = 256, seed: int = 0) -> Objective:
    """Winkler interval score for the central ``level`` predictive interval (lower is better).

    Args:
        level: central coverage of the interval (e.g. 0.9 for a 90% interval).
        ensemble: number of predictive draws used to read off the interval endpoints (an exact
            positive integer).
        seed: RNG seed for the ensemble.
    """
    ensemble = _positive_int(ensemble, "ensemble")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1).")
    alpha = 1.0 - level

    def pw(model: Any, data: Any) -> np.ndarray:
        y = _as_array(data)
        f = sample_ensemble(model, y.shape[0], ensemble, seed=seed)
        lo = np.quantile(f, alpha / 2.0, axis=1)
        hi = np.quantile(f, 1.0 - alpha / 2.0, axis=1)
        return np.asarray(_scoring.interval_score(lo, hi, y, alpha, mean=False), dtype=float)

    return _ScalarObjective(f"interval@{level}", True, pw)


def calibration_objective(*, ensemble: int = 256, seed: int = 0, bins: int = 10) -> Objective:
    """PIT calibration error of the predictive distribution (scalar-only, lower is better).

    Uses the rank-based Probability Integral Transform of a sampled ensemble: under a calibrated
    continuous forecast the PIT values are Uniform(0, 1), and ``pit_calibration_error`` measures the
    histogram's mean absolute deviation from uniform. There is no per-observation paired vector
    for a histogram statistic, so ``pointwise`` returns ``None`` and the verify gate falls back to a
    bare scalar comparison with no p-value or CI, reported for human review only -- see this module's
    docstring and :mod:`mixle.evolve.verify` module docstring point 8.

    (For *classification* models the natural calibration scalar is
    :func:`mixle.inference.calibration.expected_calibration_error`; this builder targets the
    continuous-predictive case, which is the common one for the streaming/auto-select loop.)

    ``ensemble`` and ``bins`` must both be exact positive integers -- they are the budget the
    reported calibration error is computed from.
    """
    ensemble = _positive_int(ensemble, "ensemble")
    bins = _positive_int(bins, "bins")

    def pw(model: Any, data: Any) -> None:
        return None

    def sc(model: Any, data: Any) -> float:
        y = _as_array(data)
        f = sample_ensemble(model, y.shape[0], ensemble, seed=seed)
        pit = _cal.pit_ensemble(y, f, seed=seed)
        return float(_cal.pit_calibration_error(pit, bins=bins))

    return _ScalarObjective("calibration", True, pw, sc)


def _realized_loss(
    loss: Callable[[Any, Any], float], action: Any, data: Any, *, vectorized: bool | None
) -> tuple[float, bool]:
    """Mean loss of a fixed ``action`` against the actual observed ``data``, plus the resolved
    calling convention.

    Delegates to :func:`mixle.inference.decision._loss_samples` -- the same evaluation the Bayes
    action itself is chosen with -- but reduces over real outcomes rather than posterior draws,
    which is what makes :func:`decision_regret_objective` check a chosen action against reality
    instead of against the same model's own beliefs.

    This used to re-implement that evaluation, and the re-implementation swallowed every exception
    from the array call before retrying the loss once per outcome. A loss that was simply broken --
    or a backend that was down -- was therefore called ``len(data) + 1`` times per model scored
    instead of failing, and the original error was discarded. Returning the resolved convention
    lets the caller probe at most once for the whole search rather than once per candidate model.
    """
    values, mode = _loss_samples(loss, action, list(_as_array(data)), vectorized=vectorized, context="decision_regret")
    return float(values.mean()), mode


def decision_regret_objective(
    loss: Callable[[Any, Any], float],
    actions: Sequence[Any],
    *,
    n: int = 2000,
    seed: int = 0,
    vectorized: bool | None = None,
) -> Objective:
    """Realized decision regret of a model's chosen action against actual ``data`` (scalar-only, lower
    is better).

    For the chosen Bayes action ``a* = bayes_action(posterior(model), loss, actions)`` -- optimal under
    the model's OWN predictive belief -- this reports the realized loss ``mean(loss(a*, y_i) for y_i in
    data)`` against the actual ``data``, not draws from that same model's own posterior. Scoring against
    self-generated draws would let a confidently wrong model look flawless: its chosen action and its
    "outcomes" both come from the same wrong belief, so they always agree with each other regardless of
    how far that belief is from reality. Checking the chosen action against real, shared ``data`` is what
    makes this a genuine promotion metric rather than a self-consistency check. Scalar-only (this is a
    single realized-loss number per model, not a per-observation quantity), so ``pointwise`` returns
    ``None``.

    Args:
        loss: ``loss(action, draw) -> float`` (or a numpy-vectorized ``loss(action, draws) -> array``).
        actions: the finite candidate-action set.
        n: posterior draws used only to pick the Bayes-optimal action under the model's own predictive
            belief; the reported score is the realized loss of that action on ``data``.
        seed: RNG seed for the action-selection draws.
        vectorized: the loss's calling convention, forwarded to :func:`bayes_action` and used for
            the realized-loss evaluation. ``True`` -> called once with the whole outcome array;
            ``False`` -> called once per outcome; ``None`` (default) -> read ``loss.vectorized`` if
            the loss declares it, else auto-detect by probing the array call once. A probe is a
            real invocation of the loss, and a search scores many models against the same loss, so
            the resolved convention is reused for the rest of the search rather than re-probed per
            model. A loss that keeps state or has side effects should declare its convention.
    """
    from mixle.inference.posterior import posterior as _posterior

    # resolved once for the whole search: every model scored shares this loss, so re-probing per
    # model would charge the loss one wasted call per candidate for as long as the search runs
    mode: list[bool | None] = [vectorized if vectorized is not None else _declared_vectorized(loss)]

    def pw(model: Any, data: Any) -> None:
        return None

    def sc(model: Any, data: Any) -> float:
        post = _posterior(model, over="predictive")
        decision = bayes_action(post, loss, list(actions), n=n, seed=seed, vectorized=mode[0])
        # bayes_action reports the convention it resolved; take it rather than probing again. The
        # realized-loss pass used to rediscover the same fact, so an undeclared loss was invoked one
        # extra time per model scored purely to learn something already known one line above.
        mode[0] = decision.get("vectorized", mode[0])
        score, mode[0] = _realized_loss(loss, decision["action"], data, vectorized=mode[0])
        return score

    return _ScalarObjective("decision_regret", True, pw, sc)


__all__ = [
    "Objective",
    "nll_objective",
    "log_score_objective",
    "crps_objective",
    "interval_objective",
    "calibration_objective",
    "decision_regret_objective",
    "pointwise_log_density",
    "sample_ensemble",
]
