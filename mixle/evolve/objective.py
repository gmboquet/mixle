"""The *measure* contract for self-improvement: a model-agnostic fitness.

An :class:`Objective` turns ``(model, data)`` into a single comparable scalar **and**, when it can, an
per-observation paired vector. The paired vector is what lets the verify gate
(:mod:`mixle.evolve.verify`) run a *paired* significance test instead of comparing two bare score
totals. Objectives that cannot produce a paired vector (a pure summary like a calibration-error
scalar) set ``pointwise`` to return ``None``; the gate then falls back to a CI-exclusion check on the
bootstrapped scalar.

Every builder here is a thin adapter over an existing, verified scorer:

* ``nll_objective``        -> per-obs ``-log p(y_i)`` from ``model.seq_log_density``.
* ``log_score_objective``  -> :func:`mixle.inference.scoring.log_score` on the predictive densities.
* ``crps_objective``       -> :func:`mixle.inference.scoring.crps_ensemble` on a sampled ensemble.
* ``interval_objective``   -> :func:`mixle.inference.scoring.interval_score` (Winkler) on ensemble quantiles.
* ``calibration_objective``-> PIT-based calibration error (:func:`mixle.inference.calibration`), scalar-only.
* ``decision_regret_objective`` -> realized regret of :func:`mixle.inference.decision.bayes_action`.

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
from mixle.inference.decision import bayes_action


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
    """
    sampler = model.sampler(seed)
    row = np.asarray(sampler.sample(int(m)), dtype=float).reshape(-1)
    return np.broadcast_to(row, (int(n), row.shape[0])).copy()


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
    """Logarithmic score (log loss) of the predictive density at the realised outcomes."""

    def pw(model: Any, data: Any) -> np.ndarray:
        prob = np.exp(pointwise_log_density(model, data))
        return np.asarray(_scoring.log_score(prob, mean=False), dtype=float)

    return _ScalarObjective("log_score", True, pw)


def crps_objective(*, ensemble: int = 256, seed: int = 0) -> Objective:
    """Continuous Ranked Probability Score from a sampled predictive ensemble (lower is better).

    Args:
        ensemble: number of predictive draws per observation.
        seed: RNG seed for the ensemble (reproducible).
    """

    def pw(model: Any, data: Any) -> np.ndarray:
        y = _as_array(data)
        f = sample_ensemble(model, y.shape[0], ensemble, seed=seed)
        return np.asarray(_scoring.crps_ensemble(f, y, mean=False), dtype=float)

    return _ScalarObjective("crps", True, pw)


def interval_objective(level: float = 0.9, *, ensemble: int = 256, seed: int = 0) -> Objective:
    """Winkler interval score for the central ``level`` predictive interval (lower is better).

    Args:
        level: central coverage of the interval (e.g. 0.9 for a 90% interval).
        ensemble: number of predictive draws used to read off the interval endpoints.
        seed: RNG seed for the ensemble.
    """
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
    for a histogram statistic, so ``pointwise`` returns ``None`` and the verify gate scores this on the
    bootstrapped scalar.

    (For *classification* models the natural calibration scalar is
    :func:`mixle.inference.calibration.expected_calibration_error`; this builder targets the
    continuous-predictive case, which is the common one for the streaming/auto-select loop.)
    """

    def pw(model: Any, data: Any) -> None:
        return None

    def sc(model: Any, data: Any) -> float:
        y = _as_array(data)
        f = sample_ensemble(model, y.shape[0], ensemble, seed=seed)
        pit = _cal.pit_ensemble(y, f, seed=seed)
        return float(_cal.pit_calibration_error(pit, bins=bins))

    return _ScalarObjective("calibration", True, pw, sc)


def decision_regret_objective(
    loss: Callable[[Any, Any], float],
    actions: Sequence[Any],
    *,
    n: int = 2000,
    seed: int = 0,
) -> Objective:
    """Realized decision regret under a fitted predictive posterior (scalar-only, lower is better).

    For the chosen Bayes action ``a* = bayes_action(posterior(model), loss, actions)`` this reports the
    expected loss ``E_draw[ loss(a*, draw) ]`` -- the realized cost of acting optimally under the
    model's own belief. A better-calibrated model yields a lower expected loss for the same loss and
    action set, so it is a first-class promotion metric. Scalar-only (the regret is an expectation over
    posterior draws, not a per-observation quantity), so ``pointwise`` returns ``None``.

    Args:
        loss: ``loss(action, draw) -> float`` (or a numpy-vectorized ``loss(action, draws) -> array``).
        actions: the finite candidate-action set.
        n: posterior draws for the Monte-Carlo expectation.
        seed: RNG seed.
    """
    from mixle.inference.posterior import posterior as _posterior

    def pw(model: Any, data: Any) -> None:
        return None

    def sc(model: Any, data: Any) -> float:
        post = _posterior(model, over="predictive")
        result = bayes_action(post, loss, list(actions), n=n, seed=seed)
        return float(result["expected_loss"])

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
