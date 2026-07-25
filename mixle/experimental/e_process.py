"""P9 (experimental) -- anytime-valid receipts via e-processes.

Conformal/coverage receipts are fixed-sample, but much of mixle streams: streaming EM, drift
monitoring, training curves, the evolution loop's champion/challenger gate. A fixed-sample test
that you *peek at* every step spends its error budget silently -- after enough peeks a level-alpha
test rejects a true null with probability far above alpha.

E-processes fix this. An **e-value** is a non-negative statistic with expectation <= 1 under the
null; an **e-process** ``(E_t)`` is a sequence of them that forms a non-negative supermartingale
under the null with ``E_0 = 1``. Ville's inequality then gives, for the whole trajectory at once,

    P_null( sup_t  E_t  >=  1/alpha )  <=  alpha.

So the rule "reject the first time ``E_t >= 1/alpha``" controls the type-I error at ``alpha``
*no matter when you look or when you stop* -- continuous monitoring and optional stopping are
free, with no alpha-spending bookkeeping.

Why this is native to mixle: a likelihood ratio ``q(x)/p(x)`` between two densities is exactly an
e-value when ``p`` is the null (``E_null[q(X)/p(X)] = integral q = 1``), so a running product of
mixle density ratios IS an e-process. This module provides:

* :class:`EProcess` -- a running-product e-process constructed from typed predictable null and
  alternative conditional densities plus explicit normalization evidence, with the
  ``E_t >= 1/alpha`` stopping rule and the anytime-valid guarantee;
* :func:`normal_mixture_eprocess` / :class:`MeanShiftDetector` -- the closed-form Robbins
  normal-mixture e-process for detecting a mean shift of unknown size, and a drift detector built
  on it.

This is exploratory ``mixle.experimental`` code (see the P9 card): its graduation receipt is the
empirically-verified anytime type-I control in ``e_process_test.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validated_alpha(alpha: Any) -> float:
    if (
        isinstance(alpha, (bool, np.bool_))
        or not np.isscalar(alpha)
        or not np.isfinite(alpha)
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("alpha must be finite and satisfy 0 < alpha < 1.")
    return float(alpha)


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar.")
    return float(value)


def _positive_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _safe_exp(log_value: float) -> float:
    if log_value > float(np.log(np.finfo(float).max)):
        return float("inf")
    if log_value < float(np.log(np.nextafter(0.0, 1.0))):
        return 0.0
    return float(np.exp(log_value))


@dataclass(frozen=True)
class TestMartingaleValidity:
    """Evidence that a predictable likelihood ratio defines a test martingale."""

    null_hypothesis: str
    alternative_hypothesis: str
    assumptions: tuple[str, ...]
    normalization_evidence: str
    construction: str = "predictable-conditional-likelihood-ratio/v1"

    def __post_init__(self) -> None:
        text_fields = {
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "normalization_evidence": self.normalization_evidence,
        }
        if any(not isinstance(value, str) or not value.strip() for value in text_fields.values()):
            raise ValueError("martingale hypotheses and normalization evidence must be non-empty strings.")
        if not self.assumptions or any(
            not isinstance(assumption, str) or not assumption.strip() for assumption in self.assumptions
        ):
            raise ValueError("martingale validity requires at least one non-empty assumption.")


@dataclass(frozen=True)
class PredictableLikelihoodRatio:
    """Conditional densities evaluated from history available before the next observation."""

    log_alternative: Callable[[tuple[Any, ...], Any], float]
    log_null: Callable[[tuple[Any, ...], Any], float]
    validity: TestMartingaleValidity

    def __post_init__(self) -> None:
        if not callable(self.log_alternative) or not callable(self.log_null):
            raise TypeError("log_alternative and log_null must be callable.")
        if not isinstance(self.validity, TestMartingaleValidity):
            raise TypeError("validity must be a TestMartingaleValidity receipt.")

    def log_ratio(self, history: tuple[Any, ...], observation: Any) -> float:
        log_alt = _finite_scalar(self.log_alternative(history, observation), "alternative log density")
        log_null = _finite_scalar(self.log_null(history, observation), "null log density")
        return _finite_scalar(log_alt - log_null, "conditional log likelihood ratio")


@dataclass(frozen=True)
class EProcessReceipt:
    """Anytime decision plus construction assumptions and numerical status."""

    e_value: float
    peak_e_value: float
    log_e_value: float
    peak_log_e_value: float
    n: int
    alpha: float
    threshold: float
    rejected: bool
    validity: TestMartingaleValidity
    numerical_status: str
    guarantee: str = "anytime-valid: P_null(ever reject) <= alpha by Ville's inequality"

    def __getitem__(self, key: str) -> Any:
        """Preserve historical receipt-style string lookup."""
        return getattr(self, key)


class EProcess:
    """A running product from a declared predictable conditional likelihood ratio.

    Construction requires a typed validity receipt. The process itself evaluates both conditional
    densities from the pre-observation history, preventing arbitrary log-ratio streams from being
    mislabeled as anytime-valid evidence.
    """

    def __init__(self, martingale: PredictableLikelihoodRatio) -> None:
        if not isinstance(martingale, PredictableLikelihoodRatio):
            raise TypeError("EProcess requires a PredictableLikelihoodRatio.")
        self.martingale = martingale
        self._log_e = 0.0  # log E_0 = 0, i.e. E_0 = 1
        self._peak_log_e = 0.0
        self._t = 0
        self._history: list[Any] = []

    def update(self, observation: Any) -> float:
        """Evaluate and multiply the next predictable conditional likelihood ratio."""
        log_ratio = self.martingale.log_ratio(tuple(self._history), observation)
        self._log_e += log_ratio
        self._t += 1
        self._history.append(observation)
        if self._log_e > self._peak_log_e:
            self._peak_log_e = self._log_e
        return self.e_value

    @property
    def e_value(self) -> float:
        return _safe_exp(self._log_e)

    @property
    def log_e_value(self) -> float:
        return float(self._log_e)

    @property
    def n(self) -> int:
        return self._t

    def rejects(self, alpha: float = 0.05) -> bool:
        """Whether the CURRENT e-value clears the ``1/alpha`` threshold (an anytime-valid reject)."""
        alpha = _validated_alpha(alpha)
        return bool(self._log_e >= -np.log(alpha))

    def ever_rejected(self, alpha: float = 0.05) -> bool:
        """Whether the e-process has crossed ``1/alpha`` at any point so far (peeking-safe)."""
        alpha = _validated_alpha(alpha)
        return bool(self._peak_log_e >= -np.log(alpha))

    def receipt(self, alpha: float = 0.05) -> EProcessReceipt:
        """Return the decision, validity assumptions, and finite/overflow numerical status."""
        alpha = _validated_alpha(alpha)
        current = self.e_value
        peak = _safe_exp(self._peak_log_e)
        numerical_status = "finite" if np.isfinite(current) and np.isfinite(peak) else "overflow"
        return EProcessReceipt(
            e_value=current,
            peak_e_value=peak,
            log_e_value=self.log_e_value,
            peak_log_e_value=float(self._peak_log_e),
            n=self._t,
            alpha=alpha,
            threshold=float(1.0 / alpha),
            rejected=self.ever_rejected(alpha),
            validity=self.martingale.validity,
            numerical_status=numerical_status,
        )


def normal_mixture_log_e(sum_centered: float, t: int, *, sigma: float, tau: float) -> float:
    """Log of the Robbins two-sided normal-mixture e-value for a Gaussian mean shift.

    Tests ``H0: mean == mu0`` for a stream of ``N(mean, sigma^2)`` observations, mixing the
    alternative mean over a ``N(mu0, tau^2)`` prior (so an unknown-size shift is covered without
    choosing it in advance). ``sum_centered = sum_i (x_i - mu0)`` over the ``t`` observations.

    Closed form (a non-negative martingale under H0 with value 1 at t=0)::

        E_t = sqrt( s2 / (s2 + t*T2) ) * exp( T2 * S^2 / (2*s2*(s2 + t*T2)) )

    with ``s2 = sigma^2``, ``T2 = tau^2``, ``S = sum_centered``.
    """
    sum_centered = _finite_scalar(sum_centered, "sum_centered")
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) or int(t) < 0:
        raise ValueError("t must be a non-negative integer.")
    t = int(t)
    sigma = _positive_scalar(sigma, "sigma")
    tau = _positive_scalar(tau, "tau")
    if t == 0:
        return 0.0
    s2 = sigma**2
    tau2 = tau**2
    denom = s2 + t * tau2
    log_e = 0.5 * np.log(s2 / denom) + (tau2 * sum_centered**2) / (2.0 * s2 * denom)
    if not np.isfinite(log_e):
        raise FloatingPointError("normal-mixture log e-value overflowed.")
    return float(log_e)


def normal_mixture_log_eprocess(stream: Any, *, mu0: float, sigma: float, tau: float) -> np.ndarray:
    """Return stable running log e-values for the Robbins normal-mixture process."""
    mu0 = _finite_scalar(mu0, "mu0")
    sigma = _positive_scalar(sigma, "sigma")
    tau = _positive_scalar(tau, "tau")
    try:
        xs = np.asarray(list(stream), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("stream must be a one-dimensional iterable of finite observations.") from exc
    if xs.ndim != 1 or not np.all(np.isfinite(xs)):
        raise ValueError("stream must be a one-dimensional iterable of finite observations.")
    centered = np.cumsum(xs - mu0)
    ts = np.arange(1, len(xs) + 1)
    s2 = sigma**2
    tau2 = tau**2
    denom = s2 + ts * tau2
    log_e = 0.5 * np.log(s2 / denom) + (tau2 * centered**2) / (2.0 * s2 * denom)
    if not np.all(np.isfinite(log_e)):
        raise FloatingPointError("normal-mixture log e-value path overflowed.")
    return log_e


def normal_mixture_eprocess(stream: Any, *, mu0: float, sigma: float, tau: float) -> np.ndarray:
    """Return the running e-values of the Robbins normal-mixture e-process over ``stream``.

    ``result[i]`` is ``E_{i+1}`` after seeing ``stream[:i+1]``. Element 0 corresponds to one
    observation; the process starts at ``E_0 = 1`` implicitly.
    """
    log_e = normal_mixture_log_eprocess(stream, mu0=mu0, sigma=sigma, tau=tau)
    max_log = float(np.log(np.finfo(float).max))
    min_log = float(np.log(np.nextafter(0.0, 1.0)))
    values = np.empty_like(log_e)
    finite = (log_e <= max_log) & (log_e >= min_log)
    values[finite] = np.exp(log_e[finite])
    values[log_e > max_log] = np.inf
    values[log_e < min_log] = 0.0
    return values


@dataclass
class DriftReport:
    """Outcome of a drift scan: whether/when the e-process crossed ``1/alpha``."""

    detected: bool
    detection_time: int | None  # 1-indexed observation count at first crossing, else None
    alpha: float
    peak_e_value: float
    final_e_value: float
    peak_log_e_value: float = 0.0
    final_log_e_value: float = 0.0
    numerical_status: str = "finite"
    assumptions: tuple[str, ...] = (
        "independent Gaussian observations under the null",
        "known positive observation sigma",
        "normal mean-shift mixture with fixed positive tau",
    )
    guarantee: str = field(default="anytime-valid: false-alarm probability <= alpha under the null (Ville)")


class MeanShiftDetector:
    """An anytime-valid detector for a shift in a Gaussian stream's mean, built on the e-process.

    The null is ``mean == mu0`` with known ``sigma``; ``tau`` sets the scale of shifts the mixture
    is most sensitive to. Because it is an e-process, you may test after every observation and stop
    whenever it fires; the false-alarm probability over the whole run is still at most ``alpha``.
    """

    def __init__(self, *, mu0: float, sigma: float, tau: float = 1.0, alpha: float = 0.05) -> None:
        self.mu0 = _finite_scalar(mu0, "mu0")
        self.sigma = _positive_scalar(sigma, "sigma")
        self.tau = _positive_scalar(tau, "tau")
        self.alpha = _validated_alpha(alpha)

    def scan(self, stream: Any) -> DriftReport:
        log_e_values = normal_mixture_log_eprocess(stream, mu0=self.mu0, sigma=self.sigma, tau=self.tau)
        if log_e_values.size == 0:
            return DriftReport(False, None, self.alpha, 1.0, 1.0)
        log_threshold = -np.log(self.alpha)
        crossings = np.flatnonzero(log_e_values >= log_threshold)
        detected = bool(crossings.size > 0)
        detection_time = int(crossings[0] + 1) if detected else None
        peak_log = float(np.max(log_e_values))
        final_log = float(log_e_values[-1])
        peak = _safe_exp(peak_log)
        final = _safe_exp(final_log)
        return DriftReport(
            detected=detected,
            detection_time=detection_time,
            alpha=self.alpha,
            peak_e_value=peak,
            final_e_value=final,
            peak_log_e_value=peak_log,
            final_log_e_value=final_log,
            numerical_status="finite" if np.isfinite(peak) and np.isfinite(final) else "overflow",
        )
