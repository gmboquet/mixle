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

* :class:`EProcess` -- the generic running-product e-process from per-step log density ratios,
  with the ``E_t >= 1/alpha`` stopping rule and the anytime-valid guarantee;
* :func:`normal_mixture_eprocess` / :class:`MeanShiftDetector` -- the closed-form Robbins
  normal-mixture e-process for detecting a mean shift of unknown size, and a drift detector built
  on it.

This is exploratory ``mixle.experimental`` code (see the P9 card): its graduation receipt is the
empirically-verified anytime type-I control in ``e_process_test.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class EProcess:
    """A generic e-process: the running product of per-step density ratios ``q/p``.

    Feed it, one observation at a time, the log density under the alternative and under the null
    (or their difference directly). Under the null, ``e_value`` is a non-negative martingale with
    mean 1, so the stopping rule :meth:`rejects` controls type-I error at any stopping time.
    """

    def __init__(self) -> None:
        self._log_e = 0.0  # log E_0 = 0, i.e. E_0 = 1
        self._peak_log_e = 0.0
        self._t = 0

    def update(self, log_ratio: float) -> float:
        """Multiply in one factor ``exp(log_ratio) = q(x_t)/p(x_t)``; return the new ``e_value``."""
        self._log_e += float(log_ratio)
        self._t += 1
        if self._log_e > self._peak_log_e:
            self._peak_log_e = self._log_e
        return self.e_value

    def update_densities(self, log_alt: float, log_null: float) -> float:
        """Update from separate alternative/null log-densities of the current observation."""
        return self.update(log_alt - log_null)

    @property
    def e_value(self) -> float:
        return float(np.exp(self._log_e))

    @property
    def log_e_value(self) -> float:
        return float(self._log_e)

    @property
    def n(self) -> int:
        return self._t

    def rejects(self, alpha: float = 0.05) -> bool:
        """Whether the CURRENT e-value clears the ``1/alpha`` threshold (an anytime-valid reject)."""
        return bool(self._log_e >= -np.log(alpha))

    def ever_rejected(self, alpha: float = 0.05) -> bool:
        """Whether the e-process has crossed ``1/alpha`` at any point so far (peeking-safe)."""
        return bool(self._peak_log_e >= -np.log(alpha))

    def receipt(self, alpha: float = 0.05) -> dict[str, Any]:
        """A small anytime-valid receipt: current/peak e-value, n, and the reject decision."""
        return {
            "e_value": self.e_value,
            "peak_e_value": float(np.exp(self._peak_log_e)),
            "n": self._t,
            "alpha": float(alpha),
            "threshold": float(1.0 / alpha),
            "rejected": self.ever_rejected(alpha),
            "guarantee": "anytime-valid: P_null(ever reject) <= alpha by Ville's inequality",
        }


def normal_mixture_log_e(sum_centered: float, t: int, *, sigma: float, tau: float) -> float:
    """Log of the Robbins two-sided normal-mixture e-value for a Gaussian mean shift.

    Tests ``H0: mean == mu0`` for a stream of ``N(mean, sigma^2)`` observations, mixing the
    alternative mean over a ``N(mu0, tau^2)`` prior (so an unknown-size shift is covered without
    choosing it in advance). ``sum_centered = sum_i (x_i - mu0)`` over the ``t`` observations.

    Closed form (a non-negative martingale under H0 with value 1 at t=0)::

        E_t = sqrt( s2 / (s2 + t*T2) ) * exp( T2 * S^2 / (2*s2*(s2 + t*T2)) )

    with ``s2 = sigma^2``, ``T2 = tau^2``, ``S = sum_centered``.
    """
    if t == 0:
        return 0.0
    s2 = float(sigma) ** 2
    tau2 = float(tau) ** 2
    denom = s2 + t * tau2
    log_e = 0.5 * np.log(s2 / denom) + (tau2 * sum_centered**2) / (2.0 * s2 * denom)
    return float(log_e)


def normal_mixture_eprocess(stream: Any, *, mu0: float, sigma: float, tau: float) -> np.ndarray:
    """Return the running e-values of the Robbins normal-mixture e-process over ``stream``.

    ``result[i]`` is ``E_{i+1}`` after seeing ``stream[:i+1]``. Element 0 corresponds to one
    observation; the process starts at ``E_0 = 1`` implicitly.
    """
    xs = np.asarray(list(stream), dtype=float)
    centered = np.cumsum(xs - float(mu0))
    ts = np.arange(1, len(xs) + 1)
    s2 = float(sigma) ** 2
    tau2 = float(tau) ** 2
    denom = s2 + ts * tau2
    log_e = 0.5 * np.log(s2 / denom) + (tau2 * centered**2) / (2.0 * s2 * denom)
    return np.exp(log_e)


@dataclass
class DriftReport:
    """Outcome of a drift scan: whether/when the e-process crossed ``1/alpha``."""

    detected: bool
    detection_time: int | None  # 1-indexed observation count at first crossing, else None
    alpha: float
    peak_e_value: float
    final_e_value: float
    guarantee: str = field(default="anytime-valid: false-alarm probability <= alpha under the null (Ville)")


class MeanShiftDetector:
    """An anytime-valid detector for a shift in a Gaussian stream's mean, built on the e-process.

    The null is ``mean == mu0`` with known ``sigma``; ``tau`` sets the scale of shifts the mixture
    is most sensitive to. Because it is an e-process, you may test after every observation and stop
    whenever it fires; the false-alarm probability over the whole run is still at most ``alpha``.
    """

    def __init__(self, *, mu0: float, sigma: float, tau: float = 1.0, alpha: float = 0.05) -> None:
        self.mu0 = float(mu0)
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.alpha = float(alpha)

    def scan(self, stream: Any) -> DriftReport:
        e_values = normal_mixture_eprocess(stream, mu0=self.mu0, sigma=self.sigma, tau=self.tau)
        if e_values.size == 0:
            return DriftReport(False, None, self.alpha, 1.0, 1.0)
        threshold = 1.0 / self.alpha
        crossings = np.flatnonzero(e_values >= threshold)
        detected = bool(crossings.size > 0)
        detection_time = int(crossings[0] + 1) if detected else None
        return DriftReport(
            detected=detected,
            detection_time=detection_time,
            alpha=self.alpha,
            peak_e_value=float(np.max(e_values)),
            final_e_value=float(e_values[-1]),
        )
