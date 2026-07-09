"""Calibration reports as a post-condition of fitting.

A fit provides parameters; it does not by itself show whether predictive
probabilities are calibrated on held-out data. :func:`calibration_report`
returns the held-out mean log-density and, when the model exposes a predictive
CDF, a probability-integral-transform (PIT) calibration check.

Calibration is opt-in because it reserves held-out data. When requested through
the higher-level fitting surfaces, the resulting report is attached to the
model or artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CalibrationReport", "calibration_report"]


@dataclass
class CalibrationReport:
    """Whether a fitted model's uncertainty is calibrated on held-out data.

    ``pit_error`` is the total-variation distance of the PIT histogram from uniform (0 = perfectly
    calibrated). It has a finite-sample floor ~``sqrt(bins/n)`` even for a perfect model, so
    :meth:`is_calibrated` judges against that floor rather than a fixed constant.
    """

    n: int
    mean_log_density: float
    pit_error: float | None = None  # TV distance of PIT from uniform; None if the model has no CDF
    pit_histogram: dict[str, Any] | None = None
    bins: int = 10
    method: str = ""
    note: str = ""

    def noise_floor(self) -> float:
        """The PIT-error a perfectly calibrated model would show at this sample size (sampling noise)."""
        return float(np.sqrt(self.bins / max(self.n, 1)))

    def is_calibrated(self, tol: float | None = None) -> bool:
        """True when the PIT error is within tolerance. Default tol = 2.5x the finite-sample noise floor
        (so genuine miscalibration, not sampling noise, is what fails). Unknown -> False, conservatively."""
        if self.pit_error is None:
            return False
        threshold = 2.5 * self.noise_floor() if tol is None else float(tol)
        return self.pit_error <= threshold

    def as_dict(self) -> dict[str, Any]:
        """Return rounded calibration metrics as JSON-compatible data."""
        d = {
            "n": self.n,
            "mean_log_density": round(self.mean_log_density, 6),
            "pit_error": None if self.pit_error is None else round(self.pit_error, 6),
            "method": self.method,
            "note": self.note,
        }
        return d

    def __str__(self) -> str:
        pit = "n/a (no CDF)" if self.pit_error is None else f"{self.pit_error:.4f}"
        return (
            f"CalibrationReport(n={self.n}, mean_log_density={self.mean_log_density:.4f}, "
            f"pit_error={pit}, method={self.method or 'log-density'})"
        )


def _scalar_cdf(model: Any) -> Any:
    """A vectorized predictive CDF ``F(y)`` if the model exposes a scalar ``cdf``, else None."""
    fn = getattr(model, "cdf", None)
    if not callable(fn):
        return None

    def cdf(ys: np.ndarray) -> np.ndarray:
        return np.asarray([float(fn(float(v))) for v in np.asarray(ys, dtype=float).ravel()], dtype=float)

    return cdf


def calibration_report(model: Any, data: Any) -> CalibrationReport:
    """The calibration of ``model`` on held-out ``data`` (see module docstring).

    ``data`` should be data the model was not fitted on -- calibration measured on the training set is
    optimistic. Runs the PIT test when the model has a scalar predictive CDF; always reports the
    held-out mean log-density.
    """
    rows = list(data)
    enc = model.dist_to_encoder().seq_encode(rows)
    ll = np.asarray(model.seq_log_density(enc), dtype=np.float64)
    mean_ll = float(ll.mean()) if ll.size else float("nan")

    cdf = _scalar_cdf(model)
    if cdf is None:
        return CalibrationReport(
            n=len(rows),
            mean_log_density=mean_ll,
            pit_error=None,
            method="log-density",
            note="model has no scalar predictive CDF; PIT calibration not applicable (multivariate/latent)",
        )

    from mixle.inference.calibration import pit_calibration_error, pit_histogram, pit_values

    y = np.asarray([float(v) for v in rows], dtype=float)
    pit = pit_values(y, cdf)
    err = float(pit_calibration_error(pit))
    hist = pit_histogram(pit)
    report = CalibrationReport(
        n=len(rows),
        mean_log_density=mean_ll,
        pit_error=err,
        pit_histogram={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in hist.items()},
        bins=10,
        method="PIT",
    )
    report.note = (
        f"calibrated (PIT error {err:.3f} within the {report.noise_floor():.3f} noise floor)"
        if report.is_calibrated()
        else f"PIT deviates from uniform ({err:.3f} vs floor {report.noise_floor():.3f}) -- intervals are off"
    )
    return report
