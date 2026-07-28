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

from mixle.utils.optional_deps import HAS_PANDAS, pandas, require

__all__ = ["CalibrationReport", "calibration_report"]

#: Minimum expected observations per PIT bin before a uniformity verdict is meaningful. The usual
#: rule of thumb for a histogram goodness-of-fit statistic; below it the finite-sample noise floor
#: swamps the statistic's own range.
MIN_EXPECTED_PER_BIN = 5


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
    # Scoring validity is a required dimension of the verdict, not a detail of the mean: a NaN
    # log-density is not a score, whereas -inf is a legitimate "this row is impossible under the
    # model". A model whose per-row scores are unavailable cannot satisfy a calibration gate however
    # well its CDF happens to line up with the holdout.
    n_invalid_scores: int = 0

    def noise_floor(self) -> float:
        """The PIT-error a perfectly calibrated model would show at this sample size (sampling noise)."""
        return float(np.sqrt(self.bins / max(self.n, 1)))

    def scoring_is_valid(self) -> bool:
        """Whether every held-out row got a usable (finite or explicitly impossible) log-density."""
        return self.n_invalid_scores == 0

    def max_pit_error(self) -> float:
        """The largest PIT error this histogram statistic can attain: all mass in one bin.

        ``sum_b |freq_b - 1/bins|`` peaks at ``(1 - 1/bins) + (bins-1)/bins == 2(1 - 1/bins)`` -- 1.8
        for the default ten bins. A tolerance at or above this bound accepts *every* possible
        histogram, so the gate is not a test at all.
        """
        return 2.0 * (1.0 - 1.0 / self.bins)

    def has_enough_data(self) -> bool:
        """Whether the holdout can support a per-bin uniformity judgement at all.

        The histogram statistic needs a usable expected count per bin; below ``MIN_EXPECTED_PER_BIN``
        the noise floor ``sqrt(bins/n)`` grows until ``2.5x`` it exceeds :meth:`max_pit_error` and the
        acceptance region becomes vacuous -- at ten bins and a single observation the threshold was
        7.91 against a statistic that cannot exceed 1.8, so even maximally clumped PIT values were
        reported calibrated.
        """
        return self.n >= MIN_EXPECTED_PER_BIN * self.bins

    def is_calibrated(self, tol: float | None = None) -> bool:
        """True when the PIT error is within tolerance AND the model actually scored the holdout.

        Default tol = 2.5x the finite-sample noise floor (so genuine miscalibration, not sampling
        noise, is what fails). Unknown -> False, conservatively. An invalid predictive score on any
        row -> False: the PIT histogram alone cannot certify a model whose density is unavailable.
        Too little held-out data for :meth:`has_enough_data` -> False: absence of evidence is not
        evidence of calibration.

        Raises:
            ValueError: if ``tol`` is not finite and nonnegative, or is at/above :meth:`max_pit_error`
                (a tolerance that no histogram can exceed would approve any model).
        """
        if tol is not None:
            tol = float(tol)
            if not np.isfinite(tol) or tol < 0.0:
                raise ValueError(f"calibration tol must be finite and nonnegative, got {tol!r}")
            if tol >= self.max_pit_error():
                raise ValueError(
                    f"calibration tol {tol!r} is at or above the largest attainable PIT error "
                    f"{self.max_pit_error()!r} for {self.bins} bins, so it would accept every possible "
                    "model; pass a tolerance inside the statistic's range."
                )
        if self.pit_error is None or not self.scoring_is_valid() or not self.has_enough_data():
            return False
        threshold = 2.5 * self.noise_floor() if tol is None else tol
        # With has_enough_data() satisfied the default threshold is at most 2.5*sqrt(1/5) ~= 1.118,
        # strictly inside the statistic's 1.8 range, so the default gate can never be vacuous either.
        return self.pit_error <= threshold

    def as_dict(self) -> dict[str, Any]:
        """Return rounded calibration metrics as JSON-compatible data."""
        d = {
            "n": self.n,
            "mean_log_density": round(self.mean_log_density, 6),
            "pit_error": None if self.pit_error is None else round(self.pit_error, 6),
            "n_invalid_scores": self.n_invalid_scores,
            "method": self.method,
            "note": self.note,
        }
        return d

    def to_dataframe(self) -> Any:
        """Return this report as a ``pandas.DataFrame``.

        When a PIT histogram was computed (``method == "PIT"``), returns one row per bin --
        ``bin_left``, ``bin_right``, ``count``, ``density``, ``uniform`` -- the natural columnar shape
        of a calibration report (a table you'd plot). Otherwise (no scalar predictive CDF, so no PIT
        histogram) returns a single summary row with the same unrounded fields as :meth:`as_dict`.
        Requires the ``pandas`` extra (``pip install mixle[pandas]``).
        """
        if not HAS_PANDAS:
            require("pandas", "pandas")
        if self.pit_histogram is not None:
            edges = np.asarray(self.pit_histogram["edges"], dtype=float)
            return pandas.DataFrame(
                {
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "count": np.asarray(self.pit_histogram["counts"]),
                    "density": np.asarray(self.pit_histogram["density"], dtype=float),
                    "uniform": np.asarray(self.pit_histogram["uniform"], dtype=float),
                }
            )
        return pandas.DataFrame(
            [
                {
                    "n": self.n,
                    "mean_log_density": self.mean_log_density,
                    "pit_error": self.pit_error,
                    "n_invalid_scores": self.n_invalid_scores,
                    "method": self.method,
                    "note": self.note,
                }
            ]
        )

    def to_parquet(self, path: Any, **kwargs: Any) -> None:
        """Write this report to a Parquet file; see :meth:`to_dataframe`.

        ``kwargs`` forward to ``DataFrame.to_parquet`` (e.g. ``engine=``, ``compression=``). Needs a
        Parquet engine in addition to pandas -- ``pip install mixle[arrow]`` (pyarrow) or fastparquet.
        """
        self.to_dataframe().to_parquet(path, **kwargs)

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


def _is_lattice_law(cdf: Any, y: np.ndarray) -> bool:
    """Whether the predictive CDF is a step function on the integers over the observed values.

    ``F(Y)`` is uniform only for a CONTINUOUS predictive law. When the law has atoms, ``F(Y)`` can
    only ever land on the finitely many values the atoms produce, so its histogram is clumped by
    construction and the ordinary PIT reports a large "error" for an exactly correct model.

    Detected from the CDF itself rather than from the model's type, so it works for any model that
    exposes ``cdf``: the values must all be integers, and ``F`` must be FLAT strictly between
    consecutive integers (``F(y-0.5) == F(y-0.25)``), which is exactly what a lattice CDF does and
    what a continuous one does not.
    """
    if y.size == 0 or not np.all(np.isfinite(y)) or not np.allclose(y, np.round(y)):
        return False
    return bool(np.allclose(cdf(y - 0.5), cdf(y - 0.25), atol=1e-12, rtol=0.0))


def _pit_for_law(cdf: Any, y: np.ndarray, seed: int) -> tuple[np.ndarray, str, str]:
    """PIT values appropriate to the predictive law: randomized on a lattice, ordinary otherwise.

    The randomized PIT ``u = F(y-) + V (F(y) - F(y-))``, ``V ~ U(0,1)``, is exactly Uniform(0,1)
    under a correctly specified DISCRETE law -- it spreads each atom's probability mass evenly across
    the interval the CDF jumps over. For an integer lattice the left limit ``F(y-)`` is ``F(y-1)``
    exactly. The randomness is seeded and reported so the verdict is reproducible.
    """
    from mixle.inference.calibration import pit_values

    if not _is_lattice_law(cdf, y):
        return pit_values(y, cdf), "PIT", ""
    lo, hi = cdf(y - 1.0), cdf(y)
    v = np.random.RandomState(seed).uniform(size=y.shape)
    u = np.clip(lo + v * np.maximum(hi - lo, 0.0), 0.0, 1.0)
    return pit_values(y, u), "randomized-PIT", f" [randomized PIT for a discrete predictive law, seed={seed}]"


def _holdout_log_densities(model: Any, rows: list) -> tuple[np.ndarray, int]:
    """One log-density per held-out row, plus how many of them are unusable.

    The predictive contract is exactly one score per row. A model returning a single score for the
    whole holdout, or a batch of the wrong width, is a contract violation and is rejected rather than
    broadcast into a plausible-looking mean. NaN scores are counted as invalid; ``-inf`` is NOT --
    it is the model's legitimate statement that the row is impossible under it.
    """
    ll = np.atleast_1d(np.asarray(model.seq_log_density(model.dist_to_encoder().seq_encode(rows)), dtype=np.float64))
    if ll.shape != (len(rows),):
        raise ValueError(
            f"model returned {ll.shape} log-densities for {len(rows)} held-out records; a calibration "
            "check needs exactly one predictive score per row"
        )
    return ll, int(np.isnan(ll).sum())


def calibration_report(model: Any, data: Any, *, seed: int = 0) -> CalibrationReport:
    """The calibration of ``model`` on held-out ``data`` (see module docstring).

    ``data`` should be data the model was not fitted on -- calibration measured on the training set is
    optimistic. Runs the PIT test when the model has a scalar predictive CDF; always reports the
    held-out mean log-density.

    A predictive law with ATOMS is handled with the randomized PIT rather than the continuous one:
    ``F(Y)`` is uniform only for a continuous law, so routing a discrete model through the ordinary
    PIT reports a large error for an exactly correct model. ``seed`` fixes that randomization (it is
    recorded in ``note``) and is unused for a continuous law, whose method is unchanged.

    Scoring validity is checked first and carried in the report. The verdict used to consult only the
    PIT histogram, so a model whose CDF was exact but whose per-row log-densities were all NaN --
    or which returned one score for a thousand rows -- passed the gate as ``"calibrated"``. Exactly
    one usable score per row is now required for a calibrated verdict; a cardinality mismatch raises
    and NaN scores are counted into ``n_invalid_scores``, which forces :meth:`is_calibrated` to False.

    Raises:
        ValueError: if the model's batch scoring does not return exactly one log-density per row.
    """
    rows = list(data)
    ll, n_invalid = _holdout_log_densities(model, rows)
    mean_ll = float(ll.mean()) if ll.size else float("nan")

    cdf = _scalar_cdf(model)
    if cdf is None:
        return CalibrationReport(
            n=len(rows),
            mean_log_density=mean_ll,
            pit_error=None,
            method="log-density",
            n_invalid_scores=n_invalid,
            note="model has no scalar predictive CDF; PIT calibration not applicable (multivariate/latent)",
        )

    from mixle.inference.calibration import pit_calibration_error, pit_histogram

    y = np.asarray([float(v) for v in rows], dtype=float)
    pit, method, method_note = _pit_for_law(cdf, y, seed)
    err = float(pit_calibration_error(pit))
    hist = pit_histogram(pit)
    report = CalibrationReport(
        n=len(rows),
        mean_log_density=mean_ll,
        pit_error=err,
        pit_histogram={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in hist.items()},
        bins=10,
        method=method,
        n_invalid_scores=n_invalid,
    )
    if not report.scoring_is_valid():
        report.note = (
            f"predictive scoring is unavailable on {n_invalid} of {len(rows)} held-out rows "
            f"(PIT error {err:.3f}) -- not calibrated: a uniform PIT cannot certify a model whose "
            "own log-density is missing"
        )
    elif not report.has_enough_data():
        report.note = (
            f"only {len(rows)} held-out row(s) for {report.bins} PIT bins "
            f"(need {MIN_EXPECTED_PER_BIN * report.bins}) -- not calibrated: too little data to judge "
            "uniformity, so no verdict is claimed either way"
        )
    else:
        report.note = (
            f"calibrated (PIT error {err:.3f} within the {report.noise_floor():.3f} noise floor)"
            if report.is_calibrated()
            else f"PIT deviates from uniform ({err:.3f} vs floor {report.noise_floor():.3f}) -- intervals are off"
        )
    report.note += method_note
    return report
