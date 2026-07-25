"""Production model monitoring: watch for drift, retrain, and swap -- with provenance and a DOE hook.

A :class:`Monitor` wraps a fitted model + its estimator + a reference (training) sample. Feed it each
production batch via :meth:`check` (drift report only) or :meth:`update` (check, and on drift retrain a
fresh model -- recording a new provenance header -- and swap it in). Every action is appended to a history
for audit. :meth:`suggest_samples` ties in ``mixle.doe`` so a drift signal (or any model objective) can
drive *where to collect new data* -- space-filling by default, or active-learning against a model objective.
"""

from __future__ import annotations

import math
import time
from numbers import Real
from typing import Any

from mixle.inference.production.drift import DriftReport, detect_drift
from mixle.inference.production.provenance import Header, fit_with_provenance


class Monitor:
    """Stateful monitor for one deployed model: drift detection + retrain-and-swap + DOE-driven sampling."""

    def __init__(
        self,
        model: Any,
        estimator: Any,
        reference: Any,
        *,
        psi_threshold: float = 0.25,
        ks_threshold: float = 0.2,
        loglik_shift_threshold: float = -0.5,
    ) -> None:
        def threshold(value: Any, name: str, *, unit: bool = False) -> float:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite real number")
            value = float(value)
            if not math.isfinite(value) or (unit and not 0.0 <= value <= 1.0):
                domain = " in [0, 1]" if unit else ""
                raise ValueError(f"{name} must be a finite real number{domain}")
            return value

        self.model = model
        self.estimator = estimator
        self.reference = self._materialize(reference, "reference")
        self.thresholds = {
            "psi_threshold": threshold(psi_threshold, "psi_threshold"),
            "ks_threshold": threshold(ks_threshold, "ks_threshold", unit=True),
            "loglik_shift_threshold": threshold(loglik_shift_threshold, "loglik_shift_threshold"),
        }
        self.history: list[dict] = []
        self.header: Header | None = getattr(model, "header", None)

    @staticmethod
    def _materialize(data: Any, name: str) -> list:
        records = getattr(data, "records", None)
        source = records() if callable(records) else data
        batch = list(source)
        if not batch:
            raise ValueError(f"{name} data must contain at least one observation")
        return batch

    def check(self, current: Any) -> DriftReport:
        """Drift report of ``current`` (production) data against the reference under the current model."""
        batch = self._materialize(current, "current")
        return detect_drift(self.model, self.reference, batch, **self.thresholds)

    def update(self, current: Any, *, retrain: bool = True, combine_reference: bool = True, **fit_kw: Any) -> dict:
        """Check drift on ``current`` and, if drift is flagged and ``retrain``, fit a fresh model (with a
        new provenance header) and swap it in. Returns ``{report, action, model, header}`` and appends to
        :attr:`history`. ``combine_reference`` retrains on reference + current (else current only)."""
        if not isinstance(retrain, bool) or not isinstance(combine_reference, bool):
            raise TypeError("retrain and combine_reference must be booleans")
        batch = self._materialize(current, "current")
        report = detect_drift(self.model, self.reference, batch, **self.thresholds)
        action = "none"
        header = self.header
        if report.drift and retrain:
            train = (self.reference + batch) if combine_reference else batch
            new_model, header = fit_with_provenance(train, self.estimator, **fit_kw)
            self.model = new_model
            self.reference = train
            self.header = header
            action = "retrained"
        entry = {
            "time": time.time(),
            "n_current": len(batch),
            "drift": report.drift,
            "score": report.score,
            "action": action,
        }
        self.history.append(entry)
        return {
            "report": report,
            "action": action,
            "model": self.model,
            "header": header,
            "processed_count": len(batch),
        }

    def suggest_samples(
        self, bounds: Any, n: int = 10, *, method: str = "lhs", objective: Any = None, seed: int | None = None
    ) -> Any:
        """Use ``mixle.doe`` to propose where to collect new data (e.g. after drift, or to meet an
        objective). ``method='lhs'``/``'sobol'`` gives a space-filling batch over ``bounds`` (list of
        ``(lo, hi)``); pass an ``objective(x)->float`` to switch to active learning (ALC/ALM) that targets
        where the model is most informative."""
        from mixle import doe as _doe

        if objective is not None:
            return _doe.active_learning_design(objective, bounds, max_evals=n, seed=seed)
        if method == "sobol":
            return _doe.sobol_design(bounds, n, seed)
        return _doe.latin_hypercube(bounds, n, seed)
