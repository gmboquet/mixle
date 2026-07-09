"""Certified model creation from data.

``create`` fits a model through the ordinary Mixle inference machinery and
returns a :class:`CreatedModel` artifact rather than a bare distribution. The
artifact contains the fitted model, an estimation certificate, optional
calibration, optional uncertainty quantification, and provenance.

The ``budget`` and ``device`` arguments are recorded as constraints and bias the
automatic structure search toward smaller, independence-first models. They do
not replace task-specific compression or edge-distillation workflows; they make
the creation boundary explicit when a caller already knows the deployment
envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CreatedModel:
    """A certified model artifact: the fitted model plus its guarantees and provenance."""

    model: Any
    certificate: Any
    strategy: str
    calibration: Any | None = None
    uq: Any | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def guarantee(self) -> Any:
        """The aggregate estimation guarantee (MIN over blocks) -- the artifact's summary claim."""
        return self.certificate.guarantee

    def why(self) -> str:
        """One line summarizing how the artifact was estimated."""
        return self.certificate.why_not_adam() if hasattr(self.certificate, "why_not_adam") else ""

    def is_calibrated(self) -> bool | None:
        """Whether the calibration holdout judged the model calibrated (None if not checked)."""
        if self.calibration is None:
            return None
        return bool(self.calibration.is_calibrated())


def create(
    data: Any,
    *,
    budget: Any | None = None,
    device: Any | None = None,
    calibrate: float | None = None,
    quantify_uq: bool = False,
    max_its: int = 25,
    seed: int = 0,
) -> CreatedModel:
    """Create a certified model from ``data`` (see module docstring).

    ``data`` is a list of records/scalars. ``calibrate`` (a fraction in (0,1)) reserves a holdout for a
    PIT calibration check. ``quantify_uq=True`` attaches an auto-selected UQ handle. ``budget`` /
    ``device`` (any object; recorded verbatim) constrain the fit toward a smaller model. Returns a
    :class:`CreatedModel` bundling the fit, its certificate, and the requested post-conditions.
    """
    import numpy as np

    from mixle.inference.calibrate_fit import calibration_report
    from mixle.inference.estimation import optimize
    from mixle.inference.planning import certify

    rows = list(data)
    constrained = budget is not None or device is not None
    # A budget/device envelope caps model complexity: keep fields independent (structure search off) so
    # the artifact stays small. Task-specific compression still belongs in
    # mixle.task.edge when labels and a hard footprint are supplied.
    structure = "off" if constrained else "auto"

    fit_rows, holdout = rows, None
    if calibrate is not None:
        frac = float(calibrate)
        if not 0.0 < frac < 1.0:
            raise ValueError("calibrate must be a fraction in (0, 1)")
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rows))
        n_hold = max(1, int(round(frac * len(rows))))
        hold_idx, fit_idx = order[:n_hold], order[n_hold:]
        fit_rows = [rows[i] for i in fit_idx]
        holdout = [rows[i] for i in hold_idx]

    model = optimize(fit_rows, out=None, max_its=max_its, structure=structure, rng=np.random.RandomState(seed))
    cert = certify(model)

    calibration = None
    if holdout is not None:
        try:
            calibration = calibration_report(model, holdout)
        except Exception:  # noqa: BLE001 - calibration is a best-effort post-condition, never fatal
            calibration = None

    uq_handle = None
    if quantify_uq:
        try:
            from mixle.inference.uq import uq as _uq

            uq_handle = _uq(model, fit_rows)
        except Exception:  # noqa: BLE001 - UQ is optional; absence is explicit, not a crash
            uq_handle = None

    # M2 precondition: pooling rows into one model assumes exchangeability -- test it, record the
    # verdict next to the artifact (a warning, never a silent refusal).
    try:
        from mixle.data.exchangeability import exchangeability_check

        exch = exchangeability_check(rows, seed=seed).as_dict()
    except Exception:  # noqa: BLE001 - the precondition check must never break a fit
        exch = None

    return CreatedModel(
        model=model,
        certificate=cert,
        strategy="edge-constrained" if constrained else "structured",
        calibration=calibration,
        uq=uq_handle,
        provenance={
            "n": len(rows),
            "n_fit": len(fit_rows),
            "structure": structure,
            "budget": repr(budget) if budget is not None else None,
            "device": repr(device) if device is not None else None,
            "seed": seed,
            "exchangeability": exch,
        },
    )
