"""The claim-bearing honesty ledger: one registry that save and load both iterate.

Three findings in this release were the same defect: a claim-bearing field written at save time
and reconstructed by hand -- or not at all -- at load time (STAT-R1 dropped ``tau``;
STAT-RR13-1 reloaded an uncertified threshold as certified; STAT-RR13-2 reset the
selection-reuse count). Hand-maintaining the two sides independently is the mechanism, so this
module removes it: each solution shape declares its ledger ONCE, and both the save path and the
load path iterate the same declaration. A field cannot be persisted without being restored, and
every restored value passes the same validation the live object enforces.

The lifecycle contract test iterates these registries too, so a newly declared field is
round-trip-tested automatically -- the test cannot go stale against the registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LedgerField:
    """One claim-bearing field: its name, its missing-artifact default, and its validator."""

    name: str
    default: Any
    validate: Callable[[Any], Any]


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"ledger count must be a non-negative integer, got {value!r}")
    return int(value)


def _regime(*allowed: str) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if value not in allowed:
            raise ValueError(f"ledger regime is unrecognized: {value!r} (allowed: {sorted(allowed)})")
        return str(value)

    return check


# Classification (mixle.task.solve.Solution): the threshold's certifying regime and the
# selection-reuse count. "fresh-harvest" is NOT a classification regime -- its escalations are
# per-query threshold-selected and can never serve as calibration evidence (D-0155).
CLASSIFICATION_LEDGER: tuple[LedgerField, ...] = (
    LedgerField("selection_uses", 0, _non_negative_int),
    LedgerField(
        "calibration_evidence",
        "solve-split",
        _regime("solve-split", "fresh-evidence", "reused-after-adaptive-harvest"),
    ),
)

# Regression (mixle.task.regress.RegressionSolution): the same two claims; "fresh-harvest" IS a
# valid regime here because the regression gate is all-or-none, so a closed route harvests the
# raw serving stream (D-0155), and there is no reused regime -- regression improve() refuses to
# certify on reused rows outright.
REGRESSION_LEDGER: tuple[LedgerField, ...] = (
    LedgerField("selection_uses", 0, _non_negative_int),
    LedgerField(
        "calibration_evidence",
        "solve-split",
        _regime("solve-split", "fresh-harvest", "fresh-evidence"),
    ),
)


def write_ledger(obj: Any, fields: tuple[LedgerField, ...]) -> dict[str, Any]:
    """Serialize ``obj``'s ledger fields, validating each on the way OUT.

    Validating at save time means a corrupted live object cannot mint a well-formed artifact
    carrying an uninterpretable claim.
    """
    return {field.name: field.validate(getattr(obj, field.name)) for field in fields}


def read_ledger(meta: dict[str, Any], fields: tuple[LedgerField, ...]) -> dict[str, Any]:
    """Reconstruct ledger keyword arguments from artifact metadata, validating each on the way IN.

    Missing fields take the declared default (artifacts predating a field's introduction);
    present-but-invalid values are refused rather than coerced -- silent defaulting is exactly
    how an uncertified threshold reloaded as certified (STAT-RR13-1).
    """
    out: dict[str, Any] = {}
    for field in fields:
        if field.name in meta:
            out[field.name] = field.validate(meta[field.name])
        else:
            out[field.name] = field.default
    return out


__all__ = [
    "LedgerField",
    "CLASSIFICATION_LEDGER",
    "REGRESSION_LEDGER",
    "write_ledger",
    "read_ledger",
]
