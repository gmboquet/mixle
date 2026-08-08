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
    """One claim-bearing field: its name and its validator.

    There is deliberately NO missing-artifact default: a live object's initialization default
    ("solve-split", zero uses) describes a freshly solved model, and reusing it as the meaning
    of an ABSENT field would present an artifact whose calibration history is unknown as clean,
    certified evidence (STAT-RR14-1). An artifact that does not carry its ledger is refused.
    """

    name: str
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


# Classification (mixle.task.solve.Solution): the threshold's certifying regime, the
# selection-reuse count, and the answered-slice measurement counts. "fresh-harvest" is NOT a
# classification regime -- its escalations are per-query threshold-selected and can never serve
# as calibration evidence (D-0155). The answered-slice counts are claim-bearing measurements
# (STAT-RR16-2): dropping them on reload would present an artifact with an unknown measurement
# as one that measured nothing, so they ride the same refuse-on-missing registry.
CLASSIFICATION_LEDGER: tuple[LedgerField, ...] = (
    LedgerField("selection_uses", _non_negative_int),
    LedgerField(
        "calibration_evidence",
        _regime("solve-split", "fresh-evidence", "reused-after-adaptive-harvest"),
    ),
    LedgerField("sel_rows", _non_negative_int),
    LedgerField("answered_sel_n", _non_negative_int),
    LedgerField("answered_sel_correct", _non_negative_int),
)

# Regression (mixle.task.regress.RegressionSolution): the same two claims; "fresh-harvest" IS a
# valid regime here because the regression gate is all-or-none, so a closed route harvests the
# raw serving stream (D-0155), and there is no reused regime -- regression improve() refuses to
# certify on reused rows outright.
REGRESSION_LEDGER: tuple[LedgerField, ...] = (
    LedgerField("selection_uses", _non_negative_int),
    LedgerField(
        "calibration_evidence",
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

    Present-but-invalid values are refused rather than coerced (silent coercion is how an
    uncertified threshold reloaded as certified, STAT-RR13-1) -- and MISSING fields are refused
    outright: an artifact without its ledger has an unknown calibration history, and loading it
    under a fresh-solve default presented exactly such an artifact -- a threshold produced after
    adaptive reuse, selection evidence spent four times -- as clean, single-use, certified
    evidence (STAT-RR14-1). The 0.8.0 artifact format REQUIRES the ledger; there are no
    published pre-ledger artifacts, and an unpublished one is re-solved, not reinterpreted.
    """
    out: dict[str, Any] = {}
    for field in fields:
        if field.name not in meta:
            raise ValueError(
                f"artifact is missing the claim-bearing ledger field {field.name!r}: its "
                "calibration history is unknown and cannot present as certified evidence -- "
                "re-solve to produce a current artifact"
            )
        out[field.name] = field.validate(meta[field.name])
    return out


def _clopper_pearson_interval(successes: int, n: int, level: float) -> tuple[float, float]:
    """Exact Clopper-Pearson two-sided interval for a binomial proportion.

    Lives here (not in a shape module) because every shape's answered-slice measurement quotes
    it and ``regress`` already imports from ``solve`` -- a shape-to-shape import would cycle.
    """
    from scipy.stats import beta as _beta

    if n <= 0:
        raise ValueError("interval needs a positive denominator")
    tail = (1.0 - level) / 2.0
    lower = 0.0 if successes == 0 else float(_beta.ppf(tail, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(_beta.ppf(1.0 - tail, successes + 1, n - successes))
    return lower, upper


def conformal_scope(statement: str) -> dict[str, Any]:
    """Machine-readable scope block for any report that carries a conformal coverage label.

    Every report surface that names a coverage contract attaches this block so the claim's
    scope travels WITH the claim (STAT-RR15-2: bare contract labels like "joint_exact_set"
    advertised coverage without stating that split conformal is marginal under exchangeability,
    is NOT an accuracy guarantee conditional on answering locally, and is voided by
    distribution shift). One shared constructor, not per-report prose, so the scope statement
    cannot drift between shapes.
    """
    return {
        "statement": statement,
        "marginal": True,
        "assumptions": ["exchangeability of calibration rows and incoming queries"],
        "conditional_on_answering_locally": False,
        "conditional_accuracy_guarantee": False,
        "voided_by": "distribution shift / non-exchangeability",
    }


__all__ = [
    "LedgerField",
    "CLASSIFICATION_LEDGER",
    "REGRESSION_LEDGER",
    "write_ledger",
    "read_ledger",
    "conformal_scope",
]
