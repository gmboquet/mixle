"""Exchangeability diagnostics for fitted and synthetic-data workflows.

Fitting one distribution to a dataset, or synthesizing "more rows like these",
assumes that row order does not carry information. When the data has a trend or
a regime shift, that assumption is false and a pooled marginal model can
misrepresent the process.

:func:`exchangeability_check` tests the assumption with numeric probes: a
permutation test for rank correlation between value and row position, plus a
first-half/second-half location-shift test. The aggregate label is one of:

* ``exchangeable``: no order signal found at the tested level;
* ``trend``: value co-moves with position;
* ``shift``: the halves differ in location.

:func:`mixle.inference.create` and :func:`mixle.inference.synthesize` record
the verdict in provenance so downstream consumers can see when pooling deserves
review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExchangeabilityReport:
    """The verdict per numeric field, plus the aggregate label the preconditions record."""

    label: str  # 'exchangeable' | 'trend' | 'shift'
    fields: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exchangeable(self) -> bool:
        """Return ``True`` when no tested order signal was detected."""
        return self.label == "exchangeable"

    def as_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dictionary."""
        return {"label": self.label, "exchangeable": self.exchangeable, "fields": self.fields}


def _numeric_columns(rows: list[Any]) -> dict[str, np.ndarray]:
    """Extract the numeric field(s): scalars -> one column; tuples/lists -> each numeric position."""
    first = rows[0]
    if isinstance(first, (int, float, np.integer, np.floating)):
        return {"value": np.asarray([float(r) for r in rows], dtype=float)}
    cols: dict[str, np.ndarray] = {}
    if isinstance(first, (tuple, list)):
        for j, v in enumerate(first):
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                try:
                    cols[f"field[{j}]"] = np.asarray([float(r[j]) for r in rows], dtype=float)
                except (TypeError, ValueError, IndexError):
                    continue
    return cols


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    sx, sy = rx.std(), ry.std()
    if sx <= 0 or sy <= 0:
        return 0.0
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def _perm_pvalue(x: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    """Permutation p-value of |rank-corr(position, value)| -- exact null: order carries no signal."""
    pos = np.arange(len(x), dtype=float)
    observed = abs(_rank_corr(pos, x))
    rng = np.random.RandomState(seed)
    hits = 1  # add-one: the observed permutation counts (valid, slightly conservative)
    for _ in range(n_perm):
        if abs(_rank_corr(pos, rng.permutation(x))) >= observed:
            hits += 1
    return observed, hits / (n_perm + 1)


def _halves_shift_pvalue(x: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    """Permutation p-value of |mean(first half) - mean(second half)| -- a regime-change probe."""
    n = len(x)
    half = n // 2
    observed = abs(float(np.mean(x[:half]) - np.mean(x[half:])))
    rng = np.random.RandomState(seed)
    hits = 1
    for _ in range(n_perm):
        p = rng.permutation(x)
        if abs(float(np.mean(p[:half]) - np.mean(p[half:]))) >= observed:
            hits += 1
    return observed, hits / (n_perm + 1)


def exchangeability_check(data: Any, *, alpha: float = 0.01, n_perm: int = 200, seed: int = 0) -> ExchangeabilityReport:
    """Test whether row ORDER carries information (see module docstring). Small n -> exchangeable (no power).

    ``alpha`` is deliberately strict (0.01): the check should flag clear violations, not manufacture
    warnings from noise. Non-numeric-only data passes vacuously (order tests need a numeric surface)."""
    rows = list(data)
    if len(rows) < 20:
        return ExchangeabilityReport(label="exchangeable", fields=[{"note": "n < 20: no power to test"}])
    cols = _numeric_columns(rows)
    if not cols:
        return ExchangeabilityReport(label="exchangeable", fields=[{"note": "no numeric fields to test"}])

    fields: list[dict[str, Any]] = []
    worst = "exchangeable"
    for name, x in cols.items():
        tr_stat, tr_p = _perm_pvalue(x, n_perm=n_perm, seed=seed)
        sh_stat, sh_p = _halves_shift_pvalue(x, n_perm=n_perm, seed=seed + 1)
        verdict = "exchangeable"
        if tr_p < alpha or sh_p < alpha:
            # disambiguate: a genuine trend persists WITHIN each half; a step change does not.
            half = len(x) // 2
            _s1, p1 = _perm_pvalue(x[:half], n_perm=n_perm, seed=seed + 2)
            _s2, p2 = _perm_pvalue(x[half:], n_perm=n_perm, seed=seed + 3)
            within_trend = p1 < alpha or p2 < alpha
            verdict = "trend" if within_trend else ("shift" if sh_p < alpha else "trend")
        fields.append(
            {
                "field": name,
                "verdict": verdict,
                "trend_rank_corr": round(tr_stat, 4),
                "trend_p": round(tr_p, 4),
                "shift_p": round(sh_p, 4),
            }
        )
        if verdict == "trend" or (verdict == "shift" and worst == "exchangeable"):
            worst = verdict
    return ExchangeabilityReport(label=worst, fields=fields)
