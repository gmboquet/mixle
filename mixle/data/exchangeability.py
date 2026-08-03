"""Exchangeability diagnostics for fitted and synthetic-data workflows.

Fitting one distribution to a dataset, or synthesizing "more rows like these",
assumes that row order does not carry information. When the data has a trend or
a regime shift, that assumption is false and a pooled marginal model can
misrepresent the process.

:func:`exchangeability_check` tests the assumption with numeric probes: a
permutation test for rank correlation between value and row position, plus a
first-half/second-half location-shift test. The aggregate label is one of:

* ``exchangeable``: the probes ran and found no order signal at the tested level;
* ``trend``: value co-moves with position;
* ``shift``: the halves differ in location;
* ``inconclusive``: no probe could run at all -- too few rows, no numeric field in the
  record shape, or every numeric field was non-finite (NaN/Inf). This is deliberately
  distinct from ``exchangeable``: it means the assumption was never tested, not that it
  was tested and held.

:func:`mixle.inference.create` and :func:`mixle.inference.synthesize` record
the verdict in provenance so downstream consumers can see when pooling deserves
review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import rankdata


@dataclass
class ExchangeabilityReport:
    """The verdict per numeric field, plus the aggregate label the preconditions record."""

    label: str  # 'exchangeable' | 'trend' | 'shift' | 'inconclusive'
    fields: list[dict[str, Any]] = field(default_factory=list)
    n_examined: int = 0
    max_records: int | None = None
    bounded: bool = False
    #: How the multiplicity across the primary test family was corrected, and how large that family
    #: was. Recorded because the label is only interpretable against it (MXR-080-1887).
    multiplicity: dict[str, Any] = field(default_factory=dict)

    _LABELS = ("exchangeable", "trend", "shift", "inconclusive")

    def __post_init__(self) -> None:
        """Refuse a report whose label is outside the closed vocabulary it is read against.

        The label was an unchecked string, so a directly-constructed report could carry one nothing
        compares equal to -- making ``exchangeable`` silently False for a reason no reader could see
        (MXR-080-1887).
        """
        if self.label not in self._LABELS:
            raise ValueError(f"ExchangeabilityReport label must be one of {self._LABELS}, got {self.label!r}")
        if not isinstance(self.fields, list):
            raise TypeError(f"ExchangeabilityReport fields must be a list, got {type(self.fields).__name__}")

    @property
    def exchangeable(self) -> bool:
        """Return ``True`` only when the probes ran and did not reject exchangeability.

        This is FAILURE TO REJECT, not certification (MXR-080-1887). The probes test for a monotone
        trend and for a between-halves shift; data can violate exchangeability in ways neither probe
        looks for, and a real violation can go undetected for want of power. ``True`` means "these
        tests, at this corrected level, found no order signal" -- it is not evidence the assumption
        holds. :attr:`multiplicity` records the correction the verdict was decided under, without
        which the level is not interpretable.

        ``False`` both for a detected ``trend``/``shift`` AND for ``inconclusive`` (nothing could be
        tested) -- a caller that only inspects this boolean can never mistake "we could not check"
        for "we checked and found no signal".
        """
        return self.label == "exchangeable"

    def as_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dictionary."""
        return {
            "label": self.label,
            "exchangeable": self.exchangeable,
            "fields": self.fields,
            "n_examined": self.n_examined,
            "max_records": self.max_records,
            "bounded": self.bounded,
            "multiplicity": dict(self.multiplicity),
        }


def _numeric_columns(rows: list[Any]) -> dict[str, np.ndarray]:
    """Extract the numeric field(s): scalars -> one column; tuples/lists -> each numeric position;
    dict/mapping rows -> each numeric-valued key (mirrors the tuple/list extraction below)."""
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
    elif isinstance(first, dict):
        for k, v in first.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                try:
                    cols[str(k)] = np.asarray([float(r[k]) for r in rows], dtype=float)
                except (TypeError, ValueError, KeyError):
                    continue
    return cols


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, using MID-RANKS for tied values.

    ``argsort(argsort(x))`` was used here (MXR-080-1602). That assigns arbitrary DISTINCT ranks to
    tied values -- whichever order argsort happened to break the tie in -- so it is not the Spearman
    statistic on tied data, and because the same transform is reapplied to every permuted sample it
    also changes the permutation null the statistic is judged against. On the 20-value tied sequence
    ``[1,1,1,1,1,3,2,0,0,0,0,2,1,2,3,3,3,3,3,3]`` it reported a trend correlation of 0.5233 where
    the mid-rank Spearman value is 0.6001, and with 999 permutations at seed 19 that understated
    trend evidence came back as ``p=0.042`` -- passing an ``alpha=0.01`` screen -- against ``p=0.008``
    for a correct mid-rank permutation. ``rankdata`` is the tie-correct transform.
    """
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
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


def exchangeability_check(
    data: Any, *, alpha: float = 0.01, n_perm: int = 200, seed: int = 0, max_records: int = 10_000
) -> ExchangeabilityReport:
    """Test whether row ORDER carries information (see module docstring).

    ``alpha`` is deliberately strict (0.01): the check should flag clear violations, not manufacture
    warnings from noise. A dataset too small to have testing power (n < 20), with no numeric field in
    its record shape (dict/tuple/list/scalar rows are all supported), or whose only numeric field(s)
    are entirely non-finite (NaN/Inf) is reported ``inconclusive`` -- untested, never a silent,
    vacuous "exchangeable". A field containing SOME non-finite values is reported ``invalid`` and
    excluded from testing; any other, clean numeric field in the same record shape is still tested
    normally and can still produce a real verdict."""
    import itertools

    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise ValueError(f"alpha must be a finite number in (0, 1), got {alpha!r}")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be a finite number in (0, 1), got {alpha!r}")
    if isinstance(n_perm, (bool, np.bool_)) or not isinstance(n_perm, (int, np.integer)) or n_perm < 1:
        raise ValueError(f"n_perm must be a positive integer, got {n_perm!r}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an integer in [0, 2**32), got {seed!r}")
    if isinstance(max_records, (bool, np.bool_)) or not isinstance(max_records, (int, np.integer)) or max_records < 20:
        raise ValueError(f"max_records must be an integer >= 20, got {max_records!r}")
    source = data.records() if hasattr(data, "records") and callable(data.records) else data
    rows = list(itertools.islice(source, int(max_records)))
    receipt = {"n_examined": len(rows), "max_records": int(max_records), "bounded": len(rows) == max_records}
    if len(rows) < 20:
        return ExchangeabilityReport(label="inconclusive", fields=[{"note": "n < 20: no power to test"}], **receipt)
    cols = _numeric_columns(rows)
    if not cols:
        return ExchangeabilityReport(label="inconclusive", fields=[{"note": "no numeric fields to test"}], **receipt)

    # MULTIPLICITY (MXR-080-1887). Two primary tests run per numeric field, and the old code compared
    # each raw p-value to `alpha` independently -- so with k fields, 2k chances to reject at level
    # alpha each, and ANY single one flipped the whole report. The advertised alpha was therefore not
    # the report's error rate; over enough columns of genuinely exchangeable data the report reads
    # "trend" as a matter of course. Every primary p-value is corrected together, once, as one family
    # -- the discipline mixle.evolve.population already applies to simultaneous challengers.
    #
    # The within-half probes below are deliberately NOT in this family. They run only after a
    # rejection and choose between the "trend" and "shift" LABELS; they cannot themselves cause a
    # rejection, so including them would penalize the family for tests that decide only wording.
    primary: list[tuple[str, np.ndarray, float, float, float, float]] = []
    fields: list[dict[str, Any]] = []
    for name, x in cols.items():
        finite = np.isfinite(x)
        if not finite.all():
            n_bad = int((~finite).sum())
            fields.append(
                {
                    "field": name,
                    "verdict": "invalid",
                    "note": "%d of %d value(s) are non-finite (NaN/Inf): cannot test" % (n_bad, len(x)),
                }
            )
            continue
        tr_stat, tr_p = _perm_pvalue(x, n_perm=n_perm, seed=seed)
        sh_stat, sh_p = _halves_shift_pvalue(x, n_perm=n_perm, seed=seed + 1)
        primary.append((name, x, tr_stat, tr_p, sh_stat, sh_p))

    tested_any = bool(primary)
    correction = {"method": "holm", "family_size": 2 * len(primary), "alpha": alpha}
    if primary:
        from mixle.inference.multiple_testing import adjust_pvalues

        raw = np.asarray([p for _n, _x, _ts, p, _ss, _sp in primary] + [sp for *_rest, sp in primary], dtype=float)
        adjusted = adjust_pvalues(raw, method="holm", alpha=alpha)["pvals_adjusted"]
        half = len(primary)
        trend_adjusted = adjusted[:half]
        shift_adjusted = adjusted[half:]
    else:
        trend_adjusted = shift_adjusted = np.asarray([], dtype=float)

    worst = "exchangeable"
    for index, (name, x, tr_stat, tr_p_raw, sh_stat, sh_p_raw) in enumerate(primary):
        tr_p = float(trend_adjusted[index])
        sh_p = float(shift_adjusted[index])
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
                # Both the raw and the family-corrected p-value are reported: the corrected one is
                # what the verdict was decided on, and the raw one is what a reader would otherwise
                # try to compare against alpha themselves and get a different answer (MXR-080-1887).
                "trend_p": round(tr_p, 4),
                "shift_p": round(sh_p, 4),
                "trend_p_raw": round(tr_p_raw, 4),
                "shift_p_raw": round(sh_p_raw, 4),
            }
        )
        if verdict == "trend" or (verdict == "shift" and worst == "exchangeable"):
            worst = verdict
    receipt["multiplicity"] = correction
    return ExchangeabilityReport(label=worst if tested_any else "inconclusive", fields=fields, **receipt)
