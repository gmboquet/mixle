"""Model / data drift detection for production: is current data still the data the model was trained on?

Two complementary views:

* **Feature drift** -- per-field distribution shift between a reference (training) sample and a current
  (production) sample: Population Stability Index (PSI), Kolmogorov-Smirnov, and Jensen-Shannon. These
  are model-agnostic and operate on the schema's fields.
* **Score drift** -- the model-native signal: the distribution of the model's own log-density on current
  data versus on reference data. A fitted mixle model *is* the reference distribution, so if current data
  scores systematically lower (or its log-likelihood distribution shifts) the world has moved away from
  the model -- exactly when to retrain.

:func:`detect_drift` combines both into a :class:`DriftReport` with a single ``drift`` flag against
thresholds, suitable for a monitoring loop (see :class:`mixle.inference.production.monitor.Monitor`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.utils.immutable import detach_receipt_container


def population_stability_index(reference: Any, current: Any, *, bins: int = 10) -> float:
    """PSI between two 1-D numeric samples (bin edges from the reference quantiles).

    Rule of thumb: < 0.1 no shift, 0.1-0.25 moderate, > 0.25 significant."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("inf")
    eps = 1e-6
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        # The reference feature is literally constant (a single distinct value): quantile binning
        # has nothing left to bin against, so this used to unconditionally report 0.0 ("no drift")
        # regardless of `current` -- even a current sample that has moved entirely away from that
        # one value. Compare the mass still AT the reference value vs. everywhere else instead --
        # the degenerate two-bin case of the same PSI formula below, not a silent short-circuit.
        v = float(ref[0])
        c_at = float(np.mean(np.isclose(cur, v)))
        rp = np.array([1.0, 0.0]) + eps
        cp = np.array([c_at, 1.0 - c_at]) + eps
        return float(np.sum((cp - rp) * np.log(cp / rp)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0].astype(float)
    c = np.histogram(cur, edges)[0].astype(float)
    rp = r / r.sum() + eps
    cp = c / c.sum() + eps
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def ks_statistic(reference: Any, current: Any) -> float:
    """Two-sample Kolmogorov-Smirnov statistic in [0, 1] (larger = more shift)."""
    from scipy.stats import ks_2samp

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 1.0
    return float(ks_2samp(ref, cur).statistic)


def js_divergence(reference: Any, current: Any, *, bins: int = 20) -> float:
    """Jensen-Shannon divergence (bits) between two 1-D numeric samples (shared histogram support)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("inf")
    lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(ref, edges)[0].astype(float) + 1e-12
    q = np.histogram(cur, edges)[0].astype(float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _log_densities(model: Any, rows: list) -> np.ndarray:
    """One log-density per row of an ALREADY-MATERIALIZED population.

    ``rows`` must be a list, not an iterator: the batch path is tried first and, if it raises, the
    scalar path retries over these same rows. Passing a one-shot stream meant the retry iterated an
    iterator the batch attempt had already drained, so a transient batch failure silently reported no
    scores at all. The dispatch is non-destructive by construction now, and the chosen path's output
    is required to have exactly one score per row.
    """
    try:
        enc = model.dist_to_encoder().seq_encode(rows)
        scores = np.asarray(model.seq_log_density(enc), dtype=float)
    except Exception:  # noqa: BLE001 - the batch path is optional; retry the SAME rows scalar-wise
        scores = np.asarray([model.log_density(x) for x in rows], dtype=float)
    scores = np.atleast_1d(scores)
    if scores.shape != (len(rows),):
        raise ValueError(
            f"model scoring returned {scores.shape} log-densities for {len(rows)} records; drift "
            "analysis needs exactly one score per record to compare populations"
        )
    return scores


def score_drift(model: Any, reference: Any, current: Any) -> dict:
    """The model-native drift signal: how the model's log-density distribution shifts from reference to
    current data. Returns the KS statistic between the two log-likelihood samples and their mean shift
    (mean current log-density minus mean reference; negative => current data is less likely under the
    model).

    Both populations are materialized here exactly once, so a one-shot iterable is safe and the
    reported counts describe the records actually scored. Alongside the shift statistics the report
    carries the evidence behind them -- how many records each population held and what fraction of
    each was unscorable -- because a shift measured over a handful of surviving finite scores is not
    the same claim as one measured over the whole sample.
    """
    ref_rows = list(reference)
    cur_rows = list(current)
    ll_ref = _log_densities(model, ref_rows)
    ll_cur = _log_densities(model, cur_rows)
    fr = ll_ref[np.isfinite(ll_ref)]
    fc = ll_cur[np.isfinite(ll_cur)]
    return {
        "ks": ks_statistic(fr, fc),
        "mean_loglik_shift": float(fc.mean() - fr.mean()) if fr.size and fc.size else float("-inf"),
        "mean_loglik_reference": float(fr.mean()) if fr.size else None,
        "mean_loglik_current": float(fc.mean()) if fc.size else None,
        "fraction_unscorable_current": float(np.mean(~np.isfinite(ll_cur))) if ll_cur.size else 0.0,
        "fraction_unscorable_reference": float(np.mean(~np.isfinite(ll_ref))) if ll_ref.size else 0.0,
        "n_reference": int(ll_ref.size),
        "n_current": int(ll_cur.size),
        "n_scorable_reference": int(fr.size),
        "n_scorable_current": int(fc.size),
    }


@dataclass(frozen=True)
class DriftReport:
    """Drift decision, aggregate score, feature details, and thresholds.

    ``reasons`` names every condition that raised the ``drift`` flag, so a verdict driven by lost
    scoring coverage is distinguishable from one driven by a genuine distribution shift.
    """

    drift: bool
    score: dict
    per_feature: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    processed_count: int = 0
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # A receipt is a record. Detaching severs the caller's alias, so a mutation after
        # construction cannot rewrite evidence that was already recorded; `frozen=True` above
        # stops the field being rebound through the receipt itself. Containers keep their
        # concrete types -- see detach_receipt_container for why (MXR-080-1876).
        object.__setattr__(self, "score", detach_receipt_container(self.score))
        object.__setattr__(self, "per_feature", detach_receipt_container(self.per_feature))
        object.__setattr__(self, "thresholds", detach_receipt_container(self.thresholds))
        object.__setattr__(self, "reasons", detach_receipt_container(self.reasons))

    def __str__(self) -> str:
        flag = "DRIFT" if self.drift else "ok"
        feats = ", ".join(f"{k}: psi={v['psi']:.3f}/ks={v['ks']:.3f}" for k, v in self.per_feature.items())
        why = f"\n  reasons: {', '.join(self.reasons)}" if self.reasons else ""
        return (
            f"DriftReport[{flag}]  score: ks={self.score.get('ks'):.3f}, "
            f"mean_loglik_shift={self.score.get('mean_loglik_shift'):.3f}"
            + why
            + (f"\n  features: {feats}" if feats else "")
        )


def _raw_columns(records: Any, n_fields: int) -> list[list[Any]]:
    """Split tuple/scalar records into per-field lists of raw (un-coded) values."""
    rows = list(records)
    if not rows:
        return [[] for _ in range(max(n_fields, 1))]
    if not isinstance(rows[0], (tuple, list)):
        return [list(rows)]
    return [[r[i] for r in rows] for i in range(len(rows[0]))]


def _numeric_pair(ref_values: list[Any], cur_values: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Encode one field's reference and current values onto a SHARED numeric axis.

    Coding each side independently by first-seen order is what made complete population
    replacement invisible: twenty reference rows of only "a" and twenty current rows of only "b"
    each start their own code map, so both become all-zero vectors and every divergence on them is
    exactly zero. The reference defines the vocabulary (it is the schema the model was trained on)
    and any level the reference never saw shares one explicit "unseen" code above it, so a brand
    new category registers as movement rather than as code 0 again.
    """
    try:
        return np.asarray(ref_values, dtype=float), np.asarray(cur_values, dtype=float)
    except (TypeError, ValueError):
        codes: dict[Any, int] = {}
        for v in ref_values:  # reference-first, so codes are the training vocabulary
            codes.setdefault(v, len(codes))
        unseen = len(codes)  # one shared bin for every level absent from the reference
        ref = np.asarray([codes[v] for v in ref_values], dtype=float)
        cur = np.asarray([codes.get(v, unseen) for v in cur_values], dtype=float)
        return ref, cur


def detect_drift(
    model: Any,
    reference: Any,
    current: Any,
    *,
    psi_threshold: float = 0.25,
    ks_threshold: float = 0.2,
    loglik_shift_threshold: float = -0.5,
    min_scorable_fraction: float = 0.5,
    unscorable_shift_threshold: float = 0.1,
    per_feature: bool = True,
) -> DriftReport:
    """Combine score drift and per-feature drift into a single :class:`DriftReport`.

    ``drift`` is flagged if the score-distribution KS exceeds ``ks_threshold``, OR the mean log-likelihood
    drops by more than ``-loglik_shift_threshold`` (i.e. ``mean_loglik_shift < loglik_shift_threshold``),
    OR any feature's PSI exceeds ``psi_threshold``, OR the current sample lost too much scoring
    coverage to support a no-drift verdict.

    Coverage is part of the verdict, not just of the report. ``score_drift`` drops unscorable records
    before comparing distributions, so a current sample of one matching score plus nine records the
    model cannot score at all used to produce KS 0, mean shift 0, and ``drift=False`` -- a confident
    "no drift" from 10% of the evidence. Two coverage conditions now fail closed:

    * fewer than ``min_scorable_fraction`` of current records scored finitely, and
    * the current unscorable rate exceeds the reference's by more than ``unscorable_shift_threshold``
      (records moving outside the model's support IS the world moving away from the model).

    Both populations are materialized once and reused for scoring and for the per-field columns, so a
    one-shot iterable is safe: consuming the stream twice previously left the feature pass with empty
    columns and manufactured a PSI-infinity DRIFT verdict out of two identical inputs.
    """
    reference = list(reference)
    current = list(current)
    if not reference:
        raise ValueError("drift detection requires a non-empty reference dataset")
    if not current:
        raise ValueError("drift detection requires a non-empty current dataset")
    # Every comparison below is a bare `>` or `<` against one of these, so an unvalidated threshold
    # decides the verdict silently: NaN makes every comparison False and suppresses drift entirely,
    # and a negative psi_threshold flags drift on identical data, since PSI is non-negative by
    # construction. A threshold that cannot be met, or cannot fail, is not a detector.
    for label, value, upper in (
        ("psi_threshold", psi_threshold, None),
        ("ks_threshold", ks_threshold, 1.0),
        ("min_scorable_fraction", min_scorable_fraction, 1.0),
        ("unscorable_shift_threshold", unscorable_shift_threshold, 1.0),
    ):
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0 or (upper is not None and numeric > upper):
            expected = "a finite non-negative number" if upper is None else f"a finite number in [0, {upper:g}]"
            raise ValueError(f"{label} must be {expected}; got {value!r}")
    if not np.isfinite(float(loglik_shift_threshold)):
        # A shift threshold is legitimately negative -- it is how far the mean log-likelihood may
        # fall -- so only finiteness is required of it.
        raise ValueError(f"loglik_shift_threshold must be a finite number; got {loglik_shift_threshold!r}")
    score = score_drift(model, reference, current)
    reasons: list[str] = []
    if score["ks"] > ks_threshold:
        reasons.append(f"score KS {score['ks']:.3f} > {ks_threshold}")
    if score["mean_loglik_shift"] < loglik_shift_threshold:
        reasons.append(f"mean log-likelihood shift {score['mean_loglik_shift']:.3f} < {loglik_shift_threshold}")

    scorable_current = 1.0 - score["fraction_unscorable_current"]
    if scorable_current < min_scorable_fraction:
        reasons.append(
            f"only {scorable_current:.3f} of current records were scorable (< {min_scorable_fraction}); "
            "too little current evidence to certify no drift"
        )
    unscorable_shift = score["fraction_unscorable_current"] - score["fraction_unscorable_reference"]
    if unscorable_shift > unscorable_shift_threshold:
        reasons.append(
            f"unscorable rate rose {unscorable_shift:.3f} above the reference (> {unscorable_shift_threshold})"
        )

    feats: dict = {}
    if per_feature:
        try:
            from mixle.data.schema import Schema

            names = [f.name for f in Schema.for_model(model).fields]
        except Exception:  # noqa: BLE001
            names = None
        n_fields = len(names) if names else 1
        ref_raw = _raw_columns(reference, n_fields)
        cur_raw = _raw_columns(current, n_fields)
        for i, (rv, cv) in enumerate(zip(ref_raw, cur_raw)):
            nm = names[i] if names and i < len(names) else f"field_{i}"
            # One shared code axis per field, built from the reference: coding the two populations
            # independently made a complete category replacement read as zero drift.
            rc, cc = _numeric_pair(rv, cv)
            psi = population_stability_index(rc, cc)
            feats[nm] = {"psi": psi, "ks": ks_statistic(rc, cc)}
            if psi > psi_threshold:
                reasons.append(f"feature {nm!r} PSI {psi:.3f} > {psi_threshold}")

    return DriftReport(
        drift=bool(reasons),
        score=score,
        per_feature=feats,
        thresholds={
            "psi": psi_threshold,
            "ks": ks_threshold,
            "loglik_shift": loglik_shift_threshold,
            "min_scorable_fraction": min_scorable_fraction,
            "unscorable_shift": unscorable_shift_threshold,
        },
        processed_count=len(current),
        reasons=reasons,
    )
