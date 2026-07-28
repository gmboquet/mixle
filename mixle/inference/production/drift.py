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


@dataclass
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


def _columns(records: Any, n_fields: int) -> list[np.ndarray]:
    """Split tuple/scalar records into per-field 1-D arrays (best effort; non-numeric -> codes)."""
    rows = list(records)
    if not rows:
        return [np.array([]) for _ in range(max(n_fields, 1))]
    if not isinstance(rows[0], (tuple, list)):
        return [_numeric(rows)]
    return [_numeric([r[i] for r in rows]) for i in range(len(rows[0]))]


def _numeric(values: list[Any]) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):  # categorical -> integer codes by first-seen order
        codes: dict[Any, int] = {}
        return np.asarray([codes.setdefault(v, len(codes)) for v in values], dtype=float)


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
        ref_cols = _columns(reference, len(names) if names else 1)
        cur_cols = _columns(current, len(names) if names else 1)
        for i, (rc, cc) in enumerate(zip(ref_cols, cur_cols)):
            nm = names[i] if names and i < len(names) else f"field_{i}"
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
