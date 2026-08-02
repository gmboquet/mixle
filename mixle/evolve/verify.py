"""The champion/challenger gate: "did the challenger SIGNIFICANTLY beat the champion, no regression?"

This is the single decision rule the whole loop turns on, and the anti-regression guarantee lives
here. It is pure glue over existing statistics -- it owns the *policy* (alpha, the effect-size floor,
the calibration no-regression rule, multiplicity) and never edits the underlying tests:

1. **Paired significance** -- :func:`mixle.inference.model_comparison.paired_score_difference` on the
   objective's per-observation vectors, scored from the *same* held-out batch in the same order.
2. **Pairing integrity** -- both vectors are required to have equal length (they are produced from one
   ``data`` argument, so the order cannot diverge); a mismatch is a hard error, not a silent skip.
3. **Effect-size floor** -- ``|mean_diff| >= min_effect``; a statistically significant but practically
   negligible win does not promote.
4. **Non-nested cross-check** -- for family swaps (``nonnested=True``) the challenger must additionally
   win :func:`vuong_test` *and* :func:`clarke_test` (BIC-corrected) on the pointwise log-likelihoods.
5. **ELPD band** -- when LOO/WAIC pointwise arrays are supplied, :func:`compare_elpd`'s 2-SE band is the
   conservative tie rule.
6. **Calibration no-regression** -- a more-accurate-but-less-calibrated challenger is refused: its
   calibration scalar must not exceed the champion's by ``calib_tol``. :attr:`Verdict.
   calibration_status` is ``'unavailable'`` only for an explicit, checked applicability decision --
   e.g. the calibration objective is PIT-based and needs a continuous predictive distribution, so it
   does not apply to a categorical/discrete model -- since that is not itself evidence of a
   regression. An *unexpected* failure inside the computation itself (applicability was confirmed,
   yet it still raised) is ``'error'``, not ``'unavailable'``: an implementation defect must not
   silently read as calibrated, so ``'error'`` blocks promotion the same as ``'failed'`` -- see
   :func:`_calibration_no_regression` and :attr:`Verdict.calibrated`.
7. **Multiplicity** -- comparing one champion/challenger pair produces exactly one p-value, so this
   function cannot correct it for multiplicity by itself: every method in
   :mod:`mixle.inference.multiple_testing` is the identity transform at family size 1 (bonferroni
   multiplies alpha by 1; BH ranks the one p-value against itself). A caller running many simultaneous
   challengers at once (e.g. one population generation) must pool the RAW p-values from every comparison
   and adjust them together, ONCE, via :func:`mixle.inference.multiple_testing.adjust_pvalues`, then
   compare each candidate's own adjusted p-value to ``alpha`` itself -- see
   :meth:`mixle.evolve.population.Population.step`, which does exactly this. That pool must first drop
   any :attr:`Verdict.p_value` that is ``nan`` -- see the scalar-only note below --
   :func:`~mixle.inference.multiple_testing.adjust_pvalues` rejects non-finite input outright rather
   than silently mishandling it, so an un-filtered pool crashes instead of under-correcting.
8. **Scalar-only objectives** -- an objective whose ``pointwise`` returns ``None`` (e.g.
   :func:`~mixle.evolve.objective.calibration_objective`,
   :func:`~mixle.evolve.objective.decision_regret_objective`) has no per-observation vector to pair, so
   no paired test can run: :attr:`Verdict.favored` is still decided from a bare
   scalar-delta-vs-``min_effect`` comparison (reported for a human to review), but :attr:`Verdict.
   p_value`/``ci`` are set to ``nan`` as an explicit "not applicable" sentinel (there being no null
   hypothesis test to report a p-value or CI *for*) -- not a failure to compute one. Downstream code
   that pools p-values across many verdicts (point 7 above) must treat this ``nan`` as "exclude from
   the pool", not "coerce to 0 (very significant)" or "coerce to 1 (never significant)": both would
   misrepresent a comparison that was never run as one that was. A bare scalar delta alone carries no
   sampling-uncertainty estimate -- no replication, bootstrap, or resampling evidence backs it -- so it
   cannot support the same evidence-bearing promotion guarantee a paired p-value gives:
   :attr:`Verdict.promote` is ``False`` for every scalar-only verdict regardless of ``favored`` (see
   :attr:`Verdict.has_statistical_evidence`). A scalar objective that gains its own documented
   resampling/replication contract (e.g. multiple independent fit replicates, or a bootstrap over the
   held-out data) would be a legitimate way to earn auto-promotion eligibility in the future; none
   does today, so a scalar-only win is flagged in evidence for a human to act on, not auto-promoted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.evolve.objective import Objective, calibration_objective, pointwise_log_density
from mixle.inference.model_comparison import (
    clarke_test,
    compare_elpd,
    paired_score_difference,
    vuong_test,
)
from mixle.utils.immutable import detach_receipt_container


@dataclass(frozen=True)
class Verdict:
    """The outcome of a single champion/challenger comparison."""

    favored: str  # 'challenger' | 'champion' | 'tie'
    delta: float  # objective improvement, champion_scalar - challenger_scalar (>0 == challenger better)
    p_value: float  # nan for a scalar-only objective (no paired test exists) -- see module docstring point 8
    ci: tuple[float, float]  # (nan, nan) alongside a nan p_value, for the same reason
    calibration_status: str  # 'passed'|'failed'|'unavailable'|'error' -- see _calibration_no_regression
    evidence: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # detach, not freeze: promotion evidence is serialized and compared by concrete type
        # downstream (MXR-080-1876).
        object.__setattr__(self, "evidence", detach_receipt_container(self.evidence))

    @property
    def calibrated(self) -> bool:
        """True unless calibration regressed, or its computation broke unexpectedly.

        Kept for callers that only need the promotion-relevant boolean; ``calibration_status``
        carries the more precise state a bare bool cannot: 'unavailable' (calibration wasn't
        requested, or is genuinely inapplicable to this model/data -- an explicit, checked
        applicability decision, not itself evidence of a regression, so it does not block
        promotion, the same as 'passed') is distinct from both 'failed' (calibration was computed
        for both models and the challenger's is worse by more than ``calib_tol``) and 'error' (the
        computation raised unexpectedly despite looking applicable -- an implementation defect, not
        a legitimate skip). 'failed' and 'error' are the only statuses that block promotion; an
        'error' must not silently read the same as a pass.
        """
        return self.calibration_status in ("passed", "unavailable")

    @property
    def has_statistical_evidence(self) -> bool:
        """True unless this verdict came from a scalar-only objective with no paired test to run.

        A scalar-only objective (module docstring point 8) has no per-observation vector to pair,
        so ``p_value``/``ci`` are the explicit ``nan`` "not applicable" sentinel: there is no
        sampling-uncertainty estimate -- no p-value, CI, replication, or bootstrap -- behind the raw
        scalar delta at all. Automatic promotion requires this to be True; a scalar-only win is
        still reported (``favored``/``delta``/``evidence['scalar_only']``) for a human to review,
        but it can never auto-promote on its own -- see :attr:`promote`.
        """
        return not np.isnan(self.p_value)

    @property
    def promote(self) -> bool:
        """True iff the challenger is favored, calibration didn't fail or error, and the verdict is
        backed by an actual statistical test (see :attr:`has_statistical_evidence`) -- the single
        promotion predicate."""
        return self.favored == "challenger" and self.calibrated and self.has_statistical_evidence

    def as_dict(self) -> dict[str, Any]:
        """Serialize the verdict into JSON-compatible primitive fields."""
        return {
            "favored": self.favored,
            "delta": self.delta,
            "p_value": self.p_value,
            "ci": list(self.ci),
            "calibrated": self.calibrated,
            "calibration_status": self.calibration_status,
            "has_statistical_evidence": self.has_statistical_evidence,
            "evidence": self.evidence,
        }


def _calibration_no_regression(
    champion: Any,
    challenger: Any,
    data: Any,
    *,
    calib_tol: float,
    seed: int,
) -> tuple[str, dict]:
    """Challenger calibration must not be worse than the champion's by more than ``calib_tol``.

    Returns ('passed' | 'failed' | 'unavailable' | 'error', evidence).

    'unavailable' is an explicit, checked applicability decision -- calibration genuinely does not
    apply to this champion/challenger/data triple -- never a guess from whatever exception a broken
    computation happens to raise. Two checked cases land here: ``data`` cannot be interpreted as a
    continuous numeric response (calibration_objective is PIT-based; a categorical/discrete model's
    class labels are not a valid PIT input -- e.g. a self-evolution loop fitting categorical
    models), or the computation raises ``AttributeError`` -- Python's own structural signal that
    some required capability (e.g. ``.sampler()``) is genuinely missing from a model, as opposed to
    the ambiguous ``ValueError`` the non-numeric-data case above could equally be mistaken for.
    Neither is evidence of a regression, so neither blocks promotion, matching 'passed'.

    'error' is the opposite: the applicability checks above passed (numeric data, no missing-
    capability ``AttributeError``), yet the computation still raised. That is not "doesn't apply",
    it is an unexpected failure inside the calibration computation itself -- an implementation
    defect must not silently read as calibrated, so 'error' blocks promotion the same as 'failed'
    (see :attr:`Verdict.calibrated`).

    Only 'failed' (calibration was actually computed for both models, and the challenger's is
    worse) and 'error' block promotion.
    """
    try:
        np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        # explicit applicability check: PIT calibration needs a continuous numeric response, so
        # categorical/discrete `data` genuinely cannot be scored by this objective.
        return "unavailable", {"calibration": "unavailable", "reason": f"data is not numeric: {exc}"}
    try:
        obj = calibration_objective(seed=seed)
        champ_cal = obj.scalar(champion, data)
        chal_cal = obj.scalar(challenger, data)
    except AttributeError as exc:
        # a genuinely missing model capability (e.g. no .sampler()): AttributeError is Python's own
        # structural signal that the referenced attribute does not exist -- "this objective does not
        # apply to this model", not a guess from an inherently ambiguous exception type.
        return "unavailable", {"calibration": "unavailable", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001
        # applicability was already confirmed above (numeric data, and not a missing-capability
        # AttributeError): this is an unexpected failure in the computation itself, not a legitimate
        # "doesn't apply" case -- must not silently pass as calibrated, unlike 'unavailable' above.
        return "error", {"calibration": "error", "reason": str(exc)}
    ok = bool(chal_cal <= champ_cal + calib_tol)
    status = "passed" if ok else "failed"
    return status, {"champion_calib": champ_cal, "challenger_calib": chal_cal, "calib_tol": calib_tol, "ok": ok}


def _check_policy(*, alpha: float, min_effect: float, calib_tol: float) -> None:
    """Validate the gate's policy knobs before any of them can authorize a promotion.

    None of these had a domain check, and each fails in a different, quiet direction:
    ``alpha=2`` promotes unconditionally because every valid p-value is below it; a negative
    ``min_effect`` makes the effect-size floor vacuous (``abs(mean_diff) >= -1`` is always true), and
    a negative ``calib_tol`` inverts the no-regression rule into a requirement that the challenger be
    strictly *better* calibrated; a NaN in any of them turns its comparison into a silent ``False``,
    which reads as "no evidence" for alpha but as "no floor breached" nowhere consistent. These are
    release-promotion policy, so they fail loudly rather than degrade.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be a finite significance level in (0, 1), got {alpha!r}")
    if not math.isfinite(min_effect) or min_effect < 0.0:
        raise ValueError(f"min_effect must be a finite, non-negative effect-size floor, got {min_effect!r}")
    if not math.isfinite(calib_tol) or calib_tol < 0.0:
        raise ValueError(f"calib_tol must be a finite, non-negative tolerance, got {calib_tol!r}")


def _check_pointwise(vec: Any, side: str, objective: Objective, n_rows: int | None) -> np.ndarray:
    """Require exactly one score per held-out row: a flat ``(len(data),)`` vector, nothing else.

    Both vectors used to be flattened with ``.reshape(-1)`` and then compared only to each other, so
    any pair of same-sized arrays passed. Two ``(2, 2)`` score matrices were accepted as four paired
    observations for a 100-row dataset, and the challenger promoted on them. That lets accidental
    broadcasting, an aggregation that collapsed rows, or fabricated replication decide a release
    promotion, and the paired test's own assumption -- one score per held-out observation, paired by
    position -- is silently violated with no trace in the verdict.

    Enforced here rather than in the objectives themselves because this is the gate: it is the
    function that turns a vector into a promotion decision, and a custom :class:`Objective` is
    caller-supplied code.
    """
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"objective {objective.name!r} returned a {arr.shape} {side} score array; the paired gate "
            "requires a flat (len(data),) vector of one score per held-out row, not a shape that has "
            "to be flattened or broadcast to become one."
        )
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ValueError(
            f"objective {objective.name!r} returned {arr.shape[0]} {side} scores for {n_rows} held-out "
            "rows; the paired gate requires exactly one score per row, paired by position."
        )
    return arr


def challenger_beats_champion(
    champion: Any,
    challenger: Any,
    data: Any,
    *,
    objective: Objective,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    require_calibration: bool = True,
    nonnested: bool = False,
    multiplicity: str | None = None,
    calib_tol: float = 1.0e-3,
    seed: int = 0,
    elpd_pointwise: tuple[np.ndarray, np.ndarray] | None = None,
) -> Verdict:
    """Decide whether ``challenger`` significantly and non-regressively beats ``champion`` on ``data``.

    Args:
        champion, challenger: two fitted models scored on the *same* held-out ``data``.
        data: the held-out responses (one batch -> both models scored in the same order).
        objective: the :class:`~mixle.evolve.objective.Objective` to compare on.
        alpha: significance level for the paired test (and the CI is at ``1 - alpha``).
        min_effect: practical effect-size floor on ``|mean score difference|``.
        require_calibration: if True, run the calibration no-regression check.
        nonnested: if True (a family swap), additionally require Vuong + Clarke to favor the challenger.
        multiplicity: must stay ``None``. A single champion/challenger comparison produces exactly one
            p-value, so it can never be corrected for multiplicity in isolation -- passing a method name
            here raises rather than silently no-op'ing (every :mod:`mixle.inference.multiple_testing`
            method is the identity transform at family size 1). Kept as a parameter only to fail loudly
            on the mistake instead of TypeError'ing on removal; pool the raw p-values from every
            simultaneous comparison and call :func:`mixle.inference.multiple_testing.adjust_pvalues`
            once yourself, then compare each adjusted p-value to ``alpha``.
        calib_tol: tolerance on the calibration-error increase the challenger may carry.
        seed: RNG seed for the (sampled) calibration scalars.
        elpd_pointwise: optional ``(pointwise_champion, pointwise_challenger)`` LOO/WAIC arrays; when
            given, the :func:`compare_elpd` 2-SE band is required to also favor the challenger.

    Returns:
        A :class:`Verdict`. ``verdict.promote`` is the single promotion predicate; it is always
        ``False`` for a scalar-only objective, regardless of ``favored`` -- see module docstring
        point 8 and :attr:`Verdict.has_statistical_evidence`.

    Raises:
        ValueError: if a policy value is outside its domain (see :func:`_check_policy`), or if a
            pointwise vector does not carry exactly one score per held-out row (see
            :func:`_check_pointwise`).
    """
    _check_policy(alpha=alpha, min_effect=min_effect, calib_tol=calib_tol)
    champ_vec = objective.pointwise(champion, data)
    chal_vec = objective.pointwise(challenger, data)

    evidence: dict[str, Any] = {"objective": objective.name, "alpha": alpha, "min_effect": min_effect}

    if champ_vec is None or chal_vec is None:
        # scalar-only objective (e.g. calibration / decision regret): compare scalars directly, no
        # paired test available. `favored` still reports which side the raw scalar delta favors (for
        # evidence/telemetry -- e.g. a caller logging "gate verdict: favored=... promote=..."), but
        # p_value/ci stay the nan "not applicable" sentinel, so Verdict.promote can never fire from
        # this branch alone: a bare scalar comparison carries no sampling-uncertainty estimate -- no
        # replication, bootstrap, or resampling evidence -- so it cannot support the same evidence-
        # bearing promotion guarantee a paired p-value gives (see Verdict.has_statistical_evidence and
        # module docstring point 8). A clear scalar win is reported for a human to review, not
        # auto-promoted.
        champ_s = objective.scalar(champion, data)
        chal_s = objective.scalar(challenger, data)
        delta = (champ_s - chal_s) if objective.lower_is_better else (chal_s - champ_s)
        favored = "challenger" if delta > max(min_effect, 0.0) else "champion" if delta < -min_effect else "tie"
        calibration_status = "unavailable"
        if require_calibration:
            calibration_status, cal_ev = _calibration_no_regression(
                champion, challenger, data, calib_tol=calib_tol, seed=seed
            )
            evidence["calibration"] = cal_ev
        evidence["scalar_only"] = {"champion": champ_s, "challenger": chal_s}
        return Verdict(favored, float(delta), float("nan"), (float("nan"), float("nan")), calibration_status, evidence)

    n_rows = len(data) if hasattr(data, "__len__") else None
    champ_vec = _check_pointwise(champ_vec, "champion", objective, n_rows)
    chal_vec = _check_pointwise(chal_vec, "challenger", objective, n_rows)
    if champ_vec.shape[0] != chal_vec.shape[0]:
        raise ValueError(
            "pairing-integrity violation: champion vector has %d entries, challenger has %d -- both "
            "must be scored from the same held-out batch in the same order." % (champ_vec.shape[0], chal_vec.shape[0])
        )

    paired = paired_score_difference(
        champ_vec, chal_vec, lower_is_better=objective.lower_is_better, ci_level=1.0 - alpha
    )
    evidence["paired"] = paired

    p_value = float(paired["p_value"])
    if multiplicity is not None:
        # a single champion/challenger pair is exactly one p-value: adjust_pvalues on a length-1 array
        # is the identity transform for every method (bonferroni multiplies alpha by 1; BH ranks the
        # value against itself), so "adjusting" it here would silently do nothing while looking like a
        # real correction. Refuse instead -- see the module docstring's Multiplicity note and
        # mixle.evolve.population.Population.step for the correct pooled-family pattern.
        raise ValueError(
            "challenger_beats_champion compares exactly one pair, so it has exactly one p-value -- "
            "multiplicity correction is a no-op at family size 1 for every method in "
            "mixle.inference.multiple_testing. Pool the raw p-values from every simultaneous comparison "
            "and call adjust_pvalues(..., method=%r) ONCE across the whole family, then compare each "
            "candidate's own adjusted p-value to alpha yourself (see Population.step)." % multiplicity
        )

    # mean_diff = mean(champion - challenger). For lower-is-better the challenger is better when its
    # score is smaller, i.e. champion - challenger > 0, so delta = +mean_diff. For higher-is-better the
    # challenger is better when champion - challenger < 0, so delta = -mean_diff. ``delta`` is thus
    # normalised so positive always means "challenger better".
    mean_diff = float(paired["mean_diff"])
    delta = mean_diff if objective.lower_is_better else -mean_diff

    significant = p_value < alpha
    favored_paired = paired["favored"] == "B"  # 'B' is the challenger in paired_score_difference
    effect_ok = abs(mean_diff) >= min_effect

    favored = (
        "challenger"
        if (significant and favored_paired and effect_ok)
        else ("champion" if (significant and paired["favored"] == "A") else "tie")
    )

    # non-nested robustness cross-check for family swaps
    if favored == "challenger" and nonnested:
        try:
            ll_champ = pointwise_log_density(champion, data)
            ll_chal = pointwise_log_density(challenger, data)
            vuong = vuong_test(ll_chal, ll_champ, correction="bic")
            clarke = clarke_test(ll_chal, ll_champ, correction="bic")
            evidence["vuong"] = vuong
            evidence["clarke"] = clarke
            # in these calls challenger is 'A', champion is 'B' -> challenger wins iff favored == 'A'
            if not (vuong["favored"] == "A" and clarke["favored"] == "A"):
                favored = "tie"
        except Exception as exc:  # noqa: BLE001
            # a required cross-check that could not run is not a pass: refuse promotion rather than
            # silently keeping the earlier paired-test "challenger" verdict unverified.
            evidence["nonnested_error"] = str(exc)
            favored = "tie"

    # ELPD 2-SE band (Bayesian models with LOO/WAIC pointwise arrays)
    if favored == "challenger" and elpd_pointwise is not None:
        pw_champ, pw_chal = elpd_pointwise
        elpd = compare_elpd(np.asarray(pw_chal, dtype=float), np.asarray(pw_champ, dtype=float))
        evidence["elpd"] = elpd
        if elpd["favored"] != "A":  # challenger is 'A' here
            favored = "tie"

    calibration_status = "unavailable"
    if require_calibration:
        calibration_status, cal_ev = _calibration_no_regression(
            champion, challenger, data, calib_tol=calib_tol, seed=seed
        )
        evidence["calibration"] = cal_ev

    ci = (float(paired["ci_low"]), float(paired["ci_high"]))
    return Verdict(favored, float(delta), p_value, ci, calibration_status, evidence)


__all__ = ["Verdict", "challenger_beats_champion"]
