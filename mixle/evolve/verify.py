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
   calibration scalar must not exceed the champion's by ``calib_tol``. Best-effort: :attr:`Verdict.
   calibration_status` is ``'unavailable'`` (not ``'failed'``) when the comparison could not even be
   computed -- e.g. the calibration objective is PIT-based and needs a continuous predictive
   distribution, so it does not apply to a categorical/discrete model -- since that is not itself
   evidence of a regression.
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
   no paired test can run: the verdict is decided from a bare scalar-delta-vs-``min_effect`` comparison
   instead, and :attr:`Verdict.p_value`/``ci`` are set to ``nan`` as an explicit "not applicable"
   sentinel (there being no null hypothesis test to report a p-value or CI *for*) -- not a failure to
   compute one. Downstream code that pools p-values across many verdicts (point 7 above) must treat
   this ``nan`` as "exclude from the pool", not "coerce to 0 (very significant)" or "coerce to 1 (never
   significant)": both would misrepresent a comparison that was never run as one that was.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class Verdict:
    """The outcome of a single champion/challenger comparison."""

    favored: str  # 'challenger' | 'champion' | 'tie'
    delta: float  # objective improvement, champion_scalar - challenger_scalar (>0 == challenger better)
    p_value: float  # nan for a scalar-only objective (no paired test exists) -- see module docstring point 8
    ci: tuple[float, float]  # (nan, nan) alongside a nan p_value, for the same reason
    calibration_status: str  # 'passed' | 'failed' | 'unavailable' -- see _calibration_no_regression
    evidence: dict = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        """True unless calibration was actually computed and found to have regressed.

        Kept for callers that only need the promotion-relevant boolean; ``calibration_status``
        carries the more precise three-way state a bare bool cannot: 'unavailable' (calibration
        wasn't requested, or couldn't be computed for this model/objective -- not itself evidence
        of a regression, so it does not block promotion, the same as 'passed') is distinct from
        'failed' (calibration was computed for both models and the challenger's is worse by more
        than ``calib_tol`` -- the only status that blocks promotion).
        """
        return self.calibration_status != "failed"

    @property
    def promote(self) -> bool:
        """True iff the challenger is favored *and* calibration didn't fail -- the promotion predicate."""
        return self.favored == "challenger" and self.calibrated

    def as_dict(self) -> dict[str, Any]:
        """Serialize the verdict into JSON-compatible primitive fields."""
        return {
            "favored": self.favored,
            "delta": self.delta,
            "p_value": self.p_value,
            "ci": list(self.ci),
            "calibrated": self.calibrated,
            "calibration_status": self.calibration_status,
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

    Returns ('passed' | 'failed' | 'unavailable', evidence). 'unavailable' covers both "this
    objective doesn't apply to this model" (e.g. calibration_objective is PIT-based and needs a
    numeric, continuous predictive distribution -- a categorical/discrete model has no such thing)
    and any other failure to even compute the comparison; there is no exception type that reliably
    tells those apart (a missing .sampler() raises AttributeError, but a categorical model's labels
    failing float conversion raises ValueError -- the same exception a genuine bug in the
    calibration computation itself could raise), so this is deliberately best-effort and does not
    block promotion on its own, the same as 'passed'. Only 'failed' -- calibration was actually
    computed for both models, and the challenger's is worse -- blocks promotion.
    """
    try:
        obj = calibration_objective(seed=seed)
        champ_cal = obj.scalar(champion, data)
        chal_cal = obj.scalar(challenger, data)
    except Exception as exc:  # noqa: BLE001
        return "unavailable", {"calibration": "unavailable", "reason": str(exc)}
    ok = bool(chal_cal <= champ_cal + calib_tol)
    status = "passed" if ok else "failed"
    return status, {"champion_calib": champ_cal, "challenger_calib": chal_cal, "calib_tol": calib_tol, "ok": ok}


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
        A :class:`Verdict`. ``verdict.promote`` is the single promotion predicate.
    """
    champ_vec = objective.pointwise(champion, data)
    chal_vec = objective.pointwise(challenger, data)

    evidence: dict[str, Any] = {"objective": objective.name, "alpha": alpha, "min_effect": min_effect}

    if champ_vec is None or chal_vec is None:
        # scalar-only objective (e.g. calibration / decision regret): compare scalars directly, no
        # paired test available -- favor the challenger only on a clear scalar improvement.
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

    champ_vec = np.asarray(champ_vec, dtype=float).reshape(-1)
    chal_vec = np.asarray(chal_vec, dtype=float).reshape(-1)
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
