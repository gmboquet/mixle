"""Calibration gate -- a deployable IC-6 verifier that challenges whether a posterior's reported
uncertainty is actually *earned*, instead of checking only its shape.

Motivation (found empirically, across several end-to-end pipelines): the UQ in this codebase is
*locally honest* -- every module computes a real posterior -- but was *globally unaudited*. A
systematically overconfident or miscalibrated posterior sailed straight through to a confident-looking
final answer, because nothing in the flow ever challenged the confidence. The calibration *primitives*
already existed (:mod:`mixle.inference.calibration`: ``pit_ensemble`` / ``interval_coverage`` /
``coverage_curve`` / ``pit_calibration_error``); what was missing was a *gate* that composes them into
a pass/fail verifier a pipeline (e.g. :func:`mixle.task.knowledge_routing.route_task`) can actually
stop on. This module is that gate -- a thin composition, not new statistics.

**Honest boundary, stated up front rather than buried.** This gate catches:

  * *overconfidence / underconfidence* of a posterior relative to HELD-OUT DATA it can be checked
    against (:func:`posterior_predictive_calibration`), and
  * *inference-algorithm bugs* -- an inference that is not self-consistent under its own generative
    model -- via simulation-based calibration (:func:`simulation_based_calibration`).

It does **not** catch model misspecification under genuine non-uniqueness. If the data you can hold
out simply cannot distinguish the true state (the textbook example: surface gravity cannot resolve a
source's depth -- shallow and deep bodies produce nearly identical surface fields), then a biased
posterior fits the held-out data perfectly and this gate will *correctly* pass it. Catching that
requires either data that can see the biased dimension (e.g. one borehole) or a physics-based prior
-- not more calibration checking. This gate reports what the available data can support; it never
manufactures confidence the data cannot justify, and it never claims to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.inference.calibration import (
    coverage_curve,
    interval_coverage,
    pit_calibration_error,
    pit_ensemble,
)

__all__ = [
    "CalibrationVerdict",
    "posterior_predictive_calibration",
    "simulation_based_calibration",
    "CalibrationVerifier",
]


@dataclass
class CalibrationVerdict:
    """The outcome of a calibration check. Carries a ``passed`` flag (so it duck-types the IC-6
    ``Verdict`` the routing/orchestration layer reads via ``.passed``) plus the diagnostics that
    justify it -- a gate should say *why* it failed, not just that it did.

    ``null_threshold`` is the key to honest small-sample behaviour: ``pit_error`` is compared against
    the value a *genuinely calibrated* posterior of this exact sample size would produce (the high
    quantile of the finite-sample null), not a fixed constant that would false-alarm on calibrated
    data at small ``n``. ``low_power`` flags when the sample is so small that even a badly
    miscalibrated posterior could not be distinguished from a calibrated one -- in which case a
    ``passed=True`` means "not detectably miscalibrated with this little data," NOT "calibrated."
    """

    passed: bool
    pit_error: float  # deviation of the PIT/rank histogram from uniform (0 == perfectly calibrated)
    null_threshold: float  # the sample-size-aware threshold pit_error is judged against
    coverage_error: float  # DIAGNOSTIC: max |empirical - nominal| across the coverage curve
    reference_level: float
    coverage_at_reference: float  # DIAGNOSTIC: empirical coverage of the reference-level central interval
    mean_interval_width: float
    n_points: int
    low_power: bool = False
    reasons: list[str] = field(default_factory=list)
    kind: str = "calibration"

    @property
    def score(self) -> float:
        """A 0..1 calibration score (1 == at/below the calibrated-null level), for callers that rank
        rather than gate. Normalized by the null threshold so it's comparable across sample sizes."""
        if self.null_threshold <= 0:
            return 1.0 if self.pit_error == 0 else 0.0
        return float(max(0.0, 1.0 - self.pit_error / (2.0 * self.null_threshold)))


def _uniformity_null_threshold(
    n: int, *, bins: int = 10, quantile: float = 0.99, n_null: int = 500, seed: int = 12345
) -> float:
    """The value :func:`~mixle.inference.calibration.pit_calibration_error` would reach on genuinely
    uniform data of size ``n`` -- Monte-Carlo'd at the ``quantile`` upper tail. Comparing an observed
    PIT/rank error against THIS (rather than a fixed constant) is what makes the gate a proper
    finite-sample test: it asks "is this posterior worse-calibrated than 99% of genuinely calibrated
    posteriors of the same sample size would be?", which is scale-correct at every ``n``."""
    rs = RandomState(seed)
    errs = [pit_calibration_error(rs.uniform(0.0, 1.0, size=int(n)), bins=bins) for _ in range(int(n_null))]
    return float(np.quantile(errs, quantile))


def _validate_gate_parameters(
    *,
    reference_level: float | None = None,
    null_quantile: float,
    tolerance: float | None,
    bins: int,
) -> None:
    if reference_level is not None and (not np.isfinite(reference_level) or not 0.0 < reference_level < 1.0):
        raise ValueError("reference_level must be finite and strictly between 0 and 1")
    if not np.isfinite(null_quantile) or not 0.0 < null_quantile < 1.0:
        raise ValueError("null_quantile must be finite and strictly between 0 and 1")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0.0):
        raise ValueError("calibration tolerance must be finite and nonnegative")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer greater than one")


def posterior_predictive_calibration(
    ensemble: np.ndarray,
    held_out_y: np.ndarray,
    *,
    reference_level: float = 0.90,
    null_quantile: float = 0.99,
    pit_tol: float | None = None,
    bins: int = 10,
    low_power_threshold: float = 1.0,
    pit_seed: int | RandomState | None = 0,
) -> CalibrationVerdict:
    """Check a posterior-predictive ensemble against held-out observations it never saw.

    ``ensemble`` is ``(k, m)``: for each of ``k`` held-out points, ``m`` posterior-predictive draws
    (push ``m`` draws from the posterior through the forward model to the observation of each held-out
    point). ``held_out_y`` is the ``(k,)`` array of real observed values at those points.

    The decision is a proper finite-sample uniformity test on the randomized PIT of the held-out data
    (:func:`~mixle.inference.calibration.pit_ensemble`). ``pit_error``
    (:func:`~mixle.inference.calibration.pit_calibration_error`, deviation of the rank histogram from
    uniform) is compared against :func:`_uniformity_null_threshold` -- the ``null_quantile`` upper tail
    of what a *genuinely calibrated* posterior of this exact sample size would produce -- NOT a fixed
    constant (a fixed constant is below the finite-sample noise floor at small ``k`` and would
    false-alarm on calibrated data). Pass ``pit_tol`` to override with a fixed threshold if you have a
    reason to.

    Coverage numbers (:func:`~mixle.inference.calibration.coverage_curve` /
    :func:`~mixle.inference.calibration.interval_coverage`) are computed and reported as human-readable
    diagnostics and to label the *direction* of any miscalibration (over- vs under-confident), but the
    pass/fail itself is the PIT test, which subsumes them and is scale-correct.

    ``low_power``: when the null threshold is at/above ``low_power_threshold`` (the PIT error is
    bounded, so a large threshold means "even gross miscalibration is within null noise at this k"),
    the held-out set is too small to detect miscalibration; ``passed=True`` then means "not
    detectably miscalibrated with this little data," not "calibrated," and ``reasons`` says so.
    """
    ens = np.asarray(ensemble, dtype=float)
    y = np.asarray(held_out_y, dtype=float)
    _validate_gate_parameters(
        reference_level=reference_level,
        null_quantile=null_quantile,
        tolerance=pit_tol,
        bins=bins,
    )
    if not np.isfinite(low_power_threshold) or low_power_threshold < 0.0:
        raise ValueError("low_power_threshold must be finite and nonnegative")
    if ens.ndim != 2:
        raise ValueError(f"ensemble must be (k, m); got shape {ens.shape}")
    if y.ndim != 1:
        raise ValueError(f"held_out_y must be a one-dimensional (k,) array; got shape {y.shape}")
    if ens.shape[0] == 0:
        raise ValueError("calibration requires at least one held-out point")
    if ens.shape[1] == 0:
        raise ValueError("calibration requires at least one posterior-predictive draw per point")
    if y.shape[0] != ens.shape[0]:
        raise ValueError(f"held_out_y has {y.shape[0]} points but ensemble has {ens.shape[0]}")
    if not np.all(np.isfinite(ens)) or not np.all(np.isfinite(y)):
        raise ValueError("ensemble and held_out_y must contain only finite values")

    k = int(y.shape[0])
    pit = pit_ensemble(y, ens, randomize=True, seed=pit_seed)
    pit_err = float(pit_calibration_error(pit, bins=bins))
    threshold = (
        float(pit_tol) if pit_tol is not None else _uniformity_null_threshold(k, bins=bins, quantile=null_quantile)
    )
    low_power = pit_tol is None and threshold >= low_power_threshold

    curve = coverage_curve(ens, y)
    coverage_error = float(np.max(np.abs(curve["empirical"] - curve["nominal"])))
    lo = np.quantile(ens, (1.0 - reference_level) / 2.0, axis=1)
    hi = np.quantile(ens, (1.0 + reference_level) / 2.0, axis=1)
    ref = interval_coverage(lo, hi, y)
    coverage_at_ref = float(ref["coverage"])

    passed = pit_err <= threshold
    reasons: list[str] = []
    if not passed:
        direction = (
            "overconfident (intervals too narrow -- reports false certainty)"
            if coverage_at_ref < reference_level
            else "underconfident (intervals too wide)"
        )
        reasons.append(
            f"miscalibrated: PIT error {pit_err:.3f} > null threshold {threshold:.3f} for k={k}; "
            f"{reference_level:.0%} interval covers {coverage_at_ref:.1%} of held-out points -- {direction}"
        )
    else:
        reasons.append(
            f"not detectably miscalibrated: PIT error {pit_err:.3f} <= null threshold {threshold:.3f} for k={k}; "
            f"{reference_level:.0%} interval covers {coverage_at_ref:.1%} of held-out points"
        )
    if low_power:
        reasons.append(
            f"LOW POWER: k={k} held-out points is too few to detect miscalibration reliably "
            f"(null threshold {threshold:.2f} admits gross miscalibration) -- a pass here means "
            f"'undetectable with this little data', not 'calibrated'. Hold out more points."
        )

    return CalibrationVerdict(
        passed=passed,
        pit_error=pit_err,
        null_threshold=threshold,
        coverage_error=coverage_error,
        reference_level=float(reference_level),
        coverage_at_reference=coverage_at_ref,
        mean_interval_width=float(ref["mean_width"]),
        n_points=k,
        low_power=low_power,
        reasons=reasons,
    )


def simulation_based_calibration(
    prior_sampler: Callable[[RandomState], np.ndarray],
    simulate: Callable[[np.ndarray, RandomState], np.ndarray],
    fit: Callable[[np.ndarray], np.ndarray],
    *,
    n_sims: int = 200,
    param_index: int = 0,
    null_quantile: float = 0.99,
    error_tol: float | None = None,
    bins: int = 10,
    seed: int | RandomState | None = 0,
) -> CalibrationVerdict:
    """Simulation-based calibration (Talts et al. 2018): does the inference recover its own generative
    model? Draw ``theta ~ prior``, simulate ``y ~ p(y|theta)``, refit a posterior, and record the rank
    of the true ``theta`` among the posterior draws. Under a correct inference those ranks are Uniform;
    a systematically over/under-dispersed posterior makes them pile up at the middle or the edges.

    ``prior_sampler(rng) -> theta`` (a ``(d,)`` parameter vector, or scalar-as-``(1,)``);
    ``simulate(theta, rng) -> y`` (any shape the fitter accepts); ``fit(y) -> posterior_draws``
    (``(n_draws, d)`` or ``(n_draws,)``). ``param_index`` selects which parameter's rank to test.

    Unlike :func:`posterior_predictive_calibration`, this needs no held-out real data -- it tests the
    inference *machinery* against synthetic ground truth it generates itself. It therefore catches a
    different class: an inference bug / wrong likelihood / mis-scaled posterior, even on data that would
    look perfectly fit. (It does not test whether the model matches *reality* -- that is
    :func:`posterior_predictive_calibration`'s job, on real held-out data.)
    """
    _validate_gate_parameters(
        null_quantile=null_quantile,
        tolerance=error_tol,
        bins=bins,
    )
    if isinstance(n_sims, bool) or not isinstance(n_sims, (int, np.integer)) or n_sims <= 0:
        raise ValueError("n_sims must be a positive integer")
    if isinstance(param_index, bool) or not isinstance(param_index, (int, np.integer)) or param_index < 0:
        raise ValueError("param_index must be a nonnegative integer")

    rng = seed if isinstance(seed, RandomState) else RandomState(seed)
    ranks = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        theta = np.atleast_1d(np.asarray(prior_sampler(rng), dtype=float))
        if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
            raise ValueError("prior_sampler must return a non-empty finite scalar or one-dimensional vector")
        if param_index >= theta.size:
            raise ValueError(f"param_index {param_index} is outside the sampled parameter vector")
        y = simulate(theta, rng)
        try:
            simulated = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("simulate must return numeric data") from exc
        if simulated.size == 0 or not np.all(np.isfinite(simulated)):
            raise ValueError("simulate must return non-empty finite data")
        draws = np.atleast_1d(np.asarray(fit(y), dtype=float))
        if draws.ndim not in (1, 2) or draws.size == 0 or not np.all(np.isfinite(draws)):
            raise ValueError("fit must return non-empty finite one- or two-dimensional posterior draws")
        if draws.ndim == 1:
            draws_param = draws
            theta_param = float(theta[0]) if theta.shape[0] == 1 else float(theta[param_index])
        else:
            if param_index >= draws.shape[1]:
                raise ValueError(f"param_index {param_index} is outside the fitted posterior draws")
            draws_param = draws[:, param_index]
            theta_param = float(theta[param_index])
        n_draws = draws_param.shape[0]
        ranks[i] = float(np.sum(draws_param < theta_param)) / n_draws  # normalized rank in [0, 1)

    # a uniform rank histogram means calibrated inference; reuse the same PIT-uniformity metric,
    # judged against the same sample-size-aware null threshold (n_sims here plays the role of k).
    sbc_error = float(pit_calibration_error(np.clip(ranks, 0.0, 1.0), bins=bins))
    threshold = (
        float(error_tol)
        if error_tol is not None
        else _uniformity_null_threshold(int(n_sims), bins=bins, quantile=null_quantile)
    )
    passed = sbc_error <= threshold
    reason = (
        f"SBC ranks consistent with uniform (error {sbc_error:.3f} <= null threshold {threshold:.3f} for "
        f"{n_sims} sims): inference is self-consistent under its own generative model"
        if passed
        else f"SBC ranks NOT uniform (error {sbc_error:.3f} > null threshold {threshold:.3f}): inference is "
        f"mis-dispersed (over/under-confident) under its own generative model"
    )
    return CalibrationVerdict(
        passed=passed,
        pit_error=sbc_error,
        null_threshold=threshold,
        coverage_error=float("nan"),
        reference_level=float("nan"),
        coverage_at_reference=float("nan"),
        mean_interval_width=float("nan"),
        n_points=int(n_sims),
        reasons=[reason],
        kind="calibration-sbc",
    )


class CalibrationVerifier:
    """An IC-6-shaped verifier (``.verify(claim, context) -> dict``) wrapping
    :func:`posterior_predictive_calibration`, so a calibration gate drops straight into
    :func:`mixle.task.knowledge_routing.route_task` (or any IC-6 consumer) as a real verifier that
    can *fail* a miscalibrated result -- the thing the routing verifiers used in the experiments only
    ever checked structurally.

    The tool result being verified must carry, on ``claim["payload"]`` (or ``context``), an
    ``"ensemble"`` ``(k, m)`` posterior-predictive draw matrix and a ``"held_out_y"`` ``(k,)`` array.
    A payload without them fails closed with an explicit reason -- a calibration verifier that silently
    passes anything it can't actually check would defeat its own purpose.
    """

    def __init__(
        self, *, reference_level: float = 0.90, null_quantile: float = 0.99, pit_tol: float | None = None
    ) -> None:
        self.reference_level = reference_level
        self.null_quantile = null_quantile
        self.pit_tol = pit_tol

    def verify(self, claim: Any, context: Any = None) -> dict[str, Any]:
        payload = (claim or {}).get("payload", {}) if isinstance(claim, dict) else {}
        source = payload if ("ensemble" in payload and "held_out_y" in payload) else (context or {})
        ensemble = source.get("ensemble") if isinstance(source, dict) else None
        held_out_y = source.get("held_out_y") if isinstance(source, dict) else None
        if ensemble is None or held_out_y is None:
            return {
                "passed": False,
                "score": 0.0,
                "kind": "calibration",
                "reasons": [
                    "no ensemble/held_out_y to calibrate against -- failing closed rather than passing an unchecked posterior"
                ],
            }
        try:
            verdict = posterior_predictive_calibration(
                np.asarray(ensemble),
                np.asarray(held_out_y),
                reference_level=self.reference_level,
                null_quantile=self.null_quantile,
                pit_tol=self.pit_tol,
            )
        except (TypeError, ValueError, FloatingPointError) as exc:
            return {
                "passed": False,
                "score": 0.0,
                "kind": "calibration",
                "reasons": [f"calibration input rejected: {exc}"],
            }
        return {
            "passed": verdict.passed,
            "score": verdict.score,
            "kind": verdict.kind,
            "reasons": verdict.reasons,
            "low_power": verdict.low_power,
        }
