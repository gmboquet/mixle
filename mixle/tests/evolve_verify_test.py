"""The champion/challenger verify gate (mixle.evolve.verify)."""

import math
import unittest

import numpy as np

from mixle.evolve import (
    challenger_beats_champion,
    crps_objective,
    nll_objective,
)
from mixle.inference.estimation import optimize
from mixle.stats import GaussianDistribution


def _fit(data, mu=0.0, sigma2=1.0):
    return optimize(list(data), GaussianDistribution(mu, sigma2).estimator(), out=None)


class VerifyGateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 600))

    def test_accepts_real_improvement(self):
        # a clearly-wrong champion vs the MLE challenger -> challenger wins, positive delta.
        champion = GaussianDistribution(0.0, 1.0)
        challenger = _fit(self.data, 3.0, 2.0)
        verdict = challenger_beats_champion(champion, challenger, self.data, objective=nll_objective())
        self.assertEqual(verdict.favored, "challenger")
        self.assertTrue(verdict.promote)
        self.assertGreater(verdict.delta, 0.0)

    def test_rejects_noise(self):
        # the same fitted model against itself must tie (no spurious promotion).
        model = _fit(self.data, 3.0, 2.0)
        verdict = challenger_beats_champion(model, model, self.data, objective=nll_objective())
        self.assertEqual(verdict.favored, "tie")
        self.assertFalse(verdict.promote)

    def test_min_effect_floor_blocks_negligible_win(self):
        # a microscopic perturbation may be "significant" on a large n yet practically negligible; a
        # large min_effect floor must refuse it.
        champion = _fit(self.data, 3.0, 2.0)
        challenger = _fit(self.data, 3.0001, 2.0)
        verdict = challenger_beats_champion(champion, challenger, self.data, objective=nll_objective(), min_effect=10.0)
        self.assertFalse(verdict.promote)

    def test_worse_challenger_favors_champion(self):
        champion = _fit(self.data, 3.0, 2.0)
        challenger = GaussianDistribution(0.0, 1.0)  # worse
        verdict = challenger_beats_champion(champion, challenger, self.data, objective=nll_objective())
        self.assertEqual(verdict.favored, "champion")
        self.assertLess(verdict.delta, 0.0)

    def test_pairing_integrity_guard(self):
        # an objective whose pointwise vectors differ in length must raise (cannot pair).
        champion = _fit(self.data, 3.0, 2.0)

        class _RaggedObjective:
            name = "ragged"
            lower_is_better = True

            def pointwise(self, model, data):
                # deliberately return mismatched lengths for champion vs challenger
                n = len(list(data))
                return np.zeros(n if model is champion else n - 1)

            def scalar(self, model, data):
                return 0.0

        with self.assertRaises(ValueError):
            challenger_beats_champion(
                champion,
                GaussianDistribution(3.0, 2.0),
                self.data,
                objective=_RaggedObjective(),
                require_calibration=False,
            )

    def test_crps_objective_paired_vector(self):
        champion = GaussianDistribution(0.0, 1.0)
        challenger = _fit(self.data, 3.0, 2.0)
        verdict = challenger_beats_champion(
            champion, challenger, self.data, objective=crps_objective(seed=0), require_calibration=False
        )
        self.assertEqual(verdict.favored, "challenger")

    def test_multiplicity_on_a_single_pair_raises_instead_of_silently_no_opping(self):
        # a single champion/challenger comparison produces exactly one p-value; every method in
        # mixle.inference.multiple_testing is the identity transform at family size 1 (bonferroni
        # multiplies alpha by 1; BH ranks the lone value against itself), so "correcting" it here would
        # silently do nothing while looking like a real correction -- the exact bug this guards against
        # (see mixle.evolve.population.Population.step, which pools a whole generation's raw p-values
        # and corrects them together ONCE instead of per-candidate). Refuse outright rather than no-op.
        champion = _fit(self.data, 3.0, 2.0)
        challenger = _fit(self.data, 3.0, 2.0)
        with self.assertRaises(ValueError):
            challenger_beats_champion(champion, challenger, self.data, objective=nll_objective(), multiplicity="bh")


class _DelegatingWrapper:
    """Delegates every attribute to a real model except the ones explicitly overridden -- lets a
    test inject one broken capability into an otherwise-working model, rather than hand-building a
    complete fake distribution."""

    def __init__(self, inner, **overrides):
        self._inner = inner
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._inner, name)


class FailOpenGuardTest(unittest.TestCase):
    """The anti-regression guarantee: the non-nested cross-check, a MANDATORY comparison the caller
    explicitly opts into via nonnested=True (unlike calibration, which is deliberately best-effort --
    see the comment on _calibration_no_regression), must refuse promotion if it crashes rather than
    silently letting the earlier paired-test verdict stand unverified. Reuses
    test_accepts_real_improvement's champion/challenger/data (a clearly-better challenger, so the
    paired test alone would promote) and injects a crash into the cross-check specifically."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 600))
        self.champion = GaussianDistribution(0.0, 1.0)
        self.challenger = _fit(self.data, 3.0, 2.0)

    def test_nonnested_crash_refuses_promotion_not_a_silent_pass(self):
        # crps_objective's own paired computation goes through .sampler(), not seq_log_density, so
        # breaking seq_log_density here hits only the non-nested cross-check's own, independent
        # pointwise_log_density(...) calls -- not the earlier paired test that decides "challenger"
        # in the first place. (test_crps_objective_paired_vector already confirms this exact
        # champion/challenger/data setup makes crps_objective favor the challenger.)
        def _broken_seq_log_density(*_a, **_kw):
            raise RuntimeError("boom: a genuine bug in the non-nested cross-check's own computation")

        broken_challenger = _DelegatingWrapper(self.challenger, seq_log_density=_broken_seq_log_density)
        verdict = challenger_beats_champion(
            self.champion,
            broken_challenger,
            self.data,
            objective=crps_objective(seed=0),
            require_calibration=False,
            nonnested=True,
        )
        # the paired test alone would favor the challenger, but the required non-nested cross-check
        # for this family swap never completed -- that must downgrade to a tie, not stay "challenger".
        self.assertEqual(verdict.favored, "tie")
        self.assertFalse(verdict.promote)
        self.assertIn("nonnested_error", verdict.evidence)


class CalibrationStatusTest(unittest.TestCase):
    """Verdict.calibration_status carries the honest state _calibration_no_regression can produce
    ('passed' | 'failed' | 'unavailable' | 'error' -- see CalibrationErrorGuardTest for the 'error'
    case); Verdict.calibrated is a derived, backward-compatible view of it (True for both 'passed'
    and 'unavailable', False for 'failed' and 'error' -- the only two statuses that block
    promotion)."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 600))
        self.champion = GaussianDistribution(0.0, 1.0)
        self.challenger = _fit(self.data, 3.0, 2.0)

    def test_not_requested_is_unavailable(self):
        verdict = challenger_beats_champion(
            self.champion, self.challenger, self.data, objective=nll_objective(), require_calibration=False
        )
        self.assertEqual(verdict.calibration_status, "unavailable")
        self.assertTrue(verdict.calibrated)
        self.assertTrue(verdict.promote)
        self.assertNotIn("calibration", verdict.evidence)

    def test_genuinely_computed_and_better_is_passed(self):
        # the well-fit challenger's calibration is (correctly) better than the badly-fit champion's.
        verdict = challenger_beats_champion(self.champion, self.challenger, self.data, objective=nll_objective())
        self.assertEqual(verdict.calibration_status, "passed")
        self.assertTrue(verdict.calibrated)
        self.assertTrue(verdict.promote)

    def test_genuinely_computed_and_worse_is_failed_and_blocks_promotion(self):
        # NLL alone still favors this challenger (same well-fit model), but its predictive ensemble
        # is deliberately degenerate (a point mass) -- accurate on average, badly calibrated. This
        # is the real scenario calib_tol exists to catch: favored="challenger" from the paired test,
        # yet a genuine calibration regression must still refuse promotion.
        class _DegenerateSampler:
            def sample(self, m, seed=None):
                return np.full(m, 3.0)

        badly_calibrated = _DelegatingWrapper(self.challenger, sampler=lambda seed=None: _DegenerateSampler())
        verdict = challenger_beats_champion(self.champion, badly_calibrated, self.data, objective=nll_objective())
        self.assertEqual(verdict.favored, "challenger")
        self.assertEqual(verdict.calibration_status, "failed")
        self.assertFalse(verdict.calibrated)
        self.assertFalse(verdict.promote)

    def test_inapplicable_to_this_model_is_unavailable_not_failed(self):
        # the legitimate skip case: calibration can't even be attempted for this model, which must
        # not itself count as a calibration regression -- promotion proceeds as if it passed.
        def _raise_attribute_error(*_a, **_kw):
            raise AttributeError("'GaussianDistribution' object has no attribute 'sampler'")

        no_sampler = _DelegatingWrapper(self.challenger, sampler=_raise_attribute_error)
        verdict = challenger_beats_champion(self.champion, no_sampler, self.data, objective=nll_objective())
        self.assertEqual(verdict.calibration_status, "unavailable")
        self.assertTrue(verdict.calibrated)
        self.assertTrue(verdict.promote)
        self.assertEqual(verdict.evidence["calibration"]["calibration"], "unavailable")

    def test_as_dict_carries_both_the_derived_bool_and_the_raw_status(self):
        verdict = challenger_beats_champion(self.champion, self.challenger, self.data, objective=nll_objective())
        d = verdict.as_dict()
        self.assertEqual(d["calibration_status"], "passed")
        self.assertEqual(d["calibrated"], True)


class _ScalarOnlyModel:
    """Bare placeholder 'fitted model' carrying just a scalar score -- stands in for a real model
    scored by a scalar-only objective (calibration_objective / decision_regret_objective in
    production, neither of which has a per-observation vector to pair)."""

    def __init__(self, score):
        self.score = score


class _ScalarOnlyObjective:
    """Mimics calibration_objective / decision_regret_objective's shape: pointwise() always returns
    None, so every comparison goes through challenger_beats_champion's scalar-only branch, whose
    Verdict.p_value/ci are the nan "not applicable" sentinel -- see verify.py's module docstring
    point 8."""

    name = "scalar_only"
    lower_is_better = True

    def pointwise(self, model, data):
        return None

    def scalar(self, model, data):
        return model.score


class ScalarOnlyPromotionGuardTest(unittest.TestCase):
    """A scalar-only objective (module docstring point 8) has no per-observation vector to pair, so
    p_value/ci are the nan "not applicable" sentinel: zero sampling uncertainty, replication, or
    bootstrap evidence backs a bare scalar comparison. Verdict.promote must never fire from this
    branch alone, however clear the scalar delta looks -- favored/delta/evidence['scalar_only'] are
    still reported so a human can act on a genuine win, but auto-promotion requires an actual
    statistical test (see Verdict.has_statistical_evidence)."""

    def test_audit_example_flags_but_does_not_auto_promote(self):
        # the audit's own repro: champion score 1, challenger score 0 (lower is better) -- a clear
        # scalar win, with p_value/ci nan and no sampling uncertainty, replication, or bootstrap
        # evidence behind it at all.
        champion = _ScalarOnlyModel(1.0)
        challenger = _ScalarOnlyModel(0.0)
        verdict = challenger_beats_champion(
            champion, challenger, [0.0], objective=_ScalarOnlyObjective(), require_calibration=False
        )
        self.assertEqual(verdict.favored, "challenger")  # still flagged/reported
        self.assertFalse(verdict.has_statistical_evidence)
        self.assertFalse(verdict.promote)  # but never auto-promoted
        self.assertTrue(math.isnan(verdict.p_value))
        self.assertTrue(all(math.isnan(c) for c in verdict.ci))

    def test_champion_favored_and_tie_still_computed_correctly(self):
        # the guard only touches `promote`; `favored` must still reflect the raw scalar comparison
        # in every direction, not just the "challenger wins" case.
        obj = _ScalarOnlyObjective()
        worse = challenger_beats_champion(
            _ScalarOnlyModel(1.0), _ScalarOnlyModel(2.0), [0.0], objective=obj, require_calibration=False
        )
        self.assertEqual(worse.favored, "champion")
        self.assertFalse(worse.promote)

        tie = challenger_beats_champion(
            _ScalarOnlyModel(1.0), _ScalarOnlyModel(1.0), [0.0], objective=obj, require_calibration=False
        )
        self.assertEqual(tie.favored, "tie")
        self.assertFalse(tie.promote)

    def test_as_dict_carries_has_statistical_evidence(self):
        verdict = challenger_beats_champion(
            _ScalarOnlyModel(1.0),
            _ScalarOnlyModel(0.0),
            [0.0],
            objective=_ScalarOnlyObjective(),
            require_calibration=False,
        )
        d = verdict.as_dict()
        self.assertFalse(d["has_statistical_evidence"])

    def test_paired_objective_is_unaffected_by_the_scalar_only_guard(self):
        # negative control: a REAL paired objective (a genuine per-observation vector, a real
        # p-value) must be completely unaffected by this guard -- has_statistical_evidence is True
        # and a genuine improvement still promotes exactly as before (same setup as
        # VerifyGateTest.test_accepts_real_improvement).
        rng = np.random.RandomState(0)
        data = list(rng.normal(3.0, 2.0, 600))
        champion = GaussianDistribution(0.0, 1.0)
        challenger = _fit(data, 3.0, 2.0)
        verdict = challenger_beats_champion(champion, challenger, data, objective=nll_objective())
        self.assertTrue(verdict.has_statistical_evidence)
        self.assertEqual(verdict.favored, "challenger")
        self.assertTrue(verdict.promote)


class CalibrationErrorGuardTest(unittest.TestCase):
    """_calibration_no_regression must distinguish an explicit, checked applicability decision
    ('unavailable' -- calibration genuinely does not apply, e.g. a missing model capability or
    non-numeric data, so it does not block promotion) from an unexpected failure inside the
    computation itself ('error' -- an implementation defect that must block promotion like
    'failed'), instead of collapsing every exception into the same non-blocking 'unavailable'
    status regardless of cause."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 600))
        self.champion = GaussianDistribution(0.0, 1.0)
        self.challenger = _fit(self.data, 3.0, 2.0)

    def test_unexpected_exception_is_error_and_blocks_promotion(self):
        # a genuine implementation defect during calibration (NOT a missing capability, NOT
        # non-numeric data) -- e.g. a bug that makes ensemble sampling itself blow up -- must fail
        # closed, not silently read as calibrated. Same champion/challenger/data as
        # VerifyGateTest.test_accepts_real_improvement, so NLL alone would promote; only the
        # calibration check should refuse it here.
        def _broken_sampler(seed=None):
            raise RuntimeError("boom: a genuine implementation defect, unrelated to applicability")

        broken = _DelegatingWrapper(self.challenger, sampler=_broken_sampler)
        verdict = challenger_beats_champion(self.champion, broken, self.data, objective=nll_objective())
        self.assertEqual(verdict.favored, "challenger")  # the paired test alone still favors it
        self.assertEqual(verdict.calibration_status, "error")
        self.assertFalse(verdict.calibrated)
        self.assertFalse(verdict.promote)
        self.assertEqual(verdict.evidence["calibration"]["calibration"], "error")

    def test_missing_capability_is_still_unavailable_not_error(self):
        # negative control, contrasting directly with the 'error' case above: the pre-existing
        # legitimate-exemption scenario (see CalibrationStatusTest.
        # test_inapplicable_to_this_model_is_unavailable_not_failed) must still land on
        # 'unavailable', never the new 'error' -- a genuinely missing model capability is an
        # applicability decision, not an implementation defect.
        def _raise_attribute_error(*_a, **_kw):
            raise AttributeError("'GaussianDistribution' object has no attribute 'sampler'")

        no_sampler = _DelegatingWrapper(self.challenger, sampler=_raise_attribute_error)
        verdict = challenger_beats_champion(self.champion, no_sampler, self.data, objective=nll_objective())
        self.assertEqual(verdict.calibration_status, "unavailable")
        self.assertTrue(verdict.calibrated)
        self.assertTrue(verdict.promote)

    def test_non_numeric_data_is_still_unavailable_not_error(self):
        # negative control: the real-world case this fix must not regress -- a self-evolution loop
        # scoring categorical models over class-label data with a main objective that itself
        # tolerates that data fine (its own pointwise score does not depend on data's content) --
        # while the SEPARATE, always-PIT-based calibration_objective this file's calibration check
        # runs internally cannot apply to non-numeric data. That must stay the non-blocking
        # 'unavailable' exemption, not the new 'error'.
        champion, challenger = self.champion, self.challenger

        class _IgnoresDataObjective:
            name = "ignores_data"
            lower_is_better = True

            def pointwise(self, model, data):
                # deliberately independent of `data`'s content -- only cares which model it scores,
                # so a non-numeric `data` cannot break the main paired comparison.
                return np.zeros(20) if model is champion else np.full(20, -1.0)

        categorical_data = ["a", "b"] * 10
        verdict = challenger_beats_champion(champion, challenger, categorical_data, objective=_IgnoresDataObjective())
        self.assertEqual(verdict.calibration_status, "unavailable")
        self.assertTrue(verdict.calibrated)
        self.assertIn("not numeric", verdict.evidence["calibration"]["reason"])

    def test_as_dict_reflects_error_status(self):
        def _broken_sampler(seed=None):
            raise RuntimeError("boom")

        broken = _DelegatingWrapper(self.challenger, sampler=_broken_sampler)
        verdict = challenger_beats_champion(self.champion, broken, self.data, objective=nll_objective())
        d = verdict.as_dict()
        self.assertEqual(d["calibration_status"], "error")
        self.assertEqual(d["calibrated"], False)


class _FixedVectorObjective:
    """An objective returning caller-chosen score arrays, ignoring the model and the data."""

    name = "fixed"
    lower_is_better = True

    def __init__(self, champ, chal):
        self._champ, self._chal = np.asarray(champ, dtype=float), np.asarray(chal, dtype=float)

    def pointwise(self, model, data):
        return self._champ if model == "champ" else self._chal

    def scalar(self, model, data):
        return float(np.mean(self.pointwise(model, data)))


class PointwiseRowBindingTest(unittest.TestCase):
    """MXR-080-1765: pointwise scores must be one-per-held-out-row, not merely mutually same-sized."""

    def test_a_score_matrix_is_not_four_paired_observations_for_a_hundred_rows(self):
        objective = _FixedVectorObjective([[2.0, 2.0], [2.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            challenger_beats_champion("champ", "chal", list(range(100)), objective=objective, require_calibration=False)

    def test_a_short_vector_cannot_stand_in_for_the_held_out_set(self):
        objective = _FixedVectorObjective([2.0, 2.0, 2.1, 2.2], [1.0, 1.1, 1.0, 1.2])
        with self.assertRaises(ValueError):
            challenger_beats_champion("champ", "chal", list(range(100)), objective=objective, require_calibration=False)

    def test_one_finite_score_per_row_is_accepted(self):
        rows = 40
        rng = np.random.RandomState(0)
        objective = _FixedVectorObjective(rng.normal(2.0, 0.1, rows), rng.normal(1.0, 0.1, rows))
        verdict = challenger_beats_champion(
            "champ", "chal", list(range(rows)), objective=objective, require_calibration=False
        )
        self.assertEqual(verdict.favored, "challenger")


class GatePolicyDomainTest(unittest.TestCase):
    """MXR-080-1766: policy knobs that authorize promotion must be inside their own domains."""

    def setUp(self):
        rows = 40
        rng = np.random.RandomState(1)
        self.data = list(range(rows))
        self.objective = _FixedVectorObjective(rng.normal(2.0, 0.1, rows), rng.normal(1.0, 0.1, rows))

    def _gate(self, **kwargs):
        return challenger_beats_champion(
            "champ", "chal", self.data, objective=self.objective, require_calibration=False, **kwargs
        )

    def test_an_alpha_outside_zero_one_cannot_promote(self):
        # alpha=2 promoted unconditionally: every valid p-value is below it.
        for bad in (2.0, 0.0, 1.0, -0.1, float("nan"), float("inf")):
            with self.subTest(alpha=repr(bad)), self.assertRaises(ValueError):
                self._gate(alpha=bad)

    def test_a_negative_or_nan_min_effect_is_rejected(self):
        for bad in (-1.0, float("nan"), float("inf")):
            with self.subTest(min_effect=repr(bad)), self.assertRaises(ValueError):
                self._gate(min_effect=bad)

    def test_a_negative_or_nan_calib_tol_is_rejected(self):
        for bad in (-1.0, float("nan"), float("inf")):
            with self.subTest(calib_tol=repr(bad)), self.assertRaises(ValueError):
                self._gate(calib_tol=bad)

    def test_ordinary_policy_values_still_work(self):
        self.assertEqual(self._gate(alpha=0.05, min_effect=0.0, calib_tol=1e-3).favored, "challenger")


if __name__ == "__main__":
    unittest.main()
