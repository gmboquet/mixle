"""The champion/challenger verify gate (mixle.evolve.verify)."""

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


if __name__ == "__main__":
    unittest.main()
