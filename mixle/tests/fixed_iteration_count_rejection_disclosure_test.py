# T3-02: optimize(delta=None, max_its=N) is documented as "run a fixed iteration count", but the
# EM loop's monotone acceptance gate runs regardless of delta -- a rejected (non-improving or
# non-finite) step still ends the loop early, silently, even though delta=None opted out of the
# convergence-driven early stop. examples/production_example.py's own checkpointing walkthrough
# hits this: optimize(seqs, est, max_its=9, delta=None, ..., on_step=reg.checkpointer('run',
# every=3)) is documented to checkpoint at iterations 3, 6, 9, but silently checkpoints once.

import unittest
import warnings

import numpy as np

from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator


def _production_example_mixture_seqs():
    # Mirrors examples/production_example.py's exact RNG draw sequence leading up to its
    # checkpointing section, so this reproduces the same early-stop the finding evidenced there.
    rng = np.random.RandomState(0)
    rng.normal(3.0, 2.0, 4000)
    rng.normal(9.0, 2.0, 4000)
    rng.normal(3.0, 2.0, 500)
    rng.normal(9.0, 2.0, 500)
    return np.concatenate([rng.normal(-5, 1, 3000), rng.normal(5, 1, 3000)]).tolist()


class FixedIterationCountRejectionDisclosureTest(unittest.TestCase):
    def test_delta_none_stopped_short_by_a_rejected_step_warns_and_says_so(self):
        seqs = _production_example_mixture_seqs()
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(seqs, est, max_its=9, delta=None, out=None, rng=np.random.RandomState(2))
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]

        provenance = model.fit_provenance()
        # The defect: fewer than max_its iterations actually ran despite delta=None.
        self.assertLess(provenance.iterations, provenance.max_iterations)
        self.assertFalse(provenance.converged)
        self.assertIsNone(provenance.delta)

        # The fix: exactly that shortfall is now disclosed, not silent.
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("delta=None", notes[0])
        self.assertIn(str(provenance.iterations), notes[0])
        self.assertIn(str(provenance.max_iterations), notes[0])

    def test_on_step_checkpoints_stop_early_matching_the_disclosed_shortfall(self):
        # The concrete, user-visible symptom from production_example.py: a checkpoint callback
        # invoked every 3 iterations over max_its=9 should fire 3 times (iterations 3, 6, 9); the
        # pre-fix behavior fires it once because the loop exits after iteration 3.
        seqs = _production_example_mixture_seqs()
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        seen_iters = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # this test only cares about on_step, not the warning
            optimize(
                seqs,
                est,
                max_its=9,
                delta=None,
                out=None,
                rng=np.random.RandomState(2),
                on_step=lambda step: seen_iters.append(step.iter),
            )
        # Documents the current (pre-existing, unfixed-by-design per the finding's guidance) loop
        # behavior: this is exactly the shortfall the new warning discloses, not a regression this
        # test introduces.
        self.assertEqual(seen_iters, [1, 2, 3])

    def test_an_ordinary_full_budget_delta_none_run_stays_quiet(self):
        # Regression guard for over-scoping: well-separated clusters accept every EM step, so a
        # delta=None run should use its whole max_its budget and the loop shortfall warning must
        # NOT fire on this ordinary, non-degenerate fit.
        rng = np.random.default_rng(1000)
        data = np.r_[rng.normal(0.0, 1.0, 300), rng.normal(6.0, 1.0, 300)].tolist()
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(data, est, max_its=3, delta=None, out=None, rng=np.random.RandomState(0))
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]

        provenance = model.fit_provenance()
        self.assertEqual(provenance.iterations, provenance.max_iterations)
        self.assertEqual(notes, [])

    def test_a_real_delta_hitting_the_cap_still_warns_as_before(self):
        # Regression guard: the pre-existing max_its-cap-unconverged warning (a DIFFERENT case,
        # requested_delta is not None) must still fire unchanged by this fix.
        rng = np.random.default_rng(11)
        data = np.r_[rng.normal(0, 1, 400), rng.normal(2.5, 1, 400), rng.normal(5, 1, 400)].tolist()
        est = MixtureEstimator([GaussianEstimator() for _ in range(3)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(data, est)  # default delta, default max_its=10
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]

        provenance = model.fit_provenance()
        self.assertFalse(provenance.converged)
        self.assertEqual(provenance.iterations, provenance.max_iterations)
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("max_its cap", notes[0])
        # The new "requested_delta is None" warning's own trigger phrase must not have fired here --
        # a real delta was in force, so this must be the pre-existing cap warning alone (which
        # separately mentions "delta=None" only as its own suggested remedy).
        self.assertNotIn("was called with delta=None", notes[0])


if __name__ == "__main__":
    unittest.main()
