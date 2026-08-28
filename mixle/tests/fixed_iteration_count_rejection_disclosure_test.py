# T3-02: optimize(delta=None, max_its=N) is documented as "run a fixed iteration count", but the
# EM loop's monotone acceptance gate runs regardless of delta -- a rejected (non-improving or
# non-finite) step still ends the loop early, silently, even though delta=None opted out of the
# convergence-driven early stop. examples/production_example.py's own checkpointing walkthrough
# hits this: optimize(seqs, est, max_its=9, delta=None, ..., on_step=reg.checkpointer('run',
# every=3)) is documented to checkpoint at iterations 3, 6, 9, but silently checkpoints once.
#
# CI found the original version of this file's integration-level tests fragile: they mirrored
# examples/production_example.py's exact RNG draw sequence to reproduce a rejection at iteration 3,
# but "does step N's likelihood decrease by more than the 1e-12 acceptance tolerance" depends on
# the exact floating-point trajectory the platform's BLAS produces -- identical RNG draws do not
# guarantee an identical rejection point (or any rejection at all) across BLAS backends. The
# disclosure LOGIC itself (_warn_if_capped_unconverged) is now covered directly and
# deterministically below; the integration tests that exercise a real optimize() call check that
# the disclosure is self-consistent with whatever actually happened, rather than asserting one
# specific platform's iteration count as a hardcoded expectation.

import unittest
import warnings

import numpy as np

from mixle.inference import optimize
from mixle.inference.estimation import _FitTrace, _warn_if_capped_unconverged
from mixle.stats import GaussianEstimator, MixtureEstimator


def _production_example_mixture_seqs():
    # Mirrors examples/production_example.py's exact RNG draw sequence leading up to its
    # checkpointing section, so this reproduces the same scenario the finding evidenced there --
    # though, per the note above, whether a step is actually rejected on a given platform is not
    # itself asserted as a fixed outcome.
    rng = np.random.RandomState(0)
    rng.normal(3.0, 2.0, 4000)
    rng.normal(9.0, 2.0, 4000)
    rng.normal(3.0, 2.0, 500)
    rng.normal(9.0, 2.0, 500)
    return np.concatenate([rng.normal(-5, 1, 3000), rng.normal(5, 1, 3000)]).tolist()


class WarnIfCappedUnconvergedDirectTest(unittest.TestCase):
    """Deterministic, platform-independent coverage of the disclosure logic itself: construct the
    _FitTrace the EM loop would have produced in each case directly, rather than relying on real
    EM/BLAS dynamics to reproduce a specific one."""

    def test_requested_delta_none_with_a_shortfall_warns_with_the_true_counts(self):
        trace = _FitTrace()
        trace.iterations = 3
        trace.converged = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_capped_unconverged(trace, max_its=9, delta=None, requested_delta=None)
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("delta=None", notes[0])
        self.assertIn("iterations=3", notes[0])
        self.assertIn("max_iterations=9", notes[0])

    def test_requested_delta_none_that_used_the_full_budget_stays_quiet(self):
        trace = _FitTrace()
        trace.iterations = 9
        trace.converged = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_capped_unconverged(trace, max_its=9, delta=None, requested_delta=None)
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(notes, [])

    def test_a_real_requested_delta_with_a_shortfall_is_not_mistaken_for_the_delta_none_case(self):
        # requested_delta is not None here even though the LOOP's own delta (a surrogate estimator
        # can force this to None internally) might be -- the two must not be conflated.
        trace = _FitTrace()
        trace.iterations = 3
        trace.converged = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_capped_unconverged(trace, max_its=9, delta=None, requested_delta=1.0e-9)
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(notes, [])

    def test_a_real_requested_delta_hitting_the_cap_still_gets_the_pre_existing_warning(self):
        trace = _FitTrace()
        trace.iterations = 9
        trace.converged = False
        trace.objective_gain = 0.5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_capped_unconverged(trace, max_its=9, delta=1.0e-9, requested_delta=1.0e-9)
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("max_its cap", notes[0])
        self.assertNotIn("was called with delta=None", notes[0])


class FixedIterationCountRejectionDisclosureTest(unittest.TestCase):
    def test_delta_none_stopped_short_by_a_rejected_step_warns_and_says_so(self):
        seqs = _production_example_mixture_seqs()
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(seqs, est, max_its=9, delta=None, out=None, rng=np.random.RandomState(2))
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]

        provenance = model.fit_provenance()
        self.assertFalse(provenance.converged)
        self.assertIsNone(provenance.delta)
        if provenance.iterations >= provenance.max_iterations:
            self.skipTest(
                "this platform's EM trajectory did not reject a step on this data/seed -- the "
                "disclosure logic itself is covered deterministically by "
                "WarnIfCappedUnconvergedDirectTest above"
            )

        # The fix: exactly the ACTUAL shortfall this platform produced is disclosed, not silent --
        # not a specific hardcoded iteration count.
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("delta=None", notes[0])
        self.assertIn("iterations=%d" % provenance.iterations, notes[0])
        self.assertIn("max_iterations=%d" % provenance.max_iterations, notes[0])

    def test_on_step_checkpoints_stop_early_matching_the_disclosed_shortfall(self):
        # The concrete, user-visible symptom from production_example.py: a checkpoint callback
        # invoked every 3 iterations should fire once per 3 iterations actually run -- if a
        # rejection cuts the run short, on_step must stop exactly where fit_provenance() says it did
        # (not silently keep going, and not report more calls than iterations that actually ran).
        seqs = _production_example_mixture_seqs()
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        seen_iters = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = optimize(
                seqs,
                est,
                max_its=9,
                delta=None,
                out=None,
                rng=np.random.RandomState(2),
                on_step=lambda step: seen_iters.append(step.iter),
            )
        provenance = model.fit_provenance()
        self.assertEqual(seen_iters, list(range(1, provenance.iterations + 1)))
        if provenance.iterations >= provenance.max_iterations:
            self.skipTest("this platform's EM trajectory ran the full budget -- see the test above")
        self.assertEqual(seen_iters[-1], provenance.iterations)

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
