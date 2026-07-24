"""VerifiableOracle + optimize_under_oracle: the honesty boundary for de novo optimization (workstream
I1-I3). Proven here against a cheap closed-form oracle (a scored parameter vector), per the plan's own
build order, before any domain oracle exists.
"""

import threading
import time
import unittest
from unittest import mock

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.doe.oracle import (
    DesignCandidate,
    DesignRun,
    LateOracleResult,
    OracleResult,
    VerifiableOracle,
    optimize_under_oracle,
)


def _quadratic_bowl_oracle(target, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)

    def score_fn(x):
        d2 = float(np.sum((np.asarray(x, dtype=float) - target) ** 2))
        noisy = -d2 + (rng.normal(0, noise) if noise else 0.0)
        return OracleResult(score=noisy, receipt={"target_dist2": d2}, cost=1.0)

    return VerifiableOracle(name="quadratic_bowl", tier="executable", score_fn=score_fn, fidelity="exact, noiseless")


class VerifiableOracleConstructionTest(unittest.TestCase):
    def test_valid_tiers_construct(self):
        for tier in ("executable", "simulation", "held_out_truth", "real_measurement"):
            VerifiableOracle(name="ok", tier=tier, score_fn=lambda x: OracleResult(0.0))  # must not raise

    def test_self_graded_tier_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as ctx:
            VerifiableOracle(name="bad", tier="self_graded", score_fn=lambda x: OracleResult(0.0))
        self.assertIn("self_graded", str(ctx.exception))

    def test_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            VerifiableOracle(name="bad", tier="vibes", score_fn=lambda x: OracleResult(0.0))

    def test_oracle_is_callable(self):
        oracle = VerifiableOracle(name="ok", tier="executable", score_fn=lambda x: OracleResult(score=float(x[0])))
        result = oracle(np.array([3.0]))
        self.assertIsInstance(result, OracleResult)
        self.assertEqual(result.score, 3.0)


class NoOracleGuardTest(unittest.TestCase):
    def test_no_oracle_refuses_rather_than_fabricates(self):
        with self.assertRaises(ValueError) as ctx:
            optimize_under_oracle(None, [(-1.0, 1.0)])
        self.assertIn("no verifiable objective", str(ctx.exception))


class OracleTimeoutAbandonmentTest(unittest.TestCase):
    """FAULT-a ``oracle_timeout``: a genuinely-hung ``score_fn`` must abstain promptly rather than block
    the caller, and the worker thread abandoned to reach that abstention must be a daemon thread -- so
    a hung oracle call cannot keep the whole *process* alive after everything else is done, even though
    Python has no API to forcibly kill the thread itself. See ``VerifiableOracle._call_with_timeout``'s
    docstring for why this must be a raw ``threading.Thread`` and not a
    ``concurrent.futures.ThreadPoolExecutor`` (the latter's module-global exit hook joins its worker
    threads, untimed, regardless of daemon status, which reintroduces the exact hang this guards
    against).
    """

    def test_hung_score_fn_abstains_promptly_and_leaks_only_a_daemon_thread(self):
        hang_forever = threading.Event()  # deliberately never .set() -- score_fn blocks on this forever

        def never_returns(_candidate):
            hang_forever.wait()
            return OracleResult(score=999.0)  # unreachable

        oracle = VerifiableOracle(name="hung", tier="executable", score_fn=never_returns, timeout=0.2)

        threads_before = set(threading.enumerate())
        t0 = time.monotonic()
        result = oracle(np.array([1.0]))
        elapsed = time.monotonic() - t0
        threads_after = set(threading.enumerate())

        # the call returns promptly (close to the timeout budget), not after blocking on the hang
        self.assertLess(elapsed, 5.0)
        self.assertEqual(result.score, float("-inf"))
        self.assertEqual(result.cost, 0.0)
        self.assertEqual(result.receipt.get("degraded_mode"), "oracle_timeout")
        self.assertEqual(result.receipt.get("oracle_id"), "hung")

        # exactly one worker thread was abandoned (score_fn is genuinely hung and never returns) --
        # confirm it exists, and that it is a daemon thread so it cannot block process exit.
        leaked = threads_after - threads_before
        self.assertEqual(len(leaked), 1, f"expected exactly one leaked worker thread, got {leaked}")
        leaked_thread = leaked.pop()
        self.assertTrue(leaked_thread.is_alive())
        self.assertTrue(
            leaked_thread.daemon,
            "the abandoned oracle worker thread must be a daemon thread, or a hung oracle call would "
            "keep the whole process alive even after everything else has finished",
        )
        # not joined/waited on -- it is a daemon thread and the test process will exit past it fine.

    def test_score_fn_that_finishes_within_budget_is_unaffected_and_leaves_no_thread(self):
        def fast(x):
            return OracleResult(score=float(x[0]), receipt={"ok": True})

        oracle = VerifiableOracle(name="fast", tier="executable", score_fn=fast, timeout=5.0)
        threads_before = set(threading.enumerate())
        result = oracle(np.array([3.0]))
        threads_after = set(threading.enumerate())

        self.assertEqual(result.score, 3.0)
        self.assertEqual(result.receipt, {"ok": True})
        self.assertEqual(result.cost, 1.0)
        self.assertEqual(threads_after, threads_before)  # nothing leaked on the happy path

    def test_score_fn_exception_propagates_and_is_not_mistaken_for_a_timeout(self):
        def boom(_candidate):
            raise RuntimeError("score_fn blew up")

        oracle = VerifiableOracle(name="boom", tier="executable", score_fn=boom, timeout=5.0)
        with self.assertRaises(RuntimeError) as ctx:
            oracle(np.array([1.0]))
        self.assertIn("score_fn blew up", str(ctx.exception))


@unittest.skipUnless(_HAS_TORCH, "the BO proposal model needs torch")
class DesignLoopTest(unittest.TestCase):
    def test_finds_the_target_and_returns_a_receipted_run(self):
        target = np.array([0.5, -1.0])
        oracle = _quadratic_bowl_oracle(target)
        run = optimize_under_oracle(
            oracle,
            [(-3.0, 3.0), (-3.0, 3.0)],
            n_init=6,
            n_iter=20,
            seed=0,
            n_candidates=256,
            fit_kwargs={"max_its": 60},
        )
        self.assertIsInstance(run, DesignRun)
        self.assertEqual(run.oracle_calls, 26)
        self.assertLess(run.best.result.receipt["target_dist2"], 0.5)  # genuinely converged, not a lucky init draw

    def test_report_names_the_oracle_identity_and_fidelity(self):
        oracle = _quadratic_bowl_oracle(np.array([1.0, 1.0]))
        run = optimize_under_oracle(
            oracle, [(-3.0, 3.0)] * 2, n_init=6, n_iter=10, seed=1, n_candidates=256, fit_kwargs={"max_its": 60}
        )
        rep = run.report()
        self.assertEqual(rep["oracle"], "quadratic_bowl")
        self.assertEqual(rep["tier"], "executable")
        self.assertEqual(rep["fidelity"], "exact, noiseless")
        self.assertEqual(rep["oracle_calls"], 16)
        self.assertIn("target_dist2", rep["best_receipt"])

    def test_beats_random_search_at_matched_oracle_call_budget(self):
        """The I acceptance: on a known closed-form oracle, the loop beats random search at matched budget."""
        target = np.array([0.5, -1.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        oracle = _quadratic_bowl_oracle(target)
        budget = 26  # n_init=6 + n_iter=20, matching the loop's own call count

        run = optimize_under_oracle(
            oracle, bounds, n_init=6, n_iter=20, seed=0, n_candidates=256, fit_kwargs={"max_its": 60}
        )
        bo_dist2 = run.best.result.receipt["target_dist2"]

        rng = np.random.RandomState(0)
        random_pts = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(budget, 2))
        random_best = min(float(np.sum((p - target) ** 2)) for p in random_pts)

        self.assertLess(bo_dist2, random_best)  # acquisition-driven search beats matched-budget random search

    def test_negative_result_is_kept_in_the_history_not_hidden(self):
        """A candidate the surrogate visited but that scored poorly stays in the run log."""
        oracle = _quadratic_bowl_oracle(np.array([0.0, 0.0]))
        run = optimize_under_oracle(
            oracle, [(-3.0, 3.0)] * 2, n_init=6, n_iter=10, seed=2, n_candidates=256, fit_kwargs={"max_its": 60}
        )
        scores = run.scores()
        self.assertEqual(len(scores), run.oracle_calls)
        self.assertGreater(
            scores.max() - scores.min(), 0.0
        )  # some candidates were verifiably worse, and are still there


class OracleResultValidationTest(unittest.TestCase):
    """MXR-080-0189: OracleResult validates its own shape at construction -- the boundary a score_fn's
    raw, untrusted return value crosses, so a wrong type or an out-of-range value is caught immediately
    rather than silently reaching a GP fit or a cost report.
    """

    def test_valid_result_constructs(self):
        r = OracleResult(score=1.5, receipt={"k": "v"}, cost=2.0)
        self.assertEqual(r.score, 1.5)
        self.assertFalse(r.abstained)

    def test_zero_cost_is_allowed(self):
        OracleResult(score=1.0, cost=0.0)  # must not raise -- zero is a valid, common cost

    def test_non_numeric_score_is_rejected(self):
        for bad in ("nope", None, [1.0], {"x": 1}):
            with self.assertRaises(TypeError):
                OracleResult(score=bad)

    def test_bool_score_is_rejected(self):
        # bool is an int subclass in Python but is never a meaningful score
        with self.assertRaises(TypeError):
            OracleResult(score=True)

    def test_non_finite_score_is_rejected_unless_abstained(self):
        with self.assertRaises(ValueError):
            OracleResult(score=float("inf"))
        with self.assertRaises(ValueError):
            OracleResult(score=float("-inf"))  # abstained defaults to False
        with self.assertRaises(ValueError):
            OracleResult(score=float("nan"))

    def test_non_finite_score_is_accepted_when_abstained(self):
        r = OracleResult(score=float("-inf"), abstained=True)
        self.assertTrue(r.abstained)
        self.assertEqual(r.score, float("-inf"))

    def test_negative_cost_is_rejected(self):
        with self.assertRaises(ValueError):
            OracleResult(score=1.0, cost=-0.01)

    def test_non_finite_cost_is_rejected(self):
        with self.assertRaises(ValueError):
            OracleResult(score=1.0, cost=float("inf"))
        with self.assertRaises(ValueError):
            OracleResult(score=1.0, cost=float("nan"))

    def test_non_numeric_cost_is_rejected(self):
        with self.assertRaises(TypeError):
            OracleResult(score=1.0, cost="free")

    def test_non_dict_receipt_is_rejected(self):
        with self.assertRaises(TypeError):
            OracleResult(score=1.0, receipt=["not", "a", "dict"])
        with self.assertRaises(TypeError):
            OracleResult(score=1.0, receipt="also not a dict")


class VerifiableOracleTimeoutValidationTest(unittest.TestCase):
    """MXR-080-0189: the timeout value itself is validated at construction."""

    def test_none_timeout_is_allowed(self):
        VerifiableOracle(name="ok", tier="executable", score_fn=lambda x: OracleResult(0.0), timeout=None)

    def test_positive_timeout_is_allowed(self):
        VerifiableOracle(name="ok", tier="executable", score_fn=lambda x: OracleResult(0.0), timeout=1.5)

    def test_zero_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: OracleResult(0.0), timeout=0.0)

    def test_negative_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: OracleResult(0.0), timeout=-1.0)

    def test_non_finite_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: OracleResult(0.0), timeout=float("inf"))
        with self.assertRaises(ValueError):
            VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: OracleResult(0.0), timeout=float("nan"))

    def test_non_numeric_timeout_is_rejected(self):
        with self.assertRaises(TypeError):
            VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: OracleResult(0.0), timeout="soon")


class VerifiableOracleCandidateValidationTest(unittest.TestCase):
    def test_none_candidate_is_rejected(self):
        oracle = VerifiableOracle(name="ok", tier="executable", score_fn=lambda x: OracleResult(score=float(x[0])))
        with self.assertRaises(ValueError) as ctx:
            oracle(None)
        self.assertIn("None", str(ctx.exception))

    def test_none_candidate_is_rejected_under_a_timeout_too(self):
        oracle = VerifiableOracle(
            name="ok", tier="executable", score_fn=lambda x: OracleResult(score=float(x[0])), timeout=5.0
        )
        with self.assertRaises(ValueError):
            oracle(None)


class VerifiableOracleReturnTypeValidationTest(unittest.TestCase):
    """MXR-080-0189: score_fn's raw return value is validated at the __call__ boundary -- a malformed
    return is a hard error, never silently treated as a score or fabricated into an OracleResult.
    """

    def test_non_oracleresult_return_is_rejected(self):
        oracle = VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: 3.0)  # a raw float
        with self.assertRaises(TypeError) as ctx:
            oracle(np.array([1.0]))
        self.assertIn("OracleResult", str(ctx.exception))

    def test_none_return_is_rejected(self):
        oracle = VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: None)
        with self.assertRaises(TypeError):
            oracle(np.array([1.0]))

    def test_dict_return_is_rejected(self):
        oracle = VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: {"score": 3.0})
        with self.assertRaises(TypeError):
            oracle(np.array([1.0]))

    def test_malformed_return_under_a_timeout_budget_is_a_hard_error_not_a_timeout(self):
        """A malformed return that arrives well within budget must surface as a TypeError, not be
        mistaken for (or silently reinterpreted as) an oracle_timeout abstention."""
        oracle = VerifiableOracle(name="bad", tier="executable", score_fn=lambda x: "not a result", timeout=5.0)
        with self.assertRaises(TypeError) as ctx:
            oracle(np.array([1.0]))
        self.assertIn("OracleResult", str(ctx.exception))


class _RecordingBOStub:
    """Minimal ask/tell stand-in for optimize_under_oracle's loop, isolated from the real
    BayesianOptimizer/torch (torch is not installed in this environment). The fix under test lives
    entirely in optimize_under_oracle's own loop logic -- does it call tell() with an abstained score?
    -- not in the GP surrogate itself. Mirrors BayesianOptimizer.ask()'s documented contract that ask()
    may be called repeatedly without requiring an intervening tell() (its init-design dispensing is
    gated on points asked, not points told), so skipping tell() for an abstention is safe to model here
    exactly as it is against the real optimizer.
    """

    def __init__(self, bounds, **kwargs):
        self.bounds = bounds
        self.told_x: list[np.ndarray] = []
        self.told_y: list[float] = []
        self._i = 0

    def ask(self, q: int = 1):
        self._i += 1
        return np.array([float(self._i)])

    def tell(self, x, y):
        self.told_x.append(np.asarray(x))
        self.told_y.append(float(np.asarray(y).reshape(-1)[0]))
        return self


class OracleAbstentionNeverTrainedAsTruthTest(unittest.TestCase):
    """MXR-080-0189's central fix: optimize_under_oracle must never feed an abstained (e.g. timed-out)
    result to the proposal model's tell() as if it were a genuine observation, even though the
    abstention is still kept in the receipted run.history. Uses _RecordingBOStub in place of
    BayesianOptimizer (torch is unavailable in this environment, and the bug/fix under test is entirely
    in optimize_under_oracle's own loop, not the GP surrogate) via optimize_under_oracle's local
    ``from mixle.doe.optimizer import BayesianOptimizer`` import, which re-resolves at call time and so
    picks up the patched stand-in.
    """

    def test_an_always_timing_out_oracle_is_never_told_as_a_real_observation(self):
        def always_hangs(_candidate):
            time.sleep(2.0)
            return OracleResult(score=999.0)  # unreachable within budget

        oracle = VerifiableOracle(name="always_hangs", tier="executable", score_fn=always_hangs, timeout=0.05)
        with mock.patch("mixle.doe.optimizer.BayesianOptimizer", _RecordingBOStub):
            run = optimize_under_oracle(oracle, [(-10.0, 10.0)], n_init=4, n_iter=0, seed=0)

        # every attempt is receipted (transparency: kept, not hidden)...
        self.assertEqual(run.oracle_calls, 4)
        self.assertTrue(all(c.result.abstained for c in run.history))
        self.assertEqual(run.scores().tolist(), [float("-inf")] * 4)
        # ...but none of them was ever fed to the fit as ground truth.
        self.assertEqual(run.genuine_history, [])

    def test_opt_tell_is_never_called_with_a_negative_infinite_score(self):
        """Direct assertion on the actual bug mechanism: inspect what optimize_under_oracle told the
        proposal model, not just the run's own bookkeeping. Captures the stub instance
        optimize_under_oracle constructs internally via a subclass that stashes itself on init, since
        the function does not expose its BayesianOptimizer instance directly."""

        def always_hangs(_candidate):
            time.sleep(2.0)
            return OracleResult(score=999.0)

        oracle = VerifiableOracle(name="always_hangs", tier="executable", score_fn=always_hangs, timeout=0.05)
        captured = {}

        class _Spy(_RecordingBOStub):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                captured["stub"] = self

        with mock.patch("mixle.doe.optimizer.BayesianOptimizer", _Spy):
            run = optimize_under_oracle(oracle, [(-10.0, 10.0)], n_init=4, n_iter=0, seed=0)

        self.assertEqual(run.oracle_calls, 4)  # the budget was still spent -- just never trained on
        self.assertEqual(captured["stub"].told_y, [])  # zero tell() calls -- not even one -inf slipped through

    def test_a_genuine_non_abstained_run_is_told_every_observation_as_before(self):
        """Negative control: without any timeout, every candidate is genuine and every one is told --
        the fix must not withhold real observations from the fit."""

        def instant(x):
            return OracleResult(score=float(x[0]), receipt={"ok": True})

        oracle = VerifiableOracle(name="instant", tier="executable", score_fn=instant)  # no timeout at all
        captured = {}

        class _Spy(_RecordingBOStub):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                captured["stub"] = self

        with mock.patch("mixle.doe.optimizer.BayesianOptimizer", _Spy):
            run = optimize_under_oracle(oracle, [(-10.0, 10.0)], n_init=5, n_iter=0, seed=0)

        self.assertEqual(run.oracle_calls, 5)
        self.assertTrue(all(not c.result.abstained for c in run.history))
        self.assertEqual(len(run.genuine_history), 5)
        self.assertEqual(len(captured["stub"].told_y), 5)  # every genuine observation was told
        self.assertTrue(all(np.isfinite(y) for y in captured["stub"].told_y))

    def test_a_mixed_run_trains_only_on_the_genuine_observations(self):
        """Some calls time out, some do not -- the genuine ones must still be told, and only those."""
        calls = {"n": 0}

        def every_third_call_hangs(_candidate):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                time.sleep(2.0)
            return OracleResult(score=10.0 + calls["n"])

        oracle = VerifiableOracle(name="flaky", tier="executable", score_fn=every_third_call_hangs, timeout=0.05)
        captured = {}

        class _Spy(_RecordingBOStub):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                captured["stub"] = self

        with mock.patch("mixle.doe.optimizer.BayesianOptimizer", _Spy):
            run = optimize_under_oracle(oracle, [(-10.0, 10.0)], n_init=9, n_iter=0, seed=0)

        self.assertEqual(run.oracle_calls, 9)
        self.assertEqual(sum(1 for c in run.history if c.result.abstained), 3)  # calls 3, 6, 9 hung
        told_y = captured["stub"].told_y
        self.assertEqual(len(told_y), 6)  # only the 6 genuine calls were told
        self.assertTrue(all(np.isfinite(y) for y in told_y))
        self.assertNotIn(float("-inf"), told_y)


class DesignRunAbstentionBookkeepingTest(unittest.TestCase):
    """MXR-080-0189: DesignRun.best/report() must never surface an abstention as a genuine result,
    while scores()/history keep the full, unfiltered receipted record (kept, not hidden)."""

    @staticmethod
    def _run(history_specs, oracle_name="test", tier="executable", fidelity=None):
        run = DesignRun(oracle_name=oracle_name, oracle_tier=tier, oracle_fidelity=fidelity)
        for x_val, score, abstained in history_specs:
            result = (
                OracleResult(score=float("-inf"), abstained=True, cost=0.0)
                if abstained
                else OracleResult(score=score, cost=1.0)
            )
            run.history.append(DesignCandidate(x=np.array([x_val]), result=result))
        return run

    def test_best_skips_abstained_candidates_even_when_they_would_not_win_anyway(self):
        run = self._run([(0.0, 1.0, False), (1.0, None, True), (2.0, 5.0, False)])
        self.assertEqual(run.best.x.tolist(), [2.0])
        self.assertEqual(run.best.result.score, 5.0)

    def test_best_raises_a_specific_error_when_every_candidate_abstained(self):
        run = self._run([(0.0, None, True), (1.0, None, True)])
        with self.assertRaises(ValueError) as ctx:
            _ = run.best
        self.assertIn("abstained", str(ctx.exception))

    def test_best_raises_the_original_empty_history_error_when_history_is_empty(self):
        run = DesignRun(oracle_name="test", oracle_tier="executable", oracle_fidelity=None)
        with self.assertRaises(ValueError) as ctx:
            _ = run.best
        self.assertIn("empty", str(ctx.exception))

    def test_genuine_history_excludes_abstentions_full_history_does_not(self):
        run = self._run([(0.0, 1.0, False), (1.0, None, True)])
        self.assertEqual(len(run.history), 2)
        self.assertEqual(len(run.genuine_history), 1)
        self.assertEqual(run.genuine_history[0].x.tolist(), [0.0])

    def test_report_counts_abstained_calls_separately_from_oracle_calls(self):
        run = self._run([(0.0, 1.0, False), (1.0, None, True), (2.0, 3.0, False)])
        rep = run.report()
        self.assertEqual(rep["oracle_calls"], 3)
        self.assertEqual(rep["abstained_calls"], 1)
        self.assertEqual(rep["best_score"], 3.0)  # the abstention never wins "best"

    def test_scores_keeps_abstentions_visible_not_hidden(self):
        """Matches the file's existing 'kept in history, not hidden' convention -- scores() is the
        full, unfiltered receipted record, unlike best()/genuine_history."""
        run = self._run([(0.0, 1.0, False), (1.0, None, True)])
        self.assertEqual(run.scores().tolist(), [1.0, float("-inf")])


class TimeoutAbstentionReceiptTest(unittest.TestCase):
    """MXR-080-0189: the abstention receipt documents whether cancellation was requested and whether
    this particular score_fn even has a chance of honoring it -- explicit, not silently unaccounted."""

    def test_abstention_receipt_documents_cancellation_and_lack_of_cooperative_support(self):
        hang_forever = threading.Event()

        def hangs(_c):
            hang_forever.wait()
            return OracleResult(score=1.0)  # unreachable

        oracle = VerifiableOracle(name="hangs", tier="executable", score_fn=hangs, timeout=0.05)
        result = oracle(np.array([1.0]))

        self.assertTrue(result.abstained)
        self.assertIs(result.receipt["cancel_requested"], True)
        self.assertIs(result.receipt["cooperative_cancel_supported"], False)  # `hangs` takes 1 arg

    def test_abstention_receipt_reports_cooperative_support_when_score_fn_accepts_cancel_event(self):
        hang_forever = threading.Event()

        def cooperative(_c, cancel_event=None):
            hang_forever.wait()
            return OracleResult(score=1.0)  # unreachable

        oracle = VerifiableOracle(name="cooperative", tier="executable", score_fn=cooperative, timeout=0.05)
        result = oracle(np.array([1.0]))

        self.assertIs(result.receipt["cooperative_cancel_supported"], True)


class CooperativeCancellationTest(unittest.TestCase):
    """MXR-080-0189's cancellable-execution-boundary requirement: a score_fn that opts in by accepting
    cancel_event observes it get set once the caller times out (best-effort, cooperative -- Python
    cannot forcibly kill the underlying thread, see VerifiableOracle's docstring)."""

    def test_cancel_event_is_set_on_timeout_and_observed_by_a_cooperative_score_fn(self):
        observed = threading.Event()  # set by the worker once IT sees cancel_event get set

        def cooperative(_candidate, cancel_event):
            self.assertFalse(cancel_event.is_set())  # not set yet -- score_fn is still within budget
            if cancel_event.wait(timeout=2.0):  # blocks until the caller times out and cancels
                observed.set()
            return OracleResult(score=1.0)

        oracle = VerifiableOracle(name="cooperative", tier="executable", score_fn=cooperative, timeout=0.05)
        self.assertTrue(oracle._accepts_cancel_event)

        result = oracle(np.array([1.0]))

        self.assertTrue(result.abstained)  # the caller still abstains promptly regardless
        self.assertTrue(observed.wait(timeout=2.0), "the cooperative score_fn never observed cancel_event get set")

    def test_a_score_fn_that_does_not_accept_cancel_event_is_unaffected(self):
        """Every pre-existing caller's score_fn shape (single positional arg) must keep working
        unchanged -- cancel_event is strictly opt-in, detected via introspection, never forced."""
        hang_forever = threading.Event()

        def plain(_candidate):
            hang_forever.wait()
            return OracleResult(score=1.0)  # unreachable

        oracle = VerifiableOracle(name="plain", tier="executable", score_fn=plain, timeout=0.05)
        self.assertFalse(oracle._accepts_cancel_event)
        result = oracle(np.array([1.0]))
        self.assertTrue(result.abstained)

    def test_cancel_event_via_kwargs_is_also_detected(self):
        def cooperative_kwargs(_candidate, **kwargs):
            kwargs["cancel_event"].wait(timeout=2.0)
            return OracleResult(score=1.0)  # unreachable

        oracle = VerifiableOracle(name="kwargs", tier="executable", score_fn=cooperative_kwargs, timeout=0.05)
        self.assertTrue(oracle._accepts_cancel_event)


class LateResultAccountingTest(unittest.TestCase):
    """MXR-080-0189: an abandoned oracle call's eventual outcome is captured in late_results rather
    than silently lost, even though Python cannot forcibly cancel the underlying thread."""

    @staticmethod
    def _wait_until(predicate, timeout=3.0, interval=0.02):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return predicate()

    def test_a_call_that_finishes_on_time_never_appears_in_late_results(self):
        oracle = VerifiableOracle(
            name="fast", tier="executable", score_fn=lambda x: OracleResult(score=1.0), timeout=5.0
        )
        oracle(np.array([1.0]))
        time.sleep(0.1)
        self.assertEqual(oracle.late_results, [])

    def test_a_late_success_is_recorded_with_its_real_cost(self):
        def finishes_late(_candidate):
            time.sleep(0.2)  # comfortably past the oracle's 0.05s timeout
            return OracleResult(score=42.0, receipt={"late": True}, cost=7.5)

        oracle = VerifiableOracle(name="late", tier="executable", score_fn=finishes_late, timeout=0.05)
        candidate = np.array([3.0])
        result = oracle(candidate)
        self.assertTrue(result.abstained)  # the caller still gets a prompt abstention

        self.assertTrue(self._wait_until(lambda: len(oracle.late_results) == 1))
        late = oracle.late_results[0]
        self.assertIsInstance(late, LateOracleResult)
        self.assertTrue(late.ok)
        self.assertIsNone(late.error)
        self.assertEqual(late.result.score, 42.0)
        self.assertEqual(late.result.cost, 7.5)
        self.assertEqual(late.result.receipt, {"late": True})
        np.testing.assert_array_equal(late.candidate, candidate)

    def test_a_late_exception_is_recorded_not_raised_into_the_void(self):
        def fails_late(_candidate):
            time.sleep(0.2)
            raise RuntimeError("late external failure")

        oracle = VerifiableOracle(name="late_fail", tier="executable", score_fn=fails_late, timeout=0.05)
        oracle(np.array([1.0]))

        self.assertTrue(self._wait_until(lambda: len(oracle.late_results) == 1))
        late = oracle.late_results[0]
        self.assertFalse(late.ok)
        self.assertIsNone(late.result)
        self.assertIsInstance(late.error, RuntimeError)
        self.assertIn("late external failure", str(late.error))

    def test_a_late_malformed_return_is_recorded_as_a_validation_error(self):
        def returns_garbage_late(_candidate):
            time.sleep(0.2)
            return "not an OracleResult"

        oracle = VerifiableOracle(name="late_bad", tier="executable", score_fn=returns_garbage_late, timeout=0.05)
        oracle(np.array([1.0]))

        self.assertTrue(self._wait_until(lambda: len(oracle.late_results) == 1))
        late = oracle.late_results[0]
        self.assertFalse(late.ok)
        self.assertIsInstance(late.error, TypeError)


if __name__ == "__main__":
    unittest.main()
