"""Regression coverage for a thread-liveness race in ``SimulatedRank.join()``: ``join()`` must not
report a step "done" while the worker thread that ran it is still alive.

``_done`` (a ``threading.Event``) is set from INSIDE the worker thread's ``finally`` block, strictly
BEFORE that thread actually returns and CPython marks it not-alive. Under ordinary CPU contention that
gap can widen enough for a caller who only waits on ``_done`` to see ``join()`` "succeed" while
``self._thread.is_alive()`` still reads ``True``. ``ElasticTrainingJob.run_step()`` calls
``start_step()`` again on that same rank object at the top of the very next step; its liveness guard
(``if self._thread is not None and self._thread.is_alive(): raise RuntimeError(...)``) then fires
spuriously -- with ``kill_ranks`` empty throughout -- killing an otherwise perfectly healthy job. The
fix makes ``join()`` additionally join the real ``Thread`` object (against the remaining time budget)
before it can report success, so it cannot return while the thread is still alive.

The bug reproduces "in the wild" only under real, timing-dependent CPU contention (thousands of steps,
tens of concurrent processes) -- far too slow and flaky for a unit test. This file instead simulates
the exact CPython scheduling gap directly and deterministically: ``SimulatedRank``'s own post-backward
rendezvous (``started.set(); go.wait()``) already pins "the worker has finished its real work and is
about to signal completion" to a known, controllable point, so a small monkeypatch on that specific
rank's ``_done.set`` -- call the real ``set()`` (so any waiter sees the flag immediately, exactly like
production), then sleep briefly before returning (so the OS thread does not actually finish, and
``is_alive()`` does not go false, until after that sleep) -- reproduces the race on demand, with no
dependency on real machine load or timing luck.
"""

from __future__ import annotations

import itertools
import time
import unittest
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from mixle.utils.parallel.fault_tolerant_training import SimulatedRank, StepResult

IN_FEATURES = 4
NUM_CLASSES = 3
BATCH_SIZE = 2

# How long the worker thread "lingers" (real time) after signaling `_done`, before it actually returns
# and CPython marks it not-alive -- the simulated CPython scheduling gap. Comfortably larger than the
# "too short to finish" timeouts used below, and comfortably smaller than the generous "let it finish
# for real" timeouts, so the tests read correctly regardless of ordinary system load (independently
# verified against 8-way simulated CPU contention while drafting this file).
LINGER_S = 0.2


class _TinyLinear(torch.nn.Module):
    """The smallest real ``nn.Module`` that can complete a genuine forward+backward+clip pass through
    ``SimulatedRank._run()``. Deliberately not the project's causal LM (see
    ``fault_tolerant_training_test.py``): these tests only need ``start_step()`` to reach its
    post-backward rendezvous quickly and repeatedly -- the race under test lives entirely AFTER that
    point, inside ``join()``'s handling of ``_done`` versus the thread's actual liveness.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(IN_FEATURES, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _canonical_state_dict() -> dict[str, torch.Tensor]:
    return _TinyLinear().state_dict()


def _batch_fn() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return x, y


def _make_rank(rank_id: int = 0) -> SimulatedRank:
    return SimulatedRank(rank_id, _TinyLinear, _batch_fn)


def _run_rank_to_rendezvous(rank: SimulatedRank, step: int) -> None:
    """Start a step and block until the worker has finished its forward+backward and is parked at the
    rendezvous (``go.wait()``) -- the same deterministic "mid-step" point ``ElasticTrainingJob``'s own
    chaos test relies on (see the class docstring on ``SimulatedRank``)."""
    rank.start_step(step, _canonical_state_dict())
    assert rank.wait_started(timeout=5.0), "worker did not reach the post-backward rendezvous in time"


def _install_lingering_done(rank: SimulatedRank, linger_s: float = LINGER_S) -> None:
    """Patch THIS rank's current ``_done`` Event so that when the worker's ``finally`` block sets it,
    the flag becomes visible to any waiter immediately -- exactly like production -- but the worker
    thread itself does not return (and therefore ``is_alive()`` does not go false) until ``linger_s``
    later. Must be called only after the worker is confirmed parked at the rendezvous (see
    ``_run_rank_to_rendezvous``), so there is no race between installing this patch and the worker
    reaching ``finally: done.set()`` on its own.
    """
    done = rank._done
    real_set = done.set

    def _set_then_linger() -> None:
        real_set()
        time.sleep(linger_s)

    done.set = _set_then_linger


def _pre_fix_join(self: SimulatedRank, timeout: float = 5.0) -> StepResult | None:
    """Faithful reproduction of ``SimulatedRank.join()``'s entire body before this fix: waits only on
    ``_done`` and returns -- it never confirms the worker's OS thread has actually finished. Kept as a
    standalone function (rather than trusting a prose description of "the old behavior") so the
    contrast below is against the literal pre-fix algorithm.
    """
    if not self._done.wait(timeout=max(0.0, timeout)):
        raise TimeoutError(f"rank {self.rank_id} did not finish before the step deadline")
    if self._error is not None:
        raise RuntimeError(f"rank {self.rank_id} step {self._error!r} failed") from self._error
    return self._result


class ThreadLivenessRaceReproductionTest(unittest.TestCase):
    """The core regression: the pre-fix ``join()`` algorithm lets a dead-but-not-yet-reaped thread pass
    as "done", spuriously tripping ``start_step()``'s liveness guard on the very next step; the fixed
    (current) ``join()`` does not."""

    def test_pre_fix_join_reports_success_while_thread_alive_then_start_step_raises_spuriously(self) -> None:
        rank = _make_rank()
        _run_rank_to_rendezvous(rank, step=0)
        _install_lingering_done(rank)
        rank.release()

        with mock.patch.object(SimulatedRank, "join", _pre_fix_join):
            result = rank.join(timeout=5.0)

        # The pre-fix join() call "succeeded" ...
        self.assertIsInstance(result, StepResult)
        self.assertEqual(result.step, 0)
        # ... yet the worker's OS thread genuinely has not finished yet. This is the race: a caller
        # that only trusts `_done` cannot tell the difference between this and a truly-finished step.
        self.assertTrue(rank._thread.is_alive())

        # ElasticTrainingJob.run_step() calls start_step() again on exactly this object at the top of
        # the next step; its liveness guard fires on the still-alive thread, with kill_ranks empty
        # throughout -- spuriously killing an otherwise perfectly healthy rank.
        with self.assertRaisesRegex(RuntimeError, "still has a step in flight"):
            rank.start_step(1, _canonical_state_dict())

        # Cleanup: let the real thread finish for real before the test ends.
        rank._thread.join(timeout=LINGER_S + 5.0)
        self.assertFalse(rank._thread.is_alive())

    def test_fixed_join_waits_for_thread_death_so_the_next_start_step_succeeds(self) -> None:
        rank = _make_rank()
        _run_rank_to_rendezvous(rank, step=0)
        _install_lingering_done(rank)
        rank.release()

        result = rank.join(timeout=5.0)  # the real, current (fixed) join()

        self.assertIsInstance(result, StepResult)
        self.assertEqual(result.step, 0)
        # The fix's entire point: join() cannot report success while the thread is still alive.
        self.assertFalse(rank._thread.is_alive())

        # The rank is genuinely reusable for the next step -- no spurious "still has a step in flight".
        _run_rank_to_rendezvous(rank, step=1)
        rank.release()
        second_result = rank.join(timeout=5.0)
        self.assertIsInstance(second_result, StepResult)
        self.assertEqual(second_result.step, 1)


class JoinDeadlineTimeoutTest(unittest.TestCase):
    """The fix must not turn a genuinely-too-short deadline into either a hang or a false success, and
    must handle its two edge inputs (a zero timeout; a rank whose worker never started) sanely."""

    def test_join_raises_timeout_error_rather_than_hanging_or_falsely_succeeding(self) -> None:
        rank = _make_rank()
        _run_rank_to_rendezvous(rank, step=0)
        _install_lingering_done(rank)
        rank.release()

        too_short = LINGER_S / 4  # generous margin: the thread cannot possibly die this fast
        with self.assertRaisesRegex(TimeoutError, "did not finish before the step deadline"):
            rank.join(timeout=too_short)
        self.assertTrue(rank._thread.is_alive())  # correctly did NOT report a false success

        rank._thread.join(timeout=LINGER_S + 5.0)
        self.assertFalse(rank._thread.is_alive())

    def test_join_timeout_zero_is_a_non_blocking_check(self) -> None:
        # Not done yet -> immediate TimeoutError; no artificial delay involved.
        rank = _make_rank()
        _run_rank_to_rendezvous(rank, step=0)
        with self.assertRaises(TimeoutError):
            rank.join(timeout=0.0)
        rank.release()
        rank.join(timeout=5.0)  # let it finish cleanly

        # Already fully (really) finished -> succeeds immediately.
        rank2 = _make_rank(1)
        _run_rank_to_rendezvous(rank2, step=0)
        rank2.release()
        rank2._thread.join(timeout=5.0)
        self.assertFalse(rank2._thread.is_alive())
        result = rank2.join(timeout=0.0)
        self.assertIsInstance(result, StepResult)

    def test_join_tolerates_a_never_started_thread(self) -> None:
        # `_thread` is None whenever `_done` could only have been set through some path other than
        # start_step()'s own worker -- join() must not crash calling is_alive()/join() on None.
        rank = _make_rank()
        rank._done.set()
        result = rank.join(timeout=1.0)
        self.assertIsNone(result)


class JoinRemainingBudgetArithmeticTest(unittest.TestCase):
    """``join()`` computes ONE deadline up front and spends the REMAINING budget on ``thread.join()``,
    not the full original timeout again -- otherwise a step whose completion signal arrives late could
    let the total call consume up to ~2x the caller's requested deadline instead of respecting it.

    Uses a mocked clock so the deadline arithmetic itself is pinned exactly, and a spy on the real
    ``Thread.join`` to record what timeout it actually receives -- this makes the assertion depend on
    neither real machine contention nor timing luck: ``_done`` is set for real, and fast, before the
    clock is ever mocked, and unlike ``time.monotonic``, ``threading``'s own internal waiting captured
    its timing function at import time and is unaffected by patching ``time.monotonic`` afterward.
    """

    def test_thread_join_receives_the_remaining_budget_not_the_full_timeout_again(self) -> None:
        rank = _make_rank()
        _run_rank_to_rendezvous(rank, step=0)
        rank.release()
        self.assertTrue(rank._done.wait(timeout=5.0))  # real, but fast -- no artificial linger needed

        real_thread_join = rank._thread.join
        captured_timeouts: list[float | None] = []

        def _spy_join(timeout: float | None = None) -> None:
            captured_timeouts.append(timeout)
            return real_thread_join(timeout=timeout)

        rank._thread.join = _spy_join

        requested_timeout = 10.0
        elapsed_during_done_wait = 9.95  # pretend `_done.wait()` itself consumed almost the whole budget
        base = 1_000_000.0
        fake_now = itertools.chain(
            [base, base + elapsed_during_done_wait], itertools.repeat(base + elapsed_during_done_wait)
        )

        with mock.patch("mixle.utils.parallel.fault_tolerant_training.time.monotonic", side_effect=fake_now):
            try:
                rank.join(timeout=requested_timeout)
            except TimeoutError:
                # Acceptable: only ~0.05s of (real) remaining budget was left, and a correct join()
                # that genuinely cannot observe the thread's death that fast SHOULD time out -- that
                # does not change what argument it computed and passed to thread.join(), which is the
                # only thing this test checks.
                pass

        self.assertEqual(len(captured_timeouts), 1)
        expected_remaining = requested_timeout - elapsed_during_done_wait
        self.assertAlmostEqual(captured_timeouts[0], expected_remaining, places=6)

        rank._thread.join(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
