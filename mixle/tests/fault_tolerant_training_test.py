"""Fault-tolerant gradient training (F2): chaos test, bitwise loader state, loss continuity, async-snapshot
stall budget.

Mirrors ``resilient_em_test``'s (K4) shape but for the gradient-training path: real concurrent (thread)
ranks, a deterministic post-backward kill rendezvous, and real DCP save/load -- everything except FSDP2
sharding and a real multi-node all-reduce (out of scope on a laptop, called out in
``fault_tolerant_training``'s module docstring) is exercised for real, at small scale.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from mixle.models.transformer import build_causal_lm
from mixle.utils.parallel.dcp_checkpoint import save_sharded
from mixle.utils.parallel.fault_tolerant_training import (
    ElasticTrainingJob,
    LoaderState,
    load_checkpoint,
    save_checkpoint_async,
    synthetic_batch_for_state,
)
from mixle.utils.parallel.training_health import TrainingHealthMonitor

VOCAB, D_MODEL, N_LAYER, N_HEAD, BLOCK = 32, 16, 2, 2, 8


def _tiny_model():
    torch.manual_seed(0)
    return build_causal_lm(vocab=VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, block=BLOCK)


def _batch_fn(rank: int, state: LoaderState):
    return synthetic_batch_for_state(state, vocab=VOCAB, block=BLOCK, batch_size=4)


class ChaosTestKillRankMidStep(unittest.TestCase):
    """Kill a rank at a KNOWN step, mid-step (post-backward, pre-optimizer-step rendezvous), and confirm
    the job continues rather than hard-failing -- then elastically respawn the dead rank from the last
    checkpoint and confirm it rejoins."""

    def test_job_continues_with_fewer_ranks_after_a_mid_step_kill(self):
        with tempfile.TemporaryDirectory() as d:
            job = ElasticTrainingJob(_tiny_model, world_size=3, batch_fn_for_rank=_batch_fn, checkpoint_dir=d)

            for step in range(3):
                rec = job.run_step(step)
                self.assertEqual(rec["newly_dead"], [])
                self.assertEqual(sorted(rec["survivors"]), [0, 1, 2])

            # Checkpoint while all 3 ranks are healthy -- this is what a respawned rank will resume from.
            handle = job.checkpoint()
            handle.wait(timeout=10)
            self.assertTrue(handle.done)
            pre_kill_loader_states = {r: s for r, s in job.loader_states.items()}

            # KNOWN kill point: rank 1 dies mid-step 3 (after its backward pass -- real compute happened --
            # but before its gradient is folded into the step's average).
            injected_step = 3
            rec = job.run_step(injected_step, kill_ranks=frozenset({1}))
            self.assertEqual(rec["newly_dead"], [1])
            self.assertEqual(sorted(rec["survivors"]), [0, 2])
            self.assertIn(1, job.dead_ranks)
            self.assertTrue(torch.isfinite(torch.tensor(rec["loss"])))  # the step produced a real result

            # The job does NOT hard-fail: it keeps running with the surviving ranks.
            for step in range(injected_step + 1, injected_step + 3):
                rec = job.run_step(step)
                self.assertEqual(sorted(rec["survivors"]), [0, 2])

            # Elastic restart: bring rank 1 back from the checkpoint saved before it died, not from
            # scratch -- its loader state resumes exactly where the checkpoint left it.
            restored_state = job.respawn_rank(1)
            self.assertEqual(restored_state, pre_kill_loader_states[1])
            self.assertNotIn(1, job.dead_ranks)
            self.assertTrue(job.pending_restart)

            rec = job.run_step(injected_step + 3)
            self.assertEqual(sorted(rec["survivors"]), [0, 1, 2])  # rank 1 rejoined
            self.assertTrue(rec["restart"])  # this step is the one F4 evaluates for continuity


class BitwiseLoaderStateTest(unittest.TestCase):
    """The captured/restored loader state round-trips bitwise-exactly: the batch produced right after a
    restore is identical to what the uninterrupted run would have produced at that same position."""

    def test_state_round_trip_reproduces_identical_next_batch(self):
        state = LoaderState(seed=7, epoch=2, rank=1, world_size=4, batch_idx=5)
        d = state.to_dict()
        restored = LoaderState.from_dict(d)
        self.assertEqual(state, restored)

        x1, y1 = synthetic_batch_for_state(state, vocab=VOCAB, block=BLOCK, batch_size=4)
        x2, y2 = synthetic_batch_for_state(restored, vocab=VOCAB, block=BLOCK, batch_size=4)
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))

    def test_reconstructing_position_by_advancing_matches_direct_construction(self):
        # An "uninterrupted run" advancing one batch at a time must land on the exact same LoaderState (and
        # therefore the exact same next batch) as directly asking for that position after a restart.
        uninterrupted = LoaderState(seed=3, epoch=0, rank=0, world_size=2, batch_idx=0)
        for _ in range(9):
            uninterrupted = uninterrupted.advanced()
        direct = LoaderState(seed=3, epoch=0, rank=0, world_size=2, batch_idx=9)
        self.assertEqual(uninterrupted, direct)

        x1, y1 = synthetic_batch_for_state(uninterrupted, vocab=VOCAB, block=BLOCK, batch_size=4)
        x2, y2 = synthetic_batch_for_state(direct, vocab=VOCAB, block=BLOCK, batch_size=4)
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))

    def test_checkpoint_resume_restores_exact_loader_position(self):
        with tempfile.TemporaryDirectory() as d:
            job = ElasticTrainingJob(_tiny_model, world_size=2, batch_fn_for_rank=_batch_fn, checkpoint_dir=d)
            for step in range(4):
                job.run_step(step)
            expected = job.loader_states[0]  # batch_idx == 4 after 4 successful steps

            handle = job.checkpoint()
            handle.wait(timeout=10)
            fresh_model = _tiny_model()
            fresh_opt = torch.optim.SGD(fresh_model.parameters(), lr=1e-2)
            loader_state = load_checkpoint(fresh_model, fresh_opt, d)
            self.assertEqual(loader_state, expected)


class ResumeContinuityBothDirectionsTest(unittest.TestCase):
    """F4's continuity check, wired into the real resume path: PASSES for a correctly-implemented resume
    (model + optimizer state genuinely restored) and FAILS for a deliberately broken one (state silently
    dropped) -- both directions, per the acceptance criterion."""

    def _fixed_batch(self):
        torch.manual_seed(1)
        x = torch.randint(0, VOCAB, (4, BLOCK))
        y = torch.randint(0, VOCAB, (4,))
        return x, y

    def _train_steps(self, model, opt, monitor, x, y, n, start_step, restart=False):
        for i in range(n):
            opt.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            monitor.observe_step(start_step + i, float(loss.item()), restart=(restart and i == 0))

    # Trained long enough that the rolling window (below) has fully aged out the volatile early-descent
    # losses by the time of the restart -- otherwise the window's own MAD is inflated by the initial loss
    # and a genuine post-restart blowup (which coincidentally lands near that same initial value, since
    # both the "well-behaved" and "broken" scenarios overfit the exact same fixed batch) would not look
    # anomalous relative to it.
    N_PRETRAIN_STEPS = 15
    LOSS_WINDOW = 8

    def test_well_implemented_resume_preserves_continuity(self):
        with tempfile.TemporaryDirectory() as d:
            x, y = self._fixed_batch()
            model = _tiny_model()
            opt = torch.optim.Adam(model.parameters(), lr=5e-2)
            monitor = TrainingHealthMonitor(loss_window=self.LOSS_WINDOW, loss_min_periods=4, loss_z_thresh=6.0)

            # Overfit a single fixed batch for a while -- loss decreases reliably and reproducibly.
            self._train_steps(model, opt, monitor, x, y, n=self.N_PRETRAIN_STEPS, start_step=0)

            handle = save_checkpoint_async(model, opt, d, LoaderState(seed=0, epoch=0, rank=0, world_size=1))
            handle.wait(timeout=10)

            # A correct resume: a FRESH model/optimizer object, but state genuinely restored from disk.
            resumed_model = _tiny_model()
            resumed_opt = torch.optim.Adam(resumed_model.parameters(), lr=5e-2)
            load_checkpoint(resumed_model, resumed_opt, d)

            self._train_steps(
                resumed_model, resumed_opt, monitor, x, y, n=3, start_step=self.N_PRETRAIN_STEPS, restart=True
            )

            self.assertTrue(monitor.continuity_ok())
            self.assertFalse(any(a.kind == "restart_discontinuity" for a in monitor.anomalies))

    def test_broken_resume_that_drops_state_is_flagged(self):
        x, y = self._fixed_batch()
        model = _tiny_model()
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)
        monitor = TrainingHealthMonitor(loss_window=self.LOSS_WINDOW, loss_min_periods=4, loss_z_thresh=6.0)

        # Overfit -- loss drops well below its starting value and stays low/stable.
        self._train_steps(model, opt, monitor, x, y, n=self.N_PRETRAIN_STEPS, start_step=0)
        pre_restart_loss = monitor.records[-1].loss

        # A BROKEN resume: silently drop the checkpoint entirely (e.g. a bug that skips load_checkpoint) --
        # continue "training" from a freshly initialized model instead of the trained one.
        broken_model = _tiny_model()
        broken_opt = torch.optim.Adam(broken_model.parameters(), lr=5e-2)
        self._train_steps(broken_model, broken_opt, monitor, x, y, n=3, start_step=self.N_PRETRAIN_STEPS, restart=True)
        post_restart_loss = monitor.records[self.N_PRETRAIN_STEPS].loss

        self.assertGreater(post_restart_loss, pre_restart_loss + 1.0)  # a real, sizeable jump
        self.assertFalse(monitor.continuity_ok())
        self.assertTrue(any(a.kind == "restart_discontinuity" for a in monitor.anomalies))


class AsyncSnapshotStallBudgetTest(unittest.TestCase):
    """Measure real wall-clock stall: the async path must return (and let training continue) in well under
    a slow disk's write latency; a synchronous save must not."""

    SLOW_DISK_S = 0.35
    STALL_BUDGET_S = 0.15  # generous vs. SLOW_DISK_S, tight enough to prove the write is NOT awaited

    def _slow_dcp_save(self, real_save):
        def _wrapped(*args, **kwargs):
            time.sleep(self.SLOW_DISK_S)
            return real_save(*args, **kwargs)

        return _wrapped

    def test_async_save_returns_well_under_slow_disk_latency(self):
        import torch.distributed.checkpoint as dcp

        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        loader_state = LoaderState(seed=0, epoch=0, rank=0, world_size=1)

        real_save = dcp.save
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch("torch.distributed.checkpoint.save", side_effect=self._slow_dcp_save(real_save)),
        ):
            t0 = time.perf_counter()
            handle = save_checkpoint_async(model, opt, d, loader_state)
            call_elapsed = time.perf_counter() - t0

            # The caller (i.e. the training loop) is unblocked well before the "disk" finishes writing.
            self.assertLess(call_elapsed, self.STALL_BUDGET_S)
            self.assertLess(handle.prepare_time_s, self.STALL_BUDGET_S)
            self.assertFalse(handle.done)  # the slow write is genuinely still in flight

            # A training-step-shaped bit of work can run immediately, concurrently with the write.
            t1 = time.perf_counter()
            _ = torch.randn(64, 64) @ torch.randn(64, 64)
            training_step_elapsed = time.perf_counter() - t1
            self.assertLess(training_step_elapsed, self.STALL_BUDGET_S)

            handle.wait(timeout=10)
            self.assertTrue(handle.done)
            self.assertTrue(Path(d, "loader_state.json").exists())

    def test_synchronous_save_blocks_for_the_full_slow_disk_latency(self):
        import torch.distributed.checkpoint as dcp

        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)

        real_save = dcp.save
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch("torch.distributed.checkpoint.save", side_effect=self._slow_dcp_save(real_save)),
        ):
            t0 = time.perf_counter()
            save_sharded(model, opt, d)
            elapsed = time.perf_counter() - t0
            self.assertGreaterEqual(elapsed, self.SLOW_DISK_S)


if __name__ == "__main__":
    unittest.main()
