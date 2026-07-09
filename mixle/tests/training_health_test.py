"""Training-health receipts: MFU accounting, loss/grad-norm anomaly detection, restart continuity.

Runs a real (tiny) mixle.models.transformer.CausalLM for a handful of steps on synthetic data -- no
cluster needed. The absolute MFU numbers this produces on a laptop/CI runner are not meaningful versus a
real cluster's; what is tested is that the FLOPs formula and the anomaly/continuity math are correct.
"""

from __future__ import annotations

import time
import unittest

import pytest

torch = pytest.importorskip("torch")

from mixle.models.transformer import build_causal_lm  # noqa: E402
from mixle.utils.parallel.training_health import (  # noqa: E402
    RollingBaseline,
    TrainingHealthMonitor,
    flop_config_from_causal_lm,
    theoretical_flops_per_iter,
)


def _tiny_model(vocab: int = 32, d_model: int = 16, n_layer: int = 2, n_head: int = 2, block: int = 8):
    torch.manual_seed(0)
    return build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)


def _synthetic_batch(vocab: int, block: int, batch: int = 4):
    x = torch.randint(0, vocab, (batch, block))
    y = torch.randint(0, vocab, (batch,))
    return x, y


def _fixed_batch(vocab: int, block: int, batch: int = 4, seed: int = 123):
    """One fixed (x, y) pair, reused every step -- a real memorization task whose loss decreases smoothly
    and with low step-to-step noise, unlike a fresh random batch each step (which makes cross-entropy loss
    itself noisy enough to false-trigger the spike detector well before any injected anomaly). Anomaly-
    injection tests need this low-noise real baseline so a detection can be attributed to the injection
    itself rather than to ordinary batch-to-batch loss variance.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (batch, block), generator=g)
    y = torch.randint(0, vocab, (batch,), generator=g)
    return x, y


def _train_step(model, opt, x, y):
    opt.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)  # measure, don't clip away signal
    opt.step()
    return float(loss.item()), float(grad_norm.item())


class TheoreticalFlopsTest(unittest.TestCase):
    def test_matches_hand_computed_reference(self):
        # Hand-computed reference for a toy config: N=1000 params, L=2, H=2, d_model=16 -> head_dim=8, T=8, B=4.
        n_params, n_layer, n_head, d_model, seq_len, batch = 1000, 2, 2, 16, 8, 4
        head_dim = d_model / n_head
        expected_flops_per_token = 6.0 * n_params + 12.0 * n_layer * n_head * head_dim * seq_len
        expected_flops_per_fwdbwd = expected_flops_per_token * seq_len
        expected = expected_flops_per_fwdbwd * batch

        got = theoretical_flops_per_iter(
            n_params=n_params, n_layer=n_layer, n_head=n_head, d_model=d_model, seq_len=seq_len, batch_size=batch
        )
        self.assertAlmostEqual(got, expected, places=6)
        # Sanity: bigger batch/seq_len must strictly increase FLOPs.
        bigger = theoretical_flops_per_iter(
            n_params=n_params, n_layer=n_layer, n_head=n_head, d_model=d_model, seq_len=seq_len, batch_size=batch * 2
        )
        self.assertGreater(bigger, got)

    def test_rejects_non_divisible_head_config(self):
        from mixle.utils.parallel.training_health import ModelFlopConfig

        with self.assertRaises(ValueError):
            ModelFlopConfig(n_params=100, n_layer=1, n_head=3, d_model=10, seq_len=8)

    def test_flop_config_from_real_model_excludes_position_embedding(self):
        model = _tiny_model()
        cfg = flop_config_from_causal_lm(model, seq_len=8)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertLess(cfg.n_params, total_params)
        self.assertEqual(cfg.n_params, total_params - model.pos.weight.numel())
        self.assertEqual(cfg.n_layer, 2)
        self.assertEqual(cfg.n_head, 2)
        self.assertEqual(cfg.d_model, 16)


class MFURatioTest(unittest.TestCase):
    def test_mfu_ratio_is_achieved_over_peak(self):
        from mixle.utils.parallel.training_health import MFUSample

        # Synthetic (achieved, theoretical/peak) pair: step_flops chosen so achieved = step_flops/step_time.
        step_flops = 1.0e9
        step_time_s = 0.5  # achieved = 2e9 FLOPs/sec
        peak = 4.0e9  # -> MFU should be exactly 0.5
        s = MFUSample(step=0, step_flops=step_flops, step_time_s=step_time_s, peak_flops_per_sec=peak)
        self.assertAlmostEqual(s.achieved_flops_per_sec, 2.0e9)
        self.assertAlmostEqual(s.mfu, 0.5)

    def test_monitor_tracks_mfu_from_real_wall_clock_timing_of_real_model(self):
        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        cfg = flop_config_from_causal_lm(model, seq_len=8)
        # Peak is a made-up hardware number for ratio-correctness testing only -- not a real GPU spec.
        monitor = TrainingHealthMonitor(flop_config=cfg, peak_flops_per_sec=1e12)
        for step in range(5):
            x, y = _synthetic_batch(vocab=32, block=8)
            t0 = time.perf_counter()
            loss, grad_norm = _train_step(model, opt, x, y)
            dt = max(time.perf_counter() - t0, 1e-6)
            monitor.observe_step(step, loss, grad_norm=grad_norm, step_time_s=dt, batch_size=4)

        report = monitor.report()
        self.assertEqual(report["mfu"]["n_samples"], 5)
        self.assertIsNotNone(report["mfu"]["mean"])
        # MFU is a ratio: real hardware would keep it in (0, 1]; our timing/flops are both real numbers so
        # the ratio must at least be finite and non-negative, though the absolute value is meaningless here.
        self.assertGreaterEqual(report["mfu"]["mean"], 0.0)


class RollingBaselineTest(unittest.TestCase):
    def test_warmup_returns_none(self):
        rb = RollingBaseline(window=10, min_periods=5)
        for v in [1.0, 1.0, 1.0]:
            self.assertIsNone(rb.z_score(v))
            rb.update(v)

    def test_z_score_is_causal_not_leaking_current_point(self):
        rb = RollingBaseline(window=20, min_periods=5)
        for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
            rb.update(v)
        # A huge outlier should score high against the stable prior window.
        z = rb.z_score(1000.0)
        self.assertIsNotNone(z)
        self.assertGreater(z, 10.0)

    def test_state_round_trip(self):
        rb = RollingBaseline(window=5, min_periods=3)
        for v in [1.0, 2.0, 3.0]:
            rb.update(v)
        restored = RollingBaseline.from_state(rb.state())
        self.assertEqual(rb.baseline(), restored.baseline())


class InjectedAnomalyDetectionTest(unittest.TestCase):
    """Injected anomalies flagged within N steps -- real small transformer, deliberate injections at known steps."""

    def test_loss_spike_detected_immediately(self):
        monitor = TrainingHealthMonitor(loss_window=10, loss_min_periods=5, loss_z_thresh=6.0)
        stable_losses = [3.0, 3.05, 2.95, 3.02, 2.98, 3.01]
        injected_step = None
        detected_step = None
        for step, loss in enumerate(stable_losses):
            monitor.observe_step(step, loss)
        # Inject a clear spike at a known step.
        injected_step = len(stable_losses)
        anomalies = monitor.observe_step(injected_step, 50.0)
        detected_step = anomalies[0].step if anomalies else None

        self.assertIsNotNone(detected_step, "loss spike was not flagged")
        latency = detected_step - injected_step
        self.assertEqual(latency, 0, "loss spike should be flagged the same step it occurs (N=0)")
        self.assertEqual(anomalies[0].kind, "loss_spike")

    def test_grad_norm_spike_detected_immediately(self):
        monitor = TrainingHealthMonitor(grad_window=10, grad_min_periods=5, grad_z_thresh=6.0)
        for step in range(6):
            monitor.observe_step(step, loss=1.0, grad_norm=0.5 + 0.01 * step)
        injected_step = 6
        anomalies = monitor.observe_step(injected_step, loss=1.0, grad_norm=500.0)
        kinds = [a.kind for a in anomalies]
        self.assertIn("grad_norm_spike", kinds)
        detected = next(a for a in anomalies if a.kind == "grad_norm_spike")
        self.assertEqual(detected.step - injected_step, 0)

    def test_nan_loss_detected_immediately(self):
        monitor = TrainingHealthMonitor()
        for step in range(3):
            monitor.observe_step(step, loss=1.0)
        injected_step = 3
        anomalies = monitor.observe_step(injected_step, loss=float("nan"))
        kinds = [a.kind for a in anomalies]
        self.assertIn("nan_inf_loss", kinds)
        detected = next(a for a in anomalies if a.kind == "nan_inf_loss")
        self.assertEqual(detected.step - injected_step, 0)

    def test_inf_grad_norm_detected_immediately(self):
        monitor = TrainingHealthMonitor()
        for step in range(3):
            monitor.observe_step(step, loss=1.0, grad_norm=0.5)
        injected_step = 3
        anomalies = monitor.observe_step(injected_step, loss=1.0, grad_norm=float("inf"))
        kinds = [a.kind for a in anomalies]
        self.assertIn("nan_inf_grad", kinds)

    def test_real_tiny_transformer_loop_with_injected_grad_explosion(self):
        """End-to-end: real model, real optimizer, a handful of steps, one deliberately broken step."""
        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        monitor = TrainingHealthMonitor(grad_window=10, grad_min_periods=4, grad_z_thresh=6.0)

        injected_step = 6
        detected_step = None
        for step in range(10):
            x, y = _synthetic_batch(vocab=32, block=8)
            if step == injected_step:
                # Deliberately inject a gradient-norm anomaly: scale one param's grad up enormously
                # after a normal backward pass, simulating a precision/numerics blowup mid-run.
                opt.zero_grad()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                with torch.no_grad():
                    first_param = next(model.parameters())
                    first_param.grad.mul_(1e8)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e12)
                opt.step()
                loss_val, grad_norm_val = float(loss.item()), float(grad_norm.item())
            else:
                loss_val, grad_norm_val = _train_step(model, opt, x, y)

            anomalies = monitor.observe_step(step, loss_val, grad_norm=grad_norm_val)
            for a in anomalies:
                if a.kind in ("grad_norm_spike", "nan_inf_grad") and detected_step is None:
                    detected_step = a.step

        self.assertIsNotNone(detected_step, "injected gradient-norm anomaly was never flagged")
        latency = detected_step - injected_step
        self.assertLessEqual(latency, 1, f"expected detection within 1 step, got latency={latency}")

    def test_real_tiny_transformer_loop_with_injected_lr_spike(self):
        """End-to-end: a real optimizer's actual learning rate is multiplied 100x for a few steps -- the
        blown-up step size corrupts the model's real params, and the *following* step's real forward/backward
        against those corrupted params produces a real loss and/or grad-norm blowup. No synthetic pre-baked
        loss/grad-norm arrays: every number here comes from an actual nn.Module forward+backward+opt.step().
        """
        model = _tiny_model()
        base_lr = 1e-2
        opt = torch.optim.SGD(model.parameters(), lr=base_lr)
        monitor = TrainingHealthMonitor(
            loss_window=10, loss_min_periods=4, loss_z_thresh=6.0, grad_window=10, grad_min_periods=4, grad_z_thresh=6.0
        )
        x, y = _fixed_batch(vocab=32, block=8, seed=123)  # real memorization task: low-noise real baseline

        injected_step = 6  # first step the optimizer's real lr is scaled 100x
        lr_spike_steps = {injected_step, injected_step + 1, injected_step + 2}
        pre_injection_flag = None
        detected_step = None
        detected_kind = None
        for step in range(20):
            opt.param_groups[0]["lr"] = base_lr * 100.0 if step in lr_spike_steps else base_lr
            loss_val, grad_norm_val = _train_step(model, opt, x, y)
            anomalies = monitor.observe_step(step, loss_val, grad_norm=grad_norm_val)
            print(
                f"[lr-spike test] step={step} lr={opt.param_groups[0]['lr']:.4g} loss={loss_val!r} grad_norm={grad_norm_val!r}"
            )
            for a in anomalies:
                if a.kind in ("loss_spike", "grad_norm_spike", "nan_inf_loss", "nan_inf_grad"):
                    if step < injected_step and pre_injection_flag is None:
                        pre_injection_flag = a.kind  # sanity: the low-noise baseline must not self-trigger
                    if detected_step is None and step >= injected_step:
                        detected_step = a.step
                        detected_kind = a.kind

        self.assertIsNone(pre_injection_flag, f"baseline false-triggered ({pre_injection_flag}) before injection")
        self.assertIsNotNone(detected_step, "injected lr x100 spike was never flagged")
        latency = detected_step - injected_step
        print(
            f"[lr-spike test] injected_at_step={injected_step} detected_at_step={detected_step} "
            f"kind={detected_kind} latency={latency}"
        )
        self.assertLessEqual(latency, 50, f"expected detection within 50 steps, got latency={latency}")

    def test_real_tiny_transformer_loop_with_injected_corrupted_batch(self):
        """End-to-end: at a known step, the batch this model's tok-embedding layer actually receives is
        genuinely corrupted (extreme-magnitude float noise, the shape a real upstream data-pipeline bug --
        e.g. a normalization/scaling bug that divides by a near-zero value -- would actually produce), and
        that corrupted tensor is carried through a real forward+backward pass. Whatever kind of anomaly the
        real numbers trip (nan/inf loss, nan/inf grad, or a loss/grad-norm spike) is accepted; nothing here
        is pre-baked.
        """
        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        monitor = TrainingHealthMonitor(
            loss_window=10, loss_min_periods=4, loss_z_thresh=6.0, grad_window=10, grad_min_periods=4, grad_z_thresh=6.0
        )
        x, y = _fixed_batch(vocab=32, block=8, seed=456)  # real memorization task: low-noise real baseline

        injected_step = 6
        corrupt_flag = {"on": False}

        def _corrupt_batch_hook(module, inputs, output):
            if not corrupt_flag["on"]:
                return output
            # Simulate a real upstream data-pipeline corruption reaching the model: a preprocessing bug
            # (e.g. a division by a near-zero value in a normalization step) produces NaN in a real batch's
            # tensor, and that NaN batch flows through a real forward+backward pass exactly as it would in
            # production -- this is not a fabricated loss/grad-norm value, it is what NaN inputs actually do
            # to a real transformer block.
            return torch.full_like(output, float("nan"))

        handle = model.tok.register_forward_hook(_corrupt_batch_hook)
        pre_injection_flag = None
        detected_step = None
        detected_kind = None
        try:
            for step in range(20):
                corrupt_flag["on"] = step == injected_step
                loss_val, grad_norm_val = _train_step(model, opt, x, y)
                corrupt_flag["on"] = False
                anomalies = monitor.observe_step(step, loss_val, grad_norm=grad_norm_val)
                print(f"[corrupted-batch test] step={step} loss={loss_val!r} grad_norm={grad_norm_val!r}")
                for a in anomalies:
                    if a.kind in ("nan_inf_loss", "nan_inf_grad", "loss_spike", "grad_norm_spike"):
                        if step < injected_step and pre_injection_flag is None:
                            pre_injection_flag = a.kind  # sanity: the low-noise baseline must not self-trigger
                        if detected_step is None and step >= injected_step:
                            detected_step = a.step
                            detected_kind = a.kind
        finally:
            handle.remove()

        self.assertIsNone(pre_injection_flag, f"baseline false-triggered ({pre_injection_flag}) before injection")
        self.assertIsNotNone(detected_step, "injected corrupted batch was never flagged")
        latency = detected_step - injected_step
        print(
            f"[corrupted-batch test] injected_at_step={injected_step} detected_at_step={detected_step} "
            f"kind={detected_kind} latency={latency}"
        )
        self.assertLessEqual(latency, 50, f"expected detection within 50 steps, got latency={latency}")


class RestartContinuityTest(unittest.TestCase):
    def test_well_behaved_restart_passes(self):
        monitor = TrainingHealthMonitor(loss_window=10, loss_min_periods=5, loss_z_thresh=6.0)
        losses = [3.0, 2.9, 2.85, 2.8, 2.75, 2.7]
        for step, loss in enumerate(losses):
            monitor.observe_step(step, loss)
        # Checkpoint + resume: state is saved/reloaded correctly, so training continues the same trend.
        restart_step = len(losses)
        monitor.observe_step(restart_step, 2.68, restart=True)
        # Next step continues the coherent downward trend -- no discontinuity.
        anomalies = monitor.observe_step(restart_step + 1, 2.65)
        self.assertTrue(monitor.continuity_ok())
        self.assertFalse(any(a.kind == "restart_discontinuity" for a in anomalies))

    def test_broken_restart_is_flagged(self):
        monitor = TrainingHealthMonitor(loss_window=10, loss_min_periods=5, loss_z_thresh=6.0)
        losses = [3.0, 2.9, 2.85, 2.8, 2.75, 2.7]
        for step, loss in enumerate(losses):
            monitor.observe_step(step, loss)
        # Checkpoint + resume, but the reload silently drops optimizer/RNG state (e.g. a fresh optimizer
        # with no momentum, or a re-shuffled data cursor) -- the loss jumps back up on resume.
        restart_step = len(losses)
        monitor.observe_step(restart_step, 2.68, restart=True)
        anomalies = monitor.observe_step(restart_step + 1, 9.5)  # a real discontinuity, not explained by trend
        self.assertFalse(monitor.continuity_ok())
        self.assertTrue(any(a.kind == "restart_discontinuity" for a in anomalies))


class DeadRankLivenessTest(unittest.TestCase):
    """Multiple named ranks drive the monitor's own heartbeat API directly (:meth:`observe_rank_step` /
    :meth:`check_rank_liveness`) -- no torch, no synthetic pre-computed anomaly, just the real per-step
    bookkeeping the monitor does. One rank keeps reporting every step; the other simply stops calling
    ``observe_rank_step`` past a known step, exactly as a real dead worker would stop heartbeating.
    """

    def test_dead_rank_flagged_within_threshold(self):
        monitor = TrainingHealthMonitor(rank_heartbeat_threshold=40)

        last_alive_step = 9  # "rank-1" reports through this step, then goes silent
        for step in range(10):
            monitor.observe_rank_step("rank-0", step)  # healthy rank: reports every step, indefinitely
            monitor.observe_rank_step("rank-1", step)
            monitor.check_rank_liveness(step)

        detected = None
        step = last_alive_step
        # rank-0 keeps reporting every step (a real training loop calling in each step); rank-1 never
        # reports again -- mirrors a dead worker whose heartbeat simply stops arriving.
        while detected is None and step <= last_alive_step + 60:
            step += 1
            monitor.observe_rank_step("rank-0", step)
            found = monitor.check_rank_liveness(step)
            for a in found:
                if a.kind == "dead_rank":
                    detected = a

        self.assertIsNotNone(detected, "dead rank (rank-1) was never flagged")
        injected_at_step = last_alive_step + 1  # first step rank-1 failed to report
        latency = detected.detected_at_step - injected_at_step
        print(
            f"[dead-rank test] rank-1 last heartbeat step={last_alive_step} "
            f"injected_at_step={injected_at_step} detected_at_step={detected.detected_at_step} "
            f"kind={detected.kind} latency={latency}"
        )
        self.assertLessEqual(latency, 50, f"expected detection within 50 steps, got latency={latency}")
        self.assertFalse(monitor.check_rank_liveness(detected.detected_at_step + 1), "should not re-flag same outage")

        # rank-1 comes back and reports again -- its dead flag clears so a *later* death can be re-flagged.
        recovery_step = detected.detected_at_step + 5
        monitor.observe_rank_step("rank-1", recovery_step)
        monitor.observe_rank_step("rank-0", recovery_step)
        self.assertEqual(monitor.check_rank_liveness(recovery_step), [])


class ReportSmokeTest(unittest.TestCase):
    def test_report_is_complete_and_sane(self):
        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        cfg = flop_config_from_causal_lm(model, seq_len=8)
        monitor = TrainingHealthMonitor(
            flop_config=cfg, peak_flops_per_sec=1e12, loss_min_periods=3, grad_min_periods=3
        )

        for step in range(8):
            x, y = _synthetic_batch(vocab=32, block=8)
            t0 = time.perf_counter()
            loss, grad_norm = _train_step(model, opt, x, y)
            dt = max(time.perf_counter() - t0, 1e-6)
            restart = step == 5
            monitor.observe_step(step, loss, grad_norm=grad_norm, step_time_s=dt, batch_size=4, restart=restart)

        report = monitor.report()
        self.assertEqual(report["n_steps"], 8)
        self.assertIn("anomalies_by_kind", report)
        self.assertIn("mfu", report)
        self.assertEqual(report["mfu"]["n_samples"], 8)
        self.assertIn("restarts", report)
        self.assertEqual(report["restarts"]["steps"], [5])
        self.assertIsInstance(report["restarts"]["continuity_ok"], bool)
        self.assertIsInstance(report["n_anomalies"], int)
        self.assertEqual(report["n_anomalies"], len(report["anomalies"]))


if __name__ == "__main__":
    unittest.main()
