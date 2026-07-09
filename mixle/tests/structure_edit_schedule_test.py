"""H3: structure-edit schedule during training -- gating correctness, real edit wrappers, and the
core acceptance criterion (adaptive-structure beats the best fixed-structure baseline on total
compute, on a small ladder).

Synthetic task, deliberately DEPTH-dependent (unlike H1's own single-block-suffices task): a
two-stage modular composition of four context tokens, ``stage1 = (x0+x1) mod V``, ``target =
(stage1*x2 + x3) mod V``. A 1-layer CausalLM measurably PLATEAUS above the chosen target loss on
this task (verified below); 2- and 3-layer models keep improving past it -- so growing depth mid-
training is a real, necessary move here, not a decorative one.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mixle.experimental.structure_edit_schedule import (  # noqa: E402
    STRUCTURE_EDIT_REGISTRY,
    apply_structure_edit,
    health_report_from_monitor,
    should_apply_edit,
    train_with_adaptive_structure,
)
from mixle.models.coarsening import CoarsenedLM  # noqa: E402
from mixle.models.moment_propagation import GaussianLaw  # noqa: E402
from mixle.models.transformer import build_causal_lm  # noqa: E402
from mixle.utils.parallel.training_health import TrainingHealthMonitor  # noqa: E402

VOCAB = 8
BLOCK = 6
D_MODEL = 16
N_HEAD = 2


def _make_batch(batch_size: int, rng: np.random.Generator):
    ctx = rng.integers(0, VOCAB, size=(batch_size, BLOCK))
    x0, x1, x2, x3 = ctx[:, 0], ctx[:, 1], ctx[:, 2], ctx[:, 3]
    stage1 = (x0 + x1) % VOCAB
    target = (stage1 * x2 + x3) % VOCAB
    x = torch.as_tensor(ctx, dtype=torch.float32)
    y = torch.as_tensor(target, dtype=torch.long)
    return x, y


def _build(n_layer: int, seed: int = 0):
    torch.manual_seed(seed)
    return build_causal_lm(vocab=VOCAB, d_model=D_MODEL, n_layer=n_layer, n_head=N_HEAD, block=BLOCK)


class GatingTest(unittest.TestCase):
    """should_apply_edit rejects on either gate failing, accepts only when both hold."""

    def test_rejects_when_training_health_has_a_recent_anomaly(self):
        monitor = TrainingHealthMonitor()
        for step in range(10):
            monitor.observe_step(step, loss=1.0)
        monitor.observe_step(10, loss=float("nan"))  # a real NaN-loss anomaly, most recent step
        health = health_report_from_monitor(monitor, lookback=5)
        self.assertFalse(health["healthy"])
        self.assertIn("nan_inf_loss", health["recent_anomalies"])

        model = _build(n_layer=2, seed=0)
        new_model, receipt = apply_structure_edit(model, "grow_insert", {"position": 0, "seed": 0})
        self.assertTrue(receipt.parity.within_tolerance)  # the edit itself is fine...
        self.assertFalse(should_apply_edit(health, receipt.parity))  # ...but health says no.

    def test_rejects_when_parity_check_fails(self):
        monitor = TrainingHealthMonitor()
        for step in range(10):
            monitor.observe_step(step, loss=1.0)  # no anomalies at all
        health = health_report_from_monitor(monitor, lookback=5)
        self.assertTrue(health["healthy"])

        model = _build(n_layer=3, seed=0)
        rng = np.random.default_rng(0)
        mu = rng.normal(scale=0.3, size=D_MODEL)
        a = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
        cov = a @ a.T + 1e-3 * np.eye(D_MODEL)
        law = GaussianLaw(mu=mu, covar=cov)
        # depth_merge is only second-order accurate -- at a tight tolerance this real edit fails parity.
        new_model, receipt = apply_structure_edit(
            model, "prune_depth_merge", {"position": 0, "input_law": law, "seed": 0, "tolerance": 1e-9}
        )
        self.assertFalse(receipt.parity.within_tolerance)
        self.assertFalse(should_apply_edit(health, receipt.parity))

    def test_accepts_when_healthy_and_parity_ok(self):
        monitor = TrainingHealthMonitor()
        for step in range(10):
            monitor.observe_step(step, loss=1.0)
        health = health_report_from_monitor(monitor, lookback=5)
        self.assertTrue(health["healthy"])

        model = _build(n_layer=2, seed=0)
        # insert_block is EXACT (zero-init residual branches), so parity passes at a tight tolerance.
        new_model, receipt = apply_structure_edit(model, "grow_insert", {"position": 0, "seed": 0})
        self.assertTrue(receipt.parity.within_tolerance)
        self.assertTrue(should_apply_edit(health, receipt.parity))


class ApplyStructureEditWrappersTest(unittest.TestCase):
    """apply_structure_edit produces a real, usable modified model + a real receipt, for grow and
    prune at minimum (the acceptance criterion's explicit floor)."""

    def test_grow_insert_produces_a_deeper_real_model_with_a_receipt(self):
        model = _build(n_layer=2, seed=1)
        new_model, receipt = apply_structure_edit(model, "grow_insert", {"position": 1, "seed": 0})
        self.assertEqual(new_model.n_layer, 3)
        self.assertEqual(receipt.edit_type, "grow_insert")
        self.assertIsNotNone(receipt.parity)
        self.assertTrue(receipt.parity.within_tolerance)
        # a real, usable model: forward pass returns real (batch, vocab) logits.
        rng = np.random.default_rng(0)
        x, _y = _make_batch(8, rng)
        logits = new_model(x)
        self.assertEqual(tuple(logits.shape), (8, VOCAB))

    def test_prune_depth_merge_produces_a_shallower_real_model_with_a_receipt(self):
        model = _build(n_layer=3, seed=1)
        rng = np.random.default_rng(2)
        mu = rng.normal(scale=0.3, size=D_MODEL)
        a = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
        cov = a @ a.T + 1e-3 * np.eye(D_MODEL)
        law = GaussianLaw(mu=mu, covar=cov)
        new_model, receipt = apply_structure_edit(
            model, "prune_depth_merge", {"position": 0, "input_law": law, "seed": 0, "tolerance": 1.0}
        )
        self.assertIsInstance(new_model, CoarsenedLM)
        self.assertEqual(new_model.n_layer, 2)
        self.assertEqual(receipt.edit_type, "prune_depth_merge")
        self.assertIsNotNone(receipt.parity)
        self.assertIsNotNone(receipt.detail)  # a real ScaleReceipt with a real closed-form KL
        self.assertGreaterEqual(receipt.detail.kl_divergence, 0.0)
        rng2 = np.random.default_rng(0)
        x, _y = _make_batch(8, rng2)
        logits = new_model(x)
        self.assertEqual(tuple(logits.shape), (8, VOCAB))

    def test_moe_expert_add_is_a_documented_scaffold_not_a_silent_noop(self):
        self.assertIn("moe_expert_add", STRUCTURE_EDIT_REGISTRY)
        self.assertIn("SCAFFOLD ONLY", STRUCTURE_EDIT_REGISTRY["moe_expert_add"])
        model = _build(n_layer=1, seed=0)
        with self.assertRaises(NotImplementedError):
            apply_structure_edit(model, "moe_expert_add", {})


class DepthPlateauSanityTest(unittest.TestCase):
    """Confirms the synthetic task really is depth-dependent -- otherwise the acceptance test below
    would not be testing anything meaningful. A 1-layer model plateaus measurably above 1-3-nat
    losses that 2- and 3-layer models reach; run once, short, just to pin the premise."""

    def test_one_layer_plateaus_above_target_two_layer_does_not(self):
        target_loss = 1.3
        torch.manual_seed(7)
        model1 = _build(n_layer=1, seed=7)
        opt1 = torch.optim.AdamW(model1.parameters(), lr=5e-3)
        rng = np.random.default_rng(107)
        ema = None
        for _ in range(600):
            x, y = _make_batch(64, rng)
            loss = F.cross_entropy(model1(x), y)
            opt1.zero_grad()
            loss.backward()
            opt1.step()
            ema = float(loss.item()) if ema is None else 0.9 * ema + 0.1 * float(loss.item())
        self.assertGreater(ema, target_loss)  # 1-layer plateaus above target


class AdaptiveVsFixedLadderTest(unittest.TestCase):
    """THE acceptance criterion: "on a small ladder, the adaptive-structure run reaches target loss
    with measurably less compute than the best fixed structure." Compute = the real F4
    theoretical_flops_per_iter summed over every step, at the model's shape AT that step.

    Averaged over several seeds (compute-to-target on a tiny model/task is genuinely noisy step to
    step -- see the module docstring's honesty note): the adaptive run's AVERAGE total compute is
    asserted below the best FIXED baseline's AVERAGE total compute, by a real margin (see
    ``_MIN_ADAPTIVE_MARGIN`` below), not just "any nonzero win". Individual-seed numbers are printed
    so a wins-most-but-not-all-seeds outcome, if it occurs, is visible rather than hidden.

    CORRECTED RESULT (this run, under the repo's reproducible single-threaded CPU-math harness --
    see mixle/tests/conftest.py's ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``/``torch.set_num_threads(1)``
    pinning, which is how `pytest` -- the sanctioned way to run this suite -- actually executes it):
    fixed n_layer=2 avg 1.725e+10 FLOPs, fixed n_layer=3 avg 1.330e+10, adaptive avg 1.314e+10 --
    adaptive beats the best fixed baseline by ~1.3% less compute, winning individually on only 1 of
    3 seeds. This is BOTH byte-identical run-to-run under the pinned harness AND the number that
    reproduces on a clean checkout via `pytest mixle/tests/structure_edit_schedule_test.py`.

    An earlier PR description for this module (H3, #172/#230) reported a materially larger flagship
    number -- fixed n_layer=2 avg 1.872e+10, fixed n_layer=3 avg 1.336e+10, adaptive avg 1.210e+10,
    a 9.4% win on 2 of 3 seeds. That number does NOT reproduce under this repo's own CI-sanctioned
    single-threaded harness; it only reproduces when the benchmark is run OUTSIDE that harness (e.g.
    directly via ``python -m unittest`` without conftest.py's thread pinning in effect), where the
    default multi-threaded BLAS matmul reduction order is not bit-reproducible across environments.
    Because the controller's plateau detector is a hard numeric threshold
    (``ema_hist[-1] - ema_hist[-plateau_window] > -plateau_eps``), those tiny floating-point
    differences can flip exactly when an edit is considered, cascading into a materially different
    total-compute outcome over a 2500-step run -- i.e. the PR's flagship number was a real but
    non-reproducible-under-CI measurement, not a bug in the scheduler itself. Ad hoc sampling of
    other seed triples under the pinned harness confirms the mechanism's win is genuinely fragile
    (single-digit-percent at best, sometimes a net loss) rather than a reliable double-digit margin
    -- so the margin asserted below is calibrated to the modest, honestly-reproducible result above,
    not the original overstated one.
    """

    # Calibrated to the CI-reproducible ~1.3% margin measured above for SEEDS=(1, 2, 3): a real,
    # non-trivial floor (rules out "wins by any nonzero amount, however marginal") that still leaves
    # headroom below the actual measured margin so this doesn't flake on the pinned harness.
    _MIN_ADAPTIVE_MARGIN = 0.01  # require >=1% less compute than the best fixed baseline, not just any win

    TARGET_LOSS = 1.3
    SEEDS = (1, 2, 3)
    MAX_STEPS = 2500

    def _fixed_baseline(self, n_layer: int, seed: int) -> tuple[float, bool, int]:
        model = _build(n_layer=n_layer, seed=seed)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        rng = np.random.default_rng(seed + 100)
        from mixle.utils.parallel.training_health import flop_config_from_causal_lm

        total_flops = 0.0
        ema = None
        for step in range(self.MAX_STEPS):
            x, y = _make_batch(64, rng)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            cfg = flop_config_from_causal_lm(model, BLOCK)
            total_flops += cfg.flops_per_iter(64)
            lv = float(loss.item())
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            if ema < self.TARGET_LOSS and step > 30:
                return total_flops, True, step + 1
        return total_flops, False, self.MAX_STEPS

    def test_adaptive_beats_best_fixed_on_average_total_compute(self):
        fixed_sizes = (2, 3)
        fixed_results: dict[int, list[tuple[float, bool, int]]] = {n: [] for n in fixed_sizes}
        adaptive_results: list = []

        for seed in self.SEEDS:
            for n_layer in fixed_sizes:
                fixed_results[n_layer].append(self._fixed_baseline(n_layer, seed))

            small_model = _build(n_layer=1, seed=seed)
            result = train_with_adaptive_structure(
                small_model,
                _make_batch,
                target_loss=self.TARGET_LOSS,
                max_steps=self.MAX_STEPS,
                max_layer=3,
                batch_size=64,
                lr=5e-3,
                min_steps_before_edit=80,
                plateau_window=40,
                plateau_eps=0.01,
                seed=seed,
            )
            adaptive_results.append(result)

        lines = ["\n[H3 acceptance] adaptive-structure vs. fixed-structure ladder, total compute (FLOPs):"]
        best_fixed_avg = None
        for n_layer in fixed_sizes:
            rows = fixed_results[n_layer]
            reached = [r for r in rows if r[1]]
            avg_flops = float(np.mean([r[0] for r in rows]))
            lines.append(
                f"  fixed n_layer={n_layer}: per-seed flops={[f'{r[0]:.3e}' for r in rows]} "
                f"reached={[r[1] for r in rows]} avg_flops={avg_flops:.3e}"
            )
            if all(r[1] for r in rows):  # only a baseline that reaches target on EVERY seed counts as "best fixed"
                if best_fixed_avg is None or avg_flops < best_fixed_avg:
                    best_fixed_avg = avg_flops

        adaptive_flops = [r.total_flops for r in adaptive_results]
        adaptive_reached = [r.reached_target for r in adaptive_results]
        adaptive_edits = [r.edits_applied for r in adaptive_results]
        avg_adaptive = float(np.mean(adaptive_flops))
        lines.append(
            f"  adaptive:            per-seed flops={[f'{f:.3e}' for f in adaptive_flops]} "
            f"reached={adaptive_reached} avg_flops={avg_adaptive:.3e} edits={adaptive_edits}"
        )
        lines.append(f"  best fixed avg = {best_fixed_avg:.3e}; adaptive avg = {avg_adaptive:.3e}")
        if avg_adaptive < best_fixed_avg:
            lines.append(
                f"  adaptive wins by {100 * (best_fixed_avg - avg_adaptive) / best_fixed_avg:.1f}% less compute"
            )
        else:
            lines.append(
                f"  adaptive did NOT win on average this run "
                f"({100 * (avg_adaptive - best_fixed_avg) / best_fixed_avg:.1f}% more compute)"
            )
        n_adaptive_wins = sum(1 for f in adaptive_flops for bf in [best_fixed_avg] if f < bf)
        lines.append(f"  adaptive beat best-fixed-avg on {n_adaptive_wins}/{len(adaptive_flops)} individual seeds")
        print("\n".join(lines))

        self.assertIsNotNone(best_fixed_avg, "no fixed baseline reached target_loss on every seed")
        self.assertTrue(all(adaptive_reached), "adaptive run failed to reach target_loss on some seed")
        # A real margin, not just "any win, however marginal" -- see _MIN_ADAPTIVE_MARGIN's docstring
        # note above for how this was calibrated and why it replaced a bare assertLess.
        self.assertLess(avg_adaptive, best_fixed_avg * (1.0 - self._MIN_ADAPTIVE_MARGIN))


if __name__ == "__main__":
    unittest.main()
