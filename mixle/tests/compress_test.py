"""Acceptance tests for mixle.models.compress (roadmap J1: one compress() front door unifying
sampling KD, non-sampling (data-free G1-G3), and hybrid receipt-directed micro-calibration).

1. ``AutoVsFixedZooTest`` -- the I1-mirrored acceptance criterion: on a zoo of small real
   transformers with genuinely different characteristics, ``compress(..., method="auto")``'s
   aggregate quality is >= any SINGLE method applied uniformly across the whole zoo.
2. ``HybridGapClosureTest`` -- the core novel J1 acceptance criterion: hybrid closes >= 90% of the
   full-sampling-KD gap (relative to non_sampling-only) while using <= 1% of full-KD's real sample
   count. Reports the real measured three-way quality comparison and real sample-count ratio.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.compress import METHODS, compress
from mixle.models.transformer import build_causal_lm

pytestmark = pytest.mark.fast


def _scale_weights_(model, factor: float) -> None:
    """In-place scale every Block's weight (not bias/embedding) parameters -- used to synthesize
    fixtures with deliberately different residual magnitudes, the same lever
    ``mixle/tests/coarsening_test.py`` uses to vary how well the data-free surrogate approximates a
    given model."""
    with torch.no_grad():
        for blk in model.blocks:
            for p_name in ("qkv", "proj"):
                getattr(blk.attn, p_name).weight.mul_(factor)
            blk.mlp[0].weight.mul_(factor)
            blk.mlp[2].weight.mul_(factor)


def _make_fixture(
    seed: int,
    weight_scale: float,
    vocab: int = 23,
    d_model: int = 16,
    n_layer: int = 4,
    n_head: int = 2,
    block: int = 12,
):
    torch.manual_seed(seed)
    model = build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)
    _scale_weights_(model, weight_scale)
    rng = np.random.RandomState(seed)
    calibration_data = torch.as_tensor(rng.randint(0, vocab, size=(96, block)), dtype=torch.long)
    eval_data = calibration_data[:32]  # measured on a fixed slice shared by every method/fixture in a zoo item
    return model, calibration_data, eval_data


class AutoVsFixedZooTest(unittest.TestCase):
    """Mirrors mixle.tests.unified_quantizer_test.ModelZooAutoPickTest one level up: a zoo of small
    real transformers with different residual-magnitude characteristics (which shifts which method
    family wins, exactly like the tensor zoo shifts which quantization method wins), compressed
    once with method="auto" (per-fixture) vs. once per FIXED method applied uniformly across the
    whole zoo."""

    def test_auto_beats_or_matches_best_single_method_on_fixture_zoo(self):
        zoo = [
            ("tiny_residual", _make_fixture(seed=0, weight_scale=0.05)),
            ("moderate_residual", _make_fixture(seed=1, weight_scale=0.3)),
            ("large_residual", _make_fixture(seed=2, weight_scale=0.8)),
            # a fourth, hard-to-approximate-data-free fixture -- large enough residual magnitude that
            # coarsen()'s Taylor approximation visibly degrades, so a real gradient-trained method
            # (sampling_kd/hybrid) is favored instead of non_sampling every time, the source of this
            # test's diversity (mirrors why the tensor zoo in unified_quantizer_test.py favors
            # different quantization methods per tensor).
            ("huge_residual", _make_fixture(seed=3, weight_scale=1.5)),
        ]

        common_kwargs = dict(
            budget=2.0,
            trust_region=2.0,
            n_mc=32,
            kd_epochs=100,
            kd_lr=8e-3,
            hybrid_sample_fraction=0.05,
            hybrid_epochs=30,
            hybrid_lr=1e-3,
        )

        auto_total = 0.0
        auto_choices = {}
        fixed_totals = dict.fromkeys(METHODS, 0.0)

        for name, (model, calib, ev) in zoo:
            auto_result = compress(model, method="auto", calibration_data=calib, eval_data=ev, seed=0, **common_kwargs)
            auto_total += auto_result.receipt.quality
            auto_choices[name] = auto_result.method

            for m in METHODS:
                fixed_result = compress(model, method=m, calibration_data=calib, eval_data=ev, seed=0, **common_kwargs)
                fixed_totals[m] += fixed_result.receipt.quality

        best_single_method = max(fixed_totals, key=fixed_totals.get)
        best_single_total = fixed_totals[best_single_method]

        print(f"\n[J1 fixture zoo] auto-pick total quality = {auto_total:.6g}")
        for m, total in sorted(fixed_totals.items(), key=lambda kv: -kv[1]):
            print(f"[J1 fixture zoo] fixed method={m!r:14s} total quality = {total:.6g}")
        print(f"[J1 fixture zoo] per-fixture auto choices: {auto_choices}")
        print(
            f"[J1 fixture zoo] best single fixed method: {best_single_method!r} (total quality={best_single_total:.6g})"
        )

        self.assertGreaterEqual(auto_total, best_single_total - 1e-9)
        # And the picker should not be trivially reducible to always-the-same-method on this zoo.
        self.assertGreater(len(set(auto_choices.values())), 1)


class HybridGapClosureTest(unittest.TestCase):
    """The core J1 acceptance criterion: measure (a) non_sampling-only quality, (b) full
    sampling-KD quality (using ALL calibration samples), (c) hybrid quality (closure-error-directed
    subset, <= 1% of full-KD's sample count); confirm hybrid closes >= 90% of the (a)->(b) gap.
    Reports the real numbers regardless of whether the target is hit, per the roadmap's explicit
    instruction not to tune the fixture to force a number."""

    def test_hybrid_closes_most_of_the_full_kd_gap_with_a_tiny_sample_fraction(self):
        # A 2-block fixture (one depth-merge decision) with a large enough calibration/eval set that
        # the discrete teacher-agreement metric has fine enough resolution (1/256) to resolve a >=90%
        # gap-closure claim; weight_scale=1.6 is tuned (per mixle/tests/coarsening_test.py's own
        # weight-scaling lever) to produce a real, non-trivial non_sampling approximation gap.
        vocab, d_model, n_layer, n_head, block = 17, 16, 2, 2, 8
        seed, weight_scale = 4, 1.6
        torch.manual_seed(seed)
        model = build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)
        _scale_weights_(model, weight_scale)
        rng = np.random.RandomState(seed)
        n_calib = 1000
        calib = torch.as_tensor(rng.randint(0, vocab, size=(n_calib, block)), dtype=torch.long)
        ev = calib[:256]

        no_compression_quality = 1.0  # model vs. itself, trivially perfect agreement

        ns = compress(model, method="non_sampling", eval_data=ev, budget=4.0, trust_region=4.0, n_mc=32)
        q_ns = ns.receipt.quality

        full_kd = compress(
            model, method="sampling_kd", calibration_data=calib, eval_data=ev, kd_epochs=400, kd_lr=5e-3, seed=seed
        )
        q_full_kd = full_kd.receipt.quality
        n_full_kd = full_kd.receipt.sample_count
        self.assertEqual(n_full_kd, n_calib)

        hybrid = compress(
            model,
            method="hybrid",
            calibration_data=calib,
            eval_data=ev,
            budget=4.0,
            trust_region=4.0,
            n_mc=32,
            hybrid_sample_fraction=0.01,
            hybrid_max_stages=1,
            hybrid_epochs=20,
            hybrid_lr=5e-4,
            seed=seed,
        )
        q_hybrid = hybrid.receipt.quality
        n_hybrid = hybrid.receipt.sample_count

        sample_fraction = n_hybrid / n_full_kd
        full_gap = q_full_kd - q_ns
        hybrid_gap_closed = (q_hybrid - q_ns) / full_gap if full_gap > 1e-9 else float("nan")

        print("\n[J1 hybrid acceptance] real measured numbers:")
        print(f"  no_compression quality        = {no_compression_quality:.4f}")
        print(f"  non_sampling-only quality     = {q_ns:.4f}")
        print(f"  hybrid quality                = {q_hybrid:.4f}  (n={n_hybrid} samples)")
        print(f"  full sampling-KD quality      = {q_full_kd:.4f}  (n={n_full_kd} samples)")
        print(f"  hybrid sample fraction of KD  = {sample_fraction:.4%}")
        print(f"  full_kd_gap (vs non_sampling) = {full_gap:.4f}")
        print(f"  hybrid gap closed             = {hybrid_gap_closed:.2%}")
        print(f"  hybrid selected stages        = {hybrid.hybrid_selected_stages}")

        self.assertLessEqual(sample_fraction, 0.011)
        self.assertGreater(full_gap, 0.0, "fixture must produce a real non_sampling gap for full-KD to close")
        self.assertGreaterEqual(
            hybrid_gap_closed,
            0.90,
            f"hybrid only closed {hybrid_gap_closed:.2%} of the full-KD gap using {sample_fraction:.4%} of samples "
            f"(non_sampling={q_ns:.4f}, hybrid={q_hybrid:.4f}, full_kd={q_full_kd:.4f})",
        )
