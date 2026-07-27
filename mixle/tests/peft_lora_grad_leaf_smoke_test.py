"""Smoke test for ``examples/peft_lora_grad_leaf.py``: the real-checkpoint receipt still holds.

Reuses the example's own building blocks (a real ``peft``-wrapped tiny HF checkpoint dropped into
``GradLeaf``) rather than re-deriving them, so this pins the actual example against regressions instead
of a parallel hand-rolled copy. Skips cleanly (no network, or peft/transformers missing) rather than
failing CI machines without internet access.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from peft_lora_grad_leaf import build_peft_wrapped_module, toy_token_sequences  # noqa: E402

from mixle.inference.estimation import optimize  # noqa: E402
from mixle.models import GradLeaf  # noqa: E402


def _offline_or_skip():
    try:
        build_peft_wrapped_module(seed=0)
    except Exception as exc:  # pragma: no cover - depends on network/HF Hub availability  # noqa: BLE001
        pytest.skip(f"tiny HF checkpoint unavailable (likely offline): {exc}")


class PeftLoraGradLeafSmokeTest(unittest.TestCase):
    def test_only_the_lora_adapter_trains_on_a_real_checkpoint(self):
        _offline_or_skip()
        module = build_peft_wrapped_module(seed=0)

        base_before = {k: v.clone() for k, v in module.lm.base_model.model.state_dict().items() if "lora_" not in k}
        adapters_before = {
            name: parameter.detach().clone()
            for name, parameter in module.lm.named_parameters()
            if "lora_" in name and parameter.requires_grad
        }

        train = toy_token_sequences(
            vocab=module.lm.config.vocab_size,
            block=8,
            n=32,
            rng=np.random.RandomState(0),
        )
        held_out = toy_token_sequences(
            vocab=module.lm.config.vocab_size,
            block=8,
            n=32,
            rng=np.random.RandomState(1),
        )
        leaf = GradLeaf(module, m_steps=40, lr=0.1)
        before_ll = float(np.mean(leaf.seq_log_density(np.stack(held_out))))

        fitted = optimize(train, leaf, max_its=2, out=None)

        after_ll = float(np.mean(fitted.seq_log_density(np.stack(held_out))))
        base_after = {k: v for k, v in fitted.module.lm.base_model.model.state_dict().items() if "lora_" not in k}
        for k in base_before:  # the frozen checkpoint never moved
            np.testing.assert_array_equal(base_after[k].numpy(), base_before[k].numpy())

        adapter_deltas = {
            name: float((parameter.detach() - adapters_before[name]).abs().max())
            for name, parameter in fitted.module.lm.named_parameters()
            if name in adapters_before
        }
        self.assertTrue(any(delta > 0.0 for delta in adapter_deltas.values()))
        self.assertGreater(after_ll, before_ll)

    def test_adapter_includes_a_normalized_initial_token_law(self):
        source = (Path(__file__).resolve().parents[2] / "examples" / "peft_lora_grad_leaf.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("initial = -torch.log", source)
        self.assertIn("self.lm.config.vocab_size", source)


if __name__ == "__main__":
    unittest.main()
