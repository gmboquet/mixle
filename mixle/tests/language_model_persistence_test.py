"""LM.save persists an inference artifact, not a training checkpoint (worklist M11.3).

Model persistence and training resume are different contracts with different guarantees. ``LM.save`` writes
the *inference artifact* -- architecture config + learned weights -- so a loaded model can score and
generate. It is deliberately NOT a training-resume checkpoint: it stores no optimizer, scheduler, step,
RNG, data-loader position, or scaler state (that contract lives in
``mixle.utils.parallel.fault_tolerant_training`` / ``dcp_checkpoint``). This test pins that boundary, so
``LM.save`` cannot silently start being described or used as resumable training without the payload
gaining the checkpoint fields -- which would fail here.
"""

import pickle
import tempfile
import unittest

import pytest

pytest.importorskip("torch")

from mixle.models.language_model import LM  # noqa: E402

# Fields that belong to a TRAINING checkpoint, never to the inference artifact.
_TRAINING_ONLY_FIELDS = {
    "optimizer",
    "opt_state",
    "optimizer_state",
    "scheduler",
    "lr_scheduler",
    "step",
    "global_step",
    "epoch",
    "rng",
    "rng_state",
    "loader",
    "loader_state",
    "data_loader",
    "scaler",
    "grad_scaler",
    "distributed",
    "health",
}


def _tiny_lm():
    return LM(vocab=16, d_model=8, n_layer=1, n_head=2, block=4, device="cpu")


class _NotAWeight:
    """A plain object outside torch's weights_only allowlist -- stands in for an exploit payload."""


class LMPersistenceContractTest(unittest.TestCase):
    def test_to_dict_is_inference_artifact_only(self):
        payload = _tiny_lm().to_dict()
        # architecture config + weights, exactly
        self.assertEqual(set(payload), {"vocab", "d_model", "n_layer", "n_head", "block", "device", "state_dict"})
        # and nothing that belongs to a training-resume checkpoint
        self.assertEqual(set(payload) & _TRAINING_ONLY_FIELDS, set())

    def test_save_load_round_trips_a_scoring_ready_model(self):
        import torch

        lm = _tiny_lm()
        x = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            before = lm.module(x)
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/lm.pt"
            lm.save(path)
            loaded = LM.load(path)
        with torch.no_grad():
            after = loaded.module(torch.zeros(1, 4, dtype=torch.long))
        # a loaded artifact reproduces the model's outputs (scoring-ready), no training state needed
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_load_rejects_a_non_weights_payload(self):
        # LM.load must use torch.load(weights_only=True): the payload is plain config scalars plus a
        # tensor state_dict, never an arbitrary object, so a file smuggling a non-tensor/container
        # object (the shape an exploit would need) must be refused rather than unpickled.
        import torch

        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/lm.pt"
            torch.save({"payload": _NotAWeight()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                LM.load(path)


if __name__ == "__main__":
    unittest.main()
