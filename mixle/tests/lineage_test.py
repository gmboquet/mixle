"""Model fingerprinting + verifiable EM iteration lineage (model_hash chain)."""

import json
import os
import tempfile
import unittest

import numpy as np

from mixle.data import model_hash
from mixle.inference import optimize
from mixle.inference.production import Registry, fit_with_provenance, verify_lineage
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureEstimator


def _data():
    rng = np.random.RandomState(0)
    return np.concatenate([rng.normal(-3.0, 1.0, 700), rng.normal(4.0, 1.0, 700)]).tolist()


def _est():
    return MixtureEstimator([GaussianEstimator(), GaussianEstimator()])


class ModelHashTest(unittest.TestCase):
    def test_deterministic_and_parameter_sensitive(self):
        self.assertEqual(model_hash(GaussianDistribution(0.0, 1.0)), model_hash(GaussianDistribution(0.0, 1.0)))
        self.assertNotEqual(model_hash(GaussianDistribution(0.0, 1.0)), model_hash(GaussianDistribution(0.1, 1.0)))

    def test_ignores_attached_header(self):
        # a fitted model carries a .header; the fingerprint is of the parameters, not the header
        model, _ = fit_with_provenance(_data(), _est(), max_its=3, delta=None, seed=1)
        self.assertEqual(model.header.model_hash, model_hash(model))


class LineageTraceTest(unittest.TestCase):
    def test_trace_is_a_valid_hash_chain(self):
        _, header = fit_with_provenance(_data(), _est(), max_its=6, delta=None, seed=1)
        trace = header.training["convergence"]
        self.assertIsNone(trace[0]["parent_hash"])  # the root has no parent
        for prev, cur in zip(trace, trace[1:]):  # iteration i+1 names i as its parent
            self.assertEqual(cur["parent_hash"], prev["model_hash"])
        self.assertTrue(verify_lineage(header))
        self.assertTrue(verify_lineage(header.to_dict()))  # also accepts the dict form

    def test_verify_detects_a_broken_link(self):
        _, header = fit_with_provenance(_data(), _est(), max_its=6, delta=None, seed=1)
        header.training["convergence"][3]["parent_hash"] = "deadbeef"
        self.assertFalse(verify_lineage(header))

    def test_lineage_false_skips_fingerprinting(self):
        _, header = fit_with_provenance(_data(), _est(), max_its=4, delta=None, seed=1, lineage=False)
        self.assertNotIn("model_hash", header.training["convergence"][0])
        self.assertTrue(verify_lineage(header))  # vacuously intact -- nothing recorded


class CheckpointChainTest(unittest.TestCase):
    def test_checkpoints_chain_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            optimize(
                _data(),
                _est(),
                max_its=9,
                delta=None,
                out=None,
                rng=np.random.RandomState(1),
                on_step=reg.checkpointer("run", every=3),
            )
            metas = [reg.metadata("run", v) for v in reg.versions("run")]
            self.assertIsNone(metas[0]["parent_hash"])
            for prev, cur in zip(metas, metas[1:]):
                self.assertEqual(cur["parent_hash"], prev["model_hash"])
            self.assertTrue(reg.verify_chain("run"))

    def test_verify_chain_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            optimize(
                _data(),
                _est(),
                max_its=6,
                delta=None,
                out=None,
                rng=np.random.RandomState(1),
                on_step=reg.checkpointer("run", every=2),
            )
            # corrupt the stored fingerprint of the latest checkpoint: the re-hash of the loaded model
            # no longer matches what was recorded
            path = os.path.join(d, "run", reg.versions("run")[-1] + ".json")
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            payload["metadata"]["model_hash"] = "0" * 64
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            self.assertFalse(reg.verify_chain("run"))


if __name__ == "__main__":
    unittest.main()
