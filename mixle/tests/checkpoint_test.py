"""EM checkpointing: the ``optimize(on_step=...)`` hook, ``Registry.checkpointer``, and resume."""

import tempfile
import unittest

import numpy as np

from mixle.inference import EMStep, optimize
from mixle.inference.production import Registry
from mixle.stats import GaussianEstimator, MixtureEstimator


def _data():
    rng = np.random.RandomState(0)
    return np.concatenate([rng.normal(-3.0, 1.0, 800), rng.normal(4.0, 1.0, 800)]).tolist()


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


class OnStepHookTest(unittest.TestCase):
    def _collect(self, **opt_kw):
        steps: list[EMStep] = []
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        optimize(
            _data(), est, max_its=5, delta=None, out=None, rng=np.random.RandomState(1), on_step=steps.append, **opt_kw
        )
        return steps

    def test_hook_fires_each_iteration_fused_path(self):
        steps = self._collect()  # default MLE path -> fused loop
        self.assertEqual([s.iter for s in steps], [1, 2, 3, 4, 5])
        for s in steps:  # every emitted model is real and scorable
            self.assertIsInstance(s, EMStep)
            self.assertTrue(np.isfinite(_ll(s.model, _data())))

    def test_hook_fires_each_iteration_standard_path(self):
        # reuse_estep_ll=False forces the standard (non-fused) loop, which emits via the same hook
        steps = self._collect(reuse_estep_ll=False)
        self.assertEqual([s.iter for s in steps], [1, 2, 3, 4, 5])
        # the standard loop reports the per-step training log-likelihood gain
        self.assertTrue(all(np.isfinite(s.log_density) for s in steps))


class CheckpointerTest(unittest.TestCase):
    def test_checkpointer_writes_every_n_with_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
            optimize(
                _data(),
                est,
                max_its=10,
                delta=None,
                out=None,
                rng=np.random.RandomState(1),
                on_step=reg.checkpointer("run", every=3),
            )
            self.assertEqual(reg.versions("run"), ["v1", "v2", "v3"])  # iters 3, 6, 9
            meta = reg.metadata("run")  # latest checkpoint's metadata, no model deserialization
            self.assertEqual(meta["checkpoint_iter"], 9)
            self.assertIn("log_density", meta)

    def test_resume_from_checkpoint_does_not_regress(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
            data = _data()
            optimize(
                data,
                est,
                max_its=3,
                delta=None,
                out=None,
                rng=np.random.RandomState(1),
                on_step=reg.checkpointer("run"),
            )
            mid, _ = reg.get("run")  # latest checkpoint == the iter-3 model
            resumed = optimize(data, est, max_its=5, delta=None, out=None, prev_estimate=mid)
            self.assertGreaterEqual(_ll(resumed, data), _ll(mid, data) - 1e-6)


if __name__ == "__main__":
    unittest.main()
