"""Engine-resident E-step parity for the hidden association model (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.combinator.conditional import ConditionalDistribution, ConditionalDistributionEstimator
from mixle.stats.latent.hidden_association import HiddenAssociationDistribution, HiddenAssociationEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


def _flatten(value):
    out = []
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, (tuple, list)):
            stack.extend(v)
        elif isinstance(v, dict):
            stack.extend(val for _, val in sorted(v.items(), key=lambda kv: str(kv[0])))
        elif v is None:
            continue
        else:
            out.append(np.asarray(v, dtype=np.float64).ravel())
    return np.concatenate(out) if out else np.zeros(0)


class HiddenAssociationEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = HiddenAssociationDistribution(
            cond_dist=ConditionalDistribution(
                {
                    "a": CategoricalDistribution({"x": 0.80, "y": 0.20}),
                    "b": CategoricalDistribution({"x": 0.25, "y": 0.75}),
                }
            ),
            len_dist=CategoricalDistribution({0.0: 0.10, 2.0: 0.30, 3.0: 0.60}),
        )
        self.data = [
            ([("a", 2.0), ("b", 1.0)], [("x", 1.0), ("y", 2.0)]),
            ([("b", 3.0)], [("y", 2.0)]),
            ([("a", 1.0)], []),
            ([("a", 1.0), ("b", 2.0)], [("x", 3.0), ("y", 1.0)]),
        ]
        self.weights = np.array([1.0, 0.6, 1.4, 0.9])
        self.est = HiddenAssociationEstimator(
            cond_estimator=ConditionalDistributionEstimator({"a": CategoricalEstimator(), "b": CategoricalEstimator()}),
            len_estimator=CategoricalEstimator(),
        )
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "HiddenAssociationKernel")
                value = kernel.accumulate(enc, self.weights)
                self.assertTrue(
                    np.allclose(_flatten(hv[0]), _flatten(value[0]), atol=1.0e-7),
                    "%s conditional suff-stats differ" % name,
                )
                self.assertTrue(
                    np.allclose(_flatten(hv[2]), _flatten(value[2]), atol=1.0e-8), "%s size suff-stats differ" % name
                )


if __name__ == "__main__":
    unittest.main()
