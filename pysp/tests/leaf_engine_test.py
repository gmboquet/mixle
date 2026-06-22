"""Engine-resident accumulation parity for non-exponential leaf families (numpy + torch).

These leaves have no exponential-family generated-suff-stat path; this checks that their
seq_update_engine produces statistics identical to the host seq_update on numpy and torch.
"""

import unittest

import numpy as np

from pysp.engines import NUMPY_ENGINE
from pysp.stats.base.integer_uniform_spike import IntegerUniformSpikeDistribution
from pysp.stats.base.laplace import LaplaceDistribution
from pysp.stats.base.pareto import ParetoDistribution
from pysp.stats.base.point_mass import PointMassDistribution
from pysp.stats.base.uniform import UniformDistribution

try:
    from pysp.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:
    _TORCH = None


def _flatten(v):
    out, stack = [], [v]
    while stack:
        u = stack.pop()
        if isinstance(u, (tuple, list)):
            stack.extend(u)
        elif u is None:
            continue
        else:
            a = np.asarray(u, dtype=np.float64).ravel()
            if a.size:
                out.append(np.sort(a))
    return np.concatenate(out) if out else np.zeros(0)


class LeafEngineTestCase(unittest.TestCase):
    def _check(self, dist, data, atol=1.0e-9):
        engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])
        est = dist.estimator()
        enc = dist.dist_to_encoder().seq_encode(data)
        weights = np.linspace(0.5, 1.5, len(data))
        host = est.accumulator_factory().make()
        host.seq_update(enc, weights, dist)
        hv = _flatten(host.value())
        for name, engine in engines:
            with self.subTest(dist=type(dist).__name__, engine=name):
                kernel = dist.kernel(engine=engine, estimator=est)
                value = kernel.accumulate(enc, weights)
                self.assertTrue(
                    np.allclose(hv, _flatten(value), atol=atol), "%s %s suff-stats differ" % (type(dist).__name__, name)
                )

    def test_pareto(self):
        d = ParetoDistribution(xm=1.0, alpha=2.5)
        self._check(d, list(d.sampler(seed=1).sample(40)))

    def test_uniform(self):
        d = UniformDistribution(-2.0, 5.0)
        self._check(d, list(d.sampler(seed=2).sample(40)))

    def test_laplace(self):
        d = LaplaceDistribution(0.5, 1.3)
        self._check(d, list(d.sampler(seed=3).sample(40)))

    def test_int_spike(self):
        d = IntegerUniformSpikeDistribution(k=3, num_vals=10, p=0.6, min_val=0)
        self._check(d, list(d.sampler(seed=4).sample(60)))

    def test_point_mass(self):
        d = PointMassDistribution(7.0)
        self._check(d, [7.0] * 20)


if __name__ == "__main__":
    unittest.main()
