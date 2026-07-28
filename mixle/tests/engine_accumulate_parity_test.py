"""Host-vs-engine E-step parity across the full distribution catalog (numpy + torch).

Drives every torch-ready distribution from the backend-scoring catalog through both the host
seq_update and the engine kernel accumulate, asserting identical sufficient statistics. This is the
end-to-end check that structural wrappers keep nested families engine-resident.
"""

import dataclasses
import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.tests.backend_scoring_test import BackendScoringTestCase

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


def _flatten(v):
    """Flatten a nested suff-stat structure into ``(numbers, labels)`` by in-order traversal.

    Lists/tuples are traversed element-wise (so a Python list of scalars and an equivalent ndarray
    flatten to the same order); dicts are visited by sorted string key; dataclasses -- which is what
    the typed ``*Statistics`` records are -- by declared field order, which is identical on both
    sides because both produce the same type; ndarrays are raveled.

    Sufficient statistics carry two kinds of leaf. Numeric quantities get compared within tolerance.
    Identity labels (a ConditionalBranchStatistics ``key``, a MarkovChainStatistics ``states``) are
    structural and have to match exactly -- coercing them to float64 raises, and dropping them would
    let a kernel that accumulated the right counts against the wrong state ordering pass as parity.
    So they are collected separately and compared for equality.
    """
    nums, labels = [], []

    def rec(u):
        if u is None:
            return
        if isinstance(u, dict):
            for _, val in sorted(u.items(), key=lambda kv: str(kv[0])):
                rec(val)
        elif dataclasses.is_dataclass(u) and not isinstance(u, type):
            for field in dataclasses.fields(u):
                rec(getattr(u, field.name))
        elif isinstance(u, (str, bytes)):
            labels.append(u.decode("utf-8", "replace") if isinstance(u, bytes) else u)
        elif isinstance(u, (tuple, list)):
            for el in u:
                rec(el)
        else:
            a = np.asarray(u)
            if a.dtype.kind in "biufc":
                a = a.astype(np.float64).ravel()
                if a.size:
                    nums.append(a)
            else:  # str/object arrays are label content, not measurements
                labels.extend(str(x) for x in np.ravel(a).tolist())

    rec(v)
    return (np.concatenate(nums) if nums else np.zeros(0)), tuple(labels)


def _catalog():
    cases = []
    for meth in ["backend_leaf_cases", "stacked_mixture_cases"]:
        fn = getattr(BackendScoringTestCase, meth, None)
        if fn is None:
            continue
        for c in fn():
            dist, data = (c[1], c[2]) if len(c) == 3 else (c[0], c[1])
            cases.append((type(dist).__name__, dist, data))
    return cases


class EngineAccumulateParityTestCase(unittest.TestCase):
    def test_catalog_parity(self):
        engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])
        checked = 0
        for name, dist, data in _catalog():
            if "torch" not in dist.supported_engines():
                continue
            try:
                est = dist.estimator()
            except Exception:  # noqa: BLE001
                continue
            enc = dist.dist_to_encoder().seq_encode(data)
            weights = np.linspace(0.5, 1.5, len(data))
            host = est.accumulator_factory().make()
            host.seq_update(enc, weights, dist)
            hv, hlabels = _flatten(host.value())
            for ename, engine in engines:
                with self.subTest(dist=name, engine=ename):
                    kernel = dist.kernel(engine=engine, estimator=est)
                    value, vlabels = _flatten(kernel.accumulate(enc, weights))
                    self.assertEqual(hlabels, vlabels, "%s/%s suff-stat labels differ" % (name, ename))
                    self.assertEqual(hv.shape, value.shape, "%s/%s suff-stat shape mismatch" % (name, ename))
                    self.assertTrue(
                        np.allclose(hv, value, atol=1.0e-6, rtol=1.0e-6), "%s/%s suff-stats differ" % (name, ename)
                    )
            checked += 1
        self.assertGreater(checked, 20, "expected to exercise many catalog distributions")


if __name__ == "__main__":
    unittest.main()
