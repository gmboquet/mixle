"""Edge distillation (mixle.task.edge): device budgets, structure x process search, design meta-model."""

import json
import unittest

import numpy as np

from mixle.task import (
    DesignModel,
    DeviceSpec,
    EdgeFootprint,
    EdgeSpace,
    distill_designer,
    distill_for_edge,
    footprint,
)


def _make_records(n, seed):
    """Records (x: float, tag: str) with a rule label learnable by both student families."""
    rng = np.random.RandomState(seed)
    recs, labels = [], []
    for _ in range(n):
        x = float(rng.normal())
        tag = "p" if rng.random() < 0.5 else "q"
        recs.append((x, tag))
        labels.append("a" if (tag == "p") == (x > 0) else "b")
    return recs, labels


class RuleTeacher:
    """A 'teacher' implementing the rule directly (stands in for a large model)."""

    def __call__(self, records):
        if isinstance(records, list):
            return ["a" if (t == "p") == (x > 0) else "b" for (x, t) in records]
        x, t = records
        return "a" if (t == "p") == (x > 0) else "b"


def _tiny_space():
    return EdgeSpace(
        families=("mlp", "structured"),
        dim_choices=(64, 128),
        hidden_range=(4, 24),
        epochs_range=(30, 90),
        components_range=(1, 2),
        max_its_range=(8, 20),
    )


class FootprintTest(unittest.TestCase):
    def test_mlp_footprint_matches_closed_form(self):
        from mixle.task import distill_records_from_labels

        recs, labels = _make_records(60, 0)
        student = distill_records_from_labels(recs, labels, dim=64, hidden=[8], epochs=10, lr=1e-2, seed=0)
        fp = footprint(student)
        # Linear(64->8) + Linear(8->2): params = 64*8+8 + 8*2+2; macs = 64*8 + 8*2
        self.assertEqual(fp.bytes, 4 * (64 * 8 + 8 + 8 * 2 + 2))
        self.assertEqual(fp.ops, 64 * 8 + 8 * 2)
        self.assertFalse(fp.torch_free)

    def test_structured_footprint_is_measured_and_torch_free(self):
        from mixle.task import distill_structured_from_labels

        recs, labels = _make_records(120, 1)
        student = distill_structured_from_labels(recs, labels, seed=0)
        fp = footprint(student)
        self.assertTrue(fp.torch_free)
        self.assertGreater(fp.bytes, 0)
        # 2 labels x 1 component x (2 fields + 1) factor evaluations
        self.assertEqual(fp.ops, 2 * 1 * 3)


class DeviceSpecTest(unittest.TestCase):
    def test_feasibility_and_violations(self):
        dev = DeviceSpec(max_bytes=1000, max_ops=50)
        ok = EdgeFootprint(bytes=800, ops=40, torch_free=False)
        too_big = EdgeFootprint(bytes=2000, ops=40, torch_free=False)
        self.assertTrue(dev.feasible(ok))
        self.assertFalse(dev.feasible(too_big))
        v = dev.violations(too_big)
        self.assertEqual(len(v), 2)
        self.assertAlmostEqual(v[0], 1.0)  # (2000-1000)/1000
        self.assertLess(v[1], 0.0)

    def test_torch_free_gate(self):
        dev = DeviceSpec(torch_free=True)
        self.assertFalse(dev.feasible(EdgeFootprint(10, 1, torch_free=False)))
        self.assertTrue(dev.feasible(EdgeFootprint(10, 1, torch_free=True)))


class EdgeSpaceTest(unittest.TestCase):
    def test_decode_covers_both_families(self):
        sp = _tiny_space()
        fam0, r0 = sp.decode(np.array([0.0, 0.5, 0.5, 0.5, 0.5]))
        fam1, r1 = sp.decode(np.array([0.99, 0.5, 0.5, 0.5, 0.5]))
        self.assertEqual(fam0, "mlp")
        self.assertIn("epochs", r0)
        self.assertEqual(fam1, "structured")
        self.assertIn("n_components", r1)
        # decode is deterministic and in-range
        self.assertIn(r0["dim"], sp.dim_choices)
        self.assertTrue(sp.components_range[0] <= r1["n_components"] <= sp.components_range[1])

    def test_signature_changes_with_space(self):
        self.assertNotEqual(_tiny_space().signature(), EdgeSpace().signature())


class DesignModelTest(unittest.TestCase):
    def _seeded(self):
        # synthetic ledger: quality peaks at x0=0.2; designs with x0 > 0.6 blow the budget
        dm = DesignModel("sig", n_constraints=1)
        rng = np.random.RandomState(0)
        for _ in range(20):
            p = rng.uniform(0, 1, size=2)
            quality = float(np.exp(-((p[0] - 0.2) ** 2) * 8.0))
            viol = float(p[0] - 0.6)  # feasible iff x0 <= 0.6
            dm.add(p, quality, [viol], fidelity="screen")
        return dm

    def test_predict_shapes_and_feasibility_gradient(self):
        dm = self._seeded()
        out = dm.predict([[0.1, 0.5], [0.9, 0.5]])
        self.assertEqual(out["mean"].shape, (2,))
        self.assertEqual(out["sd"].shape, (2,))
        # deep in the feasible region vs deep in the infeasible region
        self.assertGreater(out["p_feasible"][0], out["p_feasible"][1])

    def test_propose_prefers_feasible_high_quality(self):
        dm = self._seeded()
        picks = np.array([dm.propose([(0.0, 1.0)] * 2, seed=s) for s in range(6)])
        # the feasibility-weighted acquisition should mostly stay out of the infeasible zone
        self.assertGreaterEqual(np.mean(picks[:, 0] <= 0.65), 0.5)

    def test_json_roundtrip(self):
        dm = self._seeded()
        clone = DesignModel.from_json(json.loads(json.dumps(dm.to_json())))
        self.assertEqual(len(clone), len(dm))
        self.assertEqual(clone.signature, dm.signature)
        np.testing.assert_allclose(clone.X, dm.X)

    def test_cold_propose_is_random_in_bounds(self):
        dm = DesignModel("sig", 1)
        p = dm.propose([(0.0, 1.0)] * 3, seed=0)
        self.assertEqual(p.shape, (3,))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))


class DistillForEdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, _ = _make_records(240, 10)
        cls.val, _ = _make_records(120, 11)
        cls.teacher = RuleTeacher()

    def test_search_returns_feasible_student_and_valid_pareto(self):
        dev = DeviceSpec(max_bytes=200_000)  # generous: both families fit; search optimizes quality
        res = distill_for_edge(
            self.teacher, self.train, self.val, dev, space=_tiny_space(), n_init=3, n_iter=2, promote=2, seed=0
        )
        self.assertTrue(res.feasible)
        self.assertTrue(dev.feasible(res.footprint))
        self.assertGreater(res.agreement, 0.7)  # the rule is learnable
        # the winner was re-trained at full fidelity
        self.assertIn("full", {t["fidelity"] for t in res.trials})
        # Pareto front: sorted by bytes and non-dominated (agreement strictly improves with bytes)
        front = res.pareto
        self.assertGreater(len(front), 0)
        byte_sizes = [f["bytes"] for f in front]
        self.assertEqual(byte_sizes, sorted(byte_sizes))
        agrees = [f["agreement"] for f in front]
        self.assertTrue(all(a2 >= a1 for a1, a2 in zip(agrees, agrees[1:])))
        # the callable student actually classifies
        self.assertIn(res.model(self.val[0]), ("a", "b"))

    def test_torch_free_device_forces_structured_family(self):
        dev = DeviceSpec(torch_free=True)
        res = distill_for_edge(
            self.teacher, self.train, self.val, dev, space=_tiny_space(), n_init=2, n_iter=1, promote=1, seed=0
        )
        self.assertEqual(res.family, "structured")
        self.assertTrue(res.footprint.torch_free)
        self.assertTrue(res.feasible)
        self.assertGreater(res.agreement, 0.7)

    def test_design_model_warm_start_accumulates_and_matches_cold(self):
        dev = DeviceSpec(max_bytes=200_000)
        cold = distill_for_edge(
            self.teacher, self.train, self.val, dev, space=_tiny_space(), n_init=3, n_iter=2, promote=1, seed=0
        )
        n_ledger = len(cold.design)
        warm = distill_for_edge(
            self.teacher,
            self.train,
            self.val,
            dev,
            space=_tiny_space(),
            design=cold.design,
            n_init=2,
            n_iter=1,
            promote=1,
            seed=1,
        )
        self.assertGreater(len(warm.design), n_ledger)  # knowledge accumulates across searches
        self.assertGreaterEqual(warm.agreement, 0.9 * cold.agreement)  # cheap warm run holds the line

    def test_incompatible_design_model_is_rejected(self):
        dev = DeviceSpec(max_bytes=200_000)
        with self.assertRaises(ValueError):
            distill_for_edge(
                self.teacher, self.train, self.val, dev, space=_tiny_space(), design=DesignModel("other", 1), seed=0
            )

    def test_torch_free_with_mlp_only_space_raises(self):
        with self.assertRaises(ValueError):
            distill_for_edge(
                self.teacher,
                self.train,
                self.val,
                DeviceSpec(torch_free=True),
                space=EdgeSpace(families=("mlp",)),
                seed=0,
            )


class DistillDesignerTest(unittest.TestCase):
    def test_designer_is_a_torch_free_student_over_the_ledger(self):
        # a ledger with a clear pattern: designs with x0 < 0.5 are good, above are weak
        dm = DesignModel("sig", n_constraints=1)
        rng = np.random.RandomState(2)
        for _ in range(40):
            p = rng.uniform(0, 1, size=3)
            good = p[0] < 0.5
            dm.add(p, 0.9 if good else 0.3, [(-0.5 if good else 0.5)])
        designer = distill_designer(dm)
        self.assertEqual(designer.payload, "json")  # torch-free: runs anywhere
        # it judges new designs, and gets the planted pattern mostly right
        preds_good = [designer((0.1, 0.5, 0.5)), designer((0.2, 0.3, 0.8))]
        preds_weak = [designer((0.9, 0.5, 0.5)), designer((0.8, 0.3, 0.8))]
        self.assertTrue(all(p in ("good", "weak") for p in preds_good + preds_weak))
        self.assertGreaterEqual(sum(p == "good" for p in preds_good) + sum(p == "weak" for p in preds_weak), 3)

    def test_needs_enough_rows_and_two_classes(self):
        dm = DesignModel("sig", 0)
        for i in range(4):
            dm.add([i / 4, 0.5], 0.5, [])
        with self.assertRaises(ValueError):
            distill_designer(dm)
        uniform = DesignModel("sig", 0)
        for i in range(10):
            uniform.add([i / 10, 0.5], 0.9, [])  # every design equally good -> one class
        with self.assertRaises(ValueError):
            distill_designer(uniform)


if __name__ == "__main__":
    unittest.main()
