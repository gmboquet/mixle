"""Edge distillation (mixle.task.edge): device budgets, structure x process search, design meta-model."""

import json
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")  # the edge-distillation stack builds torch students; skip cleanly where torch is absent

from mixle.task import (  # noqa: E402
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
        bits_choices=(32, 8),
        components_range=(1, 2),
        max_its_range=(8, 20),
    )


def _fast_edge_space():
    """Same cube as :func:`_tiny_space`, minus the mixture-of-dependency-trees arm.

    ``components_range=(1, 2)`` lets ``distill_for_edge`` propose a 2-component structured student,
    whose fit (hard EM: multiple restarts x iterations, each relearning a dependency forest per
    cluster) costs an order of magnitude more than every other candidate in this space combined --
    profiling a single search call showed >75% of wall time inside that one fit. None of the
    `DistillForEdgeTest` searches below assert anything about mixture components specifically (they
    check feasibility, agreement, Pareto validity, warm-start parity, and cross-task transfer), so
    pinning ``components_range=(1, 1)`` removes the expensive arm without touching what's verified.
    """
    sp = _tiny_space()
    sp.components_range = (1, 1)
    return sp


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
        fam0, r0 = sp.decode(np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.0]))
        fam1, r1 = sp.decode(np.array([0.99, 0.5, 0.5, 0.5, 0.5, 0.5]))
        self.assertEqual(fam0, "mlp")
        self.assertIn("epochs", r0)
        self.assertEqual(fam1, "structured")
        self.assertIn("n_components", r1)
        # decode is deterministic and in-range
        self.assertIn(r0["dim"], sp.dim_choices)
        self.assertTrue(sp.components_range[0] <= r1["n_components"] <= sp.components_range[1])

    def test_bits_axis_decodes_both_precisions(self):
        sp = _tiny_space()
        _, fp32 = sp.decode(np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.0]))
        _, int8 = sp.decode(np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.99]))
        self.assertEqual(fp32["bits"], 32)
        self.assertEqual(int8["bits"], 8)

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

    def test_prefilter_vetoes_weak_designs(self):
        # the designer loop closed: a judge labeling x0 > 0.5 'weak' steers proposals below it
        dm = self._seeded()
        judge = lambda pt: "weak" if pt[0] > 0.5 else "good"  # noqa: E731
        picks = np.array([dm.propose([(0.0, 1.0)] * 2, seed=s, prefilter=judge) for s in range(8)])
        self.assertGreaterEqual(np.mean(picks[:, 0] <= 0.5), 0.875)  # at most one veto survives retries

    def test_prefilter_that_rejects_everything_still_returns(self):
        dm = self._seeded()
        p = dm.propose([(0.0, 1.0)] * 2, seed=0, prefilter=lambda pt: "weak", max_tries=3)
        self.assertEqual(p.shape, (2,))  # the judge advises; the surrogate still decides

    def test_cold_propose_respects_prefilter(self):
        dm = DesignModel("sig", 1)
        judge = lambda pt: "weak" if pt[0] > 0.3 else "good"  # noqa: E731
        picks = np.array([dm.propose([(0.0, 1.0)] * 2, seed=s, prefilter=judge) for s in range(8)])
        self.assertGreaterEqual(np.mean(picks[:, 0] <= 0.3), 0.75)  # random draws filtered too


class CrossTaskDesignModelTest(unittest.TestCase):
    """Task fingerprints: one ledger, many tasks — the surrogate conditions on which task it is."""

    def _two_task_ledger(self):
        # task A (fingerprint 0.0): quality peaks at x0 = 0.2; task B (fingerprint 3.0): peaks at 0.8
        dm = DesignModel("sig", n_constraints=0, n_fingerprint=1)
        rng = np.random.RandomState(0)
        for _ in range(16):
            p = rng.uniform(0, 1, size=1)
            dm.add(p, float(np.exp(-((p[0] - 0.2) ** 2) * 12)), [], fingerprint=[0.0], task="A")
            q = rng.uniform(0, 1, size=1)
            dm.add(q, float(np.exp(-((q[0] - 0.8) ** 2) * 12)), [], fingerprint=[3.0], task="B")
        return dm

    def test_add_validates_fingerprint_length(self):
        dm = DesignModel("sig", 0, n_fingerprint=2)
        with self.assertRaises(ValueError):
            dm.add([0.5], 1.0, [], fingerprint=[0.0])  # wrong length
        dm.add([0.5], 1.0, [], fingerprint=[0.0, 1.0])
        self.assertEqual(len(dm.X[0]), 3)  # design coord + 2 fingerprint coords

    def test_predict_conditions_on_the_task(self):
        dm = self._two_task_ledger()
        pts = [[0.2], [0.8]]
        at_a = dm.predict(pts, fingerprint=[0.0])["mean"]
        at_b = dm.predict(pts, fingerprint=[3.0])["mean"]
        self.assertGreater(at_a[0], at_a[1])  # task A prefers x0=0.2
        self.assertGreater(at_b[1], at_b[0])  # task B prefers x0=0.8 — same ledger, flipped answer

    def test_propose_returns_design_dims_and_conditions_on_task(self):
        dm = self._two_task_ledger()
        picks_a = np.array([dm.propose([(0.0, 1.0)], seed=s, fingerprint=[0.0]) for s in range(6)])
        picks_b = np.array([dm.propose([(0.0, 1.0)], seed=s, fingerprint=[3.0]) for s in range(6)])
        self.assertEqual(picks_a.shape, (6, 1))  # fingerprint coords stripped from the proposal
        # proposals track each task's own optimum from the SHARED ledger
        self.assertLess(np.median(picks_a[:, 0]), np.median(picks_b[:, 0]))

    def test_json_roundtrip_keeps_fingerprint_dims(self):
        dm = self._two_task_ledger()
        clone = DesignModel.from_json(json.loads(json.dumps(dm.to_json())))
        self.assertEqual(clone.n_fingerprint, 1)
        np.testing.assert_allclose(clone.X, dm.X)


class LatencyProbeTest(unittest.TestCase):
    """measure_* turn ops budgets into measured milliseconds; DeviceSpec.for_latency converts back."""

    def test_measured_seconds_and_throughput_are_positive(self):
        from mixle.task import distill_structured_from_labels, measure_inference_seconds, measure_ops_per_second

        recs, labels = _make_records(150, 0)
        student = distill_structured_from_labels(recs, labels, seed=0)
        # secs and rate come from two independent wall-clock probes; under a loaded parallel runner a
        # scheduler preemption can inflate either one arbitrarily. The positivity/finiteness invariants
        # hold on every attempt; the cross-probe consistency check (rate IS ops per measured second,
        # not off by a unit conversion) retries with a generous same-order window instead of a tight
        # one-shot ratio.
        for _ in range(5):
            secs = measure_inference_seconds(student, recs[:20], repeats=2)
            self.assertGreater(secs, 0.0)
            rate = measure_ops_per_second(student, recs[:20], repeats=2)
            self.assertTrue(np.isfinite(rate))
            self.assertGreater(rate, 0.0)
            ratio = (footprint(student).ops / secs) / rate
            if 0.05 <= ratio <= 20.0:
                break
        else:
            self.fail(f"ops/secs vs measured rate disagreed by >20x on every attempt (last ratio {ratio:.3g})")

    def test_for_latency_converts_budget_arithmetic(self):
        dev = DeviceSpec.for_latency(10.0, 2_000_000.0, max_bytes=5000, torch_free=True)
        self.assertEqual(dev.max_ops, 20_000)  # 2e6 ops/s * 0.01 s
        self.assertEqual(dev.max_bytes, 5000)
        self.assertTrue(dev.torch_free)
        with self.assertRaises(ValueError):
            DeviceSpec.for_latency(0.0, 1e6)
        with self.assertRaises(ValueError):
            DeviceSpec.for_latency(10.0, -1.0)

    def test_probe_requires_inputs(self):
        from mixle.task import distill_structured_from_labels, measure_inference_seconds

        recs, labels = _make_records(60, 0)
        student = distill_structured_from_labels(recs, labels, seed=0)
        with self.assertRaises(ValueError):
            measure_inference_seconds(student, [])


class TaskFingerprintTest(unittest.TestCase):
    def test_record_task_fingerprint(self):
        from mixle.task import task_fingerprint

        recs, labels = _make_records(240, 0)
        fp = task_fingerprint(recs, labels)
        self.assertEqual(len(fp), 5)
        self.assertAlmostEqual(fp[0], np.log10(240), places=6)  # log10 examples
        self.assertEqual(fp[1], 2.0)  # two labels
        self.assertEqual(fp[2], 2.0)  # two fields
        self.assertAlmostEqual(fp[3], 0.5)  # one of two fields categorical
        self.assertGreater(fp[4], 0.9)  # near-balanced labels -> entropy ~ 1

    def test_text_task_fingerprint(self):
        from mixle.task import task_fingerprint

        fp = task_fingerprint(["some text"] * 50, ["a"] * 40 + ["b"] * 10)
        self.assertEqual(fp[2], 1.0)  # text = one field
        self.assertEqual(fp[3], 1.0)  # fully categorical
        self.assertLess(fp[4], 0.9)  # imbalanced labels -> entropy < 1


class DistillForEdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, _ = _make_records(240, 10)
        cls.val, _ = _make_records(120, 11)
        cls.teacher = RuleTeacher()

    def test_search_returns_feasible_student_and_valid_pareto(self):
        dev = DeviceSpec(max_bytes=200_000)  # generous: both families fit; search optimizes quality
        res = distill_for_edge(
            self.teacher, self.train, self.val, dev, space=_fast_edge_space(), n_init=3, n_iter=2, promote=2, seed=0
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

    def test_torch_free_device_yields_torch_free_student(self):
        # a torch-free device admits structured students AND int8-quantized MLPs (numpy inference);
        # whichever wins, the deployed artifact must carry no torch dependence.
        dev = DeviceSpec(torch_free=True)
        res = distill_for_edge(
            self.teacher, self.train, self.val, dev, space=_fast_edge_space(), n_init=3, n_iter=2, promote=2, seed=0
        )
        self.assertTrue(res.footprint.torch_free)
        if res.family == "mlp":
            self.assertEqual(res.recipe["bits"], 8)  # fp32 MLPs are pinned out on torch-free devices
        self.assertTrue(res.feasible)
        self.assertGreater(res.agreement, 0.7)

    def test_quantization_unlocks_byte_budgets_fp32_cannot_meet(self):
        # 1000 bytes, MLP-only space. The smallest fp32 MLP here is (64*4+4 + 4*2+2)*4 = 1080 bytes
        # -- over budget BY CONSTRUCTION, so the fp32 arm must come back infeasible. The int8 arm has
        # the same architecture space at ~1/4 the bytes, so the search must find a fitting student.
        dev = DeviceSpec(max_bytes=1000)

        def arm(bits):
            return distill_for_edge(
                self.teacher,
                self.train,
                self.val,
                dev,
                space=EdgeSpace(
                    families=("mlp",),
                    dim_choices=(64, 128),
                    hidden_range=(4, 12),
                    epochs_range=(30, 90),
                    bits_choices=bits,
                ),
                n_init=3,
                n_iter=2,
                promote=2,
                seed=0,
            )

        fp32 = arm((32,))
        int8 = arm((8,))
        self.assertFalse(fp32.feasible)  # deterministic: every fp32 candidate exceeds the budget
        self.assertTrue(int8.feasible)  # quantization brings the same shapes under it
        self.assertLessEqual(int8.footprint.bytes, 1000)
        self.assertEqual(int8.recipe["bits"], 8)
        self.assertTrue(int8.footprint.torch_free)
        self.assertGreater(int8.agreement, 0.7)  # squeezed 4x, still matches the teacher

    def test_design_model_warm_start_accumulates_and_matches_cold(self):
        dev = DeviceSpec(max_bytes=200_000)
        cold = distill_for_edge(
            self.teacher, self.train, self.val, dev, space=_fast_edge_space(), n_init=3, n_iter=2, promote=1, seed=0
        )
        n_ledger = len(cold.design)
        warm = distill_for_edge(
            self.teacher,
            self.train,
            self.val,
            dev,
            space=_fast_edge_space(),
            design=cold.design,
            n_init=2,  # halved seeding: the warm surrogate already covers the space
            n_iter=3,  # one extra BO iteration: without the mixture arm's extra candidate diversity,
            # this keeps the warm search's promoted finalists as reliably good as the cold baseline
            promote=2,
            seed=1,
        )
        self.assertGreater(len(warm.design), n_ledger)  # knowledge accumulates across searches
        self.assertGreaterEqual(warm.agreement, 0.9 * cold.agreement)  # cheaper warm run holds the line

    def test_designer_prefilter_passes_through_the_front_door(self):
        # a judge vetoing the structured half of the cube (p0 > 0.5) rides along without breaking
        # the search; the winner still fits the device and matches the teacher.
        dev = DeviceSpec(max_bytes=200_000)
        judge = lambda pt: "weak" if pt[0] > 0.5 else "good"  # noqa: E731
        res = distill_for_edge(
            self.teacher,
            self.train,
            self.val,
            dev,
            space=_tiny_space(),
            designer=judge,
            n_init=3,
            n_iter=2,
            promote=1,
            seed=0,
        )
        self.assertTrue(res.feasible)
        self.assertGreater(res.agreement, 0.7)

    def test_incompatible_design_model_is_rejected(self):
        dev = DeviceSpec(max_bytes=200_000)
        with self.assertRaises(ValueError):
            distill_for_edge(
                self.teacher, self.train, self.val, dev, space=_tiny_space(), design=DesignModel("other", 1), seed=0
            )
        # an un-fingerprinted ledger (old shape) is also rejected with the same clear error
        with self.assertRaises(ValueError):
            distill_for_edge(
                self.teacher,
                self.train,
                self.val,
                dev,
                space=_tiny_space(),
                design=DesignModel(_tiny_space().signature(), 1, n_fingerprint=0),
                seed=0,
            )

    def test_design_knowledge_transfers_across_different_tasks(self):
        # task A: 2-field records; task B: 3-field records (different fingerprint) -- ONE ledger
        # serves both: no compatibility error, rows accumulate, and B's search still succeeds.
        dev = DeviceSpec(max_bytes=200_000)
        a = distill_for_edge(
            self.teacher, self.train, self.val, dev, space=_fast_edge_space(), n_init=2, n_iter=1, promote=1, seed=0
        )
        rng = np.random.RandomState(5)
        recs3 = [(float(rng.normal()), float(rng.normal()), "p" if rng.random() < 0.5 else "q") for _ in range(200)]

        class T3:
            def __call__(self, records):
                if isinstance(records, list):
                    return ["a" if (t == "p") == (x > 0) else "b" for (x, _y, t) in records]
                x, _y, t = records
                return "a" if (t == "p") == (x > 0) else "b"

        n_after_a = len(a.design)
        b = distill_for_edge(
            T3(),
            recs3[:140],
            recs3[140:],
            dev,
            space=_fast_edge_space(),
            design=a.design,
            n_init=16,  # more LHS seeds (cheap: no surrogate fit) so a good candidate for this
            # harder 3-field task is reliably among the screens, without paying for extra BO
            # iterations (each of which fits a Gaussian-process surrogate -- the expensive step)
            n_iter=1,
            promote=6,  # promote more of those screens to full fidelity so the winner is picked
            # from a wide field rather than gambling on the single top screen
            seed=1,
        )
        self.assertGreater(len(b.design), n_after_a)  # shared ledger keeps growing across tasks
        self.assertTrue(b.feasible)
        self.assertGreater(b.agreement, 0.6)
        # the two tasks are distinguishable inside the ledger via their fingerprint coords
        fps = {tuple(row[-5:]) for row in b.design.X}
        self.assertGreaterEqual(len(fps), 2)

    def test_torch_free_with_fp32_only_mlp_space_raises(self):
        # an mlp-only space with NO quantized precision cannot serve a torch-free device...
        with self.assertRaises(ValueError):
            distill_for_edge(
                self.teacher,
                self.train,
                self.val,
                DeviceSpec(torch_free=True),
                space=EdgeSpace(families=("mlp",), bits_choices=(32,)),
                seed=0,
            )
        # ...but with int8 available, an mlp-only space IS servable torch-free (numpy inference)
        res = distill_for_edge(
            self.teacher,
            self.train,
            self.val,
            DeviceSpec(torch_free=True),
            space=EdgeSpace(families=("mlp",), dim_choices=(64, 128), hidden_range=(4, 24), epochs_range=(30, 90)),
            n_init=2,
            n_iter=1,
            promote=1,
            seed=0,
        )
        self.assertEqual(res.family, "mlp")
        self.assertEqual(res.recipe["bits"], 8)
        self.assertTrue(res.footprint.torch_free)


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
