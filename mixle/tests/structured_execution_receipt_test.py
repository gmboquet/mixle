"""A structured-execution receipt attests admission, not device affinity (MXR-080-0647)."""

import unittest

from mixle.experimental.typed_runtime.measurement import WorkMeasurement
from mixle.experimental.typed_runtime.structured_execution import StructuredEstimationReceipt


def _real_placement_and_work():
    """A genuine plan and work measurement, since the receipt now requires typed evidence."""
    from typed_structured_execution_test import _topology

    from mixle.experimental.typed_runtime.compiler import compile_update_graph
    from mixle.experimental.typed_runtime.topology import plan_structured_placement
    from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator

    model = MixtureDistribution([GaussianDistribution(float(i) - 1.5, 1.0) for i in range(4)], [0.25] * 4)
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(4)])
    graph = compile_update_graph(model, estimator, nobs=40.0)
    placement = plan_structured_placement(graph, _topology(), n_data=40)
    work = WorkMeasurement(
        node_type="MixtureDistribution",
        update_kind=graph.node(graph.root_node).contract.update_kind,
        backend="typed_model_parallel",
        wall_time_seconds=0.01,
        compute_units=graph.node(graph.root_node).cost.compute_units,
        observations=40.0,
        operation_count=1,
        extra={},
    )
    node_ids = tuple(row.node_id for row in placement.placements)
    devices = tuple(sorted({shard.device_id for row in placement.placements for shard in row.shards}))
    return placement, work, node_ids, devices


class ReceiptFieldTest(unittest.TestCase):
    def _receipt(self, **overrides) -> StructuredEstimationReceipt:
        placement, work, node_ids, devices = _real_placement_and_work()
        fields = dict(
            placement=placement,
            observations=40.0,  # must match work.observations: one run, one row count (MXR-080-0647)
            num_workers=2,
            worker_device_ids=devices[:2],
            execution_backend="local_numpy_thread_pool",
            parallel_node_ids=node_ids[:1],
            parallel_statistics_hash="abc",
            reference_statistics_hash=None,
            parallel_model_hash="def",
            reference_model_hash=None,
            exact_parity=None,
            work=work,
        )
        fields.update(overrides)
        return StructuredEstimationReceipt(**fields)

    def test_untyped_placement_or_work_is_refused(self):
        """A receipt whose evidence is None carries no evidence, yet every field reads as a claim."""
        for field in ("placement", "work"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, f"receipt {field} must be a"):
                    self._receipt(**{field: None})

    def test_an_invented_device_or_node_is_refused(self):
        with self.assertRaisesRegex(ValueError, "placement plan never placed"):
            self._receipt(num_workers=1, worker_device_ids=("gpu:99",))
        with self.assertRaisesRegex(ValueError, "placement plan does not contain"):
            self._receipt(parallel_node_ids=("no-such-node",))

    def test_an_atomic_run_may_name_no_parallel_node(self):
        # A graph with no shardable axis executes atomically and the executor emits exactly this.
        self.assertEqual(self._receipt(parallel_node_ids=()).parallel_node_ids, ())

    def test_a_repeated_parallel_node_is_refused(self):
        placement, _work, node_ids, _devices = _real_placement_and_work()
        del placement
        with self.assertRaisesRegex(ValueError, "repeats a parallel node id"):
            self._receipt(parallel_node_ids=(node_ids[0], node_ids[0]))

    def test_a_baseline_receipt_constructs(self):
        self.assertEqual(self._receipt().num_workers, 2)

    def test_more_workers_than_placed_slots_is_refused(self):
        # "workers may exceed placed topology devices, and receipts still claim them".
        _placement, _work, _nodes, devices = _real_placement_and_work()
        with self.assertRaisesRegex(ValueError, "cannot exceed the placement"):
            self._receipt(num_workers=999, worker_device_ids=devices[:1])

    def test_fewer_workers_than_slots_is_still_allowed(self):
        # A caller may ask for less than the placement capacity; that is a plan, not a forgery.
        self.assertEqual(self._receipt(num_workers=1).num_workers, 1)

    def test_a_non_positive_worker_count_is_refused(self):
        for bad in (0, -3, True, 1.5):
            with self.subTest(num_workers=bad):
                with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                    self._receipt(num_workers=bad)

    def test_a_repeated_device_slot_is_refused(self):
        _placement, _work, _nodes, devices = _real_placement_and_work()
        with self.assertRaisesRegex(ValueError, "repeats a worker device id"):
            self._receipt(num_workers=2, worker_device_ids=(devices[0], devices[0]))

    def test_a_blank_device_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "names nothing|never placed"):
            self._receipt(num_workers=1, worker_device_ids=("  ",))

    def test_an_anonymous_backend_or_hash_is_refused(self):
        for field in ("execution_backend", "parallel_statistics_hash", "parallel_model_hash"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must name something"):
                    self._receipt(**{field: ""})


class AffinityClaimTest(unittest.TestCase):
    """The work measurement must not report placement the executor does not enforce."""

    def _run(self):
        import numpy as np
        from typed_structured_execution_test import _topology

        from mixle.experimental.typed_runtime.structured_execution import run_structured_estimation_step
        from mixle.stats import (
            GaussianDistribution,
            GaussianEstimator,
            MixtureDistribution,
            MixtureEstimator,
            seq_encode,
        )

        rng = np.random.RandomState(5)
        model = MixtureDistribution([GaussianDistribution(float(i) - 1.5, 1.0) for i in range(4)], [0.25] * 4)
        estimator = MixtureEstimator([GaussianEstimator() for _ in range(4)])
        data = [float(rng.randn()) for _ in range(60)]
        return run_structured_estimation_step(
            seq_encode(data, model=model), estimator, model, _topology(), num_workers=3
        )

    def test_admission_and_affinity_are_reported_as_two_separate_facts(self):
        extra = self._run().receipt.work.extra
        # The old key asserted enforcement of a placement the ThreadPoolExecutor fold cannot pin.
        self.assertNotIn("placement_enforced", extra)
        self.assertIs(extra["placement_admitted"], True)
        self.assertIs(extra["device_affinity_enforced"], False)

    def test_the_device_list_is_still_published_for_the_admitted_slots(self):
        receipt = self._run().receipt
        self.assertEqual(receipt.worker_device_ids, ("cpu:0", "cpu:1", "cpu:2"))
        self.assertEqual(receipt.num_workers, 3)


if __name__ == "__main__":
    unittest.main()
