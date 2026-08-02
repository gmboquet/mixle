"""A structured-execution receipt attests admission, not device affinity (MXR-080-0647)."""

import unittest

from mixle.experimental.typed_runtime.structured_execution import StructuredEstimationReceipt


class ReceiptFieldTest(unittest.TestCase):
    def _receipt(self, **overrides) -> StructuredEstimationReceipt:
        fields = dict(
            placement=None,
            observations=10.0,
            num_workers=2,
            worker_device_ids=("d0", "d1"),
            execution_backend="local_numpy_thread_pool",
            parallel_node_ids=("root",),
            parallel_statistics_hash="abc",
            reference_statistics_hash=None,
            parallel_model_hash="def",
            reference_model_hash=None,
            exact_parity=None,
            work=None,
        )
        fields.update(overrides)
        return StructuredEstimationReceipt(**fields)

    def test_a_baseline_receipt_constructs(self):
        self.assertEqual(self._receipt().num_workers, 2)

    def test_more_workers_than_placed_slots_is_refused(self):
        # "workers may exceed placed topology devices, and receipts still claim them".
        with self.assertRaisesRegex(ValueError, "cannot exceed the placement"):
            self._receipt(num_workers=999, worker_device_ids=("d0",))

    def test_fewer_workers_than_slots_is_still_allowed(self):
        # A caller may ask for less than the placement capacity; that is a plan, not a forgery.
        self.assertEqual(self._receipt(num_workers=1).num_workers, 1)

    def test_a_non_positive_worker_count_is_refused(self):
        for bad in (0, -3, True, 1.5):
            with self.subTest(num_workers=bad):
                with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                    self._receipt(num_workers=bad)

    def test_a_repeated_device_slot_is_refused(self):
        with self.assertRaisesRegex(ValueError, "repeats a worker device id"):
            self._receipt(num_workers=2, worker_device_ids=("d0", "d0"))

    def test_a_blank_device_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "names nothing"):
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
