import importlib
import os
import tempfile
import unittest

import numpy as np

from pysp.parallel import (
    CalibrationCatalog,
    DeviceSpec,
    EncodedDataHandle,
    LocalEncodedData,
    Resources,
    calibrate_resources,
    encoded_data,
    estimate_model_nbytes,
    is_encoded_data_handle,
    model_sharding_plan,
    plan,
)
from pysp.stats import (
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
    seq_estimate,
    seq_initialize,
    seq_log_density_sum,
)
from pysp.utils.estimation import optimize

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from pysp.engines import TorchEngine
    from pysp.stats import ResidentEncodedPayload
else:
    torch = None
    TorchEngine = None
    ResidentEncodedPayload = None


class PlacementPlanningTestCase(unittest.TestCase):
    def test_resources_from_external_orchestrator_shapes_need_no_optional_imports(self):
        class FakeComm:
            def Get_size(self):
                return 3

        class FakeDaskClient:
            def scheduler_info(self):
                return {
                    "workers": {
                        "tcp://a": {"nthreads": 2, "memory_limit": 1000},
                        "tcp://b": {"nthreads": 4, "memory_limit": 2000},
                    }
                }

        class FakeSparkContext:
            defaultParallelism = 5

        mpi = Resources.from_mpi(FakeComm(), memory_bytes=900)
        dask = Resources.from_dask(FakeDaskClient())
        spark = Resources.from_spark(FakeSparkContext(), memory_bytes=5000)

        self.assertEqual(len(mpi.devices), 3)
        self.assertEqual(mpi.devices[0].memory_bytes, 300)
        self.assertEqual(tuple(device.throughput for device in dask.devices), (2.0, 4.0))
        self.assertEqual(dask.devices[1].memory_bytes, 2000)
        self.assertEqual(len(spark.devices), 5)
        self.assertEqual(spark.devices[0].memory_bytes, 1000)

    def test_plan_estimates_rows_and_bytes_from_model_encoder(self):
        model = GaussianDistribution(0.0, 1.0)
        data = list(np.linspace(-2.0, 2.0, 25))

        placement = plan(
            data=data, model=model, estimator=model.estimator(), resources=Resources.single_cpu(memory_bytes=10_000_000)
        )

        self.assertEqual(placement.total_rows, len(data))
        self.assertEqual(sum(shard.size for shard in placement.shards), len(data))
        self.assertGreater(placement.encoded_row_bytes, 0.0)
        self.assertGreater(placement.model_bytes, 0)
        self.assertGreaterEqual(placement.statistic_bytes, 0)
        self.assertIn("Placement(", str(placement))
        self.assertEqual(placement.to_dict()["total_rows"], len(data))

    def test_weighted_resources_get_proportional_chunks(self):
        resources = Resources.from_specs(
            (
                DeviceSpec("cpu:slow", memory_bytes=1_000_000, throughput=1.0),
                DeviceSpec("cpu:fast", memory_bytes=1_000_000, throughput=3.0),
            )
        )
        data = list(np.linspace(0.0, 1.0, 100))

        placement = plan(
            data=data, model=GaussianDistribution(0.0, 1.0), estimator=GaussianEstimator(), resources=resources
        )
        slow_rows = sum(shard.size for shard in placement.for_device("cpu:slow"))
        fast_rows = sum(shard.size for shard in placement.for_device("cpu:fast"))

        self.assertEqual(slow_rows + fast_rows, 100)
        self.assertLess(slow_rows, fast_rows)
        self.assertEqual(slow_rows, 25)
        self.assertEqual(fast_rows, 75)

    def test_manual_num_chunks_round_robins_over_weighted_devices(self):
        resources = Resources.from_specs(
            (
                DeviceSpec("cpu:slow", memory_bytes=1_000_000, throughput=1.0),
                DeviceSpec("cpu:fast", memory_bytes=1_000_000, throughput=3.0),
            )
        )
        placement = plan(
            data=list(range(40)),
            model=GaussianDistribution(0.0, 1.0),
            estimator=GaussianEstimator(),
            resources=resources,
            num_chunks=4,
        )

        self.assertEqual(len(placement.shards), 4)
        self.assertEqual(sum(shard.size for shard in placement.shards), 40)
        self.assertEqual(sum(1 for shard in placement.shards if shard.device.name == "cpu:fast"), 3)

    def test_memory_cap_splits_large_shard(self):
        data = list(np.linspace(-2.0, 2.0, 200))
        model = MixtureDistribution(
            [
                GaussianDistribution(-1.0, 1.0),
                GaussianDistribution(1.0, 1.0),
            ],
            [0.5, 0.5],
        )
        resources = Resources.single_cpu(memory_bytes=1_000)

        placement = plan(data=data, model=model, estimator=model.estimator(), resources=resources, safety_factor=1.0)

        self.assertGreater(len(placement.shards), 1)
        self.assertEqual(sum(shard.size for shard in placement.shards), len(data))
        self.assertTrue(all(shard.size > 0 for shard in placement.shards))

    def test_estimate_model_nbytes_sees_distribution_parameters(self):
        model = MixtureDistribution(
            [
                GaussianDistribution(-1.0, 1.0),
                GaussianDistribution(1.0, 1.0),
            ],
            [0.25, 0.75],
        )

        self.assertGreater(estimate_model_nbytes(model), 0)

    def test_calibrate_resources_updates_throughput_from_kernel_scoring(self):
        model = GaussianDistribution(0.0, 1.0)
        data = list(np.linspace(-2.0, 2.0, 20))
        resources = Resources.single_cpu(memory_bytes=1_000_000, throughput=1.0)

        calibrated = calibrate_resources(data, model, resources=resources, sample_size=10, repeats=1)

        self.assertEqual(len(calibrated.devices), 1)
        self.assertGreater(calibrated.devices[0].throughput, 0.0)
        self.assertTrue(np.isfinite(calibrated.devices[0].throughput))

    def test_calibrate_resources_can_time_estep_and_em_workloads(self):
        model = GaussianDistribution(0.0, 1.0)
        data = list(np.linspace(-2.0, 2.0, 20))
        resources = Resources.single_cpu(memory_bytes=1_000_000, throughput=1.0)

        estep = calibrate_resources(data, model, resources=resources, sample_size=10, repeats=1, workload="estep")
        em = calibrate_resources(
            data, model, resources=resources, estimator=GaussianEstimator(), sample_size=10, repeats=1, workload="em"
        )

        self.assertEqual(len(estep.devices), 1)
        self.assertEqual(len(em.devices), 1)
        self.assertGreater(estep.devices[0].throughput, 0.0)
        self.assertGreater(em.devices[0].throughput, 0.0)
        self.assertTrue(np.isfinite(estep.devices[0].throughput))
        self.assertTrue(np.isfinite(em.devices[0].throughput))

    def test_calibrate_resources_rejects_unknown_workload(self):
        with self.assertRaises(ValueError):
            calibrate_resources([0.0, 1.0], GaussianDistribution(0.0, 1.0), workload="unknown")

    def test_resources_json_round_trip_preserves_calibration(self):
        resources = Resources.from_specs(
            (
                DeviceSpec("cpu:0", memory_bytes=1_000_000, throughput=12.5, precision="float32"),
                DeviceSpec(
                    "cuda:0", kind="cuda", memory_bytes=2_000_000, engine="torch", throughput=40.0, precision="float32"
                ),
            )
        )

        loaded = Resources.from_json(resources.to_json(sort_keys=True))

        self.assertEqual(loaded.devices, resources.devices)
        with tempfile.NamedTemporaryFile() as f:
            resources.save(f.name, sort_keys=True)
            from_disk = Resources.load(f.name)
        self.assertEqual(from_disk.devices, resources.devices)

    def test_calibration_catalog_records_model_workload_resources(self):
        model = GaussianDistribution(0.0, 1.0)
        data = list(np.linspace(-2.0, 2.0, 20))
        resources = Resources.single_cpu(memory_bytes=1_000_000, throughput=1.0)
        catalog = CalibrationCatalog()

        calibrated = calibrate_resources(
            data,
            model,
            resources=resources,
            sample_size=10,
            repeats=1,
            workload="score",
            precision="float64",
            catalog=catalog,
        )

        self.assertEqual(len(catalog.records), 1)
        record = catalog.latest(model_type="GaussianDistribution", workload="score", precision="float64")
        self.assertIsNotNone(record)
        self.assertEqual(record.sample_size, 10)
        self.assertEqual(record.repeats, 1)
        self.assertEqual(record.row_count, len(data))
        self.assertEqual(record.model_bytes, estimate_model_nbytes(model))
        self.assertEqual(record.resources.devices, calibrated.devices)
        self.assertEqual(catalog.resources_for("GaussianDistribution", "score", "float64").devices, calibrated.devices)

    def test_calibration_catalog_path_round_trip(self):
        model = GaussianDistribution(0.0, 1.0)
        data = list(np.linspace(-2.0, 2.0, 20))
        resources = Resources.single_cpu(memory_bytes=1_000_000, throughput=1.0)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "calibration.json")
            calibrated = calibrate_resources(
                data,
                model,
                resources=resources,
                estimator=GaussianEstimator(),
                sample_size=10,
                repeats=1,
                workload="em",
                catalog_path=path,
            )
            loaded = CalibrationCatalog.load(path)

        self.assertEqual(len(loaded.records), 1)
        record = loaded.latest(model_type="GaussianDistribution", workload="em")
        self.assertIsNotNone(record)
        self.assertEqual(record.estimator_type, "GaussianEstimator")
        self.assertGreater(record.statistic_bytes, 0)
        self.assertEqual(record.resources.devices, calibrated.devices)

    def test_model_sharding_plan_covers_components_by_throughput(self):
        model = MixtureDistribution(
            [
                GaussianDistribution(-2.0, 1.0),
                GaussianDistribution(-1.0, 1.0),
                GaussianDistribution(0.0, 1.0),
                GaussianDistribution(1.0, 1.0),
                GaussianDistribution(2.0, 1.0),
            ],
            [0.2] * 5,
        )
        resources = Resources.from_specs(
            (
                DeviceSpec("cpu:slow", throughput=1.0),
                DeviceSpec("cpu:fast", throughput=3.0),
            )
        )

        shards = model_sharding_plan(model, resources, estimator=model.estimator())

        self.assertEqual([(s.component_start, s.component_stop) for s in shards], [(0, 1), (1, 5)])
        self.assertEqual(sum(s.num_components for s in shards), model.num_components)
        self.assertGreater(shards[1].parameter_bytes, shards[0].parameter_bytes)
        self.assertGreater(shards[1].statistic_bytes, shards[0].statistic_bytes)

    def test_model_sharding_plan_requires_component_model(self):
        with self.assertRaises(ValueError):
            model_sharding_plan(GaussianDistribution(0.0, 1.0), Resources.single_cpu())


class LocalEncodedDataTestCase(unittest.TestCase):
    def test_encoded_data_protocol_and_factory_preserve_existing_handles(self):
        model = GaussianDistribution(0.0, 1.0)
        data = list(np.linspace(-1.0, 1.0, 8))

        handle = encoded_data(data, model=model, estimator=GaussianEstimator())

        try:
            self.assertIsInstance(handle, EncodedDataHandle)
            self.assertIsInstance(handle, LocalEncodedData)
            self.assertTrue(is_encoded_data_handle(handle))
            self.assertFalse(is_encoded_data_handle(data))
            self.assertIs(encoded_data(handle), handle)
        finally:
            handle.close()

    def test_local_handle_dispatch_matches_serial_log_density_and_estimate(self):
        model = GaussianDistribution(0.5, 1.5)
        data = list(np.linspace(-2.0, 2.0, 40))
        estimator = GaussianEstimator()
        enc_serial = seq_encode(data, model=model)

        with LocalEncodedData(
            data, model=model, estimator=estimator, resources=Resources.local(num_cpus=2), sub_chunks=2
        ) as enc:
            cnt_h, ll_h = seq_log_density_sum(enc, model)
            fitted_h = seq_estimate(enc, estimator, model)

        cnt_s, ll_s = seq_log_density_sum(enc_serial, model)
        fitted_s = seq_estimate(enc_serial, estimator, model)

        self.assertEqual(cnt_h, cnt_s)
        self.assertAlmostEqual(ll_h, ll_s, places=10)
        self.assertAlmostEqual(fitted_h.mu, fitted_s.mu, places=10)
        self.assertAlmostEqual(fitted_h.sigma2, fitted_s.sigma2, places=10)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_local_handle_keeps_engine_resident_payloads_for_scoring(self):
        model = MixtureDistribution(
            [
                GaussianDistribution(-1.0, 0.8),
                GaussianDistribution(1.5, 1.2),
            ],
            [0.45, 0.55],
        )
        data = model.sampler(seed=8).sample(size=50)
        estimator = model.estimator()
        enc_serial = seq_encode(data, model=model)
        engine = TorchEngine(dtype=torch.float64)

        with LocalEncodedData(data, model=model, estimator=estimator, engine=engine, sub_chunks=2) as enc:
            chunk_payload = enc.shards[0].chunks[0][1]
            self.assertIsInstance(chunk_payload, ResidentEncodedPayload)
            self.assertIsInstance(chunk_payload.engine_payload, torch.Tensor)
            self.assertIsInstance(chunk_payload.host_payload, np.ndarray)
            cnt_h, ll_h = seq_log_density_sum(enc, model)
            fitted_h = seq_estimate(enc, estimator, model)

        cnt_s, ll_s = seq_log_density_sum(enc_serial, model)
        fitted_s = seq_estimate(enc_serial, estimator, model)

        self.assertEqual(cnt_h, cnt_s)
        self.assertAlmostEqual(ll_h, ll_s, places=10)
        np.testing.assert_allclose(fitted_h.w, fitted_s.w, rtol=1.0e-10, atol=1.0e-10)
        for got, exp in zip(fitted_h.components, fitted_s.components):
            self.assertAlmostEqual(got.mu, exp.mu, places=10)
            self.assertAlmostEqual(got.sigma2, exp.sigma2, places=10)

    def test_local_handle_initialize_and_optimize_use_unified_protocol(self):
        truth = MixtureDistribution(
            [
                GaussianDistribution(-2.0, 0.5),
                GaussianDistribution(2.0, 0.5),
            ],
            [0.45, 0.55],
        )
        data = truth.sampler(seed=4).sample(size=120)
        start = MixtureDistribution(
            [
                GaussianDistribution(-1.0, 2.0),
                GaussianDistribution(1.0, 2.0),
            ],
            [0.5, 0.5],
        )
        estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        enc_serial = seq_encode(data, model=start)
        _, ll0 = seq_log_density_sum(enc_serial, start)

        with LocalEncodedData(
            data, model=start, estimator=estimator, resources=Resources.local(num_cpus=2), sub_chunks=2
        ) as enc:
            initialized = seq_initialize(enc, estimator, np.random.RandomState(2), p=0.5)
            fitted = optimize(None, estimator, enc_data=enc, prev_estimate=start, max_its=8, delta=None)
            _, ll1 = seq_log_density_sum(enc, fitted)

        self.assertIsInstance(initialized, MixtureDistribution)
        self.assertGreater(ll1, ll0)
        self.assertAlmostEqual(float(np.sum(fitted.w)), 1.0)


if __name__ == "__main__":
    unittest.main()
