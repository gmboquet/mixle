import unittest

import numpy as np
import pytest

from mixle.utils.parallel.training_contracts import (
    CollectiveKind,
    DistributedUpdate,
    ParallelAxis,
    ParallelPlan,
    ParameterLayout,
    PayloadKind,
    StateLayout,
    StepReceipt,
)
from mixle.utils.parallel.training_launchers import (
    LightningFabricLauncher,
    RayTrainLauncher,
    _TopologyBoundFabric,
    attest_worker_coordinate,
)

pytestmark = pytest.mark.fast


class ParallelPlanContractTest(unittest.TestCase):
    def test_dimensions_are_canonical_exact_positive_integers(self):
        plan = ParallelPlan(tp=np.int64(2))
        self.assertIs(type(plan.tp), int)
        self.assertEqual(plan.axis_sizes[ParallelAxis.TP], 2)
        self.assertEqual(plan.as_dict()["tp"], 2)
        for value in (1.9, True, 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ParallelPlan(tp=value)

    def test_world_size_validation_does_not_truncate(self):
        with self.assertRaises(ValueError):
            ParallelPlan().validate_world_size(1.9)


class TrainingReceiptContractTest(unittest.TestCase):
    def test_parameter_layout_rejects_duplicate_axes_and_invalid_shapes(self):
        with self.assertRaises(ValueError):
            ParameterLayout("p", (2,), placements=(("tp", "a"), ("tp", "b")))
        for shape in ((2.5,), (-1,), (True,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                ParameterLayout("p", shape)

    def test_distributed_update_rejects_invalid_typed_identity(self):
        base = dict(
            node_id="n0",
            payload=PayloadKind.GRADIENT,
            collective=CollectiveKind.ALL_REDUCE,
            mesh_axes=(ParallelAxis.TP,),
            state_layout=StateLayout.SHARDED,
            exact=False,
        )
        with self.assertRaises(ValueError):
            DistributedUpdate(**(base | {"mesh_axes": (ParallelAxis.TP, ParallelAxis.TP)}))
        with self.assertRaises(TypeError):
            DistributedUpdate(**(base | {"payload": "gradient"}))
        with self.assertRaises(ValueError):
            DistributedUpdate(**(base | {"numerics_sample_count": -1}))

    def test_step_receipt_rejects_impossible_measurements(self):
        base = dict(
            step=1,
            loss=0.5,
            local_examples=2,
            local_tokens=4,
            microbatches=1,
            accumulation_steps=1,
            data_parallel_size=2,
            optimizer="adamw",
            precision="bf16",
        )
        for updates in (
            {"loss": -1.0},
            {"loss": np.nan},
            {"step": 1.5},
            {"local_examples": -1},
            {"collective_bytes": -1},
        ):
            with self.subTest(updates=updates), self.assertRaises((TypeError, ValueError)):
                StepReceipt(**(base | updates))


class LauncherTopologyContractTest(unittest.TestCase):
    def test_every_rank_maps_to_one_unique_planned_coordinate(self):
        plan = ParallelPlan(dp_replicate=2, tp=2)
        attestations = [
            attest_worker_coordinate(plan, actual_world_size=4, rank=rank)
            for rank in range(4)
        ]
        self.assertEqual(len({attestation.coordinate for attestation in attestations}), 4)
        with self.assertRaises(ValueError):
            attest_worker_coordinate(plan, actual_world_size=2, rank=0)
        with self.assertRaises(ValueError):
            attest_worker_coordinate(plan, actual_world_size=4, rank=4)

    def test_lightning_rejects_topology_mismatch_before_optional_import(self):
        with self.assertRaises(ValueError):
            LightningFabricLauncher().create(
                plan=ParallelPlan(dp_replicate=4),
                devices=1,
                num_nodes=1,
            )
        with self.assertRaises(ValueError):
            LightningFabricLauncher().create(
                plan=ParallelPlan(dp_replicate=4),
                devices="auto",
                num_nodes=2,
            )

    def test_ray_rejects_resource_conflicts_before_optional_import(self):
        launcher = RayTrainLauncher()
        with self.assertRaises(ValueError):
            launcher.launch(
                lambda config: config,
                plan=ParallelPlan(),
                use_gpu=False,
                resources_per_worker={"GPU": 1.0},
            )
        with self.assertRaises(ValueError):
            launcher.launch(
                lambda config: config,
                plan=ParallelPlan(),
                use_gpu=True,
                resources_per_worker={"GPU": 0.0},
            )

    def test_fabric_binding_produces_runtime_attestation(self):
        class FakeFabric:
            world_size = 2
            global_rank = 1

            def launch(self):
                return "launched"

        binding = _TopologyBoundFabric(FakeFabric(), ParallelPlan(dp_replicate=2))
        with self.assertRaises(RuntimeError):
            binding.attest_topology()
        self.assertEqual(binding.launch(), "launched")
        self.assertEqual(binding.attest_topology().rank, 1)


if __name__ == "__main__":
    unittest.main()
