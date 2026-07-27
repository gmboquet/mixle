"""End-to-end typed structured model-parallel estimation tests."""

import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    ClusterTopology,
    LinkProfile,
    TopologyDevice,
    run_structured_estimation_step,
)
from mixle.experimental.typed_runtime.structured_execution import _model_hash
from mixle.stats import (
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    PoissonDistribution,
    PoissonEstimator,
    seq_encode,
)
from mixle.utils.parallel.planner import DeviceSpec

pytestmark = [pytest.mark.experimental]


def _topology():
    devices = tuple(
        TopologyDevice(
            "cpu:%d" % index,
            "host",
            "local-island",
            "local",
            "local",
            DeviceSpec("cpu:%d" % index, memory_bytes=1_000_000_000, throughput=1.0),
        )
        for index in range(3)
    )
    links = tuple(
        LinkProfile(source.device_id, target.device_id, 1.0e-6, 10.0e9)
        for source in devices
        for target in devices
        if source != target
    )
    return ClusterTopology(devices, links)


def test_component_parallel_step_is_bitwise_equal_to_serial_statistics_and_model():
    rng = np.random.RandomState(5)
    model = MixtureDistribution(
        [GaussianDistribution(float(index) - 1.5, 1.0) for index in range(4)],
        [0.25] * 4,
    )
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(4)])
    data = [float(rng.randn() + 2.0 * (rng.randint(4) - 1.5)) for _ in range(200)]
    encoded = seq_encode(data, model=model)

    result = run_structured_estimation_step(encoded, estimator, model, _topology(), num_workers=3)

    assert result.receipt.exact_parity is True
    assert result.receipt.parallel_statistics_hash == result.receipt.reference_statistics_hash
    assert result.receipt.parallel_model_hash == result.receipt.reference_model_hash
    assert result.receipt.parallel_node_ids == ("n0000",)
    assert result.receipt.worker_device_ids == ("cpu:0", "cpu:1", "cpu:2")
    assert result.receipt.execution_backend == "local_numpy_thread_pool"
    assert len(result.receipt.placement.placement("n0000").shards) == 3
    assert result.receipt.work.backend == "typed_model_parallel"
    json.dumps(result.receipt.as_dict(), allow_nan=False)


def test_undeclared_factor_axis_runs_atomically_and_matches_serial_path():
    rng = np.random.RandomState(8)
    model = CompositeDistribution((GaussianDistribution(0.0, 1.0), PoissonDistribution(1.0)))
    estimator = CompositeEstimator((GaussianEstimator(), PoissonEstimator()))
    data = [(float(rng.randn()), int(rng.poisson(3.0))) for _ in range(120)]
    result = run_structured_estimation_step(seq_encode(data, model=model), estimator, model, _topology())
    assert result.receipt.exact_parity is True
    assert result.receipt.parallel_node_ids == ()
    assert result.receipt.num_workers == 1


def test_worker_count_must_be_positive_and_within_placed_capacity():
    model = MixtureDistribution(
        [GaussianDistribution(float(index), 1.0) for index in range(4)],
        [0.25] * 4,
    )
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(4)])
    encoded = seq_encode([0.0, 1.0], model=model)
    with pytest.raises(ValueError, match="positive"):
        run_structured_estimation_step(encoded, estimator, model, _topology(), num_workers=0)
    with pytest.raises(ValueError, match="placement capacity"):
        run_structured_estimation_step(encoded, estimator, model, _topology(), num_workers=4)


def test_model_hash_has_no_unstable_string_fallback():
    with pytest.raises(TypeError, match="deterministic to_json"):
        _model_hash(object())


def test_bare_encoding_requires_explicit_weights_and_can_skip_reference():
    model = GaussianDistribution(0.0, 1.0)
    estimator = GaussianEstimator()
    payload = model.dist_to_encoder().seq_encode([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="explicit weights"):
        run_structured_estimation_step(payload, estimator, model, _topology())

    result = run_structured_estimation_step(
        payload,
        estimator,
        model,
        _topology(),
        weights=np.array([1.0, 2.0, 1.0]),
        verify_reference=False,
    )
    assert result.receipt.observations == pytest.approx(4.0)
    assert result.receipt.exact_parity is None
