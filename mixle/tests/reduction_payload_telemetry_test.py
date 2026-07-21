"""Worklist D8.4 -- measure reduction/serialization overhead so scale-out claims are grounded.

When a fit is distributed, each worker reduces its shard to a *sufficient-statistic
payload* -- exactly ``pickle.dumps((count, accumulator.value()))`` -- which is reduced to
the root (a gather-and-fold or an O(log W) reduce tree, backend-dependent), and the
re-estimated model broadcast back (see ``mixle/utils/parallel/mpi.py``). The economics of
distribution hinge on one fact: that payload is O(model parameters), **independent of the
dataset size N**, while per-worker compute is O(N / workers). So distribution pays off once
per-shard compute dominates the fixed reduce + fold + broadcast overhead.

This test measures the payload directly (no cluster needed) and pins the property the
scale-out guidance depends on: the payload does not grow with N. It also checks the
payload round-trips through pickle (the reduce path's serialization) and times
serialize/deserialize so a regression that bloats the payload is caught.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pytest

from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
)

_SCALE_OUT_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "scale-out-economics.rst"

_PROTO = pickle.HIGHEST_PROTOCOL


def _reduce_payload(estimator, prev_model, data):
    """Reproduce a worker's reduce payload: (count, accumulator.value()) pickled.

    Mirrors ``MPIEncodedData._local_update`` so the measured bytes are the real gather
    payload, not a proxy.
    """
    enc = prev_model.dist_to_encoder().seq_encode(data)
    accumulator = estimator.accumulator_factory().make()
    accumulator.seq_update(enc, np.ones(len(data)), prev_model)
    local = (float(len(data)), accumulator.value())
    return pickle.dumps(local, protocol=_PROTO), local


def _ser_deser_us(local) -> tuple[float, float]:
    blob = pickle.dumps(local, protocol=_PROTO)
    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
        pickle.dumps(local, protocol=_PROTO)
    ser = (time.perf_counter() - t) / reps * 1e6
    t = time.perf_counter()
    for _ in range(reps):
        pickle.loads(blob)
    deser = (time.perf_counter() - t) / reps * 1e6
    return ser, deser


def _gauss():
    return GaussianEstimator(), GaussianDistribution(0.0, 1.0)


def _mixture(k=5):
    est = MixtureEstimator([GaussianEstimator() for _ in range(k)])
    model = MixtureDistribution([GaussianDistribution(float(i), 1.0) for i in range(k)], [1.0 / k] * k)
    return est, model


def test_payload_is_constant_in_dataset_size() -> None:
    """The gather payload must not grow with N -- the premise of the crossover claim."""
    for make in (_gauss, _mixture):
        est, model = make()
        small = np.random.RandomState(0).normal(0, 1, 2_000).tolist()
        large = np.random.RandomState(1).normal(0, 1, 200_000).tolist()
        blob_small, _ = _reduce_payload(est, model, small)
        blob_large, _ = _reduce_payload(est, model, large)
        # 100x the data must not meaningfully change the sufficient-statistic payload.
        assert abs(len(blob_large) - len(blob_small)) <= 16, (
            f"{make.__name__}: reduce payload grew with N "
            f"({len(blob_small)} B -> {len(blob_large)} B); it must be O(model), not O(N)"
        )


def test_payload_scales_with_model_not_data() -> None:
    """A bigger model has a bigger payload; that is the axis the payload depends on."""
    data = np.random.RandomState(0).normal(0, 1, 5_000).tolist()
    g_est, g_model = _gauss()
    m_est, m_model = _mixture(5)
    g_blob, _ = _reduce_payload(g_est, g_model, data)
    m_blob, _ = _reduce_payload(m_est, m_model, data)
    assert len(m_blob) > len(g_blob), (
        "a 5-component mixture should carry a larger sufficient-statistic payload than a "
        "single Gaussian on the same data"
    )


def test_payload_round_trips_and_is_small() -> None:
    """The payload must survive the reduce path's pickle and stay small for simple models."""
    est, model = _gauss()
    data = np.random.RandomState(0).normal(0, 1, 10_000).tolist()
    blob, local = _reduce_payload(est, model, data)
    count, stats = pickle.loads(blob)
    assert count == 10_000.0
    assert stats is not None
    # A single Gaussian's sufficient statistics are a few hundred bytes, flat in N.
    assert len(blob) < 4_096, f"Gaussian reduce payload unexpectedly large: {len(blob)} B"
    ser, deser = _ser_deser_us(local)
    assert ser >= 0 and deser >= 0  # timing is telemetry, not asserted to a threshold


def test_scale_out_doc_states_when_the_backend_helps() -> None:
    """D8.4 acceptance: scale-out docs state when the backend is expected to help."""
    if not _SCALE_OUT_DOC.is_file():
        pytest.skip("docs/scale-out-economics.rst not found")
    text = _SCALE_OUT_DOC.read_text(encoding="utf-8").lower()
    # It must talk about the payload being independent of N, and about when to distribute.
    assert "payload" in text and ("o(model)" in text or "not o(data)" in text or "not with" in text or "flat" in text)
    assert "helps" in text and "does not help" in text, (
        "the doc must state both when distribution helps and when it does not (losses as well as wins)"
    )
