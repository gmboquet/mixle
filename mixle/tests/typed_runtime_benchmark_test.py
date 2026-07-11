"""Stage-0 measurement and failure-oracle tests."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    BenchmarkPoint,
    FailureKind,
    FailureLedger,
    FailureReceipt,
    ObjectiveTarget,
    TargetDirection,
    TimeToTargetTrace,
    UpdateKind,
    WorkMeasurement,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


class TimeToTargetTest:
    def test_trace_reports_time_operations_and_evaluations_separately(self):
        trace = TimeToTargetTrace(
            "gaussian-local",
            "typed-blocks",
            ObjectiveTarget("held-out-log-likelihood", TargetDirection.MAXIMIZE, -10.0),
        )
        trace.record(BenchmarkPoint(0, -20.0, 0.0))
        trace.record(
            BenchmarkPoint(
                1,
                -9.5,
                0.4,
                operation_count=7,
                model_evaluations=2,
                bytes_read=100,
                accepted_updates=1,
            )
        )
        trace.record(
            BenchmarkPoint(
                2,
                -9.0,
                0.8,
                operation_count=12,
                model_evaluations=4,
                bytes_read=200,
                accepted_updates=2,
            )
        )

        payload = trace.as_dict()
        assert payload["achieved"] is True
        assert payload["time_to_target_seconds"] == pytest.approx(0.4)
        assert payload["operations_to_target"] == 7
        assert payload["model_evaluations_to_target"] == 2
        json.dumps(payload, allow_nan=False)

    def test_minimization_target_and_tolerance(self):
        target = ObjectiveTarget("error", TargetDirection.MINIMIZE, 0.1, tolerance=0.01)
        assert not target.reached(0.12)
        assert target.reached(0.11)
        assert not target.reached(float("nan"))

    def test_non_cumulative_trace_is_rejected(self):
        trace = TimeToTargetTrace(
            "fixture",
            "candidate",
            ObjectiveTarget("objective", TargetDirection.MAXIMIZE, 1.0),
        )
        trace.record(BenchmarkPoint(1, 0.0, 1.0, operation_count=10))
        with pytest.raises(ValueError, match="counters cannot decrease"):
            trace.record(BenchmarkPoint(2, 0.5, 2.0, operation_count=9))
        with pytest.raises(ValueError, match="advance"):
            trace.record(BenchmarkPoint(1, 0.5, 2.0, operation_count=11))


class FailureOracleTest:
    def test_ledger_keeps_missed_negative_controls_visible(self):
        ledger = FailureLedger()
        caught = FailureReceipt(
            "proposal-commit",
            "stale-version",
            FailureKind.VERSION_MISMATCH,
            "base-version equality",
            expected_failure=True,
            detected=True,
            observed="proposal rejected",
        )
        missed = FailureReceipt(
            "proposal-commit",
            "bad-objective",
            FailureKind.OBJECTIVE_REGRESSION,
            "canary objective",
            expected_failure=True,
            detected=False,
            observed="regressing update committed",
        )
        clean = FailureReceipt(
            "proposal-commit",
            "valid-control",
            FailureKind.VERSION_MISMATCH,
            "base-version equality",
            expected_failure=False,
            detected=False,
            observed="valid proposal committed",
        )
        for receipt in (caught, missed, clean):
            ledger.record(receipt)

        assert ledger.failed_oracles == (missed,)
        assert ledger.as_dict()["all_oracles_passed"] is False
        with pytest.raises(ValueError, match="already recorded"):
            ledger.record(caught)


def test_work_measurement_has_explicit_io_operation_and_staleness_counters():
    measurement = WorkMeasurement(
        "GaussianDistribution",
        UpdateKind.EXACT_CLOSED_FORM,
        "cpu",
        0.01,
        operation_count=5,
        bytes_read=80,
        bytes_written=16,
        collective_bytes=8,
        staleness_steps=2,
    )
    assert measurement.as_dict() | {} == measurement.as_dict()
    assert measurement.as_dict()["operation_count"] == 5
    assert measurement.as_dict()["collective_bytes"] == 8
    assert measurement.as_dict()["staleness_steps"] == 2

    with pytest.raises(ValueError, match="non-negative"):
        WorkMeasurement("GaussianDistribution", UpdateKind.EXACT_CLOSED_FORM, "cpu", 0.01, bytes_read=-1)
