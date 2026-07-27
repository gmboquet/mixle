"""Benchmark evidence times fit only and fails closed on invalid likelihood parity."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "benchmarks"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bench():
    module = _load(_BENCH / "_bench.py", "_benchmark_contract_bench")
    sys.modules["_bench"] = module
    yield module
    sys.modules.pop("_bench", None)


@pytest.fixture
def runner(bench):
    scripts = str(_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return _load(_BENCH / "run_benchmarks.py", "_benchmark_contract_runner")


def test_timer_excludes_prepare_and_evaluate_and_checks_every_fit(bench, monkeypatch):
    clock = [0.0]
    calls = {"prepare": 0, "fit": 0, "evaluate": 0}

    def prepare():
        calls["prepare"] += 1
        clock[0] += 100.0
        return calls["prepare"]

    def fit(state):
        calls["fit"] += 1
        clock[0] += 2.0
        return state

    def evaluate(state):
        calls["evaluate"] += 1
        clock[0] += 100.0
        return -3.0, 4

    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    result = bench.timed(bench.BenchmarkWorkload(prepare, fit, evaluate), reps=3)
    assert result["sec"] == 2.0
    assert result["sec_min"] == 2.0
    assert calls == {"prepare": 4, "fit": 4, "evaluate": 4}
    assert len(result["repetitions"]) == 3


def test_timer_rejects_unvalidated_or_nonreproducible_repetitions(bench):
    values = iter([(-1.0, 2), (-1.0, 2), (-2.0, 2)])
    workload = bench.BenchmarkWorkload(lambda: None, lambda state: state, lambda state: next(values))
    result = bench.timed(workload, reps=2)
    assert result["failed"] == "NonReproducibleEvaluation"
    with pytest.raises(ValueError):
        bench.timed(workload, reps=0)
    with pytest.raises(TypeError):
        bench.timed(lambda: None, reps=1)


def test_mixle_adapter_uses_preencoded_fit_and_validates_repetition(bench):
    data, initial = bench.make_full_cov_gmm(80, 2, 2)
    result = bench.timed(bench.gmm_mixle(data, initial, 1), reps=1)
    assert result.get("failed") is None
    assert result["sec"] >= 0.0
    assert result["repetitions"][0]["mean_ll"] == pytest.approx(result["mean_ll"])


def test_parity_requires_mixle_and_an_independent_matching_reference(runner):
    valid = {
        "mixle": {"mean_ll": -4.0, "sec": 0.1},
        "reference": {"mean_ll": -4.0001, "sec": 0.2},
    }
    delta, ok = runner._check_parity("valid", valid)
    assert ok and delta < runner.LL_TOL
    for invalid in (
        {"mixle": {"mean_ll": -4.0, "sec": 0.1}},
        {
            "mixle": {"failed": "ValueError", "mean_ll": None},
            "one": {"mean_ll": -4.0},
            "two": {"mean_ll": -4.0},
        },
        {"reference": {"mean_ll": -4.0}},
        {
            "mixle": {"mean_ll": -4.0, "sec": 0.1},
            "reference": {"mean_ll": -3.0, "sec": 0.2},
        },
    ):
        with pytest.raises(runner.BenchmarkValidationError):
            runner._check_parity("invalid", invalid)


def test_output_directory_is_created_and_writable_before_compute(runner, tmp_path):
    target = tmp_path / "new" / "results"
    assert runner._prepare_output_directory(target) == target.resolve()
    assert target.is_dir()
