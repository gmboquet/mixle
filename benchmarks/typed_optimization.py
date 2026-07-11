"""Benchmark selective and geometry-aware optimization against honest baselines.

The three panels deliberately distinguish:

* an exact implementation optimization that should preserve the fixed-iteration result;
* a selective block scheduler that must reach a shared objective target;
* an experimental routed neural optimizer compared with AdamW at identical batch semantics.

Run ``python benchmarks/typed_optimization.py --quick`` for the short matrix or omit
``--quick`` for larger local workloads. Results are provenance stamped.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mixle.experimental.typed_runtime import run_graph_memory_pilot  # noqa: E402
from mixle.inference import optimize  # noqa: E402
from mixle.inference.block_em import run_block_em  # noqa: E402
from mixle.inference.em import PosteriorTransformEM, observed_log_likelihood  # noqa: E402
from mixle.stats import (  # noqa: E402
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
    seq_encode,
)
from scripts.benchmark_provenance import stamp_result  # noqa: E402

ProblemFactory = Callable[[], tuple[Any, Any, Any]]
_SEEDS = (11, 19, 29, 37, 43, 53, 61)


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _median_available(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return float(statistics.median(values)) if values else None


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("mixle", "numpy", "scipy", "torch"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _scalar_problem(seed: int, nobs: int) -> tuple[Any, Any, Any]:
    truth = MixtureDistribution(
        [GaussianDistribution(-5.0, 0.6), GaussianDistribution(5.0, 0.6)],
        [0.5, 0.5],
    )
    data = truth.sampler(seed=seed).sample(size=nobs)
    locations = (-0.3, 0.3, -14.0, 14.0, -40.0, 40.0, -70.0, 70.0)
    model = MixtureDistribution(
        [GaussianDistribution(location, 3.0) for location in locations],
        [0.4, 0.4] + [0.025] * 6,
    )
    estimator = MixtureEstimator([GaussianEstimator() for _ in locations])
    return model, estimator, seq_encode(data, model=model)


def _multivariate_problem(seed: int, nobs: int, dimension: int) -> tuple[Any, Any, Any]:
    covariance = np.eye(dimension)
    truth = MixtureDistribution(
        [
            MultivariateGaussianDistribution(np.full(dimension, -4.0), 0.7 * covariance),
            MultivariateGaussianDistribution(np.full(dimension, 4.0), 0.7 * covariance),
        ],
        [0.5, 0.5],
    )
    data = truth.sampler(seed=seed).sample(size=nobs)
    locations = (-0.3, 0.3, -12.0, 12.0, -25.0, 25.0, -40.0, 40.0)
    components = [
        MultivariateGaussianDistribution(np.full(dimension, location), 3.0 * covariance) for location in locations
    ]
    model = MixtureDistribution(components, [0.4, 0.4] + [0.025] * 6)
    estimator = MixtureEstimator([MultivariateGaussianEstimator(dim=dimension) for _ in components])
    return model, estimator, seq_encode(data, model=model)


def _fit_with_reuse(factory: ProblemFactory, rounds: int, reuse: bool) -> tuple[float, float]:
    model, estimator, encoded = factory()
    started = time.perf_counter()
    fitted = optimize(
        None,
        estimator,
        enc_data=encoded,
        prev_estimate=model,
        max_its=rounds,
        delta=None,
        out=None,
        reuse_estep_ll=reuse,
        structure="off",
        schedule="full",
    )
    elapsed = time.perf_counter() - started
    return elapsed, observed_log_likelihood(encoded)(fitted)


def _reuse_case(name: str, factory: ProblemFactory, rounds: int, reps: int) -> dict[str, Any]:
    _fit_with_reuse(factory, 2, False)
    _fit_with_reuse(factory, 2, True)
    rows = []
    for repeat in range(reps):
        if repeat % 2:
            reuse_seconds, reuse_objective = _fit_with_reuse(factory, rounds, True)
            baseline_seconds, baseline_objective = _fit_with_reuse(factory, rounds, False)
        else:
            baseline_seconds, baseline_objective = _fit_with_reuse(factory, rounds, False)
            reuse_seconds, reuse_objective = _fit_with_reuse(factory, rounds, True)
        rows.append(
            {
                "baseline_seconds": baseline_seconds,
                "reuse_seconds": reuse_seconds,
                "speedup": baseline_seconds / reuse_seconds,
                "objective_delta": reuse_objective - baseline_objective,
            }
        )
    return {
        "case": name,
        "rounds": rounds,
        "repetitions": reps,
        "runs": rows,
        "median_speedup": _median(rows, "speedup"),
        "maximum_absolute_objective_delta": max(abs(float(row["objective_delta"])) for row in rows),
    }


def benchmark_estep_reuse(*, quick: bool, reps: int) -> dict[str, Any]:
    scalar_n = 4_000 if quick else 40_000
    mvn_n = 1_000 if quick else 4_000
    scalar_rounds = 12 if quick else 30
    mvn_rounds = 10 if quick else 20
    return {
        "hypothesis": "reusing an exact E-step likelihood reduces wall time without changing the result",
        "comparison": "reuse_estep_ll=True / reuse_estep_ll=False",
        "cases": [
            _reuse_case(
                "scalar-gaussian-mixture",
                lambda: _scalar_problem(17, scalar_n),
                scalar_rounds,
                reps,
            ),
            _reuse_case(
                "full-covariance-gaussian-mixture",
                lambda: _multivariate_problem(17, mvn_n, 16),
                mvn_rounds,
                reps,
            ),
        ],
    }


def _full_tree_trace(factory: ProblemFactory, rounds: int) -> dict[str, Any]:
    model, estimator, encoded = factory()
    objective = observed_log_likelihood(encoded)
    start_objective = objective(model)
    strategy = PosteriorTransformEM()
    values = []
    cumulative_seconds = []
    elapsed = 0.0
    for _ in range(rounds):
        started = time.perf_counter()
        model = strategy.step(encoded, estimator, model, objective=objective).model
        values.append(objective(model))
        elapsed += time.perf_counter() - started
        cumulative_seconds.append(elapsed)
    return {
        "start_objective": start_objective,
        "values": values,
        "cumulative_seconds": cumulative_seconds,
        "components": model.num_components,
    }


def _block_trace(factory: ProblemFactory, rounds: int) -> dict[str, Any]:
    model, estimator, encoded = factory()
    start_objective = observed_log_likelihood(encoded)(model)
    _, history = run_block_em(
        encoded,
        estimator,
        model,
        max_its=rounds,
        delta=None,
        budget_fraction=0.5,
    )
    return {
        "start_objective": start_objective,
        "values": [row.objective for row in history],
        "cumulative_seconds": list(np.cumsum([row.wall_time_seconds for row in history])),
        "cumulative_evaluations": list(np.cumsum([row.n_log_density_evals for row in history])),
        "active_fraction": [row.active_fraction for row in history],
    }


def _first_target(values: list[float], target: float) -> int:
    return next(index for index, value in enumerate(values) if value >= target)


def _block_case(
    name: str,
    factory_for_seed: Callable[[int], tuple[Any, Any, Any]],
    rounds: int,
    reps: int,
    target_fraction: float = 0.99,
) -> dict[str, Any]:
    rows = []
    for repeat, seed in enumerate(_SEEDS[:reps]):
        factory = lambda seed=seed: factory_for_seed(seed)
        if repeat % 2:
            block = _block_trace(factory, rounds)
            full = _full_tree_trace(factory, rounds)
        else:
            full = _full_tree_trace(factory, rounds)
            block = _block_trace(factory, rounds)
        start = min(float(full["start_objective"]), float(block["start_objective"]))
        shared_terminal = min(float(full["values"][-1]), float(block["values"][-1]))
        if shared_terminal <= start:
            raise RuntimeError("benchmark fixture did not improve its objective.")
        target = start + target_fraction * (shared_terminal - start)
        full_index = _first_target(full["values"], target)
        block_index = _first_target(block["values"], target)
        full_evaluations = 2 * int(full["components"]) * (full_index + 1)
        block_evaluations = int(block["cumulative_evaluations"][block_index])
        full_seconds = float(full["cumulative_seconds"][full_index])
        block_seconds = float(block["cumulative_seconds"][block_index])
        rows.append(
            {
                "seed": seed,
                "target_objective": target,
                "full_seconds_to_target": full_seconds,
                "block_seconds_to_target": block_seconds,
                "wall_time_speedup": full_seconds / block_seconds,
                "full_evaluations_to_target": full_evaluations,
                "block_evaluations_to_target": block_evaluations,
                "work_speedup": full_evaluations / block_evaluations,
                "full_rounds_to_target": full_index + 1,
                "block_rounds_to_target": block_index + 1,
                "mean_block_active_fraction": float(statistics.mean(block["active_fraction"][: block_index + 1])),
            }
        )
    return {
        "case": name,
        "round_budget": rounds,
        "shared_improvement_fraction": target_fraction,
        "runs": rows,
        "median_wall_time_speedup": _median(rows, "wall_time_speedup"),
        "median_work_speedup": _median(rows, "work_speedup"),
    }


def benchmark_block_scheduling(*, quick: bool, reps: int) -> dict[str, Any]:
    scalar_n = 4_000 if quick else 40_000
    mvn_n = 1_000 if quick else 16_000
    scalar_rounds = 80 if quick else 150
    mvn_rounds = 50 if quick else 100
    return {
        "hypothesis": "selective block updates reduce time-to-target after scheduler and gate overhead",
        "comparison": "budget_fraction=0.5 block EM / full-tree EM",
        "cases": [
            _block_case(
                "scalar-gaussian-mixture",
                lambda seed: _scalar_problem(seed, scalar_n),
                scalar_rounds,
                reps,
            ),
            _block_case(
                "full-covariance-gaussian-mixture",
                lambda seed: _multivariate_problem(seed, mvn_n, 16),
                mvn_rounds,
                reps,
            ),
        ],
    }


def benchmark_geometry_routing(*, quick: bool, reps: int) -> dict[str, Any]:
    batches = ((8, 1), (16, 2), (64, 1)) if quick else ((8, 1), (8, 4), (16, 2), (32, 2), (64, 1))
    cases = []
    for microbatch, accumulation in batches:
        rows = []
        for seed in _SEEDS[:reps]:
            receipt = run_graph_memory_pilot(
                seed=seed,
                source_nodes=128,
                train_examples=128,
                test_examples=64,
                updates=60,
                microbatch_size=microbatch,
                accumulation_steps=accumulation,
                target_accuracy=0.9,
            )
            adamw = receipt.graph_adamw
            routed = receipt.graph_routed
            both_reached = adamw.time_to_target_seconds is not None and routed.time_to_target_seconds is not None
            rows.append(
                {
                    "seed": seed,
                    "adamw_target_achieved": adamw.time_to_target_seconds is not None,
                    "routed_target_achieved": routed.time_to_target_seconds is not None,
                    "adamw_seconds_to_target": adamw.time_to_target_seconds,
                    "routed_seconds_to_target": routed.time_to_target_seconds,
                    "wall_time_speedup": (
                        adamw.time_to_target_seconds / routed.time_to_target_seconds if both_reached else None
                    ),
                    "adamw_updates_to_target": adamw.time_to_target_updates,
                    "routed_updates_to_target": routed.time_to_target_updates,
                    "update_speedup": (
                        adamw.time_to_target_updates / routed.time_to_target_updates if both_reached else None
                    ),
                    "adamw_final_accuracy": adamw.test_accuracy,
                    "routed_final_accuracy": routed.test_accuracy,
                }
            )
        cases.append(
            {
                "microbatch_size": microbatch,
                "accumulation_steps": accumulation,
                "effective_batch_examples": microbatch * accumulation,
                "runs": rows,
                "adamw_target_achievement_rate": statistics.mean(row["adamw_target_achieved"] for row in rows),
                "routed_target_achievement_rate": statistics.mean(row["routed_target_achieved"] for row in rows),
                "joint_target_runs": sum(
                    row["adamw_target_achieved"] and row["routed_target_achieved"] for row in rows
                ),
                "median_wall_time_speedup": _median_available(rows, "wall_time_speedup"),
                "median_update_speedup": _median_available(rows, "update_speedup"),
            }
        )
    return {
        "hypothesis": "typed Muon/AdamW routing beats AdamW time-to-target on a heterogeneous MoE",
        "comparison": "routed optimizer / AdamW at identical stochastic batch semantics",
        "timing_scale": "synthetic microfixture",
        "wall_time_claim_eligible": False,
        "claim_exclusion_reason": "target crossings are too short for a production optimizer speed claim",
        "cases": cases,
    }


def run_benchmarks(*, quick: bool, reps: int, panels: tuple[str, ...]) -> dict[str, Any]:
    if reps < 1 or reps > len(_SEEDS):
        raise ValueError("reps must be between 1 and %d." % len(_SEEDS))
    unknown = set(panels) - {"reuse", "block", "geometry"}
    if unknown:
        raise ValueError("unknown benchmark panels: %s" % sorted(unknown))
    selected: dict[str, Any] = {}
    if "reuse" in panels:
        selected["estep_reuse"] = benchmark_estep_reuse(quick=quick, reps=reps)
    if "block" in panels:
        selected["block_scheduling"] = benchmark_block_scheduling(quick=quick, reps=reps)
    if "geometry" in panels:
        selected["geometry_routing"] = benchmark_geometry_routing(quick=quick, reps=reps)
    return stamp_result(
        {
            "benchmark": "typed_optimization",
            "schema_version": 1,
            "quick": quick,
            "repetitions": reps,
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "packages": _versions(),
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                },
            },
            "panels": selected,
            "interpretation": {
                "speedup_above_one": "candidate is faster or uses fewer updates/work than baseline",
                "wall_time_is_claim_metric": True,
                "work_speedup_alone_is_not_acceleration": True,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run smaller local fixtures")
    parser.add_argument("--reps", type=int, default=3, help="independent repeats/seeds")
    parser.add_argument("--panels", default="reuse,block,geometry", help="comma-separated panel names")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "typed_optimization.json",
    )
    args = parser.parse_args()
    panels = tuple(name.strip() for name in args.panels.split(",") if name.strip())
    result = run_benchmarks(quick=args.quick, reps=args.reps, panels=panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
