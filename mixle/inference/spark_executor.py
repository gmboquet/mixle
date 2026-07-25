"""Spark transport for distributed heterogeneous EM: ``RDD.treeReduce`` over the verified combine-tree.

The same sharded-E-step + k-way tree-reduce algorithm as :mod:`mixle.inference.heterogeneous_executor`,
run on a Spark cluster: shards become an RDD, each is scored to a fixed-size ``(count, sufficient-stat)``
payload by ``map``, and those fold with ``RDD.treeReduce`` -- the reduction happens IN Spark across
``O(log W)`` levels, never a single-root ``collect`` to the driver (the OOM fan-in the scaling audit
flagged). ``treeReduce``'s combiner runs on freshly-deserialized payloads, so the in-place ``combine()``
is safe (the HMM-stat aliasing hazard does not bite).
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

from mixle.inference.heterogeneous_executor import _shard_bounds, _shard_estep


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _replayable_data(data: Any) -> Any:
    """Return one non-empty, sized, sliceable dataset without consuming it more than once."""
    if hasattr(data, "__len__") and hasattr(data, "__getitem__"):
        replayable = data
    else:
        replayable = tuple(data)
    if len(replayable) == 0:
        raise ValueError("Spark EM requires at least one observation")
    return replayable


def _make_shards(data: Any, n_shards: int) -> tuple[Any, list[Any]]:
    replayable = _replayable_data(data)
    n_shards = _positive_integer(n_shards, "n_shards")
    if n_shards > len(replayable):
        raise ValueError(
            f"n_shards ({n_shards}) must not exceed the observation count ({len(replayable)}); "
            "empty Spark partitions are not valid EM work"
        )
    shards = [replayable[lo:hi] for lo, hi in _shard_bounds(len(replayable), None, n_shards)]
    if any(len(shard) == 0 for shard in shards) or sum(len(shard) for shard in shards) != len(replayable):
        raise RuntimeError("Spark shard construction did not preserve every observation exactly once")
    return replayable, shards


def _checked_estep(estimator: Any, model: Any, shard: Any) -> tuple[int, Any]:
    count, value = _shard_estep(estimator, model, shard)
    if count != len(shard) or count <= 0:
        raise RuntimeError(
            f"Spark E-step reported {count} processed observations for a non-empty shard of {len(shard)}"
        )
    return count, value


def spark_em_step(sc: Any, estimator: Any, model: Any, data: Any, n_shards: int = 8, depth: int = 2) -> Any:
    """One EM step on Spark: parallelize shards, map the E-step, ``treeReduce`` the combine, estimate."""
    depth = _positive_integer(depth, "depth")
    replayable, shards = _make_shards(data, n_shards)
    if not callable(getattr(sc, "parallelize", None)):
        raise TypeError("sc must provide a callable parallelize method")
    factory = estimator.accumulator_factory()

    def estep(shard: Any) -> tuple[int, Any]:
        return _checked_estep(estimator, model, shard)

    def combine(a: tuple[int, Any], b: tuple[int, Any]) -> tuple[int, Any]:
        acc = factory.make().from_value(a[1])
        acc.combine(b[1])
        return a[0] + b[0], acc.value()

    rdd = sc.parallelize(shards, len(shards))
    count, value = rdd.map(estep).treeReduce(combine, depth=depth)
    if count != len(replayable):
        raise RuntimeError(
            f"Spark reduction processed {count} observations, expected exactly {len(replayable)}"
        )
    return estimator.estimate(float(count), value)


def spark_fit(sc: Any, model: Any, data: Any, max_its: int = 10, n_shards: int = 8, depth: int = 2) -> Any:
    """Run ``max_its`` EM iterations on Spark; the shard RDD is cached once and re-scored each iteration."""
    max_its = _positive_integer(max_its, "max_its")
    depth = _positive_integer(depth, "depth")
    replayable, shards = _make_shards(data, n_shards)
    if not callable(getattr(sc, "parallelize", None)):
        raise TypeError("sc must provide a callable parallelize method")
    rdd = sc.parallelize(shards, len(shards)).cache()
    estimator = model.estimator()
    factory = estimator.accumulator_factory()
    current = model
    try:
        for _ in range(max_its):
            model_i = current  # capture the current estimate for this iteration's closure

            def estep(shard: Any, _m: Any = model_i) -> tuple[int, Any]:
                return _checked_estep(estimator, _m, shard)

            def combine(a: tuple[int, Any], b: tuple[int, Any]) -> tuple[int, Any]:
                acc = factory.make().from_value(a[1])
                acc.combine(b[1])
                return a[0] + b[0], acc.value()

            count, value = rdd.map(estep).treeReduce(combine, depth=depth)
            if count != len(replayable):
                raise RuntimeError(
                    f"Spark reduction processed {count} observations, expected exactly {len(replayable)}"
                )
            current = estimator.estimate(float(count), value)
    finally:
        rdd.unpersist()
    return current
