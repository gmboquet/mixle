"""Spark transport for distributed heterogeneous EM: ``RDD.treeReduce`` over the verified combine-tree.

The same sharded-E-step + k-way tree-reduce algorithm as :mod:`mixle.inference.heterogeneous_executor`,
run on a Spark cluster: shards become an RDD, each is scored to a fixed-size ``(count, sufficient-stat)``
payload by ``map``, and those fold with ``RDD.treeReduce`` -- the reduction happens IN Spark across
``O(log W)`` levels, never a single-root ``collect`` to the driver (the OOM fan-in the scaling audit
flagged). ``treeReduce``'s combiner runs on freshly-deserialized payloads, so the in-place ``combine()``
is safe (the HMM-stat aliasing hazard does not bite).
"""

from __future__ import annotations

from typing import Any

from mixle.inference.heterogeneous_executor import _shard_bounds, _shard_estep


def _make_shards(data: Any, n_shards: int) -> list[Any]:
    return [data[lo:hi] for lo, hi in _shard_bounds(len(data), None, n_shards) if hi > lo]


def spark_em_step(sc: Any, estimator: Any, model: Any, data: Any, n_shards: int = 8, depth: int = 2) -> Any:
    """One EM step on Spark: parallelize shards, map the E-step, ``treeReduce`` the combine, estimate."""
    shards = _make_shards(data, n_shards)
    factory = estimator.accumulator_factory()

    def estep(shard: Any) -> tuple[int, Any]:
        return _shard_estep(estimator, model, shard)

    def combine(a: tuple[int, Any], b: tuple[int, Any]) -> tuple[int, Any]:
        acc = factory.make().from_value(a[1])
        acc.combine(b[1])
        return a[0] + b[0], acc.value()

    rdd = sc.parallelize(shards, len(shards))
    count, value = rdd.map(estep).treeReduce(combine, depth=depth)
    return estimator.estimate(float(count), value)


def spark_fit(sc: Any, model: Any, data: Any, max_its: int = 10, n_shards: int = 8, depth: int = 2) -> Any:
    """Run ``max_its`` EM iterations on Spark; the shard RDD is cached once and re-scored each iteration."""
    shards = _make_shards(data, n_shards)
    rdd = sc.parallelize(shards, len(shards)).cache()
    estimator = model.estimator()
    factory = estimator.accumulator_factory()
    current = model
    try:
        for _ in range(max_its):
            model_i = current  # capture the current estimate for this iteration's closure

            def estep(shard: Any, _m: Any = model_i) -> tuple[int, Any]:
                return _shard_estep(estimator, _m, shard)

            def combine(a: tuple[int, Any], b: tuple[int, Any]) -> tuple[int, Any]:
                acc = factory.make().from_value(a[1])
                acc.combine(b[1])
                return a[0] + b[0], acc.value()

            count, value = rdd.map(estep).treeReduce(combine, depth=depth)
            current = estimator.estimate(float(count), value)
    finally:
        rdd.unpersist()
    return current
