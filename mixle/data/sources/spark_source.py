"""Spark-backed sampling helpers for sequence-encodable Mixle distributions.

The functions in this module broadcast distribution objects to Spark workers,
draw reproducible partition-level samples, and return RDDs that can feed larger
distributed estimation workflows.
"""

try:
    from pyspark import SparkConf, SparkContext
except ImportError:
    SparkContext = SparkConf = None  # pip install mixle[spark]
import pickle
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint


def _exact_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _seed(seed: Any) -> int | None:
    if seed is None:
        return None
    value = _exact_int("seed", seed, minimum=0)
    if value >= 2**32:
        raise ValueError(f"seed must be less than 2**32, got {seed!r}")
    return value


def take_sample(rdd: Any, with_replacement: bool, n: int, seed: int | None = None):
    """Take a deterministic-order random sample from a Spark RDD.

    Raises ``ValueError`` if ``with_replacement`` is ``False`` and ``rdd`` has fewer than ``n``
    elements: Spark's own ``RDD.takeSample(False, n, ...)`` silently caps the result at the RDD's true
    cardinality in that case (returning every element instead of the requested count) rather than
    raising, so this checks the returned length itself and fails loudly with the shortfall instead of
    silently handing the caller fewer records than they asked for (MXR-080-0066). ``with_replacement``
    sampling always returns exactly ``n`` elements and is unaffected.
    """
    if not isinstance(with_replacement, (bool, np.bool_)):
        raise ValueError(f"with_replacement must be Boolean, got {with_replacement!r}")
    n = _exact_int("n", n, minimum=0)
    seed = _seed(seed)
    rng = RandomState(seed)
    sample = rdd.zipWithUniqueId().takeSample(with_replacement, n, rng.randint(0, maxrandint))
    sidx = np.argsort([u[1] for u in sample])
    sample = [sample[i][0] for i in sidx]
    if not with_replacement and len(sample) < n:
        raise ValueError("requested %d without replacement, only %d available" % (n, len(sample)))
    # Shuffle by the ACTUAL returned cardinality, not the requested `n` -- indexing a length-`n`
    # permutation into a shorter list is exactly the IndexError MXR-080-0066 reported.
    sidx = np.argsort(rng.uniform(size=len(sample)))
    return [sample[i] for i in sidx]


def sample_seq_as_rdd(sc, dist, seq_len, count_per_split, num_splits, seed=None):
    """Sample fixed-length sequences from a distribution into a Spark RDD."""
    seq_len = _exact_int("seq_len", seq_len, minimum=1)
    count_per_split = _exact_int("count_per_split", count_per_split, minimum=1)
    num_splits = _exact_int("num_splits", num_splits, minimum=1)
    seed = _seed(seed)
    distB = sc.broadcast(dist)
    seeds = RandomState(seed).randint(0, maxrandint, size=num_splits)

    def fmap(u):
        ddist = distB.value
        sampler = [ddist.sampler(seed=h) for h in u]
        return iter([v for h in sampler for v in h.sample_seq(seq_len, size=count_per_split)])

    return sc.parallelize(seeds, num_splits).mapPartitions(fmap, True)


def sample_rdd(sc, dist, count_per_split, num_splits, seed=None):
    """Sample independent draws from a distribution into a Spark RDD."""
    count_per_split = _exact_int("count_per_split", count_per_split, minimum=1)
    num_splits = _exact_int("num_splits", num_splits, minimum=1)
    seed = _seed(seed)
    dd = pickle.dumps(dist, protocol=0)
    distB = sc.broadcast(dd)
    seeds = RandomState(seed).randint(0, maxrandint, size=num_splits)

    def fmap(u):
        ddist = pickle.loads(distB.value)
        sampler = [ddist.sampler(seed=h) for h in u]
        return iter([v for h in sampler for v in h.sample(size=count_per_split)])

    return sc.parallelize(seeds, num_splits).mapPartitions(fmap, True)
