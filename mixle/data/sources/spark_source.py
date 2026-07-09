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


def take_sample(rdd: Any, with_replacement: bool, n: int, seed: int | None = None):
    """Take a deterministic-order random sample from a Spark RDD."""
    rng = RandomState(seed)
    sample = rdd.zipWithUniqueId().takeSample(with_replacement, n, rng.randint(0, maxrandint))
    sidx = np.argsort([u[1] for u in sample])
    sample = [sample[i][0] for i in sidx]
    sidx = np.argsort(rng.uniform(size=n))
    return [sample[i] for i in sidx]


def sample_seq_as_rdd(sc, dist, seq_len, count_per_split, num_splits, seed=None):
    """Sample fixed-length sequences from a distribution into a Spark RDD."""
    distB = sc.broadcast(dist)
    seeds = RandomState(seed).randint(0, maxrandint, size=num_splits)

    def fmap(u):
        ddist = distB.value
        sampler = [ddist.sampler(seed=h) for h in u]
        return iter([v for h in sampler for v in h.sample_seq(seq_len, size=count_per_split)])

    return sc.parallelize(seeds, num_splits).mapPartitions(fmap, True)


def sample_rdd(sc, dist, count_per_split, num_splits, seed=None):
    """Sample independent draws from a distribution into a Spark RDD."""
    dd = pickle.dumps(dist, protocol=0)
    distB = sc.broadcast(dd)
    seeds = RandomState(seed).randint(0, maxrandint, size=num_splits)

    def fmap(u):
        ddist = pickle.loads(distB.value)
        sampler = [ddist.sampler(seed=h) for h in u]
        return iter([v for h in sampler for v in h.sample(size=count_per_split)])

    return sc.parallelize(seeds, num_splits).mapPartitions(fmap, True)
