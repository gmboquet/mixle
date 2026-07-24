"""Regression tests for the undersized without-replacement Spark sample crash (MXR-080-0066).

``take_sample`` draws ``rdd.zipWithUniqueId().takeSample(with_replacement, n, seed)``, then reorders
the result deterministically by a second ``rng.uniform`` shuffle. Spark's own ``RDD.takeSample`` with
``with_replacement=False`` silently caps its result at the RDD's true cardinality when ``n`` exceeds
it (returning every element instead of the requested count) -- it does not raise. The final shuffle
nevertheless generated a permutation of length ``n`` (the REQUESTED count, always) and indexed the
(possibly shorter) returned list with it, raising ``IndexError: list index out of range`` whenever
Spark's result came back short. The audit's own repro was a two-item mock ``takeSample`` result for
``n=5``.

The fix shuffles by the ACTUAL returned cardinality (``len(sample)``) rather than the requested ``n``,
and makes the undersized-request contract explicit: without replacement, a shortfall now raises
``ValueError`` naming both numbers, instead of leaking an opaque ``IndexError`` (or silently handing
the caller fewer records than they asked for). This mirrors the "raise rather than silently truncate"
contract already established elsewhere in this codebase for the same kind of request-exceeds-population
condition (e.g. ``mixle.evolve.population._auto_split`` / ``mixle.evolve.concept_discovery``).

No real ``pyspark``/JVM is needed: ``take_sample`` only ever calls ``.zipWithUniqueId().takeSample(...)``
on its ``rdd`` argument, so a minimal fake standing in for exactly that surface is enough to drive it
deterministically and without the JVM startup cost the real Spark-backed tests pay
(``spark_executor_test.py``, ``spark_encoded_data_test.py``, both marked ``spark``/``optional``).
"""

import pytest

from mixle.data.sources.spark_source import take_sample


class _FakeZippedRDD:
    """Stands in for ``rdd.zipWithUniqueId()``: ``take_sample`` only ever calls ``.takeSample(...)`` on it."""

    def __init__(self, result):
        self._result = result

    def takeSample(self, with_replacement, n, seed):
        return self._result


class _FakeRDD:
    """Stands in for a Spark RDD: ``take_sample`` only ever calls ``.zipWithUniqueId()`` on it."""

    def __init__(self, zipped_result):
        self._zipped_result = zipped_result

    def zipWithUniqueId(self):
        return _FakeZippedRDD(self._zipped_result)


# --------------------------------------------------------------------------- the crash itself


def test_undersized_without_replacement_raises_value_error_not_indexerror():
    """The audit's own repro: a two-item mock ``takeSample`` result for a without-replacement n=5
    request. Previously raised ``IndexError: list index out of range``."""
    rdd = _FakeRDD([("a", 0), ("b", 1)])

    with pytest.raises(ValueError, match=r"requested 5.*only 2 available"):
        take_sample(rdd, False, 5, seed=0)


def test_undersized_error_names_the_actual_shortfall():
    rdd = _FakeRDD([("only-one", 0)])

    with pytest.raises(ValueError, match=r"requested 3.*only 1 available"):
        take_sample(rdd, False, 3, seed=0)


def test_zero_items_returned_without_replacement_raises_cleanly():
    """Edge of the edge case: an empty ``takeSample`` result must raise the same targeted error,
    not (say) a different crash from indexing/shuffling an empty list against a nonzero n."""
    rdd = _FakeRDD([])

    with pytest.raises(ValueError, match=r"requested 4.*only 0 available"):
        take_sample(rdd, False, 4, seed=0)


# --------------------------------------------------------------------------- negative controls (unaffected paths)


def test_normal_sized_without_replacement_sample_unchanged():
    """Population == n (the common case) must still work exactly as before: all n items returned,
    deterministically shuffled by the seed, none dropped or duplicated."""
    rdd = _FakeRDD([(chr(ord("a") + i), i) for i in range(5)])

    out = take_sample(rdd, False, 5, seed=0)

    assert len(out) == 5
    assert sorted(out) == ["a", "b", "c", "d", "e"]


def test_with_replacement_sample_of_requested_size_is_unaffected():
    """With replacement, Spark's takeSample always returns exactly n elements (never short), so the
    new undersized check must never trigger on this path."""
    rdd = _FakeRDD([("a", 0), ("a", 1), ("b", 2), ("a", 3), ("b", 4)])

    out = take_sample(rdd, True, 5, seed=0)

    assert len(out) == 5
    assert set(out) <= {"a", "b"}


def test_same_seed_is_deterministic():
    """The shuffle-by-actual-cardinality fix must preserve the function's documented determinism."""
    items = [(chr(ord("a") + i), i) for i in range(5)]

    out1 = take_sample(_FakeRDD(items), False, 5, seed=42)
    out2 = take_sample(_FakeRDD(items), False, 5, seed=42)

    assert out1 == out2
