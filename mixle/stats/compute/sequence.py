"""Vectorized sequence-driver primitives over the pdist protocol.

The module-level ``seq_*`` API — encode a dataset, score it, run a vectorized EM E-step, initialize an
accumulator — dispatching over the (encoder, estimator, distribution) contracts in
:mod:`mixle.stats.compute.pdist`. These were defined inline in ``mixle.stats.__init__``; they live here
so the fitting machinery (``mixle.inference``) can import them WITHOUT importing the whole
``mixle.stats`` package — which previously forced ``mixle.inference`` to resolve lazily to dodge a
half-initialized ``mixle.stats``. ``mixle.stats`` re-exports them, so the public
``mixle.stats.seq_estimate`` / ``seq_log_density_sum`` / … API is unchanged.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    validate_estimator_keys,
)
from mixle.utils.optional_deps import RDD_TYPES, pyspark

T = TypeVar("T")
T_D = TypeVar("T_D", bound=SequenceEncodableProbabilityDistribution)


def seq_encode(
    data: Sequence[T] | pyspark.rdd.RDD,
    encoder: DataSequenceEncoder | None = None,
    estimator: ParameterEstimator | None = None,
    model: SequenceEncodableProbabilityDistribution | None = None,
    num_chunks: int = 1,
    chunk_size: int | None = None,
) -> pyspark.rdd.RDD | list[tuple[int, Any]]:
    """Sequence encode a sequence of iid observations from a distribution corresponding to 'encoder'.

    Takes data of type Union[Sequence[T], pyspark.rdd.RDD], where the data type of the DataSequenceEncoder object's
    corresponding distribution is type T.

    If not RDD, returns a List[Tuple[int, T1]], with each list entry being a tuple containing the number of observations
    in the sequence (chunk_size), and an encoded sequence of the observations having type T1. The list has length
    num_chunks.

    RDD version with receive the Tuple of chunk_size and encoded data of type T1 for each corresponding node.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Sequence of iid observations of data type consistent with
            'encoder'.
        encoder (Optional[DataSequenceEncoder]): A DataSequenceEncoder object for sequence encoding iid sequences.
        estimator (Optional[ParameterEstimator]): An estimator to create DataSequenceEncoder from.
        model (Optional[SequenceEncodableProbabilityDistribution]): A distribution to create DataSequenceEncoder from.
        num_chunks (int): Number of chunks to split the data into. Useful for distributed data sets.
        chunk_size (Optional[int]): Approximate size of chunks to determine num_chunks above.

    Returns:
        Encoded data ready for vectorized ``seq_*`` methods.

    """
    # tolerate a model or estimator passed positionally in the encoder slot
    if isinstance(encoder, SequenceEncodableProbabilityDistribution):
        model, encoder = encoder, None
    elif isinstance(encoder, ParameterEstimator):
        estimator, encoder = encoder, None

    if encoder is None:
        if model is not None:
            encoder = model.dist_to_encoder()
        elif estimator is not None:
            encoder = estimator.accumulator_factory().make().acc_to_encoder()
        else:
            raise ValueError("At least one arg: encoder, estimator, or dist must be passed.")

    # DataSource branch (additive) -- a structured/typed source routes through its structure-aware
    # encoder and returns the same [(count, payload)] shape; the bare-list and RDD paths are untouched.
    # Imported lazily so stats does not depend on mixle.data at module load (data depends on stats).
    from mixle.data.core import DataSource

    if isinstance(data, DataSource):
        return data.encode(encoder, num_chunks=num_chunks, chunk_size=chunk_size)

    if isinstance(data, RDD_TYPES):
        sc = data.context
        temp_encoder = pickle.dumps(encoder, protocol=0)
        encoder_broadcast = sc.broadcast(temp_encoder)

        enc_data = (
            data.glom()
            .map(lambda x: list(x))
            .map(lambda x: (len(x), pickle.loads(encoder_broadcast.value).seq_encode(x)))
        )

        return enc_data

    else:
        sz = len(data)
        if chunk_size is not None:
            num_chunks_loc = int(np.ceil(float(sz) / float(chunk_size)))
        else:
            num_chunks_loc = num_chunks

        rv = []
        for i in range(num_chunks_loc):
            data_loc = [data[i] for i in range(i, sz, num_chunks_loc)]
            enc_data = encoder.seq_encode(data_loc)
            rv.append((len(data_loc), enc_data))

        return rv


def seq_log_density_sum(
    enc_data: list[tuple[int, T]] | pyspark.rdd.RDD, estimate: SequenceEncodableProbabilityDistribution
) -> tuple[float, float]:
    """Vectorized evaluation of total count and total log-density over encoded data.

    The returned pair contains the observation count represented by ``enc_data`` and the sum of
    ``log_density`` over those encoded observations.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimate (SequenceEncodableProbabilityDistribution): Distribution to use for log_density evaluations. Must
            be consistent with enc_data.

    Returns:
        Tuple of sum of total obs, and sum of log_density of estimate at all encoded data observations.

    """
    if hasattr(enc_data, "pysp_seq_log_density_sum"):
        # parallel-backend handle (mixle.utils.parallel.multiprocessing / parallel_mpi)
        return enc_data.pysp_seq_log_density_sum(estimate)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context
        estimate_broadcast = sc.broadcast(pickle.dumps(estimate, protocol=0))

        def acc(itr):

            rv = 0.0
            cnt = 0.0
            estimate_loc = pickle.loads(estimate_broadcast.value)
            for sz, x in itr:
                rv += estimate_loc.seq_log_density(x).sum()
                cnt += sz

            return [(cnt, rv)]

        return enc_data.mapPartitions(acc).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))

    else:
        return sum([u[0] for u in enc_data]), sum([estimate.seq_log_density(u[1]).sum() for u in enc_data])


def seq_log_density(
    enc_data: list[tuple[int, T]] | pyspark.rdd.RDD,
    estimate: Sequence[SequenceEncodableProbabilityDistribution] | SequenceEncodableProbabilityDistribution,
) -> list[np.ndarray]:
    """Vectorized evaluation of 'estimate' log-density for each observation in enc_data.

    If 'estimate' is input as a List of numpy arrays. Each list entry corresponds to the seq_log_density calls of all
    the encoded data for each List entry of estimate.

    If 'estimate' is a single SequenceEncodableProbabilityDistribution instance. The log_density of every observation
    in the 'enc_data' data set is returned as a list.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimate (SequenceEncodableProbabilityDistribution): Distribution to use for log_density evaluations. Must
            be consistent with enc_data.

    Returns:
        List[np.ndarray[float]] or List[float] depending on input.

    """
    is_list = issubclass(type(estimate), Sequence)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context
        temp_estimate = pickle.dumps(estimate, protocol=0)
        estimate_broadcast = sc.broadcast(temp_estimate)

        def acc(itr):
            loc_estimate = pickle.loads(estimate_broadcast.value)
            if is_list:
                return [np.asarray([ee.seq_log_density(x) for ee in loc_estimate]) for sz, x in itr]
            else:
                return [loc_estimate.seq_log_density(x) for sz, x in itr]

        return enc_data.mapPartitions(acc).collect()

    else:
        if is_list:
            return [np.asarray([ee.seq_log_density(u[1]) for ee in estimate]) for u in enc_data]
        else:
            return [estimate.seq_log_density(u[1]) for u in enc_data]


def log_density(
    data: Sequence[T] | pyspark.rdd.RDD,
    model: SequenceEncodableProbabilityDistribution,
) -> np.ndarray:
    """Per-observation log-density of 'model' over raw (unencoded) 'data'.

    Convenience wrapper that encodes 'data' with the model's own encoder, evaluates the vectorized
    seq_log_density, and returns a single flat numpy array aligned to the input order -- the common need that
    otherwise requires the seq_encode / seq_log_density / np.concatenate boilerplate. For a distributed RDD the
    densities are collected to the driver in partition order.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Raw iid observations of data type consistent with 'model'.
        model (SequenceEncodableProbabilityDistribution): Distribution to score the observations under.

    Returns:
        np.ndarray of per-observation log-densities.

    """
    # num_chunks=1 keeps the result aligned to the input order (multi-chunk encoding interleaves observations)
    enc_data = seq_encode(data, model=model, num_chunks=1)
    parts = seq_log_density(enc_data, model)
    return np.concatenate([np.atleast_1d(np.asarray(p, dtype=float)) for p in parts])


def density(
    data: Sequence[T] | pyspark.rdd.RDD,
    model: SequenceEncodableProbabilityDistribution,
) -> np.ndarray:
    """Per-observation density of 'model' over raw (unencoded) 'data'.

    Exponentiated companion to log_density(); returns a flat numpy array of densities aligned to the input order.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Raw iid observations of data type consistent with 'model'.
        model (SequenceEncodableProbabilityDistribution): Distribution to score the observations under.

    Returns:
        np.ndarray of per-observation densities.

    """
    return np.exp(log_density(data, model))


def seq_estimate(
    enc_data: list[tuple[int, T]] | pyspark.rdd.RDD, estimator: ParameterEstimator, prev_estimate: T_D
) -> T_D:
    """Perform vectorized E-step in EM algorithm for encoded sequence of observations in 'enc_data'.

    Arg estimator must be consistent with prev_estimate. That is, prev_estimate must be an estimate that could be
    obtained from estimator.

    Arg enc_data must type consistent with estimator and prev_estimate (result of seq_encode() call).

    Returns the next iteration of EM algorithm with vectorized calls to "seq_update()" of the corresponding
    SequenceEncodableStatsiticAccumulator objects.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimator (ParameterEstimator): Model to be estimated from 'enc_data'.
        prev_estimate (SequenceEncodableProbabilityDistribution): Previous estimate of EM algorithm.

    Returns:
        SequenceEncodableProbabilityDistribution object.

    """
    validate_estimator_keys(estimator)

    if hasattr(enc_data, "pysp_seq_estimate"):
        # parallel-backend handle (mixle.utils.parallel.multiprocessing / parallel_mpi)
        return enc_data.pysp_seq_estimate(estimator, prev_estimate)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context

        estimator_broadcast = sc.broadcast(estimator)
        estimate_broadcast = sc.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            local_estimate = pickle.loads(estimate_broadcast.value)

            for sz, x in itr:
                counts_for_split = counts_for_split + sz
                accumulator_for_split.seq_update(x, np.ones(sz), local_estimate)

            rv = pickle.dumps((counts_for_split, accumulator_for_split.value()), protocol=0)

            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimator_broadcast.value.accumulator_factory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))

            return rv

        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = 0.0
        accumulator = estimator.accumulator_factory().make()

        for stuff in temp.collect():
            nobs_for_split, stats_for_split = pickle.loads(stuff)
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        estimate_broadcast.destroy()
        estimator_broadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return estimator.estimate(nobs, accumulator.value())

    else:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0

        for sz, x in enc_data:
            nobs += sz
            accumulator.seq_update(x, np.ones(sz), prev_estimate)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(None, accumulator.value())


def seq_initialize(
    enc_data: list[tuple[int, T]] | pyspark.rdd.RDD,
    estimator: ParameterEstimator,
    rng: np.random.RandomState,
    p: float = 0.1,
) -> SequenceEncodableProbabilityDistribution:
    """Vectorized initialization of a model corresponding to ParameterEstimator for encoded sequences of iid data
        observations.

    Arg enc_data must type consistent with estimator (result of seq_encode() call).
    Arg estimator must be of data type consistent with encoded sequence data type in 'enc_data'.

    Vectorized initialization of SequenceEncodableProbabilityDistribution corresponding to 'estimator' from enc_data.
    Observations in the encoded sequence enc_data are kept with probability p.

    This functions relies on calls to SequenceEncodableStatisticAccumulator.seq_initialize(), which is a vectorized
    initialization of the SequenceEncodableStatisticAccumulator object.

    This method should produce the same initialized model as a call to initialize() if the data sets are the same.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimator (ParameterEstimator): Model to be estimated from 'enc_data'.
        rng (RandomState): RandomState object for setting seed.
        p (float): Proportion of data to randomly sample for initializing model.

    Returns:
        SequenceEncodableProbabilityDistribution object consistent with 'estimator'.

    """
    validate_estimator_keys(estimator)

    if hasattr(enc_data, "pysp_seq_initialize"):
        # parallel-backend handle (mixle.utils.parallel.multiprocessing / parallel_mpi)
        return enc_data.pysp_seq_initialize(estimator, rng, p)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context
        num_partitions = enc_data.getNumPartitions()
        seeds = rng.randint(2**31, size=num_partitions)

        estimator_broadcast = sc.broadcast(estimator)
        seeds_broadcast = sc.broadcast(pickle.dumps(seeds, protocol=0))

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            rng_loc = np.random.RandomState(seeds_broadcast.value[split_index])
            rng_loc_w = np.random.RandomState(seed=rng_loc.randint(2**31))

            for sz, x in itr:
                w = np.zeros(sz, dtype=float)
                w_1 = rng_loc_w.rand(sz) <= p
                w[w_1] = 1.0

                counts_for_split += np.sum(w)
                accumulator_for_split.seq_initialize(x, w, rng_loc)

            rv = pickle.dumps((counts_for_split, accumulator_for_split.value()), protocol=0)
            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimator_broadcast.value.accumulator_factory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))

            return rv

        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = 0.0
        accumulator = estimator.accumulator_factory().make()

        for stuff in temp.collect():
            nobs_for_split, stats_for_split = pickle.loads(stuff)
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        seeds_broadcast.destroy()
        estimator_broadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return estimator.estimate(nobs, accumulator.value())

    else:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        rng_w = np.random.RandomState(seed=rng.randint(2**31 - 1))

        for sz, enc_x in enc_data:
            w = rng_w.binomial(n=1, p=p, size=sz).astype(dtype=np.float64)
            accumulator.seq_initialize(enc_x, w, rng)
            nobs += float(w.sum())  # count the kept (weight-1) observations, matching the RDD/non-seq paths

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())


def initialize(
    data: Sequence[T] | pyspark.rdd.RDD, estimator: ParameterEstimator, rng: np.random.RandomState, p: float = 0.1
) -> SequenceEncodableProbabilityDistribution:
    """Randomly initialize a model corresponding to ParameterEstimator for iid observations data.

    Note: ParameterEstimator must be of data type T, matching the input data.

    This function sequentially iterates over the entire data set 'data', repeatedly calling initialize() method
    of the SequenceEncodableStatisticAccumulator object created from 'estimator'. Data points are weighted 0 or 1 with
    probability p.

    Seq_initialize() is much more efficient, and should produce the same initialized model for the same data sets.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Set of iid observations compatible with 'estimator'.
        estimator (ParameterEstimator): ParameterEstimator object for desired model to be estimated from data.
        rng (RandomState): RandomState object for setting seed.
        p (float): Proportion of data to randomly sample for initializing model.

    Returns:
        SequenceEncodableProbabilityDistribution object consistent with 'estimator'.

    """
    validate_estimator_keys(estimator)

    if isinstance(data, RDD_TYPES):
        factory = estimator.accumulator_factory()
        sc = data.context

        num_partitions = data.getNumPartitions()
        seeds = rng.randint(2**31, size=num_partitions)

        estimator_broadcast = sc.broadcast(estimator)
        seeds_broadcast = sc.broadcast(seeds)

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            rng_loc = np.random.RandomState(seeds_broadcast.value[split_index])
            rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))

            for x in itr:
                w = rng_w.binomial(n=1, p=p)  # partition-local rng; the driver's rng is identical on every split
                counts_for_split += w
                accumulator_for_split.initialize(x, w, rng_loc)

            return iter([(counts_for_split, accumulator_for_split.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobs_for_split, stats_for_split in temp.collect():
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        rng_w = np.random.RandomState(seed=rng.randint(2**31))

        for i, x in enumerate(idata):
            w = rng_w.binomial(n=1, p=p)
            nobs += w
            accumulator.initialize(x, w, rng)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())


def estimate(
    data: Sequence[T] | pyspark.rdd.RDD,
    estimator: ParameterEstimator,
    prev_estimate: SequenceEncodableProbabilityDistribution | None = None,
) -> SequenceEncodableProbabilityDistribution:
    """Perform E-step in EM algorithm by iterating over all observations in 'data'.

    Arg estimator must be consistent with prev_estimate. That is, prev_estimate must be an estimate that could be
    obtained from estimator.

    Data must type consistent with estimator and prev_estimate.

    Returns the next iteration of EM algorithm by iterating over each observation of data. See seq_estimate() for
    a more computationally efficient implementation.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Sequence of iid observations of data type consistent with
            'estimator' and/or 'prev_estimate'.
        estimator (ParameterEstimator): Model to be estimated from 'data'.
        prev_estimate (Optional[SequenceEncodableProbabilityDistribution]): Previous estimate of EM algorithm. Must
            be included for distributions that require initialization.

    Returns:
        SequenceEncodableProbabilityDistribution object.

    """
    validate_estimator_keys(estimator)

    # accumulators distinguish estimate-free updates with `estimate is None`;
    # substituting a NullDistribution here would defeat those guards. Lazy import keeps the compute
    # layer free of concrete-distribution dependencies at module load.
    from mixle.stats.combinator.null_dist import NullDistribution

    if isinstance(prev_estimate, NullDistribution):
        prev_estimate = None

    if isinstance(data, RDD_TYPES):
        sc = data.context
        factory = estimator.accumulator_factory()
        estimator_broadcast = sc.broadcast(estimator)

        temp_estimate = pickle.dumps(prev_estimate, protocol=0)
        temp_estimate_b = sc.broadcast(temp_estimate)

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            loc_prev_estimate = pickle.loads(temp_estimate_b.value)

            for x in itr:
                counts_for_split = counts_for_split + 1.0
                accumulator_for_split.update(x, 1.0, estimate=loc_prev_estimate)

            return iter([(counts_for_split, accumulator_for_split.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobs_for_split, stats_for_split in temp.collect():
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        return estimator.estimate(nobs, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0

        for x in idata:
            nobs += 1.0
            accumulator.update(x, 1.0, estimate=prev_estimate)

        return estimator.estimate(nobs, accumulator.value())
