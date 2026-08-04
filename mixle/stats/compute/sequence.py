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
from contextlib import suppress
from typing import Any, TypeVar

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    merge_accumulator_keys,
    validate_estimator_keys,
)
from mixle.utils.optional_deps import RDD_TYPES, pyspark
from mixle.utils.vector import validate_initialization_probability, validated_initialized_observations

T = TypeVar("T")
T_D = TypeVar("T_D", bound=SequenceEncodableProbabilityDistribution)


def _release_broadcasts(*broadcasts: Any) -> None:
    """Best-effort release of Spark broadcasts without masking the primary result or failure."""
    for broadcast in broadcasts:
        if broadcast is None:
            continue
        with suppress(Exception):
            broadcast.destroy()


def _checked_encode(encoder: DataSequenceEncoder, rows: Any, path: str) -> tuple[int, Any]:
    declared = len(rows)
    payload = encoder.seq_encode(rows)
    observed = encoder.row_count(payload)
    if isinstance(observed, (bool, np.bool_)) or not isinstance(observed, (int, np.integer)):
        raise TypeError("%s encoder row_count() must return an integer." % path)
    if int(observed) != declared:
        raise ValueError(
            "%s encoded-row conservation failed: input has %d rows but payload reports %d." % (path, declared, observed)
        )
    return declared, payload


def _validate_encoded_chunks(chunks: Any, encoder: DataSequenceEncoder, path: str) -> Any:
    if isinstance(chunks, RDD_TYPES):
        return chunks
    checked = []
    for index, (declared, payload) in enumerate(chunks):
        observed = encoder.row_count(payload)
        if isinstance(declared, (bool, np.bool_)) or not isinstance(declared, (int, np.integer)):
            raise TypeError("%s chunk %d count must be an integer." % (path, index))
        if isinstance(observed, (bool, np.bool_)) or not isinstance(observed, (int, np.integer)):
            raise TypeError("%s chunk %d encoder row_count() must return an integer." % (path, index))
        if declared < 0 or int(observed) != int(declared):
            raise ValueError(
                "%s chunk %d encoded-row conservation failed: declared %d rows but payload reports %d."
                % (path, index, declared, observed)
            )
        checked.append((int(declared), payload))
    return checked


def _validated_update_chunk(
    accumulator: Any,
    declared: Any,
    payload: Any,
    path: str,
) -> tuple[int, np.ndarray]:
    if isinstance(declared, (bool, np.bool_)) or not isinstance(declared, (int, np.integer)):
        raise TypeError("%s chunk count must be a non-negative integer." % path)
    count = int(declared)
    if count < 0:
        raise ValueError("%s chunk count must be a non-negative integer." % path)
    observed = accumulator.acc_to_encoder().row_count(payload)
    if isinstance(observed, (bool, np.bool_)) or not isinstance(observed, (int, np.integer)):
        raise TypeError("%s encoder row_count() must return an integer." % path)
    if int(observed) != count:
        raise ValueError(
            "%s chunk metadata reports %d rows but its encoded payload contains %d." % (path, count, observed)
        )
    weights = np.ones(count, dtype=np.float64)
    if len(weights) != count:
        raise AssertionError("%s internal weight construction did not conserve row count." % path)
    return count, weights


def _partition_random_states(seeds: Any, split_index: int) -> tuple[np.random.RandomState, np.random.RandomState]:
    seed_values = np.asarray(seeds)
    if seed_values.ndim != 1 or not np.issubdtype(seed_values.dtype, np.integer):
        raise TypeError("partition seeds must be a one-dimensional integer array.")
    if isinstance(split_index, bool) or not isinstance(split_index, (int, np.integer)):
        raise TypeError("partition index must be an integer.")
    index = int(split_index)
    if not 0 <= index < len(seed_values):
        raise IndexError("partition index %d is outside the %d generated seeds." % (index, len(seed_values)))
    rng = np.random.RandomState(int(seed_values[index]))
    weight_rng = np.random.RandomState(seed=int(rng.randint(2**31)))
    return rng, weight_rng


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
    from mixle.data.partition import num_chunks_for

    # Validate both controls before selecting a backend. For streams whose size is
    # unknown here, zero is sufficient because only the control contract matters.
    control_size = len(data) if not isinstance(data, (DataSource, RDD_TYPES)) else 0
    validated_num_chunks = num_chunks_for(control_size, num_chunks=num_chunks, chunk_size=chunk_size)

    if isinstance(data, DataSource):
        chunks = data.encode(encoder, num_chunks=num_chunks, chunk_size=chunk_size)
        return _validate_encoded_chunks(chunks, encoder, type(data).__name__)

    if isinstance(data, RDD_TYPES):
        temp_encoder = pickle.dumps(encoder, protocol=0)

        def encode_partition(rows):
            return _checked_encode(pickle.loads(temp_encoder), rows, "RDD partition")  # nosec B301 # IPC: temp_encoder is the encoder this function pickled three lines above; the closure holding it is itself shipped to Spark executors by pickle

        enc_data = data.glom().map(lambda x: list(x)).map(encode_partition)

        return enc_data

    else:
        sz = len(data)
        num_chunks_loc = validated_num_chunks

        if num_chunks_loc <= 1:
            # single chunk: hand the data straight to the encoder -- the old element-by-element
            # rebuild copied (and, for arrays, boxed) the whole dataset for nothing (audit E-3)
            return [_checked_encode(encoder, data, "local chunk 0")]

        rv = []
        for i in range(num_chunks_loc):
            # a stride slice is C-speed for lists and a zero-copy view for arrays; the previous
            # per-element comprehension boxed every ndarray row and dominated encode time
            data_loc = data[i::num_chunks_loc]
            rv.append(_checked_encode(encoder, data_loc, "local chunk %d" % i))

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
        # parallel-backend handle (mixle.utils.parallel.multiprocessing / mixle.utils.parallel.mpi)
        return enc_data.pysp_seq_log_density_sum(estimate)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context
        estimate_broadcast = sc.broadcast(pickle.dumps(estimate, protocol=0))

        def acc(itr):

            rv = 0.0
            cnt = 0.0
            estimate_loc = pickle.loads(estimate_broadcast.value)  # nosec B301 # IPC: estimate_broadcast is the Spark broadcast this function created from its own pickle.dumps of the current estimate
            for sz, x in itr:
                rv += estimate_loc.seq_log_density(x).sum()
                cnt += sz

            return [(cnt, rv)]

        try:
            return enc_data.mapPartitions(acc).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        finally:
            _release_broadcasts(estimate_broadcast)

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
            loc_estimate = pickle.loads(estimate_broadcast.value)  # nosec B301 # IPC: estimate_broadcast is the Spark broadcast this function created from its own pickle.dumps of the current estimate
            if is_list:
                return [np.asarray([ee.seq_log_density(x) for ee in loc_estimate]) for sz, x in itr]
            else:
                return [loc_estimate.seq_log_density(x) for sz, x in itr]

        try:
            return enc_data.mapPartitions(acc).collect()
        finally:
            _release_broadcasts(estimate_broadcast)

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
    if not parts:
        return np.empty(0, dtype=float)
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
        # parallel-backend handle (mixle.utils.parallel.multiprocessing / mixle.utils.parallel.mpi)
        return enc_data.pysp_seq_estimate(estimator, prev_estimate)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context

        estimator_broadcast = sc.broadcast(estimator)
        estimate_broadcast = sc.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0
            local_estimate = pickle.loads(estimate_broadcast.value)  # nosec B301 # IPC: estimate_broadcast is the Spark broadcast this function created from its own pickle.dumps of the current estimate

            for chunk_index, (sz, x) in enumerate(itr):
                count, weights = _validated_update_chunk(
                    accumulator_for_split,
                    sz,
                    x,
                    "RDD partition %d chunk %d" % (split_index, chunk_index),
                )
                counts_for_split += count
                accumulator_for_split.seq_update(x, weights, local_estimate)

            rv = pickle.dumps((counts_for_split, accumulator_for_split.value()), protocol=0)

            return [rv]

        def red(x, y):
            xx = pickle.loads(x)  # nosec B301 # IPC: a treeReduce operand -- either a payload the acc closure above pickled on an executor or a partial this same reducer produced
            yy = pickle.loads(y)  # nosec B301 # IPC: a treeReduce operand -- either a payload the acc closure above pickled on an executor or a partial this same reducer produced
            accumulator = estimator_broadcast.value.accumulator_factory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))

            return rv

        try:
            # Fold in Spark via treeReduce (O(log W) levels) rather than a single-root collect --
            # the driver-memory/OOM risk at high partition counts flagged by the scaling audit and
            # fixed the same way in mixle.inference.spark_executor's spark_em_step/spark_fit.
            nobs, stats_value = pickle.loads(enc_data.mapPartitionsWithIndex(acc, True).treeReduce(red))  # nosec B301 # IPC: the final operand of the treeReduce on this line, produced by the acc/red closures above

            accumulator = estimator.accumulator_factory().make()
            accumulator.combine(stats_value)

            stats_dict = dict()
            accumulator.key_merge(stats_dict)
            accumulator.key_replace(stats_dict)

            return estimator.estimate(nobs, accumulator.value())
        finally:
            _release_broadcasts(estimate_broadcast, estimator_broadcast)

    else:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0

        for chunk_index, (sz, x) in enumerate(enc_data):
            count, weights = _validated_update_chunk(accumulator, sz, x, "local chunk %d" % chunk_index)
            nobs += count
            accumulator.seq_update(x, weights, prev_estimate)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())


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
    p = validate_initialization_probability(p)

    if hasattr(enc_data, "pysp_seq_initialize"):
        # parallel-backend handle (mixle.utils.parallel.multiprocessing / mixle.utils.parallel.mpi)
        return enc_data.pysp_seq_initialize(estimator, rng, p)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context
        num_partitions = enc_data.getNumPartitions()
        seeds = rng.randint(2**31, size=num_partitions)

        estimator_broadcast = sc.broadcast(estimator)
        seeds_broadcast = sc.broadcast(seeds)

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            rng_loc, rng_loc_w = _partition_random_states(seeds_broadcast.value, split_index)

            for sz, x in itr:
                w = np.zeros(sz, dtype=float)
                w_1 = rng_loc_w.rand(sz) <= p
                w[w_1] = 1.0

                counts_for_split += np.sum(w)
                accumulator_for_split.seq_initialize(x, w, rng_loc)

            rv = pickle.dumps((counts_for_split, accumulator_for_split.value()), protocol=0)
            return [rv]

        def red(x, y):
            xx = pickle.loads(x)  # nosec B301 # IPC: a treeReduce operand -- either a payload the acc closure above pickled on an executor or a partial this same reducer produced
            yy = pickle.loads(y)  # nosec B301 # IPC: a treeReduce operand -- either a payload the acc closure above pickled on an executor or a partial this same reducer produced
            accumulator = estimator_broadcast.value.accumulator_factory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))

            return rv

        try:
            # Fold in Spark via treeReduce (O(log W) levels) rather than a single-root collect --
            # the driver-memory/OOM risk at high partition counts flagged by the scaling audit and
            # fixed the same way in mixle.inference.spark_executor's spark_em_step/spark_fit.
            nobs, stats_value = pickle.loads(enc_data.mapPartitionsWithIndex(acc, True).treeReduce(red))  # nosec B301 # IPC: the final operand of the treeReduce on this line, produced by the acc/red closures above

            accumulator = estimator.accumulator_factory().make()
            accumulator.combine(stats_value)

            stats_dict = dict()
            accumulator.key_merge(stats_dict)
            accumulator.key_replace(stats_dict)

            return estimator.estimate(validated_initialized_observations(nobs), accumulator.value())
        finally:
            _release_broadcasts(seeds_broadcast, estimator_broadcast)

    else:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        rng_w = np.random.RandomState(seed=rng.randint(2**31 - 1))

        # Streams chunk by chunk, exactly as before, and repairs only the degenerate outcome at the
        # end. The mask is Bernoulli(p) with p defaulting to 0.1, so selecting nothing at all is not
        # rare on small data -- at two observations it is the majority outcome (0.81) -- and an
        # initialization that selected nothing carries no information about the data by construction.
        # Leaves disagreed about what to do with it: a Gaussian quietly returned a degenerate
        # estimate while RandomForestEstimator and the streaming transformer leaf raised "weights
        # must contain at least one positive value", so the copy-paste example in the docs for those
        # two families failed most of the time and the error named weights the caller never supplied.
        # _initialize_with_support_fallback cannot cover this: it inspects a model seq_initialize
        # RETURNED, and here seq_initialize raises.
        #
        # An earlier version of this drew every chunk's mask up front so the whole pass could be
        # checked before any of it reached the accumulator. That works but holds the entire mask set
        # in memory at once, which is a real cost on the large inputs this path exists for, so the
        # repair is deferred to the end instead -- matching the iterable branch below.
        fallback: tuple[Any, ...] = ()
        for sz, enc_x in enc_data:
            if not fallback:
                fallback = (enc_x,)
            w = rng_w.binomial(n=1, p=p, size=sz).astype(dtype=np.float64)
            accumulator.seq_initialize(enc_x, w, rng)
            nobs += float(w.sum())  # count the kept (weight-1) observations, matching the RDD/non-seq paths

        if nobs == 0.0 and fallback:
            # Nothing at all was selected. Seed the accumulator from the first chunk's leading row so
            # the estimator downstream has something to fit, rather than an empty initialization that
            # only some leaves tolerate.
            seed_mask = np.zeros(int(enc_data[0][0]), dtype=np.float64)
            seed_mask[0] = 1.0
            accumulator.seq_initialize(fallback[0], seed_mask, rng)
            nobs = 1.0

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(validated_initialized_observations(nobs), accumulator.value())


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
    p = validate_initialization_probability(p)

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

        try:
            temp = data.mapPartitionsWithIndex(acc, True)
            nobs = 0.0
            accumulator = factory.make()

            for nobs_for_split, stats_for_split in temp.collect():
                nobs = nobs + nobs_for_split
                accumulator.combine(stats_for_split)

            stats_dict = dict()
            accumulator.key_merge(stats_dict)
            accumulator.key_replace(stats_dict)

            return estimator.estimate(validated_initialized_observations(nobs), accumulator.value())
        finally:
            _release_broadcasts(seeds_broadcast, estimator_broadcast)

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        rng_w = np.random.RandomState(seed=rng.randint(2**31))

        # Retains the first observation as a fallback so the same guarantee as the seq_ path above
        # holds without buffering the data: a pass that selects nothing still hands the accumulator
        # one observation instead of an empty initialization that some leaves accept and others
        # reject. See the note in the seq_ branch.
        #
        # Deliberately NOT reservoir-sampled. Drawing a uniform index per observation would be the
        # nicer choice in isolation, but it consumes from rng_w on every iteration and therefore
        # shifts the whole Bernoulli mask that follows it -- silently changing the initialization of
        # every fit that reaches this path, not just the degenerate one this guards. A fixed choice
        # touches no random state at all, and in the only case it is ever used the pass had selected
        # nothing, so there is no distribution left to preserve.
        fallback: tuple[Any, ...] = ()
        for x in idata:
            if not fallback:
                fallback = (x,)
            w = rng_w.binomial(n=1, p=p)
            nobs += w
            accumulator.initialize(x, w, rng)

        if nobs == 0.0 and fallback:
            accumulator.initialize(fallback[0], 1.0, rng)
            nobs = 1.0

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(validated_initialized_observations(nobs), accumulator.value())


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
            loc_prev_estimate = pickle.loads(temp_estimate_b.value)  # nosec B301 # IPC: temp_estimate_b is the Spark broadcast this function created from its own pickle.dumps of the current estimate

            for x in itr:
                counts_for_split = counts_for_split + 1.0
                accumulator_for_split.update(x, 1.0, estimate=loc_prev_estimate)

            return iter([(counts_for_split, accumulator_for_split.value())])

        try:
            temp = data.mapPartitionsWithIndex(acc, True)
            nobs = 0.0
            accumulator = factory.make()

            for nobs_for_split, stats_for_split in temp.collect():
                nobs = nobs + nobs_for_split
                accumulator.combine(stats_for_split)

            merge_accumulator_keys(accumulator)
            return estimator.estimate(nobs, accumulator.value())
        finally:
            _release_broadcasts(temp_estimate_b, estimator_broadcast)

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0

        for x in idata:
            nobs += 1.0
            accumulator.update(x, 1.0, estimate=prev_estimate)

        merge_accumulator_keys(accumulator)
        return estimator.estimate(nobs, accumulator.value())
