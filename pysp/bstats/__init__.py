"""Bayesian estimation package for pysparkplug (pysp.bstats).

Re-exports the Bayesian (conjugate-prior / variational) distributions and
estimators, and provides driver functions for fitting them to data held
locally (any iterable), in a pandas DataFrame, or in a pyspark RDD:

  - estimate / seq_estimate: one accumulate-then-estimate step (EM/VB update),
  - initialize: random-weight initialization of a model from data,
  - seq_encode: chunked sequence-encoding of data for vectorized updates,
  - seq_log_density / seq_log_density_sum: log-density evaluation over
    encoded data.

bstats estimators expose accumulator_factory() and estimate(suff_stat);
legacy stats-style estimators expose accumulatorFactory() and
estimate(nobs, suff_stat). The drivers below dispatch to whichever form an
estimator provides, preferring the bstats snake_case API.
"""

__all__ = [
    "BernoulliDistribution",
    "BernoulliEstimator",
    "BernoulliSampler",
    "BernoulliSetDistribution",
    "BernoulliSetEstimator",
    "BernoulliSetSampler",
    "BetaDistribution",
    "BetaSampler",
    "BinomialDistribution",
    "BinomialEstimator",
    "BinomialSampler",
    "CategoricalDistribution",
    "CategoricalEstimator",
    "CategoricalSampler",
    "CompositeDistribution",
    "CompositeEstimator",
    "CompositeSampler",
    "DiagonalGaussianDistribution",
    "DiagonalGaussianEstimator",
    "DiagonalGaussianSampler",
    "DictDirichletDistribution",
    "DirichletDistribution",
    "DirichletEstimator",
    "DirichletSampler",
    "DirichletProcessMixtureDistribution",
    "DirichletProcessMixtureEstimator",
    "DirichletProcessMixtureSampler",
    "ExponentialDistribution",
    "ExponentialEstimator",
    "ExponentialSampler",
    "GaussianDistribution",
    "GaussianEstimator",
    "GaussianSampler",
    "GammaDistribution",
    "GammaEstimator",
    "GammaSampler",
    "GeometricDistribution",
    "GeometricEstimator",
    "GeometricSampler",
    "HiddenMarkovModelDistribution",
    "HiddenMarkovModelEstimator",
    "HiddenMarkovModelSampler",
    "HierarchicalDirichletProcessMixtureDistribution",
    "HierarchicalDirichletProcessMixtureEstimator",
    "HierarchicalDirichletProcessMixtureSampler",
    "IgnoredDistribution",
    "IgnoredEstimator",
    "IgnoredSampler",
    "IntegerCategoricalDistribution",
    "IntegerCategoricalEstimator",
    "IntegerCategoricalSampler",
    "LogGaussianDistribution",
    "LogGaussianEstimator",
    "LogGaussianSampler",
    "MarkovChainDistribution",
    "MarkovChainEstimator",
    "MarkovChainSampler",
    "MixtureDistribution",
    "MixtureEstimator",
    "MixtureSampler",
    "mixture_prior",
    "MultivariateGaussianDistribution",
    "MultivariateGaussianEstimator",
    "MultivariateGaussianSampler",
    "MultivariateNormalGammaDistribution",
    "MultivariateNormalGammaSampler",
    "NormalGammaDistribution",
    "NormalGammaSampler",
    "NormalWishartDistribution",
    "NormalWishartSampler",
    "NullDistribution",
    "NullEstimator",
    "NullSampler",
    "OptionalDistribution",
    "OptionalEstimator",
    "OptionalSampler",
    "PoissonDistribution",
    "PoissonEstimator",
    "PoissonSampler",
    "SequenceDistribution",
    "SequenceEstimator",
    "SequenceSampler",
    "BayesianStreamingEstimator",
    "posterior_carry",
    "forgetting",
    "estimate",
    "seq_estimate",
    "initialize",
    "seq_log_density_sum",
    "seq_encode",
    "seq_log_density",
    "load_models",
    "dump_models",
]


import inspect
import pickle

import numpy as np

from pysp.arithmetic import *
from pysp.bstats.bernoulli import BernoulliDistribution, BernoulliEstimator, BernoulliSampler
from pysp.bstats.beta import BetaDistribution, BetaSampler
from pysp.bstats.binomial import BinomialDistribution, BinomialEstimator, BinomialSampler
from pysp.bstats.catdirichlet import DictDirichletDistribution
from pysp.bstats.categorical import CategoricalDistribution, CategoricalEstimator, CategoricalSampler
from pysp.bstats.composite import CompositeDistribution, CompositeEstimator, CompositeSampler
from pysp.bstats.dirichlet import DirichletDistribution, DirichletEstimator, DirichletSampler
from pysp.bstats.dmvn import DiagonalGaussianDistribution, DiagonalGaussianEstimator, DiagonalGaussianSampler
from pysp.bstats.dpm import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
    DirichletProcessMixtureSampler,
)
from pysp.bstats.exponential import ExponentialDistribution, ExponentialEstimator, ExponentialSampler
from pysp.bstats.gamma import GammaDistribution, GammaEstimator, GammaSampler
from pysp.bstats.gaussian import GaussianDistribution, GaussianEstimator, GaussianSampler
from pysp.bstats.geometric import GeometricDistribution, GeometricEstimator, GeometricSampler
from pysp.bstats.hdpm import (
    HierarchicalDirichletProcessMixtureDistribution,
    HierarchicalDirichletProcessMixtureEstimator,
    HierarchicalDirichletProcessMixtureSampler,
)
from pysp.bstats.hidden_markov import (
    HiddenMarkovModelDistribution,
    HiddenMarkovModelEstimator,
    HiddenMarkovModelSampler,
)
from pysp.bstats.ignored import IgnoredDistribution, IgnoredEstimator, IgnoredSampler
from pysp.bstats.int_range import IntegerCategoricalDistribution, IntegerCategoricalEstimator, IntegerCategoricalSampler
from pysp.bstats.log_gaussian import LogGaussianDistribution, LogGaussianEstimator, LogGaussianSampler
from pysp.bstats.markov_chain import MarkovChainDistribution, MarkovChainEstimator, MarkovChainSampler
from pysp.bstats.mixture import MixtureDistribution, MixtureEstimator, MixtureSampler, mixture_prior
from pysp.bstats.mvn import MultivariateGaussianDistribution, MultivariateGaussianEstimator, MultivariateGaussianSampler
from pysp.bstats.mvngamma import MultivariateNormalGammaDistribution, MultivariateNormalGammaSampler
from pysp.bstats.normgamma import NormalGammaDistribution, NormalGammaSampler
from pysp.bstats.normwishart import NormalWishartDistribution, NormalWishartSampler
from pysp.bstats.nulldist import NullDistribution, NullEstimator, NullSampler
from pysp.bstats.optional import OptionalDistribution, OptionalEstimator, OptionalSampler
from pysp.bstats.poisson import PoissonDistribution, PoissonEstimator, PoissonSampler
from pysp.bstats.sequence import SequenceDistribution, SequenceEstimator, SequenceSampler
from pysp.bstats.setdist import BernoulliSetDistribution, BernoulliSetEstimator, BernoulliSetSampler


def load_models(x):
    """Reconstruct a model or collection of models from dump_models() JSON."""
    from pysp.utils.serialization import from_json

    return from_json(x)


def dump_models(x):
    """Serialize a bstats model or collection of models to safe strict JSON."""
    from pysp.utils.serialization import to_json

    return to_json(x)


def _is_pandas_dataframe(data):
    """Return True when data is a pandas DataFrame.

    Dispatches on the type's module/name rather than isinstance so pandas
    need not be imported here, and so the check works across pandas versions
    (pandas 3 changed str(type(df)) from 'pandas.core.frame.DataFrame' to
    'pandas.DataFrame').

    Args:
        data: Candidate data container.

    Returns:
        bool: True when data is a pandas DataFrame.
    """
    t = type(data)
    return t.__name__ == "DataFrame" and t.__module__.split(".", 1)[0] == "pandas"


def _accumulator_factory(estimator):
    """Return the estimator's accumulator factory.

    Prefers the bstats snake_case accumulator_factory(); falls back to the
    legacy camelCase accumulatorFactory() for older stats-style estimators.

    Args:
        estimator: ParameterEstimator-like object.

    Returns:
        Factory object with a make() method producing accumulators.
    """
    factory_fn = getattr(estimator, "accumulator_factory", None)
    if factory_fn is None:
        factory_fn = estimator.accumulatorFactory
    return factory_fn()


def _estimator_estimate(estimator, nobs, suff_stat):
    """Call estimator.estimate() with whichever arity it supports.

    bstats estimators expose estimate(suff_stat); legacy stats-style
    estimators expose estimate(nobs, suff_stat). The arity is detected from
    the signature, defaulting to the bstats single-argument form when the
    signature cannot be inspected.

    Args:
        estimator: ParameterEstimator-like object.
        nobs (Optional[float]): Observation count passed only to legacy
            two-argument estimators.
        suff_stat: Sufficient statistics from an accumulator's value().

    Returns:
        The estimated distribution.
    """
    try:
        params = inspect.signature(estimator.estimate).parameters
        npos = len([p for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        npos = 1

    if npos >= 2:
        return estimator.estimate(nobs, suff_stat)
    else:
        return estimator.estimate(suff_stat)


def _local_estimate(data, estimator, prev_estimate=None):
    """Accumulate sufficient statistics over a local iterable and estimate.

    Args:
        data: Iterable of observations.
        estimator: bstats ParameterEstimator.
        prev_estimate: Previous model estimate passed to accumulator updates
            (required by estimators whose update step depends on the current
            model, e.g. mixtures).

    Returns:
        The estimated distribution.
    """
    idata = iter(data)
    accumulator = _accumulator_factory(estimator).make()
    nobs = 0.0

    for x in idata:
        nobs += 1.0
        accumulator.update(x, 1.0, estimate=prev_estimate)

    stats_dict = dict()
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)

    return _estimator_estimate(estimator, nobs, accumulator.value())


def estimate(data, estimator, prev_estimate=None):
    """Perform one accumulate-then-estimate step (EM/VB update) over data.

    Dispatches on the data container: pyspark RDDs are accumulated per
    partition and combined on the driver, pandas DataFrames use the
    accumulator's df_update path, and any other iterable is processed
    locally.

    Args:
        data: Iterable of observations, pandas DataFrame, or pyspark RDD.
        estimator: bstats ParameterEstimator (legacy camelCase/two-argument
            estimators are also supported).
        prev_estimate: Previous model estimate passed to accumulator updates.

    Returns:
        The estimated distribution.
    """
    if "pyspark.rdd" in str(type(data)):
        sc = data.context
        factory = _accumulator_factory(estimator)
        estimatorBroadcast = sc.broadcast(estimator)

        temp_estimate = pickle.dumps(prev_estimate, protocol=0)
        temp_estimateB = sc.broadcast(temp_estimate)

        def acc(splitIndex, itr):
            accumulatorForSplit = _accumulator_factory(estimatorBroadcast.value).make()
            countsForSplit = 0.0
            loc_prev_estimate = pickle.loads(temp_estimateB.value)

            for x in itr:
                countsForSplit = countsForSplit + 1.0
                accumulatorForSplit.update(x, 1.0, estimate=loc_prev_estimate)

            return iter([(countsForSplit, accumulatorForSplit.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobsForSplit, statsForSplit in temp.collect():
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return _estimator_estimate(estimator, nobs, accumulator.value())

    elif _is_pandas_dataframe(data):
        accumulator = _accumulator_factory(estimator).make()
        accumulator.df_update(data, np.ones(len(data)), estimate=prev_estimate)
        return _estimator_estimate(estimator, None, accumulator.value())

    elif hasattr(data, "__iter__"):
        return _local_estimate(data, estimator, prev_estimate)


def seq_encode(data, model, num_chunks=1, chunk_size=None):
    """Sequence-encode data with model.seq_encode for vectorized updates.

    Args:
        data: Iterable of observations or pyspark RDD.
        model: Distribution whose seq_encode() is used to encode the data.
        num_chunks (int): Number of chunks for local data (ignored for RDDs).
        chunk_size (Optional[int]): If given, overrides num_chunks so each
            chunk holds at most chunk_size observations.

    Returns:
        For local data, a list of (count, encoded_chunk) pairs; for RDDs, an
        RDD of (count, encoded_partition) pairs.
    """
    if "pyspark.rdd" in str(type(data)):
        sc = data.context

        temp_model = pickle.dumps(model, protocol=0)
        modelBroadcast = sc.broadcast(temp_model)

        enc_data = (
            data.glom().map(lambda x: list(x)).map(lambda x: (len(x), pickle.loads(modelBroadcast.value).seq_encode(x)))
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
            enc_data = model.seq_encode(data_loc)
            rv.append((len(data_loc), enc_data))

        return rv


def seq_log_density_sum(enc_data, estimate):
    """Total log-density of encoded data under a model estimate.

    Args:
        enc_data: Output of seq_encode() (local list or RDD of
            (count, encoded_chunk) pairs).
        estimate: Distribution used to evaluate seq_log_density.

    Returns:
        Tuple (count, total_log_density) summed over all chunks.
    """
    if "pyspark.rdd" in str(type(enc_data)):
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


def seq_log_density(enc_data, estimate, is_list=False):
    """Per-observation log-densities of encoded data under a model estimate.

    Args:
        enc_data: Output of seq_encode() (local list or RDD of
            (count, encoded_chunk) pairs).
        estimate: Distribution, or, when is_list is True, a list of
            distributions evaluated jointly on each chunk.
        is_list (bool): If True, evaluate each distribution in estimate and
            stack the per-chunk results.

    Returns:
        List of per-chunk numpy arrays of log-densities.
    """
    if "pyspark.rdd" in str(type(enc_data)):
        sc = enc_data.context
        temp_estimate = pickle.dumps(estimate, protocol=0)
        estimateBroadcast = sc.broadcast(temp_estimate)

        def acc(itr):
            loc_estimate = pickle.loads(estimateBroadcast.value)
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


def seq_estimate(enc_data, estimator, prev_estimate):
    """Perform one vectorized accumulate-then-estimate step on encoded data.

    Args:
        enc_data: Output of seq_encode() (local list or RDD of
            (count, encoded_chunk) pairs).
        estimator: bstats ParameterEstimator (legacy camelCase/two-argument
            estimators are also supported).
        prev_estimate: Previous model estimate passed to seq_update.

    Returns:
        The estimated distribution.
    """
    if "pyspark.rdd" in str(type(enc_data)):
        sc = enc_data.context

        estimatorBroadcast = sc.broadcast(estimator)
        estimateBroadcast = sc.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def acc(splitIndex, itr):
            accumulatorForSplit = _accumulator_factory(estimatorBroadcast.value).make()
            countsForSplit = zero
            local_estimate = pickle.loads(estimateBroadcast.value)

            for sz, x in itr:
                countsForSplit = countsForSplit + sz
                accumulatorForSplit.seq_update(x, np.ones(sz), local_estimate)

            rv = pickle.dumps((countsForSplit, accumulatorForSplit.value()), protocol=0)
            # return [(countsForSplit, accumulatorForSplit.value())]
            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = _accumulator_factory(estimatorBroadcast.value).make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))
            # return (nobs, vals)
            return rv

        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = zero
        accumulator = _accumulator_factory(estimator).make()

        for stuff in temp.collect():
            nobsForSplit, statsForSplit = pickle.loads(stuff)
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        estimateBroadcast.destroy()
        estimatorBroadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return _estimator_estimate(estimator, nobs, accumulator.value())

    else:
        accumulator = _accumulator_factory(estimator).make()
        nobs = 0.0

        data_update = []

        for sz, x in enc_data:
            nobs += sz
            accumulator.seq_update(x, np.ones(sz), prev_estimate)
            # x_update = accumulator.seq_update(x, np.ones(sz), prev_estimate)
            # data_update.append((sz, x_update))

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return _estimator_estimate(estimator, nobs, accumulator.value())


def initialize(data, estimator, rng, p):
    """Initialize a model by accumulating a random subsample of data.

    Each observation is included with probability p (weight 1.0, otherwise
    0.0) and routed through the accumulator's initialize path, then a first
    estimate is formed from the resulting sufficient statistics.

    Args:
        data: Iterable of observations, pandas DataFrame, or pyspark RDD.
        estimator: bstats ParameterEstimator (legacy camelCase/two-argument
            estimators are also supported).
        rng (numpy.random.RandomState): Source of subsampling randomness.
        p (float): Inclusion probability for each observation.

    Returns:
        The initialized distribution estimate.
    """
    if "pyspark.rdd" in str(type(data)):
        factory = _accumulator_factory(estimator)
        sc = data.context

        num_partitions = data.getNumPartitions()
        seeds = rng.randint(maxrandint, size=num_partitions)

        estimatorBroadcast = sc.broadcast(estimator)
        seedsBroadcast = sc.broadcast(seeds)

        def acc(splitIndex, itr):
            accumulatorForSplit = _accumulator_factory(estimatorBroadcast.value).make()
            countsForSplit = zero
            rng_loc = np.random.RandomState(seedsBroadcast.value[splitIndex])

            for x in itr:
                w = 1.0 if rng_loc.rand() <= p else 0.0
                countsForSplit += w
                accumulatorForSplit.initialize(x, w, rng_loc)

            return iter([(countsForSplit, accumulatorForSplit.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = zero
        accumulator = factory.make()

        for nobsForSplit, statsForSplit in temp.collect():
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return _estimator_estimate(estimator, nobs, accumulator.value())

    elif _is_pandas_dataframe(data):
        accumulator = _accumulator_factory(estimator).make()
        accumulator.df_initialize(data, rng.rand(len(data)) * p, rng)
        return _estimator_estimate(estimator, None, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = _accumulator_factory(estimator).make()
        nobs = 0.0

        for x in idata:
            w = 1.0 if rng.rand() <= p else 0.0
            nobs += w
            accumulator.initialize(x, w, rng)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return _estimator_estimate(estimator, nobs, accumulator.value())


from pysp.bstats.bestimation import BayesianStreamingEstimator, forgetting, posterior_carry
