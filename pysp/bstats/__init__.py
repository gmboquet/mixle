__all__ = ['BernoulliDistribution', 'BernoulliEstimator', 'BernoulliSampler',
           'BernoulliSetDistribution', 'BernoulliSetEstimator', 'BernoulliSetSampler',
           'BetaDistribution', 'BetaSampler',
           'BinomialDistribution', 'BinomialEstimator', 'BinomialSampler',
           'CategoricalDistribution', 'CategoricalEstimator', 'CategoricalSampler',
           'CompositeDistribution', 'CompositeEstimator', 'CompositeSampler',
           'DiagonalGaussianDistribution', 'DiagonalGaussianEstimator', 'DiagonalGaussianSampler',
           'DictDirichletDistribution',
           'DirichletDistribution', 'DirichletEstimator', 'DirichletSampler',
           'DirichletProcessMixtureDistribution', 'DirichletProcessMixtureEstimator', 'DirichletProcessMixtureSampler',
           'ExponentialDistribution', 'ExponentialEstimator', 'ExponentialSampler',
           'GaussianDistribution', 'GaussianEstimator', 'GaussianSampler',
           'GammaDistribution', 'GammaEstimator', 'GammaSampler',
           'GeometricDistribution', 'GeometricEstimator', 'GeometricSampler',
           'HiddenMarkovModelDistribution', 'HiddenMarkovModelEstimator', 'HiddenMarkovModelSampler',
           'HierarchicalDirichletProcessMixtureDistribution', 'HierarchicalDirichletProcessMixtureEstimator',
           'HierarchicalDirichletProcessMixtureSampler',
           'IgnoredDistribution', 'IgnoredEstimator', 'IgnoredSampler',
           'IntegerCategoricalDistribution', 'IntegerCategoricalEstimator', 'IntegerCategoricalSampler',
           'LogGaussianDistribution', 'LogGaussianEstimator', 'LogGaussianSampler',
           'MarkovChainDistribution', 'MarkovChainEstimator', 'MarkovChainSampler',
           'MixtureDistribution', 'MixtureEstimator', 'MixtureSampler',
           'MultivariateGaussianDistribution', 'MultivariateGaussianEstimator', 'MultivariateGaussianSampler',
           'MultivariateNormalGammaDistribution', 'MultivariateNormalGammaSampler',
           'NormalWishartDistribution', 'NormalWishartSampler',
           'NullDistribution', 'NullEstimator', 'NullSampler',
           'OptionalDistribution', 'OptionalEstimator', 'OptionalSampler',
           'PoissonDistribution', 'PoissonEstimator', 'PoissonSampler',
           'SequenceDistribution', 'SequenceEstimator', 'SequenceSampler',
           'estimate', 'seq_estimate', 'initialize', 'seq_log_density_sum', 'seq_encode', 'seq_log_density']


from pysp.arithmetic import *

from pysp.bstats.beta         import BetaDistribution, BetaSampler
from pysp.bstats.bernoulli    import BernoulliDistribution, BernoulliEstimator, BernoulliSampler
from pysp.bstats.binomial     import BinomialDistribution, BinomialEstimator, BinomialSampler
from pysp.bstats.categorical  import CategoricalDistribution, CategoricalEstimator, CategoricalSampler
from pysp.bstats.composite    import CompositeDistribution, CompositeEstimator, CompositeSampler
from pysp.bstats.catdirichlet import DictDirichletDistribution
from pysp.bstats.dirichlet    import DirichletDistribution, DirichletEstimator, DirichletSampler
from pysp.bstats.dmvn         import DiagonalGaussianDistribution, DiagonalGaussianEstimator, DiagonalGaussianSampler
from pysp.bstats.exponential  import ExponentialDistribution, ExponentialEstimator, ExponentialSampler
from pysp.bstats.gaussian     import GaussianDistribution, GaussianEstimator, GaussianSampler
from pysp.bstats.gamma        import GammaDistribution, GammaEstimator, GammaSampler
from pysp.bstats.geometric    import GeometricDistribution, GeometricEstimator, GeometricSampler
from pysp.bstats.ignored      import IgnoredDistribution, IgnoredEstimator, IgnoredSampler
from pysp.bstats.intrange     import IntegerCategoricalDistribution, IntegerCategoricalEstimator, IntegerCategoricalSampler
from pysp.bstats.log_gaussian import LogGaussianDistribution, LogGaussianEstimator, LogGaussianSampler
from pysp.bstats.markovchain  import MarkovChainDistribution, MarkovChainEstimator, MarkovChainSampler
from pysp.bstats.hdpm         import (HierarchicalDirichletProcessMixtureDistribution,
                                      HierarchicalDirichletProcessMixtureEstimator,
                                      HierarchicalDirichletProcessMixtureSampler)
from pysp.bstats.hidden_markov import HiddenMarkovModelDistribution, HiddenMarkovModelEstimator, HiddenMarkovModelSampler
from pysp.bstats.mixture      import MixtureDistribution, MixtureEstimator, MixtureSampler
from pysp.bstats.mvn          import MultivariateGaussianDistribution, MultivariateGaussianEstimator, MultivariateGaussianSampler
from pysp.bstats.mvngamma     import MultivariateNormalGammaDistribution, MultivariateNormalGammaSampler
from pysp.bstats.normwishart  import NormalWishartDistribution, NormalWishartSampler
from pysp.bstats.nulldist     import NullDistribution, NullEstimator, NullSampler
from pysp.bstats.optional     import OptionalDistribution, OptionalEstimator, OptionalSampler
from pysp.bstats.poisson      import PoissonDistribution, PoissonEstimator, PoissonSampler
from pysp.bstats.sequence     import SequenceDistribution, SequenceEstimator, SequenceSampler
from pysp.bstats.setdist      import BernoulliSetDistribution, BernoulliSetEstimator, BernoulliSetSampler

from pysp.bstats.dpm import DirichletProcessMixtureDistribution, DirichletProcessMixtureEstimator, DirichletProcessMixtureSampler


import numpy as np
import _pickle
import pickle

def load_models(x):
    return eval(x)

def dump_models(x):
    return str(x)



def _local_estimate(data, estimator, prev_estimate=None):
    idata = iter(data)
    accumulator = estimator.accumulator_factory().make()
    nobs = 0.0

    for x in idata:
        nobs += 1.0
        accumulator.update(x, 1.0, estimate=prev_estimate)

    stats_dict = dict()
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)

    return estimator.estimate(accumulator.value())


def estimate(data, estimator, prev_estimate=None):

    if 'pyspark.rdd' in str(type(data)):

        sc = data.context
        factory          = estimator.accumulatorFactory()
        estimatorBroadcast = sc.broadcast(estimator)

        temp_estimate  = pickle.dumps(prev_estimate, protocol=0)
        temp_estimateB = sc.broadcast(temp_estimate)

        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulatorFactory().make()
            countsForSplit      = 0.0
            loc_prev_estimate   = pickle.loads(temp_estimateB.value)

            for x in itr:
                countsForSplit = countsForSplit + 1.0
                accumulatorForSplit.update(x, 1.0, estimate=loc_prev_estimate)

            return(iter([(countsForSplit, accumulatorForSplit.value())])	)

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobsForSplit,statsForSplit in temp.collect():
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)

        return estimator.estimate(nobs, accumulator.value())

    elif 'pandas.core.frame.DataFrame' in str(type(data)):
        accumulator = estimator.accumulatorFactory().make()
        accumulator.df_update(data, np.ones(len(data)), estimate=prev_estimate)
        return estimator.estimate(None, accumulator.value())

    elif(hasattr(data, '__iter__')):
        return _local_estimate(data, estimator, prev_estimate)

def seq_encode(data, model, num_chunks=1, chunk_size=None):

    if 'pyspark.rdd' in str(type(data)):
        sc = data.context

        temp_model = pickle.dumps(model, protocol=0)
        modelBroadcast = sc.broadcast(temp_model)

        enc_data = data.glom().map(lambda x: list(x)).map(lambda x: (len(x), pickle.loads(modelBroadcast.value).seq_encode(x)))

        return enc_data

    else:
        sz = len(data)
        if chunk_size is not None:
            num_chunks_loc = int(np.ceil(float(sz)/float(chunk_size)))
        else:
            num_chunks_loc = num_chunks

        rv = []
        for i in range(num_chunks_loc):
            data_loc = [data[i] for i in range(i,sz,num_chunks_loc)]
            enc_data = model.seq_encode(data_loc)
            rv.append((len(data_loc), enc_data))

        return rv


def seq_log_density_sum(enc_data, estimate):

    if 'pyspark.rdd' in str(type(enc_data)):
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

        return enc_data.mapPartitions(acc).reduce(lambda a,b : (a[0]+b[0], a[1]+b[1]))

    else:

        return sum([u[0] for u in enc_data]), sum([estimate.seq_log_density(u[1]).sum() for u in enc_data])



def seq_log_density(enc_data, estimate, is_list=False):

    if 'pyspark.rdd' in str(type(enc_data)):
        sc = enc_data.context
        temp_estimate = pickle.dumps(estimate, protocol=0)
        estimateBroadcast = sc.broadcast(temp_estimate)

        def acc(itr):
            loc_estimate = pickle.loads(estimateBroadcast.value)
            if is_list:
                return [np.asarray([ee.seq_log_density(x) for ee in loc_estimate]) for sz, x in itr]
            else:
                return [loc_estimate.seq_log_density(x) for sz,x in itr]

        return enc_data.mapPartitions(acc).collect()

    else:

        if is_list:
            return [np.asarray([ee.seq_log_density(u[1]) for ee in estimate]) for u in enc_data]
        else:
            return [estimate.seq_log_density(u[1]) for u in enc_data]



def seq_estimate(enc_data, estimator, prev_estimate):

    if 'pyspark.rdd' in str(type(enc_data)):
        sc = enc_data.context

        estimatorBroadcast = sc.broadcast(estimator)
        estimateBroadcast  = sc.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulatorFactory().make()
            countsForSplit = zero
            local_estimate = pickle.loads(estimateBroadcast.value)

            for sz, x in itr:
                countsForSplit = countsForSplit + sz
                accumulatorForSplit.seq_update(x, np.ones(sz), local_estimate)

            rv = pickle.dumps((countsForSplit, accumulatorForSplit.value()), protocol=0)
            #return [(countsForSplit, accumulatorForSplit.value())]
            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimatorBroadcast.value.accumulatorFactory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))
            #return (nobs, vals)
            return rv


        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = zero
        accumulator = estimator.accumulatorFactory().make()

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

        return(estimator.estimate(nobs, accumulator.value()))

    else:

        accumulator = estimator.accumulator_factory().make()
        nobs        = 0.0

        data_update = []

        for sz, x in enc_data:
            nobs += sz
            accumulator.seq_update(x, np.ones(sz), prev_estimate)
            #x_update = accumulator.seq_update(x, np.ones(sz), prev_estimate)
            #data_update.append((sz, x_update))

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(accumulator.value())


def initialize(data, estimator, rng, p):

    if 'pyspark.rdd' in str(type(data)):
        factory = estimator.accumulator_factory()
        sc = data.context

        num_partitions = data.getNumPartitions()
        seeds = rng.randint(maxrandint, size=num_partitions)

        estimatorBroadcast = sc.broadcast(estimator)
        seedsBroadcast = sc.broadcast(seeds)


        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulator_factory().make()
            countsForSplit      = zero
            rng_loc = np.random.RandomState(seedsBroadcast.value[splitIndex])

            for x in itr:
                w = 1.0 if rng_loc.rand() <= p else 0.0
                countsForSplit += w
                accumulatorForSplit.initialize(x, w, rng_loc)

            return iter([(countsForSplit, accumulatorForSplit.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = zero
        accumulator = factory.make()

        for nobsForSplit,statsForSplit in temp.collect():
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)


        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return(estimator.estimate(nobs, accumulator.value()))

    elif 'pandas.core.frame.DataFrame' in str(type(data)):
        accumulator = estimator.accumulatorFactory().make()
        accumulator.df_initialize(data, rng.rand(len(data)) * p, rng)
        return estimator.estimate(None, accumulator.value())

    elif(hasattr(data, '__iter__')):
        idata       = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs        = 0.0

        for x in idata:
            w = 1.0 if rng.rand() <= p else 0.0
            nobs += w
            accumulator.initialize(x, w, rng)


        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(accumulator.value())


