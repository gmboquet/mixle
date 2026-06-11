from pyspark import SparkContext, SparkConf
from pysp.stats import *
from pysp.stats.rdd_sampler import sample_rdd
from pysp.utils.estimation import best_of
import numpy as np

if __name__ == '__main__':

	conf = SparkConf().setAppName("mixture_example")
	sc   = SparkContext(conf=conf)

	# Disable INFO/WARN printing
	log4j = sc._jvm.org.apache.log4j
	log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

	rng = np.random.RandomState(2)

	v = 0.1
	d11  = CompositeDistribution((GaussianDistribution(-20.0, 1.0), CategoricalDistribution({'a': 0.5 - v, 'b': 0.5 - v, 'c': v, 'd': v})))
	d12  = CompositeDistribution((GaussianDistribution( 0.0, 1.0), CategoricalDistribution({'a': v, 'b': v, 'c': 0.5 - v, 'd': 0.5 - v})))
	d13  = CompositeDistribution((GaussianDistribution( 20.0, 1.0), CategoricalDistribution({'a': v, 'b': 0.5 - v, 'c': 0.5 - v, 'd': v})))

	v = 0.1
	d21 = CompositeDistribution((GaussianDistribution(-5.0, 1.0), CategoricalDistribution({'a': 0.5 - v, 'b': 0.5 - v, 'c': v, 'd': v})))
	d22 = CompositeDistribution((GaussianDistribution( 0.0, 1.0), CategoricalDistribution({'a': v, 'b': v, 'c': 0.5 - v, 'd': 0.5 - v})))
	d23 = CompositeDistribution((GaussianDistribution( 5.0, 1.0), CategoricalDistribution({'a': v, 'b': 0.5 - v, 'c': 0.5 - v, 'd': v})))

	dist1 = HiddenMarkovModelDistribution([d11, d12, d13], [0.4, 0.4, 0.2], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], None, len_dist=PoissonDistribution(8.0))
	dist2 = HiddenMarkovModelDistribution([d21, d22, d23], [0.4, 0.4, 0.2], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], None, len_dist=PoissonDistribution(8.0))
	dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])

	sampler = dist.sampler(seed=rng.randint(2**32 - 1))

	data = sample_rdd(sc, dist, 200, 10, seed=rng.randint(2**32 - 1))

	train_data, valid_data = data.randomSplit([0.9, 0.1], seed=rng.randint(2**32-1))


	est1  = CompositeEstimator((GaussianEstimator(), CategoricalEstimator(pseudo_count=1.0)))
	est2  = HiddenMarkovEstimator([est1]*3, pseudo_count=(1.0, 1.0), keys=('init_key', 'trans_key', None))
	iest  = MixtureEstimator([est2]*2, pseudo_count=1.0)

	est1 = CompositeEstimator((GaussianEstimator(), CategoricalEstimator()))
	est2 = HiddenMarkovEstimator([est1]*3, keys=('init_key', 'trans_key', None))
	est  = MixtureEstimator([est2]*2)

	ll, mm = best_of(train_data, valid_data, est, 20, 100, 0.01, 1.0e-8, rng, init_estimator=iest)

	print(str(mm))
