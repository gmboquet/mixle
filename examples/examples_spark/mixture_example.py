from pyspark import SparkContext, SparkConf
from pysp.stats import *
from pysp.stats.rdd_sampler import sample_rdd
from pysp.utils.estimation import empirical_kl_divergence
import numpy as np

if __name__ == '__main__':

	conf = SparkConf().setAppName("mixture_example")
	sc   = SparkContext(conf=conf)

	# Disable INFO/WARN printing
	log4j = sc._jvm.org.apache.log4j
	log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

	rng = np.random.RandomState(2)

	# Create some data

	d10 = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
	d11 = OptionalDistribution(CategoricalDistribution({'a': 0.5, 'b': 0.4, 'c': 0.1}), p=0.1)
	d12 = MarkovChainDistribution({'a' : 0.5, 'b' : 0.5}, {'a' : { 'a' : 0.2, 'b' : 0.8}, 'b' : { 'a' : 0.8, 'b' : 0.2}}, len_dist=PoissonDistribution(8.0))
	d13 = BernoulliSetDistribution({'a' : 0.1, 'b': 0.3})
	d1  = CompositeDistribution([d11, d10, d12, d13])

	d20 = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(6.0, 1.0)], [0.5, 0.5])
	d21 = OptionalDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.1, 'c': 0.8}), p=0.2)
	d22 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.8, 'b': 0.2}, 'b': {'a': 0.2, 'b': 0.8}}, len_dist=PoissonDistribution(8.0))
	d23 = BernoulliSetDistribution({'a': 0.9, 'b': 0.8})
	d2  = CompositeDistribution([d21, d20, d22, d23])

	dist = MixtureDistribution([d1, d2], [0.5, 0.5])


	# Compare to the local method of generating samples
	#
	#   sampler = dist.sampler(seed=rng.randint(2**32 - 1))
	#   data = sampler.sample(size=2000)

	data = sample_rdd(sc, dist, 200, 10, seed=rng.randint(2**32 - 1))

	# Compare to the local method of forming a random split
	#
	#    train_data, valid_data = partition_data(data, [0.9, 0.1], rng)

	train_data, valid_data = data.randomSplit([0.9, 0.1], seed=rng.randint(2**32-1))

	# Make the component estimator
	iest = MixtureEstimator([CompositeEstimator([OptionalEstimator(CategoricalEstimator(pseudo_count=1.0), est_prob=True, pseudo_count=1.0), MixtureEstimator([GaussianEstimator()]*2), MarkovChainEstimator(pseudo_count=1.0, len_estimator=PoissonEstimator()), BernoulliSetEstimator()])]*2, pseudo_count=1.0)
	est  = MixtureEstimator([CompositeEstimator([OptionalEstimator(CategoricalEstimator(), est_prob=True), MixtureEstimator([GaussianEstimator()]*2), MarkovChainEstimator(len_estimator=PoissonEstimator()), BernoulliSetEstimator()])]*2)


	# Estimate parameters
	# Note: Checkout pysp.utils.estimation.best_of/optimize for methods that handle this computation

	mm = initialize(train_data, iest, rng, 0.05)

	enc_data  = seq_encode(train_data, model=mm)
	enc_vdata = seq_encode(valid_data, model=mm)
	_, old_ll = seq_log_density_sum(enc_vdata, mm)
	_, old_tll = seq_log_density_sum(enc_data, mm)

	dll = np.inf
	its_cnt = 0
	while dll > 1.0e-8:

		its_cnt += 1

		mm_next  = seq_estimate(enc_data, est, mm)
		_, tll    = seq_log_density_sum(enc_data, mm_next)
		_, ll    = seq_log_density_sum(enc_vdata, mm_next)
		kl, _ ,_ = empirical_kl_divergence(mm_next, dist, enc_vdata)

		dll = tll - old_tll

		if dll >= 0:
			mm  = mm_next

		print('Iteration %d. LL=%f, delta LL=%e, val LL=%f, KL[Est||True|data]=%e' % (its_cnt+1, tll, dll, ll, kl))
		old_tll = tll

	print(str(mm))

