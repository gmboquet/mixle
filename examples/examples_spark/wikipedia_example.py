"""Run distributed topic-model estimation for the Wikipedia example corpus."""

import os
import sys
import time
import numpy as np
from pyspark import SparkContext, SparkConf
from pysp.stats import *
import pysp.utils.optsutil as ops

data_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')


def load_wiki_data():

	sword_loc = os.path.join(data_loc, 'stop_words')
	sword = set([''])
	for f in [os.path.join(sword_loc,'mallet.txt')]:
		fin = open(os.path.join(sword_loc, f), 'rt')
		sword.update(fin.read().split('\n'))
		fin.close()

	wiki_loc  = os.path.join(data_loc, 'wiki_example')
	files     = [os.path.join(wiki_loc,u) for u in filter(lambda v: v.endswith('.txt'), os.listdir(wiki_loc))]
	data      = ops.flatMap(lambda x: x, [list(map(lambda u: list(filter(lambda v: v not in sword, u.split(' '))), ops.textFile(f))) for f in files])
	data      = list(filter(lambda u: len(u) > 0, data))

	words = sorted(set([u for v in data for u in v]))

	return data, words


if __name__ == '__main__':

	conf = SparkConf().setAppName("wikipedia_example")
	sc   = SparkContext(conf=conf)

	# Disable INFO/WARN printing
	log4j = sc._jvm.org.apache.log4j
	log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

	num_topics = 10
	print_cnt = 10
	rng = np.random.RandomState(2)
	#out = open('/Users/boquet1/PycharmProjects/wiki_debug.log', 'wt')
	out = sys.stdout


	data, words = load_wiki_data()

	avg_size = np.mean([len(u) for u in data])

	out.write('#words = %d / #docs = %d / avg w/doc = %f\n' % (len(words), len(data), avg_size))


	word_map = dict()
	data = [ops.map_to_integers(u, word_map) for u in data]
	data_cnt = [list(ops.countByValue(u).items()) for u in data]
	word_map_inv = ops.get_inv_map(word_map)


	estimator0 = IntegerCategoricalEstimator(minVal=0, maxVal=(len(word_map)-1), pseudo_count=0.001)
	estimator1 = LDAEstimator([estimator0]*num_topics, keys=(None, 'topics'), gamma_threshold=1.0e-8)

	estimator = estimator1

	# The only difference between the local and spark versions is that we parallelize the data
	data_cnt = sc.parallelize([list(ops.countByValue(u).items()) for u in data], 4)


	imm = initialize(data_cnt, estimator, rng, 0.1)

	enc_data   = seq_encode(data_cnt, model=imm)
	prev_model = imm

	dcnt, lob_sum = seq_log_density_sum(enc_data, imm)
	old_elob = lob_sum / dcnt


	for kk in range(300):
		t0 = time.time()
		mm = seq_estimate(enc_data, estimator, prev_estimate=prev_model)
		t1 = time.time()

		dcnt, lob_sum = seq_log_density_sum(enc_data, mm)
		elob = lob_sum/dcnt

		prev_model = mm
		out.write('Iteration %d\tE[LoB]=%e\tdelta E[LoB]=%e\tdelta time=%f\n'%(kk+1, elob, elob-old_elob, t1-t0))

		old_elob = elob

		if (kk+1) % print_cnt == 0:

			topics = mm.topics

			for i in np.argsort(-mm.alpha):
				sidx = np.argsort(-topics[i].logPVec)
				top_words = ', '.join(['%s (%f)'%(word_map_inv[j], np.exp(topics[i].logPVec[j])) for j in sidx[:10]])
				out.write('Topic %d [%f]: %s\n'%(i, mm.alpha[i], top_words))


		out.flush()
