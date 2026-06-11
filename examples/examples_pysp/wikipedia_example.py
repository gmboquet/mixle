"""Fit an LDA to data from wikipedia."""
import os
import sys
import time

import numpy as np

import pysp.utils.optsutil as ops
from pysp.stats import *

data_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')


def load_wiki_data():
    sword_loc = os.path.join(data_loc, 'stop_words')
    sword = set([''])
    for f in [os.path.join(sword_loc, 'mallet.txt')]:
        fin = open(os.path.join(sword_loc, f), 'rt')
        sword.update(fin.read().split('\n'))
        fin.close()

    wiki_loc = os.path.join(data_loc, 'wiki_example')
    files = [os.path.join(wiki_loc, u) for u in filter(lambda v: v.endswith('.txt'), os.listdir(wiki_loc))]
    data = ops.flatMap(lambda x: x,
                       [list(map(lambda u: list(filter(lambda v: v not in sword, u.split(' '))), ops.textFile(f))) for f
                        in files])
    data = list(filter(lambda u: len(u) > 0, data))

    words = sorted(set([u for v in data for u in v]))

    return data, words


if __name__ == '__main__':

    num_topics = 10
    print_cnt = 10
    rng = np.random.RandomState(2)
    # out = open('/Users/boquet1/PycharmProjects/wiki_debug.log', 'wt')
    out = sys.stdout

    data, words = load_wiki_data()

    avg_size = np.mean([len(u) for u in data])

    out.write('#words = %d / #docs = %d / avg w/doc = %f\n' % (len(words), len(data), avg_size))

    word_map = dict()
    data = [ops.map_to_integers(u, word_map) for u in data]
    data_cnt = [list(ops.count_by_value(u).items()) for u in data]

    word_map_inv = ops.get_inv_map(word_map)

    # estimator0 = CategoricalEstimator(pseudo_count=0.001, suff_stat={w : 1.0/len(words) for w in words})
    estimator0 = IntegerCategoricalEstimator(min_val=0, max_val=(len(word_map) - 1), pseudo_count=0.001)
    estimator1 = LDAEstimator([estimator0] * num_topics, keys=(None, 'topics'), gamma_threshold=1.0e-8)
    # estimator  = MixtureEstimator([estimator1]*2, pseudo_count=1.0)
    estimator = estimator1

    enc_data = seq_encode(data_cnt, estimator=estimator, num_chunks=1)

    imm = seq_initialize(enc_data, estimator, p=0.1, rng=np.random.RandomState(1))
    prev_model = imm

    dcnt, lob_sum = seq_log_density_sum(enc_data, imm)
    old_elob = lob_sum / dcnt

    for kk in range(300):

        t0 = time.time()
        mm = seq_estimate(enc_data, estimator, prev_estimate=prev_model)
        t1 = time.time()
        dcnt, lob_sum = seq_log_density_sum(enc_data, mm)
        elob = lob_sum / dcnt

        prev_model = mm
        out.write(
            'Iteration %d\tE[LoB]=%e\tdelta E[LoB]=%e\tdelta time=%f\n' % (kk + 1, elob, elob - old_elob, t1 - t0))

        old_elob = elob

        if (kk + 1) % print_cnt == 0:

            # out.write('Weights = %s\n'%(str(','.join(map(str, mm.w)))))
            # out.write('Alpha_2 = %s\n'%(str(','.join(map(str, mm.components[0].alpha)))))
            # out.write('Alpha_1 = %s\n'%(str(','.join(map(str, mm.components[1].alpha)))))
            # topics = mm.components[0].topics

            topics = mm.topics

            for i in np.argsort(-mm.alpha):
                sidx = np.argsort(-topics[i].log_p_vec)
                top_words = ', '.join(['%s (%f)' % (word_map_inv[j], np.exp(topics[i].log_p_vec[j])) for j in sidx[:10]])
                out.write('Topic %d [%f]: %s\n' % (i, mm.alpha[i], top_words))

        out.flush()
