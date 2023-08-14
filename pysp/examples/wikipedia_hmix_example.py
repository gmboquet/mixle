import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
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

    num_topics = 9
    print_cnt = 5
    rng = np.random.RandomState(2)
    out = sys.stdout

    data, words = load_wiki_data()

    avg_size = np.mean([len(u) for u in data])

    out.write('#words = %d / #docs = %d / avg w/doc = %f\n' % (len(words), len(data), avg_size))

    words_map = dict(zip(words, range(len(words))))
    inv_words_map = [None] * len(words_map)
    for w, i in words_map.items():
        inv_words_map[i] = w

    # data_cnt  = [list(ops.countByValue(u).items()) for u in data]
    data_cnt = [ops.map_to_integers(list(u), words_map) for u in data]

    estimator = HierarchicalMixtureEstimator([IntegerCategoricalEstimator(min_val=0, max_val=(len(words) - 1),
                                                                          pseudo_count=0.001, suff_stat=(
        0, np.ones(len(words)) / float(len(words))))] * num_topics, num_topics)

    imm = initialize(data_cnt, estimator, rng, 0.1)

    enc_data = seq_encode(data_cnt, model=imm, num_chunks=1)
    prev_model = imm

    dcnt, lob_sum = seq_log_density_sum(enc_data, imm)
    old_elob = lob_sum / dcnt

    for kk in range(5):

        t0 = time.time()
        mm = seq_estimate(enc_data, estimator, prev_estimate=prev_model)
        t1 = time.time()
        dcnt, lob_sum = seq_log_density_sum(enc_data, mm)
        elob = lob_sum / dcnt

        prev_model = mm
        out.write('Iteration %d\tE[LOB]=%e\tdelta E[LOB]=%e\tdt=%f\n' % (kk + 1, elob, elob - old_elob, t1 - t0))

        old_elob = elob

        if (kk + 1) % print_cnt == 0:

            alpha = np.dot(mm.taus.T, mm.w)
            alpha /= alpha.sum()

            out.write('Alpha = %s\n' % (str(','.join(map(str, alpha)))))

            topics = mm.topics

            for i in range(num_topics):
                sidx = np.argsort(-topics[i].p_vec)
                top_words = ', '.join(['%s (%f)' % (inv_words_map[j], topics[i].p_vec[j]) for j in sidx[:10]])
                out.write('Topic %d: %s\n' % (i, top_words))

        out.flush()
