"""LDA example of fake data ranking top words in each document for model fits."""
import sys
import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'
import numpy as np

import pysp.utils.optsutil as ops
from pysp.stats import *


def make_fake_data(num_topics, num_docs, snr, p_alpha, seed):
    word_per_doc = 100
    num_words = 10
    num_topics = num_topics

    rng = np.random.RandomState(seed)

    alpha1 = p_alpha * np.ones(num_topics)
    alpha1[np.arange(num_topics) >= (num_topics / 2)] = 0.0001

    alpha2 = p_alpha * np.ones(num_topics)
    alpha2[np.arange(num_topics) < (num_topics / 2)] = 0.0001

    topics = [{k: snr * rng.rand() + (
        1.0 if (i * num_words / num_topics) <= k and ((i + 1) * num_words / num_topics) > k else 0.0) for k in
               range(num_words)} for i in range(num_topics)]
    topics = [CategoricalDistribution({str(k): v / float(sum(u.values())) for k, v in u.items()}) for u in topics]

    # Create the data
    dist1 = LDADistribution(topics, alpha1, len_dist=CategoricalDistribution({word_per_doc: 1.0}))
    dist2 = LDADistribution(topics, alpha2, len_dist=CategoricalDistribution({word_per_doc: 1.0}))
    dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])

    data = dist.sampler(seed=1).sample(size=num_docs)

    words = sorted(set([u for v in data for u in v]))

    return data, words, dist


if __name__ == '__main__':

    num_topics = 10
    print_cnt = 10
    rng = np.random.RandomState(2)
    # out = open('/Users/boquet1/PycharmProjects/lda_debug.log', 'wt')
    out = sys.stdout

    data, words, dist = make_fake_data(num_topics, 50, 0.0001, 1, 1)

    avg_size = np.mean([len(u) for u in data])

    out.write('#words = %d / #docs = %d / avg w/doc = %f\n' % (len(words), len(data), avg_size))

    data_cnt = [list(ops.count_by_value(u).items()) for u in data]

    estimator1 = LDAEstimator(
        [CategoricalEstimator(pseudo_count=0.001, suff_stat={w: 1.0 / len(words) for w in words})] * num_topics,
        keys=(None, None), gamma_threshold=0.001)
    estimator = MixtureEstimator([estimator1] * 2, pseudo_count=1.0)

    enc_data = seq_encode(data_cnt, estimator.accumulator_factory().make().acc_to_encoder())
    imm = seq_initialize(enc_data, estimator, rng=np.random.RandomState(1), p=0.10)
    prev_model = imm

    dcnt, lob_sum = seq_log_density_sum(enc_data, imm)
    old_elob = lob_sum / dcnt
    d_elob = np.inf
    kk = -1

    while d_elob > 1.0e-8:
        kk += 1
        mm = seq_estimate(enc_data, estimator, prev_estimate=prev_model)

        dcnt, lob_sum = seq_log_density_sum(enc_data, mm)
        elob = lob_sum / dcnt

        prev_model = mm
        out.write('Iteration %d\tE[LOB]=%e\tdelta E[LOB]=%e\n' % (kk + 1, elob, elob - old_elob))

        old_elob = elob

        if (kk + 1) % print_cnt == 0:

            out.write('Weights = %s\n' % (str(','.join(map(str, mm.w)))))
            out.write('Alpha_2 = %s\n' % (str(','.join(map(str, mm.components[0].alpha)))))
            out.write('Alpha_1 = %s\n' % (str(','.join(map(str, mm.components[1].alpha)))))
            topics = mm.components[0].topics

            for i in range(num_topics):
                log_prob_vec = np.asarray([x for x in topics[i].pmap.values()])
                vals = np.asarray([x for x in topics[i].pmap.keys()])

                sidx = np.argsort(-log_prob_vec)
                top_words = ', '.join(
                    ['%s (%f)' % (vals[j], np.exp(log_prob_vec[j])) for j in sidx])
                out.write('Topic %d: %s\n' % (i, top_words))

        out.flush()
