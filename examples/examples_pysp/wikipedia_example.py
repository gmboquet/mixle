"""Fit an LDA topic model to a self-contained synthetic corpus.

A "corpus" of integer-encoded documents is sampled from a known LDADistribution (each topic favors a
contiguous block of the vocabulary), then recovered with an LDAEstimator over IntegerCategorical
topics. Demonstrates the manual sequence-EM loop (seq_encode / seq_initialize / seq_estimate /
seq_log_density_sum) and ranking the top words per learned topic. No external corpus required.
"""
import sys

import numpy as np

import pysp.utils.optsutil as ops
from pysp.stats import *

if __name__ == '__main__':
    num_topics = 6
    num_words = 60
    num_docs = 800
    words_per_doc = 80
    out = sys.stdout

    # True topic model: topic i puts most of its mass on a contiguous block of the vocabulary.
    rng = np.random.RandomState(1)
    block = num_words // num_topics
    true_topics = []
    for i in range(num_topics):
        p = np.full(num_words, 0.2)
        p[i * block:(i + 1) * block] += 5.0
        true_topics.append(IntegerCategoricalDistribution(0, p / p.sum()))
    true_lda = LDADistribution(true_topics, alpha=np.ones(num_topics) * 0.2,
                               len_dist=CategoricalDistribution({words_per_doc: 1.0}))

    documents = true_lda.sampler(seed=2).sample(num_docs)
    data = [list(ops.count_by_value(doc).items()) for doc in documents]
    out.write('#docs=%d  vocab=%d  words/doc=%d\n' % (num_docs, num_words, words_per_doc))

    # Estimator: one IntegerCategorical topic per latent topic, fit by sequence EM.
    topic_est = IntegerCategoricalEstimator(min_val=0, max_val=num_words - 1, pseudo_count=0.01)
    estimator = LDAEstimator([topic_est] * num_topics, gamma_threshold=1.0e-6)

    enc_data = seq_encode(data, estimator=estimator)
    model = seq_initialize(enc_data, estimator, rng=np.random.RandomState(3), p=0.1)

    count, ll_sum = seq_log_density_sum(enc_data, model)
    prev = ll_sum / count
    for it in range(40):
        model = seq_estimate(enc_data, estimator, prev_estimate=model)
        count, ll_sum = seq_log_density_sum(enc_data, model)
        elbo = ll_sum / count
        if it % 10 == 0 or abs(elbo - prev) < 1.0e-6:
            out.write('iteration %3d  E[LB]=%e  delta=%e\n' % (it + 1, elbo, elbo - prev))
        if abs(elbo - prev) < 1.0e-6:
            break
        prev = elbo

    for i in np.argsort(-model.alpha):
        log_p = model.topics[i].log_p_vec
        top = np.argsort(-log_p)[:block]
        out.write('topic %d [alpha=%.3f]: %s\n'
                  % (i, model.alpha[i], ', '.join('w%d(%.2f)' % (w, np.exp(log_p[w])) for w in top)))
