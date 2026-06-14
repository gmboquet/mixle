"""Fit a hierarchical mixture topic model to a self-contained synthetic corpus.

A HierarchicalMixtureDistribution models a bag of words as a mixture over "documents" (the outer
weights ``w``), each of which is its own mixture over shared word topics (the ``taus`` rows). Here a
synthetic corpus is sampled from a known model whose topics each favor a contiguous block of the
vocabulary, then recovered and the per-topic top words are printed. No external corpus required.
"""
import sys

import numpy as np

from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    num_topics = 6
    num_words = 60
    num_docs = 1000
    out = sys.stdout

    block = num_words // num_topics
    topics = []
    for i in range(num_topics):
        p = np.full(num_words, 0.2)
        p[i * block:(i + 1) * block] += 5.0
        topics.append(IntegerCategoricalDistribution(0, p / p.sum()))

    # Three outer "document profiles", each a different mixture over the shared topics.
    taus = [[0.7, 0.2, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.2, 0.1, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.2, 0.7]]
    w = [0.5, 0.3, 0.2]
    len_dist = CategoricalDistribution({40: 0.5, 50: 0.5})
    truth = HierarchicalMixtureDistribution(topics, w, taus, len_dist=len_dist)

    data = truth.sampler(seed=2).sample(num_docs)
    out.write('#docs=%d  vocab=%d  topics=%d  profiles=%d\n'
              % (num_docs, num_words, num_topics, len(w)))

    topic_est = IntegerCategoricalEstimator(min_val=0, max_val=num_words - 1, pseudo_count=0.01)
    estimator = HierarchicalMixtureEstimator([topic_est] * num_topics, num_mixtures=len(w),
                                             len_estimator=CategoricalEstimator())

    model = optimize(data, estimator, max_its=80, print_iter=20, rng=np.random.RandomState(3))

    overall = np.dot(np.asarray(model.taus).T, model.w)
    out.write('overall topic weights: %s\n' % np.round(overall / overall.sum(), 3))
    for i in range(num_topics):
        log_p = model.topics[i].log_p_vec
        top = np.argsort(-log_p)[:block]
        out.write('topic %d: %s\n' % (i, ', '.join('w%d(%.2f)' % (v, np.exp(log_p[v])) for v in top)))
