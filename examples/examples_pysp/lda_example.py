"""Fit an LDA topic model with the high-level optimize() helper on self-contained random data.

Documents are bags of (word, count) pairs. A known LDADistribution (each topic favoring a distinct
trio of words) generates the corpus, and an LDAEstimator over Categorical topics recovers it. For the
low-level sequence-EM loop and an integer-vocab corpus, see wikipedia_example.py.
"""
import numpy as np

import pysp.utils.optsutil as ops
from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    num_topics = 4
    vocab = ['w%d' % i for i in range(12)]

    # Topic i favors words [3i, 3i+1, 3i+2].
    topics = []
    for i in range(num_topics):
        p = np.full(len(vocab), 0.02)
        p[3 * i:3 * i + 3] = 1.0
        topics.append(CategoricalDistribution(dict(zip(vocab, p / p.sum()))))

    true_lda = LDADistribution(topics, alpha=np.ones(num_topics) * 0.3,
                               len_dist=CategoricalDistribution({40: 1.0}))

    documents = true_lda.sampler(seed=1).sample(400)
    data = [list(ops.count_by_value(doc).items()) for doc in documents]

    estimator = LDAEstimator([CategoricalEstimator(pseudo_count=0.01)] * num_topics,
                             gamma_threshold=1.0e-6)

    model = optimize(data, estimator, max_its=60, print_iter=20, init_p=0.1,
                     rng=np.random.RandomState(2))

    print('alpha: %s' % np.round(model.alpha, 3))
    for i, topic in enumerate(model.topics):
        top = sorted(topic.pmap.items(), key=lambda kv: -kv[1])[:3]
        print('topic %d: %s' % (i, ', '.join('%s(%.2f)' % (w, p) for w, p in top)))
