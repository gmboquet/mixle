"""Mixture of (author-set x bag-of-title-words) documents on self-contained random data.

Each "document" is a tuple (set of author ids, list of title words). A BernoulliSetDistribution
models which authors are present (presence/absence over a fixed label universe) and a
SequenceDistribution bag-of-words models the title; the two views are joined with a
CompositeDistribution and clustered with a MixtureDistribution. No external corpus is needed --
the data is sampled from a known three-component model and the fit recovers it.
"""
import numpy as np

from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    authors = ['a%d' % i for i in range(8)]
    words = ['w%d' % i for i in range(10)]

    def component(author_subset, word_subset):
        author_probs = {a: (0.7 if a in author_subset else 0.05) for a in authors}
        word_probs = np.full(len(words), 0.02)
        for w in word_subset:
            word_probs[words.index(w)] = 1.0
        word_probs = word_probs / word_probs.sum()
        return CompositeDistribution((
            BernoulliSetDistribution(author_probs),
            SequenceDistribution(CategoricalDistribution(dict(zip(words, word_probs))),
                                 len_dist=CategoricalDistribution({4: 0.5, 5: 0.5})),
        ))

    dist = MixtureDistribution(
        [component({'a0', 'a1'}, ['w0', 'w1', 'w2']),
         component({'a2', 'a3', 'a4'}, ['w3', 'w4', 'w5']),
         component({'a5', 'a6'}, ['w6', 'w7', 'w8', 'w9'])],
        [0.40, 0.35, 0.25])

    data = dist.sampler(seed=1).sample(1500)

    est_component = CompositeEstimator((
        BernoulliSetEstimator(pseudo_count=1.0e-3),
        SequenceEstimator(CategoricalEstimator(pseudo_count=1.0)),
    ))
    est = MixtureEstimator([est_component] * 3)

    model = optimize(data, est, max_its=100, init_p=0.10, rng=np.random.RandomState(1))

    print('mixture weights: %s' % np.round(model.w, 3))
    for k, comp in enumerate(model.components):
        top_authors = sorted(comp.dists[0].pmap.items(), key=lambda kv: -kv[1])[:3]
        top_words = sorted(comp.dists[1].dist.pmap.items(), key=lambda kv: -kv[1])[:4]
        print('component %d:' % k)
        print('  authors: %s' % ', '.join('%s=%.2f' % (a, p) for a, p in top_authors))
        print('  words:   %s' % ', '.join('%s=%.2f' % (w, p) for w, p in top_words))
