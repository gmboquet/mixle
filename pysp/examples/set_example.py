"""Fit a Mixture of a composite of Bernoulli set distribution and bag of words model for NIPS documents with authors,
represented by edges in a graph."""
import json
import os

os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':

    # Load the NIPs data
    fin = open('../../data/nips/all_submissions.json', 'rt', encoding='utf-8')
    data = json.load(fin)
    fin.close()

    # Extract the author sets
    papers = [([v['id'] for v in u['authors']], [v.lower() for v in u['title']]) for u in data]
    authors = set([v for u in papers for v in u[0]])
    words = set([v for u in papers for v in u[1]])

    est1 = BernoulliSetEstimator(pseudo_count=1.0e-8, suff_stat={u: 0.5 for u in authors})
    est2 = SequenceEstimator(CategoricalEstimator(pseudo_count=1.0, suff_stat={w: 1.0 / len(words) for w in words}))
    est3 = CompositeEstimator((est1, est2))
    est = MixtureEstimator([est3] * 10)

    model = optimize(papers, est, init_p=0.10, rng=np.random.RandomState(1), max_its=10)

    for comp in model.components:
        print(sorted(comp.dists[0].pmap.items(), key=lambda u: u[1], reverse=True)[:10])
        print(sorted(comp.dists[1].dist.pmap.items(), key=lambda u: u[1], reverse=True)[:10])
