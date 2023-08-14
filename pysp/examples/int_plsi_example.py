"""Integer PLSI example on generated data."""
import sys
import numpy as np

import pysp.utils.optsutil as ops
from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    n_docs = 10000
    num_states = 3
    num_authors = 10
    num_words = 50

    rng = np.random.RandomState(1)
    state_word_mat = rng.dirichlet(alpha=np.ones(num_words), size=num_states).T
    doc_state_mat = rng.dirichlet(alpha=np.ones(num_states), size=num_authors)
    doc_vec = rng.dirichlet(alpha=np.ones(num_authors))

    d = IntegerPLSIDistribution(state_word_mat=state_word_mat, doc_state_mat=doc_state_mat, doc_vec=doc_vec,
                                len_dist=CategoricalDistribution({5: 0.25, 6: 0.25, 10: 0.25, 12: 0.25}))
    data = d.sampler(seed=10).sample(n_docs)
    print(data[:10])

    est = IntegerPLSIEstimator(num_vals=num_words, num_states=num_states, num_docs=num_authors,
                               len_estimator=CategoricalEstimator())

    fit = optimize(data=data, estimator=est, init_p=0.10, rng=np.random.RandomState(2), max_its=200, print_iter=10)



