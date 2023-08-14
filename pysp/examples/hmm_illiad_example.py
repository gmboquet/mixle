"""Fit an HMM to text from the Iliad, comparing Numba use and fit without Numba."""
import re
import time

import numpy as np

from pysp.stats import *
from pysp.utils.estimation import optimize
from pysp.utils.optsutil import map_to_integers

if __name__ == '__main__':
    rng = np.random.RandomState(2)

    fin = open('../../data/iliad/iliad_en.txt', 'rt', encoding='utf-8')
    data = fin.read()
    words = re.split('\s+', data)
    fin.close()
    m = len(words)
    n = 100
    wmap = dict()
    chunks = [words[(i * n):min((i + 1) * n, m)] for i in range(int(len(words) / n))]
    chunks = [map_to_integers(x, wmap) for x in chunks[:100]]

    est = IntegerCategoricalEstimator(min_val=0, max_val=len(wmap) - 1, pseudo_count=1.0)
    est = HiddenMarkovEstimator([est] * 10, use_numba=False)
    imodel = optimize(chunks, est, max_its=1, rng=np.random.RandomState(1), init_p=1.0)

    t00 = time.time()
    model = optimize(chunks, est, max_its=200, prev_estimate=imodel, print_iter=200)
    t01 = time.time()
    print(t01 - t00)

    est = IntegerCategoricalEstimator(min_val=0, max_val=len(wmap) - 1, pseudo_count=1.0)
    est = HiddenMarkovEstimator([est] * 10, use_numba=True)
    imodel = optimize(chunks, est, max_its=1, rng=np.random.RandomState(1), init_p=1.0)

    t10 = time.time()
    model = optimize(chunks, est, max_its=200, prev_estimate=imodel, print_iter=200)
    t11 = time.time()
    print(t11 - t10)

    print('Speedup = %f' % ((t01 - t00) / (t11 - t10)))
