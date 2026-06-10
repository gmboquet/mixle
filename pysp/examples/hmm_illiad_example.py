"""Fit an HMM to text from the Iliad, comparing Numba use and fit without Numba.

The text (public domain, Project Gutenberg #2199) is downloaded on first run
and cached under data/iliad/ at the repository root.
"""
import os
import re
import time
import urllib.request

import numpy as np

from pysp.stats import *
from pysp.utils.estimation import optimize
from pysp.utils.optsutil import map_to_integers

ILIAD_URL = 'https://www.gutenberg.org/cache/epub/2199/pg2199.txt'
ILIAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'data', 'iliad', 'iliad_en.txt')


def load_iliad_text() -> str:
    path = os.path.normpath(ILIAD_PATH)
    if not os.path.exists(path):
        print('Downloading the Iliad from %s -> %s' % (ILIAD_URL, path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(ILIAD_URL, path)
    with open(path, 'rt', encoding='utf-8') as fin:
        return fin.read()


if __name__ == '__main__':
    rng = np.random.RandomState(2)

    data = load_iliad_text()
    words = re.split(r'\s+', data)
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
