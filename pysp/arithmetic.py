"""Backend-dispatched arithmetic helpers used by pysparkplug classes."""
from __future__ import annotations

import numpy as np

from pysp.engines import NUMPY_ENGINE, engine_of

__all__ = [
    'asarray', 'zeros', 'empty', 'arange', 'to_numpy',
    'log', 'exp', 'sqrt', 'abs', 'where', 'maximum', 'clip', 'floor', 'isnan', 'isinf',
    'dot', 'matmul', 'cumsum', 'logsumexp', 'stack',
    'bincount', 'index_add', 'unique', 'searchsorted',
    'gammaln', 'digamma', 'betaln', 'erf',
    'pi', 'maxint', 'maxrandint', 'one', 'zero', 'two', 'half', 'inf', 'eps',
]


def _dispatch(name):
    def fn(*args, **kwargs):
        engine = engine_of(args, default=NUMPY_ENGINE)
        return getattr(engine, name)(*args, **kwargs)
    fn.__name__ = name
    return fn


asarray = _dispatch('asarray')
zeros = _dispatch('zeros')
empty = _dispatch('empty')
arange = _dispatch('arange')
to_numpy = _dispatch('to_numpy')

log = _dispatch('log')
exp = _dispatch('exp')
sqrt = _dispatch('sqrt')
abs = _dispatch('abs')
where = _dispatch('where')
maximum = _dispatch('maximum')
clip = _dispatch('clip')
floor = _dispatch('floor')
isnan = _dispatch('isnan')
isinf = _dispatch('isinf')

sum = _dispatch('sum')
max = _dispatch('max')
dot = _dispatch('dot')
matmul = _dispatch('matmul')
cumsum = _dispatch('cumsum')
logsumexp = _dispatch('logsumexp')
stack = _dispatch('stack')

bincount = _dispatch('bincount')
index_add = _dispatch('index_add')
unique = _dispatch('unique')
searchsorted = _dispatch('searchsorted')

gammaln = _dispatch('gammaln')
digamma = _dispatch('digamma')
betaln = _dispatch('betaln')
erf = _dispatch('erf')


pi = np.pi
maxint = 2**31 - 1
maxrandint = 2**31 - 1
one = 1.0
zero = 0.0
two = 2.0
half = 0.5
inf = float('inf')
eps = 1.0e-8
