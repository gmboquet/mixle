"""Fit the same mixture with the legacy path, the numba kernel engine, and the
PyTorch engine, and check they agree.

The accelerated engines compile/batch the whole mixture once and reuse the
ordinary estimators for the M-step, so results match the legacy vectorized
path; see pysp/stats/kernels.py and pysp/stats/torch_engine.py for the
supported families and contracts.
"""

import time

import numpy as np

from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    rng = np.random.RandomState(1)

    # A mixed-type model exercising several engine-supported families
    d1 = CompositeDistribution((
        GaussianDistribution(-2.0, 1.0),
        GammaDistribution(2.0, 3.0),
        CategoricalDistribution({'a': 0.7, 'b': 0.2, 'c': 0.1}),
        OptionalDistribution(PoissonDistribution(4.0), p=0.2),
    ))
    d2 = CompositeDistribution((
        GaussianDistribution(2.0, 1.0),
        GammaDistribution(5.0, 1.0),
        CategoricalDistribution({'a': 0.1, 'b': 0.3, 'c': 0.6}),
        OptionalDistribution(PoissonDistribution(9.0), p=0.05),
    ))
    dist = MixtureDistribution([d1, d2], [0.5, 0.5])

    data = dist.sampler(seed=rng.randint(2 ** 31)).sample(size=50000)

    est = MixtureEstimator([CompositeEstimator((
        GaussianEstimator(), GammaEstimator(),
        CategoricalEstimator(pseudo_count=0.5),
        OptionalEstimator(PoissonEstimator(), est_prob=True),
    ))] * 2)

    # Legacy vectorized path
    t0 = time.perf_counter()
    mm = optimize(data, est, max_its=20, rng=np.random.RandomState(2), print_iter=20)
    t_legacy = time.perf_counter() - t0

    # numba kernel engine: compile once, EM on compiled scalar kernels
    from pysp.stats.kernels import CompiledMixture
    eng = CompiledMixture(mm)
    enc = eng.encode(data)
    t0 = time.perf_counter()
    mm_k = mm
    for _ in range(20):
        mm_k = eng.em_step(enc, est, model=mm_k)
    t_kernel = time.perf_counter() - t0

    # torch engine: batched tensors; also supports fit_mle / fit_map
    from pysp.stats.torch_engine import TorchMixture
    tm = TorchMixture(mm)
    tenc = tm.encode(data)
    t0 = time.perf_counter()
    mm_t = mm
    for _ in range(20):
        mm_t = tm.em_step(tenc, est, model=mm_t)
    t_torch = time.perf_counter() - t0

    enc_data = seq_encode(data, model=mm)
    for name, m in (('legacy', mm), ('kernels', mm_k), ('torch', mm_t)):
        _, ll = seq_log_density_sum(enc_data, m)
        print('%-8s ll=%.6f' % (name, ll))
    print('20 EM iterations: legacy %.2fs | kernels %.2fs | torch %.2fs'
          % (t_legacy, t_kernel, t_torch))
