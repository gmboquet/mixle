"""Fit the same mixture with the NumPy sequence path and modular compute engines.

The accelerated paths swap only the scoring/accumulation kernel. Distribution
modules still own their likelihood math and ordinary estimators still own the
M-step, so results match the ordinary vectorized sequence path.
"""

import io
import time

import numpy as np

from pysp.engines import NUMPY_ENGINE, TorchEngine, torch
from pysp.stats import *
from pysp.stats.compute.kernel import NumbaKernelFactory
from pysp.utils.estimation import optimize
from pysp.utils.fit import fit_map, fit_mle
from pysp.utils.priors import DirichletPrior, MixturePrior, NormalGammaPrior

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

    data = dist.sampler(seed=rng.randint(2 ** 31)).sample(size=20000)

    est = MixtureEstimator([CompositeEstimator((
        GaussianEstimator(), GammaEstimator(),
        CategoricalEstimator(pseudo_count=0.5),
        OptionalEstimator(PoissonEstimator(), est_prob=True),
    ))] * 2)

    # Ordinary NumPy/vectorized sequence path
    t0 = time.perf_counter()
    mm = optimize(data, est, max_its=20, rng=np.random.RandomState(2), print_iter=20,
                  out=io.StringIO())
    t_legacy = time.perf_counter() - t0

    # numba kernel engine: compile once, then use the common Kernel API
    kernel = NumbaKernelFactory().build(mm, NUMPY_ENGINE, estimator=est)
    enc = kernel.encode(data)
    t0 = time.perf_counter()
    mm_k = mm
    for _ in range(20):
        kernel.refresh(mm_k)
        mm_k = est.estimate(len(data), kernel.accumulate(enc, np.ones(len(data))))
    t_kernel = time.perf_counter() - t0

    # torch engine: the same optimize() call, just with a different engine
    if torch is not None:
        torch_eng = TorchEngine(dtype='float64')
    else:
        torch_eng = None

    t0 = time.perf_counter()
    mm_t = optimize(data, est, max_its=20, delta=None, prev_estimate=mm,
                    engine=torch_eng, out=io.StringIO()) if torch_eng is not None else None
    t_torch = time.perf_counter() - t0

    enc_data = seq_encode(data, model=mm)
    rows = [('legacy', mm, t_legacy), ('numba', mm_k, t_kernel)]
    if mm_t is not None:
        rows.append(('torch', mm_t, t_torch))
    for name, m, t in rows:
        _, ll = seq_log_density_sum(enc_data, m)
        print('%-8s ll=%.6f   20 EM iterations in %.2fs' % (name, ll, t))

    # Gradient fitting lives in pysp.utils.estimation and uses the same engine object.
    small_truth = MixtureDistribution(
        [GaussianDistribution(-2.0, 0.6), GaussianDistribution(2.0, 0.9)],
        [0.45, 0.55],
    )
    small_start = MixtureDistribution(
        [GaussianDistribution(-1.0, 2.0), GaussianDistribution(1.0, 2.0)],
        [0.5, 0.5],
    )
    small = small_truth.sampler(seed=5).sample(size=250)
    small_enc = small_start.dist_to_encoder().seq_encode(small)
    mle, _ = fit_mle(small_enc, small_start, engine=torch_eng, max_its=120, lr=0.04,
                     print_iter=1000) if torch_eng is not None else (None, None)
    map_prior = MixturePrior(
        components=[
            NormalGammaPrior(mu0=-2.0, kappa=1.0, alpha=3.0, beta=2.0),
            NormalGammaPrior(mu0=2.0, kappa=1.0, alpha=3.0, beta=2.0),
        ],
        weights=DirichletPrior([2.0, 2.0]),
    )
    mapped, _ = fit_map(small_enc, small_start, engine=torch_eng, priors=map_prior,
                        prior_strength=0.0, max_its=120, lr=0.04,
                        print_iter=1000) if torch_eng is not None else (None, None)
    if mle is not None:
        print('gradient MLE weights:', np.round(mle.w, 3))
        print('typed-prior MAP weights:', np.round(mapped.w, 3))
