"""Scaling a fit across backends: the same ``optimize`` call, distributed by ``backend=``.

mixle separates *what* you fit from *where* it runs. You write one model + estimator and the EM
fit distributes over the chosen backend with an identical result -- only the data placement changes.

This script runs the two backends that need no external cluster:

  * ``backend='local'`` -- single process (the default),
  * ``backend='mp'``    -- multiprocessing across worker processes on this machine,

and prints both fits so you can see they agree. The cluster backends are identical in spirit but need a
launcher, so they are documented (not run) at the bottom: ``'mpi'`` (mpi4py) and Spark RDDs.

We fit a CompositeDistribution (a Gaussian + a Categorical + a Poisson per record) -- a realistic tabular
record whose MLE is closed-form, so the fit is deterministic and local/mp recover the same parameters --
to within floating-point reassociation, since a data-parallel backend adds the per-shard sufficient
statistics in a different order than the single-process fold. Agreement, not bit-equality.
"""

import math

from mixle.inference import optimize
from mixle.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    PoissonDistribution,
    PoissonEstimator,
)


def _fit(data, backend, **kw):
    est = CompositeEstimator((GaussianEstimator(), CategoricalEstimator(), PoissonEstimator()))
    model = optimize(data, est, max_its=1, out=None, backend=backend, **kw)
    g, c, p = model.dists
    return g.mu, c.pmap["a"], p.lam


if __name__ == "__main__":
    truth = CompositeDistribution(
        (
            GaussianDistribution(2.0, 1.5),
            CategoricalDistribution({"a": 0.6, "b": 0.3, "c": 0.1}),
            PoissonDistribution(4.0),
        )
    )
    data = truth.sampler(1).sample(50000)
    print("true              : mu=2.00  P(a)=0.60  lam=4.00")

    local = _fit(data, "local")
    print("backend='local'   : mu=%.2f  P(a)=%.2f  lam=%.2f" % local)

    multiprocessing = _fit(data, "mp", num_workers=4)
    print("backend='mp' (x4) : mu=%.2f  P(a)=%.2f  lam=%.2f" % multiprocessing)
    # Agreement, not bit-equality. A data-parallel backend shards the rows across workers and adds the
    # per-shard sufficient statistics back together, so the summation ORDER differs from the
    # single-process fold. Floating-point addition is not associative, so the two agree to within
    # reassociation error -- here the last ulp of a ~2.0 mean -- and the library documents exactly that
    # (mixle.utils.parallel.model_parallel: "exact up to float reassociation like every data-parallel
    # backend"). Asserting == demanded a guarantee no data-parallel reduce can give, and this example
    # existed to show the backends agree, which they do.
    assert all(math.isclose(m, ell, rel_tol=1e-9, abs_tol=1e-12) for m, ell in zip(multiprocessing, local)), (
        local,
        multiprocessing,
    )

    # --- cluster backends (same optimize() call, external launcher) ---------------------------------
    # MPI -- run on N ranks; the model is the same on every rank, optimize() runs in lockstep:
    #     mpiexec -n 4 python -c "import scaling_example"   # with backend='mpi'
    #   optimize(data, est, backend='mpi', comm=MPI.COMM_WORLD, root=0)
    #
    # Spark -- data lives in an RDD partitioned across the cluster:
    #     optimize(None, est, enc_data=<encoded RDD>, backend='spark')
    #   (see mixle.data.sources.spark_source for building the encoded RDD from a SparkContext).
