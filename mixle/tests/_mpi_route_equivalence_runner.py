"""Runner: fit the SAME model+data through BOTH MPI routes and print both results (worklist D8.5).

Launched as ``mpirun -n W python _mpi_route_equivalence_runner.py``. mixle has two MPI EM transports:

  * ``mixle.inference.mpi_executor.mpi_fit`` -- a standalone ``comm.reduce`` tree-fold (each rank passes the
    full data and slices its own contiguous shard internally);
  * ``mixle.utils.parallel.mpi.MPIEncodedData`` -- the integrated ``optimize`` backend, gather-to-root (each
    rank holds its own shard).

They run the same sharded-E-step + associative-``combine`` EM, so fed the same per-rank shards from the same
initial model they must converge to the SAME parameters. This runs both and prints them for the test to
compare (rank 0). Underscore-prefixed so pytest does not collect it.
"""

import json

import numpy as np
from mpi4py import MPI

import mixle.stats as st
from mixle.inference.estimation import optimize
from mixle.inference.mpi_executor import mpi_fit
from mixle.utils.parallel.mpi import MPIEncodedData, mpi_out


def _model_and_data():
    rng = np.random.RandomState(0)
    comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(3)]
    m = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(3))))
    return m, m.sampler(1).sample(4000)


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    model, data = _model_and_data()

    # Both routes fold sufficient statistics over ALL the data every EM step -- route A slices contiguous
    # shards, route B round-robin shards (MPIEncodedData shards its full input internally). Summation is
    # partition-invariant, so the total statistics, the M-step, and the fit are the same up to
    # floating-point summation order; delta=None on route B forces the full iteration budget so both take the
    # same number of EM steps from the same initial model.
    its = 30

    # Route A: tree-fold reduce transport (mpi_executor). Each rank passes full data; slices its own shard.
    fit_a = mpi_fit(comm, model, data, max_its=its)

    # Route B: gather-to-root optimize backend (MPIEncodedData). Pass the SAME full data (it shards
    # internally), the same initial model, and the same iteration count.
    est = model.estimator()
    enc = MPIEncodedData(data, estimator=est)
    fit_b = optimize(None, est, enc_data=enc, prev_estimate=model, max_its=its, delta=None, out=mpi_out())

    if rank == 0:
        payload = {
            "size": size,
            "w_a": sorted(float(x) for x in fit_a.w),
            "mu_a": sorted(float(c.mu) for c in fit_a.components),
            "w_b": sorted(float(x) for x in fit_b.w),
            "mu_b": sorted(float(c.mu) for c in fit_b.components),
        }
        print("RESULT " + json.dumps(payload))


if __name__ == "__main__":
    main()
