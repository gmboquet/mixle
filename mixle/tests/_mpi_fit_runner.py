"""Runner for the MPI distributed-EM test -- launched as ``mpirun -n W python _mpi_fit_runner.py``.

Every rank builds the identical model + data (fixed seed), runs the MPI EM fit (each rank scoring its
contiguous shard, folded by comm.reduce), and rank 0 prints the fitted parameters as JSON for the test to
compare against the serial baseline. Underscore-prefixed so pytest does not collect it.
"""

import json

import numpy as np
from mpi4py import MPI

import mixle.stats as st
from mixle.inference.mpi_executor import mpi_fit


def _model_and_data():
    rng = np.random.RandomState(0)
    comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(3)]
    m = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(3))))
    return m, m.sampler(1).sample(4000)


def main():
    comm = MPI.COMM_WORLD
    m, data = _model_and_data()
    fit = mpi_fit(comm, m, data, max_its=12)
    if comm.Get_rank() == 0:
        payload = {
            "size": comm.Get_size(),
            "w": sorted(float(x) for x in fit.w),
            "mu": sorted(float(c.mu) for c in fit.components),
        }
        print("RESULT " + json.dumps(payload))


if __name__ == "__main__":
    main()
