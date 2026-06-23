"""Simulate data and estimate a Mixture distribution with the mpi4py backend.

Twin of examples/examples_pysp/mixture_example.py for MPI. Run with:

    mpiexec -n 4 python examples/examples_mpi/mixture_example.py

Every rank runs this same script (SPMD): each rank samples the identical
dataset (same seed), keeps its own shard data[rank::size] encoded and
resident, and optimize() runs in lockstep on all ranks - the model is
re-estimated on rank 0 each iteration and broadcast, so every rank finishes
with the identical fit. mpi_out() routes iteration logging to rank 0 only.
"""

import numpy as np

from pysp.stats import *
from pysp.inference import optimize
from pysp.utils.evaluation import partition_data
from pysp.utils.parallel.mpi import MPIEncodedData, mpi_out
from mpi4py import MPI

if __name__ == '__main__':
    rng = np.random.RandomState(1)

    # Create the example distribution (identical on every rank - same seed)

    d10 = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0)], [0.5, 0.5])
    d11 = OptionalDistribution(CategoricalDistribution({'a': 0.5, 'b': 0.4, 'c': 0.1}), p=0.1)
    d12 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.2, 'b': 0.8}, 'b': {'a': 0.8, 'b': 0.2}},
                                  len_dist=PoissonDistribution(8.0))
    d13 = BernoulliSetDistribution({'a': 0.9, 'b': 0.8})
    d14 = MultivariateGaussianDistribution([-1.0, -1.0], [[2.0, 1.0], [1.0, 2.0]])

    d1 = CompositeDistribution([d10, d11, d12, d13, d14])

    d20 = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(6.0, 1.0)], [0.5, 0.5])
    d21 = OptionalDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.1, 'c': 0.8}), p=0.2)
    d22 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.8, 'b': 0.2}, 'b': {'a': 0.2, 'b': 0.8}},
                                  len_dist=PoissonDistribution(8.0))
    d23 = BernoulliSetDistribution({'a': 0.9, 'b': 0.8})
    d24 = MultivariateGaussianDistribution([1.0, 1.0], [[2.0, 1.0], [1.0, 2.0]])

    d2 = CompositeDistribution([d20, d21, d22, d23, d24])

    dist = MixtureDistribution([d1, d2], [0.5, 0.5])

    # Sample the SAME data on every rank (fixed seed); each rank keeps only
    # its shard. For data on the root rank only, pass root_only=True.

    sampler = dist.sampler(seed=rng.randint(2 ** 31))
    data = sampler.sample(size=2000)

    train_data, valid_data = partition_data(data, [0.9, 0.1], rng)

    # Specify the model estimator (regularized for initialization, plain for EM)

    e0 = MixtureEstimator([GaussianEstimator()] * 2, pseudo_count=1.0)
    e1 = OptionalEstimator(CategoricalEstimator(pseudo_count=1.0), est_prob=False, pseudo_count=1.0)
    e2 = MarkovChainEstimator(pseudo_count=1.0, len_estimator=PoissonEstimator())
    e3 = BernoulliSetEstimator(keys='asdf')
    e4 = MultivariateGaussianEstimator()
    iest = MixtureEstimator([CompositeEstimator((e0, e1, e2, e3, e4))] * 2, pseudo_count=1.0, fixed_weights=[0.2, 0.8])

    e0 = MixtureEstimator([GaussianEstimator()] * 2)
    e1 = OptionalEstimator(CategoricalEstimator(), est_prob=False)
    e2 = MarkovChainEstimator(len_estimator=PoissonEstimator())
    e3 = BernoulliSetEstimator(keys='asdf')
    e4 = MultivariateGaussianEstimator()
    est = MixtureEstimator([CompositeEstimator((e0, e1, e2, e3, e4))] * 2, fixed_weights=[0.2, 0.8])

    # Compare to the local method:
    #
    #   mm = optimize(train_data, est, max_its=100, ...)

    enc_train = MPIEncodedData(train_data, estimator=est)
    mm = optimize(None, est, enc_data=enc_train, vdata=valid_data, max_its=100,
                  print_iter=20, init_estimator=iest, rng=np.random.RandomState(1),
                  out=mpi_out())

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(str(list(mm.w)))
        print(str(mm.components[0]))
        print(str(mm.components[1]))
