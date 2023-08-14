"""Functions for estimating and validating pysparkplug models from observed data.

Useful functions for estimating pysparkplug 'SequenceEncodableProbabilityDistributions' from 'ParameterEstimator'
objects.

"""
import numpy as np
from numpy.random import RandomState
import sys
import time

from pysp.stats import initialize, seq_estimate, seq_log_density_sum, seq_encode, seq_log_density, seq_initialize
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator

from typing import Any, Tuple, List, Union, TypeVar, Optional, IO, Sequence

T = TypeVar('T')
E0 = TypeVar('E0')


def empirical_kl_divergence(dist1: SequenceEncodableProbabilityDistribution,
                            dist2: SequenceEncodableProbabilityDistribution, enc_data: List[Tuple[int, Any]]
                            ) -> Tuple[float, float, float]:
    """Computes the emirical KL-divergence between two densities.

    Compute the KL-divergence between dist1 and dist2, for encoded sequence of data. Dists must both have the
    same encodings.

    Args:
        dist1 (SequenceEncodableProbabilityDistribution): Distribution compatible with enc_data.
        dist2 (SequenceEncodableProbabilityDistribution): Distribution compatible with enc_data.
        enc_data (List[Tuple[int, Any]]): List of Tuple containing chunk size and encoded sequence for chunked data.

    Returns:
        Tuple of KL-div estiamte, number of 'bad' likelihood values for dist1, 'bad' likelihood values for dist2.

    """

    ll = seq_log_density(enc_data, estimate=(dist1, dist2))
    ll = np.hstack(ll)

    l1 = ll[0, :]
    l2 = ll[1, :]
    g1 = np.bitwise_and(l1 != -np.inf, ~np.isnan(l1))
    g2 = np.bitwise_and(l2 != -np.inf, ~np.isnan(l2))
    gg = np.bitwise_and(g1, g2)

    max_l1 = np.max(l1[gg])
    max_l2 = np.max(l2[gg])

    p1 = np.exp(l1[gg] - max_l1)
    p1 /= p1.sum()

    p2 = np.exp(l2[gg] - max_l2)
    p2 /= p2.sum()

    r1 = (p1[gg] * (np.log(p1[gg]) - np.log(p2[gg]))).sum()
    r2 = (~g1).sum()
    r3 = (~g2).sum()

    return r1, r2, r3


def k_fold_split_index(sz: int, k: int, rng: RandomState) -> np.ndarray:
    """Returns integer numpy index vector for k-fold split. Entry j is the fold-id for the j^{th} data point.

    Args:
        sz (int): Integer length of data points in data set.
        k (int): Integer number of folds for k-folds.
        rng (RandomState): RandomState for setting seed.

    Returns:
        1-d np.ndarray[int] of indices for each data points fold-id.

    """
    idx = rng.rand(sz)
    sidx = np.argsort(idx)

    rv = np.zeros(sz, dtype=int)
    for i in range(k):
        rv[sidx[np.arange(start=i, stop=sz, step=k, dtype=int)]] = i

    return rv


def partition_data_index(sz: int, pvec: Union[List[float], np.ndarray], rng: RandomState) -> List[np.ndarray]:
    """Returns List of np.ndarray[int] containing integers indexes for data partitions proportional to pvec.

    Args:
        sz (int): Integer value of total number of data observations.
        pvec (Union[List[float], np.ndarray]): Vector of proportions for each partition.
        rng (RandomState): RandomState for setting seed of random partitioning.

    Returns:
        List of numpy arrays containing indexes of each partition.

    """
    idx = rng.rand(sz)
    sidx = np.argsort(idx)

    rv = []
    p_tot = 0
    prev_idx = 0

    for p in pvec:
        next_idx = int(round(sz * (p_tot + p), 0))
        rv.append(sidx[prev_idx:next_idx])
        p_tot += p
        prev_idx = next_idx

    return rv


def partition_data(data: Sequence[T], pvec: Union[List[float], np.ndarray], rng: RandomState) -> List[List[T]]:
    """Partitions List of data into partitions, each with size equal to the proportion of pvec.

    Args:

        data (Sequence[T]): Sequence of data observations, each entry of type T.
        pvec (Union[List[float], np.ndarray]): List of length n, containing proportion of data to be held in each data
            partition.
        rng (RandomState): RandomState for setting seed on random partitioning of data.

    Returns:
        List of List containing data partitions of proportion equal to pvec.

    """
    idx_list = partition_data_index(len(data), pvec, rng)

    return [[data[i] for i in u] for u in idx_list]


def best_of(data: Optional[Sequence[T]], vdata: Optional[Sequence[T]], est: ParameterEstimator, trials: int,
            max_its: int, init_p: float, delta: float, rng: RandomState,
            init_estimator: Optional[ParameterEstimator] = None,
            enc_data: Optional[List[Tuple[int, E0]]] = None,
            enc_vdata: Optional[Sequence[Tuple[int, E0]]] = None,
            out: IO = sys.stdout, print_iter: int = 1) -> Tuple[float, SequenceEncodableProbabilityDistribution]:
    """Performs EM algorithm for trials-number of randomized initial conditions. Returns the best model fit in terms of
        maximum log-likelihood value from validation data.

    Args:
        data (Optional[List[T]]): List of data of type T. If None is given, enc_data must be provided as
            List[Tuple[int, enc_data_type]].
        vdata (Optional[Sequence[T]]): Optional validation set.
        est (ParameterEstimator): ParameterEstimator for model to be estimated.
        trials (int): Integer number >= 1, of randomized initial conditions to perform EM algorithm for.
        max_its (int): Integer value >=1, sets the maximum number of iterations of EM to be performed as stopping criteria.
        init_p (float): Value in (0.0,1.0] for randomizing the proportion of data points used in initialization.
        delta (float): Stopping criteria for EM when |old-log-likelihood - new-log-likelihood| < delta.
        rng (RandomState): RandomState for setting seed.
        init_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator used for fitting.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data, if provided data need not be
            provided. If None, enc_data is set from data.
        enc_vdata (Optional[List[Tuple[int, E0]]]): Optional sequence encoded validation set.
        out (I0): Text output stream.
        print_iter (int): Print iterations (i.e. log-likelihood difference) every print_iter-iterations.

    Returns:
        Tuple of log-likelihood of best fitting model and the best fitting model from number of trials.

    """
    rv_ll = -np.inf
    rv_mm = None
    i_est = est if init_estimator is None else init_estimator

    if data is None and enc_data is None:
        raise Exception('Optimization called with empty data or enc_data.')

    if max_its < 1:
        max_its = 1

    if trials < 1:
        trials = 1

    for kk in range(trials):

        if enc_data is None:
            encoder = i_est.accumulator_factory().make().acc_to_encoder()
            enc_data = seq_encode(data, encoder)

            if enc_vdata is None and vdata is not None:
                enc_vdata = seq_encode(vdata, encoder)

        mm = seq_initialize(enc_data, i_est, rng, init_p)
        _, old_ll = seq_log_density_sum(enc_data, mm)

        for i in range(max_its):

            mm_next = seq_estimate(enc_data, est, mm)
            _, ll = seq_log_density_sum(enc_data, mm_next)
            dll = ll - old_ll

            if (i + 1) % print_iter == 0:
                out.write('Iteration %d. LL=%f, delta LL=%e\n' % (i + 1, ll, dll))

            if (dll >= 0) or (delta is None):
                mm = mm_next

            if (delta is not None) and (dll < delta):
                break

            old_ll = ll

        _, vll = seq_log_density_sum(enc_vdata, mm)
        out.write('Trial %d. VLL=%f\n' % (kk + 1, vll))

        if vll > rv_ll:
            rv_mm = mm
            rv_ll = vll

    return rv_ll, rv_mm


def optimize(data: Optional[Sequence[T]], estimator: ParameterEstimator, max_its: int = 10,
             delta: Optional[float] = 1.0e-9,
             init_estimator: Optional[ParameterEstimator] = None, init_p: float = 0.1,
             rng: RandomState = RandomState(), prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None,
             vdata: Optional[Sequence[T]] = None,
             enc_data: Optional[List[Tuple[int, E0]]] = None,
             enc_vdata: Optional[List[Tuple[int, E0]]] = None,
             out: IO = sys.stdout,
             print_iter: int = 1, num_chunks: int = 1) -> SequenceEncodableProbabilityDistribution:
    """Estimation of 'estimator' via EM algorithm for max_its iterations or until
        new_loglikelihood - old_loglikelihood < delta.

    Args:
        data (Optional[List[T]]): List of data type T containing observed data. Must be compatible with data type of
            estimator.
        estimator (ParameterEstimator): ParameterEstimator used to specify to-be-estimated distribution for observed
            data.
        max_its (int): Maximum number of EM iterations to be performed. Default value is 10 iterations.
        delta (Optional[float]): Stopping criteria for EM algorithm used if max_its is not set: Iterate until
            |old_loglikelihood - new_loglikelihood| < delta or iterations == max_its.
        init_estimator (Optional[ParameterEstimator]): ParameterEstimator to used to initialize EM algorithm parameters.
            If None, estimator is used. Must be consistent with estimator.
        init_p (float): Value in (0.0,1.0] for randomizing the proportion of data points used in initialization.
        rng (RandomState): RandomState used to set seed for initializing EM algorithm.
        vdata (Optional[Sequence[T]]): Optional validation set.
        prev_estimate (Optional[SeqeuenceEncodableProbabilityDistribution]): Optional model estimate used from prior
            fitting. Must be consistent with estimator.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data of form
            List[Tuple[int, E]]. Formed from data if None.
        enc_vdata (Optional[List[Tuple[int, E0]]]): Optional sequence encoded validation set.
        out (IO): IO stream to write out iterations of EM algorithm.
        print_iter (int): Print iterations (i.e. log-likelihood difference) every print_iter-iterations.
        num_chunks (int): Number of chunks for encoded data.

    Returns:
        SequenceEncodableProbabilityDistribution corresponding to estimator when stopping criteria of EM algorithm
            is met.

    """
    if data is None and enc_data is None:
        raise Exception('Optimization called with empty data or enc_data.')

    est = estimator if init_estimator is None else init_estimator

    if prev_estimate is None:
        data_encoder = est.accumulator_factory().make().acc_to_encoder()
    else:
        data_encoder = prev_estimate.dist_to_encoder()

    if enc_data is None:
        enc_data = seq_encode(data=data, encoder=data_encoder, num_chunks=num_chunks)

    if prev_estimate is None:
        if init_p <= 0.0:
            p = 0.10
        else:
            p = min(max(init_p, 0.0), 1.0)

        mm = seq_initialize(enc_data=enc_data, estimator=est, rng=rng, p=p)
    else:
        mm = prev_estimate

    _, old_ll = seq_log_density_sum(enc_data=enc_data, estimate=mm)

    if enc_vdata is None and vdata is not None:
        enc_vdata = seq_encode(vdata, data_encoder, num_chunks=num_chunks)

    if enc_vdata is not None:
        _, old_vll = seq_log_density_sum(enc_vdata, mm)
    else:
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):

        mm_next = seq_estimate(enc_data=enc_data, estimator=estimator, prev_estimate=mm)
        cnt, ll = seq_log_density_sum(enc_data=enc_data, estimate=mm_next)

        if enc_vdata is not None:
            _, vll = seq_log_density_sum(enc_vdata, mm_next)
        else:
            vll = ll

        dll = ll - old_ll

        if (dll >= 0) or (delta is None):
            mm = mm_next

        if (delta is not None) and (dll < delta):
            if enc_vdata is not None:
                out.write(
                    'Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, '
                    'ln[p_mat(Valid Data|Model)]=%e\n' % (
                    i + 1, ll, dll, vll))
            else:
                out.write('Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n' %
                          (i + 1, ll, dll))
            break

        if (i + 1) % print_iter == 0:
            if enc_vdata is not None:
                out.write('Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, '
                          'ln[p_mat(Valid Data|Model)]=%e\n' % (i + 1, ll, dll, vll))
            else:
                out.write('Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n' %
                          (i + 1, ll, dll))

        old_ll = ll

        if best_vll < vll:
            best_vll = vll
            best_model = mm

    return best_model


def iterate(data: List[T], estimator: Optional[ParameterEstimator], max_its: int,
            prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None, init_p: float = 0.1,
            rng: Optional[RandomState] = RandomState(), out: IO = sys.stdout,
            enc_data: Optional[List[Tuple[int, E0]]] = None,
            init_estimator: Optional[ParameterEstimator] = None,
            print_iter: int = 1) -> SequenceEncodableProbabilityDistribution:
    """Performs max_its-iterations of EM algorithm and returns next estimate (SequenceEncodableProbabilityDistribution).

    Args:
        data (List[T]): List of data type compatible with estimator.
        estimator (Optional[ParameterEstimator]): Optional ParameterEstimator for distribution to be estimated from
            data by EM algorithm. Can be None only if init_estimator is not None.
        max_its (int): Total number of EM iterations to be performed before returning estimate.
        prev_estimate (Optional[SequenceEncodableProbabilityDistribution]): Optional previous estimate of distribution
            for data. Must be consistent with estimator or init_estimator.
        init_p (float): Value in (0.0,1.0] for randomizing the proportion of data points used in initialization.
        rng (Optional[RandomState]): RandomState used to set seed for initializing EM algorithm.
        out (IO): IO stream to write out iterations of EM algorithm.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data of form
            List[Tuple[int, E]]. Formed from data if None.
        init_estimator (Optional[ParameterEstimator]): ParameterEstimator to used to initialize EM algorithm parameters.
            If None, estimator is used. Must be consistent with estimator.
        print_iter (bool): Print iterations (i.e. log-likelihood) ever print_iter-iterations.

    Returns:
        SequenceEncodableProbabilityDistribution corresponding to estimator/init_estimator after max_its iterations of
            EM algorithm.

    """
    if data is None and enc_data is None:
        raise Exception('Optimization called with empty data or enc_data.')

    i_est = estimator if init_estimator is None else init_estimator

    if enc_data is None:
        encoder = estimator.accumulator_factory().make().acc_to_encoder()
        enc_data = seq_encode(data, encoder)

    if prev_estimate is None:
        if init_p <= 0.0:
            p = 0.1
        else:
            p = min(max(init_p, 0.0), 1.0)

        mm = seq_initialize(enc_data, i_est, rng, init_p)
    else:
        mm = prev_estimate

    if hasattr(enc_data, 'cache'):
        enc_data.cache()

    t0 = time.time()
    for i in range(max_its):
        mm = seq_estimate(enc_data, estimator, mm)

        if (i + 1) % print_iter == 0:
            out.write('Iteration %d\t E[dT]=%f.\n' % (i + 1, (time.time() - t0) / float(i + 1)))

    return mm

