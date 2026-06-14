"""Functions for estimating and validating pysparkplug models from observed data.

Useful functions for estimating pysparkplug 'SequenceEncodableProbabilityDistributions' from 'ParameterEstimator'
objects.

"""
import copy
from dataclasses import dataclass

import numpy as np
from numpy.random import RandomState
import sys
import time

from pysp.stats import initialize, seq_estimate, seq_log_density_sum, seq_encode, seq_log_density, seq_initialize, \
    validate_estimator_keys
from pysp.stats.gradient import GradientFitError
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator
from pysp.utils.priors import as_prior_dict

from typing import Any, Tuple, List, Union, TypeVar, Optional, IO, Sequence, Mapping

T = TypeVar('T')
E0 = TypeVar('E0')


@dataclass
class GradientFitResult:
    """Optimization result for generic autograd MLE/MAP fitting."""

    model: SequenceEncodableProbabilityDistribution
    value: float
    iterations: int
    history: Tuple[float, ...] = ()
    converged: bool = False
    initial_value: Optional[float] = None
    final_delta: Optional[float] = None
    log_likelihood: Optional[float] = None
    log_prior: Optional[float] = None
    prior_strength: float = 0.0
    tag: str = 'MLE'
    best_value: Optional[float] = None
    best_iteration: Optional[int] = None
    final_gradient_norm: Optional[float] = None

    def as_tuple(self) -> Tuple[SequenceEncodableProbabilityDistribution, float]:
        """Return the historical ``(model, objective)`` shape."""
        return self.model, self.value

    @property
    def objective_change(self) -> Optional[float]:
        """Return the signed objective change from the start of optimization."""
        if self.initial_value is None:
            return None
        return self.value - self.initial_value

    @property
    def improvement(self) -> Optional[float]:
        """Return the maximization improvement from the start objective."""
        return self.objective_change

    @property
    def best_improvement(self) -> Optional[float]:
        """Return best improvement seen during optimization."""
        if self.initial_value is None or self.best_value is None:
            return None
        return self.best_value - self.initial_value

    @property
    def prior_sensitivity(self) -> Optional[float]:
        """Return the magnitude fraction of the final objective coming from the prior."""
        if self.log_likelihood is None or self.log_prior is None:
            return None
        likelihood = abs(float(self.log_likelihood))
        prior = abs(float(self.log_prior))
        total = likelihood + prior
        return 0.0 if total == 0.0 else prior / total


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

    r1 = (p1 * (np.log(p1) - np.log(p2))).sum()
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
    if data is None and enc_data is None:
        raise Exception('Optimization called with empty data or enc_data.')

    max_its = max(1, max_its)
    trials = max(1, trials)
    i_est = est if init_estimator is None else init_estimator

    # encode once and reuse across trials (each trial re-initializes from rng)
    if enc_data is None:
        encoder = _resolve_encoder(i_est)
        enc_data = seq_encode(data, encoder)
        if enc_vdata is None and vdata is not None:
            enc_vdata = seq_encode(vdata, encoder)
    elif enc_vdata is None and vdata is not None:
        enc_vdata = seq_encode(vdata, _resolve_encoder(i_est))
    score_data = enc_data if enc_vdata is None else enc_vdata

    rv_ll, rv_mm = -np.inf, None
    for kk in range(trials):
        mm = optimize(None, est, init_estimator=i_est, enc_data=enc_data, enc_vdata=enc_vdata,
                      max_its=max_its, delta=delta, init_p=init_p, rng=rng, out=out,
                      print_iter=print_iter)
        _, vll = seq_log_density_sum(score_data, mm)
        out.write('Trial %d. VLL=%f\n' % (kk + 1, vll))
        if vll > rv_ll:
            rv_ll, rv_mm = vll, mm

    return rv_ll, rv_mm


def _local_encoded_chunks(enc_data: Any) -> List[Tuple[int, Any]]:
    if hasattr(enc_data, 'as_seq_chunk'):
        return [enc_data.as_seq_chunk()]
    if isinstance(enc_data, tuple) and len(enc_data) == 2 and isinstance(enc_data[0], (int, np.integer, float)):
        return [enc_data]
    if isinstance(enc_data, list):
        return enc_data
    raise ValueError('engine-aware optimize currently supports local encoded chunks only; '
                     'distributed engine orchestration is handled by a later planner slice.')


def _engine_seq_log_density_sum(enc_data: Any,
                                estimate: SequenceEncodableProbabilityDistribution,
                                engine: Any) -> Tuple[float, float]:
    chunks = _local_encoded_chunks(enc_data)
    kernel = estimate.kernel(engine=engine)
    nobs = 0.0
    ll = 0.0
    for sz, enc in chunks:
        nobs += sz
        ll += float(np.asarray(engine.to_numpy(kernel.score(enc)), dtype=np.float64).sum())
    return nobs, ll


def _engine_seq_estimate(enc_data: Any,
                         estimator: ParameterEstimator,
                         prev_estimate: SequenceEncodableProbabilityDistribution,
                         engine: Any) -> SequenceEncodableProbabilityDistribution:
    validate_estimator_keys(estimator)
    chunks = _local_encoded_chunks(enc_data)
    kernel = prev_estimate.kernel(engine=engine, estimator=estimator)
    accumulator = estimator.accumulator_factory().make()
    nobs = 0.0
    for sz, enc in chunks:
        nobs += sz
        accumulator.combine(kernel.accumulate(enc, np.ones(sz, dtype=np.float64)))
    return estimator.estimate(nobs, accumulator.value())


def _dataframe_like(data: Any) -> bool:
    return hasattr(data, 'columns') and hasattr(data, 'loc')


def _recordish(obj: Any) -> bool:
    return obj is not None and hasattr(obj, 'fields') and hasattr(obj, 'sources')


def _dataframe_fields(fields: Any, estimator: Any, model: Any) -> Any:
    if fields is not None:
        return fields
    for obj in (model, estimator):
        if _recordish(obj):
            return tuple(zip(getattr(obj, 'fields'), getattr(obj, 'sources')))
    return None


def _data_records_for_encoding(data: Any, fields: Any, estimator: Any, model: Any) -> Any:
    if not _dataframe_like(data) and fields is None:
        return data
    from pysp.stats.dataframe import dataframe_records
    record_fields = _dataframe_fields(fields, estimator, model)
    return dataframe_records(data, fields=record_fields, as_dict=_recordish(model) or _recordish(estimator))


# --- shared EM driver ------------------------------------------------------
#
# optimize/best_of/iterate (and em.run_em) all share the same skeleton: build an
# encoder, encode the data, initialize (or reuse) a model, then iterate an E/M
# step until convergence. The helpers below factor out that skeleton so each
# entry point is a thin policy wrapper over one tested loop.

def _resolve_encoder(estimator: ParameterEstimator,
                     prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None) -> Any:
    """Return the data encoder for a fitting run (model encoder if continuing)."""
    if prev_estimate is not None:
        return prev_estimate.dist_to_encoder()
    return estimator.accumulator_factory().make().acc_to_encoder()


def _ll_sum_fn(engine: Optional[Any]):
    """Return a (enc, model) -> (count, log_likelihood) scorer for the engine."""
    if engine is None:
        return seq_log_density_sum
    return lambda enc, model: _engine_seq_log_density_sum(enc, model, engine)


def _em_step_fn(engine: Optional[Any], strategy: Optional[Any] = None):
    """Return the per-iteration (enc, estimator, model) -> model update.

    With ``strategy`` set, the update is delegated to an EM strategy object
    (``pysp.utils.em``) or any callable, which is how alternative E-steps
    (annealed, hard, Monte-Carlo, ...) plug into ``optimize`` without a circular
    import. Otherwise the standard exact E/M step is used (engine-aware).
    """
    if strategy is not None:
        step_method = getattr(strategy, 'step', None)
        if callable(step_method):
            def step(enc, estimator, model):
                result = step_method(enc, estimator, model, engine=engine)
                return getattr(result, 'model', result)
            return step
        if callable(strategy):
            return lambda enc, estimator, model: strategy(enc, estimator, model)
        raise TypeError('strategy must be an EM strategy with .step(...) or a callable '
                        '(enc, estimator, model) -> model.')
    if engine is None:
        return seq_estimate
    return lambda enc, estimator, model: _engine_seq_estimate(enc, estimator, model, engine)


def _local_fused_step(enc_data, estimator, model):
    """Local E/M step that also returns the data log-likelihood of ``model``.

    Runs the standard local accumulation pass and, when the top-level accumulator records the
    data log-likelihood during its E-step (the posterior normalizer, e.g. for mixtures), returns
    it so the caller can skip a separate convergence-LL pass. Returns
    ``(next_model, ll_of_model_or_None)``; ``None`` means the model can't report it and the caller
    should score ``model`` itself. Local (non-RDD, non-parallel-handle) encoded data only.
    """
    accumulator = estimator.accumulator_factory().make()
    accumulator._track_ll = True  # ask the accumulator to record the E-step data log-likelihood
    for sz, x in enc_data:
        accumulator.seq_update(x, np.ones(sz), model)
    stats_dict = dict()
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    nxt = estimator.estimate(None, accumulator.value())
    # Present only when the top-level accumulator recorded it (e.g. mixtures); else None -> fallback.
    return nxt, getattr(accumulator, '_seq_ll', None)


def _write_em_iter(out: Optional[IO], i: int, ll: float, dll: float, vll: float, has_vdata: bool) -> None:
    """Write one EM progress line in the historical format."""
    if out is None:
        return
    if has_vdata:
        out.write('Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, '
                  'ln[p_mat(Valid Data|Model)]=%e\n' % (i, ll, dll, vll))
    else:
        out.write('Iteration %d: ln[p_mat(Data|Model)]=%e, '
                  'ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n' % (i, ll, dll))


def _em_loop(enc_data: Any, estimator: ParameterEstimator,
             model: SequenceEncodableProbabilityDistribution,
             step_fn: Any, ll_fn: Any, max_its: int, delta: Optional[float],
             enc_vdata: Optional[Any] = None, out: Optional[IO] = sys.stdout,
             print_iter: int = 1, monotone: bool = True,
             track_best: bool = True,
             fused_step_fn: Optional[Any] = None) -> Tuple[SequenceEncodableProbabilityDistribution, float]:
    """Canonical EM iteration shared by the public estimation entry points.

    Args:
        step_fn: ``(enc, estimator, model) -> model`` E/M (or strategy) update.
        ll_fn: ``(enc, model) -> (count, log_likelihood)`` convergence objective.
        delta: stop when the training log-likelihood gain drops below this;
            ``None`` runs the full ``max_its`` iterations.
        enc_vdata: optional encoded validation set used for best-model tracking.
        monotone: when True only accept a step that does not decrease the
            training log-likelihood (the historical ``optimize`` guard).
        track_best: when True return the best-by-validation model seen; otherwise
            the final accepted model.
        fused_step_fn: optional ``(enc, estimator, model) -> (next_model, ll_of_model)``
            update that returns the data log-likelihood of ``model`` as a byproduct of
            the E-step (the posterior normalizer), avoiding a separate convergence-LL
            pass. ``ll_of_model`` may be ``None`` when the model can't report it, in
            which case this falls back to scoring ``model`` directly. See
            :func:`_fused_em_loop`.

    Returns:
        ``(chosen_model, best_validation_score)``.
    """
    if fused_step_fn is not None:
        return _fused_em_loop(enc_data, estimator, model, fused_step_fn, ll_fn, max_its,
                              delta, enc_vdata, out, print_iter, track_best)

    _, old_ll = ll_fn(enc_data, model)
    has_v = enc_vdata is not None
    best_vll = ll_fn(enc_vdata, model)[1] if has_v else old_ll
    best_model = model

    for i in range(int(max_its)):
        nxt = step_fn(enc_data, estimator, model)
        _, ll = ll_fn(enc_data, nxt)
        vll = ll_fn(enc_vdata, nxt)[1] if has_v else ll
        dll = ll - old_ll

        if (dll >= 0) or (delta is None) or (not monotone):
            model = nxt

        converged = (delta is not None) and (dll < delta)
        if converged or ((i + 1) % print_iter == 0):
            _write_em_iter(out, i + 1, ll, dll, vll, has_v)
        if converged:
            break

        old_ll = ll
        if track_best and best_vll < vll:
            best_vll = vll
            best_model = model

    return (best_model if track_best else model), best_vll


def _fused_em_loop(enc_data, estimator, model, fused_step_fn, ll_fn, max_its, delta,
                   enc_vdata, out, print_iter, track_best):
    """EM loop that reuses the E-step's likelihood normalizer instead of a separate score pass.

    Each ``fused_step_fn`` call returns ``(next_model, ll_of_model)`` where ``ll_of_model`` is the
    data log-likelihood of the *input* model, computed for free as the posterior normalizer during
    the E-step. The convergence test therefore lags the standard loop by one iteration (it compares
    the likelihood of successive accepted models), which converges to the same fixed point; the
    returned model is still the best-likelihood model seen (so quality is preserved even though
    intermediate steps are accepted unconditionally). When ``ll_of_model`` is ``None`` the model
    cannot report it and we fall back to scoring ``model`` directly for that iteration.
    """
    has_v = enc_vdata is not None
    best_model = model
    best_score = ll_fn(enc_vdata, model)[1] if has_v else None
    prev_ll = None
    nxt = None
    converged = False

    for i in range(int(max_its)):
        nxt, ll_model = fused_step_fn(enc_data, estimator, model)
        if ll_model is None:
            _, ll_model = ll_fn(enc_data, model)
        score = ll_fn(enc_vdata, model)[1] if has_v else ll_model

        if best_score is None or score >= best_score:
            best_score = score
            best_model = model

        dll = (ll_model - prev_ll) if prev_ll is not None else float('inf')
        converged = (delta is not None) and (prev_ll is not None) and (dll < delta)
        if converged or ((i + 1) % print_iter == 0):
            _write_em_iter(out, i + 1, ll_model, dll, score, has_v)
        if converged:
            break

        prev_ll = ll_model
        model = nxt

    if not converged and nxt is not None:
        # Loop ran to max_its: fold the final step into best-model tracking (one extra score pass).
        score = ll_fn(enc_vdata, nxt)[1] if has_v else ll_fn(enc_data, nxt)[1]
        if best_score is None or score >= best_score:
            best_score = score
            best_model = nxt

    chosen = best_model if track_best else (nxt if nxt is not None else model)
    return chosen, (best_score if best_score is not None else 0.0)


def optimize(data: Optional[Sequence[T]], estimator: ParameterEstimator, max_its: int = 10,
             delta: Optional[float] = 1.0e-9,
             init_estimator: Optional[ParameterEstimator] = None, init_p: float = 0.1,
             rng: RandomState = RandomState(), prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None,
             vdata: Optional[Sequence[T]] = None,
             enc_data: Optional[List[Tuple[int, E0]]] = None,
             enc_vdata: Optional[List[Tuple[int, E0]]] = None,
             out: IO = sys.stdout,
             print_iter: int = 1, num_chunks: int = 1,
             engine: Optional[Any] = None,
             precision: Optional[Any] = None,
             fields: Optional[Any] = None,
             resources: Optional[Any] = None,
             placement: Optional[Any] = None,
             sub_chunks: int = 1,
             chunk_size: Optional[int] = None,
             backend: str = 'local',
             num_workers: Optional[int] = None,
             client: Optional[Any] = None,
             comm: Optional[Any] = None,
             root: int = 0,
             root_only: bool = False,
             strategy: Optional[Any] = None,
             reuse_estep_ll: bool = False) -> SequenceEncodableProbabilityDistribution:
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
        engine (Optional[Any]): Optional ComputeEngine for local kernel scoring/accumulation. Distributed engine
            placement is intentionally deferred to the orchestrator/planner layer.
        precision (Optional[Any]): Optional floating-point precision such as ``'float32'`` or ``np.float64``.
        fields (Optional[Any]): DataFrame column/field selection. A single field yields scalar observations; several
            fields yield tuple observations unless the estimator/model is record-shaped, in which case dict records
            are produced by source column name.
        resources (Optional[Any]): Optional planner resources. When supplied with raw data, optimize encodes through
            the shared encoded-data factory so placement, sub-chunks, and per-shard engines use the orchestrator
            contract.
        placement (Optional[Any]): Optional explicit placement produced by ``pysp.parallel.plan``.
        sub_chunks (int): Number of sub-chunks per placement shard when ``resources`` or ``placement`` is supplied.
        chunk_size (Optional[int]): Approximate chunk size for ordinary local sequence encoding.
        backend (str): Encoded-data backend for raw data. ``'local'`` keeps the historical local encoding unless
            resources/placement are supplied; ``'mp'`` and ``'mpi'`` use the shared encoded-data factory.
        num_workers (Optional[int]): Worker count for ``backend='mp'`` and optional partition count hint for
            ``backend='dask'``.
        client (Optional[Any]): Existing dask.distributed client for ``backend='dask'``. If omitted, the dask backend
            uses an active default client or starts a local threaded client.
        comm (Optional[Any]): MPI communicator for ``backend='mpi'``.
        root (int): MPI root rank for ``backend='mpi'``.
        root_only (bool): MPI root-only data mode for ``backend='mpi'``.
        strategy (Optional[Any]): Optional EM strategy from ``pysp.utils.em`` (e.g. ``AnnealedEM``,
            ``HardEM``, ``MonteCarloEM``) or any callable ``(enc, estimator, model) -> model`` to use
            in place of the standard exact E/M step. ``None`` uses the standard step.
        reuse_estep_ll (bool): When True, reuse the data log-likelihood computed during the E-step
            (the posterior normalizer / forward pass) for convergence instead of running a separate
            scoring pass each iteration -- up to ~2x faster per iteration when the model reports it
            (currently top-level mixtures and HMMs on the default local engine). Convergence then
            lags by one iteration (same fixed point) and the best-likelihood model is returned. Falls
            back to the standard loop for engines/strategies/distributed backends or models that
            can't report the LL. Default False (exact historical behavior).

    Returns:
        SequenceEncodableProbabilityDistribution corresponding to estimator when stopping criteria of EM algorithm
            is met.

    """
    if precision is not None:
        from pysp.engines import engine_with_precision
        engine = engine_with_precision(engine, precision)

    backend_name = str(backend or 'local').lower()
    if data is None and enc_data is None and not (backend_name == 'mpi' and root_only):
        raise Exception('Optimization called with empty data or enc_data.')

    est = estimator if init_estimator is None else init_estimator

    if prev_estimate is None:
        data_encoder = est.accumulator_factory().make().acc_to_encoder()
    else:
        data_encoder = prev_estimate.dist_to_encoder()

    encode_model = prev_estimate
    data_for_encoding = data
    close_created_enc_data = False
    if enc_data is None:
        data_for_encoding = _data_records_for_encoding(data, fields, est, encode_model)
        if resources is not None or placement is not None or backend_name != 'local':
            from pysp.parallel import encoded_data, is_encoded_data_handle
            close_created_enc_data = not is_encoded_data_handle(data_for_encoding)
            enc_data = encoded_data(data_for_encoding, estimator=est, model=encode_model,
                                    encoder=data_encoder, placement=placement, resources=resources,
                                    engine=engine, precision=precision, num_chunks=num_chunks,
                                    sub_chunks=sub_chunks, backend=backend_name,
                                    num_workers=num_workers, client=client, comm=comm, root=root,
                                    root_only=root_only)
        else:
            enc_data = seq_encode(data=data_for_encoding, encoder=data_encoder,
                                  num_chunks=num_chunks, chunk_size=chunk_size)

    try:
        if prev_estimate is None:
            if init_p <= 0.0:
                p = 0.10
            else:
                p = min(max(init_p, 0.0), 1.0)

            mm = seq_initialize(enc_data=enc_data, estimator=est, rng=rng, p=p)
        else:
            mm = prev_estimate

        if enc_vdata is None and vdata is not None:
            vdata_for_encoding = _data_records_for_encoding(vdata, fields, est, mm)
            enc_vdata = seq_encode(vdata_for_encoding, data_encoder, num_chunks=num_chunks, chunk_size=chunk_size)

        # Fused EM (reuse the E-step likelihood normalizer instead of a separate score pass) is only
        # valid on the plain local encoded path with the default engine and exact E-step.
        fused_step_fn = None
        if reuse_estep_ll and engine is None and strategy is None and isinstance(enc_data, list):
            fused_step_fn = _local_fused_step

        best_model, _ = _em_loop(
            enc_data, estimator, mm,
            step_fn=_em_step_fn(engine, strategy),
            ll_fn=_ll_sum_fn(engine),
            max_its=max_its, delta=delta, enc_vdata=enc_vdata,
            out=out, print_iter=print_iter, fused_step_fn=fused_step_fn)

        return best_model
    finally:
        if close_created_enc_data and callable(getattr(enc_data, 'close', None)):
            enc_data.close()


def constant(rho: float):
    """Return a constant streaming step-size schedule."""
    if rho <= 0.0 or rho > 1.0:
        raise ValueError('constant(rho) requires 0 < rho <= 1.')

    def schedule(t: int) -> float:
        return float(rho)

    return schedule


def harmonic(alpha: float, offset: float = 1.0):
    """Return ``rho_t = (offset + t - 1)^(-alpha)`` for streaming EM."""
    if alpha <= 0.5 or alpha > 1.0:
        raise ValueError('harmonic(alpha) requires 0.5 < alpha <= 1.0.')
    if offset <= 0.0:
        raise ValueError('harmonic offset must be positive.')

    def schedule(t: int) -> float:
        tt = max(1, int(t))
        return float((offset + tt - 1.0) ** (-alpha))

    return schedule


def streaming_accumulate(enc_data: Any,
                         estimator: ParameterEstimator,
                         model: SequenceEncodableProbabilityDistribution) -> Tuple[float, Any]:
    """Return one batch's globally tied sufficient-stat accumulator.

    Encoded-data handles can implement ``pysp_stream_accumulate`` to do the
    local/distributed fold themselves.  Plain encoded chunks use the legacy
    in-process ``seq_update`` loop.
    """
    validate_estimator_keys(estimator)
    if hasattr(enc_data, 'pysp_stream_accumulate'):
        nobs, value = enc_data.pysp_stream_accumulate(estimator, model)
        return nobs, estimator.accumulator_factory().make().from_value(value)

    chunks = _local_encoded_chunks(enc_data)
    acc = estimator.accumulator_factory().make()
    nobs = 0.0
    for sz, enc in chunks:
        nobs += sz
        acc.seq_update(enc, np.ones(sz), model)
    stats_dict = dict()
    acc.key_merge(stats_dict)
    acc.key_replace(stats_dict)
    return nobs, acc


class StreamingEstimator(object):
    """Decay-mode online estimator built from accumulator scaling and M-steps."""

    def __init__(self, estimator: ParameterEstimator, schedule=None,
                 model: Optional[SequenceEncodableProbabilityDistribution] = None,
                 init_estimator: Optional[ParameterEstimator] = None,
                 init_p: float = 0.1,
                 rng: Optional[RandomState] = None,
                 encoder=None,
                 num_chunks: int = 1) -> None:
        validate_estimator_keys(estimator)
        self.estimator = estimator
        self.init_estimator = estimator if init_estimator is None else init_estimator
        self.schedule = harmonic(0.7) if schedule is None else schedule
        self.model = model
        self.init_p = init_p
        self.rng = RandomState() if rng is None else rng
        self.encoder = encoder if encoder is not None else (model.dist_to_encoder() if model is not None else None)
        self.num_chunks = num_chunks
        self.running_accumulator = None
        self.nobs = 0.0
        self.step = 0

    def _encode_batch(self, data, enc_data):
        if enc_data is not None:
            if hasattr(enc_data, 'as_seq_chunk'):
                return [enc_data.as_seq_chunk()]
            return enc_data
        if data is None:
            raise ValueError('StreamingEstimator.update requires data or enc_data.')
        if self.encoder is None:
            self.encoder = self.model.dist_to_encoder() if self.model is not None else \
                self.init_estimator.accumulator_factory().make().acc_to_encoder()
        return seq_encode(data, encoder=self.encoder, num_chunks=self.num_chunks)

    def _ensure_model(self, enc_data):
        if self.model is None:
            p = min(max(self.init_p, 0.0), 1.0) if self.init_p > 0.0 else 0.1
            self.model = seq_initialize(enc_data, self.init_estimator, self.rng, p)
            self.encoder = self.model.dist_to_encoder()

    def update(self, data: Optional[Sequence[T]] = None, enc_data: Optional[List[Tuple[int, E0]]] = None) \
            -> SequenceEncodableProbabilityDistribution:
        """Consume one batch and return the updated model."""
        enc_batch = self._encode_batch(data, enc_data)
        self._ensure_model(enc_batch)
        batch_nobs, batch_acc = streaming_accumulate(enc_batch, self.estimator, self.model)

        if self.running_accumulator is None:
            self.running_accumulator = batch_acc
            self.nobs = batch_nobs
        else:
            rho = float(self.schedule(self.step + 1))
            if rho <= 0.0 or rho > 1.0:
                raise ValueError('streaming schedule returned %r; expected 0 < rho <= 1.' % rho)
            self.running_accumulator.scale(1.0 - rho)
            batch_acc.scale(rho)
            self.running_accumulator.combine(batch_acc.value())
            self.nobs = (1.0 - rho) * self.nobs + rho * batch_nobs

        self.model = self.estimator.estimate(self.nobs, self.running_accumulator.value())
        self.step += 1
        return self.model

    def value(self):
        """Return the running sufficient-statistic payload."""
        return None if self.running_accumulator is None else self.running_accumulator.value()

    def reset(self) -> None:
        """Drop running statistics and fitted model state."""
        self.running_accumulator = None
        self.model = None
        self.nobs = 0.0
        self.step = 0


class IncrementalEstimator(object):
    """Neal-Hinton style incremental EM over replaceable data chunks.

    Each chunk contributes a sufficient-statistic payload computed under the
    current model.  Revisiting a chunk subtracts that chunk's previous payload,
    adds the new payload, and runs the ordinary estimator M-step on the pooled
    statistics.  No distribution-specific estimation code lives here; the class
    only uses ``scale(-1)``, ``combine()``, and ``estimate()``.
    """

    def __init__(self, estimator: ParameterEstimator,
                 model: Optional[SequenceEncodableProbabilityDistribution] = None,
                 init_estimator: Optional[ParameterEstimator] = None,
                 init_p: float = 0.1,
                 rng: Optional[RandomState] = None,
                 encoder=None,
                 num_chunks: int = 1) -> None:
        validate_estimator_keys(estimator)
        self.estimator = estimator
        self.init_estimator = estimator if init_estimator is None else init_estimator
        self.model = model
        self.init_p = init_p
        self.rng = RandomState() if rng is None else rng
        self.encoder = encoder if encoder is not None else (model.dist_to_encoder() if model is not None else None)
        self.num_chunks = num_chunks
        self.running_accumulator = None
        self.chunk_values = dict()
        self.nobs_by_chunk = dict()
        self.nobs = 0.0
        self.step = 0

    def _encode_batch(self, data, enc_data):
        if enc_data is not None:
            if hasattr(enc_data, 'as_seq_chunk'):
                return [enc_data.as_seq_chunk()]
            return enc_data
        if data is None:
            raise ValueError('IncrementalEstimator.update requires data or enc_data.')
        if self.encoder is None:
            self.encoder = self.model.dist_to_encoder() if self.model is not None else \
                self.init_estimator.accumulator_factory().make().acc_to_encoder()
        return seq_encode(data, encoder=self.encoder, num_chunks=self.num_chunks)

    def _ensure_model(self, enc_data):
        if self.model is None:
            p = min(max(self.init_p, 0.0), 1.0) if self.init_p > 0.0 else 0.1
            self.model = seq_initialize(enc_data, self.init_estimator, self.rng, p)
            self.encoder = self.model.dist_to_encoder()

    def update(self, chunk_id: Any,
               data: Optional[Sequence[T]] = None,
               enc_data: Optional[List[Tuple[int, E0]]] = None) -> SequenceEncodableProbabilityDistribution:
        """Replace one chunk contribution and return the updated model."""
        if chunk_id is None:
            raise ValueError('IncrementalEstimator.update requires a non-None chunk_id.')
        enc_batch = self._encode_batch(data, enc_data)
        self._ensure_model(enc_batch)
        batch_nobs, batch_acc = streaming_accumulate(enc_batch, self.estimator, self.model)

        if self.running_accumulator is None:
            self.running_accumulator = self.estimator.accumulator_factory().make()

        if chunk_id in self.chunk_values:
            old_acc = self.estimator.accumulator_factory().make()
            old_acc.from_value(copy.deepcopy(self.chunk_values[chunk_id]))
            old_acc.scale(-1.0)
            self.running_accumulator.combine(old_acc.value())
            self.nobs -= self.nobs_by_chunk[chunk_id]

        self.running_accumulator.combine(batch_acc.value())
        self.nobs += batch_nobs
        self.chunk_values[chunk_id] = copy.deepcopy(batch_acc.value())
        self.nobs_by_chunk[chunk_id] = batch_nobs
        self.model = self.estimator.estimate(self.nobs, self.running_accumulator.value())
        self.step += 1
        return self.model

    def value(self):
        """Return the current pooled sufficient-statistic payload."""
        return None if self.running_accumulator is None else self.running_accumulator.value()

    def chunk_value(self, chunk_id: Any):
        """Return a copy of one stored chunk contribution."""
        if chunk_id not in self.chunk_values:
            raise KeyError(chunk_id)
        return copy.deepcopy(self.chunk_values[chunk_id])

    def reset(self) -> None:
        """Drop all chunk contributions and fitted model state."""
        self.running_accumulator = None
        self.chunk_values = dict()
        self.nobs_by_chunk = dict()
        self.model = None
        self.nobs = 0.0
        self.step = 0


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
        enc_data = seq_encode(data, _resolve_encoder(estimator))

    if prev_estimate is None:
        mm = seq_initialize(enc_data, i_est, rng, init_p)
    else:
        mm = prev_estimate

    if hasattr(enc_data, 'cache'):
        enc_data.cache()

    # fixed-iteration stepping with timing only (no convergence/scoring): the
    # lightweight path for callers that just want N EM steps
    t0 = time.time()
    for i in range(max_its):
        mm = seq_estimate(enc_data, estimator, mm)
        if (i + 1) % print_iter == 0:
            out.write('Iteration %d\t E[dT]=%f.\n' % (i + 1, (time.time() - t0) / float(i + 1)))

    return mm


def _torch_for_gradient_fit(engine, precision: Optional[Any] = None):
    try:
        import torch
    except ImportError as e:
        raise ImportError('fit_mle/fit_map require torch for autograd-backed engines.') from e

    if engine is None:
        from pysp.engines import TorchEngine
        engine = TorchEngine(dtype=precision or torch.float64)
    elif precision is not None:
        from pysp.engines import engine_with_precision
        engine = engine_with_precision(engine, precision)
    if not getattr(engine, 'supports_autograd', False):
        raise ValueError('fit_mle/fit_map require an engine with supports_autograd=True.')
    return torch, engine


def _tensor_param(value, engine, torch, transform=None):
    tensor = engine.asarray(value, dtype=getattr(engine, 'dtype', None))
    tensor = tensor.clone().detach()
    eps = 1.0e-8
    if transform == 'log':
        tensor = torch.log(torch.clamp(tensor, min=eps))
    elif transform == 'logit':
        tensor = torch.logit(torch.clamp(tensor, min=eps, max=1.0 - eps))
    elif transform == 'logits':
        tensor = torch.log(torch.clamp(tensor, min=eps))
    tensor.requires_grad_(True)
    return tensor


def _gradient_raw_state(dist, engine, torch, leaves):
    from pysp.stats.declarations import declaration_for

    hook = getattr(dist, 'gradient_fit_state', None)
    if callable(hook):
        state = hook(engine, torch, leaves, _gradient_raw_state, _tensor_param)
        if state is not None:
            return state

    declaration = declaration_for(dist)
    if declaration is None or not callable(getattr(dist, 'backend_seq_log_density', None)):
        return ('fixed', dist)
    if not declaration.differentiable:
        return ('fixed', dist)

    raw = {}
    fixed = {}
    for spec in declaration.parameters:
        value = getattr(dist, spec.name)
        if not spec.differentiable:
            fixed[spec.name] = value
            continue
        if _is_ordered_bound_constraint(spec.constraint):
            anchor = _ordered_bound_anchor(spec.constraint)
            delta = _ordered_bound_delta(getattr(dist, spec.name), getattr(dist, anchor), spec.constraint)
            raw_name = _coupled_raw_name(spec.name, anchor, spec.constraint)
            raw[raw_name] = _tensor_param(delta, engine, torch, transform='log')
            leaves.append(raw[raw_name])
            continue
        raw_name, transform = _raw_name_and_transform(spec.name, spec.constraint)
        raw[raw_name] = _tensor_param(value, engine, torch, transform=transform)
        leaves.append(raw[raw_name])
    return ('leaf', dist, declaration, raw, fixed)


def _raw_name_and_transform(name: str, constraint: str) -> Tuple[str, Optional[str]]:
    if _is_ordered_bound_constraint(constraint):
        return _coupled_raw_name(name, _ordered_bound_anchor(constraint), constraint), 'log'
    if constraint in ('positive', 'positive_vector', 'positive_matrix'):
        return 'log_' + name, 'log'
    if constraint == 'unit_interval':
        return 'logit_' + name, 'logit'
    if constraint in ('simplex', 'simplex_vector', 'row_simplex_matrix', 'column_simplex_matrix'):
        return name + '_logits', 'logits'
    return name, None


def _canonical_value(name: str, spec_name: str, constraint: str, raw: dict, torch):
    if constraint in ('positive', 'positive_vector', 'positive_matrix'):
        return torch.exp(raw['log_' + spec_name])
    if constraint == 'unit_interval':
        return torch.sigmoid(raw['logit_' + spec_name])
    if constraint in ('simplex', 'simplex_vector'):
        return torch.softmax(raw[spec_name + '_logits'], dim=0)
    if constraint == 'row_simplex_matrix':
        return torch.softmax(raw[spec_name + '_logits'], dim=1)
    if constraint == 'column_simplex_matrix':
        return torch.softmax(raw[spec_name + '_logits'], dim=0)
    return raw[name]


def _is_greater_than_constraint(constraint: str) -> bool:
    return str(constraint).startswith('greater_than:')


def _is_less_than_constraint(constraint: str) -> bool:
    return str(constraint).startswith('less_than:')


def _is_ordered_bound_constraint(constraint: str) -> bool:
    return _is_greater_than_constraint(constraint) or _is_less_than_constraint(constraint)


def _ordered_bound_anchor(constraint: str) -> str:
    anchor = str(constraint).split(':', 1)[1] if ':' in str(constraint) else ''
    if not anchor:
        raise ValueError('%s constraint requires an anchor parameter.' % constraint)
    return anchor


def _ordered_bound_delta(value: Any, anchor_value: Any, constraint: str) -> Any:
    if _is_greater_than_constraint(constraint):
        delta = value - anchor_value
    else:
        delta = anchor_value - value
    delta_arr = np.asarray(delta, dtype=np.float64)
    if np.any(delta_arr <= 0.0) or not np.all(np.isfinite(delta_arr)):
        raise ValueError('Initial value for %s must satisfy its ordered bound.' % constraint)
    return delta


def _coupled_raw_name(name: str, anchor: str, constraint: str) -> str:
    return 'log_' + _ordered_bound_delta_name(name, anchor, constraint)


def _ordered_bound_delta_name(name: str, anchor: str, constraint: str) -> str:
    if _is_greater_than_constraint(constraint):
        return '%s_minus_%s' % (name, anchor)
    return '%s_minus_%s' % (anchor, name)


def _gradient_shadow_state(state, torch):
    shadow_fn = getattr(state, 'shadow', None)
    if callable(shadow_fn):
        return shadow_fn(torch, _gradient_shadow_state)
    kind = state[0]
    if kind == 'leaf':
        _, template, declaration, raw, fixed = state
        shadow = object.__new__(type(template))
        shadow.__dict__.update(getattr(template, '__dict__', {}))
        params = {}
        for spec in declaration.parameters:
            if spec.name in fixed:
                params[spec.name] = fixed[spec.name]
            elif _is_ordered_bound_constraint(spec.constraint):
                anchor = _ordered_bound_anchor(spec.constraint)
                anchor_value = params.get(anchor, getattr(template, anchor))
                delta = torch.exp(raw[_coupled_raw_name(spec.name, anchor, spec.constraint)])
                if _is_greater_than_constraint(spec.constraint):
                    params[spec.name] = anchor_value + delta
                else:
                    params[spec.name] = anchor_value - delta
            else:
                raw_name, _ = _raw_name_and_transform(spec.name, spec.constraint)
                params[spec.name] = _canonical_value(raw_name, spec.name, spec.constraint, raw, torch)
            setattr(shadow, spec.name, params[spec.name])
        if 'p_vec' in params:
            shadow.log_p_vec = torch.log(params['p_vec'])
        return shadow
    if kind == 'fixed':
        return state[1]
    raise GradientFitError('Unknown gradient fit state %s.' % kind)


def _gradient_enc_chunks(enc):
    """Normalize an encoded payload into a list of per-chunk payloads to score.

    ``pysp.stats.seq_encode`` returns the chunked form ``[(size, payload), ...]``;
    an encoder's own ``seq_encode`` returns a single bare payload. The gradient
    objective sums log densities over observations, so either form reduces to a
    list of payloads whose per-chunk score sums are added together.
    """
    if (isinstance(enc, list) and enc and
            all(isinstance(c, tuple) and len(c) == 2 and
                isinstance(c[0], (int, np.integer)) for c in enc)):
        return [payload for _, payload in enc]
    return [enc]


def _gradient_score_state(state, enc, engine, torch):
    from pysp.stats.backend import backend_seq_log_density

    score_fn = getattr(state, 'score', None)
    if callable(score_fn):
        return score_fn(enc, engine, torch, _gradient_score_state)
    kind = state[0]
    if kind == 'leaf':
        return backend_seq_log_density(_gradient_shadow_state(state, torch), enc, engine)
    if kind == 'fixed':
        return engine.asarray(state[1].seq_log_density(enc))
    raise GradientFitError('Unknown gradient fit state %s.' % kind)


def _detach_value(x):
    if hasattr(x, 'detach'):
        arr = x.detach().cpu().numpy()
        return float(arr) if np.ndim(arr) == 0 else arr
    return x


def _tensor_scalar(x) -> float:
    return float(x.detach().cpu().item())


def _gradient_best_entry(history: Sequence[float]) -> Tuple[float, int]:
    values = np.asarray(history, dtype=np.float64)
    idx = int(np.nanargmax(values))
    return float(values[idx]), idx


def _gradient_objective_norm(torch, leaves: Sequence[Any], objective) -> float:
    for leaf in leaves:
        if getattr(leaf, 'grad', None) is not None:
            leaf.grad = None
    value = objective()
    value.backward()
    total = None
    for leaf in leaves:
        grad = getattr(leaf, 'grad', None)
        if grad is None:
            continue
        term = torch.sum(grad.detach() * grad.detach())
        total = term if total is None else total + term
    for leaf in leaves:
        if getattr(leaf, 'grad', None) is not None:
            leaf.grad = None
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu().item())


def _gradient_build_state(state, torch):
    build_fn = getattr(state, 'build', None)
    if callable(build_fn):
        return build_fn(torch, _gradient_build_state, _detach_value)
    kind = state[0]
    if kind == 'leaf':
        _, template, declaration, raw, fixed = state
        args = []
        params = {}
        for spec in declaration.parameters:
            if spec.name in fixed:
                value = fixed[spec.name]
            elif _is_ordered_bound_constraint(spec.constraint):
                anchor = _ordered_bound_anchor(spec.constraint)
                anchor_value = params.get(anchor, getattr(template, anchor))
                delta = torch.exp(raw[_coupled_raw_name(spec.name, anchor, spec.constraint)])
                value = anchor_value + delta if _is_greater_than_constraint(spec.constraint) else anchor_value - delta
            else:
                raw_name, _ = _raw_name_and_transform(spec.name, spec.constraint)
                value = _canonical_value(raw_name, spec.name, spec.constraint, raw, torch)
            params[spec.name] = value
            args.append(_detach_value(value))
        kwargs = {}
        if hasattr(template, 'name'):
            kwargs['name'] = getattr(template, 'name')
        if hasattr(template, 'keys'):
            kwargs['keys'] = getattr(template, 'keys')
        try:
            return type(template)(*args, **kwargs)
        except TypeError:
            kwargs.pop('keys', None)
            return type(template)(*args, **kwargs)
    if kind == 'fixed':
        return state[1]
    raise GradientFitError('Unknown gradient fit state %s.' % kind)


def _raw_l2_prior(leaves, initial_leaves, torch, prior_strength: float, engine):
    if prior_strength == 0.0:
        return _prior_zero(torch, engine, leaves[0] if leaves else None)
    penalty = _prior_zero(torch, engine, leaves[0] if leaves else None)
    for cur, start in zip(leaves, initial_leaves):
        delta = cur - start
        penalty = penalty + torch.sum(delta * delta)
    return -0.5 * float(prior_strength) * penalty


def fit_mle(enc: Any, model: SequenceEncodableProbabilityDistribution, engine=None,
            max_its: int = 500, lr: float = 0.05, optimizer: str = 'adam',
            tol: float = 1.0e-7, out: Optional[IO] = None,
            print_iter: int = 100,
            precision: Optional[Any] = None,
            return_result: bool = False) -> Any:
    """Fit converted models by maximizing backend log likelihood with autograd.

    The generic implementation handles declaration-backed tensor leaves and
    delegates structured model families to distribution-owned
    ``gradient_fit_state`` hooks.
    """
    return _fit_gradient(enc, model, engine, max_its, lr, optimizer, tol, out, print_iter,
                         tag='MLE', prior_strength=0.0, precision=precision,
                         return_result=return_result)


def fit_map(enc: Any, model: SequenceEncodableProbabilityDistribution, engine=None,
            prior_strength: float = 1.0, priors: Optional[Any] = None,
            max_its: int = 500, lr: float = 0.05,
            optimizer: str = 'adam', tol: float = 1.0e-7, out: Optional[IO] = None,
            print_iter: int = 100,
            precision: Optional[Any] = None,
            return_result: bool = False) -> Any:
    """Fit converted models with MAP priors over declaration-backed parameters.

    ``prior_strength=0`` is exactly the same objective as ``fit_mle`` when no
    explicit ``priors`` are supplied.  ``priors`` may be a legacy prior dict or
    one of the helpers from ``pysp.utils.priors``.
    """
    return _fit_gradient(enc, model, engine, max_its, lr, optimizer, tol, out, print_iter,
                         tag='MAP', prior_strength=float(prior_strength), priors=priors,
                         precision=precision, return_result=return_result)


def _fit_gradient(enc, model, engine, max_its, lr, optimizer, tol, out, print_iter, tag, prior_strength,
                  priors=None, precision: Optional[Any] = None, return_result: bool = False):
    torch, engine = _torch_for_gradient_fit(engine, precision=precision)
    if hasattr(enc, 'payload'):
        enc = enc.payload
    enc_chunks = _gradient_enc_chunks(enc)

    leaves: List[Any] = []
    state = _gradient_raw_state(model, engine, torch, leaves)
    if not leaves:
        raise GradientFitError('%s has no differentiable parameters.' % type(model).__name__)
    initial_leaves_by_id = {id(leaf): leaf.detach().clone() for leaf in leaves}
    priors = as_prior_dict(priors)

    def log_likelihood():
        total = None
        for chunk in enc_chunks:
            part = engine.sum(_gradient_score_state(state, chunk, engine, torch))
            total = part if total is None else total + part
        return total

    def log_prior():
        if priors is not None or prior_strength != 0.0:
            return _gradient_log_prior_state(state, priors, prior_strength, torch, engine, initial_leaves_by_id)
        return _prior_zero(torch, engine, leaves[0])

    def objective():
        return log_likelihood() + log_prior()

    # one gradient-descent loop, shared with objectives.optimize_torch_objective.
    # restore_best=False keeps this fit's "return the final iterate" semantics
    # (the leaves still hold the final values for the diagnostics below).
    from pysp.utils.objectives import optimize_torch_objective
    loop = optimize_torch_objective(
        leaves, objective, engine=engine, max_its=max_its, lr=lr, optimizer=optimizer,
        tol=tol, maximize=True, out=out, print_iter=print_iter,
        restore_best=False, return_result=True)
    history = list(loop.history)
    iterations = loop.iterations
    converged = loop.converged

    final_obj = history[-1]
    final_ll = _tensor_scalar(log_likelihood())
    final_lp = _tensor_scalar(log_prior())
    final_delta = history[-1] - history[-2] if len(history) > 1 else None
    best_value, best_iteration = _gradient_best_entry(history)
    final_gradient_norm = _gradient_objective_norm(torch, leaves, objective)
    result = GradientFitResult(
        _gradient_build_state(state, torch),
        final_obj,
        iterations,
        history=tuple(history),
        converged=converged,
        initial_value=history[0],
        final_delta=final_delta,
        log_likelihood=final_ll,
        log_prior=final_lp,
        prior_strength=float(prior_strength),
        tag=tag,
        best_value=best_value,
        best_iteration=best_iteration,
        final_gradient_norm=final_gradient_norm,
    )
    return result if return_result else result.as_tuple()


def _gradient_log_prior_state(state, priors, prior_strength: float, torch, engine, initial_leaves_by_id):
    """Structured log prior for declaration-backed MAP objectives."""
    prior_fn = getattr(state, 'log_prior', None)
    if callable(prior_fn):
        return prior_fn(priors, prior_strength, torch, engine, initial_leaves_by_id, _gradient_log_prior_state)
    kind = state[0]
    if kind == 'leaf':
        shadow = _gradient_shadow_state(state, torch)
        declaration, raw = state[2], state[3]
        prior_hook = getattr(shadow, 'gradient_log_prior', None)
        if callable(prior_hook):
            hook_lp = prior_hook(priors, prior_strength, torch, engine)
            if hook_lp is not None:
                return hook_lp

        lp = _prior_zero(torch, engine, next(iter(raw.values()), None))
        matched = False

        for spec in declaration.parameters:
            param_prior = _parameter_prior(priors, spec.name)
            if spec.name in state[4]:
                continue
            if _is_ordered_bound_constraint(spec.constraint):
                anchor = _ordered_bound_anchor(spec.constraint)
                delta_name = _ordered_bound_delta_name(spec.name, anchor, spec.constraint)
                param_prior = _parameter_prior(priors, delta_name) or param_prior
                if param_prior is None:
                    continue
                pfam = _prior_family(param_prior)
                if pfam == 'gamma':
                    value = torch.exp(raw[_coupled_raw_name(spec.name, anchor, spec.constraint)])
                    shape = engine.asarray(param_prior.get('shape', 1.0))
                    rate = engine.asarray(param_prior.get('rate', 0.0))
                    lp = lp + torch.sum((shape - 1.0) * torch.log(value) - rate * value)
                    matched = True
                continue
            if param_prior is None:
                continue
            raw_name, _ = _raw_name_and_transform(spec.name, spec.constraint)
            value = _canonical_value(raw_name, spec.name, spec.constraint, raw, torch)
            pfam = _prior_family(param_prior)
            if pfam == 'gamma' and spec.constraint in ('positive', 'positive_vector', 'positive_matrix'):
                shape = engine.asarray(param_prior.get('shape', 1.0))
                rate = engine.asarray(param_prior.get('rate', 0.0))
                lp = lp + torch.sum((shape - 1.0) * torch.log(value) - rate * value)
                matched = True
            elif pfam == 'beta' and spec.constraint == 'unit_interval':
                alpha = engine.asarray(param_prior.get('alpha', 1.0))
                beta = engine.asarray(param_prior.get('beta', 1.0))
                lp = lp + torch.sum((alpha - 1.0) * torch.log(value) + (beta - 1.0) * torch.log1p(-value))
                matched = True
            elif pfam == 'dirichlet' and spec.constraint in (
                    'simplex', 'simplex_vector', 'row_simplex_matrix', 'column_simplex_matrix'):
                alpha = _dirichlet_alpha_tensor(param_prior.get('alpha'), None, value, engine, torch)
                lp = lp + torch.sum((alpha - 1.0) * torch.log(value))
                matched = True

        if matched:
            return lp
        return _raw_l2_prior(_state_leaves(state), _state_initial_leaves(state, initial_leaves_by_id),
                             torch, prior_strength, engine)
    if kind == 'fixed':
        return _prior_zero(torch, engine)
    return _prior_zero(torch, engine)


def _prior_zero(torch, engine, ref=None):
    if ref is not None:
        return torch.as_tensor(0.0, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(0.0, dtype=engine.dtype, device=engine.device)


def _prior_family(prior):
    return prior.get('family') if isinstance(prior, Mapping) else None


def _prior_parameter_matches(prior, name: str) -> bool:
    return not isinstance(prior, Mapping) or prior.get('parameter') in (None, name)


def _parameter_prior(priors, name: str):
    family = _prior_family(priors)
    if family in ('gamma', 'beta', 'dirichlet') and _prior_parameter_matches(priors, name):
        return priors
    if isinstance(priors, Mapping):
        if isinstance(priors.get('parameters'), Mapping) and name in priors['parameters']:
            return as_prior_dict(priors['parameters'][name])
        if name in priors:
            return as_prior_dict(priors[name])
    return None


def _dirichlet_alpha_tensor(alpha, labels, logits, engine, torch):
    if alpha is None:
        alpha = 1.0
    if isinstance(alpha, Mapping):
        if labels is None:
            raise ValueError('Dirichlet alpha mappings require categorical labels.')
        alpha = [alpha.get(label, 1.0) for label in labels]
    alpha_t = engine.asarray(alpha)
    if alpha_t.ndim == 0:
        return alpha_t + torch.zeros_like(logits)
    return alpha_t


def _state_leaves(state):
    kind = state[0]
    if kind == 'leaf':
        return list(state[3].values())
    return []


def _state_initial_leaves(state, initial_leaves_by_id):
    return [initial_leaves_by_id[id(leaf)] for leaf in _state_leaves(state)]
