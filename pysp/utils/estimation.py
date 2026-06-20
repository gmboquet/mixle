"""Functions for estimating and validating pysparkplug models from observed data.

Useful functions for estimating pysparkplug 'SequenceEncodableProbabilityDistributions' from 'ParameterEstimator'
objects.

"""

import sys
import time
from collections.abc import Sequence
from typing import IO, Any, TypeVar

import numpy as np
from numpy.random import RandomState

from pysp.stats import (
    seq_encode,
    seq_estimate,
    seq_initialize,
    seq_log_density_sum,
    validate_estimator_keys,
)
from pysp.stats.compute.pdist import ParameterEstimator, SequenceEncodableProbabilityDistribution

T = TypeVar("T")
E0 = TypeVar("E0")


def best_of(
    data: Sequence[T] | None,
    vdata: Sequence[T] | None,
    est: ParameterEstimator,
    trials: int,
    max_its: int,
    init_p: float,
    delta: float,
    rng: RandomState,
    init_estimator: ParameterEstimator | None = None,
    enc_data: list[tuple[int, E0]] | None = None,
    enc_vdata: Sequence[tuple[int, E0]] | None = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
    reuse_estep_ll: bool = True,
    objective: str = "auto",
) -> tuple[float, SequenceEncodableProbabilityDistribution]:
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
        reuse_estep_ll (bool): Default True. Forwarded to each trial's ``optimize`` call -- reuse the
            E-step likelihood for convergence instead of a separate scoring pass (see ``optimize``).
            Set False to force the exact historical per-iteration scoring behavior.
        objective (str): Convergence/selection objective forwarded to each trial's ``optimize`` call;
            ``'auto'`` (default) selects MLE / MAP / variational Bayes from the prior (see ``optimize``).

    Returns:
        Tuple of log-likelihood of best fitting model and the best fitting model from number of trials.

    """
    if data is None and enc_data is None:
        raise Exception("Optimization called with empty data or enc_data.")

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
        mm = optimize(
            None,
            est,
            init_estimator=i_est,
            enc_data=enc_data,
            enc_vdata=enc_vdata,
            max_its=max_its,
            delta=delta,
            init_p=init_p,
            rng=rng,
            out=out,
            print_iter=print_iter,
            reuse_estep_ll=reuse_estep_ll,
            objective=objective,
        )
        _, vll = seq_log_density_sum(score_data, mm)
        if out is not None:
            out.write("Trial %d. VLL=%f\n" % (kk + 1, vll))
        if vll > rv_ll:
            rv_ll, rv_mm = vll, mm

    return rv_ll, rv_mm


def _local_encoded_chunks(enc_data: Any) -> list[tuple[int, Any]]:
    if hasattr(enc_data, "as_seq_chunk"):
        return [enc_data.as_seq_chunk()]
    if isinstance(enc_data, tuple) and len(enc_data) == 2 and isinstance(enc_data[0], (int, np.integer, float)):
        return [enc_data]
    if isinstance(enc_data, list):
        return enc_data
    raise ValueError(
        "engine-aware optimize currently supports local encoded chunks only; "
        "distributed engine orchestration is handled by a later planner slice."
    )


def _engine_seq_log_density_sum(
    enc_data: Any, estimate: SequenceEncodableProbabilityDistribution, engine: Any
) -> tuple[float, float]:
    chunks = _local_encoded_chunks(enc_data)
    kernel = estimate.kernel(engine=engine)
    nobs = 0.0
    ll = 0.0
    for sz, enc in chunks:
        nobs += sz
        ll += float(np.asarray(engine.to_numpy(kernel.score(enc)), dtype=np.float64).sum())
    return nobs, ll


def _engine_seq_estimate(
    enc_data: Any, estimator: ParameterEstimator, prev_estimate: SequenceEncodableProbabilityDistribution, engine: Any
) -> SequenceEncodableProbabilityDistribution:
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
    return hasattr(data, "columns") and hasattr(data, "loc")


def _recordish(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "fields") and hasattr(obj, "sources")


def _dataframe_fields(fields: Any, estimator: Any, model: Any) -> Any:
    if fields is not None:
        return fields
    for obj in (model, estimator):
        if _recordish(obj):
            return tuple(zip(obj.fields, obj.sources))
    return None


def _data_records_for_encoding(data: Any, fields: Any, estimator: Any, model: Any) -> Any:
    if not _dataframe_like(data) and fields is None:
        return data
    from pysp.data.dataframe import dataframe_records

    record_fields = _dataframe_fields(fields, estimator, model)
    return dataframe_records(data, fields=record_fields, as_dict=_recordish(model) or _recordish(estimator))


# --- shared EM driver ------------------------------------------------------
#
# optimize/best_of/iterate (and em.run_em) all share the same skeleton: build an
# encoder, encode the data, initialize (or reuse) a model, then iterate an E/M
# step until convergence. The helpers below factor out that skeleton so each
# entry point is a thin policy wrapper over one tested loop.


def _resolve_encoder(
    estimator: ParameterEstimator, prev_estimate: SequenceEncodableProbabilityDistribution | None = None
) -> Any:
    """Return the data encoder for a fitting run (model encoder if continuing)."""
    if prev_estimate is not None:
        return prev_estimate.dist_to_encoder()
    return estimator.accumulator_factory().make().acc_to_encoder()


def _ll_sum_fn(engine: Any | None):
    """Return a (enc, model) -> (count, log_likelihood) scorer for the engine."""
    if engine is None:
        return seq_log_density_sum
    return lambda enc, model: _engine_seq_log_density_sum(enc, model, engine)


def _em_step_fn(engine: Any | None, strategy: Any | None = None):
    """Return the per-iteration (enc, estimator, model) -> model update.

    With ``strategy`` set, the update is delegated to an EM strategy object
    (``pysp.utils.em``) or any callable, which is how alternative E-steps
    (annealed, hard, Monte-Carlo, ...) plug into ``optimize`` without a circular
    import. Otherwise the standard exact E/M step is used (engine-aware).
    """
    if strategy is not None:
        step_method = getattr(strategy, "step", None)
        if callable(step_method):

            def step(enc, estimator, model):
                result = step_method(enc, estimator, model, engine=engine)
                return getattr(result, "model", result)

            return step
        if callable(strategy):
            return lambda enc, estimator, model: strategy(enc, estimator, model)
        raise TypeError(
            "strategy must be an EM strategy with .step(...) or a callable (enc, estimator, model) -> model."
        )
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
    return nxt, getattr(accumulator, "_seq_ll", None)


def _write_em_iter(
    out: IO | None, i: int, ll: float, dll: float, vll: float, has_vdata: bool, obj_label: str | None = None
) -> None:
    """Write one EM progress line.

    With ``obj_label=None`` (plain maximum likelihood) the historical log-likelihood format is used;
    for the penalized-LL / ELBO objectives ``obj_label`` (e.g. ``'penalized-LL'``, ``'ELBO'``) names
    the quantity so the progress line is not mislabeled as a data log-likelihood.
    """
    if out is None:
        return
    if obj_label is None:
        if has_vdata:
            out.write(
                "Iteration %d: ln[p_mat(Data|Model)]=%e, ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e, "
                "ln[p_mat(Valid Data|Model)]=%e\n" % (i, ll, dll, vll)
            )
        else:
            out.write(
                "Iteration %d: ln[p_mat(Data|Model)]=%e, "
                "ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]=%e\n" % (i, ll, dll)
            )
    elif has_vdata:
        out.write("Iteration %d: %s=%e, d%s=%e, valid-%s=%e\n" % (i, obj_label, ll, obj_label, dll, obj_label, vll))
    else:
        out.write("Iteration %d: %s=%e, d%s=%e\n" % (i, obj_label, ll, obj_label, dll))


def _em_loop(
    enc_data: Any,
    estimator: ParameterEstimator,
    model: SequenceEncodableProbabilityDistribution,
    step_fn: Any,
    ll_fn: Any,
    max_its: int,
    delta: float | None,
    enc_vdata: Any | None = None,
    out: IO | None = sys.stdout,
    print_iter: int = 1,
    monotone: bool = True,
    track_best: bool = True,
    fused_step_fn: Any | None = None,
    obj_label: str | None = None,
) -> tuple[SequenceEncodableProbabilityDistribution, float]:
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
        return _fused_em_loop(
            enc_data,
            estimator,
            model,
            fused_step_fn,
            ll_fn,
            max_its,
            delta,
            enc_vdata,
            out,
            print_iter,
            track_best,
            obj_label,
        )

    _, old_ll = ll_fn(enc_data, model)
    has_v = enc_vdata is not None
    best_vll = ll_fn(enc_vdata, model)[1] if has_v else old_ll
    best_model = model

    for i in range(int(max_its)):
        nxt = step_fn(enc_data, estimator, model)
        _, ll = ll_fn(enc_data, nxt)
        vll = ll_fn(enc_vdata, nxt)[1] if has_v else ll
        dll = ll - old_ll

        # A non-finite step (e.g. a collapsed/singular covariance producing a NaN/-inf
        # log-likelihood) is never an improvement: never accept it, and do not let it
        # poison the convergence reference ``old_ll`` (which would stall every later
        # iteration on NaN comparisons). For finite ``ll`` this is the historical guard.
        ll_finite = bool(np.isfinite(ll))
        if ll_finite and ((dll >= 0) or (delta is None) or (not monotone)):
            model = nxt

        converged = (delta is not None) and (dll < delta)
        if out is not None and (converged or (print_iter and (i + 1) % print_iter == 0)):
            _write_em_iter(out, i + 1, ll, dll, vll, has_v, obj_label)
        if converged:
            break

        if ll_finite:
            old_ll = ll
        if track_best and best_vll < vll:
            best_vll = vll
            best_model = model

    return (best_model if track_best else model), best_vll


def _fused_em_loop(
    enc_data,
    estimator,
    model,
    fused_step_fn,
    ll_fn,
    max_its,
    delta,
    enc_vdata,
    out,
    print_iter,
    track_best,
    obj_label=None,
):
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

        dll = (ll_model - prev_ll) if prev_ll is not None else float("inf")
        converged = (delta is not None) and (prev_ll is not None) and (dll < delta)
        if out is not None and (converged or (print_iter and (i + 1) % print_iter == 0)):
            _write_em_iter(out, i + 1, ll_model, dll, score, has_v, obj_label)
        if converged:
            break

        # Keep the convergence reference on the last finite likelihood; a non-finite
        # ``ll_model`` (e.g. a collapsed covariance) must not stall the delta test on NaN.
        if np.isfinite(ll_model):
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


def optimize(
    data: Sequence[T] | None,
    estimator: ParameterEstimator,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    init_estimator: ParameterEstimator | None = None,
    init_p: float = 0.1,
    rng: RandomState = RandomState(),
    prev_estimate: SequenceEncodableProbabilityDistribution | None = None,
    vdata: Sequence[T] | None = None,
    enc_data: list[tuple[int, E0]] | None = None,
    enc_vdata: list[tuple[int, E0]] | None = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
    num_chunks: int = 1,
    engine: Any | None = None,
    precision: Any | None = None,
    fields: Any | None = None,
    resources: Any | None = None,
    placement: Any | None = None,
    sub_chunks: int = 1,
    chunk_size: int | None = None,
    backend: str = "local",
    num_workers: int | None = None,
    client: Any | None = None,
    comm: Any | None = None,
    root: int = 0,
    root_only: bool = False,
    strategy: Any | None = None,
    reuse_estep_ll: bool = True,
    objective: str = "auto",
) -> SequenceEncodableProbabilityDistribution:
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
        out (IO): IO stream to write out iterations of EM algorithm. Pass out=None to silence all output.
        print_iter (int): Print the log-likelihood difference every print_iter iterations; the final converged
            iteration is always reported. Pass print_iter=0 to suppress the periodic lines (keeping only the
            converged line), or out=None to silence entirely.
        num_chunks (int): Number of chunks for encoded data.
        engine (Optional[Any]): Optional ComputeEngine for local kernel scoring/accumulation. Distributed engine
            placement is intentionally deferred to the orchestrator/planner layer.
        precision (Optional[Any]): Optional floating-point precision such as ``'float32'`` or ``np.float64``.
            Pass ``'auto'`` to let ``pysp.engines.auto_precision`` choose from the data and engine:
            float32 only on a GPU torch engine with well-conditioned numeric data, else float64.
        fields (Optional[Any]): DataFrame column/field selection. A single field yields scalar observations; several
            fields yield tuple observations unless the estimator/model is record-shaped, in which case dict records
            are produced by source column name.
        resources (Optional[Any]): Optional planner resources. When supplied with raw data, optimize encodes through
            the shared encoded-data factory so placement, sub-chunks, and per-shard engines use the orchestrator
            contract.
        placement (Optional[Any]): Optional explicit placement produced by ``pysp.planner.plan``.
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
        reuse_estep_ll (bool): Default True. Reuse the data log-likelihood computed during the E-step
            (the posterior normalizer / forward pass / variational ELBO) for convergence instead of
            running a separate scoring pass each iteration -- typically ~1.5-2x faster per iteration
            for latent models (mixtures, HMMs and variants, topic models, associations, IBP, ...) on
            the default local engine. Convergence then lags by one iteration (same fixed point) and
            the best-likelihood model is returned; fixed-iteration fits (delta=None) are identical to
            the standard loop. Automatically falls back to the standard loop for engines/strategies/
            distributed backends or models that can't report the LL (no slowdown there). Set False to
            force the exact historical per-iteration scoring behavior.
        objective (str): Convergence/selection objective. ``'auto'`` (default) makes the prior the
            single switch -- a model exposing a variational ELBO (``seq_local_elbo``) is fit by
            variational Bayes (``'vb'``), an estimator carrying a parameter prior by penalized
            log-likelihood (``'map'``), and everything else by plain maximum likelihood (``'mle'``).
            Pass ``'mle'`` / ``'map'`` / ``'vb'`` to force a specific objective. ``fit`` accepts the
            same argument; both share this resolution so a Bayesian estimator is fit on the correct
            objective regardless of the verb used. (Only ``'mle'`` is compatible with the fused
            E-step shortcut; ``reuse_estep_ll`` is ignored for ``'map'``/``'vb'``.)

    Returns:
        SequenceEncodableProbabilityDistribution corresponding to estimator when stopping criteria of EM algorithm
            is met.

    """
    if precision == "auto":
        from pysp.engines import auto_precision

        precision = auto_precision(data, engine=engine)
        # When 'auto' settles on float64 with no explicit engine, keep the default host path
        # (already float64 and fastest on CPU) rather than forcing the engine path.
        if engine is None and precision == "float64":
            precision = None
    if precision is not None:
        from pysp.engines import engine_with_precision

        engine = engine_with_precision(engine, precision)

    backend_name = str(backend or "local").lower()
    if data is None and enc_data is None and not (backend_name == "mpi" and root_only):
        raise Exception("Optimization called with empty data or enc_data.")

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
        if resources is not None or placement is not None or backend_name != "local":
            from pysp.planner import encoded_data, is_encoded_data_handle

            close_created_enc_data = not is_encoded_data_handle(data_for_encoding)
            enc_data = encoded_data(
                data_for_encoding,
                estimator=est,
                model=encode_model,
                encoder=data_encoder,
                placement=placement,
                resources=resources,
                engine=engine,
                precision=precision,
                num_chunks=num_chunks,
                sub_chunks=sub_chunks,
                backend=backend_name,
                num_workers=num_workers,
                client=client,
                comm=comm,
                root=root,
                root_only=root_only,
            )
        else:
            enc_data = seq_encode(
                data=data_for_encoding, encoder=data_encoder, num_chunks=num_chunks, chunk_size=chunk_size
            )

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

        # The prior is the single switch: 'auto' uses the variational ELBO when the model exposes
        # one (seq_local_elbo), the penalized log-likelihood when the estimator carries a prior, and
        # the plain log-likelihood otherwise. So a Bayesian estimator converges/selects on the right
        # objective whether the caller reaches for optimize() or fit().
        resolved_objective = _resolve_objective(objective, estimator, mm)

        # Fused EM (reuse the E-step likelihood normalizer instead of a separate score pass) is only
        # valid for the plain-likelihood objective on the local encoded path with the default engine
        # and exact E-step -- the reused normalizer is the data LL, not the penalized LL / ELBO.
        fused_step_fn = None
        if (
            reuse_estep_ll
            and resolved_objective == "mle"
            and engine is None
            and strategy is None
            and isinstance(enc_data, list)
        ):
            fused_step_fn = _local_fused_step

        best_model, _ = _em_loop(
            enc_data,
            estimator,
            mm,
            step_fn=_em_step_fn(engine, strategy),
            ll_fn=_objective_scorer(resolved_objective, estimator, engine),
            max_its=max_its,
            delta=delta,
            enc_vdata=enc_vdata,
            out=out,
            print_iter=print_iter,
            fused_step_fn=fused_step_fn,
            obj_label={"mle": None, "map": "penalized-LL", "vb": "ELBO"}[resolved_objective],
        )

        return best_model
    finally:
        if close_created_enc_data and callable(getattr(enc_data, "close", None)):
            enc_data.close()


def _data_objective_sum(enc_data: Any, model: SequenceEncodableProbabilityDistribution) -> float:
    """Data-dependent part of the Bayesian fit objective.

    For variational models exposing ``seq_local_elbo`` (e.g. variational mixtures, DPM) this is the
    sum of per-observation local ELBO contributions; otherwise it is the observed-data log-likelihood
    at the current (MAP) parameter estimates.
    """
    if hasattr(model, "seq_local_elbo"):
        return float(sum(model.seq_local_elbo(u[1]).sum() for u in enc_data))
    _, rv = seq_log_density_sum(enc_data, model)
    return rv


def _model_objective(estimator: ParameterEstimator, model: SequenceEncodableProbabilityDistribution) -> float:
    """Prior/global part of the Bayesian fit objective.

    For MAP estimators this is the log-prior density of the estimated parameters; for variational
    estimators it is the data-independent part of the ELBO (prior cross-entropies plus variational
    entropies). Returns ``0.0`` when the estimator carries no usable prior.
    """
    fn = getattr(estimator, "model_log_density", None)
    if fn is None:
        return 0.0
    rv = fn(model)
    return 0.0 if rv is None else float(rv)


_VALID_OBJECTIVES = ("auto", "mle", "map", "vb")


def _resolve_objective(
    objective: str, estimator: ParameterEstimator, model: SequenceEncodableProbabilityDistribution
) -> str:
    """Resolve the convergence/selection objective for a fitting run.

    The prior is the single switch: with ``objective='auto'`` (the default) a model that exposes a
    variational ELBO (``seq_local_elbo``) is fit by ``'vb'``, an estimator that carries a parameter
    prior (non-zero ``model_log_density``) by ``'map'`` (penalized log-likelihood), and everything
    else by plain ``'mle'``. Pass an explicit ``'mle'`` / ``'map'`` / ``'vb'`` to override.
    """
    obj = (objective or "auto").lower()
    if obj not in _VALID_OBJECTIVES:
        raise ValueError("objective must be one of %r, got %r." % (_VALID_OBJECTIVES, objective))
    if obj != "auto":
        return obj
    if hasattr(model, "seq_local_elbo"):
        return "vb"
    if _model_objective(estimator, model) != 0.0:
        return "map"
    return "mle"


def _objective_scorer(resolved: str, estimator: ParameterEstimator, engine: Any | None):
    """Return a ``(enc, model) -> (count, score)`` scorer for the resolved objective.

    ``'mle'`` scores the plain data log-likelihood (and is the only objective compatible with the
    fused-E-step shortcut). ``'map'`` / ``'vb'`` score the penalized log-likelihood / ELBO
    ``_data_objective_sum + _model_objective`` (the data term auto-adapts to ``seq_local_elbo``).
    """
    if resolved == "mle":
        return _ll_sum_fn(engine)

    def scorer(enc: Any, model: SequenceEncodableProbabilityDistribution) -> tuple[float, float]:
        return 0.0, _data_objective_sum(enc, model) + _model_objective(estimator, model)

    return scorer


def fit(
    data: Sequence[T] | None,
    estimator: ParameterEstimator,
    max_its: int = 10,
    delta: float | None = 1.0e-6,
    init_estimator: ParameterEstimator | None = None,
    init_p: float = 0.1,
    rng: RandomState = RandomState(),
    prev_estimate: SequenceEncodableProbabilityDistribution | None = None,
    vdata: Sequence[T] | None = None,
    enc_data: list[tuple[int, E0]] | None = None,
    enc_vdata: list[tuple[int, E0]] | None = None,
    out: IO | None = sys.stdout,
    print_iter: int = 1,
    objective: str = "auto",
) -> SequenceEncodableProbabilityDistribution:
    """Fit a model in the Bayesian (variational / MAP) sense, returning the posterior-bearing model.

    This is the posterior-returning counterpart of :func:`optimize`. ``fit`` iterates the EM/VB update
    that maximizes the objective selected by ``objective`` (default ``'auto'``):

      - ``'auto'`` -- the prior is the single switch: ``'vb'`` when the model exposes ``seq_local_elbo``,
        ``'map'`` when the estimator carries a parameter prior, else ``'mle'``;
      - ``'mle'`` -- plain data log-likelihood (ignores any prior in the objective);
      - ``'map'`` / ``'vb'`` -- penalized log-likelihood / ELBO ``obj = data term + prior term``, where the
        data term is the observed-data LL (MAP) or local-ELBO contributions (variational), and the prior
        term is ``estimator.model_log_density(model)``.

    Convergence is checked on the chosen objective, so under ``'map'``/``'vb'`` the prior is part of the
    stopping rule and conjugate updates never decrease it. The returned model carries its conjugate
    posterior forward as ``model.get_prior()``. With no prior anywhere, every objective reduces to plain
    EM, so ``fit`` and ``optimize`` agree for frequentist estimators. ``optimize`` accepts the same
    ``objective`` argument; the two share this resolution so a Bayesian estimator is fit correctly
    regardless of which verb the caller reaches for.

    Args otherwise mirror :func:`optimize` (local encoded path). Returns the model with the best
    validation log-likelihood seen during the run.
    """
    if data is None and enc_data is None:
        raise Exception("fit called with empty data or enc_data.")

    est = estimator if init_estimator is None else init_estimator
    div_error = np.seterr(divide="ignore")
    try:
        encoder = _resolve_encoder(est, prev_estimate)
        if enc_data is None:
            enc_data = seq_encode(data, encoder)
        if enc_vdata is None:
            enc_vdata = enc_data if vdata is None else seq_encode(vdata, encoder)

        if prev_estimate is None:
            p = 0.10 if init_p <= 0.0 else min(max(init_p, 0.0), 1.0)
            mm = seq_initialize(enc_data=enc_data, estimator=est, rng=rng, p=p)
        else:
            mm = prev_estimate

        resolved_objective = _resolve_objective(objective, estimator, mm)

        def _obj_terms(model: SequenceEncodableProbabilityDistribution) -> tuple[float, float]:
            # (data term, prior/global term) for the resolved objective; 'mle' drops the prior term.
            if resolved_objective == "mle":
                return _ll_sum_fn(None)(enc_data, model)[1], 0.0
            return _data_objective_sum(enc_data, model), _model_objective(estimator, model)

        data_ll, model_ll = _obj_terms(mm)
        old_obj = data_ll + model_ll
        _, best_vll = seq_log_density_sum(enc_vdata, mm)
        best_model = mm

        for i in range(max(1, max_its)):
            mm_next = seq_estimate(enc_data, estimator, mm)

            data_ll, model_ll = _obj_terms(mm_next)
            obj = data_ll + model_ll
            _, vll = seq_log_density_sum(enc_vdata, mm_next)
            dobj = obj - old_obj

            if (dobj >= 0) or (delta is None):
                mm = mm_next
                if best_vll < vll:
                    best_vll = vll
                    best_model = mm

            converged = (delta is not None) and (dobj < delta)
            if out is not None and (converged or (print_iter and (i + 1) % print_iter == 0)):
                label = "Terminating" if converged else "Iteration"
                out.write(
                    "%s %d. OBJ=%f, dOBJ=%e, LL=%f, MLL=%f, VLL=%f\n"
                    % (label, i + 1, obj, dobj, data_ll, model_ll, vll)
                )
            if converged:
                break

            old_obj = obj

        return best_model
    finally:
        np.seterr(**div_error)


def constant(rho: float):
    """Return a constant streaming step-size schedule."""
    if rho <= 0.0 or rho > 1.0:
        raise ValueError("constant(rho) requires 0 < rho <= 1.")

    def schedule(t: int) -> float:
        return float(rho)

    return schedule


def harmonic(alpha: float, offset: float = 1.0):
    """Return ``rho_t = (offset + t - 1)^(-alpha)`` for streaming EM."""
    if alpha <= 0.5 or alpha > 1.0:
        raise ValueError("harmonic(alpha) requires 0.5 < alpha <= 1.0.")
    if offset <= 0.0:
        raise ValueError("harmonic offset must be positive.")

    def schedule(t: int) -> float:
        tt = max(1, int(t))
        return float((offset + tt - 1.0) ** (-alpha))

    return schedule


def posterior_carry() -> str:
    """Return the recursive-conjugate streaming mode name.

    In ``posterior_carry`` mode each fitted posterior becomes the next batch's prior, i.e. the
    stream performs exact recursive Bayesian updating: the conjugate posterior after batch ``t`` is
    fed in as the conjugate prior for batch ``t + 1``.
    """
    return "posterior_carry"


def forgetting(rho: float):
    """Return a constant forgetting / power-prior schedule for streaming Bayes.

    ``rho`` in ``(0, 1]`` down-weights each incoming batch's sufficient statistics by a constant
    factor before the conjugate ``estimate`` call, so older evidence decays geometrically. ``rho=1``
    recovers ordinary (un-forgotten) accumulation.
    """
    if rho <= 0.0 or rho > 1.0:
        raise ValueError("forgetting(rho) requires 0 < rho <= 1.")

    def schedule(t: int) -> float:
        return float(rho)

    return schedule


def _stream_accumulate(
    enc_data: Any,
    estimator: ParameterEstimator,
    model: SequenceEncodableProbabilityDistribution,
) -> tuple[float, Any]:
    """Accumulate one encoded batch's globally tied sufficient statistics.

    Returns ``(nobs, suff_stat)`` where ``suff_stat`` is the accumulator ``value()`` after the
    key-merge/key-replace pass that ties globally shared parameters across the batch.
    """
    accumulator = estimator.accumulator_factory().make()
    nobs = 0.0
    for sz, enc in enc_data:
        nobs += sz
        accumulator.seq_update(enc, np.ones(sz), model)

    stats_dict: dict[Any, Any] = dict()
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    return nobs, accumulator


class BayesianStreamingEstimator:
    """Streaming / recursive-Bayes driver over the pysp.stats estimator protocol.

    ``mode='posterior_carry'`` (the default) performs exact recursive conjugate updating: each fitted
    posterior is carried forward as the next batch's prior by rebuilding the estimator from the fitted
    model (``model.estimator()`` returns an estimator whose prior is the model's posterior).

    ``mode='forgetting'`` applies a power-prior step: the current batch's accumulated sufficient
    statistics are scaled by ``rho = schedule(step)`` (via the accumulator's ``scale``, which
    preserves structural support metadata such as a categorical's ``min_val``) before the ordinary
    conjugate ``estimate`` call, and the batch's contribution to ``nobs`` is scaled to match.

    The public surface is ``BayesianStreamingEstimator(estimator, mode=..., schedule=...)`` plus
    ``.update(data=None, enc_data=None)`` and ``.reset()``.
    """

    def __init__(
        self,
        estimator: ParameterEstimator,
        mode: str | None = "posterior_carry",
        schedule: Any | None = None,
        model: SequenceEncodableProbabilityDistribution | None = None,
        init_estimator: ParameterEstimator | None = None,
        init_p: float = 0.1,
        rng: RandomState = RandomState(),
        num_chunks: int = 1,
    ) -> None:
        self.estimator = estimator
        self.init_estimator = estimator if init_estimator is None else init_estimator
        self.mode = posterior_carry() if mode is None else mode
        self.schedule = schedule
        if self.mode == "forgetting" and self.schedule is None:
            self.schedule = forgetting(1.0)
        if self.mode not in ("posterior_carry", "forgetting"):
            raise ValueError("mode must be 'posterior_carry' or 'forgetting'.")
        self.model = model
        self.init_p = init_p
        self.rng = rng
        self.num_chunks = num_chunks
        self.step = 0
        self.nobs = 0.0
        if model is not None:
            self._carry_prior_from(model)

    def _carry_prior_from(self, model: SequenceEncodableProbabilityDistribution) -> None:
        """Carry the model's posterior forward as the estimator's prior for the next batch.

        ``model.estimator()`` returns a fresh estimator whose conjugate prior is the model's current
        posterior; rebuilding from it carries that posterior forward as the next batch's conjugate
        prior. Falls back to leaving the estimator unchanged when the model can't supply one.
        """
        make_estimator = getattr(model, "estimator", None)
        if callable(make_estimator):
            self.estimator = make_estimator()

    def _ensure_model(self, data: Sequence[T] | None, enc_data: Any | None) -> None:
        if self.model is not None:
            return
        if enc_data is None and data is None:
            raise ValueError("BayesianStreamingEstimator.update requires data for initialization.")
        p = min(max(self.init_p, 0.0), 1.0) if self.init_p > 0.0 else 0.1
        enc = enc_data if enc_data is not None else self._encode(data)
        self.model = seq_initialize(enc_data=enc, estimator=self.init_estimator, rng=self.rng, p=p)
        self._carry_prior_from(self.model)

    def _encode(self, data: Sequence[T]) -> Any:
        encoder = self.model.dist_to_encoder()
        return seq_encode(data, encoder, num_chunks=self.num_chunks)

    def _encode_batch(self, data: Sequence[T] | None, enc_data: Any | None) -> Any:
        if enc_data is not None:
            return enc_data
        if data is None:
            raise ValueError("BayesianStreamingEstimator.update requires data or enc_data.")
        return self._encode(data)

    def update(
        self, data: Sequence[T] | None = None, enc_data: Any | None = None
    ) -> SequenceEncodableProbabilityDistribution:
        """Consume one batch and return the updated posterior-bearing model."""
        self._ensure_model(data, enc_data)
        enc_batch = self._encode_batch(data, enc_data)
        batch_nobs, accumulator = _stream_accumulate(enc_batch, self.estimator, self.model)

        if self.mode == "forgetting":
            rho = float(self.schedule(self.step + 1))
            if rho <= 0.0 or rho > 1.0:
                raise ValueError("forgetting schedule returned %r; expected 0 < rho <= 1." % rho)
            accumulator.scale(rho)
            batch_nobs *= rho

        self.model = self.estimator.estimate(batch_nobs, accumulator.value())
        self._carry_prior_from(self.model)
        self.nobs += batch_nobs
        self.step += 1
        return self.model

    def reset(self) -> None:
        """Drop the current model and stream counters."""
        self.model = None
        self.step = 0
        self.nobs = 0.0


def iterate(
    data: list[T],
    estimator: ParameterEstimator | None,
    max_its: int,
    prev_estimate: SequenceEncodableProbabilityDistribution | None = None,
    init_p: float = 0.1,
    rng: RandomState | None = RandomState(),
    out: IO = sys.stdout,
    enc_data: list[tuple[int, E0]] | None = None,
    init_estimator: ParameterEstimator | None = None,
    print_iter: int = 1,
) -> SequenceEncodableProbabilityDistribution:
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
        raise Exception("Optimization called with empty data or enc_data.")

    i_est = estimator if init_estimator is None else init_estimator

    if enc_data is None:
        enc_data = seq_encode(data, _resolve_encoder(estimator))

    if prev_estimate is None:
        mm = seq_initialize(enc_data, i_est, rng, init_p)
    else:
        mm = prev_estimate

    if hasattr(enc_data, "cache"):
        enc_data.cache()

    # fixed-iteration stepping with timing only (no convergence/scoring): the
    # lightweight path for callers that just want N EM steps
    t0 = time.time()
    for i in range(max_its):
        mm = seq_estimate(enc_data, estimator, mm)
        if out is not None and print_iter and (i + 1) % print_iter == 0:
            out.write("Iteration %d\t E[dT]=%f.\n" % (i + 1, (time.time() - t0) / float(i + 1)))

    return mm
