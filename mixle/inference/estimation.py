"""Functions for estimating and validating mixle models from observed data.

Useful functions for estimating mixle 'SequenceEncodableProbabilityDistributions' from 'ParameterEstimator'
objects.

"""

from collections.abc import Sequence
from functools import partial
from typing import IO, Any, NamedTuple, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableProbabilityDistribution,
    validate_estimator_keys,
)
from mixle.stats.compute.sequence import (
    seq_encode,
    seq_estimate,
    seq_initialize,
    seq_log_density_sum,
)
from mixle.utils.aliasing import coalesce_alias

T = TypeVar("T")
E0 = TypeVar("E0")


def _resolve_rng_arg(rng: RandomState | int | None, seed: int | None) -> RandomState | None:
    """Reconcile the legacy ``rng=`` argument with its ``seed=`` alias.

    ``seed`` is the spelling every other entry point takes (``create``/``forecast``/``advi``/...),
    so the fit verbs accept it too. Passing both raises ``TypeError`` (the standard alias
    double-supply policy); an integer ``rng`` is coerced to a ``RandomState`` the way ``advi`` /
    ``nuts`` coerce theirs. ``None`` is returned unchanged so each caller keeps its own default.
    """
    value = coalesce_alias("rng", rng, "seed", seed, required=False, default=None)
    if isinstance(value, (int, np.integer)):
        return RandomState(int(value))
    return value


# --- estimator coercion -----------------------------------------------------
def _coerce_estimator(estimator: Any, data: Any) -> ParameterEstimator:
    """Resolve the ``estimator`` argument to a concrete ``ParameterEstimator``.

    The fit verbs (``optimize`` / ``fit`` / ``best_of``) accept three spellings so a model's
    *shape* need not be written twice:

      * a :class:`ParameterEstimator` -- used as-is (the historical contract);
      * a bare **torch module** with ``log_density(batch)`` -- wrapped as a
        :class:`~mixle.models.grad_leaf.GradLeaf`, so ``optimize(x, module)`` needs no contract
        code at all;
      * a distribution **prototype** (any :class:`ProbabilityDistribution`) -- its matching
        estimator tree is taken from ``proto.estimator()``, so you build the structure once and fit
        it directly;
      * ``None`` -- the estimator is inferred from raw ``data`` via
        ``mixle.utils.automatic.get_estimator``.
    """
    if isinstance(estimator, ProbabilityDistribution):
        return estimator.estimator()
    if hasattr(estimator, "log_density") and callable(getattr(estimator, "state_dict", None)):
        # a bare torch density module (scores batches, carries parameters): fit it as a gradient
        # leaf -- the module owns forward and objective, the manufactured contract owns the loop.
        from mixle.models.grad_leaf import GradLeaf, looks_like_torch_module

        if looks_like_torch_module(estimator):
            return GradLeaf(estimator).estimator()
    if estimator is None:
        if data is None:
            raise ValueError(
                "no estimator given and none can be inferred: pass a ParameterEstimator, a "
                "distribution prototype, or raw `data` (estimator inference needs raw data, not "
                "pre-encoded enc_data)."
            )
        from mixle.utils.automatic import get_estimator

        return get_estimator(data)
    return estimator


def _maybe_structured_model(
    data: Any,
    max_its: int,
    out: Any,
    rng: RandomState | None,
    *,
    delta: float | None = 1.0e-9,
    init_p: float = 0.1,
    objective: str = "auto",
    reuse_estep_ll: bool = True,
) -> tuple[Any, Any]:
    """The automatic-structure front door for ``optimize(data)`` / ``fit(data)`` with no estimator.

    For flat tuple records the independent :class:`CompositeDistribution` the automatic detector
    produces is a Naive-Bayes assumption — the one heterogeneous data most often violates. This
    discovers the cross-field dependency graph (:func:`mixle.inference.learn_bayesian_network`) and
    returns it only when it beats the independent composite by BIC on the same data; anything else —
    no edges found, non-record data, too few rows, or an expected data/numeric failure — yields
    ``None`` and the historical composite path proceeds untouched, so the default is never worse.

    Returns ``(structured, composite)``: ``structured`` is the winning dependence model or ``None``;
    ``composite`` is the fully fitted independent composite whenever the BIC gate paid for that fit
    (dependence candidates were scored), so the caller can reuse it instead of refitting the identical
    model. The keyword-only EM knobs (``delta``/``init_p``/``objective``/``reuse_estep_ll``) are
    threaded into that composite fit so it is exactly the fit the caller would otherwise run.
    """
    try:
        rows = list(data)
        if len(rows) < 40:
            return None, None
        first = rows[0]
        if not isinstance(first, tuple) or len(first) < 2:
            return None, None
        n_fields = len(first)
        if any(not isinstance(r, tuple) or len(r) != n_fields for r in rows):
            return None, None
        if any(not isinstance(v, (str, bool, int, float, np.integer, np.floating)) for v in first):
            return None, None  # nested/sequence fields: structure search handles flat records only

        from mixle.inference.bayesian_network import bayesian_network_bic, learn_bayesian_network
        from mixle.inference.structure import _num_free_params
        from mixle.utils.automatic import get_estimator

        # all-continuous records get a second dependence candidate below (a copula), which models
        # heterogeneous marginals + dependence a linear-Gaussian network cannot; other records only try the BN.
        all_continuous = all(isinstance(v, (float, np.floating)) for v in first)
        net = learn_bayesian_network(rows)
        if not net.edges() and not all_continuous:
            return None, None  # independence is what the composite already models; keep the automatic families

        composite = optimize(
            rows,
            get_estimator(rows),
            max_its=max_its,
            delta=delta,
            init_p=init_p,
            rng=rng,
            out=None,
            reuse_estep_ll=reuse_estep_ll,
            objective=objective,
        )
        enc = composite.dist_to_encoder().seq_encode(rows)
        comp_ll = float(np.sum(composite.seq_log_density(enc)))
        n_log = float(np.log(max(len(rows), 2)))
        comp_params = _num_free_params(composite)
        comp_bic = -2.0 * comp_ll + comp_params * n_log

        # candidate dependence models, each scored by BIC on the same data; the independent composite is the
        # baseline and wins ties, so this never returns a worse model than the historical default.
        candidates: list[tuple[float, Any, str]] = []
        if net.edges():
            candidates.append((bayesian_network_bic(net, rows), net, "bayesian-network"))
        if all_continuous:
            from mixle.inference.copula_structure import copula_candidates

            candidates.extend(copula_candidates(rows, composite, comp_params, comp_bic, n_log, max_its, rng))
        if not candidates:
            return None, composite
        # a later, more complex candidate (e.g. a vine) only displaces an earlier, simpler one (e.g. the
        # plain Gaussian copula core it can degenerate to exactly on low-dimensional data) when it wins by
        # more than floating-point noise -- otherwise a sub-ulp BIC difference from platform/BLAS variance
        # would nondeterministically pick the more complex, mathematically-equivalent model.
        best_bic, best_model, desc = candidates[0]
        for bic, model, name in candidates[1:]:
            if bic < best_bic - 1e-6 * max(1.0, abs(best_bic)):
                best_bic, best_model, desc = bic, model, name
        if best_bic >= comp_bic:
            return None, composite
        if out is not None:
            out.write(
                "structure: %s dependence beats independent fields (BIC %.1f < %.1f)\n" % (desc, best_bic, comp_bic)
            )
        return best_model, composite
    except (ValueError, TypeError, KeyError, IndexError, FloatingPointError, OverflowError, ZeroDivisionError):
        # The data-shape and numeric failures the structure search actually raises (np.linalg.LinAlgError
        # is a ValueError): fall back to the historical composite path. Anything else -- an AttributeError,
        # ImportError, ... -- is a structure-path regression and must surface, not be silently swallowed.
        return None, None


# --- data-encoding helpers --------------------------------------------------
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
    # Activate the engine so device-aware M-steps (e.g. NeuralLeaf) follow its device. The context
    # wraps nested component estimates too (a MixtureEstimator's per-leaf estimate runs inside it).
    from mixle.engines.base import using_active_engine

    with using_active_engine(engine):
        return estimator.estimate(nobs, accumulator.value())


def _engine_fused_step(
    enc_data: Any, estimator: ParameterEstimator, prev_estimate: Any, engine: Any
) -> tuple[Any, float | None]:
    """Engine E/M step that also returns the data log-likelihood of ``prev_estimate``.

    When the engine kernel records the E-step normalizer (the :class:`FusedKernel` does -- the data LL
    falls out of the responsibility soft-max), it is returned so the EM loop reuses it instead of a
    separate convergence-LL scoring pass. Returns ``(next_model, ll_or_None)``; ``None`` when the kernel
    can't report it (the loop then scores ``prev_estimate`` itself). Local encoded chunks only.
    """
    validate_estimator_keys(estimator)
    chunks = _local_encoded_chunks(enc_data)
    kernel = prev_estimate.kernel(engine=engine, estimator=estimator)
    accumulator = estimator.accumulator_factory().make()
    nobs = 0.0
    ll = 0.0
    have_ll = True
    for sz, enc in chunks:
        nobs += sz
        accumulator.combine(kernel.accumulate(enc, np.ones(sz, dtype=np.float64)))
        chunk_ll = getattr(kernel, "last_ll", None)
        if chunk_ll is None:
            have_ll = False
        else:
            ll += float(chunk_ll)
    # The key_merge/key_replace pass every other EM driver runs after accumulation (seq_estimate,
    # _local_fused_step; #432 added it to the posterior-transform strategies). Without it, KEYED
    # (tied) parameters silently untie on the engine-kernel path -- and the auto-fusion gate routes
    # large fused-eligible fits here, so a tied-variance mixture at scale estimated per-component
    # stats instead of pooled ones (proven: sigma2 1.36 vs 3.18 where the host ties both at 1.36).
    stats_dict: dict = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    return estimator.estimate(nobs, accumulator.value()), (ll if have_ll else None)


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
    from mixle.data.sources.pandas_source import dataframe_records

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


def _em_step_fn(engine: Any | None, strategy: Any | None = None, objective: Any | None = None):
    """Return the per-iteration (enc, estimator, model) -> model update.

    With ``strategy`` set, the update is delegated to an EM strategy object
    (``mixle.inference.em``) or any callable, which is how alternative E-steps
    (annealed, hard, Monte-Carlo, ...) plug into ``optimize`` without a circular
    import. Otherwise the standard exact E/M step is used (engine-aware).
    """
    if strategy is not None:
        from mixle.inference.em import EMStrategy

        if isinstance(strategy, EMStrategy):

            def step(enc, estimator, model):
                result = strategy.step(enc, estimator, model, engine=engine, objective=objective)
                return result

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


class EMStep(NamedTuple):
    """One accepted EM iteration, handed to an ``optimize(on_step=...)`` callback.

    ``iter`` is 1-based; ``model`` is the current accepted model -- snapshot it to checkpoint, and resume
    with ``optimize(prev_estimate=...)``; ``log_density`` is the training objective at this step; ``delta``
    is its gain over the previous step (``inf`` on the first iteration).
    """

    iter: int
    model: Any
    log_density: float
    delta: float


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
    # structured convergence record for provenance collectors (a custom ``out`` exposing em_record);
    # text output is unaffected for ordinary streams.
    rec = getattr(out, "em_record", None)
    if rec is not None:
        rec(i, float(ll), float(dll), float(vll) if has_vdata else None, obj_label)
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
    out: IO | None = None,
    print_iter: int = 1,
    monotone: bool = True,
    track_best: bool = True,
    fused_step_fn: Any | None = None,
    obj_label: str | None = None,
    on_step: Any | None = None,
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
            on_step,
        )

    from mixle.inference.transaction import MutableStateSnapshot

    _, old_ll = ll_fn(enc_data, model)
    has_v = enc_vdata is not None
    best_vll = ll_fn(enc_vdata, model)[1] if has_v else old_ll
    best_model = model
    best_state = MutableStateSnapshot.capture(best_model)

    for i in range(int(max_its)):
        transaction = MutableStateSnapshot.capture(model, estimator)
        proposal = step_fn(enc_data, estimator, model)
        nxt = getattr(proposal, "model", proposal)
        strategy_accepted = bool(getattr(proposal, "accepted", True))
        _, ll = ll_fn(enc_data, nxt)
        vll = ll_fn(enc_vdata, nxt)[1] if has_v else ll
        # ll and old_ll can both be -inf (e.g. two successive collapsed/singular-covariance steps);
        # -inf - -inf is nan by IEEE 754, which is the semantics every downstream nan-comparison below
        # already relies on (nan >= 0 and nan < delta are both False, so a nan dll is inert) -- silence
        # the resulting RuntimeWarning rather than changing behavior.
        with np.errstate(invalid="ignore"):
            dll = ll - old_ll

        # A non-finite step (e.g. a collapsed/singular covariance producing a NaN/-inf
        # log-likelihood) is never an improvement: never accept it, and do not let it
        # poison the convergence reference ``old_ll`` (which would stall every later
        # iteration on NaN comparisons). For finite ``ll`` this is the historical guard.
        ll_finite = bool(np.isfinite(ll))
        accepted = strategy_accepted and ll_finite and ((dll >= -1.0e-12) or (not monotone))
        if accepted:
            model = nxt
        else:
            transaction.restore()

        # Best-model + reference update happen BEFORE the convergence break so the accepted step on the
        # converging iteration is recorded (otherwise an immediate convergence returns the stale initial
        # model). best-model selection is by validation score; record nxt (the model that achieved vll),
        # not model (unchanged on a rejected step) -- and never select a non-finite step.
        if accepted:
            old_ll = ll
            if track_best and best_vll < vll:
                best_vll = vll
                best_model = nxt
                best_state = MutableStateSnapshot.capture(best_model)

        converged = accepted and (delta is not None) and (0.0 <= dll < delta)
        if out is not None and (converged or (print_iter and (i + 1) % print_iter == 0)):
            _write_em_iter(out, i + 1, ll, dll, vll, has_v, obj_label)
        if on_step is not None:
            reported_ll = ll if accepted else old_ll
            reported_delta = dll if accepted else 0.0
            on_step(EMStep(i + 1, model, float(reported_ll), float(reported_delta)))
        if converged or (not accepted):
            break

    if track_best:
        best_state.restore()
        return best_model, best_vll
    return model, best_vll


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
    on_step=None,
):
    """EM loop that reuses the E-step's likelihood normalizer instead of a separate score pass.

    Each ``fused_step_fn`` call returns ``(next_model, ll_of_model)`` where ``ll_of_model`` is the
    data log-likelihood of the *input* model, computed for free as the posterior normalizer during
    the E-step. The convergence test therefore lags the standard loop by one iteration (it compares
    the likelihood of successive accepted models), which converges to the same fixed point; the
    returned model is still the best-likelihood model seen. Because the fused likelihood belongs
    to the input model, a decrease is detected one iteration late; the candidate is then rejected
    and the last accepted model is retained. When ``ll_of_model`` is ``None`` the model cannot
    report it and we fall back to scoring ``model`` directly for that iteration.
    """
    has_v = enc_vdata is not None
    best_model = model
    best_score = ll_fn(enc_vdata, model)[1] if has_v else None
    prev_ll = None
    accepted_model = model
    nxt = None
    converged = False
    exhausted = True

    for i in range(int(max_its)):
        nxt, ll_model = fused_step_fn(enc_data, estimator, model)
        if ll_model is None:
            _, ll_model = ll_fn(enc_data, model)
        dll = (ll_model - prev_ll) if prev_ll is not None else float("inf")
        if prev_ll is None and not np.isfinite(ll_model):
            # Some categorical/association initializers have zero support and score -inf before
            # their first M-step fills the observed categories. The standard loop can escape that
            # state because it scores the candidate; fused scoring lags by one iteration, so allow
            # exactly this pre-finite repair step and judge the candidate on the next pass.
            model = nxt
            continue
        accepted = bool(np.isfinite(ll_model)) and (prev_ll is None or dll >= -1.0e-12)
        if not accepted:
            exhausted = False
            break

        accepted_model = model
        score = ll_fn(enc_vdata, model)[1] if has_v else ll_model
        if best_score is None or score >= best_score:
            best_score = score
            best_model = model

        converged = (delta is not None) and (prev_ll is not None) and (0.0 <= dll < delta)
        if out is not None and (converged or (print_iter and (i + 1) % print_iter == 0)):
            _write_em_iter(out, i + 1, ll_model, dll, score, has_v, obj_label)
        if on_step is not None:
            # ll_model is the log-likelihood of `model` (the fused step's INPUT, computed for free
            # as the E-step normalizer), not of `nxt` (this iteration's freshly-computed, not-yet-
            # scored output) -- report them paired correctly, matching every other use of ll_model
            # in this loop (`best_model = model` above, `score = ll_fn(enc_vdata, model)`), so an
            # on_step consumer that checkpoints model alongside log_density (as the EMStep docstring
            # explicitly recommends) doesn't persist a mismatched pair.
            on_step(EMStep(i + 1, model, float(ll_model), float(dll)))
        if converged:
            exhausted = False
            break

        prev_ll = ll_model
        model = nxt

    if exhausted and not converged and nxt is not None:
        # Loop ran to max_its: fold the final step into best-model tracking (one extra score pass).
        final_ll = ll_fn(enc_data, nxt)[1]
        if np.isfinite(final_ll) and (prev_ll is None or final_ll - prev_ll >= -1.0e-12):
            accepted_model = nxt
            score = ll_fn(enc_vdata, nxt)[1] if has_v else final_ll
            if best_score is None or score >= best_score:
                best_score = score
                best_model = nxt

    chosen = best_model if track_best else accepted_model
    return chosen, (best_score if best_score is not None else 0.0)


# --- objective resolution (MLE / MAP / VB selection + scorers) --------------
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
    # Prefer the explicit prior signal: get_prior() is None exactly when the estimator carries no
    # parameter prior. This is robust even when the log-prior happens to evaluate to 0.0 at init
    # (which the model_log_density != 0.0 heuristic below would misclassify as MLE).
    get_prior = getattr(estimator, "get_prior", None)
    if callable(get_prior):
        return "map" if get_prior() is not None else "mle"
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


def _resolve_monotone(
    monotone: bool | None,
    estimator: ParameterEstimator,
    model: SequenceEncodableProbabilityDistribution,
) -> bool:
    """Resolve whether every proposed update must improve the outer objective.

    Closed-form updates over immutable distributions use the strict generalized-EM gate. Torch-like
    modules are optimized in place with finite stochastic steps, and variational models can contain
    approximate coordinate or hyperparameter updates; their automatic policy therefore permits a
    non-monotone trajectory while retaining and restoring the best outer-objective state seen.

    An explicit boolean always wins, which lets callers audit a supposedly exact updater with
    ``monotone=True`` or deliberately use best-seen selection with ``monotone=False``.
    """
    if monotone is not None:
        return bool(monotone)

    from mixle.inference.transaction import has_mutable_state

    return (
        not has_mutable_state(model, estimator)
        and not hasattr(model, "seq_local_elbo")
        and not _contains_surrogate_update(estimator)
    )


def _contains_surrogate_update(root: Any) -> bool:
    """Whether an estimator tree optimizes an objective incompatible with outer density scoring."""
    seen: set[int] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if obj is None or isinstance(obj, (str, bytes, bytearray, int, float, complex, bool)):
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        if getattr(obj, "outer_objective_compatible", True) is False:
            return True
        # A module's internals are irrelevant here and can be a very large cyclic graph.
        if callable(getattr(obj, "state_dict", None)):
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
    return False


def _resolve_track_best(track_best: bool | None, estimator: ParameterEstimator) -> bool:
    """Resolve final-vs-best selection for the estimator's actual update objective."""
    if track_best is not None:
        return bool(track_best)
    # Observed density is not a valid selector for NCE, DPO, PINN, or another explicitly
    # surrogate-trained leaf. Their estimator owns the fitting objective, so retain its final
    # finite update instead of preferring an initially unnormalized/high-scoring model.
    return not _contains_surrogate_update(estimator)


# --- public estimation drivers (optimize / fit / best_of) -------------------
def _record_precision_plan(estimator: Any, plan: Any, out: IO | None) -> None:
    """Disclose the ``precision="minimal"`` allocation: on the estimator (which survives the fit;
    fitted models may round-trip custom serializers) and on the requested reporting stream. A
    silent float64 fallback is a receipts violation -- the decision must be observable."""
    try:
        estimator.last_precision_plan = plan
    except (AttributeError, TypeError):  # a slotted/frozen estimator: the stream still discloses
        pass
    if out is not None:
        out.write("precision=minimal: %s (%s)\n" % (np.dtype(plan.compute_dtype).name, plan.rationale))


def optimize(
    data: Sequence[T] | None,
    estimator: ParameterEstimator | ProbabilityDistribution | None = None,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    init_estimator: ParameterEstimator | ProbabilityDistribution | None = None,
    init_p: float = 0.1,
    rng: RandomState | None = None,
    prev_estimate: SequenceEncodableProbabilityDistribution | None = None,
    vdata: Sequence[T] | None = None,
    enc_data: list[tuple[int, E0]] | None = None,
    enc_vdata: list[tuple[int, E0]] | None = None,
    out: IO | None = None,
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
    on_step: Any | None = None,
    structure: str = "auto",
    schedule: str = "full",
    monotone: bool | None = None,
    track_best: bool | None = None,
    seed: int | None = None,
) -> SequenceEncodableProbabilityDistribution:
    """Fit ``estimator`` to ``data`` by a generalized-EM loop, for ``max_its`` iterations or until the
        objective improves by less than ``delta``.

    Each iteration re-estimates every part of the model by whatever its structure calls for -- closed-form
    for conjugate / exponential-family leaves, gradient descent for neural leaves, coordinate descent for
    GLMs, responsibility-weighted EM for latent structure (mixtures, HMMs) -- so a single call fits a
    heterogeneous tree without the caller choosing an algorithm. (The convergence objective is MLE by
    default; a parameter prior switches it to penalized-LL / MAP, and a variational model to the ELBO -- see
    ``objective``.)

    Args:
        data (Optional[List[T]]): List of data type T containing observed data. Must be compatible with data type of
            estimator.
        estimator (ParameterEstimator | ProbabilityDistribution | None): What to fit. A ``ParameterEstimator``
            is used as-is; a distribution **prototype** (any ``ProbabilityDistribution``) is coerced to its
            matching estimator via ``proto.estimator()`` so you build the model shape only once; ``None``
            infers an estimator from raw ``data`` (``mixle.utils.automatic.get_estimator``).
        max_its (int): Maximum number of EM iterations to be performed. Default value is 10 iterations.
        delta (Optional[float]): Stopping criteria for EM algorithm used if max_its is not set: Iterate until
            |old_loglikelihood - new_loglikelihood| < delta or iterations == max_its.
        init_estimator (Optional[ParameterEstimator]): ParameterEstimator to used to initialize EM algorithm parameters.
            If None, estimator is used. Must be consistent with estimator.
        init_p (float): Value in (0.0,1.0] for randomizing the proportion of data points used in initialization.
        rng (RandomState): RandomState used to set seed for initializing EM algorithm. ``None`` resolves to
            a FIXED seed, so the NumPy-driven parts of an un-seeded ``optimize``/``fit`` (initialization,
            EM, subsampling) are deterministic by default; pass your own RandomState when you WANT
            different initializations across calls (e.g. hand-rolled restarts). Torch-backed leaves are
            the deliberate exception: modules that consume torch's global RNG (dropout, VAE
            reparameterization draws, minibatch shuffling) follow torch's own default non-determinism --
            call ``torch.manual_seed`` yourself when a torch-backed fit must be exactly reproducible.
            An integer is accepted and coerced to ``RandomState(rng)``. Mutually exclusive with ``seed``.
        vdata (Optional[Sequence[T]]): Optional validation set.
        prev_estimate (Optional[SeqeuenceEncodableProbabilityDistribution]): Optional model estimate used from prior
            fitting. Must be consistent with estimator.
        enc_data (Optional[List[Tuple[int, E]]]): Optional encoded data of form
            List[Tuple[int, E]]. Formed from data if None.
        enc_vdata (Optional[List[Tuple[int, E0]]]): Optional sequence encoded validation set.
        out (IO | None): Stream for per-iteration EM progress lines. Defaults to ``None`` (quiet, so the
            library does not spam stdout in normal use); pass ``out=sys.stdout`` to watch convergence.
        print_iter (int): Print the log-likelihood difference every print_iter iterations; the final converged
            iteration is always reported. Pass print_iter=0 to suppress the periodic lines (keeping only the
            converged line), or out=None to silence entirely.
        num_chunks (int): Number of chunks for encoded data.
        engine (Optional[Any]): Optional ComputeEngine for local kernel scoring/accumulation. Distributed engine
            placement is intentionally deferred to the orchestrator/planner layer.
        precision (Optional[Any]): Optional floating-point precision such as ``'float32'`` or ``np.float64``.
            Pass ``'auto'`` to let ``mixle.engines.auto_precision`` choose from the data and engine:
            float32 only on a GPU torch engine with well-conditioned numeric data, else float64.
            Pass ``'minimal'`` for the data-aware CPU allocator (``mixle.inference.precision_plan``): it
            inspects the data magnitude and the model's leaf families/conditioning and runs the reduced
            float32 fused kernel where verified safe (accumulation stays float64), else float64 -- the
            "preserve accuracy with minimal compute" default for local fits.
        fields (Optional[Any]): DataFrame column/field selection. A single field yields scalar observations; several
            fields yield tuple observations unless the estimator/model is record-shaped, in which case dict records
            are produced by source column name.
        resources (Optional[Any]): Optional planner resources. When supplied with raw data, optimize encodes through
            the shared encoded-data factory so placement, sub-chunks, and per-shard engines use the orchestrator
            contract.
        placement (Optional[Any]): Optional explicit placement produced by ``mixle.utils.parallel.planner.plan``.
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
        strategy (Optional[Any]): Optional EM strategy from ``mixle.inference.em`` (e.g. ``AnnealedEM``,
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
        monotone (Optional[bool]): Outer-objective acceptance policy. ``None`` (default) uses strict
            generalized-EM acceptance for immutable closed-form updates and best-seen selection for
            mutable neural or variational/approximate updates. In best-seen mode finite downhill
            steps may be traversed, but the returned model (including mutable module parameters) is
            restored to the best selected-objective value observed. Pass ``True`` to reject the
            first decreasing step, or ``False`` to permit a non-monotone trajectory explicitly.

            Convergence contract (worklist Q5.4; pinned by ``em_convergence_contract_test``): under
            strict acceptance the accepted-round objectives are non-decreasing within tolerance and
            -- bounded above by the estimator variance floors -- the objective sequence converges;
            limit-point stationarity is the classical EM/GEM theory per family. Under best-seen
            selection the guarantee is the best visited iterate; a neural leaf fit with
            ``lr_decay`` in ``(0.5, 1]`` additionally follows a Robbins--Monro step schedule (the
            condition stochastic-approximation EM analyses require). A non-finite objective is
            never accepted and never becomes the convergence reference.
        track_best (Optional[bool]): Whether to restore the best outer-objective state seen. ``None``
            (default) does so except when an estimator explicitly declares a surrogate fitting
            objective, such as NCE; observed density is not a valid selector until such a model is
            normalized, so its final finite update is returned. Pass a boolean to override selection.
        on_step (Optional[Callable[[EMStep], None]]): Optional per-iteration callback receiving an
            :class:`EMStep` ``(iter, model, log_density, delta)`` for the accepted model. Use it to
            checkpoint a long run -- e.g. ``on_step=registry.checkpointer('run', every=5)`` -- and
            resume with ``prev_estimate=``. Called on every iteration regardless of ``print_iter``.
        structure (str): ``'auto'`` (default) makes the tagline literal for flat tuple records fit
            with no estimator: the cross-field dependency graph is discovered
            (:func:`mixle.inference.learn_bayesian_network`) and returned when it beats the
            independent composite by BIC — otherwise (no edges, non-record data, or any failure)
            the historical automatic-composite path proceeds untouched. ``'off'`` restores the
            unconditional historical behavior. Only consulted when ``estimator`` is ``None`` and no
            ``prev_estimate``/``init_estimator``/``strategy``/``enc_data`` is supplied.
        schedule (str): ``'full'`` (default) -- unchanged vanilla full-tree EM, every round scores
            and re-estimates every component. ``'auto'`` engages the block-coordinate-ascent
            scheduler (:mod:`mixle.inference.block_em`): after one full bootstrap sweep, components
            are ranked by their last observed complete-data Q gain per structural-cost unit, and
            only the highest-value ones within a per-round cost budget are re-estimated. Inactive
            component scores are cached while their parameters remain unchanged. This is a
            scheduling choice: observed likelihood is transactionally gated non-decreasing every round, and
            when there is no useful ranking to do (e.g. every component looks equally worth
            updating) the scheduler degenerates to doing exactly what ``'full'`` does. Only
            engaged when the model is a plain local-backend ``MixtureDistribution``/
            ``MixtureEstimator`` MLE fit with no explicit ``strategy``/``engine``/``resources``/
            ``placement`` -- anything else silently falls back to ``'full'`` (never an error, never
            a behavior change beyond scheduling).
        seed (Optional[int]): Integer seed for initializing the EM algorithm -- shorthand for
            ``rng=RandomState(seed)``, matching the ``seed=`` argument the samplers and the other
            entry points take. Mutually exclusive with ``rng`` (passing both raises ``TypeError``).

    Returns:
        SequenceEncodableProbabilityDistribution corresponding to estimator when stopping criteria of EM algorithm
            is met.

    """
    rng = _resolve_rng_arg(rng, seed)
    if (
        estimator is None
        and structure == "auto"
        and data is not None
        and enc_data is None
        and prev_estimate is None
        and init_estimator is None
        and strategy is None
    ):
        structured, independent_composite = _maybe_structured_model(
            data,
            max_its,
            out,
            rng,
            delta=delta,
            init_p=init_p,
            objective=objective,
            reuse_estep_ll=reuse_estep_ll,
        )
        if structured is not None:
            return structured
        # When the dependence candidates were scored and lost, the BIC gate already paid for a full
        # fit of the independent composite that this path would now refit identically -- reuse it,
        # but only when every knob it could not see is at the default that front-door fit used.
        if independent_composite is not None and (
            vdata is None
            and enc_vdata is None
            and out is None
            and num_chunks == 1
            and engine is None
            and precision is None
            and fields is None
            and resources is None
            and placement is None
            and sub_chunks == 1
            and chunk_size is None
            and backend == "local"
            and on_step is None
            and schedule == "full"
            and monotone is None
            and track_best is None
        ):
            return independent_composite
    estimator = _coerce_estimator(estimator, data)
    if init_estimator is not None:
        init_estimator = _coerce_estimator(init_estimator, data)
    rng = RandomState(0) if rng is None else rng  # fixed default: the numpy side of an un-seeded fit is deterministic
    minimal_precision_pending = False
    if precision == "minimal":
        # Data-aware allocation: inspect the data + model and run the reduced-precision fused kernel only
        # where it is verified safe; else stay float64. The accumulation is float64 either way.
        # A warm start has a model to inspect NOW; a cold start defers planning until seq_initialize
        # has produced one -- planning against ``prev_estimate=None`` silently allocated float64 to
        # every cold-start fit (the common case). The deferred decision lands immediately after the
        # model materializes, before any engine consumer runs.
        if prev_estimate is not None:
            from mixle.inference.precision_plan import recommend_compute_precision

            plan = recommend_compute_precision(prev_estimate, data)
            _record_precision_plan(estimator, plan, out)
            if plan.reduced() and engine is None:
                from mixle.engines import NumpyEngine

                engine = NumpyEngine(dtype=plan.compute_dtype, prefer_fused=True)
        else:
            minimal_precision_pending = True
        precision = None  # carried by the explicit engine (or the default float64 host path)
    elif precision == "auto":
        from mixle.engines import auto_precision

        precision = auto_precision(data, engine=engine)
        # When 'auto' settles on float64 with no explicit engine, keep the default host path
        # (already float64 and fastest on CPU) rather than forcing the engine path.
        if engine is None and precision == "float64":
            precision = None
    if precision is not None:
        from mixle.engines import engine_with_precision

        engine = engine_with_precision(engine, precision)

    backend_name = str(backend or "local").lower()
    if data is None and enc_data is None and not (backend_name == "mpi" and root_only):
        raise ValueError("Optimization called with empty data or enc_data.")
    # Empty (but non-None) data previously slipped through and silently returned the initialized
    # prior/default model -- a wrong answer, not a fit. Match ppl's "fit() received empty data."
    if data is not None and enc_data is None and hasattr(data, "__len__") and len(data) == 0:
        raise ValueError("optimize() received empty data.")

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
            from mixle.utils.parallel.planner import encoded_data, is_encoded_data_handle

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

        if minimal_precision_pending:
            # Deferred cold-start leg of precision="minimal" (see the block above). Parallel
            # backends already built their encoded handles engine-free, so they keep the
            # conservative float64 rather than switching dtype mid-flight.
            from mixle.inference.precision_plan import PrecisionPlan, recommend_compute_precision

            local_path = resources is None and placement is None and backend_name == "local"
            if engine is None and data is not None and local_path:
                plan = recommend_compute_precision(mm, data)
            else:
                plan = PrecisionPlan(np.float64, "minimal: non-local backend or engine already supplied -> float64")
            _record_precision_plan(estimator, plan, out)
            if plan.reduced() and engine is None:
                from mixle.engines import NumpyEngine

                engine = NumpyEngine(dtype=plan.compute_dtype, prefer_fused=True)

        if enc_vdata is None and vdata is not None:
            vdata_for_encoding = _data_records_for_encoding(vdata, fields, est, mm)
            enc_vdata = seq_encode(vdata_for_encoding, data_encoder, num_chunks=num_chunks, chunk_size=chunk_size)

        # The prior is the single switch: 'auto' uses the variational ELBO when the model exposes
        # one (seq_local_elbo), the penalized log-likelihood when the estimator carries a prior, and
        # the plain log-likelihood otherwise. So a Bayesian estimator converges/selects on the right
        # objective whether the caller reaches for optimize() or fit().
        resolved_objective = _resolve_objective(objective, estimator, mm)
        surrogate_update = _contains_surrogate_update(estimator)
        strict_monotone = _resolve_monotone(monotone, estimator, mm)
        select_best = _resolve_track_best(track_best, estimator)
        loop_delta = None if surrogate_update else delta

        # D3 block-EM scheduler: 'auto' dispatches to mixle.inference.block_em's greedy
        # gain-per-cost scheduler when the fit is a plain local MLE MixtureDistribution/
        # MixtureEstimator fit with none of the other execution knobs engaged (those all need
        # the standard _em_loop path); everything else silently keeps the 'full' behavior below
        # -- schedule='auto' never errors or changes what is computed, only how it is scheduled.
        if schedule == "auto":
            from mixle.inference.block_em import is_block_em_eligible
            from mixle.inference.fusion_policy import prefer_block_schedule as _prefer_block

            if (
                is_block_em_eligible(mm, estimator)
                and not surrogate_update
                and strategy is None
                and resolved_objective == "mle"
                and engine is None
                and resources is None
                and placement is None
                and backend_name == "local"
                and _prefer_block(mm, enc_data, max_its)
            ):
                from mixle.inference.block_em import run_block_em

                best_model, block_history = run_block_em(enc_data, estimator, mm, max_its=max_its, delta=delta)
                if out is not None and block_history:
                    last = block_history[-1]
                    out.write(
                        "block-em: %d rounds, final objective=%.6f, mean active fraction=%.3f\n"
                        % (
                            len(block_history),
                            last.objective,
                            float(np.mean([h.active_fraction for h in block_history])),
                        )
                    )
                return best_model

        # Cost-model auto-fusion: with no explicit engine, switch a large-enough local MLE fit of a
        # fusible model onto the single-pass fused numba kernel (parity-identical, ~1.7x once warm).
        if (
            engine is None
            and backend_name == "local"
            and resources is None
            and placement is None
            and strategy is None
            and resolved_objective == "mle"
        ):
            from mixle.inference.fusion_policy import should_auto_fuse

            if should_auto_fuse(mm, enc_data, max_its):
                from mixle.engines import FUSED_NUMPY_ENGINE

                engine = FUSED_NUMPY_ENGINE

        # Fused EM (reuse the E-step likelihood normalizer instead of a separate score pass) is only
        # valid for the plain-likelihood objective on the local encoded path with an exact E-step --
        # the reused normalizer is the data LL, not the penalized LL / ELBO. The default engine reads it
        # from the accumulator; an explicit engine reads it from a kernel that reports it (the
        # FusedKernel), and gracefully falls back to a scoring pass otherwise.
        fused_step_fn = None
        from mixle.inference.transaction import has_mutable_state

        if (
            reuse_estep_ll
            and strict_monotone
            and resolved_objective == "mle"
            and strategy is None
            and isinstance(enc_data, list)
            and not has_mutable_state(mm, estimator)
        ):
            if engine is None:
                fused_step_fn = _local_fused_step
            else:
                fused_step_fn = partial(_engine_fused_step, engine=engine)

        objective_scorer = _objective_scorer(resolved_objective, estimator, engine)
        objective_fn = lambda candidate: objective_scorer(enc_data, candidate)[1]
        best_model, _ = _em_loop(
            enc_data,
            estimator,
            mm,
            step_fn=_em_step_fn(engine, strategy, objective_fn),
            ll_fn=objective_scorer,
            max_its=max_its,
            delta=loop_delta,
            enc_vdata=enc_vdata,
            out=out,
            print_iter=print_iter,
            monotone=strict_monotone,
            track_best=select_best,
            fused_step_fn=fused_step_fn,
            obj_label={"mle": None, "map": "penalized-LL", "vb": "ELBO"}[resolved_objective],
            on_step=on_step,
        )

        return best_model
    finally:
        if close_created_enc_data and callable(getattr(enc_data, "close", None)):
            enc_data.close()


def fit(
    data: Sequence[T] | None,
    estimator: ParameterEstimator | ProbabilityDistribution | None = None,
    max_its: int = 10,
    delta: float | None = 1.0e-6,
    init_estimator: ParameterEstimator | ProbabilityDistribution | None = None,
    **kwargs: Any,
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

    ``fit`` is a thin wrapper over :func:`optimize` -- they share the one EM/objective loop. ``fit`` adds
    only the opt-in data-structure check, a Bayesian-leaning default ``delta`` (1e-6), and the exact
    per-iteration-scored loop (``reuse_estep_ll=False``). Every other :func:`optimize` keyword -- engines,
    precision, distributed ``backend``, ``on_step``, the fused E-step -- is accepted here too and forwarded
    verbatim, so reaching for a heavier knob never means switching verbs. ``estimator`` accepts the same
    three spellings as :func:`optimize` (estimator, distribution prototype, or ``None`` to infer from data).
    """
    # Resolve seed=/rng= up front (same alias policy as optimize) so the automatic-structure path
    # below sees the same RandomState the forwarded optimize call would.
    if "seed" in kwargs or "rng" in kwargs:
        kwargs["rng"] = _resolve_rng_arg(kwargs.pop("rng", None), kwargs.pop("seed", None))
    if (
        estimator is None
        and kwargs.get("structure", "auto") == "auto"
        and data is not None
        and kwargs.get("enc_data") is None
        and init_estimator is None
        and kwargs.get("prev_estimate") is None
        and kwargs.get("strategy") is None
    ):
        structured, independent_composite = _maybe_structured_model(
            data,
            max_its,
            kwargs.get("out"),
            kwargs.get("rng"),
            delta=delta,
            init_p=kwargs.get("init_p", 0.1),
            objective=kwargs.get("objective", "auto"),
            reuse_estep_ll=kwargs.get("reuse_estep_ll", False),  # fit forces the exact scored loop below
        )
        if structured is not None:
            return structured
        # Same double-fit repair as optimize's front door: the losing-candidates path already fitted
        # this exact composite (fit's delta/reuse_estep_ll/objective/init_p were threaded into it), so
        # reuse it unless some other optimize knob was passed through **kwargs.
        _threaded_or_inert = {
            "structure",
            "rng",
            "out",
            "enc_data",
            "prev_estimate",
            "strategy",
            "init_p",
            "objective",
            "reuse_estep_ll",
        }
        if independent_composite is not None and kwargs.get("out") is None and not (set(kwargs) - _threaded_or_inert):
            return independent_composite
    estimator = _coerce_estimator(estimator, data)
    if init_estimator is not None:
        init_estimator = _coerce_estimator(init_estimator, data)
    if data is None and kwargs.get("enc_data") is None:
        raise ValueError("fit called with empty data or enc_data.")
    # opt-in sample-structure check: a tagged DataSource is verified against the model it feeds (warns on
    # a mismatch, e.g. a SEQUENTIAL source handed to an i.i.d. leaf). Bare lists carry no structure tag.
    if data is not None and getattr(data, "structure", None) is not None:
        from mixle.data.structure import check_model_structure

        check_model_structure(estimator if init_estimator is None else init_estimator, data.structure)
    # fit owns these two defaults; reuse_estep_ll is forced off (exact per-iteration scoring). Everything
    # else flows through **kwargs so any optimize knob works without changing verbs.
    kwargs.setdefault("reuse_estep_ll", False)
    return optimize(
        data,
        estimator,
        max_its=max_its,
        delta=delta,
        init_estimator=init_estimator,
        **kwargs,
    )


def best_of(
    data: Sequence[T] | None,
    vdata: Sequence[T] | None,
    est: ParameterEstimator | ProbabilityDistribution | None,
    trials: int,
    max_its: int,
    init_p: float,
    delta: float,
    rng: RandomState | int | None = None,
    init_estimator: ParameterEstimator | ProbabilityDistribution | None = None,
    enc_data: list[tuple[int, E0]] | None = None,
    enc_vdata: Sequence[tuple[int, E0]] | None = None,
    out: IO | None = None,
    print_iter: int = 1,
    reuse_estep_ll: bool = True,
    objective: str = "auto",
    seed: int | None = None,
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
        rng (RandomState): RandomState for setting seed. An integer is coerced to ``RandomState(rng)``;
            ``None`` (default) resolves to the fixed default seed. Mutually exclusive with ``seed``.
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
        seed (Optional[int]): Integer seed -- shorthand for ``rng=RandomState(seed)``. Mutually
            exclusive with ``rng`` (passing both raises ``TypeError``).

    Returns:
        Tuple of log-likelihood of best fitting model and the best fitting model from number of trials.

    """
    rng = _resolve_rng_arg(rng, seed)
    if data is None and enc_data is None:
        raise ValueError("Optimization called with empty data or enc_data.")

    est = _coerce_estimator(est, data)
    if init_estimator is not None:
        init_estimator = _coerce_estimator(init_estimator, data)
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


# --- streaming / online estimation ------------------------------------------
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
    """Streaming / recursive-Bayes driver over the mixle.stats estimator protocol.

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
        rng: RandomState | None = None,
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
        self.rng = (
            RandomState(0) if rng is None else rng
        )  # fixed default: the numpy side of an un-seeded fit is deterministic
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
