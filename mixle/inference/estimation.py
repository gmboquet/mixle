"""Functions for estimating and validating mixle models from observed data.

Useful functions for estimating mixle 'SequenceEncodableProbabilityDistributions' from 'ParameterEstimator'
objects.

"""

import warnings
from collections.abc import Sequence
from copy import deepcopy
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
from mixle.utils.exact import require_exact_bool

T = TypeVar("T")
E0 = TypeVar("E0")


def _resolve_rng_arg(rng: RandomState | int | None, seed: int | None) -> RandomState | None:
    """Reconcile the legacy ``rng=`` argument with its ``seed=`` alias.

    ``seed`` is the spelling every other entry point takes (``create``/``forecast``/``advi``/...),
    so the fit verbs accept it too. Passing both raises ``TypeError`` (the standard alias
    double-supply policy); an integer ``rng`` is coerced to a ``RandomState`` the way ``advi`` /
    ``nuts`` coerce theirs. A modern :class:`numpy.random.Generator` (``np.random.default_rng``)
    is accepted as well: the internals draw through the legacy ``RandomState`` surface
    (``.randint`` partition/initialization seeding), so a ``RandomState`` is derived from one draw
    of the generator -- deterministic given the generator's state, and it advances the generator
    exactly once, matching what handing over an rng means everywhere else. ``None`` is returned
    unchanged so each caller keeps its own default.
    """
    value = coalesce_alias("rng", rng, "seed", seed, required=False, default=None)
    if isinstance(value, (int, np.integer)):
        return RandomState(int(value))
    if isinstance(value, np.random.Generator):
        return RandomState(int(value.integers(2**31 - 1)))
    return value


def _reject_masked_data(data: Any, entry: str) -> None:
    """Reject a numpy masked array whose mask actually masks something, at the public entry.

    Every encoder coerces through ``np.asarray``, which returns a MaskedArray's bare ``.data`` --
    the fill values under the mask would be fit as real observations with no error anywhere. Only
    an array with at least one masked value is rejected; a trivial (all-False) mask carries no
    missingness and fits like any ndarray.
    """
    if np.ma.isMaskedArray(data) and np.ma.is_masked(data):
        raise ValueError(
            "%s received a numpy masked array with %d masked value(s); the mask would be silently "
            "dropped and the masked entries fit as real data. Pass data.compressed() to drop the "
            "masked entries, or use OptionalEstimator/MISSING for missingness-aware fitting."
            % (entry, int(np.ma.count_masked(data)))
        )


def _estimator_carries_prior(estimator: Any, _depth: int = 0) -> bool:
    """True if any estimator in the (small, nested) estimator tree declares prior information.

    Zero total evidence is only an unfittable request when the estimator has NOTHING to fall back
    on but its arbitrary no-evidence defaults. An estimator carrying ``suff_stat`` or
    ``pseudo_count`` defines the zero-evidence posterior to BE that prior -- returning it is the
    documented MAP semantics, not a fabricated fit -- so the all-zero-weight guard must not fire
    (found by the wave-3 adversarial check: the guard's first version rejected exactly that
    legitimate state, this codebase's historical defect class). Duck-typed attribute walk, no
    concrete-family imports (compute_metadata contract).
    """
    if estimator is None or _depth > 6:
        return False
    for name in ("suff_stat", "pseudo_count"):
        value = getattr(estimator, name, None)
        # families default these to tuples OF Nones (e.g. GaussianEstimator suff_stat=(None, None)):
        # a container with no actual value in it declares no prior.
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            if any(item is not None for item in value):
                return True
        else:
            return True
    children: list[Any] = []
    for name in ("estimator", "estimators", "components", "accumulator"):
        child = getattr(estimator, name, None)
        if child is None:
            continue
        children.extend(child if isinstance(child, (list, tuple)) else [child])
    return any(_estimator_carries_prior(child, _depth + 1) for child in children)


def _reject_all_zero_observation_weights(data: Any, entry: str) -> None:
    """Reject USER-SUPPLIED observation weights that sum to zero, at the public entry only.

    A dataset of ``WeightedObservation`` rows whose weights are all ``0.0`` carries no evidence:
    the estimators' no-evidence defaults (e.g. a Gaussian at ``mu=0.0`` with the floor variance)
    would be returned as if they were a fit of the data. Weight ``0.0`` on SOME rows is the
    documented "contributes nothing" meaning and is untouched; only the all-zero total is an
    unfittable request, and it is named here -- at the entry point, where the weights are the
    user's -- because inside EM a zero total weight is the routine dead-component M-step that the
    no-evidence defaults exist to serve (raising there would crash ordinary mixture fits).

    DECLARED LIMIT: the guard reads ``data`` and therefore covers only the ``data=`` path.
    Pre-encoded ``enc_data`` is opaque here by design -- its layout is per-family knowledge this
    module is barred from having (compute_metadata contract) -- so an all-zero-weight dataset
    passed as ``enc_data`` still fits to the no-evidence defaults; that fit is not silent to
    provenance readers (``numerical_repairs()`` reports the floored variance). Estimators carrying
    real prior information (``suff_stat``/``pseudo_count``) are exempted at the call sites: zero
    evidence there returns the prior, the documented MAP semantics.
    """
    if data is None or not hasattr(data, "__len__") or len(data) == 0:
        return
    try:
        first = data[0]
    except (TypeError, KeyError, IndexError):
        return
    # Identified by type name, not isinstance: this file is barred from importing concrete
    # mixle.stats modules (compute_metadata contract), and the guard only ever needs to recognize
    # mixle's own payload wrapper -- anything else falls through to the combinator's validation.
    payload_type = type(first)
    if payload_type.__name__ != "WeightedObservation" or not payload_type.__module__.startswith("mixle.stats."):
        return
    for obs in data:
        if type(obs) is not payload_type:
            return  # mixed payloads are validated by the weighted combinator itself
        try:
            weight = float(obs.weight)
        except (TypeError, ValueError):
            return  # non-numeric weights get the combinator's own error
        if weight != 0.0:
            # any nonzero weight (non-finite ones included -- those raise downstream) is evidence
            return
    raise ValueError(
        "%s received observation weights that sum to zero: all %d WeightedObservation weights "
        "are 0.0, so the data carry no evidence to fit. Give at least one observation a positive "
        "weight, or drop the zero-weight rows." % (entry, len(data))
    )


# --- estimator coercion -----------------------------------------------------
def _coerce_estimator(estimator: Any, data: Any, fields: Any = None) -> ParameterEstimator:
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

        if fields is not None:
            # ``fields=`` selects the columns the encoder will actually fit, so inference must see
            # those same records: inferring from the full table made ``get_estimator`` size the
            # model to every column while the encoded rows carried only the selection, and the fit
            # then failed with a row-shape ContractError blaming the user's data (campaign T2-09,
            # the documented-workaround defect). ``fields is not None`` forces the same conversion
            # here that ``_data_records_for_encoding`` applies at encode time, so the two agree.
            data = _data_records_for_encoding(data, fields, None, None)
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
    returns it only when it beats the independent composite by BIC on the same data. Unsupported
    shapes, too few rows, and a discovered graph with no edges yield ``None`` and the historical
    composite path proceeds untouched. Once fitting starts, errors propagate: an internal fit
    failure must not be misreported as an unsupported-data routing decision.

    Returns ``(structured, composite)``: ``structured`` is the winning dependence model or ``None``;
    ``composite`` is the fully fitted independent composite whenever the BIC gate paid for that fit
    (dependence candidates were scored), so the caller can reuse it instead of refitting the identical
    model. The keyword-only EM knobs (``delta``/``init_p``/``objective``/``reuse_estep_ll``) are
    threaded into that composite fit so it is exactly the fit the caller would otherwise run.
    """
    rows = list(data)
    if len(rows) < 40:
        return None, None
    first = rows[0]
    if not isinstance(first, tuple) or len(first) < 2:
        return None, None
    n_fields = len(first)
    if any(not isinstance(row, tuple) or len(row) != n_fields for row in rows):
        return None, None
    scalar = (str, bool, int, float, np.integer, np.floating)
    if any(not isinstance(value, scalar) for row in rows for value in row):
        return None, None  # nested/sequence fields: structure search handles flat records only

    from mixle.inference.bayesian_network import bayesian_network_bic, learn_bayesian_network
    from mixle.inference.structure import _num_free_params
    from mixle.utils.automatic import get_estimator

    # All-continuous records get a second dependence candidate below (a copula), which models
    # heterogeneous marginals + dependence a linear-Gaussian network cannot; other records only try the BN.
    all_continuous = all(isinstance(value, (float, np.floating)) for row in rows for value in row)
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
    if not np.isfinite(comp_bic):
        raise ValueError("automatic structure selection produced a non-finite independent-model BIC.")

    # Candidate dependence models are scored by BIC on the same data. The independent composite is
    # the baseline and wins ties; a failed or non-finite fit is an error, not a routing fallback.
    candidates: list[tuple[float, Any, str]] = []
    if net.edges():
        candidates.append((bayesian_network_bic(net, rows), net, "bayesian-network"))
    if all_continuous:
        from mixle.inference.copula_structure import copula_candidates

        candidates.extend(copula_candidates(rows, composite, comp_params, comp_bic, n_log, max_its, rng))
    if not candidates:
        return None, composite
    if any(not np.isfinite(bic) for bic, _, _ in candidates):
        raise ValueError("automatic structure selection produced a non-finite candidate BIC.")
    # A later, more complex candidate only displaces an earlier, simpler one when it wins by more
    # than floating-point noise; otherwise platform/BLAS variance could select it nondeterministically.
    best_bic, best_model, desc = candidates[0]
    for bic, model, name in candidates[1:]:
        if bic < best_bic - 1e-6 * max(1.0, abs(best_bic)):
            best_bic, best_model, desc = bic, model, name
    if best_bic >= comp_bic:
        return None, composite
    if out is not None:
        out.write("structure: %s dependence beats independent fields (BIC %.1f < %.1f)\n" % (desc, best_bic, comp_bic))
    return best_model, composite


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
    # The post-accumulation key pass every EM driver must run -- same omission as _engine_fused_step
    # had: without it, keyed (tied) parameters silently untie on any engine-kernel fit routed here.
    stats_dict: dict = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    # Activate the engine so device-aware M-steps (e.g. NeuralLeaf) follow its device. The context
    # wraps nested component estimates too (a MixtureEstimator's per-leaf estimate runs inside it).
    from mixle.engines.base import using_active_engine

    with using_active_engine(engine):
        return estimator.estimate(nobs, accumulator.value())


def _engine_fused_step(
    enc_data: Any,
    estimator: ParameterEstimator,
    prev_estimate: Any,
    engine: Any,
    fused_options: dict[str, Any] | None = None,
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
    if fused_options:
        for knob, value in fused_options.items():
            if hasattr(kernel, knob):  # only the FusedKernel carries the tuning attrs
                setattr(kernel, knob, value)
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
        # A pandas Series reaches the generic path (it has no .columns), but the family encoders
        # take sequences, and OptionalDataEncoder in particular refuses the Series wholesale -- so
        # optimize(series_with_missing) failed for every Optional-wrapped auto estimator while
        # get_estimator(series) had happily accepted the same input (campaign wave 2). Iterating a
        # Series yields its VALUES (never the index), preserving both missing spellings.
        # pd.NA must mean missing on the ENCODE side too, not only in profiling: optimize() infers
        # the estimator through get_estimator/normalize_input but encodes through this function, and
        # OptionalDataEncoder identifies missing rows by sentinel identity -- _same_sentinel(pd.NA,
        # None) is False -- so a profiler that said "missing" and an encoder still seeing pd.NA
        # would disagree about the same row (campaign three, T2-1).
        from mixle.data.sources.pandas_source import column_records, normalize_pandas_missing

        if type(data).__name__ == "Series" and type(data).__module__.startswith("pandas"):
            # column_records re-derives the column's OWN missing-value convention (NaN for a
            # numeric dtype, None otherwise) rather than normalizing pandas' sentinels value by
            # value -- the same fix normalize_input's Series branch applies on the profiling side,
            # so a model whose sentinel came from profiling meets records encoded the same way
            # (campaign four, T2-02, the Series half; the two sides disagreeing raised
            # ContractError on the whole batch, which is why this and the profiling change had to
            # land together).
            return column_records(data)
        if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
            try:
                return [normalize_pandas_missing(value) for value in data]
            except TypeError:
                return data
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


def _compiled_fused_step(enc_data, estimator, model, strategy):
    """Compiled full-mixture step plus its already-computed input-model objective."""

    result = strategy.step(enc_data, estimator, model)
    metadata = result.metadata or {}
    return result.model, metadata.get("input_data_objective")


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


def _initialize_with_support_fallback(
    *, enc_data: Any, estimator: ParameterEstimator, rng: np.random.RandomState, p: float
) -> SequenceEncodableProbabilityDistribution:
    """Seed EM, falling back to the full sample when the ``p`` subsample gives an unusable start.

    ``seq_initialize`` keeps each observation with probability ``p`` -- a HARD 0/1 Bernoulli mask, so
    an unselected observation is not down-weighted, it is deleted from every component's accumulator.
    For a discrete leaf that is not a harmless thinning: a category no draw happened to select gets
    exactly zero mass, and the seeded model then assigns ``-inf`` to every observation carrying it.
    At the default ``p=0.1`` on a 60-row sample with 32 distinct categorical levels, 37 of 60 rows
    scored ``-inf``; the dataset log-likelihood is ``-inf`` and EM cannot recover, because the E-step
    needs finite component densities to form responsibilities and the M-step therefore never sees the
    data that would fill the missing categories. The result was a hard failure ("fused EM did not
    produce a finite objective from its non-finite initial model") on entirely ordinary input.

    The subsample is not what makes the components differ -- ``MixtureAccumulator.seq_initialize``
    draws Dirichlet responsibilities per observation for that, and multiplies these weights in
    afterwards. So ``p`` buys diversity nothing, costs statistical efficiency, and is the sole cause
    of the missing-support failure. Rather than change ``p``'s documented meaning (and with it every
    seeded initialization in the wild), this keeps the requested draw and only repairs the case where
    it produced a model that cannot score its own training data: re-seed once from the full sample,
    which by construction covers every observed category.

    A model that assigns zero probability to the data it was initialized from is not a valid EM
    starting point under any reading, so falling back is a strict improvement, never a silent
    downgrade. The fallback is deterministic: it reuses ``rng``, so a given seed still yields one
    reproducible starting model.
    """
    model = seq_initialize(enc_data=enc_data, estimator=estimator, rng=rng, p=p)
    if p >= 1.0:
        return model
    try:
        _, objective = seq_log_density_sum(enc_data, model)
    except (TypeError, ValueError, AttributeError):
        # Not every encoded handle supports a bare scoring pass here (parallel/RDD handles score
        # through their own driver). Those keep exactly the previous behaviour.
        return model
    if np.isfinite(objective):
        return model
    return seq_initialize(enc_data=enc_data, estimator=estimator, rng=rng, p=1.0)


def _encoded_row_count(enc_data: Any) -> int | None:
    """Rows behind an encoded dataset, or ``None`` when the encoding does not report a count."""
    try:
        if isinstance(enc_data, list):
            return int(sum(int(chunk[0]) for chunk in enc_data))
        count = getattr(enc_data, "num_records", None)
        return None if count is None else int(count)
    except Exception:  # noqa: BLE001 - provenance must never break a fit that already succeeded
        return None


def _record_fit_provenance(
    model: SequenceEncodableProbabilityDistribution,
    trace: "_FitTrace",
    *,
    algorithm: str,
    estimator: ParameterEstimator,
    objective: str,
    max_its: int,
    delta: float | None,
    enc_data: Any,
    seed: int | None,
) -> SequenceEncodableProbabilityDistribution:
    """Attach a :class:`FitProvenance` describing the run that produced ``model``.

    Never raises: a fit that has already succeeded must not fail because its receipt could not be
    written. A model that reaches a caller without provenance is reported as ``None`` from
    ``fit_provenance()``, which is honest -- the alternative would be a fabricated receipt.
    """
    try:
        from mixle.stats.compute.pdist import FitProvenance

        # Repairs are applied by estimators deep inside the M-step, so the model is what knows about
        # them; the loop only knows about iterations. Merge both, deduplicated and ordered.
        model_repairs = tuple(model.numerical_repairs()) if hasattr(model, "numerical_repairs") else ()
        repairs = tuple(dict.fromkeys(tuple(trace.repairs) + model_repairs))
        model.with_fit_provenance(
            FitProvenance(
                algorithm=algorithm,
                estimator=type(estimator).__name__,
                objective=objective,
                iterations=int(trace.iterations),
                max_iterations=int(max_its),
                converged=bool(trace.converged),
                delta=None if delta is None else float(delta),
                final_objective=trace.final_objective,
                objective_gain=trace.objective_gain,
                last_accepted_objective=getattr(trace, "last_accepted_objective", None),
                n_observations=_encoded_row_count(enc_data),
                repairs=repairs,
                seed=seed,
            )
        )
    except Exception:  # noqa: BLE001 - see docstring
        pass
    return model


def _warn_if_capped_unconverged(trace: "_FitTrace", max_its: int, delta: float | None) -> None:
    """Disclose a convergence-seeking run that exhausted its iteration cap with the objective still moving.

    A latent-variable fit truncated at ``optimize``'s default ``max_its=10`` used to present exactly like
    a finished one unless the caller thought to read ``fit_provenance()`` (campaign T4-3); the flag was
    computed and then never spoken. Scope, checked against legitimate inputs this must NOT annoy:

    * ``delta=None`` is the documented "fixed iteration count" request -- a run that stops at its cap on
      purpose. No note.
    * surrogate-trained estimators run with the loop delta disabled, so their scheduled-budget fits are
      likewise silent here.
    * a run that stopped BELOW the cap on a rejected update (``iterations < max_its``) is a different
      condition -- more iterations would not help -- and is described by ``FitProvenance.converged``'s
      docstring rather than warned about, so an ordinary e.g. Weibull fit stays quiet.

    What remains is a run the caller asked to iterate to a gain below ``delta`` that never got there:
    warn once, with both remedies.
    """
    if delta is None or trace.converged or int(trace.iterations) < int(max_its):
        return
    gain = trace.objective_gain
    gain_text = ("last objective gain %.3g" % gain) if gain is not None else "final gain unknown"
    warnings.warn(
        "optimize() stopped at the max_its cap (%d) before the objective settled (%s, delta=%g): the "
        "returned model is an unconverged fit, and its fit_provenance() reports converged=False. "
        "Raise max_its to fit to convergence, or pass delta=None to request a fixed iteration count "
        "without this note." % (int(max_its), gain_text, float(delta)),
        UserWarning,
        stacklevel=3,
    )


class _FitTrace:
    """Mutable scratch the EM loop fills in so ``optimize`` can describe the run it just performed.

    A plain object rather than a return-value change: ``_em_loop`` has several exit paths and two
    callers, and widening its tuple would put the burden of threading a third element on code that has
    nothing to do with provenance.
    """

    __slots__ = ("iterations", "converged", "final_objective", "objective_gain", "last_accepted_objective", "repairs")

    def __init__(self) -> None:
        self.iterations = 0
        self.converged = False
        self.final_objective: float | None = None
        self.objective_gain: float | None = None
        # The trajectory's last ACCEPTED objective value, recorded when best-seen selection may
        # return an earlier iterate; the loop exit then rewrites final_objective to describe the
        # model actually returned (the FitProvenance docstring's contract; campaign T4-8).
        self.last_accepted_objective: float | None = None
        self.repairs: tuple[str, ...] = ()


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
    trace: "_FitTrace | None" = None,
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
            trace,
        )

    from mixle.inference.transaction import MutableStateSnapshot

    _, old_ll = ll_fn(enc_data, model)
    old_ll = float(old_ll)
    has_v = enc_vdata is not None
    best_vll = float(ll_fn(enc_vdata, model)[1]) if has_v else old_ll
    current_is_finite = bool(np.isfinite(old_ll))
    initial_is_selectable = current_is_finite and bool(np.isfinite(best_vll))
    best_model = model if initial_is_selectable else None
    best_state = MutableStateSnapshot.capture(model) if initial_is_selectable else None
    # The TRAINING objective of best_model, kept alongside it so the receipt can describe the model
    # actually returned. trace.final_objective tracks the trajectory's last accepted value, which
    # under validation-based selection or monotone=False belongs to a model that is NOT returned
    # (campaign T4-8); the exit below reconciles the two.
    best_ll = old_ll if initial_is_selectable else None

    for i in range(int(max_its)):
        transaction = MutableStateSnapshot.capture(model, estimator)
        proposal = step_fn(enc_data, estimator, model)
        nxt = getattr(proposal, "model", proposal)
        strategy_accepted = bool(getattr(proposal, "accepted", True))
        _, candidate_ll = ll_fn(enc_data, nxt)
        ll = float(candidate_ll)
        vll = float(ll_fn(enc_vdata, nxt)[1]) if has_v else ll
        had_finite_baseline = current_is_finite
        if had_finite_baseline:
            with np.errstate(invalid="ignore"):
                dll = ll - old_ll
        else:
            # An invalid initializer is not a best state and cannot be an ordering baseline. Permit
            # exactly the first finite candidate to establish the baseline, then resume the normal
            # monotonicity contract. If no finite candidate is produced, the run fails below.
            dll = float("inf") if np.isfinite(ll) else float("nan")

        # A non-finite step (e.g. a collapsed/singular covariance producing a NaN/-inf
        # log-likelihood) is never an improvement: never accept it, and do not let it
        # poison the convergence reference. A finite proposal may repair a non-finite
        # initializer; after that, finite ``ll`` follows the historical monotonicity guard.
        ll_finite = bool(np.isfinite(ll))
        accepted = strategy_accepted and ll_finite and (not had_finite_baseline or (dll >= -1.0e-12) or (not monotone))
        if accepted:
            model = nxt
            current_is_finite = True
        else:
            transaction.restore()

        # Best-model + reference update happen BEFORE the convergence break so the accepted step on the
        # converging iteration is recorded (otherwise an immediate convergence returns the stale initial
        # model). best-model selection is by validation score; record nxt (the model that achieved vll),
        # not model (unchanged on a rejected step) -- and never select a non-finite step.
        if accepted:
            old_ll = ll
            if track_best and np.isfinite(vll) and (best_model is None or best_vll < vll):
                best_vll = vll
                best_model = nxt
                best_state = MutableStateSnapshot.capture(best_model)
                best_ll = ll
            elif not track_best:
                best_vll = vll

        converged = accepted and had_finite_baseline and (delta is not None) and (0.0 <= dll < delta)
        if out is not None and (converged or (print_iter and (i + 1) % print_iter == 0)):
            _write_em_iter(out, i + 1, ll, dll, vll, has_v, obj_label)
        if on_step is not None:
            reported_ll = ll if accepted else old_ll
            reported_delta = dll if accepted else 0.0
            on_step(EMStep(i + 1, model, float(reported_ll), float(reported_delta)))
        if trace is not None:
            # Record on every iteration, not only at a clean exit: a run that stops because a step was
            # rejected still produced these numbers, and that is precisely the case a caller most needs
            # described (MXR-080-1190/1202).
            trace.iterations = i + 1
            trace.converged = bool(converged)
            trace.final_objective = float(ll) if accepted else trace.final_objective
            trace.objective_gain = float(dll) if accepted else 0.0
        if converged or (not accepted):
            break

    if not current_is_finite:
        raise ValueError("EM did not produce a finite objective from its non-finite initial model.")
    if track_best:
        if best_model is None or best_state is None:
            raise ValueError("EM did not produce a model with a finite validation objective.")
        best_state.restore()
        if trace is not None and best_ll is not None:
            # FitProvenance.final_objective is documented as the value of the RETURNED model. The
            # loop above tracked the trajectory's last accepted value; when selection returns an
            # earlier iterate (validation selection, monotone=False) that value belongs to a model
            # the caller never sees, so keep it under its own truthful name and let the receipt
            # describe best_model, which is what is being returned (campaign T4-8).
            trace.last_accepted_objective = trace.final_objective
            trace.final_objective = float(best_ll)
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
    trace=None,
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
    best_model = None
    best_score = None
    best_train_ll = None  # training LL of best_model, so the receipt can describe the returned model
    prev_ll = None
    accepted_model = None
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
        score = float(ll_fn(enc_vdata, model)[1]) if has_v else float(ll_model)
        if np.isfinite(score) and (best_score is None or score >= best_score):
            best_score = score
            best_model = model
            best_train_ll = float(ll_model)

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
        if trace is not None:
            trace.iterations = i + 1
            trace.converged = bool(converged)
            trace.final_objective = float(ll_model)
            trace.objective_gain = float(dll)
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
            score = float(ll_fn(enc_vdata, nxt)[1]) if has_v else float(final_ll)
            if np.isfinite(score) and (best_score is None or score >= best_score):
                best_score = score
                best_model = nxt
                best_train_ll = float(final_ll)
            if trace is not None:
                # The trajectory just advanced by one accepted step; without this a capped fused fit
                # reported the value of the model one step BEHIND the one it returned (same defect
                # class as T4-8: a receipt describing a model that was not returned).
                trace.final_objective = float(final_ll)
                if prev_ll is not None:
                    trace.objective_gain = float(final_ll - prev_ll)

    if accepted_model is None:
        raise ValueError("fused EM did not produce a finite objective from its non-finite initial model.")
    if track_best and best_model is None:
        raise ValueError("fused EM did not produce a model with a finite validation objective.")
    chosen = best_model if track_best else accepted_model
    if track_best and trace is not None and best_train_ll is not None:
        # As in _em_loop: FitProvenance.final_objective must describe the RETURNED model; the
        # trajectory's own last accepted value stays visible under its truthful name (T4-8).
        trace.last_accepted_objective = trace.final_objective
        trace.final_objective = float(best_train_ll)
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
_VALID_STRUCTURES = ("auto", "off")
_VALID_SCHEDULES = ("auto", "full")


def _validate_optimize_controls(
    *,
    max_its: Any,
    delta: Any,
    init_p: Any,
    print_iter: Any,
    objective: Any,
    structure: Any,
    schedule: Any,
    reuse_estep_ll: Any,
    monotone: Any,
    track_best: Any,
) -> None:
    """Fail closed on public optimizer controls before routing or initialization."""

    def is_bool(value: Any) -> bool:
        return isinstance(value, (bool, np.bool_))

    if is_bool(max_its) or not isinstance(max_its, (int, np.integer)) or int(max_its) < 1:
        raise ValueError(f"optimize(): max_its must be a positive integer, got {max_its!r}")
    if delta is not None and (
        is_bool(delta)
        or not isinstance(delta, (int, float, np.integer, np.floating))
        or not np.isfinite(delta)
        or float(delta) < 0.0
    ):
        raise ValueError(f"optimize(): delta must be None or a finite non-negative number, got {delta!r}")
    if (
        is_bool(init_p)
        or not isinstance(init_p, (int, float, np.integer, np.floating))
        or not np.isfinite(init_p)
        or not 0.0 < float(init_p) <= 1.0
    ):
        raise ValueError(f"optimize(): init_p must be finite and in (0, 1], got {init_p!r}")
    if is_bool(print_iter) or not isinstance(print_iter, (int, np.integer)) or int(print_iter) < 0:
        raise ValueError(f"optimize(): print_iter must be a non-negative integer, got {print_iter!r}")
    for name, value, choices in (
        ("objective", objective, _VALID_OBJECTIVES),
        ("structure", structure, _VALID_STRUCTURES),
        ("schedule", schedule, _VALID_SCHEDULES),
    ):
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"optimize(): {name} must be one of {choices!r}, got {value!r}")
    for name, value, optional in (
        ("reuse_estep_ll", reuse_estep_ll, False),
        ("monotone", monotone, True),
        ("track_best", track_best, True),
    ):
        if not (is_bool(value) or (optional and value is None)):
            expected = "None or a boolean" if optional else "a boolean"
            raise TypeError(f"optimize(): {name} must be {expected}, got {value!r}")


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
    # Prefer the explicit prior signal when it says yes: get_prior() is not None is robust even
    # when the log-prior happens to evaluate to 0.0 at init (which the model_log_density != 0.0
    # heuristic below would misclassify as MLE). But get_prior() is None only rules out a prior
    # the estimator ITSELF carries at its own (outer) level -- a compound estimator (e.g. a mixture
    # whose per-component estimators have priors but whose own weight prior does not, or an HMM
    # whose per-state topic estimators have priors but whose own chain prior does not) can have
    # get_prior() is None while model_log_density is still genuinely non-zero from a nested child's
    # prior, so a None get_prior() falls through to that check rather than concluding 'mle' outright.
    get_prior = getattr(estimator, "get_prior", None)
    if callable(get_prior) and get_prior() is not None:
        return "map"
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
    strategy: Any | None = None,
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
        return require_exact_bool(monotone, "monotone")

    from mixle.inference.em import MonteCarloEM, OnlineEM
    from mixle.inference.transaction import has_mutable_state

    if isinstance(strategy, (MonteCarloEM, OnlineEM)):
        # Both are genuinely stochastic by construction (MonteCarloEM samples latent completions;
        # OnlineEM is "decay-mode stochastic/online EM" per its own docstring) over an otherwise
        # ordinary immutable estimator/model that the three checks below would not catch -- the
        # strict gate previously defaulted to True here and broke the very first noisy step,
        # silently terminating the whole run after one iteration regardless of max_its.
        return False

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
        return require_exact_bool(track_best, "track_best")
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
    fused_options: dict[str, Any] | None = None,
) -> SequenceEncodableProbabilityDistribution:
    """Fit ``estimator`` to ``data`` by a generalized-EM loop, for ``max_its`` iterations or until the
        objective improves by less than ``delta``.

    Each iteration re-estimates every part of the model by whatever its structure calls for -- closed-form
    for conjugate / exponential-family leaves, gradient descent for neural leaves, coordinate descent for
    GLMs, responsibility-weighted EM for latent structure (mixtures, HMMs) -- so a single call fits a
    heterogeneous tree without the caller choosing an algorithm. (The convergence objective defaults to the
    family-defined maximum-likelihood objective: each leaf applies its own documented estimator update,
    which for some families is a closed-form moment update rather than an iterative likelihood
    maximization, and families may apply documented numerical floors/repairs -- read them back via
    ``model.numerical_repairs()``. A parameter prior switches the objective to penalized-LL / MAP, and a
    variational model to the ELBO -- see ``objective``.)

    **Missing values.** With ``estimator=None`` (the default) auto-inference finds the gaps and wraps
    the affected leaf for you: data carrying ``None``, ``NaN``, ``pd.NA`` or ``pd.NaT`` fits an
    ``OptionalDistribution`` whose missingness rate is estimated alongside the base family. When you
    build the estimator yourself, that wrapper is yours to add, and it must name the sentinel your
    data actually carries -- the wrapper matches its ``missing_value`` by identity, so the ``None``
    default does not accept ``NaN``::

        optimize(rows, CompositeEstimator([OptionalEstimator(GaussianEstimator(),
                                                             missing_value=float("nan")),
                                           CategoricalEstimator()]))

    ``NaN`` is the spelling float arrays and numeric pandas columns use, so
    ``missing_value=float('nan')`` is the one that works for tabular numeric data;
    ``missing_value=None`` (the default) is for records that carry ``None``.
    :func:`mixle.stats.marginalized` is the same wrapper with no fitted rate, but it takes a
    DISTRIBUTION rather than an estimator and its default sentinel is not ``NaN``, so ``NaN`` data
    must spell out ``marginalized(dist, missing_value=float('nan'))``.

    **DataFrames.** ``optimize`` and ``fit`` take a pandas DataFrame or Series directly; the
    row-level stats API does not. ``mixle.stats.seq_encode`` consumes RECORDS and refuses a frame
    ("expected a sequence of 2-tuples, got DataFrame") -- converting is the caller's job, in one
    call: ``mixle.data.dataframe_records(df)`` for the records, or
    ``mixle.data.seq_encode_dataframe(df, model=...)``, which is the frame-shaped counterpart of
    ``seq_encode``. A frame's gaps are canonicalized on the way in by what each column HOLDS --
    ``NaN`` for a column of numbers, ``None`` for anything else -- so ``df`` and
    ``df.convert_dtypes()`` fit models carrying the same ``missing_value``, and each model scores
    the other frame.

    Args:
        data (Optional[List[T]]): List of data type T containing observed data. Must be compatible with data type of
            estimator.
        estimator (ParameterEstimator | ProbabilityDistribution | None): What to fit. A ``ParameterEstimator``
            is used as-is; a distribution **prototype** (any ``ProbabilityDistribution``) is coerced to its
            matching estimator via ``proto.estimator()`` so you build the model shape only once; ``None``
            infers an estimator from raw ``data`` (``mixle.utils.automatic.get_estimator``).
            Inferring from rows that do not all carry the same number of fields is ambiguous between a
            table with a malformed row and variable-length sequence data, so it is decided on the arity
            evidence rather than guessed: an overwhelmingly dominant arity raises ``ContractError``
            naming the offending row, a bare majority is fitted as a sequence and warns that it did,
            and widely spread arities are fitted as a sequence in silence. Pass
            ``mixle.utils.automatic.get_estimator(data, ragged='sequence')`` as the ``estimator`` to
            demand the sequence reading outright.
        max_its (int): Maximum number of EM iterations to be performed. Default value is 10 iterations.
        delta (Optional[float]): Stopping criteria for EM algorithm used if max_its is not set: Iterate until
            ``abs(old_loglikelihood - new_loglikelihood) < delta`` or iterations == max_its.
        init_estimator (Optional[ParameterEstimator]): ParameterEstimator to used to initialize EM algorithm parameters.
            If None, estimator is used. Must be consistent with estimator.
        init_p (float): Value in (0.0,1.0] for randomizing the proportion of data points used in initialization.
            This is a statistical knob, not an execution detail: the initialization is the estimator's
            answer on the drawn subsample, and for latent models and for families whose documented
            update is approximate rather than an exact likelihood ascent (e.g. Weibull,
            GeneralizedExtremeValue, WrappedCauchy) the accepted EM trajectory -- and therefore the
            returned parameters -- depends on that start. In the extreme, such a family's very first
            update can fail the monotone acceptance gate, and the fit then returns the initialization
            itself (``fit_provenance()`` reports ``converged=False`` with ``iterations`` below the
            cap). ``init_p=1.0`` initializes from the estimator's answer on all the data.
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
        num_chunks (int): Number of chunks for encoded data. For exact-sufficient-statistic leaves the
            chunk statistics combine exactly, so chunking changes only float summation order.
            Chunking also changes WHICH rows the ``init_p`` initialization subsample draws, however,
            so for the initialization-sensitive families described under ``init_p`` a different
            ``num_chunks`` can change the fitted parameters, not just the execution plan -- measured
            at the percent level on Weibull/GEV/WrappedCauchy fits. When varying it, compare
            ``fit_provenance().final_objective`` (or hold the start fixed with ``init_p=1.0``).
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
            ``HardEM``, ``MonteCarloEM``, ``CompiledEM``) or any callable ``(enc, estimator, model) -> model`` to use
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
            log-likelihood (``'map'``), and everything else by maximum likelihood (``'mle'``).
            ``'mle'`` is the family-defined maximum-likelihood objective: what is guaranteed is
            each leaf family's documented estimator update -- exact MLE for exponential-family
            leaves, a documented closed-form moment update for families whose class docstring says
            so -- and families may apply documented numerical floors/repairs, reported on the
            fitted model via ``numerical_repairs()``.
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
            the historical automatic-composite path proceeds untouched. A pandas DataFrame is
            converted to the same flat records first (honoring ``fields``), so a table gets the
            same structure inference whether it arrives as a DataFrame or as a list of row tuples.
            ``'off'`` restores the unconditional historical behavior. Only consulted when
            ``estimator`` is ``None`` and no
            ``prev_estimate``/``init_estimator``/``strategy``/``enc_data`` is supplied.
        schedule (str): ``'full'`` (default) performs exact full-tree EM every round and automatically
            uses whole-model or component-level compiled kernels when the local model is eligible.
            ``'auto'`` additionally engages the block-coordinate-ascent scheduler
            (:mod:`mixle.inference.block_em`) when selective work is expected to win: after one full
            bootstrap sweep, components are ranked by their last observed complete-data Q gain per
            measured whole-block cost, and only the highest-value ones within a per-round budget are
            re-estimated. Inactive component scores are cached while their parameters remain unchanged.
            This is a
            scheduling choice: observed likelihood is transactionally gated non-decreasing every round, and
            when there is no useful ranking to do (e.g. every component looks equally worth
            updating) the scheduler degenerates to doing exactly what ``'full'`` does. Only
            engaged when the model is a local-backend ``MixtureDistribution``/
            ``MixtureEstimator`` MLE/MAP fit with no explicit ``strategy``/``engine``/``resources``/
            ``placement`` -- anything else silently falls back to ``'full'`` (never an error, never
            a behavior change beyond scheduling).
        seed (Optional[int]): Integer seed for initializing the EM algorithm -- shorthand for
            ``rng=RandomState(seed)``, matching the ``seed=`` argument the samplers and the other
            entry points take. Mutually exclusive with ``rng`` (passing both raises ``TypeError``).
        fused_options (Optional[dict]): Tuning knobs for the fused compiled kernels, applied when the
            fit runs on a fused engine (an explicit fused ``engine=`` or the auto-fusion gate).
            Recognized keys: ``parallel`` (bool -- force the chunk-parallel kernels on or off,
            overriding the observation-count auto-gate; results are bit-stable either way),
            ``lse_bits`` (int -- opt SCORING into quantized log-sum-exp with a ~2**-bits relative
            bound; E-steps always stay exact), and ``lse_span`` (float, default 24.0 -- the clipped
            LSE exponent range used with ``lse_bits``). Unknown keys raise ``ValueError``. Ignored
            (with the same validation) when the fit resolves to a non-fused path.

    Returns:
        SequenceEncodableProbabilityDistribution: the fitted model. The run behind it ended in one of
            three ways -- the objective gain fell below ``delta`` (a converged fit), the ``max_its``
            cap was reached with the objective still improving (an unconverged fit; a ``UserWarning``
            says so when ``delta`` is in force), or the family's next update failed the monotone
            acceptance gate before the cap (no further progress is possible; see ``init_p``). The
            receipt distinguishing them ships on the model: ``model.fit_provenance()`` reports
            ``converged``, ``iterations``, ``final_objective`` (the returned model's objective
            value), and any numerical ``repairs``.

    **Model comparison.** A fitted core distribution intentionally carries no ``aic()``/``bic()`` --
    the core cannot count free parameters exactly for every composed model. The supported paths:
    per-observation log-densities (``model.log_density(x)``) feed the paired comparison tests
    exported by :mod:`mixle.inference` (:func:`vuong_test`, :func:`clarke_test`,
    :func:`paired_score_difference`, :func:`compare_elpd`), and the :mod:`mixle.ppl` surface fits
    the same families with a recorded parameter dimension, giving ``m.aic(data)`` / ``m.bic(data)``
    and best-first ranking via ``mixle.ppl.compare([m1, m2], data, by='bic')``.

    """
    _validate_optimize_controls(
        max_its=max_its,
        delta=delta,
        init_p=init_p,
        print_iter=print_iter,
        objective=objective,
        structure=structure,
        schedule=schedule,
        reuse_estep_ll=reuse_estep_ll,
        monotone=monotone,
        track_best=track_best,
    )
    rng = _resolve_rng_arg(rng, seed)
    if fused_options is not None:
        unknown = set(fused_options) - {"parallel", "lse_bits", "lse_span"}
        if unknown:
            raise ValueError(
                f"optimize(fused_options=...): unknown keys {sorted(unknown)}; expected a subset of ['lse_bits', 'lse_span', 'parallel']"
            )
    if (
        estimator is None
        and structure == "auto"
        and data is not None
        and enc_data is None
        and prev_estimate is None
        and init_estimator is None
        and strategy is None
    ):
        structure_rows = data
        if _dataframe_like(data):
            # A DataFrame silently bypassed structure='auto': iterating a DataFrame yields its
            # column NAMES, so _maybe_structured_model saw a few strings and declined, and the same
            # table that routes to a dependence-aware model as a list of records fit as an
            # independent composite instead -- silent and answer-changing (campaign T2-09b).
            # Convert to exactly the flat records the encoding path fits, so both spellings of the
            # same table share one structure inference.
            structure_rows = _data_records_for_encoding(data, fields, None, None)
        structured, independent_composite = _maybe_structured_model(
            structure_rows,
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
        # A converted DataFrame (structure_rows is not data) refits through the normal path: its
        # composite candidate was fit on the converted records via get_estimator(records), and this
        # close to release we do not bet the answer on that matching get_estimator(DataFrame).
        if (
            independent_composite is not None
            and structure_rows is data
            and (
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
            )
        ):
            return independent_composite
    estimator = _coerce_estimator(estimator, data, fields=fields)
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
            if plan.reduced() and engine is None:
                from mixle.engines import NumpyEngine

                engine = NumpyEngine(dtype=plan.compute_dtype, prefer_fused=True)
            actual_dtype = np.float64 if engine is None else getattr(engine, "dtype", np.float64)
            fallback = (
                "explicit_engine_override"
                if engine is not None and np.dtype(actual_dtype) != np.dtype(plan.compute_dtype)
                else None
            )
            plan.record_execution(actual_dtype, fallback=fallback)
            _record_precision_plan(estimator, plan, out)
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

    if backend is None:
        backend_name = "local"
    else:
        backend_name = str(backend).lower()
        if not backend_name.strip():
            # `backend or "local"` silently ran an empty string locally -- indistinguishable from
            # a config variable that failed to resolve. Only None means "the default"; an explicit
            # empty name is refused like any unknown backend (campaign wave 2, T4-9).
            from mixle.utils.parallel.planner import available_encoded_data_backends

            raise ValueError(
                "optimize() backend must be a non-empty backend name (or None for the local "
                "default); registered backends: %s" % ", ".join(available_encoded_data_backends())
            )
    if data is None and enc_data is None and not (backend_name == "mpi" and root_only):
        raise ValueError(
            "optimize() received no observations: data and enc_data are both None. "
            "Pass a non-empty data sequence or pre-encoded enc_data."
        )
    # Empty (but non-None) data previously slipped through and silently returned the initialized
    # prior/default model -- a wrong answer, not a fit. The message names the entry the caller
    # actually used (fit() raises its own before forwarding here) and says "no observations", the
    # one spelling both empty-input paths share.
    if data is not None and enc_data is None and hasattr(data, "__len__") and len(data) == 0:
        raise ValueError("optimize() received no observations: data is empty. Pass a non-empty data sequence.")
    if enc_data is None:
        _reject_masked_data(data, "optimize()")
        if not _estimator_carries_prior(estimator if init_estimator is None else init_estimator):
            _reject_all_zero_observation_weights(data, "optimize()")

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
            mm = _initialize_with_support_fallback(enc_data=enc_data, estimator=est, rng=rng, p=float(init_p))
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
            if plan.reduced() and engine is None:
                from mixle.engines import NumpyEngine

                engine = NumpyEngine(dtype=plan.compute_dtype, prefer_fused=True)
            actual_dtype = np.float64 if engine is None else getattr(engine, "dtype", np.float64)
            fallback = (
                "explicit_engine_override"
                if engine is not None and np.dtype(actual_dtype) != np.dtype(plan.compute_dtype)
                else None
            )
            plan.record_execution(actual_dtype, fallback=fallback)
            _record_precision_plan(estimator, plan, out)

        if enc_vdata is None and vdata is not None:
            vdata_for_encoding = _data_records_for_encoding(vdata, fields, est, mm)
            enc_vdata = seq_encode(vdata_for_encoding, data_encoder, num_chunks=num_chunks, chunk_size=chunk_size)

        # The prior is the single switch: 'auto' uses the variational ELBO when the model exposes
        # one (seq_local_elbo), the penalized log-likelihood when the estimator carries a prior, and
        # the plain log-likelihood otherwise. So a Bayesian estimator converges/selects on the right
        # objective whether the caller reaches for optimize() or fit().
        resolved_objective = _resolve_objective(objective, estimator, mm)
        surrogate_update = _contains_surrogate_update(estimator)
        strict_monotone = _resolve_monotone(monotone, estimator, mm, strategy)
        select_best = _resolve_track_best(track_best, estimator)
        loop_delta = None if surrogate_update else delta
        from mixle.inference.transaction import has_mutable_state

        compiled_full = False
        if (
            schedule in ("auto", "full")
            and strategy is None
            and resolved_objective == "mle"
            and engine is None
            and resources is None
            and placement is None
            and backend_name == "local"
            and not has_mutable_state(mm, estimator)
        ):
            from mixle.inference.fusion_policy import prefer_compiled_mixture

            compiled_full = prefer_compiled_mixture(mm, enc_data, max_its)

        # D3 block-EM scheduler: 'auto' dispatches to mixle.inference.block_em's greedy
        # gain-per-cost scheduler when the fit is a local MLE/MAP MixtureDistribution/
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
                and resolved_objective in ("mle", "map")
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
                block_trace = _FitTrace()
                block_trace.iterations = len(block_history)
                block_trace.final_objective = float(block_history[-1].objective) if block_history else None
                # Block EM stops on its own convergence test; it reports rounds, not a delta, so
                # "converged" is "it stopped before the cap" rather than an asserted gain threshold.
                block_trace.converged = bool(block_history) and len(block_history) < max_its
                _warn_if_capped_unconverged(block_trace, max_its, delta)
                return _record_fit_provenance(
                    best_model,
                    block_trace,
                    algorithm="block-em",
                    estimator=estimator,
                    objective=resolved_objective,
                    max_its=max_its,
                    delta=delta,
                    enc_data=enc_data,
                    seed=seed,
                )

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
        effective_strategy = strategy
        if compiled_full:
            from mixle.inference.em import CompiledEM

            effective_strategy = CompiledEM()
            if out is not None:
                out.write("compiled-em: component-level fused full-tree execution\n")

        if (
            reuse_estep_ll
            and strict_monotone
            and resolved_objective == "mle"
            and (strategy is None or compiled_full)
            and isinstance(enc_data, list)
            and not has_mutable_state(mm, estimator)
        ):
            if compiled_full:
                fused_step_fn = partial(_compiled_fused_step, strategy=effective_strategy)
            elif engine is None:
                fused_step_fn = _local_fused_step
            else:
                fused_step_fn = partial(_engine_fused_step, engine=engine, fused_options=fused_options)

        objective_scorer = _objective_scorer(resolved_objective, estimator, engine)
        objective_fn = lambda candidate: objective_scorer(enc_data, candidate)[1]
        trace = _FitTrace()
        best_model, _ = _em_loop(
            enc_data,
            estimator,
            mm,
            step_fn=_em_step_fn(engine, effective_strategy, objective_fn),
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
            trace=trace,
        )

        _warn_if_capped_unconverged(trace, max_its, loop_delta)
        return _record_fit_provenance(
            best_model,
            trace,
            algorithm="fused-em" if fused_step_fn is not None else "em",
            estimator=estimator,
            objective=resolved_objective,
            max_its=max_its,
            delta=loop_delta,
            enc_data=enc_data,
            seed=seed,
        )
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
      - ``'mle'`` -- the family-defined maximum-likelihood objective on the data log-likelihood (ignores
        any prior in the objective). What is guaranteed is each leaf family's documented estimator
        update -- some families document a closed-form moment update rather than an iterative
        likelihood maximization -- and families may apply documented numerical floors/repairs; read
        them back via ``model.numerical_repairs()``;
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
        structure_rows = data
        if _dataframe_like(data):
            # Same repair as optimize's front door (campaign T2-09b): a DataFrame iterates as its
            # column names, which silently skipped structure inference for exactly the tabular data
            # it exists for. Fit structure on the same flat records the encoding path will fit.
            structure_rows = _data_records_for_encoding(data, kwargs.get("fields"), None, None)
        structured, independent_composite = _maybe_structured_model(
            structure_rows,
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
        if (
            independent_composite is not None
            and structure_rows is data  # a converted DataFrame refits via the normal path (see optimize)
            and kwargs.get("out") is None
            and not (set(kwargs) - _threaded_or_inert)
        ):
            return independent_composite
    estimator = _coerce_estimator(estimator, data, fields=kwargs.get("fields"))
    if init_estimator is not None:
        init_estimator = _coerce_estimator(init_estimator, data)
    if data is None and kwargs.get("enc_data") is None:
        raise ValueError(
            "fit() received no observations: data and enc_data are both None. "
            "Pass a non-empty data sequence or pre-encoded enc_data."
        )
    # Raised here rather than in the forwarded optimize() call so the error names the entry point
    # the user actually called (same condition, same "no observations" spelling).
    if data is not None and kwargs.get("enc_data") is None and hasattr(data, "__len__") and len(data) == 0:
        raise ValueError("fit() received no observations: data is empty. Pass a non-empty data sequence.")
    if kwargs.get("enc_data") is None:
        _reject_masked_data(data, "fit()")
        if not _estimator_carries_prior(estimator if init_estimator is None else init_estimator):
            _reject_all_zero_observation_weights(data, "fit()")
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
    fused_options: dict[str, Any] | None = None,
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
        delta (float): Stopping criteria for EM when ``abs(old-log-likelihood - new-log-likelihood) < delta``.
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
        fused_options (Optional[dict]): Fused-kernel tuning knobs forwarded to every trial's
            ``optimize`` call -- see ``optimize(fused_options=...)`` for the recognized keys.

    Returns:
        Tuple of log-likelihood of best fitting model and the best fitting model from number of trials.

    """
    rng = _resolve_rng_arg(rng, seed)
    if data is None and enc_data is None:
        raise ValueError(
            "best_of() received no observations: data and enc_data are both None. "
            "Pass a non-empty data sequence or pre-encoded enc_data."
        )
    if data is not None and enc_data is None and hasattr(data, "__len__") and len(data) == 0:
        raise ValueError("best_of() received no observations: data is empty. Pass a non-empty data sequence.")
    if enc_data is None:
        _reject_masked_data(data, "best_of()")
        if not _estimator_carries_prior(est if init_estimator is None else init_estimator):
            _reject_all_zero_observation_weights(data, "best_of()")

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
            fused_options=fused_options,
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
    if not isinstance(rho, (int, float, np.integer, np.floating)) or isinstance(rho, (bool, np.bool_)):
        raise TypeError("constant(rho) requires a numeric rho.")
    if not np.isfinite(rho) or rho <= 0.0 or rho > 1.0:
        raise ValueError("constant(rho) requires 0 < rho <= 1.")

    def schedule(t: int) -> float:
        return float(rho)

    return schedule


def harmonic(alpha: float, offset: float = 1.0):
    """Return ``rho_t = (offset + t - 1)^(-alpha)`` for streaming EM."""
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or isinstance(alpha, (bool, np.bool_)):
        raise TypeError("harmonic(alpha) requires a numeric alpha.")
    if not isinstance(offset, (int, float, np.integer, np.floating)) or isinstance(offset, (bool, np.bool_)):
        raise TypeError("harmonic(offset) requires a numeric offset.")
    if not np.isfinite(alpha) or alpha <= 0.5 or alpha > 1.0:
        raise ValueError("harmonic(alpha) requires 0.5 < alpha <= 1.0.")
    if not np.isfinite(offset) or offset <= 0.0:
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

    ``rho`` in ``(0, 1]`` discounts the sufficient statistics retained from PREVIOUS batches before
    adding the next batch at full weight, so older evidence decays geometrically. ``rho=1`` recovers
    ordinary un-forgotten accumulation.
    """
    if not isinstance(rho, (int, float, np.integer, np.floating)) or isinstance(rho, (bool, np.bool_)):
        raise TypeError("forgetting(rho) requires a numeric rho.")
    if not np.isfinite(rho) or rho <= 0.0 or rho > 1.0:
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

    ``mode='forgetting'`` retains a running sufficient-statistic accumulator. Before each batch after
    the first, the retained OLD statistics and effective observation count are scaled by
    ``rho = schedule(step)``; the new batch is then added at full weight. Estimation always uses the
    original estimator/prior, avoiding both discounting new evidence and double-counting a carried
    posterior. Accumulator scaling preserves structural support metadata such as categorical bounds.

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
        resolved_mode = posterior_carry() if mode is None else mode
        if resolved_mode not in ("posterior_carry", "forgetting"):
            raise ValueError("mode must be 'posterior_carry' or 'forgetting'.")
        if resolved_mode == "posterior_carry" and schedule is not None:
            raise ValueError("schedule is only valid when mode='forgetting'.")
        if resolved_mode == "forgetting" and schedule is not None and not callable(schedule):
            raise TypeError("forgetting schedule must be callable.")
        if (
            isinstance(init_p, (bool, np.bool_))
            or not isinstance(init_p, (int, float, np.integer, np.floating))
            or not np.isfinite(init_p)
            or not 0.0 < float(init_p) <= 1.0
        ):
            raise ValueError("init_p must be finite and in (0, 1].")
        if (
            isinstance(num_chunks, (bool, np.bool_))
            or not isinstance(num_chunks, (int, np.integer))
            or int(num_chunks) < 1
        ):
            raise ValueError("num_chunks must be a positive integer.")

        self._original_estimator = estimator
        self._original_init_estimator = estimator if init_estimator is None else init_estimator
        self.estimator = estimator
        self.init_estimator = self._original_init_estimator
        self.mode = resolved_mode
        self.schedule = schedule
        if self.mode == "forgetting" and self.schedule is None:
            self.schedule = forgetting(1.0)
        self.model = model
        self.init_p = float(init_p)
        self.rng = (
            RandomState(0) if rng is None else rng
        )  # fixed default: the numpy side of an un-seeded fit is deterministic
        self._initial_rng_state = self.rng.get_state()
        self.num_chunks = int(num_chunks)
        self.step = 0
        self.nobs = 0.0
        self._history_accumulator = None
        if model is not None and self.mode == "posterior_carry":
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

    def _ensure_model(self, data: Sequence[T] | None, enc_data: Any | None) -> Any | None:
        if self.model is not None:
            return None
        if enc_data is None and data is None:
            raise ValueError("BayesianStreamingEstimator.update requires data for initialization.")
        enc = enc_data if enc_data is not None else self._encode(data)
        self.model = seq_initialize(enc_data=enc, estimator=self.init_estimator, rng=self.rng, p=self.init_p)
        return enc

    def _encode(self, data: Sequence[T]) -> Any:
        encoder = _resolve_encoder(self.init_estimator) if self.model is None else self.model.dist_to_encoder()
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
        initialization_batch = self._ensure_model(data, enc_data)
        enc_batch = initialization_batch if initialization_batch is not None else self._encode_batch(data, enc_data)
        batch_nobs, accumulator = _stream_accumulate(enc_batch, self.estimator, self.model)

        if self.mode == "forgetting":
            rho = float(self.schedule(self.step + 1))
            if not np.isfinite(rho) or rho <= 0.0 or rho > 1.0:
                raise ValueError("forgetting schedule returned %r; expected 0 < rho <= 1." % rho)
            candidate_history = self._original_estimator.accumulator_factory().make()
            if self._history_accumulator is None:
                candidate_history.from_value(deepcopy(accumulator.value()))
                effective_nobs = batch_nobs
            else:
                candidate_history.from_value(deepcopy(self._history_accumulator.value()))
                candidate_history.scale(rho)
                candidate_history.combine(accumulator.value())
                effective_nobs = rho * self.nobs + batch_nobs
            candidate_model = self._original_estimator.estimate(
                effective_nobs,
                candidate_history.value(),
            )
            self._history_accumulator = candidate_history
            self.model = candidate_model
            self.estimator = self._original_estimator
            self.nobs = effective_nobs
        else:
            self.model = self.estimator.estimate(batch_nobs, accumulator.value())
            self._carry_prior_from(self.model)
            self.nobs += batch_nobs
        self.step += 1
        return self.model

    def reset(self) -> None:
        """Restore the original estimator, RNG state, and empty stream state."""
        self.estimator = self._original_estimator
        self.init_estimator = self._original_init_estimator
        self.model = None
        self.step = 0
        self.nobs = 0.0
        self._history_accumulator = None
        self.rng.set_state(self._initial_rng_state)
