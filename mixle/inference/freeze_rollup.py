"""Freeze/roll-up cache -- per-datum log-density caching over frozen subtrees (workstream D2).

Frame (see the ConditionalJIT track, D1-D6): **the estimator tree is an IR**. D1
(:mod:`mixle.inference.node_report`) instruments every node with a per-round residual/Q-gain
report and an ``update_kind`` classification. D2 spends that report: once a subtree's D1 report
says it has stopped changing (a structurally ``"frozen"`` node, or a converged residual/Q-gain),
this module skips recomputing its per-datum log-density on every subsequent E-step round and
instead reuses a cached array -- turning E-step cost from ``O(full tree)`` into
``O(active fraction)``.

Correctness backbone (unchanged from the rest of the D-track): freeze/roll-up is a SCHEDULING
optimization only. It never changes what is computed, only how often -- a cached per-datum
log-density is byte-identical to a freshly recomputed one for the SAME parameters, and the cache
is invalidated (recomputed) the instant a subtree's parameters move again. This module tracks the
observed-data log likelihood from :func:`mixle.inference.em.observed_log_likelihood`; that directly
computed objective is the audit receipt that the cache never silently drifted from a real EM trajectory.

Scope: this module targets :class:`mixle.stats.latent.mixture.MixtureDistribution`, the
combinator with the clearest "some subtrees stop mattering" story (a component whose mixture
weight collapses near zero stops moving under EM -- its sufficient-statistic contribution is
~0 -- and stays frozen once collapsed). The freeze/roll-up mechanics (cache, invalidation,
D1-driven detection, active-fraction accounting) generalize to any combinator whose E-step
already produces one per-datum log-density array per child (Composite/Sequence), but a single,
well-tested combinator is the honest S/M-effort scope for this item; later track items
(D3's scheduler) are expected to widen it.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.node_report import node_report
from mixle.inference.transaction import MutableStateSnapshot
from mixle.stats.latent.mixture import MixtureDistribution, MixtureEstimator, _component_enc

_DEFAULT_Q_GAIN_TOL = 1.0e-6
_DEFAULT_WEIGHT_TOL = 1.0e-4
_DEFAULT_WEIGHT_DELTA_TOL = 1.0e-8
_DEFAULT_FREEZE_PATIENCE = 3
_DEFAULT_ACCEPT_TOLERANCE = 1.0e-9


def _param_signature(dist: Any) -> bytes:
    """Recursive structural fingerprint of a distribution subtree's parameters.

    Used to decide whether a cached per-datum log-density array is still valid: if a frozen
    node's owning subtree parameters have moved since the array was cached, the signature no
    longer matches and the cache is invalidated. Unlike the former shallow attribute walk, this
    includes nested combinators and torch-like modules. Module tensors use their identity and
    in-place version counter, so the check is O(number of parameter tensors), not O(parameter
    bytes); optimizer and ``load_state_dict`` updates advance those counters.
    """

    digest = hashlib.blake2b(digest_size=20)
    seen: set[int] = set()

    def add(value: Any) -> None:
        if value is None or isinstance(value, (str, bytes, bytearray, int, float, complex, bool)):
            digest.update(repr(value).encode("utf-8"))
            return
        ident = id(value)
        if ident in seen:
            digest.update(b"<seen>")
            return
        seen.add(ident)

        if isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(value.shape).encode("ascii"))
            digest.update(value.tobytes())
            return

        named_parameters = getattr(value, "named_parameters", None)
        named_buffers = getattr(value, "named_buffers", None)
        if callable(named_parameters) and callable(named_buffers):
            digest.update(type(value).__qualname__.encode("utf-8"))
            for name, tensor in list(named_parameters()) + list(named_buffers()):
                digest.update(name.encode("utf-8"))
                digest.update(repr(tuple(tensor.shape)).encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tensor.device).encode("ascii"))
                digest.update(str(getattr(tensor, "_version", 0)).encode("ascii"))
                try:
                    digest.update(str(tensor.data_ptr()).encode("ascii"))
                except RuntimeError:
                    digest.update(str(id(tensor)).encode("ascii"))
            return

        if isinstance(value, dict):
            for key in sorted(value, key=repr):
                add(key)
                add(value[key])
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                add(child)
            return
        if hasattr(value, "__dict__"):
            digest.update(type(value).__qualname__.encode("utf-8"))
            for name, child in sorted(vars(value).items()):
                if not name.startswith("_"):
                    digest.update(name.encode("utf-8"))
                    add(child)

    add(dist)
    return digest.digest()


@dataclass
class _CacheEntry:
    signature: bytes
    log_density: np.ndarray


@dataclass(frozen=True)
class DensityMatrixProfile:
    """Observed cache/evaluation work for one component-matrix construction."""

    evaluations: int
    cache_hits: int
    reused_columns: int
    zero_weight_skips: int
    evaluation_seconds: float
    cache_hit_seconds: float
    assembly_seconds: float
    component_evaluation_seconds: tuple[tuple[int, float], ...]

    @property
    def elapsed_seconds(self) -> float:
        return self.evaluation_seconds + self.cache_hit_seconds + self.assembly_seconds


@dataclass
class FreezeRollupStats:
    """One round's accounting for the freeze/roll-up cache -- the acceptance-criteria receipt.

    ``n_log_density_evals`` is the model-work receipt this module actually optimizes: the count of
    real ``component.seq_log_density(...)`` calls issued this round (each ``O(nobs)``), as opposed
    to a cache hit (``O(1)``, a dict lookup + signature compare). ``objective`` is the directly
    evaluated, transactionally monotone observed-data log likelihood for this round.
    """

    round_index: int
    n_components: int
    n_active: int
    n_frozen: int
    n_zero_weight: int
    n_log_density_evals: int
    objective: float
    accepted: bool = True

    @property
    def active_fraction(self) -> float:
        """Fraction of components genuinely (re-)evaluated this round, out of all components."""
        return self.n_active / self.n_components if self.n_components else 1.0


class FreezeRollupCache:
    """Per-mixture-component cache of per-datum log-density, keyed by component index.

    A component is eligible for caching once :func:`mixle.inference.node_report.node_report`
    reports it as D1-``"frozen"`` (the ``Neutral`` capability) or as having a converged residual/
    Q-gain (``abs(q_gain) < q_gain_tol``) sustained for ``freeze_patience`` consecutive rounds
    while its mixture weight has collapsed below ``weight_tol`` (the natural mixture-specific
    trigger: a near-zero-weight component's data-weighted M-step contribution is ~0, so its
    params -- and hence its residual/Q-gain -- stop moving). Once frozen, the caller (see
    :func:`run_em_freeze_rollup`) also skips that component's M-step entirely and carries its
    model object forward unchanged, so the cached signature can never go stale on its own; the
    cache only invalidates if a caller explicitly mutates/re-estimates a "frozen" component
    (:meth:`invalidate`) or the component's own parameter signature is found to have moved (a
    belt-and-suspenders check on every lookup, not just a documented invariant).
    """

    def __init__(
        self,
        *,
        q_gain_tol: float = _DEFAULT_Q_GAIN_TOL,
        weight_tol: float = _DEFAULT_WEIGHT_TOL,
        weight_delta_tol: float = _DEFAULT_WEIGHT_DELTA_TOL,
        freeze_patience: int = _DEFAULT_FREEZE_PATIENCE,
    ) -> None:
        self.q_gain_tol = float(q_gain_tol)
        self.weight_tol = float(weight_tol)
        self.weight_delta_tol = float(weight_delta_tol)
        self.freeze_patience = max(1, int(freeze_patience))
        self._entries: dict[int, _CacheEntry] = {}
        self._prev_residual: dict[int, float] = {}
        self._prev_weight: dict[int, float] = {}
        self._frozen_streak: dict[int, int] = {}

    def invalidate(self, idx: int | None = None) -> None:
        """Drop a cached entry (``idx=None`` clears the whole cache and freeze streaks).

        Call this if a caller explicitly re-triggers a component the cache had frozen (e.g. a
        later scheduler decides to re-activate it): the next lookup is guaranteed to recompute
        rather than silently return the stale array.
        """
        if idx is None:
            self._entries.clear()
            self._frozen_streak.clear()
            self._prev_residual.clear()
            self._prev_weight.clear()
        else:
            self._entries.pop(idx, None)
            self._frozen_streak.pop(idx, None)
            self._prev_residual.pop(idx, None)
            self._prev_weight.pop(idx, None)

    def is_frozen(self, idx: int, component: Any, weight: float) -> bool:
        """Return the D1-driven freeze verdict for component ``idx`` this round.

        Reads a fresh :class:`~mixle.inference.node_report.NodeReport` every round (cheap: a
        small Monte-Carlo self-residual, not a real-data pass) so a component that later moves
        again (unfrozen) is detected immediately -- the streak resets to 0 the instant the
        residual/Q-gain stops looking converged, the weight climbs back out of ``weight_tol``, or
        the weight itself is still moving.

        Both a converged own-residual/Q-gain AND a converged *weight* are required: a mixture
        component's own fit (mean/variance) can stabilize well before the joint E-step's
        responsibility reallocation across near-degenerate components finishes settling its
        weight (slow-manifold EM plateaus). Freezing on residual/Q-gain alone would risk locking
        in a component's weight (and hence the M-step never revisiting it) before it has actually
        reached its coordinate-ascent fixed point -- silently changing what the fit converges to,
        which the ConditionalJIT track's own correctness backbone forbids (a scheduler may change
        speed, never the answer). Requiring the weight to also have stopped moving
        (``abs(weight - prev_weight) < weight_delta_tol``) for ``freeze_patience`` consecutive
        rounds is the guard against that false-freeze failure mode.
        """
        report = node_report(component, field_path=str(idx), prev_residual=self._prev_residual.get(idx))
        self._prev_residual[idx] = report.residual
        residual_converged = report.update_kind == "frozen" or (
            report.q_gain is not None and abs(report.q_gain) < self.q_gain_tol
        )
        weight_collapsed = weight < self.weight_tol
        prev_weight = self._prev_weight.get(idx)
        weight_converged = prev_weight is not None and abs(weight - prev_weight) < self.weight_delta_tol
        self._prev_weight[idx] = weight

        converged = residual_converged and weight_collapsed and weight_converged
        streak = self._frozen_streak.get(idx, 0) + 1 if converged else 0
        self._frozen_streak[idx] = streak
        return streak >= self.freeze_patience

    def component_log_density(
        self, idx: int, component: Any, enc: Any, *, frozen: bool, compute_dtype: Any = None
    ) -> tuple[np.ndarray, bool]:
        """Return ``(log_density, was_cache_hit)`` for one component on this round.

        A cache hit costs a dict lookup + an ``O(param_count)`` signature compare -- never a call
        into ``component.seq_log_density`` (``O(nobs)``), which is the entire point of D2. Any
        mismatch between the cached signature and the component's current parameters -- frozen or
        not -- forces a recompute, so a stale cache can never silently persist past the point
        where it is wrong (acceptance criterion 3).
        """
        signature = _param_signature(component)
        entry = self._entries.get(idx)
        if frozen and entry is not None and entry.signature == signature:
            return entry.log_density, True
        log_density = _component_score(component, enc, compute_dtype)
        self._entries[idx] = _CacheEntry(signature=signature, log_density=log_density)
        return log_density, False


def _resolve_payload(enc_data: Any) -> Any:
    """Unwrap the canonical ``[(count, payload)]`` chunked ``seq_encode`` format to a bare payload.

    :func:`mixle.stats.compute.sequence.seq_encode` returns a (possibly multi-chunk) list; every
    ``seq_*`` method on a distribution (``seq_log_density``, ``seq_posterior``, ...) instead wants
    the bare per-chunk payload directly, exactly as :mod:`mixle.inference.em` unwraps it via
    ``_local_encoded_chunks``. This module currently supports single-chunk data only (the common
    ``num_chunks=1`` case, which is also all :func:`mixle.inference.em.run_em` itself assumes for
    its ``StandardEM``/``PosteriorTransformEM`` steps against a non-distributed ``enc_data``);
    multi-chunk/distributed orchestration is out of scope for this item (D2), same carve-out
    :func:`mixle.inference.estimation._local_encoded_chunks` already documents.
    """
    if (
        isinstance(enc_data, list)
        and len(enc_data) >= 1
        and isinstance(enc_data[0], tuple)
        and len(enc_data[0]) == 2
        and isinstance(enc_data[0][0], (int, np.integer, float))
    ):
        if len(enc_data) != 1:
            raise ValueError(
                "run_em_freeze_rollup currently supports single-chunk encoded data only "
                "(seq_encode(..., num_chunks=1), the default); got %d chunks." % len(enc_data)
            )
        return enc_data[0][1]
    return enc_data


def detect_frozen(cache: FreezeRollupCache, model: MixtureDistribution) -> set[int]:
    """Return the set of component indices D1 reports as frozen for ``model`` this round."""
    frozen: set[int] = set()
    for idx in range(model.num_components):
        if model.zw[idx]:
            continue  # already a structural no-op in MixtureDistribution's own seq_log_density
        if cache.is_frozen(idx, model.components[idx], weight=float(model.w[idx])):
            frozen.add(idx)
    return frozen


_FUSED_SCORING: tuple | None | bool = None


def _fused_scoring():
    """Lazily resolve the fused per-subtree scorer; False when numba/codegen is unavailable."""
    global _FUSED_SCORING
    if _FUSED_SCORING is None:
        try:
            from mixle.stats.compute.fused_codegen import (
                fused_accumulate,
                fused_seq_log_density,
                fusible,
                fusible_estep,
            )

            _FUSED_SCORING = (fused_seq_log_density, fusible, fused_accumulate, fusible_estep)
        except ImportError:  # pragma: no cover - numba optional
            _FUSED_SCORING = False
    return _FUSED_SCORING


def _component_suff_stat(
    component_estimator: Any,
    component_model: Any,
    enc: Any,
    weights: np.ndarray,
    compute_dtype: Any = None,
) -> Any:
    """One component's responsibility-weighted sufficient statistic, fused when the subtree allows.

    The M-step twin of :func:`_component_score`: a COMBINATOR component whose subtree fuses gets
    its whole E-step accumulation (inner responsibilities + per-leaf weighted statistics) in one
    nopython pass, packed in the estimator's own ``value()`` format -- eliminating the per-factor
    host walk that dominates deep components' M-step cost. Bare leaves keep the host accumulator
    (already a single vectorized pass), as do non-templated subtrees and estimators whose M-step
    needs more than fixed-width resident statistics. The fused kernel's reductions stay float64
    regardless of ``compute_dtype`` (only row arithmetic narrows).
    """
    fused = _fused_scoring()
    if fused:
        _, fusible, fused_accumulate, fusible_estep = fused
        is_combinator = (
            getattr(component_model, "components", None) is not None
            or getattr(component_model, "dists", None) is not None
        )
        # Same combinator guard and cache contract as _component_score (see the comment there);
        # accumulation reductions are float64 in both kernel families regardless of compute_dtype.
        if is_combinator and fusible(component_model) and fusible_estep(component_model):
            from mixle.stats.compute.kernel import _estimator_resident_supported

            if _estimator_resident_supported(component_estimator):
                return fused_accumulate(component_model, enc, weights, compute_dtype=compute_dtype)
    accumulator = component_estimator.accumulator_factory().make()
    accumulator.seq_update(enc, weights, component_model)
    return accumulator.value()


def _component_score(component: Any, enc: Any, compute_dtype: Any = None) -> np.ndarray:
    """One component's log-density column: the fused numba kernel when the SUBTREE fuses, else the
    host ``seq_log_density`` path.

    This is where the block/typed schedulers stop forfeiting the fused kernel the default
    ``optimize`` path already uses: each mixture component is itself a model, so a fusible
    component (deep scalar chains, composites of templated leaves) scores in one compiled pass
    while non-templated components (HMMs, GradLeaf, Laplace) keep the host path. ``compute_dtype``
    threads the validated reduced-precision band through (None = float64).
    """
    fused = _fused_scoring()
    if fused:
        fused_seq_log_density, fusible = fused[0], fused[1]
        # Only COMBINATOR components go through the fused kernel: a bare leaf is already one
        # vectorized host pass (routing it only added dispatch overhead). Both kernel families
        # share the once-per-structure-EVER compile contract: template kernels and the nested
        # scalar-tree kernels are structure-keyed and disk-cached (measured on a depth-18 chain
        # component: first-ever compile ~4.5s, then 0.10s in ANY process vs 0.34s on the host walk
        # -- 3.4x; dispatcher signature counts stay at 1-2, i.e. no per-call respecialization).
        # The nested kernels ignore compute_dtype and always run float64.
        is_combinator = (
            getattr(component, "components", None) is not None or getattr(component, "dists", None) is not None
        )
        if is_combinator and fusible(component):
            return np.asarray(fused_seq_log_density(component, enc, compute_dtype), dtype=np.float64)
    return np.asarray(component.seq_log_density(enc), dtype=np.float64)


def _combine(ll_mat: np.ndarray, log_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise log-sum-exp combine of a component log-density matrix into ``(log_density, gamma)``.

    Mirrors :meth:`MixtureDistribution.seq_log_density` and ``seq_posterior`` combined into a
    single pass (both need the same row-max/logsumexp arithmetic), since this module already has
    ``ll_mat`` in hand from the cache and would otherwise pay for that arithmetic twice.
    """
    ll = ll_mat + log_w
    ll_max = ll.max(axis=1, keepdims=True)
    bad_rows = np.isinf(ll_max.flatten())
    ll_safe = ll.copy()
    if np.any(bad_rows):
        ll_safe[bad_rows, :] = log_w
        ll_max[bad_rows] = np.max(log_w)
    shifted = ll_safe - ll_max
    np.exp(shifted, out=shifted)
    row_sum = shifted.sum(axis=1, keepdims=True)
    log_density = (np.log(row_sum) + ll_max).flatten()
    gamma = np.divide(shifted, row_sum, out=np.zeros_like(shifted), where=row_sum > 0.0)
    return log_density, gamma


def _log_density_from_matrix(ll_mat: np.ndarray, log_w: np.ndarray) -> np.ndarray:
    """Combine component scores without constructing unused responsibilities."""

    ll = ll_mat + log_w
    ll_max = ll.max(axis=1, keepdims=True)
    bad_rows = np.isinf(ll_max.flatten())
    if np.any(bad_rows):
        ll[bad_rows, :] = log_w
        ll_max[bad_rows] = np.max(log_w)
    ll -= ll_max
    np.exp(ll, out=ll)
    return (np.log(ll.sum(axis=1, keepdims=True)) + ll_max).flatten()


def _component_log_density_matrix_profiled(
    model: MixtureDistribution,
    enc_data: Any,
    cache: FreezeRollupCache,
    frozen_idx: set[int],
    compute_dtype: Any = None,
) -> tuple[np.ndarray, int, DensityMatrixProfile]:
    """Build a component score matrix and expose the assumptions behind saved work."""

    n = None
    cols: list[np.ndarray | None] = [None] * model.num_components
    evals = 0
    hits = 0
    zero_weight_skips = 0
    evaluation_seconds = 0.0
    cache_hit_seconds = 0.0
    component_seconds = []
    for idx in range(model.num_components):
        if model.zw[idx]:
            zero_weight_skips += 1
            continue
        enc_i = _component_enc(enc_data, idx)
        started = time.perf_counter()
        if idx in frozen_idx:
            log_density, hit = cache.component_log_density(
                idx, model.components[idx], enc_i, frozen=True, compute_dtype=compute_dtype
            )
        else:
            log_density = _component_score(model.components[idx], enc_i, compute_dtype)
            hit = False
        elapsed = time.perf_counter() - started
        if hit:
            hits += 1
            cache_hit_seconds += elapsed
        else:
            evals += 1
            evaluation_seconds += elapsed
            component_seconds.append((idx, elapsed))
        cols[idx] = log_density
        if n is None:
            n = len(log_density)
    if n is None:
        raise ValueError("MixtureDistribution has no components with nonzero weight.")
    assembly_started = time.perf_counter()
    ll_mat = np.full((n, model.num_components), -np.inf)
    for idx, col in enumerate(cols):
        if col is not None:
            ll_mat[:, idx] = col
    assembly_seconds = time.perf_counter() - assembly_started
    profile = DensityMatrixProfile(
        evals,
        hits,
        0,
        zero_weight_skips,
        evaluation_seconds,
        cache_hit_seconds,
        assembly_seconds,
        tuple(component_seconds),
    )
    return ll_mat, evals, profile


def _updated_component_log_density_matrix_profiled(
    model: MixtureDistribution,
    enc_data: Any,
    active_idx: set[int],
    base_matrix: np.ndarray,
    compute_dtype: Any = None,
) -> tuple[np.ndarray, int, DensityMatrixProfile]:
    """Copy a valid prior matrix and replace only columns whose components moved."""

    if base_matrix.ndim != 2 or base_matrix.shape[1] != model.num_components:
        raise ValueError("base component-score matrix does not match the candidate model.")
    indices, columns, evaluations, column_profile = _updated_component_log_density_columns_profiled(
        model,
        enc_data,
        active_idx,
        base_matrix.shape[0],
        compute_dtype=compute_dtype,
    )
    assembly_started = time.perf_counter()
    ll_mat = base_matrix.copy()
    if indices:
        ll_mat[:, indices] = columns
    assembly_seconds = column_profile.assembly_seconds + time.perf_counter() - assembly_started
    profile = DensityMatrixProfile(
        evaluations,
        0,
        model.num_components - len(active_idx),
        column_profile.zero_weight_skips,
        column_profile.evaluation_seconds,
        0.0,
        assembly_seconds,
        column_profile.component_evaluation_seconds,
    )
    return ll_mat, evaluations, profile


def _updated_component_log_density_columns_profiled(
    model: MixtureDistribution,
    enc_data: Any,
    active_idx: set[int],
    nobs: int,
    compute_dtype: Any = None,
) -> tuple[tuple[int, ...], np.ndarray, int, DensityMatrixProfile]:
    """Score moved components without cache hashing or a full-matrix copy.

    Active candidate parameters were just re-estimated, so they cannot be cache
    hits. The former path hashed every active parameter tree and populated cache
    entries that block EM never read. A compact column matrix also lets the
    caller mutate its accepted score matrix only after transaction commit.
    """

    indices = tuple(sorted(active_idx))
    columns: list[np.ndarray] = []
    evaluations = 0
    zero_weight_skips = 0
    evaluation_seconds = 0.0
    component_seconds = []
    for idx in indices:
        if model.zw[idx]:
            columns.append(np.full(nobs, -np.inf, dtype=np.float64))
            zero_weight_skips += 1
            continue
        enc_i = _component_enc(enc_data, idx)
        started = time.perf_counter()
        log_density = _component_score(model.components[idx], enc_i, compute_dtype)
        elapsed = time.perf_counter() - started
        if len(log_density) != nobs:
            raise ValueError("updated component score length does not match the base matrix.")
        columns.append(log_density)
        evaluations += 1
        evaluation_seconds += elapsed
        component_seconds.append((idx, elapsed))
    assembly_started = time.perf_counter()
    matrix = np.column_stack(columns) if columns else np.empty((nobs, 0), dtype=np.float64)
    assembly_seconds = time.perf_counter() - assembly_started
    profile = DensityMatrixProfile(
        evaluations,
        0,
        model.num_components - len(active_idx),
        zero_weight_skips,
        evaluation_seconds,
        0.0,
        assembly_seconds,
        tuple(component_seconds),
    )
    return indices, matrix, evaluations, profile


def _component_log_density_matrix(
    model: MixtureDistribution, enc_data: Any, cache: FreezeRollupCache, frozen_idx: set[int]
) -> tuple[np.ndarray, int]:
    """Return ``(ll_mat, n_log_density_evals)`` for one E-step over ``model``, cache-aware.

    ``n_log_density_evals`` counts only genuine ``seq_log_density`` calls (cache misses); a
    cache-hit component or an exact-zero-weight component (already skipped by
    ``MixtureDistribution`` itself) costs 0.
    """
    n = None
    cols: list[np.ndarray | None] = [None] * model.num_components
    evals = 0
    for idx in range(model.num_components):
        if model.zw[idx]:
            continue
        enc_i = _component_enc(enc_data, idx)
        log_density, hit = cache.component_log_density(idx, model.components[idx], enc_i, frozen=idx in frozen_idx)
        if not hit:
            evals += 1
        cols[idx] = log_density
        if n is None:
            n = len(log_density)
    if n is None:
        raise ValueError("MixtureDistribution has no components with nonzero weight.")
    ll_mat = np.full((n, model.num_components), -np.inf)
    for idx, col in enumerate(cols):
        if col is not None:
            ll_mat[:, idx] = col
    return ll_mat, evals


def _mixture_weights(estimator: MixtureEstimator, counts: np.ndarray) -> np.ndarray:
    """Replicate :meth:`MixtureEstimator.estimate`'s weight-update arithmetic from raw ``counts``.

    Needed because the freeze/roll-up M-step (:func:`_m_step`) cannot call
    ``MixtureEstimator.estimate`` directly -- that method unconditionally re-estimates every
    component from its own sufficient statistics, which is exactly the per-round cost D2 exists
    to skip for frozen components. Supports the plain-MLE / ``fixed_weights`` / ``pseudo_count`` /
    ``w_min``-floor paths (byte-identical arithmetic to the corresponding branches in
    ``MixtureEstimator.estimate``); a conjugate Dirichlet weight prior is out of scope for the
    skip-frozen path (see :func:`_m_step`).
    """
    num_components = estimator.num_components
    if estimator.fixed_weights is not None:
        w = np.asarray(estimator.fixed_weights, dtype=float)
    elif estimator.pseudo_count is not None and estimator.suff_stat is None:
        p = estimator.pseudo_count / num_components
        w = counts + p
        w = w / w.sum()
    elif estimator.pseudo_count is not None and estimator.suff_stat is not None:
        w = (counts + estimator.suff_stat * estimator.pseudo_count) / (counts.sum() + estimator.pseudo_count)
    else:
        nobs_loc = counts.sum()
        if nobs_loc == 0:
            w = np.ones(num_components) / float(num_components)
        else:
            w = counts / counts.sum()
    w = np.asarray(w, dtype=float)
    if estimator.w_min > 0.0 and estimator.fixed_weights is None:
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.maximum(w, estimator.w_min)
        w = w / w.sum()
    return w


def _m_step(
    enc_data: Any,
    estimator: MixtureEstimator,
    model: MixtureDistribution,
    gamma: np.ndarray,
    frozen_idx: set[int],
    compute_dtype: Any = None,
) -> MixtureDistribution:
    """One freeze/roll-up M-step: only ``frozen_idx``-excluded (active) components are re-estimated.

    A frozen component's model object is carried forward UNCHANGED -- no accumulator, no
    ``estimate`` call -- which is what makes its next-round cache lookup a guaranteed hit (its
    parameter signature cannot have moved because nothing touched it).
    """
    if getattr(estimator, "has_conj_prior", False) and estimator.fixed_weights is None:
        raise NotImplementedError(
            "freeze_rollup does not support a conjugate Dirichlet weight prior; use a plain "
            "MixtureEstimator(...) (no `prior=`) or mixle.inference.em.run_em for that path."
        )
    counts = gamma.sum(axis=0)
    new_components = list(model.components)
    for idx in range(model.num_components):
        if idx in frozen_idx or model.zw[idx]:
            continue
        enc_i = _component_enc(enc_data, idx)
        suff_stat = _component_suff_stat(
            estimator.estimators[idx], model.components[idx], enc_i, gamma[:, idx], compute_dtype
        )
        new_components[idx] = estimator.estimators[idx].estimate(float(counts[idx]), suff_stat)
    w = _mixture_weights(estimator, counts)
    return MixtureDistribution(new_components, w, name=estimator.name)


def run_em_freeze_rollup(
    enc_data: Any,
    estimator: MixtureEstimator,
    initial_model: MixtureDistribution,
    *,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    cache: FreezeRollupCache | None = None,
    q_gain_tol: float = _DEFAULT_Q_GAIN_TOL,
    weight_tol: float = _DEFAULT_WEIGHT_TOL,
    weight_delta_tol: float = _DEFAULT_WEIGHT_DELTA_TOL,
    freeze_patience: int = _DEFAULT_FREEZE_PATIENCE,
    accept_tolerance: float = _DEFAULT_ACCEPT_TOLERANCE,
) -> tuple[MixtureDistribution, list[FreezeRollupStats]]:
    """Run EM over a :class:`MixtureDistribution` with D1-driven freeze/roll-up E-step caching.

    Mirrors :func:`mixle.inference.em.run_em` + ``PosteriorTransformEM``'s soft-EM update (same
    GEM/ECM coordinate-ascent structure: an E-step producing responsibilities, then a per-block
    conditional M-step), with two differences that only affect SPEED, never correctness:

    1. A component D1 reports as frozen (see :meth:`FreezeRollupCache.is_frozen`) reuses last
       round's cached per-datum log-density instead of recomputing it, and is excluded from the
       M-step (its model object carries forward unchanged) -- so a round with only ``k`` of ``K``
       components active costs ``O(k/K)`` of a full round's ``seq_log_density`` work.
    2. Every round is objective-gated exactly like ``MonotonicEM``: the candidate model's
       objective is checked (again cache-aware, so this costs the same ``O(active fraction)``) and
       the step is rejected -- keeping the previous model -- if the objective would decrease
       beyond ``accept_tolerance``. This is what makes the returned ``history`` a real monotone-objective
       receipt, not just an assumption.

    Returns ``(final_model, history)`` where ``history[i]`` is round ``i``'s
    :class:`FreezeRollupStats` (including ``objective``, the observed-data log likelihood for that round,
    and ``n_log_density_evals``, the component-evaluation work receipt this module optimizes).
    """
    if not isinstance(initial_model, MixtureDistribution):
        raise TypeError("run_em_freeze_rollup requires a MixtureDistribution model.")
    cache = (
        FreezeRollupCache(
            q_gain_tol=q_gain_tol,
            weight_tol=weight_tol,
            weight_delta_tol=weight_delta_tol,
            freeze_patience=freeze_patience,
        )
        if cache is None
        else cache
    )
    enc_payload = _resolve_payload(enc_data)

    model = initial_model
    history: list[FreezeRollupStats] = []
    old_value: float | None = None

    for round_index in range(max(1, int(max_its))):
        frozen_idx = detect_frozen(cache, model)
        ll_mat, evals_e = _component_log_density_matrix(model, enc_payload, cache, frozen_idx)
        log_density, gamma = _combine(ll_mat, model.log_w)
        current_value = float(np.sum(log_density))
        if old_value is None:
            old_value = current_value

        transaction = MutableStateSnapshot.capture(model, estimator)
        candidate = _m_step(enc_payload, estimator, model, gamma, frozen_idx)
        candidate_frozen = detect_frozen(cache, candidate)
        ll_mat_c, evals_c = _component_log_density_matrix(candidate, enc_payload, cache, candidate_frozen)
        candidate_log_density = _log_density_from_matrix(ll_mat_c, candidate.log_w)
        candidate_value = float(np.sum(candidate_log_density))

        accepted = np.isfinite(candidate_value) and candidate_value + accept_tolerance >= current_value
        if accepted:
            model = candidate
            round_value = candidate_value
        else:
            transaction.restore()
            round_value = current_value

        n_frozen = len(frozen_idx)
        n_zero = int(np.count_nonzero(model.zw)) if not accepted else int(np.count_nonzero(candidate.zw))
        history.append(
            FreezeRollupStats(
                round_index=round_index,
                n_components=model.num_components,
                n_active=model.num_components - n_frozen - n_zero,
                n_frozen=n_frozen,
                n_zero_weight=n_zero,
                n_log_density_evals=evals_e + evals_c,
                objective=round_value,
                accepted=accepted,
            )
        )

        if delta is not None and 0.0 <= round_value - old_value < delta:
            break
        old_value = round_value

    return model, history
