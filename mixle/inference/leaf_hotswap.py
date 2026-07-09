"""Leaf hot-swap / analytic roll-up -- swap a plateaued gradient leaf for a closed-form surrogate,
with a retained path back on misfit (workstream D4).

Frame (see the ConditionalJIT track, D1-D6): **the estimator tree is an IR**. D1
(:mod:`mixle.inference.node_report`) instruments every node with a per-round residual/Q-gain
report and an ``update_kind`` classification -- in particular ``"gradient"`` for a
:class:`~mixle.models.grad_leaf.GradLeaf` node, whose M-step is ``m_steps`` iterations of SGD/Adam
rather than a closed-form update. D2 (:mod:`mixle.inference.freeze_rollup`) and D3
(:mod:`mixle.inference.block_em`) both spend a converged/near-zero D1 Q-gain on SCHEDULING
decisions (skip an E-step recompute, skip a turn in this round's M-step) while leaving the node's
own model object untouched. D4 goes one step further for gradient leaves specifically: once a
gradient leaf's own Q-gain has plateaued (gradient descent has stopped making meaningful progress,
so the remaining M-step compute it would otherwise spend is close to wasted), swap it for a
*closed-form* surrogate -- a moment-matched
:class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution` fit
against the SAME (responsibility-weighted) data the gradient leaf was trained on -- so every later
round's cost for that node collapses from ``O(param_count * _GRADIENT_STEPS)`` (D1's own gradient
M-step cost proxy) to ``O(param_count)`` (a closed-form MLE).

Correctness backbone (unchanged from the rest of the D-track): this is a SCHEDULING/specialization
optimization only. Swapping a node's *model object* for an approximation is more aggressive than
D2/D3's "leave the object alone, just skip recomputing/re-fitting it" story, so D4 earns back the
Neal-Hinton guarantee two ways instead of one: (1) the swap itself is gated on a genuine, locally
computed misfit RECEIPT (:func:`misfit_receipt`) -- not merely assumed to be a good approximation
-- and (2) the ORIGINAL gradient leaf is never discarded (:class:`SwapRecord.original`), so if the
receipt later shows the surrogate drifting away from the real held-out data (e.g. the underlying
regime shifts after the swap), :func:`swap_back` restores the exact retained object and gradient
fitting resumes exactly where it left off -- "never truly forget", the same policy D2's freeze/
roll-up commits to for frozen mixture components. F itself (the real Neal-Hinton free energy) is
still the audit receipt: :func:`run_em_with_hotswap` gates every round's proposal -- the swap-in
itself AND the following M-step (gradient OR closed-form), as one atomic unit -- behind the same
accept/reject monotone-F test D2/D3 already use, so a bad swap can cost speed (a rejected-and-
reverted round, or a later swap-back once already committed) but never correctness. This is the
gate that actually makes the mechanism safe: :func:`moment_matched_surrogate` on its own is a
plain Gaussian MLE fit with no guarantee of matching an arbitrary (e.g. multi-modal) gradient
leaf's held-out density -- see that function's own docstring for a worked adversarial example and
``mixle.tests.leaf_hotswap_test.MonotoneObjectiveGateCatchesBadSwapTestCase`` for the regression
test proving the gate rejects and fully reverts (not merely skips the M-step of) exactly that
case. Earlier revisions of this module applied a plateau-triggered swap to the working tree
unconditionally, before the round's accept/reject check, and only skipped ``model = candidate`` on
rejection -- so a rejected round still silently returned the corrupted surrogate; the gate now
rolls the swap itself back too when a round is rejected.

Scope: like D2/D3, this targets one gradient leaf embedded as a component of a
:class:`~mixle.stats.latent.mixture.MixtureDistribution` (mixed freely with classical families,
per :mod:`mixle.models.grad_leaf`'s whole point) -- the "tree" in ``swap_leaf(tree, leaf_path,
surrogate)`` is that mixture, and ``leaf_path`` is the component index. A single gradient leaf not
embedded in any combinator (``tree`` is the leaf itself) is also supported directly (``leaf_path``
is ignored) since it is strictly the ``num_components=1`` degenerate case of the same operation --
useful for isolating the swap/misfit/swap-back mechanics from mixture E-step bookkeeping in tests.
Generalizing past ``MixtureDistribution`` to arbitrary composite/sequence trees is the same "later
items are expected to widen it" carve-out D2 documents for its own combinator scope.

Moment-matching machinery: G1's ``moment_propagation.py`` (``origin/moment-propagation``,
unmerged into this branch's D1/D2/D3 chain -- see this module's PR description for how it was
read via ``git show`` rather than a cross-branch merge) propagates a GAUSSIAN LAW through the
specific *layer types* of ``mixle.models.transformer.CausalLM`` (Linear/LayerNorm/GELU/Attention),
which is a different, narrower object than "moment-match an arbitrary gradient leaf's behavior
against arbitrary data" -- it has no entry point that takes a generic ``GradLeaf`` and a data
sample. Reusing it here would mean either constraining D4 to transformer-shaped leaves only (out
of scope -- ``GradLeaf`` wraps ANY torch density module) or re-deriving a generic version of its
per-layer-law machinery, which is its own multi-week item. :func:`moment_matched_surrogate`
therefore uses straightforward closed-form moment matching instead: draw/collect the SAME
(possibly responsibility-weighted) data the gradient leaf was trained on and fit a
:class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution` to it via
the existing, real closed-form MLE machinery
(:class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianAccumulator` /
:class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianEstimator`) -- an
honest, fully-real "G1 machinery" scope per the roadmap item's own explicit fallback clause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference.freeze_rollup import (
    FreezeRollupCache,
    _combine,
    _component_log_density_matrix,
    _mixture_weights,
    _resolve_payload,
    detect_frozen,
)
from mixle.inference.node_report import node_report
from mixle.models.grad_leaf import GradLeaf
from mixle.stats.latent.mixture import MixtureDistribution, MixtureEstimator, _component_enc
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianAccumulator,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)

__all__ = [
    "PlateauMonitor",
    "SwapRecord",
    "LeafHotswapStats",
    "moment_matched_surrogate",
    "misfit_receipt",
    "swap_leaf",
    "swap_back",
    "run_em_with_hotswap",
]

_DEFAULT_PLATEAU_Q_GAIN_TOL = 0.02
_DEFAULT_PLATEAU_PATIENCE = 3
_DEFAULT_PLATEAU_N_MC = 2000
_DEFAULT_MISFIT_TOL = 0.15  # relative NLL degradation tolerated before swapping back
_DEFAULT_ACCEPT_TOLERANCE = 1.0e-9
_SCORE_SEED = 0  # fixed, shared across rounds/components -- see mixle.inference.block_em


def _as_matrix(data: Any) -> np.ndarray:
    """Coerce raw (possibly 1-D) data into an ``(n, d)`` float matrix, as :class:`GradLeafEncoder`
    and :class:`MultivariateGaussianDataEncoder` both already assume."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


# --------------------------------------------------------------------------------------------------
# 1. Plateau detection (D1-driven, patience-gated -- mirrors D2's freeze streak / D3's starvation
#    guard, just triggering a SWAP instead of a freeze or a scheduling skip).
# --------------------------------------------------------------------------------------------------


class PlateauMonitor:
    """Tracks, per leaf path, how many consecutive rounds a D1 :class:`NodeReport` has reported a
    near-zero Q-gain for a ``"gradient"``-update-kind node -- the "gradient descent has stopped
    making meaningful progress" signal :func:`run_em_with_hotswap` swaps on.

    Deliberately keyed and reset exactly like :class:`mixle.inference.freeze_rollup.
    FreezeRollupCache`'s own ``_frozen_streak``: a report that stops looking plateaued (the node
    moved again, or was swapped back to the original -- see :meth:`reset`) resets the streak to 0
    rather than latching a stale verdict.
    """

    def __init__(
        self,
        *,
        q_gain_tol: float = _DEFAULT_PLATEAU_Q_GAIN_TOL,
        patience: int = _DEFAULT_PLATEAU_PATIENCE,
        n_mc: int = _DEFAULT_PLATEAU_N_MC,
    ) -> None:
        self.q_gain_tol = float(q_gain_tol)
        self.patience = max(1, int(patience))
        self.n_mc = int(n_mc)
        self._prev_residual: dict[Any, float] = {}
        self._streak: dict[Any, int] = {}

    def reset(self, path: Any) -> None:
        """Clear the tracked history for ``path`` (e.g. after a swap or a swap-back)."""
        self._prev_residual.pop(path, None)
        self._streak.pop(path, None)

    def is_plateaued(self, path: Any, leaf: Any, *, nobs: float | None = None, seed: int = _SCORE_SEED) -> bool:
        """Return whether ``leaf`` (identified by ``path``) has plateaued this round.

        Only a D1 ``update_kind == "gradient"`` node can plateau in this module's sense (a
        closed-form/conjugate/frozen/em node has no "wasted SGD compute" to reclaim by swapping);
        anything else always returns ``False`` and resets the streak, so a leaf that has already
        been swapped for its (closed-form) surrogate is correctly reported as "not plateaued"
        going forward -- there is nothing left to swap.

        D1's own ``residual`` is a MONTE-CARLO estimate (``-mean(log_density(x))`` over the node's
        own self-samples, see :mod:`mixle.inference.node_report`'s module docstring) -- for a
        genuinely converged gradient leaf, round-to-round Q-gain is dominated by MC sampling noise
        rather than any real drift in the fit, so this class uses a much larger ``n_mc`` than D1's
        own default (``_DEFAULT_PLATEAU_N_MC`` vs D1's ``_DEFAULT_MC_SAMPLES=64``) to push that
        noise floor down, and a correspondingly looser ``q_gain_tol`` than D2/D3's exact-residual
        default (their residual is deterministic given unchanged parameters; this one never is).
        """
        report = node_report(
            leaf,
            field_path=str(path),
            n_mc=self.n_mc,
            seed=seed,
            nobs=nobs,
            prev_residual=self._prev_residual.get(path),
        )
        self._prev_residual[path] = report.residual
        if report.update_kind != "gradient":
            self._streak.pop(path, None)
            return False
        converged = report.q_gain is not None and abs(report.q_gain) < self.q_gain_tol
        streak = self._streak.get(path, 0) + 1 if converged else 0
        self._streak[path] = streak
        return streak >= self.patience


# --------------------------------------------------------------------------------------------------
# 2. Moment-matched closed-form surrogate.
# --------------------------------------------------------------------------------------------------


def moment_matched_surrogate(
    gradient_leaf: GradLeaf,
    data: Any,
    weights: np.ndarray | None = None,
) -> MultivariateGaussianDistribution:
    """Fit a closed-form :class:`MultivariateGaussianDistribution` that moment-matches ``data`` --
    the SAME (optionally responsibility-``weights``-ed) data ``gradient_leaf`` was trained on -- via
    the real closed-form MLE machinery (weighted mean/covariance), not a re-derived formula.

    This is a genuine fit, not a placeholder: it reads no attribute off ``gradient_leaf`` at all
    (a torch module has no portable closed-form "current mean/covariance" to read off directly --
    scoring/sampling is the only generic contract), so "moment-matched against the gradient leaf's
    current behavior" means "against the data its current fit was scoring/trained on", exactly the
    module-docstring's documented (and roadmap-sanctioned) fallback to plain closed-form moment
    matching in lieu of G1's transformer-specific law-propagation machinery.

    ``gradient_leaf`` is accepted (rather than a bare module) purely as a type/documentation
    signal of intent -- see the module docstring's "why not G1" note -- it is not otherwise used.

    IMPORTANT, honest limitation -- read before trusting this function's output alone: a single
    Gaussian can only ever be as good an approximation as the true fitted density IS Gaussian.
    Against a unimodal, roughly-Gaussian-shaped leaf (the common case for a single mixture
    component pulling its own well-separated slice of responsibility-weighted data) this is an
    excellent approximation. Against a leaf whose OWN fitted density is multi-modal, heavy-tailed,
    or otherwise non-Gaussian, this function will silently produce a POOR approximation with no
    warning -- e.g. on a genuinely bimodal ``GradLeaf`` (two well-separated modes,
    ``mixle.tests.leaf_hotswap_test.BimodalGauss``) the moment-matched surrogate collapses both
    modes into one wide Gaussian sitting between them, degrading held-out NLL by roughly 80%
    relative to the original leaf (see
    ``mixle.tests.leaf_hotswap_test.MonotoneObjectiveGateCatchesBadSwapTestCase``). This function
    provides NO guarantee, on its own, that the surrogate's held-out density tracks the original
    leaf's -- that guarantee, to the extent one exists, is earned entirely by
    :func:`run_em_with_hotswap`'s per-round monotone-F accept/reject gate (see that function's own
    docstring): a swap-plus-refit round that does not improve the real Neal-Hinton objective is
    rejected and reverted, INCLUDING the swap itself, not merely the following M-step. Do not call
    :func:`moment_matched_surrogate` outside that gated driver (or an equivalent one) and assume
    the result is a safe stand-in for ``gradient_leaf`` -- verify with a real misfit receipt
    (:func:`misfit_receipt`) against genuinely held-out data first.
    """
    if not isinstance(gradient_leaf, GradLeaf):
        raise TypeError("moment_matched_surrogate expects a GradLeaf (the D4 hot-swap target).")
    x = _as_matrix(data)
    if x.shape[0] == 0:
        raise ValueError("moment_matched_surrogate requires at least one observation.")
    w = np.ones(x.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != x.shape[0]:
        raise ValueError("weights must have the same length as data.")

    acc = MultivariateGaussianAccumulator(dim=x.shape[1])
    acc.seq_update(x, w, None)
    est = MultivariateGaussianEstimator(dim=x.shape[1])
    return est.estimate(float(w.sum()), acc.value())


# --------------------------------------------------------------------------------------------------
# 3. Swap record, swap / swap-back.
# --------------------------------------------------------------------------------------------------


@dataclass
class SwapRecord:
    """The retrievable receipt of one leaf hot-swap -- "never truly forget" (D2's own phrase for
    its freeze/roll-up cache) applied to a swapped-out gradient leaf: ``original`` is the exact
    :class:`GradLeaf` object that was in the tree before the swap, always retrievable, never
    discarded, so :func:`swap_back` can restore it byte-for-byte.
    """

    leaf_path: Any
    original: GradLeaf
    surrogate: MultivariateGaussianDistribution
    swap_round: int
    baseline_misfit: float | None
    misfit_history: list[float] = field(default_factory=list)
    swapped_back: bool = False
    swap_back_round: int | None = None


def _replace_leaf(tree: Any, leaf_path: Any, new_leaf: Any) -> Any:
    """Return a copy of ``tree`` with the node at ``leaf_path`` replaced by ``new_leaf``.

    Supports the two shapes documented in the module docstring: ``tree`` itself is the leaf
    (``leaf_path`` ignored, the degenerate ``num_components=1`` case), or ``tree`` is a
    :class:`MixtureDistribution` and ``leaf_path`` is a component index.
    """
    if isinstance(tree, MixtureDistribution):
        if not isinstance(leaf_path, (int, np.integer)):
            raise TypeError("leaf_path must be an int component index for a MixtureDistribution tree.")
        idx = int(leaf_path)
        if not (0 <= idx < tree.num_components):
            raise IndexError(f"leaf_path {idx} out of range for a {tree.num_components}-component mixture.")
        new_components = list(tree.components)
        new_components[idx] = new_leaf
        return MixtureDistribution(new_components, list(tree.w), name=tree.name)
    return new_leaf


def swap_leaf(
    tree: Any, leaf_path: Any, surrogate: MultivariateGaussianDistribution, *, round_index: int = 0
) -> tuple[Any, SwapRecord]:
    """Replace the plateaued gradient leaf at ``leaf_path`` in ``tree`` with ``surrogate``.

    Returns ``(new_tree, swap_record)``: ``new_tree`` has the surrogate in place (every generic
    D1/D2/D3 mechanism -- :func:`~mixle.inference.node_report.node_report`,
    :func:`~mixle.inference.freeze_rollup.detect_frozen`, ``seq_log_density`` -- sees an ordinary
    :class:`MultivariateGaussianDistribution` node from here on, no special-casing required
    upstream); ``swap_record.original`` retains the exact :class:`GradLeaf` that was swapped out,
    per this module's "never discard capacity" policy (see :class:`SwapRecord`).
    """
    original = (
        tree
        if leaf_path is None
        else (tree.components[int(leaf_path)] if isinstance(tree, MixtureDistribution) else tree)
    )
    if not isinstance(original, GradLeaf):
        raise TypeError(f"swap_leaf target at {leaf_path!r} is not a GradLeaf (got {type(original).__name__}).")
    new_tree = _replace_leaf(tree, leaf_path, surrogate)
    record = SwapRecord(
        leaf_path=leaf_path, original=original, surrogate=surrogate, swap_round=round_index, baseline_misfit=None
    )
    return new_tree, record


def swap_back(tree: Any, swap_record: SwapRecord, *, round_index: int = 0) -> Any:
    """Restore ``swap_record.original`` (the retained gradient leaf) into ``tree`` at
    ``swap_record.leaf_path``, undoing :func:`swap_leaf`. Marks ``swap_record`` as swapped back
    (in place) so callers/tests can assert the misfit receipt actually fired.
    """
    new_tree = _replace_leaf(tree, swap_record.leaf_path, swap_record.original)
    swap_record.swapped_back = True
    swap_record.swap_back_round = round_index
    return new_tree


# --------------------------------------------------------------------------------------------------
# 4. Misfit monitoring.
# --------------------------------------------------------------------------------------------------


def misfit_receipt(surrogate: MultivariateGaussianDistribution, holdout_data: Any) -> float:
    """A genuine, locally-computed misfit scalar for ``surrogate`` on REAL held-out data: its own
    negative log-likelihood, ``-mean(log_density(x))``. Not a placeholder -- recomputed from real
    samples every call, exactly the same style of receipt D1's ``residual`` and G1's
    ``closure_error`` both are (see the respective module docstrings). Lower is better; compared
    against :attr:`SwapRecord.baseline_misfit` (the receipt measured right after the swap) by
    :func:`should_swap_back` to decide whether the surrogate has since drifted away from the real
    data it was swapped in to approximate.
    """
    x = _as_matrix(holdout_data)
    return float(-np.mean(surrogate.seq_log_density(x)))


def should_swap_back(
    swap_record: SwapRecord, current_misfit: float, *, misfit_tol: float = _DEFAULT_MISFIT_TOL
) -> bool:
    """True iff ``current_misfit`` has degraded by more than ``misfit_tol`` (relative) versus the
    baseline misfit recorded right after the swap -- the "swap back on misfit receipts" gate.

    A missing/non-finite baseline (e.g. no held-out data was available at swap time) or a
    non-finite current misfit conservatively triggers a swap-back rather than silently keeping a
    surrogate whose fit quality this module cannot actually vouch for.
    """
    if swap_record.baseline_misfit is None or not np.isfinite(swap_record.baseline_misfit):
        return True
    if not np.isfinite(current_misfit):
        return True
    return current_misfit > swap_record.baseline_misfit * (1.0 + misfit_tol)


# --------------------------------------------------------------------------------------------------
# 5. EM-loop-level driver.
# --------------------------------------------------------------------------------------------------


@dataclass
class LeafHotswapStats:
    """One round's accounting for the hot-swap EM driver -- mirrors
    :class:`mixle.inference.freeze_rollup.FreezeRollupStats` / :class:`mixle.inference.block_em.
    BlockEMStats` (same ``n_log_density_evals`` wall-clock proxy and real Neal-Hinton
    ``objective``), plus the D4-specific ``n_gradient_m_steps`` / ``n_closed_form_m_steps`` split
    that is literally what "faster to same F" is measured against: a swapped component's per-round
    M-step cost drops from D1's ``param_count * _GRADIENT_STEPS`` proxy to ``param_count``.

    ``swapped_this_round`` is which components a plateau *proposed* a swap for this round -- it is
    recorded even when ``accepted`` is False (a rejected round rolls the swap itself back out of
    the returned model/``swap_records``, but the attempt still happened and is worth a receipt);
    use ``n_swapped`` (or check ``idx in swap_records``) for which swaps are actually COMMITTED as
    of this round.
    """

    round_index: int
    n_components: int
    n_frozen: int
    n_swapped: int
    n_log_density_evals: int
    n_gradient_m_steps: int
    n_closed_form_m_steps: int
    objective: float
    accepted: bool = True
    swapped_this_round: tuple[Any, ...] = ()
    swapped_back_this_round: tuple[Any, ...] = ()


def _m_step_hotswap(
    enc_data: Any,
    estimator: MixtureEstimator,
    model: MixtureDistribution,
    gamma: np.ndarray,
    frozen_idx: set[int],
    swapped_idx: set[int],
) -> tuple[MixtureDistribution, int, int]:
    """Like :func:`mixle.inference.freeze_rollup._m_step`, but a component in ``swapped_idx`` gets
    a closed-form moment-matched re-fit (:func:`moment_matched_surrogate`, on the same
    responsibility-weighted data a gradient M-step would have used) instead of the mixture
    estimator's own (gradient) M-step -- the actual per-round cost reduction D4 exists for.
    Returns ``(new_model, n_gradient_m_steps, n_closed_form_m_steps)``.
    """
    counts = gamma.sum(axis=0)
    new_components = list(model.components)
    n_grad = 0
    n_closed = 0
    for idx in range(model.num_components):
        if idx in frozen_idx or model.zw[idx]:
            continue
        enc_i = _component_enc(enc_data, idx)
        if idx in swapped_idx:
            x = _as_matrix(enc_i)
            acc = MultivariateGaussianAccumulator(dim=x.shape[1])
            acc.seq_update(x, gamma[:, idx], None)
            new_components[idx] = MultivariateGaussianEstimator(dim=x.shape[1]).estimate(
                float(counts[idx]), acc.value()
            )
            n_closed += 1
        else:
            acc = estimator.estimators[idx].accumulator_factory().make()
            acc.seq_update(enc_i, gamma[:, idx], model.components[idx])
            new_components[idx] = estimator.estimators[idx].estimate(float(counts[idx]), acc.value())
            if isinstance(model.components[idx], GradLeaf):
                n_grad += 1
    w = _mixture_weights(estimator, counts)
    return MixtureDistribution(new_components, w, name=estimator.name), n_grad, n_closed


def run_em_with_hotswap(
    enc_data: Any,
    estimator: MixtureEstimator,
    initial_model: MixtureDistribution,
    *,
    holdout_data: Any = None,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    cache: FreezeRollupCache | None = None,
    monitor: PlateauMonitor | None = None,
    freeze_q_gain_tol: float = 1.0e-6,
    plateau_q_gain_tol: float = _DEFAULT_PLATEAU_Q_GAIN_TOL,
    plateau_patience: int = _DEFAULT_PLATEAU_PATIENCE,
    plateau_n_mc: int = _DEFAULT_PLATEAU_N_MC,
    misfit_tol: float = _DEFAULT_MISFIT_TOL,
    accept_tolerance: float = _DEFAULT_ACCEPT_TOLERANCE,
) -> tuple[MixtureDistribution, list[LeafHotswapStats], dict[int, SwapRecord]]:
    """Run EM over a :class:`MixtureDistribution` with D4 leaf hot-swap: any component D1 reports
    as a plateaued gradient leaf (see :class:`PlateauMonitor`) is swapped for a moment-matched
    closed-form surrogate (:func:`moment_matched_surrogate`, fit against that round's
    responsibility-weighted data), which is then updated by a closed-form re-fit every round
    instead of gradient descent -- and swapped back to the retained original the instant a real
    misfit receipt (:func:`misfit_receipt` on ``holdout_data``, when supplied) shows it has drifted
    (see :func:`should_swap_back`).

    Reuses D2's ``FreezeRollupCache``/``detect_frozen`` for ordinary converged-component freezing
    (composes with D4 exactly like D3 does: a frozen component is excluded from both the swap
    check and the M-step) and the same per-round objective accept/reject gate D2/D3 use, so
    ``history`` is a real monotone-F receipt: swapping in a surrogate, or swapping back out of one,
    can only ever be accepted if the round's real Neal-Hinton objective does not decrease.

    ``holdout_data``, if given, is treated as belonging to whichever component the CURRENT model
    would assign it to most responsibly at swap time (a single fixed sample scored against every
    swapped component's surrogate) -- a simplification documented here rather than a full
    per-component responsibility-weighted holdout split, mirroring D2/D3's own explicit
    single-combinator scope carve-outs (see this module's docstring).

    Returns ``(final_model, history, swap_records)`` where ``swap_records`` maps component index to
    every :class:`SwapRecord` created during the run (including ones later swapped back), so a
    caller/test can retrieve the retained original gradient leaf and inspect ``misfit_history``.
    """
    if not isinstance(initial_model, MixtureDistribution):
        raise TypeError("run_em_with_hotswap requires a MixtureDistribution model.")
    cache = FreezeRollupCache(q_gain_tol=freeze_q_gain_tol) if cache is None else cache
    monitor = (
        PlateauMonitor(q_gain_tol=plateau_q_gain_tol, patience=plateau_patience, n_mc=plateau_n_mc)
        if monitor is None
        else monitor
    )
    enc_payload = _resolve_payload(enc_data)
    holdout_matrix = None if holdout_data is None else _as_matrix(holdout_data)

    model = initial_model
    history: list[LeafHotswapStats] = []
    swap_records: dict[int, SwapRecord] = {}
    swapped_idx: set[int] = set()
    old_value: float | None = None

    for round_index in range(max(1, int(max_its))):
        frozen_idx = detect_frozen(cache, model)

        swapped_this_round: list[int] = []
        swapped_back_this_round: list[int] = []

        # -- misfit check first: a surrogate that has drifted swaps back BEFORE this round's
        #    E-step, so the (re-activated) gradient leaf's fresh log-density is what this round
        #    actually scores/updates against, not a stale cached surrogate value.
        for idx in list(swapped_idx):
            record = swap_records[idx]
            if holdout_matrix is None:
                continue
            current = misfit_receipt(model.components[idx], holdout_matrix)
            record.misfit_history.append(current)
            if should_swap_back(record, current, misfit_tol=misfit_tol):
                model = swap_back(model, record, round_index=round_index)
                swapped_idx.discard(idx)
                monitor.reset(idx)
                swapped_back_this_round.append(idx)

        # -- plateau check: any eligible (not frozen, not already swapped) gradient leaf that has
        #    plateaued gets swapped in for a moment-matched surrogate fit on THIS round's E-step
        #    responsibilities (computed just below) -- so the swap needs one E-step first.
        ll_mat, evals_e = _component_log_density_matrix(model, enc_payload, cache, frozen_idx | swapped_idx)
        log_density, gamma = _combine(ll_mat, model.log_w)
        current_value = float(np.sum(log_density))
        if old_value is None:
            old_value = current_value

        # ``pre_swap_model`` is the receipt of "the tree as it stood when ``current_value`` was
        # measured" -- the ONLY state the round's accept/reject gate below is entitled to fall back
        # to. A newly-triggered plateau swap is proposed into ``model`` below (so the very same
        # round's closed-form M-step can immediately re-fit it against fresh responsibilities), but
        # that proposal is provisional exactly like the M-step's own ``candidate``: if the round is
        # rejected, EVERYTHING proposed this round -- the M-step AND the swap-in -- must roll back
        # together, or a swap that measurably worsens the objective would silently survive into the
        # returned model even though ``accepted`` reads False for that round (the bug this comment
        # replaces: ``model`` used to be rebound in place by the swap loop below with no matching
        # rollback path, so a rejected round still returned the swapped-in surrogate).
        pre_swap_model = model
        new_swap_ids: list[int] = []

        for idx in range(model.num_components):
            if idx in frozen_idx or idx in swapped_idx or model.zw[idx]:
                continue
            component = model.components[idx]
            if not isinstance(component, GradLeaf):
                continue
            if not monitor.is_plateaued(idx, component, nobs=1.0):
                continue
            enc_i = _component_enc(enc_payload, idx)
            surrogate = moment_matched_surrogate(component, enc_i, weights=gamma[:, idx])
            model, record = swap_leaf(model, idx, surrogate, round_index=round_index)
            if holdout_matrix is not None:
                record.baseline_misfit = misfit_receipt(surrogate, holdout_matrix)
            swap_records[idx] = record
            swapped_idx.add(idx)
            swapped_this_round.append(idx)
            new_swap_ids.append(idx)

        candidate, n_grad, n_closed = _m_step_hotswap(enc_payload, estimator, model, gamma, frozen_idx, swapped_idx)
        candidate_frozen = detect_frozen(cache, candidate)
        ll_mat_c, evals_c = _component_log_density_matrix(candidate, enc_payload, cache, candidate_frozen | swapped_idx)
        candidate_log_density, _ = _combine(ll_mat_c, candidate.log_w)
        candidate_value = float(np.sum(candidate_log_density))

        accepted = np.isfinite(candidate_value) and candidate_value + accept_tolerance >= current_value
        if accepted:
            model = candidate
            round_value = candidate_value
        else:
            # Roll back not just the M-step but this round's provisional swap-in(s) too: the gate
            # rejected the round's objective, so the surrogate that produced it never earns a place
            # in the returned model. The retained gradient leaf(s) go back into ``model`` exactly as
            # :func:`swap_back` would restore them, the streak resets so the plateau monitor can
            # retry cleanly next round, and any never-committed ``SwapRecord``/cache entry from this
            # round's rejected proposal is discarded rather than left to look like a real swap.
            model = pre_swap_model
            for idx in new_swap_ids:
                swapped_idx.discard(idx)
                swap_records.pop(idx, None)
                cache.invalidate(idx)
                monitor.reset(idx)
            round_value = current_value

        history.append(
            LeafHotswapStats(
                round_index=round_index,
                n_components=model.num_components,
                n_frozen=len(frozen_idx),
                n_swapped=len(swapped_idx),
                n_log_density_evals=evals_e + evals_c,
                n_gradient_m_steps=n_grad,
                n_closed_form_m_steps=n_closed,
                objective=round_value,
                accepted=accepted,
                swapped_this_round=tuple(swapped_this_round),
                swapped_back_this_round=tuple(swapped_back_this_round),
            )
        )

        if delta is not None and 0.0 <= round_value - old_value < delta:
            break
        old_value = round_value

    return model, history, swap_records
