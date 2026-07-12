"""Cost-model policy: when should a default-engine fit switch to the single-pass fused numba kernel?

This lives apart from ``estimation.py`` on purpose. The high-level fitting machinery is required (by the
``compute_metadata`` architectural guard) to depend only on abstract compute protocols, not on a concrete
kernel implementation like :mod:`mixle.stats.compute.fused_codegen`. This module is the one place that is
allowed to know about fusion, so it owns the fusibility query and the workload threshold.

Below ``_FUSION_MIN_WORKLOAD`` (observations x iterations) the fused kernel's one-time numba compile
(~0.1s, then disk-cached per model structure) is not amortized, so the fit stays on the host path.
Measured crossover: fusion only breaks even on a cold compile around 2-6e6 obs-iters, and is faster (~1.7x)
once warm/cached -- so this conservative gate never slows a small/medium fit while auto-using fusion for
large or repeated workloads. Parity (fused == host) is guaranteed by the fused_codegen / fused_em tests.
"""

from __future__ import annotations

from typing import Any

_FUSION_MIN_WORKLOAD = 1_500_000


def has_fusion_benefit(model: Any) -> bool:
    """A multi-factor model where single-pass fusion eliminates real per-leaf dispatch (a mixture of >1
    component, or a composite/record of >1 field). A bare leaf has nothing to fuse, so it stays on host."""
    comps = getattr(model, "components", None)
    if comps is not None and len(comps) > 1:
        return True
    dists = getattr(model, "dists", None)
    return dists is not None and len(dists) > 1


def should_auto_fuse(model: Any, enc_data: Any, max_its: int) -> bool:
    """True if the default-engine local MLE path should switch to the fused numba kernel for ``model``."""
    try:
        from mixle.utils.optional_deps import HAS_NUMBA

        # Fusion compiles a numba kernel (lazily, deep in fused_codegen), so a missing numba would only
        # surface as a crash mid-fit. Decline here and stay on the host path -- identical result, no numba.
        if not HAS_NUMBA:
            return False
        from mixle.stats.compute.fused_codegen import fusible, fusible_estep

        if not (has_fusion_benefit(model) and fusible(model) and fusible_estep(model)):
            return False
        n = sum(int(c[0]) for c in enc_data) if isinstance(enc_data, list) else 0
        return n * max(int(max_its), 1) >= _FUSION_MIN_WORKLOAD
    except Exception:  # noqa: BLE001
        return False


_BLOCK_MIN_COMPONENTS = 16
_BLOCK_MIN_COMPONENT_PARAMS = 32
_BLOCK_MIN_WEIGHTED_WORK = 5_000_000


def prefer_block_schedule(model: Any, enc_data: Any, max_its: int) -> bool:
    """Whether ``schedule="auto"`` should route an ELIGIBLE mixture to block-EM rather than the
    full-tree path (which auto-fuses when it can).

    The measured decision boundary (2026-07-12, 40k observations, 30 rounds): when the WHOLE model
    fuses, the single-pass kernel computes every component column + the log-sum-exp in one sweep of
    the data and nothing block scheduling saves can beat it (K=32 well-separated: fused full-tree
    0.15s vs block-EM 0.66-0.79s under every cost model) -- per-COLUMN fusion cannot capture the
    cross-component fusion win. Block-EM's niche is the models with NO whole-model kernel (deep
    heterogeneous components: HMMs, neural leaves, non-templated families), where sparse selection
    beat host full-tree EM by ~2.1x on a depth-21/247-node reproducer. So: block only when the
    model does not fuse whole, with enough components to amortize the per-round orchestration and
    enough workload to matter (the same floor :func:`should_auto_fuse` uses).
    """
    components = getattr(model, "components", None)
    if components is None or len(components) < 2:
        return False
    try:
        from mixle.utils.optional_deps import HAS_NUMBA

        if HAS_NUMBA:
            from mixle.stats.compute.fused_codegen import fusible

            if fusible(model):
                return False  # the whole-model fused kernel wins outright; see the docstring numbers
    except Exception:  # noqa: BLE001 - fusibility probe must never break dispatch
        pass
    # Scheduling amortizes over the COST a skipped block avoids, not the component count per se:
    # the depth-21 reproducer wins with only 3 components because each column is a deep composite
    # (hundreds of parameters), while a 4-leaf flat mixture's columns are too cheap to be worth
    # per-round orchestration. Either many blocks or expensive blocks qualify, and the workload
    # floor is parameter-WEIGHTED for the same reason (observations x iterations x mean component
    # parameter count approximates the full-tree work block selection gets to skip).
    from mixle.inference.block_em import _parameter_count

    param_counts = [max(_parameter_count(component), 1) for component in components]
    if len(components) < _BLOCK_MIN_COMPONENTS and max(param_counts) < _BLOCK_MIN_COMPONENT_PARAMS:
        return False
    n = sum(int(c[0]) for c in enc_data) if isinstance(enc_data, list) else 0
    weighted_work = n * max(int(max_its), 1) * (sum(param_counts) / len(param_counts))
    return weighted_work >= _BLOCK_MIN_WEIGHTED_WORK
