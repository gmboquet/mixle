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
