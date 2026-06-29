"""Production / MLOps layer for fitted mixle models.

The deployment-facing half of inference, kept out of the core ``mixle.inference`` namespace: reproducible
model artifacts (provenance headers + verifiable training lineage), drift detection, a versioned model
registry, a scoring service, and a drift-triggered monitor.

  - **provenance**: ``fit_with_provenance`` -> a ``Header`` (config, data hash, model hash, convergence
    trace, timing, env); ``verify_lineage`` checks the per-iteration model-hash chain.
  - **drift**: ``detect_drift`` -> ``DriftReport`` (feature PSI/KS + model score drift).
  - **registry / serving / monitor**: ``Registry`` (versioned store + alias swap + checkpointer),
    ``Service`` (batch scoring + activity logging), ``Monitor`` (drift -> retrain -> swap).

The container/Kubernetes serving layer that wraps ``Service`` lives in the separate ``mixle-deploy``
package.
"""

from __future__ import annotations

from mixle.inference.production.drift import DriftReport, detect_drift, score_drift
from mixle.inference.production.monitor import Monitor
from mixle.inference.production.provenance import (
    Header,
    build_header,
    environment_info,
    fit_with_provenance,
    verify_lineage,
)
from mixle.inference.production.registry import Registry
from mixle.inference.production.serving import Service

__all__ = [
    "fit_with_provenance",
    "verify_lineage",
    "Header",
    "build_header",
    "environment_info",
    "detect_drift",
    "score_drift",
    "DriftReport",
    "Registry",
    "Service",
    "Monitor",
]
