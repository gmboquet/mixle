"""``mixle.reason`` -- the cross-modal scientific-reasoning front door.

A scientific question is a query on a joint posterior over a shared latent that every modality is
evidence about. This package wires that idea into one call:

    answer = reason(prior, [evidence_from_modality_1, evidence_from_modality_2, ...])
    answer.mean, answer.interval(0.9)      # posterior with calibrated intervals
    answer.attribution()                   # which modality sharpened the belief (nats)
    answer.predict(H, R).epistemic         # split a prediction's uncertainty (epi vs aleatoric)

The exact core here is linear-Gaussian: each modality contributes a linear-Gaussian observation
``y = H z + noise(R)`` and the beliefs fuse by exact Kalman assimilation (a product of experts).
Learned encoders and application-specific forward models plug in by *producing* such evidence -- a
linearized ``(H, y, R)`` or a Gaussian expert -- so the front door stays stable while encoders vary
underneath it.

Built on :mod:`mixle.inference.belief` (the belief state) and
:mod:`mixle.inference.uncertainty` (the epistemic/aleatoric split).
"""

from __future__ import annotations

from typing import Any

from mixle.inference.belief import BeliefState, GaussianBelief, as_belief
from mixle.reason.anchor_harness import AnchorHarnessReport, run_anchor_harness
from mixle.reason.belief_walk import HopTransport, WalkResult, belief_walk, coverage_by_hop_count
from mixle.reason.core import (
    Evidence,
    Latent,
    LinearGaussianEvidence,
    NonlinearEvidence,
    ReasonedAnswer,
    block_selector,
    reason,
)
from mixle.reason.cross_modal import CrossModalJoint
from mixle.reason.cycle_consistency import (
    cycle_inconsistency,
    fit_cycle_transport,
    joint_cycle_consistency_receipt,
    posterior_mean_estimate,
    selective_error,
)
from mixle.reason.design import AcquisitionPlan, select_evidence_batch
from mixle.reason.discrete import DiscreteAnswer, model_evidence, reason_discrete
from mixle.reason.graph_llm import (
    GraphDistribution,
    GraphLLM,
    canonical_graph,
    fit_fact_calibrator,
)
from mixle.reason.llm import (
    ClaimAssessment,
    FactualityModel,
    InformationAssessment,
    LLMAssessment,
    LLMUncertainty,
    content_overlap,
    information_corroborator,
    sentence_claims,
)
from mixle.reason.modality import ModalityGraph, ModalityView
from mixle.reason.ontology import (
    AXIOMS,
    ConstrainedDecode,
    Ontology,
    OntologyConstrainedKG,
    constrained_decode,
)
from mixle.reason.store import CrossModalStore, RetrievalStep
from mixle.reason.task_projection import TaskReadout, read_out, task_sufficient_projection
from mixle.reason.transport_edge import (
    EdgeTransportVerdict,
    coverage_consistent_with_nominal,
    fit_conditional_transport,
    marginal_coverage,
    verify_edge_transport,
)

__all__ = [
    "AnchorHarnessReport",
    "run_anchor_harness",
    "Ontology",
    "OntologyConstrainedKG",
    "constrained_decode",
    "ConstrainedDecode",
    "AXIOMS",
    "NonlinearEvidence",
    "DiscreteAnswer",
    "model_evidence",
    "reason_discrete",
    "reason",
    "Latent",
    "Evidence",
    "LinearGaussianEvidence",
    "block_selector",
    "ReasonedAnswer",
    "GaussianBelief",
    "BeliefState",
    "AcquisitionPlan",
    "select_evidence_batch",
    "as_belief",
    "CrossModalStore",
    "RetrievalStep",
    "LLMUncertainty",
    "LLMAssessment",
    "ClaimAssessment",
    "InformationAssessment",
    "FactualityModel",
    "sentence_claims",
    "content_overlap",
    "ModalityView",
    "ModalityGraph",
    "TaskReadout",
    "task_sufficient_projection",
    "read_out",
    "cycle_inconsistency",
    "fit_cycle_transport",
    "posterior_mean_estimate",
    "selective_error",
    "CrossModalJoint",
    "joint_cycle_consistency_receipt",
    "HopTransport",
    "WalkResult",
    "belief_walk",
    "coverage_by_hop_count",
    "information_corroborator",
    "GraphLLM",
    "GraphDistribution",
    "canonical_graph",
    "fit_fact_calibrator",
    "AmortizedEncoder",
    "ScaledEmbedding",
    "CrossModalModel",
    "StructuredAdapter",
    "ProductOfExpertsFusion",
    "StructuredFusionClassifier",
    "HybridFusionClassifier",
    "fusion_flops",
    "EdgeTransportVerdict",
    "coverage_consistent_with_nominal",
    "fit_conditional_transport",
    "marginal_coverage",
    "verify_edge_transport",
]

_LAZY = {
    "AmortizedEncoder": "encoder",
    "ScaledEmbedding": "embedding",
    "CrossModalModel": "model",
    "StructuredAdapter": "adapter",
    "ProductOfExpertsFusion": "fusion",
    "StructuredFusionClassifier": "fusion",
    "HybridFusionClassifier": "fusion",
    "fusion_flops": "fusion",
}


def __getattr__(name: str) -> Any:
    # Lazy: defer importing the torch-backed modules (and building their nets) until first access, so
    # the exact linear-Gaussian core here does not construct a torch model just to be imported.
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(f"mixle.reason.{_LAZY[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
