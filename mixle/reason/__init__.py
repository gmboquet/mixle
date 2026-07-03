"""``mixle.reason`` -- the cross-modal scientific-reasoning front door.

A scientific question is a query on a joint posterior over a shared latent that every modality is
evidence about. This package wires that idea into one call:

    answer = reason(prior, [evidence_from_modality_1, evidence_from_modality_2, ...])
    answer.mean, answer.interval(0.9)      # a posterior, with honest error bars
    answer.attribution()                   # which modality sharpened the belief (nats)
    answer.predict(H, R).epistemic         # split a prediction's uncertainty (epi vs aleatoric)

The exact core here is linear-Gaussian: each modality contributes a linear-Gaussian observation
``y = H z + noise(R)`` and the beliefs fuse by exact Kalman assimilation (a product of experts).
Non-linear / learned encoders (Phase 3+, and application-specific forward models in the sibling
``mixle_pde`` package) plug in by *producing* such evidence -- a linearized ``(H, y, R)`` or a
Gaussian expert -- so the front door is stable while the encoders grow underneath it.

Design: notes/mixle-cross-modal-reasoning-design.md. Built on :mod:`mixle.inference.belief`
(the belief state) and :mod:`mixle.inference.uncertainty` (the epistemic/aleatoric split).
"""

from __future__ import annotations

from typing import Any

from mixle.inference.belief import BeliefState, GaussianBelief, as_belief
from mixle.reason.core import (
    Evidence,
    Latent,
    LinearGaussianEvidence,
    NonlinearEvidence,
    ReasonedAnswer,
    block_selector,
    reason,
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
from mixle.reason.store import CrossModalStore, RetrievalStep

__all__ = [
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
