"""Typed, provenanced, scoped storage for local knowledge and artifacts.

The substrate stores raw data, documents, model artifacts, harvested traces,
ontology triples, simulation outputs, and context packets as
:class:`SubstrateItem` objects. Each item carries a kind, provenance, access
scope, tags, links, and a retrievable text surface.
"""

from __future__ import annotations

from mixle.substrate.accum import FlywheelMeasurement, FlywheelReport, QAItem, measure_flywheel
from mixle.substrate.act import (
    Action,
    Investigation,
    Step,
    action_features,
    compute_action,
    create_action,
    delegate_action,
    investigate,
    relevance_of,
    retrieve_action,
    score_action,
    simulate_action,
)
from mixle.substrate.answer import Answer, answer_from_substrate
from mixle.substrate.belief import (
    MODEL_ASSERTION,
    MODEL_ASSERTION_CAP,
    BeliefItem,
    Claim,
    EvidenceEntry,
    assimilate,
    credence_from_history,
    harvest_knowledge,
    retract,
    retrieve_beliefs,
)
from mixle.substrate.context import (
    ContextBudget,
    ContextPacket,
    ReceiverProfile,
    assemble_context,
    assemble_for_receivers,
    compress_text,
)
from mixle.substrate.core import MODALITIES, Substrate, SubstrateItem
from mixle.substrate.eig_retrieve import eig_retrieve
from mixle.substrate.factuality import ClaimVerdict, FactualityReceipt, check_factuality
from mixle.substrate.freshness import Freshness, check_freshness, content_hash, freshness_report
from mixle.substrate.governance import Governance, approve, pending, propose, reject
from mixle.substrate.harness import (
    Harness,
    HarnessResult,
    find_harnesses,
    monitoring_harness,
    register_harness,
    support_triage_harness,
)
from mixle.substrate.ingest import (
    ingest_artifacts,
    ingest_documents,
    ingest_file,
    ingest_records,
    ingest_traces,
)
from mixle.substrate.interop import ExternalAnswer, ExternalModel, external_action
from mixle.substrate.kg_rag import kg_action, link_entities, retrieve_triples
from mixle.substrate.multihop import HopChain, HopStep, multihop
from mixle.substrate.reasoner import Reasoner
from mixle.substrate.retrieve import Retrieval, retrieve
from mixle.substrate.security import (
    SecretFinding,
    SecretScan,
    detect_secrets,
    redact_secrets,
    safe_text,
    scan_item,
    scan_substrate,
)
from mixle.substrate.spaces import PUBLIC, Space, history, merge_versions, publish, version_of, visible_scopes
from mixle.substrate.trust import LineageReport, audit_substrate, verify_lineage

__all__ = [
    "MODALITIES",
    "Substrate",
    "SubstrateItem",
    "ingest_documents",
    "ingest_artifacts",
    "ingest_traces",
    "ingest_file",
    "ingest_records",
    "ContextPacket",
    "ContextBudget",
    "ReceiverProfile",
    "assemble_context",
    "assemble_for_receivers",
    "compress_text",
    "retrieve",
    "Retrieval",
    "eig_retrieve",
    "multihop",
    "HopChain",
    "HopStep",
    "answer_from_substrate",
    "Answer",
    "measure_flywheel",
    "FlywheelReport",
    "FlywheelMeasurement",
    "QAItem",
    "investigate",
    "Investigation",
    "Action",
    "Step",
    "score_action",
    "relevance_of",
    "action_features",
    "retrieve_action",
    "compute_action",
    "simulate_action",
    "create_action",
    "delegate_action",
    "Reasoner",
    "check_factuality",
    "FactualityReceipt",
    "ClaimVerdict",
    "Space",
    "publish",
    "visible_scopes",
    "merge_versions",
    "history",
    "version_of",
    "PUBLIC",
    "verify_lineage",
    "audit_substrate",
    "LineageReport",
    "Governance",
    "propose",
    "approve",
    "reject",
    "pending",
    "detect_secrets",
    "redact_secrets",
    "safe_text",
    "scan_item",
    "scan_substrate",
    "SecretScan",
    "SecretFinding",
    "ExternalModel",
    "ExternalAnswer",
    "external_action",
    "kg_action",
    "link_entities",
    "retrieve_triples",
    "check_freshness",
    "freshness_report",
    "content_hash",
    "Freshness",
    "Harness",
    "HarnessResult",
    "support_triage_harness",
    "monitoring_harness",
    "register_harness",
    "find_harnesses",
    "harvest_knowledge",
    "assimilate",
    "retract",
    "retrieve_beliefs",
    "credence_from_history",
    "BeliefItem",
    "Claim",
    "EvidenceEntry",
    "MODEL_ASSERTION",
    "MODEL_ASSERTION_CAP",
]
