"""The knowledge substrate -- one typed, provenanced, scoped store over everything the ecosystem knows.

The local shard (workstream O1 of the frontier ecosystem plan): raw data, documents, model artifacts,
harvested traces, and (later) ontology triples / simulation outputs / context packets all live here as
typed :class:`SubstrateItem` s with provenance and access scope, retrievable through one ``search``.
This is the foundation the all-data RAG (S), context assembly (O2), and team-sharing (P) workstreams
build on.
"""

from __future__ import annotations

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
from mixle.substrate.context import ContextBudget, ContextPacket, assemble_context, compress_text
from mixle.substrate.core import MODALITIES, Substrate, SubstrateItem
from mixle.substrate.factuality import ClaimVerdict, FactualityReceipt, check_factuality
from mixle.substrate.governance import Governance, approve, pending, propose, reject
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
    "assemble_context",
    "compress_text",
    "retrieve",
    "Retrieval",
    "multihop",
    "HopChain",
    "HopStep",
    "answer_from_substrate",
    "Answer",
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
]
