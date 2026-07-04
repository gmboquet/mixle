"""The knowledge substrate -- one typed, provenanced, scoped store over everything the ecosystem knows.

The local shard (workstream O1 of the frontier ecosystem plan): raw data, documents, model artifacts,
harvested traces, and (later) ontology triples / simulation outputs / context packets all live here as
typed :class:`SubstrateItem` s with provenance and access scope, retrievable through one ``search``.
This is the foundation the all-data RAG (S), context assembly (O2), and team-sharing (P) workstreams
build on.
"""

from __future__ import annotations

from mixle.substrate.answer import Answer, answer_from_substrate
from mixle.substrate.context import ContextBudget, ContextPacket, assemble_context, compress_text
from mixle.substrate.core import MODALITIES, Substrate, SubstrateItem
from mixle.substrate.ingest import ingest_artifacts, ingest_documents, ingest_traces
from mixle.substrate.multihop import HopChain, HopStep, multihop
from mixle.substrate.retrieve import Retrieval, retrieve

__all__ = [
    "MODALITIES",
    "Substrate",
    "SubstrateItem",
    "ingest_documents",
    "ingest_artifacts",
    "ingest_traces",
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
]
