"""Experimental Mixle surfaces with explicit evidence boundaries.

Nothing in this package is part of Mixle's stable API, and presence in this inventory is not a production,
scaling, statistical-validity, or release-readiness guarantee.

The machine-readable :data:`EXPERIMENTAL_INVENTORY` uses four maturity labels:

``prototype``
    Code and an executable focused specification exist, but this inventory asserts no validated guarantee.
``locally_receipted``
    Only the narrow ``purpose`` statement was exercised by the listed focused tests at the exact revision in
    the cited durable status evidence. It is not a broader production or scientific-validity claim.
``unvalidated``
    Known audit blockers remain. No behavioral guarantee should be inferred from names, types, or docstrings.
``bookkeeping_only``
    Metadata machinery that records evidence; it does not perform or validate the underlying experiment.

Evidence IDs refer to the separate durable ``status`` ledger. Test paths are executable acceptance
specifications, not proof that an arbitrary checkout currently passes them. Callers must validate the exact
revision they deploy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentalSurface:
    """One experimental surface's narrow scope and validation boundary."""

    module: str
    purpose: str
    maturity: str
    acceptance_tests: tuple[str, ...] = ()
    evidence_id: str | None = None
    limitations: tuple[str, ...] = ()


EXPERIMENTAL_INVENTORY: tuple[ExperimentalSurface, ...] = (
    ExperimentalSurface(
        "active_causal",
        "Synthetic three-graph active causal-discovery prototype.",
        "prototype",
        ("mixle/tests/active_causal_test.py", "mixle/tests/active_causal_contract_test.py"),
    ),
    ExperimentalSurface(
        "certified_bounds",
        "Interval propagation and monotonicity checks for a restricted Gaussian/mixture grammar.",
        "prototype",
        ("mixle/tests/certified_bounds_test.py", "mixle/tests/certified_bounds_contract_test.py"),
    ),
    ExperimentalSurface(
        "context_parallel_spine",
        "Small-scale context-parallel reference for the sliding-window spine.",
        "prototype",
        ("mixle/tests/context_parallel_spine_test.py",),
        limitations=("No production multi-axis scaling guarantee.",),
    ),
    ExperimentalSurface(
        "context_spine",
        "Chunked-recurrent training protocol and local sliding-window baseline.",
        "prototype",
        ("mixle/tests/context_spine_test.py", "mixle/tests/context_spine_contract_test.py"),
    ),
    ExperimentalSurface(
        "cvi",
        "Conjugate-computation variational-update prototype for selected exponential families.",
        "prototype",
        ("mixle/tests/cvi_test.py", "mixle/tests/cvi_contract_test.py"),
    ),
    ExperimentalSurface(
        "e_process",
        "Anytime-monitoring primitives under their stated density-ratio assumptions.",
        "prototype",
        ("mixle/tests/e_process_test.py", "mixle/tests/e_process_contract_test.py"),
    ),
    ExperimentalSurface(
        "equation_discovery",
        "Synthetic scalar SINDy-style operator-discovery experiment.",
        "prototype",
        ("mixle/tests/equation_discovery_test.py", "mixle/tests/equation_discovery_contract_test.py"),
    ),
    ExperimentalSurface(
        "graduation",
        "Bookkeeping for locally supplied experiment receipts.",
        "bookkeeping_only",
        ("mixle/tests/experimental_scaffold_test.py",),
        limitations=("Eligibility metadata does not execute an evaluation.",),
    ),
    ExperimentalSurface(
        "growth_operators",
        "Candidate transformer growth operators with local parity receipts.",
        "prototype",
        ("mixle/tests/growth_operators_test.py", "mixle/tests/growth_operators_contract_test.py"),
    ),
    ExperimentalSurface(
        "kv_cache_quant",
        "KV-cache quantization and tail-modeling prototypes.",
        "prototype",
        ("mixle/tests/kv_cache_quant_test.py", "mixle/tests/kv_cache_quant_contract_test.py"),
    ),
    ExperimentalSurface(
        "law_discovery",
        "Candidate functional-form selection for synthetic simulator input/output data.",
        "prototype",
        ("mixle/tests/law_discovery_test.py",),
    ),
    ExperimentalSurface(
        "long_context_eval",
        "Synthetic long-context probes and matched-budget bookkeeping.",
        "prototype",
        ("mixle/tests/long_context_eval_test.py", "mixle/tests/long_context_eval_contract_test.py"),
        limitations=("Probe scores are not a general long-context capability certificate.",),
    ),
    ExperimentalSurface(
        "model_economy",
        "Synthetic held-out verification of exchanged fitted components.",
        "prototype",
        ("mixle/tests/model_economy_test.py", "mixle/tests/model_economy_contract_test.py"),
    ),
    ExperimentalSurface(
        "moment_closure_attention",
        "Learned moment-closure far-field attention prototype.",
        "prototype",
        ("mixle/tests/moment_closure_attention_test.py",),
    ),
    ExperimentalSurface(
        "ot_geometry",
        "Gaussian optimal-transport geometry and finite-mixture transport prototypes.",
        "prototype",
        ("mixle/tests/ot_geometry_test.py", "mixle/tests/ot_geometry_contract_test.py"),
    ),
    ExperimentalSurface(
        "pac_bayes",
        "Finite fixed-hypothesis categorical PAC-Bayes calculation under explicit assumptions.",
        "locally_receipted",
        ("mixle/tests/pac_bayes_test.py",),
        "EVID-20260725-0133",
        ("Not a certificate for fitted observation-mixture components or data-dependent priors.",),
    ),
    ExperimentalSurface(
        "program",
        "Legacy closure-based optimization-program prototype with fail-closed local contracts.",
        "prototype",
        ("mixle/tests/program_test.py", "mixle/tests/program_contract_test.py"),
        limitations=("Superseded for common cases by declarative neural/statistical fitting.",),
    ),
    ExperimentalSurface(
        "quantized_key_attention",
        "Product-quantized-key attention prototype.",
        "prototype",
        ("mixle/tests/quantized_key_attention_test.py", "mixle/tests/quantized_key_attention_contract_test.py"),
    ),
    ExperimentalSurface(
        "retrieval_memory_spine",
        "Detached frozen-past retrieval-memory prototype.",
        "prototype",
        ("mixle/tests/retrieval_memory_spine_test.py", "mixle/tests/retrieval_memory_contract_test.py"),
    ),
    ExperimentalSurface(
        "selective_scan",
        "Selective-scan sequence-module prototype.",
        "prototype",
        ("mixle/tests/selective_scan_test.py", "mixle/tests/selective_scan_contract_test.py"),
    ),
    ExperimentalSurface(
        "sketch_state_attention",
        "Frequent-Directions and normalized TensorSketch far-state references.",
        "locally_receipted",
        ("mixle/tests/sketch_state_attention_test.py", "mixle/tests/sketch_state_attention_contract_test.py"),
        "EVID-20260725-0138",
        ("TensorSketch signed estimates are not non-negative attention probabilities.",),
    ),
    ExperimentalSurface(
        "spectral_health",
        "Descriptive weight-spectrum statistics with gated tail-fit uncertainty.",
        "locally_receipted",
        ("mixle/tests/spectral_health_test.py",),
        "EVID-20260725-0139",
        ("No training-quality or memorization diagnosis is certified.",),
    ),
    ExperimentalSurface(
        "ssm_hybrid",
        "Local-attention, selective-scan, and moment-bank hybrid with routing-mass accounting.",
        "locally_receipted",
        ("mixle/tests/ssm_hybrid_test.py",),
        "EVID-20260725-0140",
        ("Routing mass is not output attribution.",),
    ),
    ExperimentalSurface(
        "structure_edit_schedule",
        "Whole-model candidate edits with parity gating and optimizer-state migration.",
        "locally_receipted",
        ("mixle/tests/structure_edit_schedule_test.py",),
        "EVID-20260725-0141",
    ),
    ExperimentalSurface(
        "summary_tree",
        "Test-scale non-overlapping mixed-radix summary frontier with conservation receipts.",
        "locally_receipted",
        ("mixle/tests/summary_tree_test.py",),
        "EVID-20260725-0142",
        ("Bounded-state evidence is focused and does not establish production quality or scaling.",),
    ),
    ExperimentalSurface(
        "tensor_network",
        "Finite discrete MPS/Born probability and marginal reference.",
        "locally_receipted",
        ("mixle/tests/tensor_network_test.py",),
        "EVID-20260725-0143",
        ("No large-scale tensor-network training guarantee.",),
    ),
    ExperimentalSurface(
        "tensor_pipeline_context_parallel",
        "Trainable in-process TP/PP/CP reference with explicit optimizer ownership and memory receipts.",
        "locally_receipted",
        ("mixle/tests/tensor_pipeline_context_parallel_test.py",),
        "EVID-20260725-0144",
        ("Not a production distributed multi-axis trainer.",),
    ),
    ExperimentalSurface(
        "tying_discovery",
        "Compatible weight-tie discovery and isolated parity-budget evaluation.",
        "locally_receipted",
        ("mixle/tests/tying_discovery_test.py",),
        "EVID-20260725-0145",
        ("Profile similarity is lossy and is not an exact permutation or parity certificate.",),
    ),
    ExperimentalSurface(
        "unlearning",
        "Commitment-backed retained-statistics re-reduction for three audited estimators.",
        "locally_receipted",
        ("mixle/tests/unlearning_test.py",),
        "EVID-20260725-0146",
        ("Does not prove physical erasure, truthful ingestion, or cleanup of other artifacts.",),
    ),
    ExperimentalSurface(
        "v_information",
        "Finite-sample polynomial-Gaussian usable-information estimates.",
        "locally_receipted",
        ("mixle/tests/v_information_test.py",),
        "EVID-20260725-0147",
        ("Repeated-split uncertainty covers holdout assignment only, not the population optimum.",),
    ),
    ExperimentalSurface(
        "wake_sleep",
        "Exact jointly supported fragments in a synthetic column-itemset grammar.",
        "locally_receipted",
        ("mixle/tests/wake_sleep_test.py",),
        "EVID-20260725-0148",
        ("Not a general program anti-unifier or external-task search-speed certificate.",),
    ),
    ExperimentalSurface(
        "typed_runtime",
        "Prototype typed optimization-runtime and coordination components.",
        "unvalidated",
        limitations=(
            "Open exhaustive-audit findings beginning at MXR-080-0630 invalidate end-to-end guarantees.",
            "Contract names and receipts must not be treated as proof until those findings are reconciled.",
        ),
    ),
)


__all__ = ["EXPERIMENTAL_INVENTORY", "ExperimentalSurface"]
