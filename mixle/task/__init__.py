"""Local task-specific models with durable artifacts and calibrated serving.

The unit is a :class:`~mixle.task.model.TaskModel`: a fitted model scoped to *one* task (classify, extract,
recommend a model shape, ...), scoped to one operational behavior and saved as a durable artifact
(:mod:`~mixle.task.artifact`) so a plain Python program can load it in a fresh process and call it. Producers:

  * :func:`~mixle.task.distill.distill` -- a teacher callable labels data and a local student is fit to match;
  * :func:`~mixle.task.tune.tune_recipe` -- ``mixle.doe`` searches the student recipe to minimize train cost.

This module's public surface re-exports the artifact contract; the model/distill/tune layers land on top.
"""

from __future__ import annotations

from mixle.task.active import ActiveResult, acquisition_scores, active_distill
from mixle.task.artifact import (
    SCHEMA_VERSION,
    TaskManifest,
    get_arrays_builder,
    get_builder,
    load_arrays,
    load_json,
    load_module,
    read_manifest,
    register_arrays_builder,
    register_builder,
    save_arrays,
    save_json,
    save_module,
)
from mixle.task.bandit import UCB1, EstimatorBandit, ThompsonBernoulli, ThompsonGaussian
from mixle.task.calibrate import ESCALATE, CalibratedTaskModel
from mixle.task.capability import (
    CapabilitySuite,
    capture_profile,
    case_jitter_invariance,
    keyboard_typo_corruption,
    whitespace_invariance,
)
from mixle.task.capacity import (
    DEFAULT_RUNGS,
    KNOWN_RUNGS,
    EmbeddingHeadIO,
    LadderResult,
    RungResult,
    WordEmbeddingFeaturizer,
    capacity_ladder,
    climb_to,
)
from mixle.task.cascade import Cascade, CascadeStats
from mixle.task.collapse import (
    CollapseVerdict,
    collapse_monitor,
    distinct_count_diversity,
    entropy_diversity,
)
from mixle.task.compose import ComposedAnswer, ComposedModel, compose
from mixle.task.data_mixture import SyntheticDomain, estimate_near_duplicate_rate, optimize_mixture, proxy_run_score
from mixle.task.density import DensityGate
from mixle.task.design import DesignedModel, design_model, spec_to_estimator
from mixle.task.design_prior import best_family, rank_design_families, record_accepted_recipe
from mixle.task.disagreement import DisagreementGate, UnionGate, fit_disagreement_gate, measure_disagreement_mass
from mixle.task.distill import (
    agreement,
    distill,
    distill_for_routing,
    distill_from_labels,
    distill_from_labels_for_routing,
    distill_records,
    distill_records_for_routing,
    distill_records_from_labels,
    distill_records_from_labels_for_routing,
    distill_structured,
    distill_structured_from_labels,
)
from mixle.task.distill_soft import distill_from_soft_labels, distill_soft, soft_agreement
from mixle.task.economics import (
    CostModel,
    RoutePlan,
    break_even_volume,
    cascade_cost_per_request,
    recommend_route,
    select_alpha_for_cost,
)

# edge distillation: structure x training-process search under a hard device budget, steered by a
# persistent design meta-model (the model that writes the model)
from mixle.task.edge import (
    FINGERPRINT_KEYS,
    DesignModel,
    DeviceSpec,
    EdgeDistillResult,
    EdgeFootprint,
    EdgeSpace,
    distill_designer,
    distill_for_edge,
    footprint,
    measure_inference_seconds,
    measure_ops_per_second,
    task_fingerprint,
)
from mixle.task.emulate import Emulator, EmulatorReceipt, emulate
from mixle.task.environment import (
    Environment,
    ExplorationEnvironment,
    GaussianStreamingBelief,
    InteractionLog,
    interact,
)
from mixle.task.explore_world import (
    DRILL_COST,
    SURVEY_COST,
    EpisodeResult,
    ExplorationWorld,
    greedy_prospectivity_policy,
    random_policy,
    run_episode,
)
from mixle.task.extract import (
    ExtractionIO,
    distill_extractor,
    extraction_f1,
    tokenize,
)
from mixle.task.generative_capability import extractive_capture_profile, validate_extraction_schema
from mixle.task.generative_text import GenerativeTextIO, distill_text_generative, distill_text_generative_from_labels
from mixle.task.harness import ExtractorHarness, MatcherHarness, replace_alerter, replace_extractor, replace_matcher
from mixle.task.imagine import (
    CeilingReport,
    ImagineResult,
    ProposalVerdict,
    StructuralCandidate,
    ceiling_report,
    propose_structure,
)
from mixle.task.inverse import InverseModel, InverseReceipts, learn_inverse
from mixle.task.irl import MaxEntIRLResult, max_ent_irl, rollout_states, state_features
from mixle.task.llm import (
    CallableLLM,
    OpenAICompatLLM,
    llm_extractor,
    llm_labeler,
    pick_label,
)
from mixle.task.model import (
    HashedNGram,
    HashedRecord,
    RecordClassifierIO,
    StructuredClassifierIO,
    TaskModel,
    TextClassifierIO,
    adapter_from_spec,
    register_adapter,
)
from mixle.task.multilabel import MultiLabelSolution, solve_multilabel
from mixle.task.orchestrate import OrchestrationResult, World, orchestrate
from mixle.task.outcome_decomposer import (
    OutcomeTrainedDecomposer,
    RoundStats,
    evaluate_greedy_heuristic,
    evaluate_plan_model,
    execute_plan,
    imitation_traces,
    train_outcome_decomposer,
)
from mixle.task.pilot_ladder import (
    PILOT_LADDER_ASSUMED_HEALTHY_PIECES,
    PILOT_LADDER_UNAVAILABLE_PIECES,
    PilotLadderResult,
    Rung,
    RungArtifacts,
    RungOutcome,
    run_pilot_ladder,
)
from mixle.task.plan import Planner, distill_planner
from mixle.task.plan_model import PlanModel, fit_plan_model
from mixle.task.plan_refine import RefinementReport, outcome_refine_planner
from mixle.task.probe_policy import ProbeHeadToHead, head_to_head_probe, myopic_eig_policy
from mixle.task.propose import ProposeVerifyResult, RoundLog, SequenceProposal, propose_verify_retrain

# post-training quantization: int8/int4 MLP weights (numpy-only inference) + LNS integer log-space
# execution for structured students (transcendental-free above the leaf boundary)
from mixle.task.quantize import (
    LNSStructuredClassifierIO,
    QuantizedClassifierIO,
    QuantizedMLP,
    lns_classifier,
    quantize_mlp,
)
from mixle.task.recommend import FieldChoice, ModelRecommendation, recommend_model
from mixle.task.refine import (
    EditTrial,
    SearchOutcome,
    apply_edge,
    blind_structure_search,
    diagnosis_directed_correction,
    fit_independent_baseline,
)
from mixle.task.regress import RegressionSolution, solve_regression
from mixle.task.replay import ExecutionTrace, TraceStep, is_bit_identical_replay, record_step, replay
from mixle.task.rl import GridWorld, QLearningResult, rollout, tabular_q_learning
from mixle.task.router import HarvestResolveResult, Router, RouterStats, resolve_from_harvest, route_stack
from mixle.task.scorecard import Scorecard, scorecard
from mixle.task.sft_plan import GenerativePlanner, sample_plans, score_plan, sft_planner
from mixle.task.solve import Solution, load_harvested, solve
from mixle.task.structured_out import StructuredSolution, solve_structured
from mixle.task.task_decomposition import (
    DecompositionProposer,
    DependencyForest,
    TaskExample,
    decomposed_predict,
    discover_decomposition,
    fit_decomposition,
    init_decomposition_proposer,
    log_decomposition_recipe,
    mdl_score,
    monolithic_predict,
    record_decomposition_outcome,
)
from mixle.task.toolcall import ToolCaller, ToolSpec, distill_tool_caller
from mixle.task.traces import AgentTrace, AgentTraces, harvest_agent_traces, parse_conversation
from mixle.task.tune import CalibratedTuneResult, RecipeSpace, TuneResult, tune_recipe, tune_recipe_for_routing
from mixle.task.vlm import (
    CallableVLM,
    OpenAICompatVLM,
    score_candidate,
    score_fn_for,
)

__all__ = [
    "ESCALATE",
    "SCHEMA_VERSION",
    "ActiveResult",
    "AgentTrace",
    "AgentTraces",
    "CalibratedTaskModel",
    "CallableLLM",
    "CapabilitySuite",
    "Cascade",
    "CascadeStats",
    "ComposedAnswer",
    "ComposedModel",
    "SyntheticDomain",
    "estimate_near_duplicate_rate",
    "optimize_mixture",
    "proxy_run_score",
    "EstimatorBandit",
    "ThompsonBernoulli",
    "ThompsonGaussian",
    "UCB1",
    "CollapseVerdict",
    "collapse_monitor",
    "distinct_count_diversity",
    "entropy_diversity",
    "CostModel",
    "DEFAULT_RUNGS",
    "DensityGate",
    "DesignModel",
    "DesignedModel",
    "DisagreementGate",
    "UnionGate",
    "best_family",
    "rank_design_families",
    "record_accepted_recipe",
    "DeviceSpec",
    "EdgeDistillResult",
    "EdgeFootprint",
    "EdgeSpace",
    "Emulator",
    "EmulatorReceipt",
    "emulate",
    "ExecutionTrace",
    "EmbeddingHeadIO",
    "ExtractionIO",
    "Environment",
    "ExplorationEnvironment",
    "GaussianStreamingBelief",
    "InteractionLog",
    "interact",
    "DRILL_COST",
    "SURVEY_COST",
    "EpisodeResult",
    "ExplorationWorld",
    "greedy_prospectivity_policy",
    "random_policy",
    "run_episode",
    "ExtractorHarness",
    "MatcherHarness",
    "CeilingReport",
    "ImagineResult",
    "ProposalVerdict",
    "StructuralCandidate",
    "ceiling_report",
    "propose_structure",
    "InverseModel",
    "InverseReceipts",
    "learn_inverse",
    "FieldChoice",
    "GenerativeTextIO",
    "extractive_capture_profile",
    "validate_extraction_schema",
    "HashedNGram",
    "HashedRecord",
    "KNOWN_RUNGS",
    "LNSStructuredClassifierIO",
    "LadderResult",
    "ModelRecommendation",
    "OpenAICompatLLM",
    "OpenAICompatVLM",
    "CallableVLM",
    "score_candidate",
    "score_fn_for",
    "QuantizedClassifierIO",
    "QuantizedMLP",
    "EditTrial",
    "SearchOutcome",
    "apply_edge",
    "blind_structure_search",
    "diagnosis_directed_correction",
    "fit_independent_baseline",
    "RecipeSpace",
    "RecordClassifierIO",
    "RoutePlan",
    "Router",
    "RungResult",
    "Scorecard",
    "RouterStats",
    "HarvestResolveResult",
    "resolve_from_harvest",
    "RegressionSolution",
    "MultiLabelSolution",
    "OrchestrationResult",
    "World",
    "OutcomeTrainedDecomposer",
    "RoundStats",
    "evaluate_greedy_heuristic",
    "evaluate_plan_model",
    "execute_plan",
    "imitation_traces",
    "train_outcome_decomposer",
    "StructuredSolution",
    "DecompositionProposer",
    "DependencyForest",
    "TaskExample",
    "decomposed_predict",
    "discover_decomposition",
    "fit_decomposition",
    "init_decomposition_proposer",
    "log_decomposition_recipe",
    "mdl_score",
    "monolithic_predict",
    "record_decomposition_outcome",
    "PILOT_LADDER_ASSUMED_HEALTHY_PIECES",
    "PILOT_LADDER_UNAVAILABLE_PIECES",
    "PilotLadderResult",
    "Rung",
    "RungArtifacts",
    "RungOutcome",
    "run_pilot_ladder",
    "GenerativePlanner",
    "Planner",
    "PlanModel",
    "fit_plan_model",
    "Solution",
    "StructuredClassifierIO",
    "TaskManifest",
    "TaskModel",
    "TraceStep",
    "ToolCaller",
    "ToolSpec",
    "TextClassifierIO",
    "CalibratedTuneResult",
    "TuneResult",
    "WordEmbeddingFeaturizer",
    "acquisition_scores",
    "active_distill",
    "adapter_from_spec",
    "agreement",
    "break_even_volume",
    "capture_profile",
    "cascade_cost_per_request",
    "case_jitter_invariance",
    "capacity_ladder",
    "climb_to",
    "compose",
    "design_model",
    "fit_disagreement_gate",
    "measure_disagreement_mass",
    "distill",
    "distill_designer",
    "distill_extractor",
    "distill_planner",
    "distill_tool_caller",
    "distill_for_edge",
    "footprint",
    "distill_for_routing",
    "distill_from_labels",
    "distill_from_labels_for_routing",
    "distill_text_generative",
    "distill_text_generative_from_labels",
    "distill_records",
    "distill_records_for_routing",
    "distill_records_from_labels",
    "distill_records_from_labels_for_routing",
    "distill_structured",
    "distill_structured_from_labels",
    "distill_from_soft_labels",
    "distill_soft",
    "soft_agreement",
    "extraction_f1",
    "harvest_agent_traces",
    "keyboard_typo_corruption",
    "parse_conversation",
    "get_arrays_builder",
    "get_builder",
    "is_bit_identical_replay",
    "llm_extractor",
    "llm_labeler",
    "load_harvested",
    "lns_classifier",
    "orchestrate",
    "pick_label",
    "record_step",
    "recommend_model",
    "replay",
    "recommend_route",
    "select_alpha_for_cost",
    "replace_alerter",
    "replace_extractor",
    "replace_matcher",
    "route_stack",
    "RefinementReport",
    "ProbeHeadToHead",
    "head_to_head_probe",
    "myopic_eig_policy",
    "outcome_refine_planner",
    "ProposeVerifyResult",
    "RoundLog",
    "SequenceProposal",
    "propose_verify_retrain",
    "GridWorld",
    "QLearningResult",
    "tabular_q_learning",
    "rollout",
    "MaxEntIRLResult",
    "max_ent_irl",
    "rollout_states",
    "state_features",
    "sample_plans",
    "score_plan",
    "scorecard",
    "sft_planner",
    "spec_to_estimator",
    "whitespace_invariance",
    "load_arrays",
    "load_json",
    "load_module",
    "quantize_mlp",
    "read_manifest",
    "register_adapter",
    "register_arrays_builder",
    "register_builder",
    "save_arrays",
    "save_json",
    "save_module",
    "solve",
    "solve_regression",
    "solve_multilabel",
    "solve_structured",
    "measure_inference_seconds",
    "measure_ops_per_second",
    "task_fingerprint",
    "FINGERPRINT_KEYS",
    "tokenize",
    "tune_recipe",
    "tune_recipe_for_routing",
]
