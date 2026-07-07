"""Small, local, task-specific models -- train or distill one, save a durable artifact, call it as a function.

The unit is a :class:`~mixle.task.model.TaskModel`: a fitted model scoped to *one* task (classify, extract,
recommend a model shape, ...), small enough to run locally and fast, with a durable artifact (:mod:`~mixle.task.artifact`)
so a plain Python program can load it in a fresh process and just call it. Producers:

  * :func:`~mixle.task.distill.distill` -- a teacher (any callable LM) labels data, a tiny student is fit to match;
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
from mixle.task.calibrate import ESCALATE, CalibratedTaskModel
from mixle.task.cascade import Cascade, CascadeStats
from mixle.task.density import DensityGate
from mixle.task.design import DesignedModel, design_model, spec_to_estimator
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
from mixle.task.economics import (
    CostModel,
    RoutePlan,
    break_even_volume,
    cascade_cost_per_request,
    recommend_route,
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
from mixle.task.extract import (
    ExtractionIO,
    distill_extractor,
    extraction_f1,
    tokenize,
)
from mixle.task.generative_text import GenerativeTextIO, distill_text_generative, distill_text_generative_from_labels
from mixle.task.harness import ExtractorHarness, MatcherHarness, replace_alerter, replace_extractor, replace_matcher
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
from mixle.task.plan import Planner, distill_planner
from mixle.task.plan_refine import RefinementReport, outcome_refine_planner

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
from mixle.task.regress import RegressionSolution, solve_regression
from mixle.task.router import Router, RouterStats, route_stack
from mixle.task.scorecard import Scorecard, scorecard
from mixle.task.sft_plan import GenerativePlanner, sample_plans, score_plan, sft_planner
from mixle.task.solve import Solution, load_harvested, solve
from mixle.task.structured_out import StructuredSolution, solve_structured
from mixle.task.toolcall import ToolCaller, ToolSpec, distill_tool_caller
from mixle.task.traces import AgentTrace, AgentTraces, harvest_agent_traces, parse_conversation
from mixle.task.tune import CalibratedTuneResult, RecipeSpace, TuneResult, tune_recipe, tune_recipe_for_routing

__all__ = [
    "ESCALATE",
    "SCHEMA_VERSION",
    "ActiveResult",
    "AgentTrace",
    "AgentTraces",
    "CalibratedTaskModel",
    "CallableLLM",
    "Cascade",
    "CascadeStats",
    "CostModel",
    "DensityGate",
    "DesignModel",
    "DesignedModel",
    "DeviceSpec",
    "EdgeDistillResult",
    "EdgeFootprint",
    "EdgeSpace",
    "ExtractionIO",
    "ExtractorHarness",
    "MatcherHarness",
    "FieldChoice",
    "GenerativeTextIO",
    "HashedNGram",
    "HashedRecord",
    "LNSStructuredClassifierIO",
    "ModelRecommendation",
    "OpenAICompatLLM",
    "QuantizedClassifierIO",
    "QuantizedMLP",
    "RecipeSpace",
    "RecordClassifierIO",
    "RoutePlan",
    "Router",
    "Scorecard",
    "RouterStats",
    "RegressionSolution",
    "MultiLabelSolution",
    "StructuredSolution",
    "GenerativePlanner",
    "Planner",
    "Solution",
    "StructuredClassifierIO",
    "TaskManifest",
    "TaskModel",
    "ToolCaller",
    "ToolSpec",
    "TextClassifierIO",
    "CalibratedTuneResult",
    "TuneResult",
    "acquisition_scores",
    "active_distill",
    "adapter_from_spec",
    "agreement",
    "break_even_volume",
    "cascade_cost_per_request",
    "design_model",
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
    "extraction_f1",
    "harvest_agent_traces",
    "parse_conversation",
    "get_arrays_builder",
    "get_builder",
    "llm_extractor",
    "llm_labeler",
    "load_harvested",
    "lns_classifier",
    "pick_label",
    "recommend_model",
    "recommend_route",
    "replace_alerter",
    "replace_extractor",
    "replace_matcher",
    "route_stack",
    "RefinementReport",
    "outcome_refine_planner",
    "sample_plans",
    "score_plan",
    "scorecard",
    "sft_planner",
    "spec_to_estimator",
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
