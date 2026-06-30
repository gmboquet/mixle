"""Small, local, task-specific models -- train or distill one, save a durable artifact, call it as a function.

The unit is a :class:`~mixle.task.model.TaskModel`: a fitted model scoped to *one* task (classify, extract,
recommend a model shape, ...), small enough to run locally and fast, with a durable artifact (:mod:`~mixle.task.artifact`)
so a plain Python program can load it in a fresh process and just call it. Producers:

  * :func:`~mixle.task.distill.distill` -- a teacher (any callable LM) labels data, a tiny student is fit to match;
  * :func:`~mixle.task.tune.tune_recipe` -- ``mixle.doe`` searches the student recipe to minimize train cost.

This module's public surface re-exports the artifact contract; the model/distill/tune layers land on top.
"""

from __future__ import annotations

from mixle.task.artifact import (
    SCHEMA_VERSION,
    TaskManifest,
    get_builder,
    load_json,
    load_module,
    read_manifest,
    register_builder,
    save_json,
    save_module,
)
from mixle.task.calibrate import ESCALATE, CalibratedTaskModel
from mixle.task.cascade import Cascade, CascadeStats
from mixle.task.density import DensityGate
from mixle.task.distill import agreement, distill
from mixle.task.economics import (
    CostModel,
    RoutePlan,
    break_even_volume,
    cascade_cost_per_request,
    recommend_route,
)
from mixle.task.model import (
    HashedNGram,
    TaskModel,
    TextClassifierIO,
    adapter_from_spec,
    register_adapter,
)
from mixle.task.recommend import FieldChoice, ModelRecommendation, recommend_model
from mixle.task.tune import RecipeSpace, TuneResult, tune_recipe

__all__ = [
    "ESCALATE",
    "SCHEMA_VERSION",
    "CalibratedTaskModel",
    "Cascade",
    "CascadeStats",
    "CostModel",
    "DensityGate",
    "FieldChoice",
    "HashedNGram",
    "ModelRecommendation",
    "RecipeSpace",
    "RoutePlan",
    "TaskManifest",
    "TaskModel",
    "TextClassifierIO",
    "TuneResult",
    "adapter_from_spec",
    "agreement",
    "break_even_volume",
    "cascade_cost_per_request",
    "distill",
    "get_builder",
    "recommend_model",
    "recommend_route",
    "load_json",
    "load_module",
    "read_manifest",
    "register_adapter",
    "register_builder",
    "save_json",
    "save_module",
    "tune_recipe",
]
