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
from mixle.task.distill import agreement, distill
from mixle.task.model import (
    HashedNGram,
    TaskModel,
    TextClassifierIO,
    adapter_from_spec,
    register_adapter,
)

__all__ = [
    "SCHEMA_VERSION",
    "HashedNGram",
    "TaskManifest",
    "TaskModel",
    "TextClassifierIO",
    "adapter_from_spec",
    "agreement",
    "distill",
    "get_builder",
    "load_json",
    "load_module",
    "read_manifest",
    "register_adapter",
    "register_builder",
    "save_json",
    "save_module",
]
