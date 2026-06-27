"""FastAPI serving layer for a pysparkplug model -- a thin HTTP wrapper over ``pysp.inference.ModelService``.

It loads the model an alias points at in a :class:`~pysp.inference.ModelRegistry` (a shared volume / object
store) and exposes it over HTTP: scoring, health (for Kubernetes probes), provenance, and drift checks.

It is *stateless*: every replica loads the same model from the shared registry, so scaling is horizontal
and a model swap is ``registry.promote(name, version)`` followed by a rolling restart (or ``POST /reload``).

Configuration is by environment variable:
  PYSP_REGISTRY_ROOT   registry directory (default ``/models``)
  PYSP_MODEL_NAME      model name in the registry (default ``model``)
  PYSP_MODEL_ALIAS     alias to serve (default ``production``)
  PYSP_REFERENCE_PATH  optional JSON array of reference records -> enables ``/drift``
  PYSP_ACTIVITY_LOG    optional path for the JSONL activity log (e.g. ``/dev/stdout`` for k8s logs)

Run: ``uvicorn app:app --host 0.0.0.0 --port 8000``.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pysp.inference import ModelRegistry, ModelService

REGISTRY_ROOT = os.environ.get("PYSP_REGISTRY_ROOT", "/models")
MODEL_NAME = os.environ.get("PYSP_MODEL_NAME", "model")
MODEL_ALIAS = os.environ.get("PYSP_MODEL_ALIAS", "production")
REFERENCE_PATH = os.environ.get("PYSP_REFERENCE_PATH")
ACTIVITY_LOG = os.environ.get("PYSP_ACTIVITY_LOG")

_state: dict[str, Any] = {"service": None}


def _records(values: list[Any]) -> list[Any]:
    """Coerce inner JSON arrays to tuples so composite/record models encode correctly."""
    return [tuple(v) if isinstance(v, list) else v for v in values]


def _load_reference() -> list[Any] | None:
    if not REFERENCE_PATH or not os.path.exists(REFERENCE_PATH):
        return None
    with open(REFERENCE_PATH) as fh:
        return _records(json.load(fh))


def _load_service() -> ModelService:
    registry = ModelRegistry(REGISTRY_ROOT)
    svc = ModelService.from_registry(
        registry, MODEL_NAME, alias=MODEL_ALIAS, reference=_load_reference(), log_path=ACTIVITY_LOG
    )
    _state["service"] = svc
    return svc


def _service() -> ModelService:
    svc = _state.get("service")
    if svc is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return svc


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _load_service()  # load the model at startup so readiness reflects a usable server
    yield


app = FastAPI(title="pysparkplug model server", version="1", lifespan=_lifespan)


class ScoreRequest(BaseModel):
    records: list[Any]


@app.get("/health")
def health() -> dict:
    """Liveness/readiness for k8s: 503 until the model is loaded, then the activity summary."""
    svc = _state.get("service")
    if svc is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS, **svc.health()}


@app.get("/info")
def info() -> dict:
    """The served model's provenance header (config, data hash, training, env)."""
    svc = _service()
    header = svc.header  # a ModelHeader (direct) or a plain dict (loaded from the registry)
    return {
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "header": header.to_dict() if hasattr(header, "to_dict") else header,
    }


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    """Per-record log-density. Non-finite (out-of-support) values become ``null`` for valid JSON."""
    svc = _service()
    lp = svc.score(_records(req.records))
    return {
        "log_density": [float(v) if math.isfinite(v) else None for v in lp],
        "n": int(len(lp)),
        "n_unscorable": int(sum(1 for v in lp if not math.isfinite(v))),
    }


@app.post("/drift")
def drift(req: ScoreRequest) -> dict:
    """Drift of the posted batch vs the configured reference sample (needs PYSP_REFERENCE_PATH)."""
    svc = _service()
    try:
        report = svc.check_drift(_records(req.records))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"drift": report.drift, "score": report.score, "per_feature": report.per_feature}


@app.post("/reload")
def reload() -> dict:
    """Re-read the alias from the registry -- the hot path for a model swap after ``promote``."""
    svc = _load_service()
    return {"reloaded": True, "model": MODEL_NAME, "alias": MODEL_ALIAS, **svc.health()}
