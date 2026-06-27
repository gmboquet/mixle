"""Reproducible model artifacts: a descriptive header logging what trained a model and how.

A :class:`ModelHeader` records the estimator/model configuration, a summary + content hash of the training
data, the data schema, the training settings and final objective, timing, and the software environment
(versions + git commit). Attach one at fit time with :func:`fit_with_provenance` (or build one for any
model + data with :func:`build_header`) so a fitted model is self-describing and a run can be reproduced
and audited. Headers are plain dicts under the hood (:meth:`ModelHeader.to_dict`), so they serialize to
JSON alongside the model.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from pysp.data.hashing import dataset_hash


def _version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        import subprocess

        import pysp

        root = os.path.dirname(os.path.dirname(os.path.abspath(pysp.__file__)))
        r = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def environment_info() -> dict:
    """Snapshot of the software/hardware environment for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "pysp_version": _version("pysp"),
        "git_commit": _git_commit(),
        "cpu_count": os.cpu_count(),
    }


def _schema_of(model: Any) -> list[tuple[str, str]]:
    try:
        from pysp.data.schema import Schema

        return [(f.name, repr(f.type)) for f in Schema.for_model(model).fields]
    except Exception:
        return []


def _final_loglik(model: Any, data: Any) -> float | None:
    try:
        import numpy as np

        enc = model.dist_to_encoder().seq_encode(list(data))
        return float(np.sum(model.seq_log_density(enc)))
    except Exception:
        return None


@dataclass
class ModelHeader:
    """A descriptive, serializable provenance record for a fitted model."""

    model_type: str
    model_summary: str
    schema: list[tuple[str, str]]
    n_records: int | None
    dataset_hash: str
    final_loglik: float | None
    training: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelHeader":
        d = dict(d)
        d["schema"] = [tuple(x) for x in d.get("schema", [])]
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})

    def __str__(self) -> str:
        tr = {k: v for k, v in self.training.items() if k != "convergence"}  # the trace is long; summarize it
        n_iter = len(self.training.get("convergence", []))
        lines = [
            f"ModelHeader[{self.model_type}]",
            f"  data: {self.n_records} records, hash={self.dataset_hash[:12]}…",
            f"  schema: {', '.join(f'{n}:{t}' for n, t in self.schema) or '(none)'}",
            f"  final_loglik: {self.final_loglik}",
            f"  training: {tr}" + (f"  [{n_iter} iters logged]" if n_iter else ""),
            f"  timing: {self.timing}",
            f"  env: python {self.environment.get('python')}, "
            f"pysp {self.environment.get('pysp_version')}, git {self.environment.get('git_commit')}",
            f"  created_at: {self.created_at}",
        ]
        return "\n".join(lines)


def build_header(
    model: Any,
    data: Any,
    *,
    training: dict | None = None,
    started: float | None = None,
    finished: float | None = None,
    final_loglik: Any = "auto",
    hash_sort: bool = False,
    hash_max_records: int | None = None,
) -> ModelHeader:
    """Build a :class:`ModelHeader` for ``model`` trained on ``data`` (does not run any fitting)."""
    n = len(data) if hasattr(data, "__len__") else None
    timing: dict = {}
    if started is not None and finished is not None:
        timing = {
            "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "finished": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
            "duration_s": round(finished - started, 6),
        }
    ll = _final_loglik(model, data) if final_loglik == "auto" else final_loglik
    return ModelHeader(
        model_type=type(model).__name__,
        model_summary=str(model),
        schema=_schema_of(model),
        n_records=n,
        dataset_hash=dataset_hash(data, sort=hash_sort, max_records=hash_max_records),
        final_loglik=ll,
        training=dict(training or {}),
        timing=timing,
        environment=environment_info(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


class _EMHistory:
    """A silent ``out`` for ``optimize`` that captures the per-iteration convergence trace (via the
    ``em_record`` hook in ``_write_em_iter``) without printing."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, _s: str) -> None:  # discard the text lines; we keep the structured records
        pass

    def flush(self) -> None:
        pass

    def em_record(self, i: int, ll: float, dll: float, vll: float | None, obj_label: str | None) -> None:
        finite_delta = dll == dll and abs(dll) != float("inf")  # null NaN/inf (e.g. the first iteration)
        rec = {"iter": int(i), "loglik": ll, "delta": (dll if finite_delta else None)}
        if vll is not None:
            rec["valid_loglik"] = vll
        if obj_label is not None:
            rec["objective"] = obj_label
        self.records.append(rec)


def fit_with_provenance(data: Any, estimator: Any, *, seed: int | None = None, **optimize_kw: Any):
    """Fit ``estimator`` on ``data`` via EM (:func:`pysp.inference.optimize`) and return
    ``(model, header)``, the model carrying a ``.header`` :class:`ModelHeader` with the data hash,
    schema, training settings + per-iteration convergence trace, timing, final log-likelihood, and
    environment. Pass your own ``out=`` to print iterations (then the trace is not captured)."""
    from pysp.inference.estimation import optimize

    capture = "out" not in optimize_kw
    collector = _EMHistory() if capture else None
    if collector is not None:
        optimize_kw["out"] = collector
        optimize_kw.setdefault("print_iter", 1)  # record every iteration, not every Nth

    training = {
        "method": "em",
        "max_its": optimize_kw.get("max_its"),
        "delta": optimize_kw.get("delta"),
        "backend": optimize_kw.get("backend", "local"),
        "seed": seed,
    }
    t0 = time.time()
    model = optimize(data, estimator, **optimize_kw)
    t1 = time.time()
    if collector is not None:
        recs = collector.records
        training["convergence"] = recs
        training["iterations"] = recs[-1]["iter"] if recs else 0
        delta = optimize_kw.get("delta")
        if delta is not None and recs:
            last = recs[-1]["delta"]
            training["converged"] = last is not None and last < delta
    header = build_header(model, data, training=training, started=t0, finished=t1)
    try:
        model.header = header
    except Exception:
        pass
    return model, header
