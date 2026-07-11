"""Reproducible model artifacts: a descriptive header logging what trained a model and how.

A :class:`Header` records the estimator/model configuration, a summary + content hash of the training
data, the data schema, the training settings and final objective, timing, and the software environment
(versions + git commit). Attach one at fit time with :func:`fit_with_provenance` (or build one for any
model + data with :func:`build_header`) so a fitted model is self-describing and a run can be reproduced
and audited. Headers are plain dicts under the hood (:meth:`Header.to_dict`), so they serialize to
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

from mixle.data.hashing import dataset_hash
from mixle.data.hashing import model_hash as _model_hash


def _version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def _git_commit() -> str | None:
    try:
        import subprocess

        import mixle

        root = os.path.dirname(os.path.dirname(os.path.abspath(mixle.__file__)))
        r = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2
        )
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def environment_info() -> dict:
    """Snapshot of the software/hardware environment for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "mixle_version": _version("mixle"),
        "git_commit": _git_commit(),
        "cpu_count": os.cpu_count(),
    }


def _schema_of(model: Any) -> list[tuple[str, str]]:
    try:
        from mixle.data.schema import Schema

        return [(f.name, repr(f.type)) for f in Schema.for_model(model).fields]
    except Exception:  # noqa: BLE001
        return []


def _resource_usage() -> dict:
    """Process CPU time and peak resident memory (best effort; empty on platforms without ``resource``)."""
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        peak_mb = ru.ru_maxrss / 1e6 if sys.platform == "darwin" else ru.ru_maxrss / 1e3  # macOS bytes, Linux KB
        return {"cpu_time_s": ru.ru_utime + ru.ru_stime, "peak_rss_mb": round(peak_mb, 1)}
    except Exception:  # noqa: BLE001
        return {}


def _records(data: Any):
    """Iterate a dataset uniformly, whether it is a list/sequence or a DataSource (``.records()``)."""
    rec = getattr(data, "records", None)
    return rec() if callable(rec) else data


def _final_loglik(model: Any, data: Any) -> float | None:
    try:
        import numpy as np

        enc = model.dist_to_encoder().seq_encode(list(_records(data)))
        return float(np.sum(model.seq_log_density(enc)))
    except Exception:  # noqa: BLE001
        return None


def _safe_model_hash(model: Any) -> str | None:
    """Fingerprint a model's serialized parameters, or ``None`` if it isn't serializable."""
    try:
        return _model_hash(model)
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Header:
    """A descriptive, serializable provenance record for a fitted model."""

    model_type: str
    model_summary: str
    schema: list[tuple[str, str]]
    n_records: int | None
    dataset_hash: str
    final_loglik: float | None
    model_hash: str | None = None
    training: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        """Return the provenance header as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Header:
        """Reconstruct a provenance header from a dictionary."""
        d = dict(d)
        d["schema"] = [tuple(x) for x in d.get("schema", [])]
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})

    def __str__(self) -> str:
        tr = {k: v for k, v in self.training.items() if k != "convergence"}  # the trace is long; summarize it
        n_iter = len(self.training.get("convergence", []))
        lines = [
            f"Header[{self.model_type}]",
            f"  data: {self.n_records} records, hash={self.dataset_hash[:12]}…",
            f"  model_hash: {self.model_hash[:12] + '…' if self.model_hash else None}",
            f"  schema: {', '.join(f'{n}:{t}' for n, t in self.schema) or '(none)'}",
            f"  final_loglik: {self.final_loglik}",
            f"  training: {tr}" + (f"  [{n_iter} iters logged]" if n_iter else ""),
            f"  timing: {self.timing}",
            f"  resources: {self.resources}",
            f"  env: python {self.environment.get('python')}, "
            f"mixle {self.environment.get('mixle_version') or self.environment.get('pysp_version')}, "
            f"git {self.environment.get('git_commit')}",
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
    resources: dict | None = None,
    hash_sort: bool = False,
    hash_max_records: int | None = None,
) -> Header:
    """Build a :class:`Header` for ``model`` trained on ``data`` (does not run any fitting)."""
    n = len(data) if hasattr(data, "__len__") else None
    timing: dict = {}
    if started is not None and finished is not None:
        timing = {
            "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "finished": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
            "duration_s": round(finished - started, 6),
        }
    ll = _final_loglik(model, data) if final_loglik == "auto" else final_loglik
    return Header(
        model_type=type(model).__name__,
        model_summary=str(model),
        schema=_schema_of(model),
        n_records=n,
        dataset_hash=dataset_hash(data, sort=hash_sort, max_records=hash_max_records),
        final_loglik=ll,
        model_hash=_safe_model_hash(model),
        training=dict(training or {}),
        timing=timing,
        resources=dict(resources or {}),
        environment=environment_info(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


class _EMHistory:
    """A silent ``out`` + ``on_step`` sink that captures the per-iteration convergence trace.

    ``em_record`` (the ``_write_em_iter`` hook) records the scalar trace -- loglik / delta / valid_loglik /
    objective. ``__call__`` (the ``optimize(on_step=...)`` hook) fingerprints the accepted model each
    iteration and chains it (``model_hash`` + the previous iteration's ``parent_hash``), so when both are
    wired the trace is a verifiable hash chain: iteration i+1 records i's hash as its parent. Records are
    merged by iteration, so either hook may be absent or fire in either order."""

    def __init__(self) -> None:
        self._by_iter: dict[int, dict] = {}
        self._prev_hash: str | None = None

    def write(self, _s: str) -> None:  # discard the text lines; we keep the structured records
        pass

    def flush(self) -> None:
        pass

    def _rec(self, i: int) -> dict:
        return self._by_iter.setdefault(int(i), {"iter": int(i)})

    def em_record(self, i: int, ll: float, dll: float, vll: float | None, obj_label: str | None) -> None:
        finite_delta = dll == dll and abs(dll) != float("inf")  # null NaN/inf (e.g. the first iteration)
        rec = self._rec(i)
        rec["loglik"] = ll
        rec["delta"] = dll if finite_delta else None
        if vll is not None:
            rec["valid_loglik"] = vll
        if obj_label is not None:
            rec["objective"] = obj_label

    def __call__(self, step: Any) -> None:
        rec = self._rec(step.iter)
        h = _safe_model_hash(step.model)
        rec["model_hash"] = h
        rec["parent_hash"] = self._prev_hash
        self._prev_hash = h

    @property
    def records(self) -> list[dict]:
        return [self._by_iter[k] for k in sorted(self._by_iter)]


def fit_with_provenance(
    data: Any, estimator: Any, *, seed: int | None = None, lineage: bool = True, **optimize_kw: Any
):
    """Fit ``estimator`` on ``data`` via EM (:func:`mixle.inference.optimize`) and return
    ``(model, header)``, the model carrying a ``.header`` :class:`Header` with the data hash, the
    final model hash, schema, training settings + per-iteration convergence trace, timing, final
    log-likelihood, and environment. Pass your own ``out=`` to print iterations (then the trace is not
    captured).

    With ``lineage=True`` (default) each iteration in the convergence trace also records the accepted
    model's ``model_hash`` and the previous iteration's ``parent_hash``, forming a verifiable hash chain
    (check it with :func:`verify_lineage`). This fingerprints the model every iteration; pass
    ``lineage=False`` to skip it for very large models. Any user ``on_step=`` is still called."""
    from mixle.inference.estimation import optimize

    capture = "out" not in optimize_kw
    collector = _EMHistory() if capture else None
    if collector is not None:
        optimize_kw["out"] = collector
        optimize_kw.setdefault("print_iter", 1)  # record every iteration, not every Nth
        if lineage:  # also fingerprint the model each iteration -> a hash chain in the trace
            user_on_step = optimize_kw.get("on_step")

            def _on_step(step: Any) -> None:
                collector(step)
                if user_on_step is not None:
                    user_on_step(step)

            optimize_kw["on_step"] = _on_step

    training = {
        "method": "em",
        "max_its": optimize_kw.get("max_its"),
        "delta": optimize_kw.get("delta"),
        "backend": optimize_kw.get("backend", "local"),
        "seed": seed,
    }
    cpu0 = _resource_usage().get("cpu_time_s")
    t0 = time.time()
    model = optimize(data, estimator, **optimize_kw)
    t1 = time.time()
    usage = _resource_usage()
    if cpu0 is not None and usage.get("cpu_time_s") is not None:
        usage = {"cpu_time_s": round(usage["cpu_time_s"] - cpu0, 3), "peak_rss_mb": usage.get("peak_rss_mb")}
    if collector is not None:
        recs = collector.records
        training["convergence"] = recs
        training["iterations"] = recs[-1]["iter"] if recs else 0
        delta = optimize_kw.get("delta")
        if delta is not None and recs:
            last = recs[-1]["delta"]
            training["converged"] = last is not None and last < delta
    header = build_header(model, data, training=training, started=t0, finished=t1, resources=usage)
    try:
        model.header = header
    except Exception:  # noqa: BLE001
        pass
    return model, header


def verify_lineage(header: Any) -> bool:
    """Check that a header's per-iteration convergence trace is an intact model-hash chain.

    Returns True when every iteration that recorded a ``model_hash`` names the previous such iteration's
    hash as its ``parent_hash`` (so iteration i+1 provably descends from i), and True vacuously when the
    trace carries no lineage (``fit_with_provenance(lineage=False)`` or a custom ``out``). Returns False
    on the first invalid link. Accepts a :class:`Header` or its ``to_dict``."""
    training = header.training if isinstance(header, Header) else dict(header or {}).get("training", {})
    prev: str | None = None
    for rec in training.get("convergence", []):
        if "model_hash" not in rec:
            continue
        if rec.get("parent_hash") != prev:
            return False
        prev = rec["model_hash"]
    return True
