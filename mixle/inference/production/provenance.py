"""Reproducible model artifacts: a descriptive header logging what trained a model and how.

A :class:`Header` records the estimator/model configuration, a summary + content hash of the training
data, the data schema, the training settings and final objective, timing, and the software environment
(versions + git commit). Attach one at fit time with :func:`fit_with_provenance` (or build one for any
model + data with :func:`build_header`) so a fitted model is self-describing and a run can be reproduced
and audited. Headers are plain dicts under the hood (:meth:`Header.to_dict`), so they serialize to
JSON alongside the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from mixle.data.hashing import dataset_hash
from mixle.data.hashing import model_hash as _model_hash
from mixle.utils.exact import require_exact_bool


def _version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def _git_state() -> dict[str, Any]:
    """Identify both the commit and any uncommitted source state used by the fit."""
    try:
        import mixle

        root = os.path.dirname(os.path.dirname(os.path.abspath(mixle.__file__)))
        commit_result = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2
        )
        commit = commit_result.stdout.strip() or None
        status_result = subprocess.run(
            ["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        status = status_result.stdout
        dirty = bool(status)
        state_digest = None
        if dirty:
            diff_result = subprocess.run(
                ["git", "-C", root, "diff", "--binary", "HEAD"],
                capture_output=True,
                timeout=2,
            )
            untracked_result = subprocess.run(
                ["git", "-C", root, "ls-files", "--others", "--exclude-standard", "-z"],
                capture_output=True,
                timeout=2,
            )
            h = hashlib.sha256()
            h.update(status.encode("utf-8", errors="surrogateescape"))
            h.update(diff_result.stdout)
            for raw_path in sorted(p for p in untracked_result.stdout.split(b"\0") if p):
                h.update(len(raw_path).to_bytes(8, "big"))
                h.update(raw_path)
                try:
                    with open(os.path.join(root, os.fsdecode(raw_path)), "rb") as f:
                        h.update(f.read())
                except OSError:
                    h.update(b"<unreadable>")
            state_digest = h.hexdigest()
        return {"git_commit": commit, "git_dirty": dirty, "git_worktree_digest": state_digest}
    except Exception:  # noqa: BLE001
        return {"git_commit": None, "git_dirty": None, "git_worktree_digest": None}


def _git_commit() -> str | None:
    """Backward-compatible commit accessor."""
    return _git_state()["git_commit"]


def _embedded_build_provenance() -> dict | None:
    """Read the provenance the wheel was built with, if this mixle came from a built artifact.

    An installed wheel is not in a repository, so the ambient-git probe legitimately finds nothing
    -- but the artifact carries its own attestation, and ignoring it reported an unknown commit for
    an installation whose identity was recorded all along (SYS-07).
    """
    try:
        import json
        from pathlib import Path

        import mixle

        root = Path(next(iter(mixle.__path__)))
        payload = json.loads((root / "_build_provenance.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent or unreadable provenance is simply not available
        return None
    if not isinstance(payload, dict) or payload.get("artifact") != "mixle.build_provenance/v1":
        return None
    commit = payload.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        return None
    return payload


def environment_info() -> dict:
    """Snapshot of the software/hardware environment for reproducibility.

    Source identity is taken from the ambient repository when there is one, and otherwise from the
    installed artifact's embedded build provenance. ``provenance_source`` names which, because
    "no repository here" and "a clean repository" are different facts and were previously reported
    the same way.
    """
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "mixle_version": _version("mixle"),
        "cpu_count": os.cpu_count(),
    }
    git_state = _git_state()
    info.update(git_state)
    if git_state.get("git_commit") is None:
        embedded = _embedded_build_provenance()
        if embedded is not None:
            info["git_commit"] = embedded["source_commit"]
            info["git_dirty"] = embedded.get("source_dirty")
            info["source_tree"] = embedded.get("source_tree")
            info["provenance_source"] = "installed-artifact-build-provenance"
        else:
            info["provenance_source"] = "unavailable"
    else:
        info["provenance_source"] = "ambient-repository"
    return info


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


def _request_value(value: Any) -> Any:
    """Canonical JSON-safe description of a fit-request value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _request_value(v) for k, v in sorted(value.items(), key=lambda item: repr(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_request_value(v) for v in value]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _fit_request_digest(request: dict[str, Any]) -> str:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
            "started": datetime.fromtimestamp(started, UTC).isoformat(),
            "finished": datetime.fromtimestamp(finished, UTC).isoformat(),
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
        created_at=datetime.now(UTC).isoformat(),
    )


def _lineage_transition_digest(record: dict[str, Any]) -> str:
    """Digest the executed transition fields recorded by the optimizer callback."""
    fields = {
        "lineage_schema": record.get("lineage_schema"),
        "run_id": record.get("run_id"),
        "iter": record.get("iter"),
        "model_hash": record.get("model_hash"),
        "parent_hash": record.get("parent_hash"),
        "parent_transition_digest": record.get("parent_transition_digest"),
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _EMHistory:
    """A silent ``out`` + ``on_step`` sink that captures the per-iteration convergence trace.

    ``em_record`` (the ``_write_em_iter`` hook) records the scalar trace -- loglik / delta / valid_loglik /
    objective. ``__call__`` (the ``optimize(on_step=...)`` hook) fingerprints the accepted model each
    iteration and chains it to the previous model and transition digests, so when both are wired the
    trace is an authenticated execution chain rather than a set of asserted parent hashes. Records are
    merged by iteration, so either hook may be absent or fire in either order."""

    def __init__(self) -> None:
        self._by_iter: dict[int, dict] = {}
        self._prev_hash: str | None = None
        self._prev_transition_digest: str | None = None
        self._run_id = secrets.token_hex(16)

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
        rec["lineage_schema"] = "mixle-em-lineage-v1"
        rec["run_id"] = self._run_id
        rec["model_hash"] = h
        rec["parent_hash"] = self._prev_hash
        rec["parent_transition_digest"] = self._prev_transition_digest
        rec["transition_digest"] = _lineage_transition_digest(rec)
        self._prev_hash = h
        self._prev_transition_digest = rec["transition_digest"]

    @property
    def records(self) -> list[dict]:
        return [self._by_iter[k] for k in sorted(self._by_iter)]

    def terminal(self, model: Any) -> dict[str, Any]:
        """Bind the optimizer's returned (possibly restored-best) model to the executed step chain."""
        record = {
            "iter": "returned",
            "lineage_schema": "mixle-em-lineage-v1",
            "run_id": self._run_id,
            "model_hash": _safe_model_hash(model),
            "parent_hash": self._prev_hash,
            "parent_transition_digest": self._prev_transition_digest,
        }
        record["transition_digest"] = _lineage_transition_digest(record)
        return record


def fit_with_provenance(
    data: Any, estimator: Any, *, seed: int | None = None, lineage: bool = True, **optimize_kw: Any
):
    """Fit ``estimator`` on ``data`` via EM (:func:`mixle.inference.optimize`) and return
    ``(model, header)``, the model carrying a ``.header`` :class:`Header` with the data hash, the
    final model hash, schema, training settings + per-iteration convergence trace, timing, final
    log-likelihood, and environment. Pass your own ``out=`` to print iterations (then the trace is not
    captured).

    With ``lineage=True`` (default) each iteration in the convergence trace records an authenticated
    transition from the previous accepted model, and a terminal record binds the model actually returned
    by the optimizer (check it with :func:`verify_lineage`). This fingerprints the model every iteration;
    pass ``lineage=False`` to skip it for very large models. Any user ``on_step=`` is still called."""
    import inspect

    from mixle.inference.estimation import optimize

    # Consume any sequence, DataSource, or one-shot iterator exactly once. This immutable request
    # snapshot is then shared by fitting, final scoring, record counting, and hashing.
    materialized_data = list(_records(data))

    # optimize()'s OWN defaults, not a hardcoded guess: a caller who relies on optimize()'s
    # defaults (doesn't pass max_its=/delta= explicitly) used to have those recorded as bare
    # None here instead of the value optimize() actually ran with -- silently breaking the audit
    # trail this function exists to build. `.get(key, default)` still returns an explicit
    # delta=None (disable early stopping) correctly, since that key IS present in optimize_kw.
    _optimize_defaults = inspect.signature(optimize).parameters
    if "rng" in optimize_kw:
        raise ValueError("fit_with_provenance requires seed= rather than a mutable rng= object")
    effective_seed = 0 if seed is None else seed
    if isinstance(effective_seed, bool) or not isinstance(effective_seed, int) or effective_seed < 0:
        raise ValueError("seed must be a nonnegative integer or None")
    optimize_kw["seed"] = effective_seed
    request_optimize_kw = dict(optimize_kw)
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
        "max_its": optimize_kw.get("max_its", _optimize_defaults["max_its"].default),
        "delta": optimize_kw.get("delta", _optimize_defaults["delta"].default),
        "backend": optimize_kw.get("backend", "local"),
        "seed": effective_seed,
        "data_materialized": True,
        "lineage_status": "recorded" if collector is not None and lineage else "not_recorded",
    }
    request = {
        "estimator_type": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "estimator_repr": repr(estimator),
        "data_hash": dataset_hash(materialized_data),
        "n_records": len(materialized_data),
        "optimize": {k: _request_value(v) for k, v in sorted(request_optimize_kw.items())},
        "lineage": require_exact_bool(lineage, "lineage"),
    }
    training["fit_request"] = request
    training["fit_request_digest"] = _fit_request_digest(request)
    cpu0 = _resource_usage().get("cpu_time_s")
    t0 = time.time()
    model = optimize(materialized_data, estimator, **optimize_kw)
    t1 = time.time()
    usage = _resource_usage()
    if cpu0 is not None and usage.get("cpu_time_s") is not None:
        usage = {"cpu_time_s": round(usage["cpu_time_s"] - cpu0, 3), "peak_rss_mb": usage.get("peak_rss_mb")}
    if collector is not None:
        recs = collector.records
        training["convergence"] = recs
        if lineage:
            training["lineage_terminal"] = collector.terminal(model)
        training["iterations"] = recs[-1]["iter"] if recs else 0
        delta = training["delta"]  # the resolved value (falls back to optimize()'s own default)
        if delta is not None and recs:
            last = recs[-1]["delta"]
            training["converged"] = last is not None and last < delta
    header = build_header(model, materialized_data, training=training, started=t0, finished=t1, resources=usage)
    try:
        model.header = header
    except Exception:  # noqa: BLE001
        pass
    return model, header


def verify_lineage(header: Any) -> bool:
    """Verify a complete execution-recorded EM transition chain.

    Missing lineage is an unverified result, never a vacuous success. Each iteration must bind its
    actual model hash to the previous model and transition digests, all records must share one run id,
    and the final executed model hash must equal the header's fitted-model hash.
    """
    raw = header.to_dict() if isinstance(header, Header) else dict(header or {})
    training = raw.get("training") or {}
    records = training.get("convergence") or []
    if training.get("lineage_status") != "recorded" or not records:
        return False
    required = {
        "iter",
        "lineage_schema",
        "run_id",
        "model_hash",
        "parent_hash",
        "parent_transition_digest",
        "transition_digest",
    }
    previous_hash: str | None = None
    previous_transition_digest: str | None = None
    previous_iter: int | None = None
    run_id: str | None = None
    for record in records:
        if not isinstance(record, dict) or not required.issubset(record):
            return False
        if record["lineage_schema"] != "mixle-em-lineage-v1":
            return False
        if run_id is None:
            run_id = record["run_id"]
        if not isinstance(run_id, str) or record["run_id"] != run_id:
            return False
        iteration = record["iter"]
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or (previous_iter is not None and iteration <= previous_iter)
        ):
            return False
        if record["parent_hash"] != previous_hash:
            return False
        if record["parent_transition_digest"] != previous_transition_digest:
            return False
        if not isinstance(record["model_hash"], str):
            return False
        if record["transition_digest"] != _lineage_transition_digest(record):
            return False
        previous_hash = record["model_hash"]
        previous_transition_digest = record["transition_digest"]
        previous_iter = iteration
    terminal = training.get("lineage_terminal")
    if not isinstance(terminal, dict) or not required.issubset(terminal):
        return False
    if terminal["iter"] != "returned":
        return False
    if terminal["lineage_schema"] != "mixle-em-lineage-v1" or terminal["run_id"] != run_id:
        return False
    if terminal["parent_hash"] != previous_hash:
        return False
    if terminal["parent_transition_digest"] != previous_transition_digest:
        return False
    if terminal["transition_digest"] != _lineage_transition_digest(terminal):
        return False
    return isinstance(raw.get("model_hash"), str) and raw["model_hash"] == terminal["model_hash"]
