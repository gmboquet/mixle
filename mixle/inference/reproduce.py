"""Reproducibility receipts -- record a fit so it can be replayed and its content re-checked (N2).

A fitted model is only reproducible if someone else can re-derive it from the same complete request.
:func:`fit_and_record` executes one materialized request; :func:`record_fit` binds an existing result to
the exact ordered data, estimator code and state, optimizer code, seed and fit policy, dependency/source
environment, and returned model. :func:`verify_reproducible` re-executes that request and requires every
declared identity and observed result to match.

The legacy ``data_fingerprint`` and ``param_fingerprint`` remain tolerance-equivalent diagnostics:
floats are rounded to ``_NDIGITS`` (10) decimal places. The v2 receipt additionally carries full exact
SHA-256 data and model digests. Therefore:

* last-bit drift may retain the diagnostic fingerprint but fails exact data/model or environment checks;
* a real content, executable, policy, dependency, or source-tree change fails its named check;
* a verdict cannot pass merely because an expected check was omitted.

A matching replay certifies identity and execution under the recorded contract, not statistical
correctness; calibration and model-quality claims require their own evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from mixle.data.hashing import dataset_hash, model_hash
from mixle.inference.integrity import (
    canonical_digest,
    dependency_manifest,
    implementation_digest,
    object_state_digest,
)
from mixle.semantics import canonical_json

_NDIGITS = 10


def _round_floats(obj: Any, ndigits: int) -> Any:
    """Recursively round floats so hashing is stable across platforms' last-bit differences."""
    if isinstance(obj, float):
        # normalize -0.0 and round; ints stay ints
        return round(obj, ndigits) + 0.0
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _canonical(obj: Any, ndigits: int) -> str:
    return json.dumps(_round_floats(obj, ndigits), sort_keys=True, separators=(",", ":"), default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def data_fingerprint(data: Any, *, ndigits: int = _NDIGITS) -> str:
    """A stable hash of a training dataset (order-sensitive; floats rounded) -- identifies the exact input."""
    return _sha(_canonical(list(data), ndigits))


def param_fingerprint(model: Any, *, ndigits: int = _NDIGITS) -> str:
    """A stable hash of a fitted model's parameters, via its ``to_json`` state (floats rounded)."""
    if not hasattr(model, "to_json"):
        # fall back to repr for models without a JSON state (still deterministic, less portable)
        return _sha(repr(model))
    raw = model.to_json()
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return _sha(str(raw))
    return _sha(_canonical(parsed, ndigits))


@dataclass
class ReproReceipt:
    """The complete executable recipe and observed result needed to re-derive a fit.

    ``max_its``/``delta`` must match whatever was actually passed to the original ``optimize()`` call
    that produced the fingerprinted model (they default to ``optimize()``'s own defaults). Without
    these, :func:`verify_reproducible` cannot replay the same fit -- an iterative estimator refit
    with a different iteration budget or convergence tolerance can land at a different (if nearby)
    optimum and spuriously report ``reproducible: False`` for a fit that would have reproduced exactly
    under its own original settings.
    """

    data_fingerprint: str
    n: int
    seed: int
    estimator: str  # type name of the estimator used (documentation; the object is supplied to verify)
    param_fingerprint: str
    max_its: int = 10
    delta: float | None = 1.0e-9
    schema_version: str = "mixle-repro-v2"
    data_digest: str = ""
    model_digest: str = ""
    estimator_implementation_digest: str = ""
    estimator_state_digest: str = ""
    optimizer_implementation_digest: str = ""
    dependency_digest: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    fit_policy: dict[str, Any] = field(default_factory=dict)
    fit_policy_digest: str = ""
    execution_digest: str = ""
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if not self.fit_policy_digest and self.fit_policy:
            self.fit_policy_digest = canonical_digest(self.fit_policy)
        if not self.receipt_digest:
            self.receipt_digest = canonical_digest(self._content())

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_fingerprint": self.data_fingerprint,
            "data_digest": self.data_digest,
            "n": self.n,
            "seed": self.seed,
            "estimator": self.estimator,
            "estimator_implementation_digest": self.estimator_implementation_digest,
            "estimator_state_digest": self.estimator_state_digest,
            "optimizer_implementation_digest": self.optimizer_implementation_digest,
            "dependency_digest": self.dependency_digest,
            "environment": self.environment,
            "fit_policy": self.fit_policy,
            "fit_policy_digest": self.fit_policy_digest,
            "param_fingerprint": self.param_fingerprint,
            "model_digest": self.model_digest,
            "execution_digest": self.execution_digest,
            "max_its": self.max_its,
            "delta": self.delta,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the receipt as JSON-compatible data."""
        return {**self._content(), "receipt_digest": self.receipt_digest}

    def matches_data(self, data: Any) -> bool:
        """Whether ``data`` is the exact dataset this fit was recorded on."""
        rows = list(data)
        return (
            len(rows) == self.n
            and data_fingerprint(rows) == self.data_fingerprint
            and bool(self.data_digest)
            and dataset_hash(rows) == self.data_digest
        )

    def matches_model(self, model: Any) -> bool:
        """Whether ``model`` has the exact parameters this receipt fingerprinted."""
        return (
            param_fingerprint(model) == self.param_fingerprint
            and bool(self.model_digest)
            and model_hash(model) == self.model_digest
        )


def _execution_environment() -> dict[str, Any]:
    from mixle.inference.production.provenance import _git_state

    return {**dependency_manifest(), "source": _git_state()}


def _estimator_implementation_digest(estimator: Any) -> str:
    estimate = getattr(estimator, "estimate", None)
    if not callable(estimate):
        raise TypeError("estimator must expose an inspectable estimate() implementation")
    return implementation_digest(estimate)


def _validate_fit_contract(
    *, seed: Any, max_its: Any, delta: Any, fit_options: dict[str, Any] | None
) -> tuple[int, int, float | None, dict[str, Any]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(max_its, bool) or not isinstance(max_its, int) or max_its < 0:
        raise ValueError("max_its must be a nonnegative integer")
    if delta is not None and (
        isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0
    ):
        raise ValueError("delta must be a finite nonnegative number or None")
    options = dict(fit_options or {})
    reserved = {"seed", "rng", "max_its", "delta", "out"} & options.keys()
    if reserved:
        raise ValueError(f"fit_options cannot override recorded controls: {sorted(reserved)}")
    # Fail closed on callbacks, mutable RNGs, address-bearing objects, or non-finite values that
    # cannot be represented as a portable replay request.
    canonical_json(options, semantic=False)
    return int(seed), int(max_its), None if delta is None else float(delta), options


def record_fit(
    model: Any,
    data: Any,
    *,
    seed: int,
    estimator: Any = None,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    fit_options: dict[str, Any] | None = None,
) -> ReproReceipt:
    """Record a :class:`ReproReceipt` for a model fitted on ``data`` with ``seed`` (see module docstring).

    Pass the same ``max_its``/``delta`` actually used for the fit being recorded (they default to
    ``optimize()``'s own defaults); :func:`verify_reproducible` replays the fit with these exact
    values."""
    if estimator is None:
        raise ValueError("estimator is required to bind a reproducible executable fit contract")
    rows = list(data)
    use_seed, use_max_its, use_delta, options = _validate_fit_contract(
        seed=seed, max_its=max_its, delta=delta, fit_options=fit_options
    )
    from mixle.inference.estimation import optimize

    est_name = f"{type(estimator).__module__}.{type(estimator).__qualname__}"
    data_digest = dataset_hash(rows)
    model_digest = model_hash(model)
    estimator_implementation = _estimator_implementation_digest(estimator)
    estimator_state = object_state_digest(estimator)
    optimizer_implementation = implementation_digest(optimize)
    environment = _execution_environment()
    dependencies = canonical_digest(environment)
    policy = {
        "seed": use_seed,
        "max_its": use_max_its,
        "delta": use_delta,
        "options": options,
    }
    policy_digest = canonical_digest(policy)
    execution_digest = canonical_digest(
        {
            "data_digest": data_digest,
            "model_digest": model_digest,
            "estimator_implementation_digest": estimator_implementation,
            "estimator_state_digest": estimator_state,
            "optimizer_implementation_digest": optimizer_implementation,
            "dependency_digest": dependencies,
            "fit_policy_digest": policy_digest,
        }
    )
    return ReproReceipt(
        data_fingerprint=data_fingerprint(rows),
        data_digest=data_digest,
        n=len(rows),
        seed=use_seed,
        estimator=est_name,
        param_fingerprint=param_fingerprint(model),
        model_digest=model_digest,
        max_its=use_max_its,
        delta=use_delta,
        estimator_implementation_digest=estimator_implementation,
        estimator_state_digest=estimator_state,
        optimizer_implementation_digest=optimizer_implementation,
        dependency_digest=dependencies,
        environment=environment,
        fit_policy=policy,
        fit_policy_digest=policy_digest,
        execution_digest=execution_digest,
    )


def fit_and_record(
    data: Any,
    estimator: Any,
    *,
    seed: int,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    fit_options: dict[str, Any] | None = None,
) -> tuple[Any, ReproReceipt]:
    """Execute one materialized fit and return the model with its complete replay receipt."""
    from mixle.inference.estimation import optimize

    rows = list(data)
    use_seed, use_max_its, use_delta, options = _validate_fit_contract(
        seed=seed, max_its=max_its, delta=delta, fit_options=fit_options
    )
    model = optimize(
        rows,
        estimator,
        seed=use_seed,
        out=None,
        max_its=use_max_its,
        delta=use_delta,
        **options,
    )
    return model, record_fit(
        model,
        rows,
        seed=use_seed,
        estimator=estimator,
        max_its=use_max_its,
        delta=use_delta,
        fit_options=options,
    )


def verify_reproducible(
    estimator: Any, data: Any, receipt: ReproReceipt, *, seed: int | None = None, max_its: int | None = None
) -> dict[str, Any]:
    """Refit ``estimator`` on ``data`` and check the fit reproduces ``receipt`` (data + parameters).

    Returns ``{reproducible, data_matches, params_match, refit_fingerprint}``. ``reproducible`` is True
    iff BOTH the data fingerprint and the refit's parameter fingerprint match the receipt -- i.e. the
    exact fit can be recovered from the recorded recipe. ``seed`` defaults to the receipt's seed.
    ``max_its`` defaults to the receipt's own recorded value (the iteration budget the original fit
    actually used); ``delta`` always replays the receipt's recorded value -- a refit run with a
    different iteration budget or convergence tolerance than the original fit can land at a
    different (if nearby) optimum and spuriously report ``reproducible: False`` for a fit that would
    have reproduced exactly under its own original settings."""
    from mixle.inference.estimation import optimize

    rows = list(data)
    use_seed = receipt.seed if seed is None else int(seed)
    use_max_its = receipt.max_its if max_its is None else int(max_its)
    options = dict(receipt.fit_policy.get("options") or {})
    replay_policy = {
        "seed": use_seed,
        "max_its": use_max_its,
        "delta": receipt.delta,
        "options": options,
    }
    current_environment = _execution_environment()
    current_estimator_implementation = _estimator_implementation_digest(estimator)
    current_estimator_state = object_state_digest(estimator)
    current_optimizer_implementation = implementation_digest(optimize)
    checks = {
        "receipt_integrity": receipt.receipt_digest == canonical_digest(receipt._content()),
        "schema_supported": receipt.schema_version == "mixle-repro-v2",
        "data_count_matches": len(rows) == receipt.n,
        "data_tolerance_matches": data_fingerprint(rows) == receipt.data_fingerprint,
        "data_digest_matches": bool(receipt.data_digest) and dataset_hash(rows) == receipt.data_digest,
        "estimator_name_matches": receipt.estimator == f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "estimator_implementation_matches": current_estimator_implementation == receipt.estimator_implementation_digest,
        "estimator_state_matches": current_estimator_state == receipt.estimator_state_digest,
        "optimizer_implementation_matches": current_optimizer_implementation == receipt.optimizer_implementation_digest,
        "dependency_environment_matches": current_environment == receipt.environment
        and canonical_digest(current_environment) == receipt.dependency_digest,
        "fit_policy_matches": canonical_digest(replay_policy) == receipt.fit_policy_digest,
    }
    refit = optimize(
        rows,
        estimator,
        out=None,
        max_its=use_max_its,
        delta=receipt.delta,
        seed=use_seed,
        **options,
    )
    refit_fp = param_fingerprint(refit)
    params_match = refit_fp == receipt.param_fingerprint
    refit_model_digest = model_hash(refit)
    model_digest_matches = bool(receipt.model_digest) and refit_model_digest == receipt.model_digest
    replay_execution_digest = canonical_digest(
        {
            "data_digest": dataset_hash(rows),
            "model_digest": refit_model_digest,
            "estimator_implementation_digest": current_estimator_implementation,
            "estimator_state_digest": current_estimator_state,
            "optimizer_implementation_digest": current_optimizer_implementation,
            "dependency_digest": canonical_digest(current_environment),
            "fit_policy_digest": canonical_digest(replay_policy),
        }
    )
    checks["params_match"] = params_match
    checks["model_digest_matches"] = model_digest_matches
    checks["execution_digest_matches"] = replay_execution_digest == receipt.execution_digest
    reproducible = all(checks.values())
    return {
        "reproducible": bool(reproducible),
        "data_matches": bool(checks["data_count_matches"] and checks["data_digest_matches"]),
        "params_match": bool(params_match),
        "refit_fingerprint": refit_fp,
        "refit_model_digest": refit_model_digest,
        "checks": checks,
        "observed_execution": True,
    }
