"""LLM-designed models: let an LLM propose the mixle structure from data -- and make mixle *validate* it.

The hardcoded auto-estimator (:func:`mixle.task.recommend.recommend_model`) is a fine fallback, but the
differentiator is flexibility: an LLM can read a data profile and propose a structure no fixed heuristic encodes
-- a mixture here, a heavy-tailed leaf there, a composite of mixed families. The risk with "LLM picks the model"
is hallucination; the answer is grounding. The LLM emits a small, **allowlisted JSON spec** (no code, no eval);
:func:`spec_to_estimator` validates and builds a real mixle estimator; :func:`design_model` checks its
shape against the observations, fits it without the holdout, and compares independent held-out density
with a separately fitted heuristic baseline before trusting it.

So the LLM proposes and mixle disposes. ``design_model(data, llm)`` returns the chosen estimator,
the accepted spec/source, an acceptance receipt, and structured evidence about a rejected proposal.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.task.llm import extract_json_object

# Allowlisted scalar families -> mixle estimator class name. No eval, no arbitrary import: a fixed map.
_FAMILY = {
    "gaussian": "GaussianEstimator",
    "student_t": "StudentTEstimator",
    "lognormal": "LogGaussianEstimator",
    "log_gaussian": "LogGaussianEstimator",
    "gamma": "GammaEstimator",
    "exponential": "ExponentialEstimator",
    "inverse_gaussian": "InverseGaussianEstimator",
    "weibull": "WeibullEstimator",
    "beta": "BetaEstimator",
    "poisson": "PoissonEstimator",
    "geometric": "GeometricEstimator",
    "bernoulli": "BernoulliEstimator",
    "binomial": "BinomialEstimator",
    "categorical": "CategoricalEstimator",
}

ALLOWED_FAMILIES = tuple(sorted(_FAMILY))

_MAX_SPEC_DEPTH = 16
_MAX_SPEC_NODES = 256
_MAX_MIXTURE_COMPONENTS = 64


def spec_to_estimator(spec: dict[str, Any]) -> Any:
    """Build a mixle estimator from an allowlisted spec dict (recursively); raise on anything off the allowlist.

    Specs:
      * ``{"family": "<name>"}``                       -- a scalar leaf (see :data:`ALLOWED_FAMILIES`);
      * ``{"type": "composite", "fields": [spec, ...]}`` -- a tuple record of sub-models;
      * ``{"type": "mixture", "k": K, "component": spec}`` -- a K-component mixture of the component model.
    """
    import mixle.stats as st

    nodes = {"count": 0}

    def build(node: Any, depth: int) -> Any:
        if not isinstance(node, dict):
            raise TypeError("every model spec node must be a JSON object")
        nodes["count"] += 1
        if depth > _MAX_SPEC_DEPTH or nodes["count"] > _MAX_SPEC_NODES:
            raise ValueError("model spec exceeds the supported complexity limits")
        if set(node) == {"family"}:
            family_value = node["family"]
            if not isinstance(family_value, str) or not family_value:
                raise TypeError("family must be a non-empty string")
            family = family_value.lower()
            if family not in _FAMILY:
                raise ValueError(f"family {family!r} not in allowlist {ALLOWED_FAMILIES}")
            return getattr(st, _FAMILY[family])()

        kind = node.get("type")
        if kind == "composite":
            if set(node) != {"type", "fields"}:
                raise ValueError("composite spec accepts exactly 'type' and 'fields'")
            fields = node["fields"]
            if not isinstance(fields, list) or not fields:
                raise ValueError("composite spec needs a non-empty 'fields' list")
            return st.CompositeEstimator(tuple(build(field, depth + 1) for field in fields))
        if kind == "mixture":
            if set(node) != {"type", "k", "component"}:
                raise ValueError("mixture spec accepts exactly 'type', 'k', and 'component'")
            k_value = node["k"]
            if isinstance(k_value, (bool, np.bool_)) or not isinstance(k_value, Integral):
                raise TypeError("mixture k must be an integer")
            k = int(k_value)
            if not 1 <= k <= _MAX_MIXTURE_COMPONENTS:
                raise ValueError(f"mixture k must be in [1, {_MAX_MIXTURE_COMPONENTS}]")
            return st.MixtureEstimator([build(node["component"], depth + 1) for _ in range(k)])
        raise ValueError(f"unrecognized spec {node!r}")

    return build(spec, 0)


def data_profile(data: Sequence[Any], *, max_rows: int = 500) -> dict[str, Any]:
    """A compact, family-agnostic description of ``data`` for the LLM: per-field kind and a few sample values."""
    from mixle.utils.automatic import analyze_structure

    profile = analyze_structure(list(data)[:max_rows], pairwise=False, validate_marginals=False)
    fields = []
    for fp in profile.fields:
        from mixle.utils.automatic import format_path

        entry: dict[str, Any] = {"path": format_path(fp.path), "kind": fp.kind}
        if fp.numeric_mean is not None:
            entry["mean"] = round(float(fp.numeric_mean), 4)
        if fp.numeric_var is not None:
            entry["var"] = round(float(fp.numeric_var), 4)
        if fp.cardinality is not None:
            entry["cardinality"] = int(fp.cardinality)
        if fp.integer_min is not None:
            entry["min"], entry["max"] = int(fp.integer_min), int(fp.integer_max)
        fields.append(entry)
    return {"n_rows": len(data), "fields": fields}


_DESIGN_SYSTEM = (
    "You are a probabilistic modeler. Given a data profile, design a mixle model as JSON only (no prose). "
    "Use: {'family': one of " + ", ".join(ALLOWED_FAMILIES) + "} for a scalar; "
    "{'type':'composite','fields':[...]} for a record of several fields in order; "
    "{'type':'mixture','k':K,'component':{...}} for a K-cluster mixture. Output a single JSON object."
)


@dataclass(frozen=True)
class DesignAcceptanceReceipt:
    """Independent train/holdout evidence for one proposed or fallback estimator."""

    accepted: bool
    schema_compatible: bool
    n_train: int
    n_holdout: int
    train_mean_log_density: float
    holdout_mean_log_density: float
    baseline_holdout_mean_log_density: float
    holdout_regret: float
    max_holdout_regret: float
    reason: str


@dataclass(frozen=True)
class DesignFailureReceipt:
    """Structured evidence retained when an LLM proposal is rejected."""

    stage: str
    error_type: str
    message: str
    proposed_spec: dict[str, Any] | None
    reply_excerpt: str
    acceptance: DesignAcceptanceReceipt | None = None


@dataclass
class DesignedModel:
    """Chosen estimator, provenance, and independent acceptance/rejection evidence."""

    estimator: Any
    spec: dict[str, Any] | None
    source: str  # "llm" | "fallback"
    note: str = ""
    acceptance: DesignAcceptanceReceipt | None = None
    failure: DesignFailureReceipt | None = None

    def fit(self, data: Sequence[Any], **kwargs: Any) -> Any:
        """Fit the designed estimator with ``mixle.inference.optimize``."""
        from mixle.inference import optimize

        rows = list(data)
        if not rows:
            raise ValueError("fit requires non-empty data")
        return optimize(rows, self.estimator, **kwargs)


def _validate_family_values(family: str, values: list[Any]) -> None:
    def real(value: Any) -> bool:
        return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and np.isfinite(float(value))

    def integer(value: Any) -> bool:
        return not isinstance(value, (bool, np.bool_)) and isinstance(value, Integral)

    if family == "categorical":
        for value in values:
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError("categorical observations must be hashable scalars") from exc
        return
    if family in {"gaussian", "student_t"}:
        valid = all(real(value) for value in values)
    elif family in {
        "lognormal",
        "log_gaussian",
        "gamma",
        "exponential",
        "inverse_gaussian",
        "weibull",
    }:
        valid = all(real(value) and float(value) > 0.0 for value in values)
    elif family == "beta":
        valid = all(real(value) and 0.0 < float(value) < 1.0 for value in values)
    elif family in {"poisson", "binomial"}:
        valid = all(integer(value) and int(value) >= 0 for value in values)
    elif family == "geometric":
        valid = all(integer(value) and int(value) >= 1 for value in values)
    elif family == "bernoulli":
        valid = all(integer(value) and int(value) in (0, 1) for value in values)
    else:  # guarded by spec_to_estimator
        valid = False
    if not valid:
        raise ValueError(f"observations are outside the declared {family!r} family support")


def _validate_spec_against_data(spec: dict[str, Any], rows: list[Any]) -> None:
    """Validate recursive observation shape and scalar support before fitting."""
    if "family" in spec:
        if any(isinstance(row, (tuple, list, dict, np.ndarray)) for row in rows):
            raise ValueError("scalar family spec is incompatible with structured observations")
        _validate_family_values(spec["family"].lower(), rows)
        return
    if spec.get("type") == "mixture":
        _validate_spec_against_data(spec["component"], rows)
        return
    if spec.get("type") == "composite":
        fields = spec["fields"]
        normalized: list[list[Any]] = []
        for row in rows:
            if isinstance(row, np.ndarray):
                if row.ndim != 1:
                    raise ValueError("composite observations must be one-dimensional records")
                values = row.tolist()
            elif isinstance(row, (tuple, list)):
                values = list(row)
            else:
                raise ValueError("composite spec requires tuple/list observations")
            if len(values) != len(fields):
                raise ValueError(f"composite spec has {len(fields)} fields but an observation has {len(values)}")
            normalized.append(values)
        for index, field_spec in enumerate(fields):
            _validate_spec_against_data(field_spec, [row[index] for row in normalized])
        return
    raise ValueError("model spec does not match the supported schema")


def _split_validation_data(
    rows: list[Any],
    *,
    validate_rows: int,
    holdout_frac: float,
    validation_seed: int,
) -> tuple[list[Any], list[Any]]:
    if isinstance(validate_rows, (bool, np.bool_)) or not isinstance(validate_rows, Integral):
        raise TypeError("validate_rows must be an integer")
    validate_rows = int(validate_rows)
    if validate_rows < 4:
        raise ValueError("validate_rows must be at least 4")
    if (
        isinstance(holdout_frac, (bool, np.bool_))
        or not isinstance(holdout_frac, Real)
        or not np.isfinite(float(holdout_frac))
        or not 0.0 < float(holdout_frac) < 1.0
    ):
        raise ValueError("holdout_frac must be a finite number in (0, 1)")
    if isinstance(validation_seed, (bool, np.bool_)) or not isinstance(validation_seed, Integral):
        raise TypeError("validation_seed must be an integer")
    validation_seed = int(validation_seed)
    if not 0 <= validation_seed <= np.iinfo(np.uint32).max:
        raise ValueError("validation_seed must fit in an unsigned 32-bit integer")
    if len(rows) < 4:
        raise ValueError("model design requires at least four observations for independent validation")
    n = min(len(rows), validate_rows)
    indices = np.random.RandomState(validation_seed).permutation(len(rows))[:n]
    n_holdout = max(1, int(round(n * float(holdout_frac))))
    if n_holdout >= n:
        raise ValueError("holdout_frac leaves no training observations")
    holdout_ids = indices[:n_holdout]
    train_ids = indices[n_holdout:]
    return [rows[int(i)] for i in train_ids], [rows[int(i)] for i in holdout_ids]


def _fit_scores(estimator: Any, train: list[Any], holdout: list[Any], max_its: int) -> tuple[float, float]:
    from mixle.inference import optimize

    fitted = optimize(train, copy.deepcopy(estimator), max_its=max_its, out=None)

    def score(values: list[Any], name: str) -> float:
        encoded = fitted.dist_to_encoder().seq_encode(values)
        density = np.asarray(fitted.seq_log_density(encoded), dtype=np.float64).reshape(-1)
        if density.size != len(values) or not np.all(np.isfinite(density)):
            raise ValueError(f"fitted model produced invalid {name} predictive log density")
        return float(np.mean(density))

    return score(train, "training"), score(holdout, "holdout")


def _candidate_acceptance(
    estimator: Any,
    spec: dict[str, Any],
    fallback_estimator: Any,
    train: list[Any],
    holdout: list[Any],
    *,
    validation_max_its: int,
    max_holdout_regret: float,
) -> DesignAcceptanceReceipt:
    _validate_spec_against_data(spec, [*train, *holdout])
    train_score, holdout_score = _fit_scores(estimator, train, holdout, validation_max_its)
    _, baseline_holdout = _fit_scores(fallback_estimator, train, holdout, validation_max_its)
    regret = baseline_holdout - holdout_score
    accepted = regret <= max_holdout_regret
    reason = (
        "held-out predictive density is within the declared baseline budget"
        if accepted
        else f"held-out log-density regret {regret:.6g} exceeds {max_holdout_regret:.6g}"
    )
    return DesignAcceptanceReceipt(
        accepted=accepted,
        schema_compatible=True,
        n_train=len(train),
        n_holdout=len(holdout),
        train_mean_log_density=train_score,
        holdout_mean_log_density=holdout_score,
        baseline_holdout_mean_log_density=baseline_holdout,
        holdout_regret=regret,
        max_holdout_regret=max_holdout_regret,
        reason=reason,
    )


def _fallback_acceptance(
    estimator: Any,
    train: list[Any],
    holdout: list[Any],
    *,
    validation_max_its: int,
    max_holdout_regret: float,
) -> DesignAcceptanceReceipt:
    train_score, holdout_score = _fit_scores(estimator, train, holdout, validation_max_its)
    return DesignAcceptanceReceipt(
        accepted=True,
        schema_compatible=True,
        n_train=len(train),
        n_holdout=len(holdout),
        train_mean_log_density=train_score,
        holdout_mean_log_density=holdout_score,
        baseline_holdout_mean_log_density=holdout_score,
        holdout_regret=0.0,
        max_holdout_regret=max_holdout_regret,
        reason="heuristic fallback fit and produced finite independent holdout density",
    )


def design_model(
    data: Sequence[Any],
    llm: Any,
    *,
    fallback: bool = True,
    validate_rows: int = 200,
    holdout_frac: float = 0.25,
    validation_seed: int = 17,
    validation_max_its: int = 10,
    max_holdout_regret: float = 1.0,
) -> DesignedModel:
    """Accept an LLM design only on schema-compatible independent predictive evidence.

    A deterministic subset is split before fitting. The LLM estimator sees only the training part and
    must produce finite holdout density no more than ``max_holdout_regret`` nats/observation below an
    independently fitted heuristic estimator. A fallback is returned only after it passes the same
    fit/holdout validity checks. Rejected proposals remain available in ``DesignedModel.failure``.
    """
    if isinstance(data, (str, bytes)):
        raise TypeError("data must be a sequence of observations")
    rows = list(data)
    train, holdout = _split_validation_data(
        rows,
        validate_rows=validate_rows,
        holdout_frac=holdout_frac,
        validation_seed=validation_seed,
    )
    if isinstance(validation_max_its, (bool, np.bool_)) or not isinstance(validation_max_its, Integral):
        raise TypeError("validation_max_its must be an integer")
    validation_max_its = int(validation_max_its)
    if validation_max_its < 1:
        raise ValueError("validation_max_its must be positive")
    if (
        isinstance(max_holdout_regret, (bool, np.bool_))
        or not isinstance(max_holdout_regret, Real)
        or not np.isfinite(float(max_holdout_regret))
        or float(max_holdout_regret) < 0.0
    ):
        raise ValueError("max_holdout_regret must be finite and non-negative")
    max_holdout_regret = float(max_holdout_regret)

    profile = data_profile(rows)
    from mixle.task.recommend import recommend_model

    fallback_estimator = recommend_model(rows).estimator
    stage = "llm_call"
    reply = ""
    proposed_spec: dict[str, Any] | None = None
    proposal_acceptance: DesignAcceptanceReceipt | None = None
    try:
        if llm is None or not callable(getattr(llm, "complete", None)):
            raise TypeError("llm must expose complete(prompt, system=...)")
        reply = llm.complete(json.dumps(profile, indent=2), system=_DESIGN_SYSTEM)
        stage = "json_parse"
        proposed_spec = _extract_json(reply)
        stage = "spec_construction"
        estimator = spec_to_estimator(proposed_spec)
        stage = "independent_acceptance"
        proposal_acceptance = _candidate_acceptance(
            estimator,
            proposed_spec,
            fallback_estimator,
            train,
            holdout,
            validation_max_its=validation_max_its,
            max_holdout_regret=max_holdout_regret,
        )
        if not proposal_acceptance.accepted:
            raise ValueError(proposal_acceptance.reason)
        return DesignedModel(
            estimator=estimator,
            spec=proposed_spec,
            source="llm",
            acceptance=proposal_acceptance,
        )
    except Exception as exc:  # noqa: BLE001 - external proposal failures are retained as structured evidence
        failure = DesignFailureReceipt(
            stage=stage,
            error_type=type(exc).__name__,
            message=str(exc),
            proposed_spec=proposed_spec,
            reply_excerpt=reply[:500] if isinstance(reply, str) else repr(reply)[:500],
            acceptance=proposal_acceptance,
        )
        if not fallback:
            raise

    fallback_receipt = _fallback_acceptance(
        fallback_estimator,
        train,
        holdout,
        validation_max_its=validation_max_its,
        max_holdout_regret=max_holdout_regret,
    )
    note = (
        f"LLM design failed at {failure.stage} "
        f"({failure.error_type}: {failure.message}); used validated heuristic fallback"
    )
    return DesignedModel(
        estimator=fallback_estimator,
        spec=None,
        source="fallback",
        note=note,
        acceptance=fallback_receipt,
        failure=failure,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (tolerates code fences / surrounding prose)."""
    return extract_json_object(text)
