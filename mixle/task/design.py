"""LLM-designed models: let an LLM propose the mixle structure from data -- and make mixle *validate* it.

The hardcoded auto-estimator (:func:`mixle.task.recommend.recommend_model`) is a fine fallback, but the
differentiator is flexibility: an LLM can read a data profile and propose a structure no fixed heuristic encodes
-- a mixture here, a heavy-tailed leaf there, a composite of mixed families. The risk with "LLM picks the model"
is hallucination; the answer is grounding. The LLM emits a small, **allowlisted JSON spec** (no code, no eval);
:func:`spec_to_estimator` builds a real mixle estimator from it; and :func:`design_model` *fits* it before
trusting it, falling back to the heuristic when the LLM is unavailable or its spec fails to build or fit.

So the LLM proposes and mixle disposes: you get flexible, data-aware model design with a hard correctness gate
(it must parse, build, and fit), not a plausible-looking guess. ``design_model(data, llm)`` returns the chosen
estimator, the spec, and whether it came from the LLM or the fallback.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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


def spec_to_estimator(spec: dict[str, Any]) -> Any:
    """Build a mixle estimator from an allowlisted spec dict (recursively); raise on anything off the allowlist.

    Specs:
      * ``{"family": "<name>"}``                       -- a scalar leaf (see :data:`ALLOWED_FAMILIES`);
      * ``{"type": "composite", "fields": [spec, ...]}`` -- a tuple record of sub-models;
      * ``{"type": "mixture", "k": K, "component": spec}`` -- a K-component mixture of the component model.
    """
    import mixle.stats as st

    if "family" in spec:
        family = str(spec["family"]).lower()
        if family not in _FAMILY:
            raise ValueError(f"family {family!r} not in allowlist {ALLOWED_FAMILIES}")
        return getattr(st, _FAMILY[family])()

    kind = spec.get("type")
    if kind == "composite":
        fields = spec.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("composite spec needs a non-empty 'fields' list")
        return st.CompositeEstimator(tuple(spec_to_estimator(f) for f in fields))
    if kind == "mixture":
        k = int(spec.get("k", 0))
        if k < 1 or "component" not in spec:
            raise ValueError("mixture spec needs k>=1 and a 'component'")
        comp = spec["component"]
        return st.MixtureEstimator([spec_to_estimator(comp) for _ in range(k)])
    raise ValueError(f"unrecognized spec {spec!r}")


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


@dataclass
class DesignedModel:
    """The model an LLM (or the fallback) designed: the estimator, the spec it built from, and the source."""

    estimator: Any
    spec: dict[str, Any] | None
    source: str  # "llm" | "fallback"
    note: str = ""

    def fit(self, data: Sequence[Any], **kwargs: Any) -> Any:
        """Fit the designed estimator with ``mixle.inference.optimize``."""
        from mixle.inference import optimize

        return optimize(list(data), self.estimator, **kwargs)


def design_model(
    data: Sequence[Any],
    llm: Any,
    *,
    fallback: bool = True,
    validate_rows: int = 200,
) -> DesignedModel:
    """Ask ``llm`` to design a model for ``data``; build, fit-validate, and fall back to the heuristic on failure.

    The LLM sees a compact :func:`data_profile` and returns a JSON spec; the spec is built into a real estimator
    and fit on a sample to prove it works. Any failure (no LLM, invalid JSON, off-allowlist family, fit error) yields
    the heuristic :func:`mixle.task.recommend.recommend_model` estimator when ``fallback`` is set.
    """
    profile = data_profile(data)
    note = ""
    try:
        reply = llm.complete(json.dumps(profile, indent=2), system=_DESIGN_SYSTEM)
        spec = _extract_json(reply)
        estimator = spec_to_estimator(spec)
        _fit_validate(estimator, data, validate_rows)
        return DesignedModel(estimator=estimator, spec=spec, source="llm")
    except Exception as exc:  # noqa: BLE001 - any failure must degrade to the grounded fallback
        note = f"LLM design failed ({type(exc).__name__}: {exc}); used heuristic fallback"
        if not fallback:
            raise
    from mixle.task.recommend import recommend_model

    return DesignedModel(estimator=recommend_model(data).estimator, spec=None, source="fallback", note=note)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (tolerates code fences / surrounding prose)."""
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("no JSON object found in LLM reply")


def _fit_validate(estimator: Any, data: Sequence[Any], rows: int) -> None:
    from mixle.inference import optimize

    sample = list(data)[:rows]
    model = optimize(sample, estimator, max_its=3, out=None)
    enc = model.dist_to_encoder().seq_encode(sample)
    import numpy as np

    if not np.all(np.isfinite(model.seq_log_density(enc))):
        raise ValueError("fitted model produced non-finite log-density")
