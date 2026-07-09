"""Decision-oriented model recommendation for heterogeneous data samples.

``recommend_model`` wraps :func:`mixle.utils.automatic.analyze_structure` and
returns an object a program can inspect, store, and optionally fit. The
recommendation includes:

* the selected estimator;
* per-field family choices, runner-up families, and bit-scale confidence gaps;
* low-confidence fields where more data or domain review would sharpen the
  choice;
* pairwise dependencies that support modeling fields jointly; and
* profile warnings that should be reviewed before production use.

Pass ``fit=True`` to attach a fitted model to the returned recommendation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

CONFIDENT_GAP_BITS = 0.02  # a family must beat the runner-up by this many bits/obs to count as a decisive choice


@dataclass
class FieldChoice:
    """The family chosen for one field, the runner-up, and how decisive the choice was (bits/obs)."""

    path: str
    kind: str
    family: str
    runner_up: str | None
    gap_bits: float | None  # description-length advantage of family over runner_up (None when type-determined)

    @property
    def confident(self) -> bool:
        """Confident when the family is type-determined (no real contender) or clears the runner-up by a margin."""
        if self.runner_up is None:
            return True  # the data type fixes the family (e.g. a string field -> categorical)
        return self.gap_bits is not None and self.gap_bits >= CONFIDENT_GAP_BITS


@dataclass
class ModelRecommendation:
    """A model shape recommended from data: estimator, per-field choices+confidence, dependencies, and notes."""

    estimator: Any
    fields: list[FieldChoice]
    dependencies: list[tuple[str, str, float]]  # (left, right, bic_gain_bits) -- argue for joint modeling
    warnings: list[str]
    profile: Any = field(default=None, repr=False)

    def low_confidence_fields(self) -> list[FieldChoice]:
        """Fields whose family choice is not yet decisive -- where more data would most sharpen the model."""
        return [c for c in self.fields if not c.confident]

    def fit(self, data: Sequence[Any], **kwargs: Any) -> Any:
        """Fit the recommended estimator on ``data`` and return the model."""
        from mixle.inference import optimize

        return optimize(list(data), self.estimator, **kwargs)

    def explain(self) -> list[str]:
        """Plain-language lines: the underlying profile's explanation (families, bits, dependencies, warnings)."""
        return self.profile.explain() if self.profile is not None else []


def recommend_model(data: Sequence[Any], *, fit: bool = False, **analyze_kwargs: Any) -> ModelRecommendation:
    """Recommend a model shape for ``data`` (and optionally fit it); see :class:`ModelRecommendation`.

    ``analyze_kwargs`` pass through to :func:`mixle.utils.automatic.analyze_structure` (sampling, pairwise
    budget, validation). With ``fit=True`` the returned recommendation's ``estimator`` is also fit and the
    model is attached as ``.model``.
    """
    from mixle.utils.automatic import analyze_structure

    profile = analyze_structure(data, **analyze_kwargs)
    fields = [_field_choice(fp) for fp in profile.fields]
    deps = [(_fmt(h.left), _fmt(h.right), float(h.bic_gain_bits)) for h in profile.pairwise_hints]
    rec = ModelRecommendation(
        estimator=profile.recommend(),
        fields=fields,
        dependencies=deps,
        warnings=list(profile.warnings),
        profile=profile,
    )
    if fit:
        rec.model = rec.fit(data)  # type: ignore[attr-defined]
    return rec


def _fmt(path: Any) -> str:
    from mixle.utils.automatic import format_path

    return format_path(path)


def _field_choice(fp: Any) -> FieldChoice:
    scores = dict(fp.model_scores_bits or {})  # family -> bits/obs (lower is better)
    family = fp.recommendation
    runner_up, gap = None, fp.model_score_gap_bits
    if scores:
        others = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1]) if k != family]
        runner_up = others[0] if others else None
        if gap is None and runner_up is not None and family in scores:
            gap = float(scores[runner_up] - scores[family])
    return FieldChoice(
        path=_fmt(fp.path),
        kind=fp.kind,
        family=family,
        runner_up=runner_up,
        gap_bits=None if gap is None else float(gap),
    )
