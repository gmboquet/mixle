"""The language<->belief bridge (roadmap M5, part (c)): NL/record -> M0 evidence via a declared
schema, and posterior -> calibrated text through A1's :class:`~mixle.task.calibrated_generator.CalibratedGenerator`.

Two independent directions, each a thin composition of already-tested machinery -- no new extraction
or generation model is built here (see ``notes/designs/M5.md`` part (c) for the full contract):

* :func:`parse_evidence` -- an extractor callable (the same ``teacher(x) -> dict`` shape
  :func:`mixle.task.structured_out.solve_structured` decomposes) produces a raw ``{field: value}``
  dict from NL/record input; this module validates it against a caller-declared schema BEFORE it
  reaches :func:`~mixle.reason.cross_modal.CrossModalJoint.infer` /
  :func:`~mixle.reason.inference_program.run_inference_program`, so a schema violation is a clear
  ``ValueError`` here rather than a confusing downstream ``log_density`` crash. A schema field may be
  declared with the plain shorthand string ``"categorical" | "numeric"`` -- the same shape
  :attr:`~mixle.task.structured_out.StructuredSolution.schema` already returns, treated as a REQUIRED
  field with no further restriction -- or a full :class:`SchemaField` for an OPTIONAL field or a
  categorical field restricted to a declared closed set of values. Validation happens in two passes:
  the schema itself is checked (every declared field's kind, and categories if any) BEFORE the
  extractor ever runs, so a malformed schema cannot hide behind a call that happens not to exercise
  the bad field; then the extractor's actual output is checked field-by-field against it -- every
  required field must be present, every numeric value must be finite, and every categorical value must
  already be a legitimate ``str`` (a member of ``categories`` when declared) rather than an arbitrary
  object silently stringified into a fabricated label.
* :class:`PosteriorDescriber` / :func:`claim_score` -- draft ``k`` candidate :class:`Claim`\\ s at
  different ABSOLUTE precision widths (multiples of a required ``tol``, the same "caller declares the
  precision that counts" contract :func:`mixle.task.regress.solve_regression` already uses for numeric
  fields -- NOT widths relative to the posterior's own spread, which would be scale-invariant and could
  never detect "too diffuse to answer"), score each against the posterior it describes, and serve the
  best one under A1's conformal accept-or-abstain guarantee. :func:`claim_score` is exported standalone
  (not only reachable through :class:`PosteriorDescriber`) so B2 (claim-checking, built elsewhere) can
  score an ALREADY-EMITTED claim against any posterior directly, with no dependency on candidate
  generation/calibration at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.calibrated_generator import ABSTAIN, CalibratedGenerator
from mixle.utils.callables import accepts_call

__all__ = [
    "Schema",
    "SchemaField",
    "parse_evidence",
    "Claim",
    "claim_score",
    "PosteriorDescriber",
    "ABSTAIN",
]

_KINDS = ("categorical", "numeric")


@dataclass(frozen=True)
class SchemaField:
    """One declared evidence field: its value kind, whether the extractor must supply it, and (for
    categorical fields) the closed set of values that count as legitimate.

    ``kind="numeric"`` accepts only finite ``int``/``float`` (``bool`` excluded -- never a genuine
    numeric measurement; NaN/Infinity excluded -- "validated" evidence must be usable arithmetic, not
    merely type-correct). ``kind="categorical"`` accepts only values that are ALREADY a ``str`` --
    never an arbitrary object (including ``None``) silently stringified into a fabricated label -- and,
    when ``categories`` is declared, only a value already a MEMBER of that closed set (an undeclared
    label is rejected, not silently admitted as a new de facto category).

    ``required=True`` (the default) means :func:`parse_evidence` raises if the extractor omits this
    field; ``required=False`` preserves the "partial evidence is fine" contract for fields the caller
    genuinely expects may go unmentioned.
    """

    kind: str
    required: bool = True
    categories: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"schema field kind must be one of {_KINDS}, got {self.kind!r}")
        if self.categories is not None:
            if self.kind != "categorical":
                raise ValueError(f"categories is only valid for kind='categorical', got kind={self.kind!r}")
            cats = frozenset(self.categories)
            if not cats:
                raise ValueError("categories must be non-empty when declared")
            non_str = sorted(repr(c) for c in cats if not isinstance(c, str))
            if non_str:
                raise ValueError(f"categories must all be str, got non-str member(s) {non_str}")
            object.__setattr__(self, "categories", cats)


Schema = dict[str, "SchemaField | str"]  # {field_name: SchemaField(...) | "categorical" | "numeric"}


def _normalize_schema(schema: Schema) -> dict[str, SchemaField]:
    """Validate the SCHEMA ITSELF -- every declared field's kind and categories -- before the extractor
    ever runs, so an invalid schema is caught even for a field a given call's extractor happens to
    omit (previously a field's kind was only checked when the extractor's raw output actually included
    it, so e.g. a typo'd kind string on a field nothing ever populates silently never fired)."""
    if not schema:
        raise ValueError("parse_evidence needs a non-empty schema")
    out: dict[str, SchemaField] = {}
    for key, decl in schema.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"schema field names must be non-empty str, got {key!r}")
        if isinstance(decl, str):
            decl = SchemaField(kind=decl)  # shorthand -- required, no categories restriction
        elif not isinstance(decl, SchemaField):
            raise TypeError(f"schema field {key!r} must be a kind string or a SchemaField, got {type(decl).__name__}")
        out[key] = decl  # SchemaField.__post_init__ already validated kind/categories
    return out


def _validate_evidence(raw: dict[str, Any], schema: dict[str, SchemaField]) -> dict[str, Any]:
    unknown = [k for k in raw if k not in schema]
    if unknown:
        raise ValueError(f"extractor returned undeclared field(s) {sorted(unknown)!r}; schema is {sorted(schema)!r}")
    missing = sorted(k for k, decl in schema.items() if decl.required and k not in raw)
    if missing:
        raise ValueError(f"extractor omitted required field(s) {missing!r}")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        decl = schema[key]
        if decl.kind == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"field {key!r} is declared numeric but the extractor returned {value!r}")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"field {key!r} is declared numeric but the extractor returned non-finite {value!r}")
            out[key] = value
        else:  # "categorical" -- SchemaField.__post_init__ already rejected any other kind
            if not isinstance(value, str):
                raise ValueError(
                    f"field {key!r} is declared categorical but the extractor returned "
                    f"{value!r} ({type(value).__name__}), not a str -- no object is stringified into a label"
                )
            if decl.categories is not None and value not in decl.categories:
                raise ValueError(
                    f"field {key!r} value {value!r} is not one of the declared categories {sorted(decl.categories)!r}"
                )
            out[key] = value
    return out


def parse_evidence(text: Any, schema: Schema, extractor: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
    """NL scenario/constraint (or any record ``extractor`` accepts) -> validated M0/L2 evidence.

    ``extractor(text) -> {field: raw_value}`` does the actual parsing (a keyword/regex rule, a
    calibrated :func:`~mixle.task.structured_out.solve_structured` student, an LLM call -- this module
    is agnostic to how); this function's only job is enforcing the declared ``schema`` BEFORE the
    result is trusted as evidence: the schema itself is validated first (so a malformed schema raises
    even on a call whose extractor happens not to touch the bad field), then every required field must
    be present, numeric fields must be finite numbers, and categorical fields must already be a ``str``
    (and a declared category, when one is set) -- never an arbitrary object silently stringified into a
    fabricated label. The returned dict is ready to pass straight to ``CrossModalJoint.infer(...)`` or
    as ``run_inference_program``'s ``evidence=``.
    """
    fields = _normalize_schema(schema)
    raw = extractor(text)
    if not isinstance(raw, dict):
        raise TypeError(f"extractor(text) must return a dict, got {type(raw).__name__}")
    return _validate_evidence(raw, fields)


@dataclass(frozen=True)
class Claim:
    """A declared interval assertion about one posterior field: ``field`` lies in ``[lo, hi]``.

    ``probe`` caches the sample batch a :class:`PosteriorDescriber` drew to score this claim at
    generation time, so :meth:`~mixle.task.calibrated_generator.CalibratedGenerator`'s single-argument
    ``score(candidate)`` contract can call :func:`claim_score` with no extra prompt/posterior plumbing.
    A hand-authored ``Claim`` (e.g. from B2) simply omits ``probe`` and passes ``posterior=`` to
    :func:`claim_score` explicitly instead.
    """

    field: str
    lo: float
    hi: float
    probe: tuple[float, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise ValueError(f"Claim.field must be a non-empty str, got {self.field!r}")
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            raise ValueError(f"Claim bounds must be finite, got lo={self.lo!r}, hi={self.hi!r}")
        if self.hi < self.lo:
            raise ValueError(f"Claim bounds must satisfy lo <= hi, got lo={self.lo!r}, hi={self.hi!r}")

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def text(self) -> str:
        mid = 0.5 * (self.lo + self.hi)
        if self.width < 1e-9:
            return f"{self.field} is approximately {mid:.4g}"
        return f"{self.field} is between {self.lo:.4g} and {self.hi:.4g}"


def _validate_n_samples(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n!r}")


def _coerce_scalar_draws(draws: Sequence[Any]) -> np.ndarray:
    """The ONE validated path every draw source funnels through, whether the batch came from a live
    posterior's ``.sample()``, a plain sequence passed straight through, or a 1-D array passed straight
    through: each draw must already be a scalar or a length-1 scalar container -- exactly
    ``_sample_scalar``'s documented scope limit, now actually enforced identically for all three
    sources instead of ndarrays and sequences silently taking their own, unvalidated shortcuts. An
    empty batch is rejected outright rather than left to produce NaN "support" downstream (``np.mean``
    of an empty array)."""
    draws = list(draws)
    if not draws:
        raise ValueError("_sample_scalar received zero draws (empty posterior/array/sequence)")
    out = []
    for d in draws:
        if isinstance(d, (int, float, np.floating, np.integer)) and not isinstance(d, bool):
            out.append(float(d))
        elif isinstance(d, (tuple, list, np.ndarray)) and len(d) == 1:
            out.append(float(d[0]))
        else:
            raise ValueError(f"expected a scalar or single-field draw, got {d!r}")
    return np.asarray(out, dtype=float)


def _sample_scalar(posterior: Any, n: int, seed: int | None) -> np.ndarray:
    """Extract ``n`` scalar draws of a single field from any M0/L2/M5 posterior-like object, or pass a
    plain array/sequence of scalars through unchanged. A real, documented scope limit (not silently
    wrong for other shapes): this only handles objects whose draws are already scalars or 1-tuples --
    exactly what a single-field :class:`~mixle.reason.inference_program.ProgramPosterior` or a
    single-target :class:`~mixle.stats.latent.mixture.MixtureDistribution`
    (``CrossModalJoint.infer(..., [field])``) produce. A multi-dimensional array is a shape violation,
    not a batch to flatten -- rejected outright rather than silently averaged over via boolean
    broadcasting downstream."""
    _validate_n_samples(n)
    if isinstance(posterior, np.ndarray):
        if posterior.ndim != 1:
            raise ValueError(
                f"_sample_scalar only accepts a 1-D array of scalar draws, got shape {posterior.shape}; "
                "a multi-dimensional array is not silently flattened into a scalar batch"
            )
        return _coerce_scalar_draws(posterior.tolist())
    if isinstance(posterior, Sequence) and not isinstance(posterior, (str, bytes)):
        return _coerce_scalar_draws(posterior)
    if hasattr(posterior, "sample"):
        if accepts_call(posterior.sample, n, seed=seed):
            draws = posterior.sample(n, seed=seed)
        else:
            draws = posterior.sample(n)
    elif hasattr(posterior, "sampler"):
        draws = posterior.sampler(seed=seed).sample(n)
    else:
        raise TypeError(f"do not know how to sample a field from {type(posterior).__name__}")
    return _coerce_scalar_draws(draws)


def claim_score(claim: Claim, posterior: Any = None, *, n_samples: int = 200, seed: int | None = 0) -> float:
    """How well ``claim`` is supported by the posterior it describes: coverage of ``[claim.lo,
    claim.hi]`` under fresh posterior draws, PER UNIT WIDTH (``coverage / width``) -- a density-like
    score, not a linear coverage-minus-penalty one. The ratio form matters, not just tie-breaking: a
    coverage-minus-linear-penalty score is shift-invariant under softmax whenever coverage SATURATES
    to the same constant across every candidate width, which happens in BOTH the confident regime (a
    sharp posterior's mass fits inside every candidate width, coverage saturates near 1) and the
    clueless regime (a posterior far more diffuse than the widest candidate has near-locally-uniform
    density, so coverage saturates near ``density(center) * width`` for every candidate) -- softmax
    over a constant offset cannot tell those two regimes apart. ``coverage / width`` does not saturate
    the same way: in the confident regime it grows as ``1 / width`` (the NARROWEST candidate wins
    decisively), while in the clueless regime ``coverage / width -> density(center)``, the SAME
    value for every candidate width (a uniform, non-committal softmax) -- exactly the "no candidate is
    more informative than any other" signal :meth:`PosteriorDescriber.describe` abstains on.

    Reusable standalone by B2's claim-checking: pass ``posterior=`` to score an independently-authored
    claim against any posterior; a :class:`PosteriorDescriber`-generated claim can instead be
    re-scored with no ``posterior`` argument, reusing the sample batch cached at generation time.
    """
    if posterior is not None:
        values = _sample_scalar(posterior, n_samples, seed)
    elif claim.probe:
        values = np.asarray(claim.probe, dtype=float)
    else:
        raise ValueError("claim_score needs either posterior=... or a claim with cached probe samples")
    coverage = float(np.mean((values >= claim.lo) & (values <= claim.hi)))
    return coverage / max(claim.width, 1e-12)


class PosteriorDescriber:
    """Posterior -> calibrated text for one field, via A1's :class:`CalibratedGenerator`.

    ``tol`` is the caller's required precision (the same "the caller states what precision counts as
    an answer" contract :func:`~mixle.task.regress.solve_regression` uses) -- candidate claim widths
    are ABSOLUTE multiples of ``tol``, not relative to the posterior's own spread, so a genuinely
    diffuse posterior (spread >> ``tol``) cannot fake confidence by simply widening every candidate in
    lockstep: none of them will cover well enough to clear the calibrated threshold, and
    :meth:`describe` abstains (acceptance criterion (d)).
    """

    def __init__(
        self,
        field_name: str,
        *,
        tol: float,
        k: int = 3,
        alpha: float = 0.1,
        width_multiples: tuple[float, ...] = (1.0, 3.0, 10.0),
        n_probe: int = 300,
        seed: int = 0,
    ) -> None:
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if k > len(width_multiples):
            raise ValueError(f"k={k} exceeds the number of configured width_multiples ({len(width_multiples)})")
        self.field_name = field_name
        self.tol = float(tol)
        self.width_multiples = width_multiples[:k]
        self.n_probe = n_probe
        self._gen = CalibratedGenerator(self._generate, self._score, alpha=alpha, k=k, seed=seed)

    def _generate(self, posterior: Any, k: int, rng: Any = None) -> list[Claim]:
        base_seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else None
        center_probe = _sample_scalar(posterior, self.n_probe, base_seed)
        mean = float(np.mean(center_probe))
        claims = []
        for i, mult in enumerate(self.width_multiples):
            half = mult * self.tol
            score_seed = None if base_seed is None else base_seed + i + 1
            probe = _sample_scalar(posterior, self.n_probe, score_seed)
            claims.append(Claim(field=self.field_name, lo=mean - half, hi=mean + half, probe=tuple(probe.tolist())))
        return claims

    def _score(self, claim: Claim) -> float:
        return claim_score(claim)

    def calibrate(self, calibration_set: Sequence[tuple[Any, float]], *, seed: int | None = None) -> PosteriorDescriber:
        """Fit the conformal threshold from ``(posterior, true_value)`` held-out pairs.

        MXR-080-0291: ``is_correct`` is EXACTLY "does the emitted interval cover the truth" --
        ``claim.contains(true_value)``, i.e. ``claim.lo <= true_value <= claim.hi`` -- checked
        independently per candidate, non-strict on both ends. Nothing else. This module previously
        assigned correctness to a single artificial distance BAND per candidate (the true value's
        distance from the shared center had to land in an EXCLUSIVE, disjoint slice ``(prev_half,
        this_half]``), which is a different and unsound event: a truth sitting exactly on the shared
        center (distance 0) failed the narrowest band's strict ``0 < dist`` lower bound and therefore
        failed EVERY band, so the objectively best-supported truth was scored as covered by nothing;
        and a wider candidate that genuinely contains the truth (``dist <= its own half-width``) was
        marked wrong whenever a narrower nested candidate ALSO contained it, purely because the
        narrower band "claimed" that distance range first -- docking a candidate for a coverage fact
        about a DIFFERENT candidate. Both are boundary/exclusivity artifacts of the banding scheme
        itself, not properties of whether the claim actually covers the truth.

        The direct coverage check has no such artifact: every nested candidate whose interval contains
        ``true_value`` counts as correct, independently, exactly matching what :meth:`describe` /
        :func:`claim_score` actually serve. This is the standard split-conformal LAC construction
        :func:`mixle.inference.conformal.conformal_label_threshold` implements (mirrored by A1's
        :class:`~mixle.task.calibrate.CalibratedTaskModel` and, for this codebase's other
        conformal-calibrated surface, :meth:`mixle.reason.model.CrossModalModel.calibrate`,
        MXR-080-0279): calibrate against the actual served event, not a proxy for it.
        """
        calibration_set = list(calibration_set)
        posteriors = [p for p, _ in calibration_set]
        truths = [v for _, v in calibration_set]
        non_finite = [v for v in truths if not np.isfinite(v)]
        if non_finite:
            raise ValueError(f"calibration_set true_value(s) must be finite, got {non_finite!r}")
        k = len(self.width_multiples)  # exactly how many claims _generate() draws per posterior

        # `is_correct` is CalibratedGenerator.calibrate()'s correctness oracle: it is called with only
        # (posterior, claim) -- no row index -- so the matching true_value cannot be recovered by
        # looking the posterior back up (an id()-keyed dict silently collapses when the SAME posterior
        # object is reused across several calibration_set rows with different true values, e.g. one
        # fitted/mock posterior paired with several synthetic points -- a realistic setup, not just a
        # theoretical one). Instead this counts calls: CalibratedGenerator.calibrate() processes rows
        # strictly in order, generating exactly `k` candidates and calling `is_correct` exactly `k`
        # times for row i before ever moving to row i+1, so `calls // k` is that row's index --
        # correct however many times a posterior object repeats, adjacently or not.
        calls = {"n": 0}

        def is_correct(posterior: Any, claim: Claim) -> bool:
            true_value = truths[calls["n"] // k]
            calls["n"] += 1
            return claim.contains(true_value)

        self._gen.calibrate(posteriors, is_correct, seed=seed)
        return self

    def describe(self, posterior: Any, *, seed: int | None = None) -> Claim | None:
        """The best calibrated claim about ``posterior``, or :data:`ABSTAIN` (``None``) when no
        candidate width conformally clears the threshold -- i.e. the posterior is too diffuse relative
        to ``tol`` for any of this describer's claims to be trustworthy."""
        return self._gen.serve(posterior, seed=seed)
