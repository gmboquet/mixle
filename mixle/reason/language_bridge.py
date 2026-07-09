"""The language<->belief bridge (roadmap M5, part (c)): NL/record -> M0 evidence via a declared
schema, and posterior -> calibrated text through A1's :class:`~mixle.task.calibrated_generator.CalibratedGenerator`.

Two independent directions, each a thin composition of already-tested machinery -- no new extraction
or generation model is built here (see ``notes/designs/M5.md`` part (c) for the full contract):

* :func:`parse_evidence` -- an extractor callable (the same ``teacher(x) -> dict`` shape
  :func:`mixle.task.structured_out.solve_structured` decomposes) produces a raw ``{field: value}``
  dict from NL/record input; this module validates it against a caller-declared schema
  (``{field: "categorical" | "numeric"}`` -- the same shape
  :attr:`~mixle.task.structured_out.StructuredSolution.schema` already returns) BEFORE it reaches
  :func:`~mixle.reason.cross_modal.CrossModalJoint.infer` / :func:`~mixle.reason.inference_program.run_inference_program`,
  so a schema violation is a clear ``ValueError`` here rather than a confusing downstream
  ``log_density`` crash.
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

__all__ = [
    "Schema",
    "parse_evidence",
    "Claim",
    "claim_score",
    "PosteriorDescriber",
    "ABSTAIN",
]

Schema = dict[str, str]  # {field_name: "categorical" | "numeric"}
_KINDS = ("categorical", "numeric")


def _validate_evidence(raw: dict[str, Any], schema: Schema) -> dict[str, Any]:
    unknown = [k for k in raw if k not in schema]
    if unknown:
        raise ValueError(f"extractor returned undeclared field(s) {sorted(unknown)!r}; schema is {sorted(schema)!r}")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        kind = schema[key]
        if kind == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"field {key!r} is declared numeric but the extractor returned {value!r}")
            out[key] = float(value)
        elif kind == "categorical":
            out[key] = str(value)
        else:
            raise ValueError(f"schema field {key!r} has unknown kind {kind!r}; expected one of {_KINDS}")
    return out


def parse_evidence(text: Any, schema: Schema, extractor: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
    """NL scenario/constraint (or any record ``extractor`` accepts) -> validated M0/L2 evidence.

    ``extractor(text) -> {field: raw_value}`` does the actual parsing (a keyword/regex rule, a
    calibrated :func:`~mixle.task.structured_out.solve_structured` student, an LLM call -- this module
    is agnostic to how); this function's only job is enforcing the declared ``schema`` BEFORE the
    result is trusted as evidence: every returned field must be declared, numeric fields must actually
    be numbers, categorical fields are normalized to ``str``. The returned dict is ready to pass
    straight to ``CrossModalJoint.infer(...)`` or as ``run_inference_program``'s ``evidence=``.
    """
    if not schema:
        raise ValueError("parse_evidence needs a non-empty schema")
    raw = extractor(text)
    if not isinstance(raw, dict):
        raise TypeError(f"extractor(text) must return a dict, got {type(raw).__name__}")
    return _validate_evidence(raw, schema)


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


def _sample_scalar(posterior: Any, n: int, seed: int | None) -> np.ndarray:
    """Extract ``n`` scalar draws of a single field from any M0/L2/M5 posterior-like object, or pass a
    plain array/sequence of scalars through unchanged. A real, documented scope limit (not silently
    wrong for other shapes): this only handles objects whose draws are already scalars or 1-tuples --
    exactly what a single-field :class:`~mixle.reason.inference_program.ProgramPosterior` or a
    single-target :class:`~mixle.stats.latent.mixture.MixtureDistribution`
    (``CrossModalJoint.infer(..., [field])``) produce."""
    if isinstance(posterior, np.ndarray):
        return posterior.astype(float)
    if isinstance(posterior, Sequence) and posterior and isinstance(posterior[0], (int, float, np.floating)):
        return np.asarray(posterior, dtype=float)
    if hasattr(posterior, "sample"):
        try:
            draws = posterior.sample(n, seed=seed)
        except TypeError:
            draws = posterior.sample(n)
    elif hasattr(posterior, "sampler"):
        draws = posterior.sampler(seed=seed).sample(n)
    else:
        raise TypeError(f"do not know how to sample a field from {type(posterior).__name__}")
    out = []
    for d in draws:
        if isinstance(d, (int, float, np.floating, np.integer)):
            out.append(float(d))
        elif isinstance(d, (tuple, list, np.ndarray)) and len(d) == 1:
            out.append(float(d[0]))
        else:
            raise ValueError(f"expected a scalar or single-field draw, got {d!r}")
    return np.asarray(out, dtype=float)


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

        ``is_correct`` assigns EXCLUSIVE correctness across the ``k`` nested candidate widths --
        the true value's distance from the shared center falls in exactly one of the ``k`` disjoint
        precision BANDS ``(0, tol], (tol, 3*tol], ...`` (widths are nested/overlapping as claims, but
        only the tightest band a true value actually falls in counts as "correct"). Without this, a
        wide claim trivially contains the true value whenever a narrower nested claim does too, so
        every calibration point would credit ALL of them at once and the conformal threshold would
        never learn to prefer -- or reject -- any one candidate.
        """
        posteriors = [p for p, _ in calibration_set]
        truth = {id(p): v for p, v in calibration_set}
        half_widths = sorted(mult * self.tol for mult in self.width_multiples)

        def is_correct(posterior: Any, claim: Claim) -> bool:
            dist = abs(truth[id(posterior)] - 0.5 * (claim.lo + claim.hi))
            half = 0.5 * claim.width
            idx = min(range(len(half_widths)), key=lambda j: abs(half_widths[j] - half))
            band_lo = half_widths[idx - 1] if idx > 0 else 0.0
            return band_lo < dist <= half_widths[idx]

        self._gen.calibrate(posteriors, is_correct, seed=seed)
        return self

    def describe(self, posterior: Any, *, seed: int | None = None) -> Claim | None:
        """The best calibrated claim about ``posterior``, or :data:`ABSTAIN` (``None``) when no
        candidate width conformally clears the threshold -- i.e. the posterior is too diffuse relative
        to ``tol`` for any of this describer's claims to be trustworthy."""
        return self._gen.serve(posterior, seed=seed)
