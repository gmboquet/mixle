"""Application harnesses -- named, one-call replacements for the rigid-code shapes everyone has.

Each harness takes the code currently doing the job (the **teacher**) plus example data, and returns a
deployable object with the same honesty contract as :func:`~mixle.task.solve.solve`: answer locally when
confident, fall back to the teacher when not, report measured numbers.

  * :func:`replace_extractor` -- the regex/parser scraper: ``teacher(text) -> {field: value}``. Distills a
    token-level tagger (:func:`~mixle.task.extract.distill_extractor`); a prediction missing required
    fields falls back to the teacher.
  * :func:`replace_alerter` -- the threshold rule over a sliding window: ``teacher(window) -> label``.
    Windows the series and runs the full ``solve()`` loop (conformal + OOD gate) over window-records.
  * :func:`replace_matcher` -- the dedup/matching rule: ``teacher(a, b) -> label``. Encodes each pair as
    one record (both sides + numeric difference features) and runs ``solve()``.

Alerter and matcher return a real :class:`~mixle.task.solve.Solution` (deploy/serve/improve all work);
the extractor returns an :class:`ExtractorHarness` with the same call-or-fallback behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.task.extract import distill_extractor, extraction_f1
from mixle.task.solve import Solution, solve


# --- extractor -----------------------------------------------------------------------------------------
@dataclass
class ExtractorHarness:
    """A distilled extractor in front of the parser it replaces: local extraction or teacher fallback."""

    model: Any
    teacher: Callable[[str], dict]
    fields: list[str]
    required: list[str]
    holdout_f1: float
    n_fallback: int = 0
    n_requests: int = 0

    def __call__(self, text: str) -> dict:
        self.n_requests += 1
        out = self.model(text)
        if all(out.get(f) for f in self.required):
            return out
        self.n_fallback += 1
        return self.teacher(text)

    def report(self) -> dict[str, Any]:
        """Return extraction holdout quality and fallback metrics."""
        return {
            "holdout_f1": round(self.holdout_f1, 4),
            "requests": self.n_requests,
            "fallbacks": self.n_fallback,
            "fallback_rate": (self.n_fallback / self.n_requests) if self.n_requests else 0.0,
        }

    def save(self, path: str) -> str:
        """Persist the wrapped extraction model artifact."""
        return self.model.save(path)


def replace_extractor(
    teacher: Callable[[str], dict],
    texts: Sequence[str],
    fields: Sequence[str],
    *,
    required: Sequence[str] | None = None,
    holdout: float = 0.25,
    seed: int = 0,
    **distill_kw: Any,
) -> ExtractorHarness:
    """Replace a regex/parser scraper with a distilled token-level extractor + teacher fallback.

    The teacher labels the training texts; a held-out slice measures field-level F1 against the teacher.
    At call time a prediction missing any ``required`` field (default: all ``fields``) falls back to the
    teacher — the same never-silently-wrong shape as ``solve()``.
    """
    items = [str(t) for t in texts]
    if len(items) < 8:
        raise ValueError("replace_extractor needs at least 8 example texts")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_hold = max(2, int(round(len(items) * holdout)))
    hold = [items[i] for i in order[:n_hold]]
    train = [items[i] for i in order[n_hold:]]

    def batch_safe(arg):  # a rigid per-text parser raises when probed with the whole list
        if isinstance(arg, list):
            return [teacher(t) for t in arg]
        return teacher(arg)

    model = distill_extractor(batch_safe, train, list(fields), seed=seed, **distill_kw)
    gold = [teacher(t) for t in hold]
    f1 = extraction_f1(model, gold, hold)
    return ExtractorHarness(
        model=model,
        teacher=teacher,
        fields=list(fields),
        required=list(required) if required is not None else list(fields),
        holdout_f1=float(f1),
    )


# --- alerter -------------------------------------------------------------------------------------------
def _windows(series: Sequence[float], window: int) -> list[tuple]:
    xs = [float(v) for v in series]
    return [tuple(xs[i : i + window]) for i in range(len(xs) - window + 1)]


def replace_alerter(
    teacher: Callable[[Sequence[float]], Any],
    series: Sequence[float],
    *,
    window: int = 16,
    stride: int = 1,
    **solve_kw: Any,
) -> Solution:
    """Replace a threshold/heuristic alert rule over a sliding window with a calibrated model.

    ``teacher(window) -> label`` (e.g. ``"alert"``/``"ok"``) labels every window of the historical
    series; the returned :class:`Solution` is called with a window (the latest ``window`` samples) and
    answers locally only when conformally confident and in-distribution — otherwise it runs the rule.
    """
    wins = _windows(series, int(window))[:: max(1, int(stride))]
    if len(wins) < 8:
        raise ValueError("replace_alerter needs a series long enough for at least 8 windows")
    return solve(lambda w: teacher(list(w)), wins, **solve_kw)


# --- matcher -------------------------------------------------------------------------------------------
def _pair_record(a: Any, b: Any) -> dict:
    """One record from a pair: both sides prefixed, plus |difference| features for shared numeric keys."""
    da = a if isinstance(a, dict) else {f"f{i}": v for i, v in enumerate(a)}
    db = b if isinstance(b, dict) else {f"f{i}": v for i, v in enumerate(b)}
    rec: dict[str, Any] = {}
    for k, v in da.items():
        rec[f"a_{k}"] = v
    for k, v in db.items():
        rec[f"b_{k}"] = v
    for k in set(da) & set(db):
        va, vb = da[k], db[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            rec[f"d_{k}"] = abs(float(va) - float(vb))
        else:
            rec[f"same_{k}"] = str(va) == str(vb)
    return rec


def replace_matcher(
    teacher: Callable[[Any, Any], Any],
    pairs: Sequence[tuple[Any, Any]],
    **solve_kw: Any,
) -> MatcherHarness:
    """Replace a record-matching/dedup rule with a calibrated model over encoded pairs.

    ``teacher(a, b) -> label`` (e.g. ``"match"``/``"no-match"``) labels the example pairs; each pair is
    encoded as one record (both sides plus numeric-difference and same-value features, which is where
    matchers earn their keep). Call the result with ``(a, b)``: confident pairs answer locally,
    everything else runs the rule.
    """
    ps = list(pairs)
    if len(ps) < 8:
        raise ValueError("replace_matcher needs at least 8 example pairs")
    records = [_pair_record(a, b) for a, b in ps]
    labels = [str(teacher(a, b)) for a, b in ps]
    table = {repr(r): y for r, y in zip(records, labels)}
    sol = solve(lambda rec: table[repr(rec)], records, **solve_kw)
    return MatcherHarness(solution=sol, teacher=teacher)


@dataclass
class MatcherHarness:
    """A calibrated pair-matcher in front of the rule it replaces."""

    solution: Solution
    teacher: Callable[[Any, Any], Any]

    def __call__(self, a: Any, b: Any) -> Any:
        rec = _pair_record(a, b)
        local = self.solution.cascade.model.decide(rec)
        if local is not None:
            self.solution.cascade.stats.n_requests += 1
            return local
        self.solution.cascade.stats.n_requests += 1
        self.solution.cascade.stats.n_escalated += 1
        return self.teacher(a, b)

    @property
    def holdout_agreement(self) -> float:
        """Return held-out agreement of the pairwise matcher solution."""
        return self.solution.holdout_agreement

    def report(self) -> dict[str, Any]:
        """Return the underlying matcher solution report."""
        return self.solution.report()
