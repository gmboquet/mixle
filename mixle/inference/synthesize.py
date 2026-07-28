"""Verified synthetic dataset creation.

``synthesize`` draws inputs from a generative source, optionally labels them
with a teacher callable, and optionally filters them through a verifier. The
result is a :class:`Dataset` that carries its verifier so consumers can recheck
rows independently.

Supported sources include:

* a fitted model with a ``sampler``;
* a list of real inputs, from which a generator is inferred and sampled;
* a callable ``() -> input`` or ``rng -> input``.

Without ``label`` the result is unlabeled. Without ``verify`` every draw is
accepted. ``max_tries`` bounds rejection sampling so an impossible verifier
raises :class:`IncompleteSynthesisError` instead of looping indefinitely --
a returned :class:`Dataset` therefore always holds exactly the requested number
of verified rows. The partially built dataset travels on the exception's
``dataset`` attribute for callers that want to inspect what did pass.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """A verified synthetic dataset: inputs, optional labels, and the verifier that vouched for them.

    ``inputs`` and ``labels`` stay ordinary mutable lists, but a labeled dataset must always hold exactly
    one label per input. That alignment is checked at construction *and* again before every operation that
    walks the rows, so a post-construction edit that leaves the two out of step is reported instead of
    silently truncating the rows that get iterated, paired, or rechecked.
    """

    inputs: list[Any]
    labels: list[Any] | None = None
    verify: Callable[..., bool] | None = None
    acceptance_rate: float = 1.0
    n_rejected: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.require_aligned()

    def require_aligned(self) -> None:
        """Raise unless every input carries exactly one label (vacuous for an unlabeled dataset)."""
        if self.labels is not None and len(self.labels) != len(self.inputs):
            raise ValueError(
                f"dataset has {len(self.inputs)} inputs but {len(self.labels)} labels; "
                "a labeled dataset must carry exactly one label per input"
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def __iter__(self):
        self.require_aligned()
        if self.labels is None:
            return iter(self.inputs)
        return iter(zip(self.inputs, self.labels, strict=True))

    def pairs(self) -> list[tuple[Any, Any]]:
        """``(input, label)`` pairs -- raises if the dataset is unlabeled or misaligned."""
        if self.labels is None:
            raise ValueError("dataset is unlabeled; pass label= to synthesize() to get pairs")
        self.require_aligned()
        return list(zip(self.inputs, self.labels, strict=True))

    def recheck(self) -> bool:
        """Re-run the attached verifier over every row.

        Returns True when every row still passes, or when no verifier is attached. Raises if the dataset
        is misaligned -- a partial recheck that skipped unlabeled inputs would report a pass it never made.
        """
        self.require_aligned()
        if self.verify is None:
            return True
        return all(_check(self.verify, x, y) for x, y in _rows(self.inputs, self.labels))


class IncompleteSynthesisError(RuntimeError):
    """Synthesis exhausted its attempt budget before producing the requested number of verified rows.

    ``dataset`` holds the rows that did pass, so a caller that genuinely wants a short dataset can take it
    deliberately rather than receiving one that looks like the full request.
    """

    def __init__(self, message: str, dataset: Dataset) -> None:
        super().__init__(message)
        self.dataset = dataset


def _rows(inputs: list, labels: list | None):
    if labels is None:
        for x in inputs:
            yield x, None
    else:
        yield from zip(inputs, labels, strict=True)


def _check(verify: Callable[..., bool], x: Any, y: Any) -> bool:
    """Call the verifier with whichever arity it wants: ``verify(x)`` or ``verify(x, y)``."""
    try:
        n = len(inspect.signature(verify).parameters)
    except (TypeError, ValueError):
        n = 1
    return bool(verify(x, y) if n >= 2 else verify(x))


def _exact_count(value: Any, label: str) -> int:
    """An exact non-negative integer count -- ``bool`` and fractional values are requests, not counts."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an exact integer count, got {value!r}")
    count = int(value)
    if count < 0:
        raise ValueError(f"{label} must be non-negative, got {count}")
    return count


def _draws(source: Any, n: int, real_inputs: list | None, seed: int) -> list:
    """Produce ``n`` candidate inputs from a model / real-input list / callable source."""
    if callable(source) and not hasattr(source, "sampler"):
        rng = np.random.RandomState(seed)
        wants_rng = False
        try:
            wants_rng = len(inspect.signature(source).parameters) >= 1
        except (TypeError, ValueError):
            pass
        return [source(rng) if wants_rng else source() for _ in range(n)]

    if real_inputs is not None:
        from mixle.inference.estimation import optimize
        from mixle.utils.automatic import get_estimator

        gen = optimize(real_inputs, get_estimator(real_inputs), max_its=25, out=None, rng=np.random.RandomState(seed))
        draws = list(gen.sampler(seed=seed).sample(max(n + n // 2, n)))
        seen = {repr(x) for x in real_inputs}
        out: list = []
        for x in draws:
            r = repr(x)
            if r not in seen:
                seen.add(r)
                out.append(x)
            if len(out) >= n:
                break
        return out

    sampler = source.sampler(seed=seed)
    return list(sampler.sample(int(n)))


def synthesize(
    source: Any,
    *,
    label: Callable[[Any], Any] | None = None,
    verify: Callable[..., bool] | None = None,
    n: int = 100,
    max_tries: int | None = None,
    seed: int = 0,
) -> Dataset:
    """Build a verified dataset of ``n`` accepted rows from a generative ``source`` (see module docstring).

    ``source`` is a fitted model (sampled), a list of real inputs (a generator is inferred over them), or
    a callable draw function. ``label`` (optional) is the teacher applied to each input. ``verify``
    (optional) accepts ``verify(x)`` or ``verify(x, label)`` and gates each row -- rejected rows are
    resampled up to ``max_tries`` total draws. The verifier is attached to the returned :class:`Dataset`
    so consumers can :meth:`~Dataset.recheck` independently.

    ``n`` must be an exact non-negative integer (``bool`` is not an integer here). A returned dataset always
    holds exactly ``n`` verified rows; if the attempt budget, an impossible verifier, a short sampler, or
    real-input deduplication leaves it short, :class:`IncompleteSynthesisError` is raised with the partial
    dataset attached rather than a shortfall being returned as an ordinary result.
    """
    n = _exact_count(n, "n")
    real_inputs = source if isinstance(source, (list, tuple)) else None
    max_tries = _exact_count(max_tries, "max_tries") if max_tries is not None else max(4 * n, 50)

    inputs: list[Any] = []
    labels: list[Any] | None = [] if label is not None else None
    tried = 0
    rejected = 0
    round_seed = seed
    while len(inputs) < n and tried < max_tries:
        want = n - len(inputs)
        batch = _draws(source, min(want * 2, max_tries - tried) or 1, real_inputs, round_seed)
        round_seed += 1
        for x in batch:
            tried += 1
            y = label(x) if label is not None else None
            if verify is not None and not _check(verify, x, y):
                rejected += 1
                continue
            inputs.append(x)
            if labels is not None:
                labels.append(y)
            if len(inputs) >= n:
                break
        if not batch:
            break

    accepted = len(inputs)
    rate = accepted / (accepted + rejected) if (accepted + rejected) else 1.0

    # M2 precondition: "more rows like these" assumes the source rows are exchangeable -- when the source
    # is real data, test that and record the verdict with the dataset (a warning, never a refusal).
    exch = None
    if real_inputs is not None:
        try:
            from mixle.data.exchangeability import exchangeability_check

            exch = exchangeability_check(real_inputs, seed=seed).as_dict()
        except Exception:  # noqa: BLE001 - the precondition check must never break synthesis
            exch = None

    dataset = Dataset(
        inputs=inputs,
        labels=labels,
        verify=verify,
        acceptance_rate=float(rate),
        n_rejected=rejected,
        provenance={"requested": n, "produced": accepted, "tried": tried, "seed": seed, "exchangeability": exch},
    )
    if accepted != n:
        raise IncompleteSynthesisError(
            f"synthesize() produced {accepted} of the {n} requested verified rows after {tried} draws "
            f"({rejected} rejected, max_tries={max_tries}); raise max_tries, loosen verify, or take the "
            "partial rows deliberately from this error's .dataset",
            dataset,
        )
    return dataset
