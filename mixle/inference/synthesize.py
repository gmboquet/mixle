"""``synthesize()`` -- a dataset factory: sample inputs, label them, keep only what verifies (F2).

The creation verb for *data*. Give it a generative source (a fitted model, or real records it should
learn a generator from) and it produces fresh inputs; give it a ``label`` callable (a teacher) and it
labels them; give it a ``verify`` predicate and it *rejection-samples* -- only rows that pass the
verifier enter the dataset. The verifier travels with the result, so a consumer can re-check every row
(the verified-data-factory discipline: never ship a synthetic row you cannot re-verify).

This is the same loop behind ``mixle.task.solve(synthesize=...)`` promoted to a first-class verb, so it
can feed a fit, an eval set, or an active-acquisition round independent of any one task. Sources:

* a **fitted model** with a ``sampler`` -- sampled directly;
* a **list of real inputs** -- a generative model is inferred over them
  (:func:`mixle.utils.automatic.get_estimator`) and sampled, deduped against the reals;
* a **callable** ``() -> input`` (or ``rng -> input``) -- called per draw (a hand-written simulator).

Everything is optional past the source: no ``label`` gives an unlabeled dataset; no ``verify`` accepts
every draw (acceptance rate 1.0). ``max_tries`` bounds the rejection loop so an impossible verifier
fails loudly with a partial dataset rather than looping forever.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """A verified synthetic dataset: inputs, optional labels, and the verifier that vouched for them."""

    inputs: list[Any]
    labels: list[Any] | None = None
    verify: Callable[..., bool] | None = None
    acceptance_rate: float = 1.0
    n_rejected: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.inputs)

    def __iter__(self):
        if self.labels is None:
            return iter(self.inputs)
        return iter(zip(self.inputs, self.labels))

    def pairs(self) -> list[tuple[Any, Any]]:
        """``(input, label)`` pairs -- raises if the dataset is unlabeled."""
        if self.labels is None:
            raise ValueError("dataset is unlabeled; pass label= to synthesize() to get pairs")
        return list(zip(self.inputs, self.labels))

    def recheck(self) -> bool:
        """Re-run the attached verifier over every row -- the consumer's independent audit.

        Returns True iff every row still passes (or there is no verifier, vacuously). This is the
        point of shipping the verifier with the data: trust is re-derivable, not asserted."""
        if self.verify is None:
            return True
        return all(_check(self.verify, x, y) for x, y in _rows(self.inputs, self.labels))


def _rows(inputs: list, labels: list | None):
    if labels is None:
        for x in inputs:
            yield x, None
    else:
        yield from zip(inputs, labels)


def _check(verify: Callable[..., bool], x: Any, y: Any) -> bool:
    """Call the verifier with whichever arity it wants: ``verify(x)`` or ``verify(x, y)``."""
    try:
        n = len(inspect.signature(verify).parameters)
    except (TypeError, ValueError):
        n = 1
    return bool(verify(x, y) if n >= 2 else verify(x))


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
    """
    real_inputs = source if isinstance(source, (list, tuple)) else None
    max_tries = int(max_tries) if max_tries is not None else max(4 * n, 50)

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
    return Dataset(
        inputs=inputs,
        labels=labels,
        verify=verify,
        acceptance_rate=float(rate),
        n_rejected=rejected,
        provenance={"requested": n, "produced": accepted, "tried": tried, "seed": seed},
    )
