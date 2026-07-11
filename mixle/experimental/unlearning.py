"""P5 (experimental) -- exact machine unlearning for closed-form leaves, with a certificate.

Additive sufficient statistics make deletion algebraic. To unlearn a shard, you do NOT subtract
its stored statistics from the total: floating-point addition is not associative, so
``T_all - T_j`` is not the retained reduction and can even produce an invalid parameter (a
negative variance). The exact operation is to **re-reduce the retained shards' stored statistics
in canonical order and re-estimate** -- the result is the model you would have fit had the shard
never been seen, bit-for-bit, for closed-form families.

Because mixle accumulators are additive (``combine`` folds one shard's ``value()`` into another),
this is a few lines over the existing machinery, and mixle's deterministic fit turns the claim
into a *certificate*: :func:`certify_unlearning` re-reduces the retained stored statistics and
checks the result equals the from-scratch retained reduction exactly.

Scope: exact for closed-form (single-M-step) leaves. Iterative-EM latent models are out of scope
for the exact certificate -- stored E-step statistics were computed under the pre-deletion
parameter trajectory, so re-reducing them is only a warm start, not an exact refit (see the P5
card and ``experiments/unlearning_certificate/``). This module certifies the closed-form case and
refuses to over-claim the latent one.

Exploratory ``mixle.experimental`` code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StoredShard:
    """One shard's deletable sufficient statistic: its observation count and accumulator value."""

    n: float
    value: Any


@dataclass
class UnlearningCertificate:
    """Certificate that unlearning ``exclude`` equals the never-saw-it retained reduction."""

    bitwise_exact: bool
    method: str
    n_excluded: int
    n_retained_shards: int
    n_shards_total: int
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bitwise_exact": self.bitwise_exact,
            "method": self.method,
            "n_excluded": self.n_excluded,
            "n_retained_shards": self.n_retained_shards,
            "n_shards_total": self.n_shards_total,
            "note": self.note,
        }


def _encoder(estimator: Any, model: Any = None) -> Any:
    if model is not None:
        return model.dist_to_encoder()
    return estimator.accumulator_factory().make().acc_to_encoder()


def shard_statistic(estimator: Any, shard: Any, *, model: Any = None) -> StoredShard:
    """Compute one shard's stored sufficient statistic via the estimator's accumulator."""
    rows = list(shard)
    enc = _encoder(estimator, model)
    acc = estimator.accumulator_factory().make()
    acc.seq_update(enc.seq_encode(rows), np.ones(len(rows)), model)
    return StoredShard(n=float(len(rows)), value=acc.value())


def _reduce(estimator: Any, stored: list[StoredShard], indices: list[int]) -> tuple[float, Any]:
    """Fold the given shards' stored values in ascending (canonical) index order."""
    acc = estimator.accumulator_factory().make()
    nobs = 0.0
    for i in sorted(indices):  # canonical order is the contract that makes the reduction exact
        acc.combine(stored[i].value)
        nobs += stored[i].n
    return nobs, acc.value()


def unlearn(estimator: Any, stored: list[StoredShard], *, exclude: Any) -> Any:
    """Exact unlearn: re-reduce the retained stored statistics in canonical order, then estimate."""
    excl = set(exclude)
    keep = [i for i in range(len(stored)) if i not in excl]
    nobs, value = _reduce(estimator, stored, keep)
    return estimator.estimate(nobs, value)


def _exact_equal(a: Any, b: Any) -> bool:
    """Recursive bitwise equality over the nested arrays/tuples/dicts an accumulator value holds."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa, bb = np.asarray(a), np.asarray(b)
        return aa.shape == bb.shape and bool(np.array_equal(aa, bb))
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_exact_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_exact_equal(a[k], b[k]) for k in a)
    return bool(a == b)


def certify_unlearning(
    estimator: Any, shards: list[Any], *, exclude: Any, model: Any = None
) -> tuple[Any, UnlearningCertificate]:
    """Unlearn ``exclude`` and certify the result equals the from-scratch retained reduction.

    Returns ``(unlearned_model, certificate)``. ``bitwise_exact`` is True iff re-reducing the
    retained shards' stored statistics yields exactly the same reduced statistic (and estimated
    model) as freshly re-encoding and reducing the retained shards -- i.e. the excluded shard
    leaves no residue at all.
    """
    excl = set(exclude)
    keep = [i for i in range(len(shards)) if i not in excl]

    stored = [shard_statistic(estimator, s, model=model) for s in shards]
    nobs_u, value_u = _reduce(estimator, stored, keep)
    unlearned = estimator.estimate(nobs_u, value_u)

    # From-scratch reference: re-encode only the retained shards, reduce in the same canonical order.
    fresh = [shard_statistic(estimator, shards[i], model=model) for i in keep]
    nobs_s, value_s = _reduce(estimator, fresh, list(range(len(fresh))))
    scratch = estimator.estimate(nobs_s, value_s)

    bitwise = _exact_equal(value_u, value_s) and _model_state(unlearned) == _model_state(scratch)
    note = (
        "closed-form re-reduce: excluded shard leaves no residue; identical to the never-saw-it fit"
        if bitwise
        else "not bitwise -- estimator may be iterative (latent); the exact certificate needs a "
        "closed-form leaf and a fixed canonical reduction order"
    )
    cert = UnlearningCertificate(
        bitwise_exact=bitwise,
        method="re-reduce",
        n_excluded=len(excl),
        n_retained_shards=len(keep),
        n_shards_total=len(shards),
        note=note,
    )
    return unlearned, cert


def _model_state(model: Any) -> str:
    """A full-precision serialization used for the bitwise model comparison."""
    if hasattr(model, "to_json"):
        raw = model.to_json()
        return raw if isinstance(raw, str) else repr(raw)
    return repr(model)
