"""Reproducibility receipts -- record a fit so it can be replayed and checked bit-for-bit (N2).

A fitted model is only trustworthy if someone else can re-derive it. :func:`record_fit` captures the
minimal recipe -- a fingerprint of the training data, the seed, the estimator, and a fingerprint of the
fitted parameters -- into a :class:`ReproReceipt`. :func:`verify_reproducible` refits from that recipe
and confirms the parameters come out identical: replay-based reproducibility, the same discipline the
certificate applies to *how* a model was estimated, applied to *whether the exact fit can be recovered*.

Fingerprints are canonical: data and parameters are serialized with floats rounded to a fixed precision
before hashing, so last-bit platform noise doesn't make an otherwise-identical fit look different, while
any real change to the data or the fitted parameters flips the hash. A model is fingerprinted through its
own ``to_json`` (its complete state), so this works for any serializable mixle distribution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

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
    """The recipe to re-derive a fit: data + seed + estimator, plus the parameter fingerprint to check."""

    data_fingerprint: str
    n: int
    seed: int
    estimator: str  # type name of the estimator used (documentation; the object is supplied to verify)
    param_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        """Return the receipt as JSON-compatible data."""
        return {
            "data_fingerprint": self.data_fingerprint,
            "n": self.n,
            "seed": self.seed,
            "estimator": self.estimator,
            "param_fingerprint": self.param_fingerprint,
        }

    def matches_data(self, data: Any) -> bool:
        """Whether ``data`` is the exact dataset this fit was recorded on."""
        return data_fingerprint(data) == self.data_fingerprint

    def matches_model(self, model: Any) -> bool:
        """Whether ``model`` has the exact parameters this receipt fingerprinted."""
        return param_fingerprint(model) == self.param_fingerprint


def record_fit(model: Any, data: Any, *, seed: int, estimator: Any = None) -> ReproReceipt:
    """Record a :class:`ReproReceipt` for a model fitted on ``data`` with ``seed`` (see module docstring)."""
    rows = list(data)
    est_name = type(estimator).__name__ if estimator is not None else type(model).__name__
    return ReproReceipt(
        data_fingerprint=data_fingerprint(rows),
        n=len(rows),
        seed=int(seed),
        estimator=est_name,
        param_fingerprint=param_fingerprint(model),
    )


def verify_reproducible(
    estimator: Any, data: Any, receipt: ReproReceipt, *, seed: int | None = None, max_its: int = 25
) -> dict[str, Any]:
    """Refit ``estimator`` on ``data`` and check the fit reproduces ``receipt`` (data + parameters).

    Returns ``{reproducible, data_matches, params_match, refit_fingerprint}``. ``reproducible`` is True
    iff BOTH the data fingerprint and the refit's parameter fingerprint match the receipt -- i.e. the
    exact fit can be recovered from the recorded recipe. ``seed`` defaults to the receipt's seed."""
    import numpy as np

    from mixle.inference.estimation import optimize

    rows = list(data)
    use_seed = receipt.seed if seed is None else int(seed)
    data_matches = data_fingerprint(rows) == receipt.data_fingerprint
    refit = optimize(rows, estimator, out=None, max_its=max_its, rng=np.random.RandomState(use_seed))
    refit_fp = param_fingerprint(refit)
    params_match = refit_fp == receipt.param_fingerprint
    return {
        "reproducible": bool(data_matches and params_match),
        "data_matches": bool(data_matches),
        "params_match": bool(params_match),
        "refit_fingerprint": refit_fp,
    }
