"""Worklist M11.4 -- verify provenance and replay claims across processes and tampering.

The reproducibility receipt (:mod:`mixle.inference.reproduce`) hashes canonical serialized
content -- training data, seed, estimator, fitted parameters -- so a fit can be replayed
and its content re-checked. M11.4's acceptance is that **tampering and environment drift
are distinguishable in reports**. These tests pin that:

  * a clean replay reproduces (data + params match), including in a *separate process*;
  * altered data, altered parameters, and an altered seed/estimator are each detected;
  * sub-tolerance drift (last-bit platform noise) does NOT flip the fingerprint, while a
    real change above tolerance does -- the exact property that separates drift from
    tampering in a report;
  * a matching fingerprint certifies identity-of-content within tolerance, NOT
    correctness -- a deliberately bad fit still matches its own receipt.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from mixle.inference.estimation import optimize
from mixle.inference.reproduce import (
    ReproReceipt,
    data_fingerprint,
    param_fingerprint,
    record_fit,
    verify_reproducible,
)
from mixle.stats import GaussianEstimator


def _fit(seed_data: int = 0, seed_fit: int = 7, n: int = 200):
    data = [float(x) for x in np.random.RandomState(seed_data).normal(3.0, 2.0, n)]
    model = optimize(data, GaussianEstimator(), out=None, rng=np.random.RandomState(seed_fit))
    return data, model


def test_clean_replay_reproduces() -> None:
    data, model = _fit()
    receipt = record_fit(model, data, seed=7, estimator=GaussianEstimator())
    report = verify_reproducible(GaussianEstimator(), data, receipt, seed=7)
    assert report["reproducible"] is True
    assert report["data_matches"] and report["params_match"]


def test_altered_data_is_detected() -> None:
    data, model = _fit()
    receipt = record_fit(model, data, seed=7, estimator=GaussianEstimator())

    tampered = data[:]
    tampered[0] += 0.5  # a real change, well above rounding tolerance
    assert receipt.matches_data(data) is True
    assert receipt.matches_data(tampered) is False

    report = verify_reproducible(GaussianEstimator(), tampered, receipt, seed=7)
    assert report["data_matches"] is False
    assert report["reproducible"] is False


def test_altered_parameters_are_detected() -> None:
    data, model = _fit()
    receipt = record_fit(model, data, seed=7, estimator=GaussianEstimator())
    assert receipt.matches_model(model) is True

    # A model with different parameters must not match the receipt.
    other, other_model = _fit(seed_data=1)
    assert receipt.matches_model(other_model) is False
    assert param_fingerprint(other_model) != receipt.param_fingerprint


def test_altered_config_changes_the_receipt() -> None:
    """The recorded seed and estimator are part of the recipe, so they show in the report."""
    data, model = _fit()
    receipt = record_fit(model, data, seed=7, estimator=GaussianEstimator())
    assert receipt.seed == 7
    assert receipt.estimator == "GaussianEstimator"

    # A receipt recorded with a different seed is a different recipe.
    other = record_fit(model, data, seed=99, estimator=GaussianEstimator())
    assert other.seed != receipt.seed
    assert other.as_dict()["seed"] == 99


def test_drift_below_tolerance_is_not_tampering() -> None:
    """Sub-rounding drift keeps the fingerprint; a supra-rounding change flips it.

    This is the mechanism that makes environment drift and tampering distinguishable.
    """
    base = [1.2345678901, 2.3456789012, 3.4567890123]
    fp = data_fingerprint(base)

    drift = [base[0] + 1e-12] + base[1:]  # last-bit noise, below the 1e-10 tolerance
    tamper = [base[0] + 1e-6] + base[1:]  # a real edit, above tolerance

    assert data_fingerprint(drift) == fp, "sub-tolerance drift must not read as tampering"
    assert data_fingerprint(tamper) != fp, "a supra-tolerance change must be caught"


def test_matching_fingerprint_does_not_imply_correctness() -> None:
    """A receipt certifies identity of content, not that the model is any good.

    A model fit for far too few iterations still matches its own receipt: the hash says
    'this is the same fitted object', which is a provenance claim, not a quality claim.
    """
    data = [float(x) for x in np.random.RandomState(0).normal(3.0, 2.0, 200)]
    bad = optimize(data, GaussianEstimator(), out=None, max_its=1, rng=np.random.RandomState(7))
    receipt = record_fit(bad, data, seed=7, estimator=GaussianEstimator())
    # The under-fit model matches its own receipt exactly -- identity, not correctness.
    assert receipt.matches_model(bad) is True


def test_cross_process_replay_is_stable() -> None:
    """A receipt recorded here must reproduce byte-for-byte in a separate interpreter."""
    data, model = _fit()
    receipt = record_fit(model, data, seed=7, estimator=GaussianEstimator())

    script = (
        "import numpy as np, json, sys\n"
        "from mixle.inference.estimation import optimize\n"
        "from mixle.inference.reproduce import param_fingerprint, data_fingerprint\n"
        "from mixle.stats import GaussianEstimator\n"
        "data=[float(x) for x in np.random.RandomState(0).normal(3.0,2.0,200)]\n"
        "m=optimize(data, GaussianEstimator(), out=None, rng=np.random.RandomState(7))\n"
        "print(json.dumps({'data': data_fingerprint(data), 'param': param_fingerprint(m)}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["data"] == receipt.data_fingerprint, "data fingerprint drifted across processes"
    assert out["param"] == receipt.param_fingerprint, "param fingerprint drifted across processes"


def test_receipt_roundtrips_through_dict() -> None:
    data, model = _fit()
    receipt = record_fit(model, data, seed=7, estimator=GaussianEstimator())
    d = receipt.as_dict()
    restored = ReproReceipt(**d)
    assert restored == receipt
