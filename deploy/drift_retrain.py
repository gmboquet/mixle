"""Drift-triggered retrain-and-swap (run by the Kubernetes CronJob).

Loads the current production model + its drift reference, evaluates drift on a recent production batch, and
if drift is detected retrains a fresh model (with new provenance), registers it, and promotes it to
``production``. The serving Deployment then picks the new model up on its next ``POST /reload`` or a rolling
restart (``kubectl rollout restart deployment/pysp-model``).

``_recent_batch()`` is a stub: wire it to your real production-data store (warehouse, log sink, the
serving activity log, ...). The estimator here is a Gaussian to match seed_registry.py -- swap in yours.
"""

from __future__ import annotations

import json
import os

import numpy as np

from pysp.inference import ModelRegistry, detect_drift, fit_with_provenance
from pysp.stats import GaussianDistribution

ROOT = os.environ.get("PYSP_REGISTRY_ROOT", "./models")
NAME = os.environ.get("PYSP_MODEL_NAME", "model")


def _recent_batch() -> list:
    """STUB: return the recent production records to test for drift. Replace with a real data pull."""
    path = os.environ.get("PYSP_RECENT_BATCH_PATH")
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    # demo fallback: a shifted sample so the example actually triggers a retrain
    return np.random.RandomState(1).normal(6.0, 2.0, 1000).tolist()


def main() -> None:
    registry = ModelRegistry(ROOT)
    model, _header = registry.current(NAME, "production")
    ref_path = os.path.join(ROOT, NAME, "reference.json")
    with open(ref_path) as fh:
        reference = json.load(fh)

    current = _recent_batch()
    report = detect_drift(model, reference, current)
    print(report)

    if not report.drift:
        print("no drift -- keeping the current production model")
        return

    train = reference + current  # retrain on reference + the new regime
    new_model, header = fit_with_provenance(train, GaussianDistribution(0.0, 1.0).estimator(), max_its=50)
    version = registry.register(new_model, NAME)
    registry.promote(NAME, version, alias="production")
    with open(ref_path, "w") as fh:
        json.dump(train, fh)
    print(f"drift detected -> retrained, registered {version}, promoted to production")
    print(header)


if __name__ == "__main__":
    main()
