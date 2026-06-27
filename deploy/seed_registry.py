"""Seed the model registry so the server has something to serve (run once, or as an init Job).

Trains a model with full provenance, registers it, promotes it to the ``production`` alias, and saves the
training sample as the drift reference. Point ``PYSP_REGISTRY_ROOT`` at the same volume the server reads.

This is an EXAMPLE using a Gaussian over synthetic data; swap in your real model/estimator and your real
training data (e.g. loaded via ``pysp.data.open_source('csv'|'parquet'|'sql', ...)``).
"""

from __future__ import annotations

import json
import os

import numpy as np

from pysp.inference import ModelRegistry, fit_with_provenance
from pysp.stats import GaussianDistribution

ROOT = os.environ.get("PYSP_REGISTRY_ROOT", "./models")
NAME = os.environ.get("PYSP_MODEL_NAME", "model")


def main() -> None:
    data = np.random.RandomState(0).normal(3.0, 2.0, 2000).tolist()
    model, header = fit_with_provenance(data, GaussianDistribution(0.0, 1.0).estimator(), max_its=50)

    registry = ModelRegistry(ROOT)
    version = registry.register(model, NAME)
    registry.promote(NAME, version, alias="production")

    # persist the training sample as the drift reference the server / cronjob compare against
    ref_path = os.path.join(ROOT, NAME, "reference.json")
    with open(ref_path, "w") as fh:
        json.dump(data, fh)

    print(f"registered {NAME} {version}, promoted to production, reference -> {ref_path}")
    print(header)


if __name__ == "__main__":
    main()
