"""Trial-manifest and exact-grading contracts for equation discovery."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.equation_discovery import active_experiments, discover, discovery_rate

LINEAR = np.array([0.0, 1.0, 0.0, 0.0])


def test_discovery_rate_materializes_a_one_shot_seed_manifest_once() -> None:
    seeds = (seed for seed in range(4))
    rate = discovery_rate(
        LINEAR,
        strategy="active",
        budget=8,
        radius=2.0,
        noise=0.0,
        threshold=0.1,
        seeds=seeds,
    )
    assert rate == 1.0


def test_coefficient_error_charges_spurious_library_terms() -> None:
    true_coef = np.zeros(4)
    receipt = discover(
        true_coef,
        active_experiments(16, 2.0),
        noise=1.0,
        threshold=0.0,
        seed=2,
    )
    assert receipt.spurious_terms == receipt.recovered_terms
    assert receipt.missing_terms == frozenset()
    assert receipt.coef_error == pytest.approx(float(np.max(np.abs(receipt.recovered_coef))))
    assert receipt.coef_error > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "typo", "budget": 4, "radius": 1.0, "noise": 0.0, "threshold": 0.1},
        {"strategy": "active", "budget": 0, "radius": 1.0, "noise": 0.0, "threshold": 0.1},
        {"strategy": "active", "budget": 4, "radius": 0.0, "noise": 0.0, "threshold": 0.1},
        {"strategy": "active", "budget": 4, "radius": 1.0, "noise": -1.0, "threshold": 0.1},
        {"strategy": "active", "budget": 4, "radius": 1.0, "noise": 0.0, "threshold": -0.1},
    ],
)
def test_discovery_rate_rejects_invalid_benchmark_controls(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        discovery_rate(LINEAR, seeds=[0], **kwargs)


def test_discovery_rate_requires_nonempty_integer_seeds() -> None:
    controls = {
        "strategy": "active",
        "budget": 4,
        "radius": 1.0,
        "noise": 0.0,
        "threshold": 0.1,
    }
    with pytest.raises(ValueError):
        discovery_rate(LINEAR, seeds=[], **controls)
    with pytest.raises(ValueError):
        discovery_rate(LINEAR, seeds=[0, 1.5], **controls)
