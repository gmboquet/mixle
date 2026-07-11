"""Regression tests for the 2026-06 API-consistency fixes (audit/CONSISTENCY_AUDIT.md §A, D1).

Each test pins a contract-conformance fix so it cannot silently regress. Names carry the finding id.
"""

import numpy as np
import pytest


def _merge_key(dist):
    """Read the merge key regardless of the `.key`/`.keys` attribute-name drift (normalized in §C)."""
    val = getattr(dist, "keys", None)
    return val if val is not None else getattr(dist, "key", None)


# --------------------------------------------------------------------------- A1
def test_a1_power_law_hawkes_encoder_is_abc_and_equal():
    from mixle.stats.compute.pdist import DataSequenceEncoder
    from mixle.stats.processes.power_law_hawkes import PowerLawHawkesDataEncoder

    e1, e2 = PowerLawHawkesDataEncoder(), PowerLawHawkesDataEncoder()
    assert isinstance(e1, DataSequenceEncoder)  # was a bare object -> broke encoder interchange
    assert e1 == e2  # __eq__ added; two encoders must compare equal for batching


def test_a1_power_law_hawkes_accumulator_is_abc():
    from mixle.stats.compute.pdist import SequenceEncodableStatisticAccumulator
    from mixle.stats.processes.power_law_hawkes import (
        PowerLawHawkesAccumulatorFactory,
        PowerLawHawkesEstimator,
    )

    acc = PowerLawHawkesEstimator(window=10.0).accumulator_factory().make()
    assert isinstance(acc, SequenceEncodableStatisticAccumulator)
    assert hasattr(acc, "key_merge") and hasattr(acc, "scale")
    # the factory is now a module-level class, not a single-use nested one
    assert PowerLawHawkesAccumulatorFactory is not None


# --------------------------------------------------------------------------- A2
def test_a2_seq_ld_lambda_default_returns_list():
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    # The base default used to `pass` (return None); sequence.py calls `.extend()` on it.
    rv = GaussianDistribution(0.0, 1.0).seq_ld_lambda()
    assert isinstance(rv, list) and len(rv) >= 1


# --------------------------------------------------------------------------- A3
def test_a3_vmf_estimate_preserves_keys():
    from mixle.stats.directional.von_mises_fisher import VonMisesFisherEstimator

    est = VonMisesFisherEstimator(keys="grp")
    acc = est.accumulator_factory().make()
    rng = np.random.RandomState(0)
    x = rng.randn(200, 3)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    acc.seq_update(x, np.ones(len(x)), None)
    d = est.estimate(float(len(x)), acc.value())
    assert _merge_key(d) == "grp"  # was silently dropped


# --------------------------------------------------------------------------- A4
def test_a4_multivariate_gaussian_estimate_preserves_keys():
    from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianEstimator

    est = MultivariateGaussianEstimator(dim=2, keys="grp")
    acc = est.accumulator_factory().make()
    x = np.random.RandomState(1).randn(80, 2)
    acc.seq_update(x, np.ones(len(x)), None)
    d = est.estimate(float(len(x)), acc.value())
    assert _merge_key(d) == "grp"  # was silently dropped


def test_a4_bernoulli_set_estimate_preserves_keys():
    from mixle.stats.sets.bernoulli_set import BernoulliSetEstimator

    est = BernoulliSetEstimator(keys="grp")
    acc = est.accumulator_factory().make()
    data = [{"a", "b"}, {"a"}, {"b", "c"}, set()]
    for s in data:
        acc.update(s, 1.0, None)
    d = est.estimate(float(len(data)), acc.value())
    assert _merge_key(d) == "grp"


# --------------------------------------------------------------------------- D1
def test_d1_all_engines_provide_required_ops():
    from mixle.engines import NUMPY_ENGINE
    from mixle.engines.base import ComputeEngine

    engines = [NUMPY_ENGINE]
    try:
        from mixle.engines import SYMBOLIC_ENGINE

        engines.append(SYMBOLIC_ENGINE)
    except Exception:  # noqa: BLE001
        pass
    try:
        from mixle.engines.torch_engine import TorchEngine

        engines.append(TorchEngine())
    except Exception:  # noqa: BLE001
        pass
    for eng in engines:
        missing = [op for op in ComputeEngine.REQUIRED_OPS if getattr(eng, op, None) is None]
        assert not missing, f"{type(eng).__name__} missing ops: {missing}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
