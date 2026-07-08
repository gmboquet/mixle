"""Regression tests for two fixed defects.

1. The DPO accumulator dropped the per-pair weight, so weighted EM / mixture responsibilities / streaming decay
   silently trained an unweighted DPO loss. It now carries and applies the weight (buffering lives in the shared
   ``DataBufferAccumulator``, which DPO wires with three encoded fields plus weights).
2. Registry joined a raw name/alias onto the store root, so ``../escape`` traversed outside it. Segments are
   now constrained to a single path component.
"""

import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- DPO weights


def _dpo_accumulator():
    """The accumulator exactly as DPO estimation wires it (generic data buffer over the DPO encoding)."""
    from mixle.models.dpo_leaf import DPOModelEstimator

    est = DPOModelEstimator(policy=None, ref=None, beta=0.1, m_steps=1, lr=1e-3, device="cpu")
    return est.accumulator_factory().make()


def test_dpo_accumulator_carries_the_weight():
    lo, hi = _dpo_accumulator(), _dpo_accumulator()
    lo.update((np.zeros(3), 0, 1), 0.01, None)
    hi.update((np.zeros(3), 0, 1), 100.0, None)
    # same triple, very different weight -> the sufficient statistics must differ now
    assert lo.value()[3].tolist() != hi.value()[3].tolist()


def test_dpo_seq_update_and_combine_preserve_weights():
    enc = (np.zeros((2, 3)), np.array([0, 0]), np.array([1, 1]))
    a = _dpo_accumulator()
    a.seq_update(enc, np.array([2.0, 5.0]), None)
    assert a.value()[3].tolist() == [2.0, 5.0]

    b = _dpo_accumulator().from_value(a.value())  # round-trip the 4-tuple (x, chosen, rejected, w)
    b.combine(a.value())  # distributed merge extends every parallel buffer, weights included
    assert len(b.value()[0]) == 4 and b.value()[3].tolist() == [2.0, 5.0, 2.0, 5.0]


def test_dpo_weighting_changes_the_fitted_policy():
    """Two pairs with opposite preferences: up-weighting one flips which the policy prefers."""
    torch = pytest.importorskip("torch")
    import copy

    import torch.nn as nn

    from mixle.models.dpo_leaf import DPOLeaf, DPOModelEstimator

    torch.manual_seed(0)
    policy = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3))
    ref = copy.deepcopy(policy)
    leaf = DPOLeaf(policy, ref, beta=0.5, m_steps=300, lr=5e-2)

    x = np.array([1.0, 0.0], dtype=float)
    # pair A prefers action 0 over 1; pair B (same context) prefers 1 over 0 -- contradictory
    pairs = [(x, 0, 1), (x, 1, 0)]

    def fit_with(weights):
        est = DPOModelEstimator(copy.deepcopy(policy), copy.deepcopy(ref), 0.5, 300, 5e-2, "cpu")
        acc = est.accumulator_factory().make()
        for pair, w in zip(pairs, weights):
            acc.update(pair, w, None)
        return est.estimate(sum(weights), acc.value())

    prefer_A = fit_with([10.0, 0.1])  # A dominates -> should prefer action 0
    prefer_B = fit_with([0.1, 10.0])  # B dominates -> should prefer action 1
    xb = np.atleast_2d(x)
    assert prefer_A.prefers(xb)[0] != prefer_B.prefers(xb)[0]  # weighting steered the outcome


# --------------------------------------------------------------------------- registry traversal


def test_registry_rejects_traversal_names_and_does_not_escape_root():
    from mixle.inference.production.registry import Registry

    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        for bad in ["../escape-check", "..", "a/b", "/abs", ".", ""]:
            with pytest.raises(ValueError):
                reg._dir(bad)
        assert not os.path.exists(os.path.join(tmp, "escape-check"))  # nothing written outside root


def test_registry_rejects_traversal_through_public_api():
    from mixle.inference.production.registry import Registry

    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        with pytest.raises(ValueError):
            reg.versions("../x")
        with pytest.raises(ValueError):
            reg.get("../x")
        with pytest.raises(ValueError):
            reg.current("m", alias="../a")


def test_registry_normal_names_still_work():
    from mixle.inference.production.registry import Registry

    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        d = reg._dir("good-model.v2")
        assert os.path.isdir(d) and os.path.dirname(d) == os.path.join(tmp, "root")
