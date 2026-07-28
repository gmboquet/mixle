"""Tests for the two bridge-stack primitives: ops.product_of_experts and inference.select_best."""

import numpy as np
import pytest

from mixle import capability as cap
from mixle import ops
from mixle.inference import select_best
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


# --------------------------------------------------------------- product_of_experts: categorical
def test_poe_categorical_equals_hand_computed_normalized_product():
    a = CategoricalDistribution({"x": 0.5, "y": 0.3, "z": 0.2})
    b = CategoricalDistribution({"x": 0.1, "y": 0.6, "z": 0.3})
    poe = ops.product_of_experts([a, b])

    # hand: unnormalized product (w=1 each): x=0.05, y=0.18, z=0.06; Z=0.29
    unnorm = {"x": 0.5 * 0.1, "y": 0.3 * 0.6, "z": 0.2 * 0.3}
    total = sum(unnorm.values())
    hand = {k: v / total for k, v in unnorm.items()}

    assert isinstance(poe, CategoricalDistribution)
    assert set(poe.pmap) == set(hand)
    for k in hand:
        assert poe.pmap[k] == pytest.approx(hand[k], abs=1e-12)
    assert sum(poe.pmap.values()) == pytest.approx(1.0, abs=1e-12)
    # log_density matches the hand-computed value exactly
    assert poe.log_density("y") == pytest.approx(np.log(hand["y"]), abs=1e-12)


def test_poe_categorical_weights_are_exponents():
    a = CategoricalDistribution({"x": 0.5, "y": 0.5})
    b = CategoricalDistribution({"x": 0.8, "y": 0.2})
    # w = (2, 1): unnorm[x] = 0.5**2 * 0.8, unnorm[y] = 0.5**2 * 0.2
    poe = ops.product_of_experts([a, b], weights=[2.0, 1.0])
    unnorm = {"x": (0.5**2) * 0.8, "y": (0.5**2) * 0.2}
    total = sum(unnorm.values())
    assert poe.pmap["x"] == pytest.approx(unnorm["x"] / total, abs=1e-12)
    assert poe.pmap["y"] == pytest.approx(unnorm["y"] / total, abs=1e-12)


def test_poe_categorical_drops_labels_outside_the_support_intersection():
    a = CategoricalDistribution({"x": 0.6, "y": 0.4})
    b = CategoricalDistribution({"y": 0.7, "z": 0.3})
    poe = ops.product_of_experts([a, b])
    # only "y" has positive mass under both experts -> degenerate at "y"
    assert set(poe.pmap) == {"y"}
    assert poe.pmap["y"] == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------- product_of_experts: gaussian
def test_poe_gaussian_equals_precision_weighted_formula():
    g1 = GaussianDistribution(0.0, 1.0)
    g2 = GaussianDistribution(2.0, 4.0)
    poe = ops.product_of_experts([g1, g2])

    precision = 1.0 / 1.0 + 1.0 / 4.0
    sigma2 = 1.0 / precision
    mu = sigma2 * (0.0 / 1.0 + 2.0 / 4.0)

    assert isinstance(poe, GaussianDistribution)
    assert poe.mu == pytest.approx(mu, abs=1e-12)
    assert poe.sigma2 == pytest.approx(sigma2, abs=1e-12)


def test_poe_gaussian_weighted_precision():
    g1 = GaussianDistribution(1.0, 2.0)
    g2 = GaussianDistribution(-1.0, 0.5)
    w = [3.0, 1.0]
    poe = ops.product_of_experts([g1, g2], weights=w)

    precision = w[0] / 2.0 + w[1] / 0.5
    sigma2 = 1.0 / precision
    mu = sigma2 * (w[0] * 1.0 / 2.0 + w[1] * (-1.0) / 0.5)
    assert poe.mu == pytest.approx(mu, abs=1e-12)
    assert poe.sigma2 == pytest.approx(sigma2, abs=1e-12)


def test_poe_default_weights_are_unit_not_normalized():
    # PoE of identical Gaussians with default weights = a sharper Gaussian (precision adds), not the
    # same Gaussian -- this distinguishes raw unit exponents from weights normalized to sum to 1.
    g = GaussianDistribution(0.0, 1.0)
    poe = ops.product_of_experts([g, g])
    assert poe.mu == pytest.approx(0.0, abs=1e-12)
    assert poe.sigma2 == pytest.approx(0.5, abs=1e-12)  # 1/(1+1)


# --------------------------------------------------------------- product_of_experts: intractable cases
def test_poe_raises_for_intractable_mixed_continuous_pair():
    from mixle.stats.univariate.continuous.exponential import ExponentialDistribution

    with pytest.raises(cap.CapabilityError):
        ops.product_of_experts([GaussianDistribution(0.0, 1.0), ExponentialDistribution(1.0)])


def test_poe_raises_for_disjoint_categorical_supports():
    a = CategoricalDistribution({"x": 0.5, "y": 0.5})
    b = CategoricalDistribution({"p": 0.5, "q": 0.5})
    with pytest.raises(cap.CapabilityError):
        ops.product_of_experts([a, b])


def test_poe_raises_for_categorical_with_nonzero_default():
    a = CategoricalDistribution({"x": 0.5, "y": 0.5}, default_value=0.1)
    b = CategoricalDistribution({"x": 0.5, "y": 0.5})
    with pytest.raises(cap.CapabilityError):
        ops.product_of_experts([a, b])


# --------------------------------------------------------------- select_best
def test_select_best_picks_max_by_score():
    cands = ["a", "bb", "cccc", "dd"]
    r = select_best(cands, score=len)
    assert r.best == "cccc"
    assert r.best_index == 2
    assert list(r.scores) == [1.0, 2.0, 4.0, 2.0]


def test_select_best_picks_min_when_lower_is_better():
    cands = ["a", "bb", "cccc", "dd"]
    r = select_best(cands, score=len, lower_is_better=True)
    assert r.best == "a"
    assert r.best_index == 0


def test_select_best_is_subscriptable_like_a_dict():
    r = select_best([3, 1, 2], score=float)
    assert r["best"] == 3
    assert r["best_index"] == 0
    assert r["confident"] is None  # no conformal_alpha given


def test_select_best_confident_for_a_clear_winner():
    # one candidate dominates by a wide margin relative to the score spread
    cands = ["a", "b", "c", "winner"]
    scores = {"a": 0.1, "b": 0.15, "c": 0.12, "winner": 5.0}
    r = select_best(cands, score=lambda c: scores[c], conformal_alpha=0.1)
    assert r.best == "winner"
    assert r.confident is True


def test_select_best_not_confident_for_a_near_tie():
    cands = ["a", "b", "c", "d"]
    scores = {"a": 1.00, "b": 1.01, "c": 0.99, "d": 0.98}
    r = select_best(cands, score=lambda c: scores[c], conformal_alpha=0.1)
    assert r.best == "b"
    assert r.confident is False


def test_select_best_rejects_empty_candidates():
    with pytest.raises(ValueError):
        select_best([], score=len)


def test_select_best_rejects_bad_alpha():
    with pytest.raises(ValueError):
        select_best([1, 2], score=float, conformal_alpha=1.5)
