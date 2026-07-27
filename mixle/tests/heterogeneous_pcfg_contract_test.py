"""Termination, statistic, pruning, and ownership contracts for heterogeneous PCFGs."""

import numpy as np
import pytest

from mixle.stats import CategoricalDistribution
from mixle.stats.latent.heterogeneous_pcfg import (
    HeterogeneousPCFGAccumulator,
    HeterogeneousPCFGDistribution,
    HeterogeneousPCFGEstimator,
    InducedHeterogeneousPCFGEstimator,
)
from mixle.utils.vector import ImpossibleEvidenceError


def _point_mass():
    return CategoricalDistribution({"x": 1.0})


def _terminal_model():
    return HeterogeneousPCFGDistribution(
        binary_rules=None,
        terminal_rules=[("S", _point_mass(), 1.0)],
        start="S",
    )


def _terminal_estimator():
    emission = _point_mass()
    return HeterogeneousPCFGEstimator(
        binary_rules=None,
        terminal_rules=[("S", emission.estimator(pseudo_count=1.0), 1.0)],
        start="S",
    )


def test_supercritical_and_nonterminating_spine_grammars_are_rejected():
    emission = _point_mass()
    with pytest.raises(ValueError, match="proper"):
        HeterogeneousPCFGDistribution(
            binary_rules=[("S", "S", "S", 1.0)],
            terminal_rules=[("A", emission, 1.0)],
            start="S",
        )
    with pytest.raises(ValueError, match="infinite derivation spine"):
        HeterogeneousPCFGDistribution(
            binary_rules=[("S", "S", "A", 1.0)],
            terminal_rules=[("A", emission, 1.0)],
            start="S",
        )
    with pytest.raises(ValueError, match="spectral_radius"):
        HeterogeneousPCFGDistribution(
            binary_rules=[("S", "S", "S", 0.6)],
            terminal_rules=[("S", emission, 0.4)],
            start="S",
        )


def test_critical_and_acyclic_grammars_have_termination_certificates():
    emission = _point_mass()
    critical = HeterogeneousPCFGDistribution(
        binary_rules=[("S", "S", "S", 0.5)],
        terminal_rules=[("S", emission, 0.5)],
        start="S",
    )
    assert critical.termination_certificate.proper
    assert critical.termination_certificate.progeny_spectral_radius == pytest.approx(1.0)

    acyclic = HeterogeneousPCFGDistribution(
        binary_rules=[("S", "A", "A", 1.0)],
        terminal_rules=[("A", emission, 1.0)],
        start="S",
    )
    assert acyclic.termination_certificate.proper
    assert acyclic.sampler(2).sample() == ["x", "x"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_nonterminals": 1.5},
        {"terminal_rule_mass": 0.49},
        {"terminal_rule_mass": np.nan},
        {"rule_pseudo_count": -1.0},
        {"prune_threshold": np.inf},
        {"min_rule_prob": 1.0},
    ],
)
def test_induced_estimator_rejects_invalid_controls(kwargs):
    arguments = {
        "max_nonterminals": 2,
        "terminal_estimators": [_point_mass().estimator(pseudo_count=1.0)],
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        InducedHeterogeneousPCFGEstimator(**arguments)


@pytest.mark.parametrize(
    "terminal_counts,binary_counts",
    [
        (np.array([-1.0]), np.empty(0)),
        (np.array([np.nan]), np.empty(0)),
        (np.array([1.0, 2.0]), np.empty(0)),
        (np.array([1.0]), np.array([1.0])),
    ],
)
def test_fixed_estimator_rejects_malformed_rule_statistics(terminal_counts, binary_counts):
    estimator = _terminal_estimator()
    emission_values = estimator.accumulator_factory().make().value()[2]
    with pytest.raises(ValueError):
        estimator.estimate(None, (terminal_counts, binary_counts, emission_values))


def test_accumulator_values_are_owned_and_tied_emissions_are_cloned():
    estimator = _terminal_estimator()
    accumulator = estimator.accumulator_factory().make()
    value = accumulator.value()
    value[0][0] = 100.0
    assert accumulator.terminal_counts[0] == 0.0

    emission = _point_mass()
    factory = emission.estimator(pseudo_count=1.0).accumulator_factory()
    left = HeterogeneousPCFGAccumulator([factory.make()], 0, [0], [], keys=(None, "shared-emission"))
    right = HeterogeneousPCFGAccumulator([factory.make()], 0, [0], [], keys=(None, "shared-emission"))
    pooled = {}
    left.key_merge(pooled)
    right.key_replace(pooled)
    assert left.emission_accumulators[0] is not pooled["shared-emission"][0]
    assert right.emission_accumulators[0] is not pooled["shared-emission"][0]
    assert left.emission_accumulators[0] is not right.emission_accumulators[0]
    before = right.emission_accumulators[0].value()
    left.emission_accumulators[0].update("x", 3.0, emission)
    after = right.emission_accumulators[0].value()
    for old, new in zip(before, after, strict=True):
        np.testing.assert_array_equal(old, new)


def test_impossible_evidence_and_invalid_weights_fail_transactionally():
    model = _terminal_model()
    estimator = _terminal_estimator()
    encoded = model.dist_to_encoder().seq_encode([["x"], ["y"]])
    accumulator = estimator.accumulator_factory().make()
    before = accumulator.value()
    with pytest.raises(ImpossibleEvidenceError):
        accumulator.seq_update(encoded, np.ones(2), model)
    after = accumulator.value()
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])

    with pytest.raises(ValueError):
        accumulator.seq_update(encoded, np.array([1.0, -1.0]), model)
    accumulator.seq_update(encoded, np.array([1.0, 0.0]), model)
    assert accumulator.terminal_counts[0] == pytest.approx(1.0)


def test_pruning_fails_closed_on_empty_parent_and_projects_supercritical_counts():
    emission_estimator = _point_mass().estimator(pseudo_count=1.0)
    estimator = InducedHeterogeneousPCFGEstimator(
        max_nonterminals=1,
        terminal_estimators=[emission_estimator],
        terminal_rule_mass=0.5,
        rule_pseudo_count=0.0,
    )
    value = estimator.accumulator_factory().make().value()
    with pytest.raises(ValueError, match="removed every production"):
        estimator.estimate(None, value)

    terminal_counts = np.ones(estimator._prior.num_terminal_rules)
    binary_counts = np.full(estimator._prior.num_binary_rules, 100.0)
    model = estimator.estimate(
        1.0,
        (terminal_counts, binary_counts, value[2]),
    )
    assert model.termination_certificate.proper
    assert model.binary_probs.sum() == pytest.approx(0.5)
    assert model.terminal_probs.sum() == pytest.approx(0.5)
