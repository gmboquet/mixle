"""Probability-law contracts for deterministic weighted machines."""

import math

import numpy as np
import pytest

from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
from mixle.stats.latent.hmm_determinize import DeterminizedSequenceDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def test_public_machine_rejects_materially_unnormalized_weights():
    with pytest.raises(ValueError, match="log probability"):
        DeterminizedSequenceDistribution([{}], [{"done": math.log(2.0)}])

    with pytest.raises(ValueError, match="must sum to one"):
        DeterminizedSequenceDistribution(
            [{"continue": (math.log(0.2), 0)}],
            [{"done": math.log(0.2)}],
        )


def test_public_machine_normalizes_only_floating_point_roundoff():
    machine = DeterminizedSequenceDistribution([{}], [{"done": math.log(1.0 + 5.0e-13)}])

    assert machine.log_density(["done"]) == 0.0
    assert machine.density(["done"]) == 1.0
    assert machine.sampler(seed=3).sample(5) == [["done"]] * 5


def test_public_machine_owns_edges_and_scores_the_sampled_law():
    trans = [{"continue": (math.log(0.5), 0)}]
    accept = [{"done": math.log(0.5)}]
    machine = DeterminizedSequenceDistribution(trans, accept)
    trans[0].clear()
    accept[0]["done"] = 0.0

    assert machine.termination_certified
    assert machine.density(["done"]) == pytest.approx(0.5)
    assert machine.density(["continue", "done"]) == pytest.approx(0.25)
    assert all(sample[-1] == "done" for sample in machine.sampler(seed=7).sample(20))


@pytest.mark.parametrize(
    ("trans", "accept", "error"),
    [
        ([], [], ValueError),
        ([{}], [{}, {}], ValueError),
        ([{"x": (0.0, 1)}], [{}], ValueError),
        ([{"x": (0.0, True)}], [{}], TypeError),
        ([{"x": (0.0, 0)}], [{"x": 0.0}], ValueError),
        ([{"x": (np.nan, 0)}], [{}], ValueError),
    ],
)
def test_public_machine_rejects_invalid_geometry_and_edges(trans, accept, error):
    with pytest.raises(error):
        DeterminizedSequenceDistribution(trans, accept)


def test_public_machine_rejects_reachable_nonterminating_class():
    with pytest.raises(ValueError, match="cannot terminate"):
        DeterminizedSequenceDistribution([{"loop": (0.0, 0)}], [{}])


def test_rationalized_hmm_rows_are_normalized_as_one_exact_law():
    source = HiddenMarkovModelDistribution(
        [CategoricalDistribution({"rare": 0.01, "other": 0.25, "done": 0.74})],
        w=[1.0],
        transitions=[[1.0]],
        terminal_values={"done"},
        use_numba=False,
    )

    machine = source.determinize(max_denominator=2)

    assert machine.termination_certified
    assert machine.density(["done"]) == 1.0
    assert machine.sampler(seed=11).sample(5) == [["done"]] * 5


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_states", 0),
        ("max_states", True),
        ("max_denominator", 0),
        ("max_denominator", 2.5),
    ],
)
def test_determinization_controls_are_positive_exact_integers(name, value):
    source = HiddenMarkovModelDistribution(
        [CategoricalDistribution({"done": 1.0})],
        w=[1.0],
        transitions=[[1.0]],
        terminal_values={"done"},
        use_numba=False,
    )
    arguments = {"max_states": 8, "max_denominator": 8}
    arguments[name] = value

    with pytest.raises((TypeError, ValueError)):
        source.determinize(**arguments)


@pytest.mark.parametrize("size", [-1, 1.5, True])
def test_determinized_sampler_validates_size(size):
    sampler = DeterminizedSequenceDistribution([{}], [{"done": 0.0}]).sampler(seed=1)

    with pytest.raises((TypeError, ValueError)):
        sampler.sample(size)
