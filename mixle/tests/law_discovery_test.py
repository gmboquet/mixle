"""Law discovery selects on validation and confirms once on untouched held-out inputs."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from mixle.experimental.law_discovery import discover_law


def test_recovers_a_power_law_and_validates_out_of_sample():
    # y = 3 * x^2, mild noise. The referee (holdout R^2) must be high.
    rng = np.random.default_rng(0)
    law = discover_law(lambda x: 3.0 * x**2 + rng.normal(0, 0.5), (1.0, 10.0), n_samples=30)
    assert law.form in ("power", "quadratic")  # both express x^2
    assert law.passed
    assert law.holdout_r2 > 0.9
    if law.form == "power":
        assert abs(law.params["b"] - 2.0) < 0.2  # recovered exponent ~2


def test_recovers_a_logarithmic_law():
    rng = np.random.default_rng(1)
    law = discover_law(lambda x: 2.5 * np.log(x) + 1.0 + rng.normal(0, 0.05), (1.0, 50.0), n_samples=30)
    assert law.form == "logarithmic"
    assert law.passed
    assert abs(law.params["a"] - 2.5) < 0.3


def test_selection_is_by_validation_not_train_fit():
    # a straight line: linear must win on held-out data; an over-flexible quadratic may tie on TRAIN
    # but shouldn't beat it out-of-sample. Confirm the winner generalizes.
    law = discover_law(lambda x: 4.0 * x - 7.0, (0.0, 20.0), n_samples=24)
    assert law.holdout_r2 > 0.99
    assert law.form in ("linear", "quadratic")  # both fit a line; either is a valid generalizing law


def test_reports_failure_honestly_when_no_form_generalizes():
    # pure noise -- there is no law. Every form should fail out-of-sample; passed must be False.
    rng = np.random.default_rng(2)
    law = discover_law(lambda x: rng.normal(0, 1), (1.0, 10.0), n_samples=30, min_holdout_r2=0.9)
    assert not law.passed
    assert law.holdout_r2 < 0.9  # honest: no discovery claimed


def test_ranking_covers_the_candidate_forms_and_is_ordered_by_selection():
    law = discover_law(lambda x: 3.0 * x**2, (1.0, 10.0), n_samples=30)
    selection_scores = [score for _, score in law.ranking]
    assert selection_scores == sorted(selection_scores, reverse=True)
    assert law.ranking[0][0] == law.form
    assert law.ranking[0][1] == law.selection_r2


def test_winner_is_confirmed_on_inputs_not_used_for_selection():
    n_samples = 24
    seed = 11
    fraction = 1.0 / 3.0
    xs = np.linspace(0.0, 10.0, n_samples)
    count = int(round(n_samples * fraction))
    order = np.random.RandomState(seed).permutation(n_samples)
    confirmation_x = xs[order[count : 2 * count]]

    def simulator(x):
        offset = 50.0 if np.any(np.isclose(x, confirmation_x)) else 0.0
        return 2.0 * x + 1.0 + offset

    law = discover_law(
        simulator,
        (0.0, 10.0),
        n_samples=n_samples,
        holdout_fraction=fraction,
        forms=("linear",),
        min_holdout_r2=0.9,
        seed=seed,
    )
    assert law.selection_r2 == pytest.approx(1.0)
    assert law.holdout_r2 < 0.9
    assert not law.passed
    assert law.n_selection == count
    assert law.n_holdout == count


@pytest.mark.parametrize("fraction", [0.0, -0.1, 0.5, 1.0, float("nan")])
def test_invalid_holdout_fractions_are_rejected(fraction):
    with pytest.raises(ValueError, match="holdout_fraction"):
        discover_law(lambda x: x, (0.0, 1.0), holdout_fraction=fraction)


def test_invalid_domains_forms_and_simulator_outputs_are_rejected():
    with pytest.raises(ValueError, match="lo < hi"):
        discover_law(lambda x: x, (1.0, 1.0))
    with pytest.raises(ValueError, match="strictly positive"):
        discover_law(lambda x: x, (-1.0, 1.0), log_spaced=True)
    with pytest.raises(ValueError, match="unknown candidate"):
        discover_law(lambda x: x, (0.0, 1.0), forms=("not-a-form",))
    with pytest.raises(ValueError, match="finite scalar"):
        discover_law(lambda _x: float("nan"), (0.0, 1.0))


def test_sample_budget_must_support_three_way_evaluation():
    with pytest.raises(ValueError, match="at least two validation"):
        discover_law(lambda x: x, (0.0, 1.0), n_samples=6, holdout_fraction=0.1)
    with pytest.raises(ValueError, match="too few fitting"):
        discover_law(lambda x: x, (0.0, 1.0), n_samples=7, holdout_fraction=0.3)
