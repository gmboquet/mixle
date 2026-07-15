"""discover_law recovers a validated functional relationship from a black-box simulator, selects by
OUT-OF-SAMPLE fit (so it's a discovered law, not an overfit), and honestly reports failure when no
form generalizes. The out-of-sample selection is the referee -- these tests pin that it's real."""

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


def test_selection_is_by_holdout_not_train_fit():
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


def test_ranking_covers_the_candidate_forms_and_is_ordered_by_holdout():
    law = discover_law(lambda x: 3.0 * x**2, (1.0, 10.0), n_samples=30)
    holdouts = [h for _, h in law.ranking]
    assert holdouts == sorted(holdouts, reverse=True)  # best (referee) first
    assert law.ranking[0][0] == law.form
