"""Campaign-three regressions for the Markov-chain and Bernoulli fit paths.

T1-F3 -- ``optimize()``/``fit()`` seed EM from a hard 0/1 Bernoulli subsample (``init_p``, default
0.1). ``MarkovChainEstimator.estimate0`` used to REFUSE a transition row with no evidence, so a
five-sequence dataset with abundant evidence for every row failed on 10/10 seeds under the defaults,
with an error that blamed the user's data and named remedies (pseudo-counts, priors) that change the
statistical answer. A row no observation ever leaves appears in no likelihood factor, so it is an
exact maximizer rather than an undefined one; it is now filled uniform and DISCLOSED through
``numerical_repairs()``.

T4-04 -- ``BernoulliEstimator.estimate`` clamps an all-success / all-failure fit off the open
interval's boundary, but reported nothing: ``numerical_repairs()`` and ``fit_provenance().repairs``
were both empty while ``optimize()``'s docstring promises repairs are readable there (the Gaussian
variance floor already does this). The clamp is now recorded when, and only when, it binds.

Dependency-light by construction: numpy + mixle only, so the file runs in the core CI lane.
"""

import copy
import io
import pickle
import warnings

import numpy as np

from mixle.inference import fit, optimize
from mixle.stats import (
    BernoulliEstimator,
    GaussianEstimator,
    MarkovChainEstimator,
    MixtureEstimator,
    PoissonEstimator,
)
from mixle.stats.sequences.markov_chain import MarkovChainDistribution

# The tester's dataset. Transition evidence in FULL: row 'a' -> a 4 / b 4, row 'b' -> a 3 / b 2;
# initial states a 3 / b 2. Its exact closed-form MLE log-likelihood is therefore
# 6*ln(3/5) + 4*ln(2/5) + 8*ln(1/2) (initial vector plus row 'b' plus row 'a').
SMALL_SEQUENCES = [list("aab"), list("abbba"), list("ba"), list("aaaabab"), list("b")]
EXACT_MLE_LL = 6.0 * np.log(0.6) + 4.0 * np.log(0.4) + 8.0 * np.log(0.5)


def _data_ll(model, sequences):
    return float(sum(model.log_density(x) for x in sequences))


def _quiet_optimize(*args, **kwargs):
    """``optimize`` with its unconverged-cap note silenced; these fits are about success, not tuning."""
    kwargs.setdefault("out", io.StringIO())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return optimize(*args, **kwargs)


# --------------------------------------------------------------------------------------------
# T1-F3: the default initialization must not starve a closed-form sequence fit.
# --------------------------------------------------------------------------------------------


def test_default_optimize_reaches_the_exact_closed_form_mle_on_small_data():
    """Every seed must fit, and land on the exact MLE -- not merely stop raising.

    Before: ValueError "Markov MLE row np.str_('a') requires transition evidence, pseudo-count
    smoothing, or a prior." on 10/10 RandomState seeds, though the data has evidence for every row.
    """
    for seed in range(10):
        model = _quiet_optimize(SMALL_SEQUENCES, MarkovChainEstimator(), rng=np.random.RandomState(seed))
        assert abs(_data_ll(model, SMALL_SEQUENCES) - EXACT_MLE_LL) <= 1e-12


def test_default_fit_reaches_the_exact_closed_form_mle_on_small_data():
    """``fit()`` (no rng supplied) took the same starved path and failed the same way."""
    model = fit(SMALL_SEQUENCES, MarkovChainEstimator())
    assert abs(_data_ll(model, SMALL_SEQUENCES) - EXACT_MLE_LL) <= 1e-12


def test_default_fit_of_small_data_reports_no_repair():
    """The subsample is a starting point only: the converged fit is untouched, so it discloses nothing.

    This is the negative control for the disclosure below -- a repair channel that fires on ordinary
    fits would be as useless as one that never fires.
    """
    model = _quiet_optimize(SMALL_SEQUENCES, MarkovChainEstimator(), rng=np.random.RandomState(0))
    assert model.numerical_repairs() == ()
    assert model.fit_provenance().repairs == ()


def test_state_with_no_outgoing_transition_is_uniform_and_disclosed():
    """A terminal state is legitimate full-sample input; it was refused even at ``init_p=1.0``."""
    sequences = [list("ab"), list("ba"), list("c")]
    model = _quiet_optimize(sequences, MarkovChainEstimator(), rng=np.random.RandomState(0), init_p=1.0)

    row = model.transition_map["c"]
    assert row == {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0}

    repairs = model.numerical_repairs()
    assert repairs == ("markov-row-uniform(no outgoing transitions: 'c')",)
    # and it reaches the fit receipt, the surface a caller audits.
    assert model.fit_provenance().repairs == repairs


def test_uniform_row_names_the_state_in_plain_python_not_numpy_repr():
    """The old message rendered states as ``np.str_('a')``; the disclosure must name the user's value."""
    model = _quiet_optimize(
        [list("ab"), list("ba"), list("c")], MarkovChainEstimator(), rng=np.random.RandomState(0), init_p=1.0
    )
    (repair,) = model.numerical_repairs()
    assert "np.str_" not in repair


def test_uniform_fill_does_not_change_the_data_log_likelihood():
    """The filled row is an exact maximizer: no observation leaves 'c', so the row is not scored.

    Pins the reason the fill is legitimate rather than a guess -- any other row for 'c' gives the
    identical data log-likelihood.
    """
    sequences = [list("ab"), list("ba"), list("c")]
    fitted = _quiet_optimize(sequences, MarkovChainEstimator(), rng=np.random.RandomState(0), init_p=1.0)
    skewed = MarkovChainDistribution(
        dict(fitted.init_prob_map),
        {
            "a": dict(fitted.transition_map["a"]),
            "b": dict(fitted.transition_map["b"]),
            "c": {"a": 0.9, "b": 0.05, "c": 0.05},
        },
    )
    assert _data_ll(skewed, sequences) == _data_ll(fitted, sequences)


def test_uniform_row_disclosure_survives_pickle_and_deepcopy():
    """A copy of a repaired model must not report itself untouched.

    ``MarkovChainDistribution.__deepcopy__`` rebuilds from the probability tables, and
    ``optimize(track_best=...)`` returns a deepcopy of the best-seen model -- so the repair log has
    to be carried across explicitly or the disclosure silently disappears on that path.
    """
    accumulator = MarkovChainEstimator(levels=("a", "b")).accumulator_factory().make()
    accumulator.update(["a"], 1.0, None)
    model = MarkovChainEstimator(levels=("a", "b")).estimate(1.0, accumulator.value())
    expected = ("markov-row-uniform(no outgoing transitions: 'a', 'b')",)
    assert model.numerical_repairs() == expected
    assert pickle.loads(pickle.dumps(model)).numerical_repairs() == expected
    assert copy.deepcopy(model).numerical_repairs() == expected


def test_deepcopy_of_an_unrepaired_markov_model_stays_silent():
    """Negative control for the copy path: an ordinary model copies to an ordinary model."""
    accumulator = MarkovChainEstimator().accumulator_factory().make()
    for sequence in (["a", "b"], ["b", "a"]):
        accumulator.update(sequence, 1.0, None)
    model = MarkovChainEstimator().estimate(None, accumulator.value())
    assert model.numerical_repairs() == ()
    assert copy.deepcopy(model).numerical_repairs() == ()


def test_mixture_of_markov_leaves_survives_the_default_initialization():
    """Blast radius: a component whose responsibilities emptied a row hit the same refusal."""
    for seed in range(6):
        _quiet_optimize(
            SMALL_SEQUENCES * 8,
            MixtureEstimator([MarkovChainEstimator(len_estimator=PoissonEstimator())] * 3),
            rng=np.random.RandomState(seed),
            max_its=3,
        )


def test_pseudo_count_path_still_smooths_rather_than_filling_uniform():
    """Negative control: ``estimate1``'s smoothing is untouched by the ``estimate0`` change."""
    accumulator = MarkovChainEstimator().accumulator_factory().make()
    for sequence in [list("ab"), list("ba"), list("c")]:
        accumulator.update(sequence, 1.0, None)
    model = MarkovChainEstimator(pseudo_count=3.0).estimate(None, accumulator.value())
    assert model.transition_map["c"] == {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0}
    assert model.numerical_repairs() == ()  # smoothing is the user's declared model, not a repair


def test_empty_initial_evidence_is_still_refused_with_an_accurate_message():
    """The one guard kept: with no observation starting anywhere there is no data to fit at all."""
    accumulator = MarkovChainEstimator(levels=["a", "b"]).accumulator_factory().make()
    accumulator.update(list("ab"), 0.0, None)
    try:
        MarkovChainEstimator(levels=["a", "b"]).estimate(None, accumulator.value())
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("estimate() accepted statistics with no initial-state evidence")
    assert "no observation started" in message
    assert "pseudo_count" in message


# --------------------------------------------------------------------------------------------
# T4-04: the Bernoulli boundary clamp is applied, so it must be readable.
# --------------------------------------------------------------------------------------------


def test_all_success_bernoulli_fit_discloses_the_boundary_clamp():
    """Before: p=0.999999999999 with numerical_repairs() == () and fit_provenance().repairs == ()."""
    model = _quiet_optimize([True] * 50, estimator=BernoulliEstimator(), max_its=3)
    assert model.p < 1.0  # the clamp is still applied; only the silence is fixed
    assert model.numerical_repairs() == ("bernoulli-p-clamped(1 -> 1 - 1e-12)",)
    assert model.fit_provenance().repairs == model.numerical_repairs()


def test_all_failure_bernoulli_fit_discloses_the_boundary_clamp():
    model = _quiet_optimize([False] * 50, estimator=BernoulliEstimator(), max_its=3)
    assert model.p > 0.0
    assert model.numerical_repairs() == ("bernoulli-p-clamped(0 -> 1e-12)",)
    assert model.fit_provenance().repairs == model.numerical_repairs()


def test_interior_bernoulli_fit_reports_no_repair():
    """An ordinary fit whose p never reached the boundary must stay silent."""
    model = _quiet_optimize([True] * 30 + [False] * 20, estimator=BernoulliEstimator(), max_its=3)
    assert model.p == 0.6
    assert model.numerical_repairs() == ()
    assert model.fit_provenance().repairs == ()


def test_bernoulli_estimate_records_the_clamp_without_going_through_optimize():
    """The repair belongs to the estimator, not to the fit loop -- pin it at the estimate() call."""
    estimator = BernoulliEstimator()
    clamped = estimator.estimate(None, (50.0, 50.0))
    assert clamped.numerical_repairs() == ("bernoulli-p-clamped(1 -> 1 - 1e-12)",)
    interior = estimator.estimate(None, (50.0, 30.0))
    assert interior.numerical_repairs() == ()
    empty = estimator.estimate(None, (0.0, 0.0))  # count == 0 short-circuits to p=0.5, no clamp
    assert empty.p == 0.5
    assert empty.numerical_repairs() == ()


def test_pseudo_count_keeps_a_boundary_sample_off_the_clamp():
    """Smoothing moves p into the interior, so nothing is repaired and nothing is reported."""
    model = BernoulliEstimator(pseudo_count=2.0).estimate(None, (50.0, 50.0))
    assert 0.0 < model.p < 1.0 - 1.0e-12
    assert model.numerical_repairs() == ()


def test_bernoulli_disclosure_matches_the_gaussian_floor_it_mirrors():
    """The contract this restores: a repair that binds is readable, whichever family applied it."""
    gaussian = _quiet_optimize([1.0] * 50, estimator=GaussianEstimator(), max_its=3)
    bernoulli = _quiet_optimize([True] * 50, estimator=BernoulliEstimator(), max_its=3)
    assert gaussian.numerical_repairs()  # already true before this change
    assert bernoulli.numerical_repairs()  # was () -- the defect
