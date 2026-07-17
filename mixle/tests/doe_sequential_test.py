"""The sequential-design loop primitive: a real Bayesian sequential design (uncertainty genuinely
shrinks as data accumulates), every stop path exercised, and -- the payoff -- a test proving the loop
composes with the real voi_stopping_decision rule from mixle.analysis.real_options, i.e. the
session's decision machinery snaps together instead of being hand-wired per demo."""

import numpy as np
import pytest

from mixle.doe.sequential import DesignRound, sequential_design

# --- a genuinely Bayesian toy: estimate a scalar theta from noisy iid measurements ---
_THETA_TRUE = 2.0
_TAU = 5.0  # prior sd
_SIGMA = 1.0  # measurement noise sd


class _ScalarPosterior:
    """Minimal IC-1-ish posterior over a scalar: conjugate-Gaussian update from n measurements."""

    def __init__(self, measurements: list[float]):
        n = len(measurements)
        post_var = 1.0 / (1.0 / _TAU**2 + n / _SIGMA**2)
        post_mean = post_var * (np.sum(measurements) / _SIGMA**2)  # prior mean 0
        self.mean = np.array([post_mean])
        self.cov = np.array([[post_var]])
        self.post_sd = float(np.sqrt(post_var))
        self.n = n

    def samples(self, n, rng):
        return rng.normal(self.mean[0], self.post_sd, size=(n, 1))


def _fit(data):
    return _ScalarPosterior(data)


def _summarize(state, i):
    return {"round": i, "n": state.n, "post_sd": state.post_sd}


def _acquire(_action):
    rng = np.random.default_rng(_action)  # action carries a seed so the test is deterministic
    return float(_THETA_TRUE + rng.normal(0, _SIGMA))


def _combine(data, new):
    return data + [new]


def _initial(seed_base: int = 1000, n0: int = 2):
    return [_acquire(seed_base + j) for j in range(n0)]


def _threshold_controller(threshold: float):
    def should_continue(history: list[DesignRound]):
        sd = history[-1].summary["post_sd"]
        return {"keep_going": sd > threshold, "reason": f"post_sd={sd:.3f} vs threshold {threshold}"}

    return should_continue


def _propose_next_measurement(state, history):
    # all measurements are iid here, so "the design" is trivial -- just request another, seeded by round.
    return 5000 + len(history)


def test_uncertainty_actually_shrinks_and_the_loop_stops_when_tight_enough():
    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=_threshold_controller(0.15),
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=50,
    )
    assert result.stopped_reason == "controller_stop"
    sds = [r.summary["post_sd"] for r in result.rounds]
    assert sds == sorted(sds, reverse=True)  # strictly non-increasing: each measurement tightens the posterior
    assert result.final_state.post_sd <= 0.15
    # every round but the last proposed a next sample; the last (stopping) round proposed nothing.
    assert all(r.proposed_action is not None for r in result.rounds[:-1])
    assert result.rounds[-1].proposed_action is None


def test_budget_exhausted_when_threshold_is_never_reached():
    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=_threshold_controller(1e-6),  # unreachable
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=3,
    )
    assert result.stopped_reason == "budget_exhausted"
    assert result.n_rounds == 4  # round 0 (initial) + 3 adaptive rounds


@pytest.mark.parametrize("max_rounds", [-1, 1.5, True])
def test_invalid_round_budgets_are_rejected(max_rounds):
    with pytest.raises(ValueError, match="nonnegative integer"):
        sequential_design(
            _initial(),
            fit=_fit,
            summarize=_summarize,
            should_continue=_threshold_controller(0.15),
            propose=_propose_next_measurement,
            acquire=_acquire,
            combine=_combine,
            max_rounds=max_rounds,
        )


def test_no_proposal_stops_the_loop_even_if_controller_wants_to_continue():
    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=lambda h: {"keep_going": True, "reason": "always"},
        propose=lambda state, history: None,  # no admissible next sample
        acquire=_acquire,
        combine=_combine,
        max_rounds=10,
    )
    assert result.stopped_reason == "no_proposal"
    assert result.n_rounds == 1


def test_history_is_complete_and_ordered():
    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=_threshold_controller(0.2),
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=50,
    )
    assert [r.index for r in result.rounds] == list(range(result.n_rounds))
    assert all("post_sd" in r.summary and r.decision for r in result.rounds)


def test_composes_with_the_real_voi_stopping_decision_rule():
    """The payoff: the loop's stop-decision slot takes the actual value-of-information stopping rule
    (mixle.analysis.real_options.voi_stopping_decision) with no adapter -- the pieces built across this
    session snap together. Asserts they compose and terminate cleanly, not a specific round count."""
    from mixle.analysis.real_options import voi_stopping_decision

    def _decision_value(samples):
        return float(max(np.mean(samples[:, 0]), 0.0))  # risk-neutral go/no-go

    def voi_controller(history: list[DesignRound]):
        state = history[-1].state
        rng = np.random.default_rng(len(history))
        decision = voi_stopping_decision(
            state,
            _decision_value,
            {"variance_reduction": 0.5},
            sample_cost=0.05,
            rng=rng,
        )
        return {
            "keep_going": bool(decision.keep_sampling),
            "reason": f"voi={decision.voi_dollars:.4f}",
            "voi": decision.voi_dollars,
        }

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=voi_controller,
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=25,
    )
    assert result.stopped_reason in ("controller_stop", "budget_exhausted")
    assert result.n_rounds >= 1
    assert all("voi" in r.decision for r in result.rounds)
