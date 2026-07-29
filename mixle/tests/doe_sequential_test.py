"""The sequential-design loop primitive: a real Bayesian sequential design (uncertainty genuinely
shrinks as data accumulates), every stop path exercised, and -- the payoff -- a test proving the loop
composes with the real voi_stopping_decision rule from mixle.analysis.real_options, i.e. the
session's decision machinery snaps together instead of being hand-wired per demo.

Also covers the audit trail's integrity guarantees: stored summary/decision records survive later
mutation of the object a callback returned (copy at storage time) and survive a callback reaching into
its `history` argument and trying to rewrite a past round (copy-safe view at callback time); and every
fit/propose/acquire/combine failure mode records an explicit failed round rather than aborting silently,
under both the default fail-fast `on_error="raise"` policy and the `on_error="record_and_stop"` policy.
"""

import numpy as np
import pytest

from mixle.doe.sequential import DesignRound, SequentialDesignError, sequential_design

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
            {"method": "variance_rescaling_heuristic", "variance_reduction": 0.5},
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


# --- audit-trail integrity: negative control + copy-safety + callback-mutation resistance ---


def test_normal_run_marks_no_round_as_failed():
    """Negative control: a fully successful multi-round run leaves the failure-tracking fields alone
    and produces a correct, complete audit trail (the existing stop-path tests above already check the
    per-round summaries/decisions themselves)."""
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
    assert result.n_rounds > 1
    assert all(r.failed is False and r.failed_step is None and r.error is None for r in result.rounds)


def test_mutating_a_returned_summary_after_the_fact_does_not_corrupt_the_stored_round():
    """summarize() returns a dict; mutating that SAME object after the round has completed must not
    retroactively change the stored audit-trail entry -- the round must store a copy, not the live
    reference summarize() handed back."""
    captured = {}

    def summarize_and_capture(state, i):
        d = {"round": i, "n": state.n, "post_sd": state.post_sd}
        captured[i] = d
        return d

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=summarize_and_capture,
        should_continue=_threshold_controller(0.15),
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=50,
    )
    before = dict(result.rounds[0].summary)
    captured[0]["post_sd"] = -999.0  # tamper with the object summarize() originally returned
    captured[0]["n"] = -999
    assert result.rounds[0].summary == before


def test_should_continue_cannot_mutate_the_live_history_it_is_given():
    """should_continue is handed a copy-safe view of the audit trail: writing into a past round's
    summary/decision through that view must not change what sequential_design actually stored."""

    def tampering_should_continue(history):
        if len(history) >= 2:
            history[0].summary["post_sd"] = -999.0  # try to rewrite a PAST round
            history[0].decision["reason"] = "tampered"
        sd = history[-1].summary["post_sd"]
        return {"keep_going": sd > 0.15, "reason": "ok"}

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=tampering_should_continue,
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=50,
    )
    assert result.rounds[0].summary["post_sd"] > 0  # never tampered to -999
    assert result.rounds[0].decision.get("reason") != "tampered"


def test_propose_cannot_mutate_the_live_history_it_is_given():
    """Same aliasing guarantee, exercised through propose's `history` argument instead."""

    def tampering_propose(state, history):
        history[0].summary["n"] = -1
        return len(history)

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=_threshold_controller(0.15),
        propose=tampering_propose,
        acquire=_acquire,
        combine=_combine,
        max_rounds=50,
    )
    assert result.rounds[0].summary["n"] >= 0  # never tampered to -1


# --- explicit failure-state rounds: one test per fit/propose/acquire/combine callback ---


def test_fit_failure_is_recorded_as_an_explicit_failed_round():
    calls = {"n": 0}

    def flaky_fit(data):
        calls["n"] += 1
        if calls["n"] == 2:  # fail on round 1's fit, after round 0 completed cleanly
            raise RuntimeError("fit blew up")
        return _fit(data)

    result = sequential_design(
        _initial(),
        fit=flaky_fit,
        summarize=_summarize,
        should_continue=lambda h: {"keep_going": True, "reason": "go"},
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert result.stopped_reason == "callback_error"
    assert result.n_rounds == 2  # round 0 (clean) + round 1 (the new failure round)
    failed = result.rounds[-1]
    assert failed.index == 1
    assert failed.failed is True
    assert failed.failed_step == "fit"
    assert "fit blew up" in failed.error
    assert failed.state is None  # fit never returned a state
    assert result.rounds[0].failed is False  # round 0's record is untouched


def test_propose_failure_is_recorded_on_the_round_that_attempted_it():
    def boom_propose(state, history):
        raise ValueError("propose blew up")

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=lambda h: {"keep_going": True, "reason": "go"},
        propose=boom_propose,
        acquire=_acquire,
        combine=_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert result.stopped_reason == "callback_error"
    assert result.n_rounds == 1  # round 0's own record is marked failed, not duplicated
    failed = result.rounds[-1]
    assert failed.index == 0
    assert failed.failed_step == "propose"
    assert "propose blew up" in failed.error
    assert failed.summary  # fit/summarize/decision for round 0 completed before propose ran
    assert failed.proposed_action is None  # propose never returned one


def test_acquire_failure_is_recorded_with_the_proposal_still_on_record():
    def boom_acquire(action):
        raise ValueError("acquire blew up")

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=lambda h: {"keep_going": True, "reason": "go"},
        propose=_propose_next_measurement,
        acquire=boom_acquire,
        combine=_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert result.stopped_reason == "callback_error"
    failed = result.rounds[-1]
    assert failed.failed_step == "acquire"
    assert failed.proposed_action is not None  # propose succeeded before acquire failed
    assert "acquire blew up" in failed.error


def test_combine_failure_is_recorded_after_a_successful_acquire():
    def boom_combine(data, new):
        raise ValueError("combine blew up")

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=lambda h: {"keep_going": True, "reason": "go"},
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=boom_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert result.stopped_reason == "callback_error"
    failed = result.rounds[-1]
    assert failed.failed_step == "combine"
    assert failed.proposed_action is not None  # propose and acquire both succeeded
    assert "combine blew up" in failed.error


# --- caller-selected rethrow policy ---


def test_on_error_raise_is_the_default_and_attaches_the_partial_result_to_the_exception():
    def boom_acquire(action):
        raise KeyError("acquire blew up")

    with pytest.raises(SequentialDesignError) as excinfo:
        sequential_design(
            _initial(),
            fit=_fit,
            summarize=_summarize,
            should_continue=lambda h: {"keep_going": True, "reason": "go"},
            propose=_propose_next_measurement,
            acquire=boom_acquire,
            combine=_combine,
            max_rounds=5,
            # on_error left at its default -- "raise"
        )
    err = excinfo.value
    assert isinstance(err.__cause__, KeyError)  # original exception chained, not swallowed
    assert err.result.stopped_reason == "callback_error"
    assert err.result.rounds[-1].failed_step == "acquire"
    assert "acquire blew up" in err.result.rounds[-1].error


def test_on_error_record_and_stop_returns_a_partial_result_instead_of_raising():
    def boom_acquire(action):
        raise KeyError("acquire blew up")

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=lambda h: {"keep_going": True, "reason": "go"},
        propose=_propose_next_measurement,
        acquire=boom_acquire,
        combine=_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert result.stopped_reason == "callback_error"
    assert result.rounds[-1].failed_step == "acquire"


def test_on_error_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="on_error"):
        sequential_design(
            _initial(),
            fit=_fit,
            summarize=_summarize,
            should_continue=_threshold_controller(0.15),
            propose=_propose_next_measurement,
            acquire=_acquire,
            combine=_combine,
            max_rounds=5,
            on_error="retry_forever",
        )


# --- summary/decision well-formedness validation ---


def test_non_dict_summary_is_rejected():
    with pytest.raises(SequentialDesignError, match="summary_validation") as excinfo:
        sequential_design(
            _initial(),
            fit=_fit,
            summarize=lambda state, i: "not a dict",
            should_continue=_threshold_controller(0.15),
            propose=_propose_next_measurement,
            acquire=_acquire,
            combine=_combine,
            max_rounds=5,
        )
    assert isinstance(excinfo.value.__cause__, TypeError)
    assert excinfo.value.result.rounds[-1].failed_step == "summary_validation"


def test_non_dict_decision_is_rejected():
    with pytest.raises(SequentialDesignError, match="decision_validation") as excinfo:
        sequential_design(
            _initial(),
            fit=_fit,
            summarize=_summarize,
            should_continue=lambda h: None,
            propose=_propose_next_measurement,
            acquire=_acquire,
            combine=_combine,
            max_rounds=5,
        )
    assert isinstance(excinfo.value.__cause__, TypeError)
    assert excinfo.value.result.rounds[-1].failed_step == "decision_validation"


@pytest.mark.parametrize("failed_step", ["summarize", "should_continue"])
def test_summary_and_decision_callback_failures_return_auditable_results(failed_step):
    def boom(*args):
        raise RuntimeError(f"{failed_step} blew up")

    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=boom if failed_step == "summarize" else _summarize,
        should_continue=boom if failed_step == "should_continue" else _threshold_controller(0.15),
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert result.stopped_reason == "callback_error"
    assert result.rounds[-1].failed_step == failed_step
    assert f"{failed_step} blew up" in result.rounds[-1].error


@pytest.mark.parametrize("keep_going", ["false", 0, 1, None, np.bool_(False)])
def test_non_boolean_decision_never_triggers_acquisition(keep_going):
    acquisitions = []
    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=lambda history: {"keep_going": keep_going, "reason": "invalid"},
        propose=_propose_next_measurement,
        acquire=lambda action: acquisitions.append(action),
        combine=_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert acquisitions == []
    assert result.stopped_reason == "callback_error"
    assert result.rounds[-1].failed_step == "decision_validation"


@pytest.mark.parametrize(
    ("callback", "bad_record", "failed_step"),
    [
        ("summary", {"bad": object()}, "summary_validation"),
        ("summary", {"bad": float("nan")}, "summary_validation"),
        ("summary", {1: "integer key"}, "summary_validation"),
        ("decision", {"keep_going": False, "reason": "stop", "bad": object()}, "decision_validation"),
        ("decision", {"keep_going": False, "reason": ""}, "decision_validation"),
    ],
)
def test_audit_records_enforce_portable_json_and_required_decision_reason(callback, bad_record, failed_step):
    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=(lambda state, index: bad_record) if callback == "summary" else _summarize,
        should_continue=(lambda history: bad_record) if callback == "decision" else _threshold_controller(0.15),
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert result.stopped_reason == "callback_error"
    assert result.rounds[-1].failed_step == failed_step


def test_history_copy_failure_is_named_and_preserves_the_partial_result(monkeypatch):
    def broken_deepcopy(value):
        raise RuntimeError("copy failed")

    monkeypatch.setattr("mixle.doe.sequential.copy.deepcopy", broken_deepcopy)
    result = sequential_design(
        _initial(),
        fit=_fit,
        summarize=_summarize,
        should_continue=_threshold_controller(0.15),
        propose=_propose_next_measurement,
        acquire=_acquire,
        combine=_combine,
        max_rounds=5,
        on_error="record_and_stop",
    )
    assert result.stopped_reason == "callback_error"
    assert result.rounds[-1].failed_step == "decision_history"
