"""Population.step()'s population-wide multiplicity correction (mixle.evolve.population).

Regression coverage for the bug where a generation's challengers were each gated by a call to
challenger_beats_champion(..., multiplicity="bh") -- one call PER CANDIDATE, each seeing exactly one
p-value. Every method in mixle.inference.multiple_testing is the identity transform at family size 1
(bonferroni multiplies alpha by 1; BH ranks the lone value against itself), so that "correction" never
changed a single promotion decision. Population.step() now pools the whole generation's raw p-values
and corrects them together, ONCE, before finalizing any promotion -- see the multiplicity note on
Population.step and the loud-failure guard on challenger_beats_champion (evolve_verify_test.py).
"""

import math

import numpy as np
import pytest

from mixle.evolve.operators import Candidate
from mixle.evolve.population import OperatorBandit, Population
from mixle.evolve.verify import challenger_beats_champion
from mixle.inference.multiple_testing import adjust_pvalues

_N_OBS = 60
_N_CANDIDATES = 20
_ALPHA = 0.05


class _NullModel:
    """A bare placeholder 'fitted model'. ``noise_seed=None`` marks the champion/seed; every challenger
    carries its own ``noise_seed`` selecting an INDEPENDENT noise draw with the champion's exact same
    mean -- the true effect between champion and every challenger is exactly zero by construction, so
    any promotion is necessarily a false positive from sampling noise alone."""

    def __init__(self, noise_seed):
        self.noise_seed = noise_seed


class _NullObjective:
    """Deterministic per-observation scores: champion = a fixed baseline vector; every challenger =
    baseline + independent N(0, 1) noise. Every champion/challenger comparison is therefore a genuine
    null (no true effect) -- about ``alpha`` fraction of many such simultaneous tests should cross
    p < alpha by chance alone, exactly the setting multiplicity correction exists to guard against."""

    name = "null_diff"
    lower_is_better = True

    def __init__(self, n_obs, base_seed=999):
        self._base = np.random.RandomState(base_seed).normal(0.0, 1.0, n_obs)
        self.n_obs = n_obs

    def pointwise(self, model, data):
        # One score per row of the data actually handed in -- the gate requires that binding
        # (MXR-080-1765), and a fixture that returns a fixed-length vector regardless of `data`
        # would silently score a different set of rows than the one being gated. Slicing the fixed
        # baseline keeps the values (and hence every existing expectation at the full length)
        # identical while making the length follow `data`.
        n = len(data)
        base = self._base[:n]
        if model.noise_seed is None:
            return base.copy()
        return base + np.random.RandomState(model.noise_seed).normal(0.0, 1.0, n)

    def scalar(self, model, data):
        return float(np.mean(self.pointwise(model, data)))


class _NullOp:
    """A trivial operator whose 'improvement' is pure noise around the parent -- no true effect."""

    def __init__(self, name, noise_seed):
        self.name = name
        self.cost_hint = 1.0
        self.noise_seed = noise_seed

    def applicable(self, model, data, *, ctx):
        return True

    def propose(self, model, data, *, ctx):
        return Candidate(_NullModel(self.noise_seed), self.name)


def _null_generation():
    """20 simultaneous null-effect challengers in ONE generation (n_offspring == 20 distinct ops, so
    Population.step() proposes and compares all 20 in a single pass -- one multiplicity family)."""
    obj = _NullObjective(_N_OBS)
    seed_model = _NullModel(None)
    ops = [_NullOp(f"null_op_{i}", noise_seed=10_000 + i) for i in range(_N_CANDIDATES)]
    pop = Population([seed_model], objective=obj, operators=ops, size=2 * _N_CANDIDATES + 4, diversity_quota=0, seed=0)
    return pop, obj, seed_model, ops


def _uncorrected_promotions(obj, seed_model, ops, data):
    """The SAME 20 comparisons Population.step() just ran, gated with no multiplicity adjustment at
    all -- what naive, per-candidate-only significance testing would promote."""
    raw_pvals = []
    promotions = 0
    for op in ops:
        v = challenger_beats_champion(seed_model, _NullModel(op.noise_seed), data, objective=obj, seed=0)
        raw_pvals.append(v.p_value)
        promotions += int(v.promote)
    return promotions, np.asarray(raw_pvals)


def test_population_multiplicity_correction_reduces_false_positive_promotions():
    # Bug regression: naive uncorrected testing on 20 simultaneous null-effect challengers promotes
    # some of them by chance at alpha=0.05 (that's exactly what multiplicity correction exists to
    # guard against); a real population-wide correction, pooling all 20 raw p-values and adjusting them
    # together ONCE, must promote strictly fewer -- not the same count a per-candidate "correction"
    # (family size 1, the identity transform) would leave unchanged.
    pop, obj, seed_model, ops = _null_generation()
    data = list(range(_N_OBS))
    report = pop.step(data, data)

    uncorrected_promotions, raw_pvals = _uncorrected_promotions(obj, seed_model, ops, data)
    genuine_corrected_promotions = int(adjust_pvalues(raw_pvals, method="bh", alpha=_ALPHA)["n_reject"])

    assert report.proposals == _N_CANDIDATES
    # sanity: this scenario must actually produce false positives, or the test below proves nothing.
    assert uncorrected_promotions > 0
    # the actual behavioral difference the fix must produce: strictly fewer promotions than uncorrected.
    assert report.verified < uncorrected_promotions
    # and it must match a genuine population-wide correction over the exact same p-values -- proving
    # step() really pools and corrects once, rather than doing some other, still-wrong thing.
    assert report.verified == genuine_corrected_promotions
    # this scenario is fully deterministic (every RNG draw is seeded); pin the exact counts too, so a
    # silent behavior change anywhere in the pipeline is caught even if the relations above still hold.
    assert uncorrected_promotions == 2
    assert genuine_corrected_promotions == 0


def test_population_step_never_promotes_more_than_the_raw_gate_would():
    # Structural invariant, independent of the specific seed: every method in
    # mixle.inference.multiple_testing only ever raises a p-value relative to the raw one (never
    # lowers it), so the population-wide correction can only REVOKE a raw "challenger" verdict --
    # Population.step() must never promote a candidate the raw, uncorrected paired test itself refused.
    pop, obj, seed_model, ops = _null_generation()
    data = list(range(_N_OBS))
    report = pop.step(data, data)

    uncorrected_promotions, _ = _uncorrected_promotions(obj, seed_model, ops, data)
    assert report.verified <= uncorrected_promotions


def test_population_reward_slots_stay_aligned_with_operators_used():
    # The two-pass restructuring (raw comparison, then pooled correction) must not disturb the
    # per-slot bookkeeping GenerationReport callers rely on: one operators_used/rewards entry per
    # parent/operator pairing drawn this generation, in the same order.
    pop, obj, seed_model, ops = _null_generation()
    data = list(range(_N_OBS))
    report = pop.step(data, data)

    assert len(report.operators_used) == len(report.rewards) == _N_CANDIDATES
    assert set(report.operators_used) == {op.name for op in ops}


def test_bh_correction_of_the_same_family_is_reproducible_and_matches_module_helper():
    # A single population-level correction call, not one call per candidate: adjust_pvalues on the
    # pooled family should itself be deterministic and match the documented dispatcher behavior.
    obj = _NullObjective(_N_OBS)
    seed_model = _NullModel(None)
    ops = [_NullOp(f"null_op_{i}", noise_seed=10_000 + i) for i in range(_N_CANDIDATES)]
    data = list(range(_N_OBS))
    _, raw_pvals = _uncorrected_promotions(obj, seed_model, ops, data)

    first = adjust_pvalues(raw_pvals, method="bh", alpha=_ALPHA)["pvals_adjusted"]
    second = adjust_pvalues(raw_pvals, method="bh", alpha=_ALPHA)["pvals_adjusted"]
    np.testing.assert_array_equal(first, second)
    # a genuine pooled correction can only ever raise (or keep) each raw p-value, never lower it.
    assert np.all(np.asarray(first) >= raw_pvals - 1e-12)


# ---------------------------------------------------------------------------
# Regression: a scalar-only objective's nan p-value must not poison pass 2's pool
# ---------------------------------------------------------------------------
class _ScalarOnlyModel:
    """A bare placeholder 'fitted model' carrying just a scalar score -- stands in for a real model
    scored by a scalar-only objective (mixle.evolve.objective.calibration_objective /
    decision_regret_objective in production)."""

    def __init__(self, score):
        self.score = score


class _ScalarOnlyObjective:
    """Mimics calibration_objective / decision_regret_objective's shape: pointwise() always returns
    None (there is no per-observation vector, only a scalar), so every comparison goes through
    challenger_beats_champion's scalar-only branch, whose Verdict.p_value is nan by design -- see
    verify.py's module docstring point 8. This is what Population.step()'s pass 2 (pooling every
    compared candidate's raw p-value into one array before calling adjust_pvalues) must not choke on."""

    name = "scalar_only_null"
    lower_is_better = True

    def pointwise(self, model, data):
        return None

    def scalar(self, model, data):
        return model.score


class _ScalarOnlyOp:
    def __init__(self, name, score):
        self.name = name
        self.cost_hint = 1.0
        self.score = score

    def applicable(self, model, data, *, ctx):
        return True

    def propose(self, model, data, *, ctx):
        return Candidate(_ScalarOnlyModel(self.score), self.name)


def test_population_step_with_scalar_only_objective_does_not_crash_on_nan_pvalue_pool():
    # Bug regression: a scalar-only objective (no paired vector -- calibration_objective and
    # decision_regret_objective in mixle.evolve.objective are exactly this shape) makes every
    # compared candidate's Verdict.p_value nan (challenger_beats_champion's scalar-only branch has no
    # paired test to report a p-value for). Population.step()'s pass 2 pools every compared
    # candidate's raw p-value and hands the pool straight to adjust_pvalues, which rejects any
    # non-finite entry outright (mixle.inference.multiple_testing._prep) -- so a scalar-only objective
    # used to crash step() unconditionally, on every single generation, taking down promotion for the
    # whole generation rather than just leaving the scalar-only candidates uncorrected.
    obj = _ScalarOnlyObjective()
    champion_score = 1.0
    seed_model = _ScalarOnlyModel(champion_score)
    ops = [
        _ScalarOnlyOp("better", score=0.5),  # lower is better -> a genuine favored-direction win...
        _ScalarOnlyOp("worse", score=1.5),  # genuine regression, must not promote
        _ScalarOnlyOp("tie", score=1.0),  # no change at all, must not promote
    ]
    # 4 placeholder observations (not 1): this objective never touches `data`'s CONTENT, but
    # Population.step() now auto-splits `data` into a disjoint (fit, verify) pair by default
    # (MXR-080-0042) whenever verify_data is omitted, and mixle.evolve.improve._split requires at
    # least 4 observations to do that honestly.
    pop = Population([seed_model], objective=obj, operators=ops, size=8, diversity_quota=0, seed=0)
    report = pop.step([0.0, 1.0, 2.0, 3.0])

    assert report.proposals == 3
    # not just "didn't crash": every scalar-only verdict's OWN (unadjusted, there being no p-value for
    # a population-wide correction to act on) promotion decision must stand -- and per verify.py's
    # has_statistical_evidence guard (a bare scalar delta carries no sampling-uncertainty estimate), a
    # scalar-only verdict's promote is now ALWAYS False, regardless of favored direction: not even
    # "better"'s genuine favorable delta auto-promotes on its own.
    assert report.verified == 0
    assert report.rewards == [0.0, 0.0, 0.0]


def test_scalar_only_verdict_carries_nan_p_value_contract():
    # Pins two related contracts on challenger_beats_champion's scalar-only branch. (1) The nan
    # p_value pooling contract test_population_step_with_scalar_only_objective_does_not_crash_on_nan_
    # pvalue_pool depends on: p_value=nan (not, say, 1.0 or 0.0), so a pooling caller must treat it as
    # "exclude", never as a real, poolable value. (2) The promotion contract: nan p_value means no
    # paired test backs the verdict, so has_statistical_evidence and therefore promote are both False
    # -- regardless of favored -- for every scalar-only comparison, however favorable its raw delta.
    obj = _ScalarOnlyObjective()
    verdict = challenger_beats_champion(
        _ScalarOnlyModel(1.0), _ScalarOnlyModel(0.5), [0.0], objective=obj, require_calibration=False
    )
    assert math.isnan(verdict.p_value)
    assert all(math.isnan(c) for c in verdict.ci)
    assert verdict.favored == "challenger"
    # has_statistical_evidence is False whenever p_value is nan (see the property in verify.py) --
    # it is the gate promote (below) depends on.
    assert verdict.has_statistical_evidence is False
    # promote requires has_statistical_evidence (p_value not nan) as of verify.py's da1fec0b fix: a
    # scalar-only verdict can never auto-promote, however favorable its raw delta, since no sampling-
    # uncertainty estimate backs it.
    assert verdict.promote is False


# ---------------------------------------------------------------------------
# Regression (MXR-080-0042, Critical): verification must not silently equal training data by default
# ---------------------------------------------------------------------------
class _RecordingSpy:
    """Wraps a real Objective, recording the exact ``data`` list every ``pointwise``/``scalar`` call
    receives -- so a test can see exactly which rows Population actually scored on, without Population
    exposing its internal split as part of its public API."""

    def __init__(self, real):
        self._real = real
        self.name = real.name
        self.lower_is_better = real.lower_is_better
        self.calls: list[list] = []

    def pointwise(self, model, data):
        self.calls.append(list(data))
        return self._real.pointwise(model, data)

    def scalar(self, model, data):
        self.calls.append(list(data))
        return self._real.scalar(model, data)


def test_population_step_default_verify_is_disjoint_from_fit_data():
    # Bug regression (MXR-080-0042): Population.step()/.run() used to default verify_data to `data`
    # ITSELF (the exact same object) whenever a caller omitted it, so a challenger's parent was scored,
    # and every challenger gated, against the very rows it (or the population's own fitness bookkeeping)
    # was just fit on -- a near-tautological "held-out" test. The default must now derive a genuinely
    # disjoint (fit, verify) split instead.
    spy = _RecordingSpy(_NullObjective(_N_OBS))
    seed_model = _NullModel(None)
    ops = [_NullOp("op0", noise_seed=1)]
    data = list(range(_N_OBS))
    pop = Population([seed_model], objective=spy, operators=ops, size=4, diversity_quota=0, seed=0)

    pop.step(data)  # verify_data OMITTED -- exercises exactly the default path this fix changes

    assert spy.calls, "expected at least one scoring call"
    # the OLD (buggy) default aliased verify_data to `data`, so every scoring call would have seen the
    # FULL _N_OBS-observation data; the fix must make every one of them strictly smaller.
    assert all(len(c) < _N_OBS for c in spy.calls), (
        f"verify scoring saw the full {_N_OBS}-observation `data` -- no disjoint split happened: "
        f"{[len(c) for c in spy.calls]}"
    )


def test_population_default_verify_differs_from_the_pre_fix_full_data_alias():
    # Explicit before/after: the pre-fix default was literally `verify = data if verify_data is None
    # else verify_data` -- i.e. `data` itself. The new default (an auto-derived split) must be a STRICT
    # subset that never coincides with that old, buggy default.
    seed_model = _NullModel(None)
    ops = [_NullOp("op0", noise_seed=1)]
    data = list(range(_N_OBS))
    pop = Population([seed_model], objective=_NullObjective(_N_OBS), operators=ops, size=4, diversity_quota=0, seed=0)

    _, new_verify = pop._auto_split(data)
    old_verify = data  # the exact pre-fix default

    assert new_verify != old_verify
    assert len(new_verify) < len(old_verify)
    assert set(new_verify).issubset(set(old_verify))


def test_population_run_reuses_one_auto_split_verify_set_across_all_generations():
    # The auto-derived split must be resolved ONCE per run (not re-derived every generation) --
    # otherwise a champion's score would be measured against a different yardstick each generation,
    # corrupting the anti-regression bookkeeping worse than the train/verify overlap this fix replaces.
    # _auto_split is deterministic given (len(data), self.seed), so calling it twice with the same data
    # must reproduce the exact same split -- exactly what run() relies on internally.
    seed_model = _NullModel(None)
    pop = Population([seed_model], objective=_NullObjective(_N_OBS), size=4, diversity_quota=0, seed=0)
    data = list(range(_N_OBS))

    fit1, verify1 = pop._auto_split(data)
    fit2, verify2 = pop._auto_split(data)

    assert fit1 == fit2
    assert verify1 == verify2
    assert set(fit1).isdisjoint(set(verify1))
    assert fit1 and verify1


def test_population_step_explicit_verify_data_still_respected():
    # Negative control: a caller that explicitly WANTS verify_data == data (e.g. to reproduce the old
    # single-batch behavior on purpose, exactly as the multiplicity tests above do via pop.step(data,
    # data)) must still be able to -- this fix only changes the DEFAULT, never a caller's own explicit
    # choice.
    spy = _RecordingSpy(_NullObjective(_N_OBS))
    seed_model = _NullModel(None)
    ops = [_NullOp("op0", noise_seed=1)]
    data = list(range(_N_OBS))
    pop = Population([seed_model], objective=spy, operators=ops, size=4, diversity_quota=0, seed=0)

    pop.step(data, data)  # verify_data EXPLICITLY passed as `data` itself

    assert spy.calls
    assert all(len(c) == _N_OBS for c in spy.calls)


# ---------------------------------------------------------------------------
# Regression (MXR-080-0042): evaluate_holdout is the one genuinely evidence-bearing, one-shot check
# ---------------------------------------------------------------------------
class _GapModel:
    """A bare placeholder carrying a fixed mean offset from a shared random baseline."""

    def __init__(self, mean):
        self.mean = mean


class _GapObjective:
    """Deterministic per-observation scores: a shared low-noise baseline plus ``model.mean`` -- unlike
    _NullObjective (zero true effect by construction, for the multiplicity tests above), two _GapModels
    with different ``mean``s have a REAL, robustly-detectable paired difference, so evaluate_holdout has
    something genuine to find."""

    name = "gap"
    lower_is_better = True

    def __init__(self, n_obs, base_seed=777, noise_scale=0.05):
        self._base = np.random.RandomState(base_seed).normal(0.0, noise_scale, n_obs)

    def pointwise(self, model, data):
        return self._base + model.mean

    def scalar(self, model, data):
        return float(np.mean(self.pointwise(model, data)))


def test_evaluate_holdout_uses_only_the_holdout_data_never_the_runs_internal_data():
    # evaluate_holdout must be computed strictly from the caller-supplied holdout_data -- never from
    # whatever `data`/`verify_data` a prior step()/run() call used internally.
    spy = _RecordingSpy(_NullObjective(_N_OBS))
    seed_model = _NullModel(None)
    ops = [_NullOp("op0", noise_seed=1)]
    data = list(range(_N_OBS))
    pop = Population([seed_model], objective=spy, operators=ops, size=4, diversity_quota=0, seed=0)
    pop.step(data)
    spy.calls.clear()

    holdout_data = list(range(1000, 1000 + _N_OBS))
    pop.evaluate_holdout(holdout_data, require_calibration=False)

    assert spy.calls, "expected evaluate_holdout to score something"
    assert all(c == holdout_data for c in spy.calls), "evaluate_holdout must score ONLY the given holdout_data"


def test_evaluate_holdout_detects_a_genuine_improvement_with_real_statistical_evidence():
    # Unlike step()'s internally-reused verify_data (a heuristic search signal -- see Population's
    # class docstring), evaluate_holdout's one-shot Verdict must carry REAL statistical evidence when
    # there is a genuine, robust improvement to detect.
    obj = _GapObjective(_N_OBS)
    worse = _GapModel(1.0)
    better = _GapModel(0.0)  # lower is better -> genuinely, robustly better
    pop = Population([worse], objective=obj, size=4, diversity_quota=0, seed=0)
    pop._ensure_initialized(list(range(_N_OBS)))  # snapshot `worse` as the pre-run incumbent
    pop._champion = better  # stand in for "step()/run() already promoted a genuinely better model"

    verdict = pop.evaluate_holdout(list(range(_N_OBS)), require_calibration=False)

    assert verdict.favored == "challenger"
    assert verdict.has_statistical_evidence
    assert verdict.promote is True


def test_evaluate_holdout_does_not_promote_without_a_real_difference():
    # Negative control: no genuine gap between reference and champion -> no promotion.
    obj = _GapObjective(_N_OBS)
    same = _GapModel(1.0)
    pop = Population([same], objective=obj, size=4, diversity_quota=0, seed=0)

    verdict = pop.evaluate_holdout(list(range(_N_OBS)), require_calibration=False)

    assert verdict.favored == "tie"
    assert verdict.promote is False


class _Op:
    """A trivial named operator; the bandit only needs `name` and `cost_hint`."""

    def __init__(self, name, cost_hint=1.0):
        self.name = name
        self.cost_hint = cost_hint

    def applicable(self, model, data, *, ctx):
        del model, data, ctx
        return True

    def propose(self, model, data, *, ctx):
        del data, ctx
        return Candidate(model, {})


def test_operator_bandit_rejects_duplicate_names():
    # MXR-080-1772: `{op.name: op for op in ops}` collapsed two supplied operators into one arm.
    with pytest.raises(ValueError, match="unique"):
        OperatorBandit([_Op("same"), _Op("same")])
    assert len(OperatorBandit([_Op("a"), _Op("b")]).arms) == 2


def test_operator_bandit_refuses_non_finite_feedback():
    # One NaN update used to leave mean delta, mean cost, and the next Thompson value NaN forever.
    for bad in (float("nan"), float("inf"), float("-inf")):
        bandit = OperatorBandit([_Op("a"), _Op("b")])
        with pytest.raises(ValueError):
            bandit.reward("a", bad, 1.0)
        with pytest.raises(ValueError):
            bandit.reward("a", 1.0, bad)
    bandit = OperatorBandit([_Op("a"), _Op("b")])
    with pytest.raises(ValueError):
        bandit.reward("a", 1.0, -1.0)  # a negative cost is not a cost


def test_operator_bandit_policy_state_survives_a_rejected_update():
    bandit = OperatorBandit([_Op("a"), _Op("b")])
    bandit.reward("a", 2.0, 1.0)
    with pytest.raises(ValueError):
        bandit.reward("a", float("nan"), 1.0)
    report = bandit.report()["operators"]["a"]
    assert math.isfinite(report["mean_delta"]) and report["mean_delta"] > 0.0
    assert math.isfinite(report["mean_cost"])
    assert all(math.isfinite(bandit.value(name)) or name == "b" for name in ("a", "b"))


class _NanObjective:
    name = "nan_fitness"
    lower_is_better = True

    def pointwise(self, model, data):
        return None

    def scalar(self, model, data):
        del model, data
        return float("nan")


def test_population_rejects_nan_fitness_instead_of_ranking_it():
    # MXR-080-1771: two NaN-scored seeds plus generations=0 returned the first seed, best_score=NaN,
    # search_failed=False, and zero evaluations -- a failed search shaped like a successful one.
    pop = Population([_NullModel(None), _NullModel(1)], objective=_NanObjective(), size=4, seed=0)
    with pytest.raises(ValueError, match="finite"):
        pop.run(list(range(_N_OBS)), generations=0)


def test_population_run_reports_the_same_evaluation_semantics_as_other_backends():
    seeds = [_NullModel(None), _NullModel(1)]
    pop = Population(seeds, objective=_NullObjective(_N_OBS), operators=[_NullOp("op0", noise_seed=2)], seed=0)
    result = pop.run(list(range(_N_OBS)), generations=0)
    assert result.search_failed is False
    assert result.n_evaluations == len(seeds)  # the seed scorings really happened
    assert result.n_successes == result.n_evaluations
    assert math.isfinite(result.best_score)
