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

from mixle.evolve.operators import Candidate
from mixle.evolve.population import Population
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
        if model.noise_seed is None:
            return self._base.copy()
        return self._base + np.random.RandomState(model.noise_seed).normal(0.0, 1.0, self.n_obs)

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
        _ScalarOnlyOp("better", score=0.5),  # lower is better -> genuine improvement, should promote
        _ScalarOnlyOp("worse", score=1.5),  # genuine regression, must not promote
        _ScalarOnlyOp("tie", score=1.0),  # no change at all, must not promote
    ]
    pop = Population([seed_model], objective=obj, operators=ops, size=8, diversity_quota=0, seed=0)

    report = pop.step([0.0])  # this objective never touches `data`; a placeholder is enough

    assert report.proposals == 3
    # not just "didn't crash": the scalar-only verdict's OWN (unadjusted, there being no p-value for a
    # population-wide correction to act on) promotion decision must stand -- exactly the genuinely
    # better candidate promotes, not the tied or regressed ones.
    assert report.verified == 1
    assert report.rewards == [0.5, 0.0, 0.0]


def test_scalar_only_verdict_carries_nan_p_value_contract():
    # Pins the exact contract test_population_step_with_scalar_only_objective_does_not_crash_on_nan_
    # pvalue_pool depends on: challenger_beats_champion's scalar-only branch returns p_value=nan (not,
    # say, 1.0 or 0.0), so a pooling caller must treat it as "exclude", never as a real, poolable value.
    obj = _ScalarOnlyObjective()
    verdict = challenger_beats_champion(
        _ScalarOnlyModel(1.0), _ScalarOnlyModel(0.5), [0.0], objective=obj, require_calibration=False
    )
    assert math.isnan(verdict.p_value)
    assert all(math.isnan(c) for c in verdict.ci)
    assert verdict.favored == "challenger"
    assert verdict.promote is True
