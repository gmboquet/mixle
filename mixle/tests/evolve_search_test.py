"""mixle.evolve Phase 2-3: typed-space search + the meta-search that learns which operators help."""

import numpy as np
import pytest

from mixle.evolve import (
    Categorical,
    Integer,
    OperatorBandit,
    Population,
    Real,
    Recompose,
    SearchResult,
    Space,
    auto_select,
    challenger_beats_champion,
    default_operators,
    nll_objective,
    registered_operators,
    search,
)
from mixle.evolve.operators import Candidate
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


def test_space_sample_and_neighbors():
    sp = Space({"a": Real(0.0, 1.0), "k": Integer(1, 5), "c": Categorical(["x", "y", "z"])})
    rng = np.random.RandomState(0)
    cfg = sp.sample(rng)
    assert 0.0 <= cfg["a"] <= 1.0 and 1 <= cfg["k"] <= 5 and cfg["c"] in ("x", "y", "z")
    neighbors = sp.neighbors(cfg)
    assert neighbors and all(set(n) == set(cfg) for n in neighbors)


def test_real_rejects_infinite_bounds():
    # Bug regression: Real.__post_init__ only checked lo < hi, not finiteness, so an infinite (or
    # NaN) bound constructed fine and then poisoned sample()/neighbors() with inf/nan -- or crashed
    # deep inside NumPy's RNG with an opaque OverflowError, far from this actual root cause.
    with pytest.raises(ValueError, match="finite"):
        Real(-np.inf, np.inf)
    with pytest.raises(ValueError, match="finite"):
        Real(0.0, np.inf)
    with pytest.raises(ValueError, match="finite"):
        Real(-np.inf, 1.0)
    with pytest.raises(ValueError, match="finite"):
        Real(np.nan, 1.0)
    # negative control: finite bounds must keep working exactly as before
    r = Real(-5.0, 5.0)
    rng = np.random.RandomState(0)
    assert all(-5.0 <= r.sample(rng) <= 5.0 for _ in range(20))
    assert all(np.isfinite(n) and -5.0 <= n <= 5.0 for n in r.neighbors(0.0))


def test_categorical_rejects_duplicate_choices():
    # Bug regression: Categorical didn't reject duplicate choices, and its own notion of cardinality
    # disagreed depending which path asked -- encode() collapsed a duplicate to its first occurrence's
    # index (only `len(set)` distinct codes reachable) while sample()/bounds()/len() still treated the
    # un-deduplicated list length as the true choice count (e.g. biased sampling toward the repeated
    # value, and neighbors() could hand back the input value as its own "neighbor").
    with pytest.raises(ValueError, match="unique"):
        Categorical(["a", "b", "a"])
    with pytest.raises(ValueError, match="unique"):
        Categorical([1, 2, 3, 2])
    # negative control: unique choices must keep working exactly as before
    c = Categorical(["a", "b", "c"])
    assert len(c.choices) == 3
    assert {c.encode(v) for v in c.choices} == {0.0, 1.0, 2.0}  # every choice gets its own code
    rng = np.random.RandomState(0)
    assert all(c.sample(rng) in ("a", "b", "c") for _ in range(20))
    assert set(c.neighbors("a")) == {"b", "c"}  # excludes exactly the queried value, nothing else


def test_search_evolutionary_finds_variance():
    data = list(np.random.RandomState(1).normal(0.0, 2.0, 300))  # true variance 4
    mu = float(np.mean(data))
    sp = Space({"sigma2": Real(0.5, 12.0)})
    res = search(
        sp,
        data,
        objective=nll_objective(),
        build_fn=lambda cfg: GaussianDistribution(mu, float(cfg["sigma2"])),
        method="evolutionary",
        n_iter=40,
        seed=2,
    )
    assert isinstance(res, SearchResult)
    assert 2.5 < res.best_config["sigma2"] < 6.0  # recovers the true ~4


def test_auto_select_challenger_is_independent_of_val_split():
    # Finding-1 regression: auto_select's challenger used to be refit on `rows` (train+val) and then
    # scored on that SAME val -- so the challenger's own fit depended on val, which is not a held-out
    # comparison at all. `_split`'s permutation is a pure function of (n, seed), independent of the
    # data VALUES, so for a fixed (n, holdout, seed) we know in advance which positions land in train
    # vs val; perturbing ONLY the val-assigned positions must never move the gate's decision, since the
    # champion and challenger are both fit purely from the (untouched) train-assigned positions.
    from mixle.evolve.improve import _split

    n, holdout, seed = 200, 0.25, 0
    base_data = list(np.random.RandomState(123).normal(0.0, 1.0, n))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_verify = min(max(1, int(round(holdout * n))), n - 1)
    verify_idx = set(perm[:n_verify].tolist())

    shifted_data = list(base_data)
    for i in verify_idx:
        shifted_data[i] += 6.0  # perturb ONLY the positions _split will assign to val

    t0, v0 = _split(base_data, holdout, seed)
    t1, v1 = _split(shifted_data, holdout, seed)
    assert t0 == t1  # sanity: train split is byte-identical between the two datasets
    assert v0 != v1  # sanity: val split really did change

    obj = nll_objective()
    res_orig = auto_select(base_data, criterion=obj, verify=True, holdout=holdout, seed=seed)
    res_shift = auto_select(shifted_data, criterion=obj, verify=True, holdout=holdout, seed=seed)

    # the pre-fix bug produced verified=True with an artificially huge delta (~15) on the shifted val,
    # purely because the challenger had already fit those exact (shifted) observations. The gate
    # decision must now be immune to a perturbation that only ever touches val-assigned rows.
    assert res_orig.verified is False
    assert res_shift.verified is False
    assert res_orig.delta == pytest.approx(0.0, abs=1e-6)
    assert res_shift.delta == pytest.approx(0.0, abs=1e-6)


def test_auto_select_verify_gate_negative_control_normal_data():
    # negative control for Finding 1: the ordinary (non-adversarial) verify path must still run to
    # completion and return a real, usable result -- the leak fix must not have made the gate inert.
    rng = np.random.RandomState(9)
    data = list(rng.normal(2.0, 1.5, 400))
    res = auto_select(data, criterion=nll_objective(), verify=True, seed=0)
    assert res.model is not None
    assert "family" in res.evidence
    assert isinstance(res.verified, bool)


def test_operator_bandit_concentrates_on_winner():
    class _Op:
        def __init__(self, name, win):
            self.name, self.cost_hint, self._win = name, 1.0, win

        def applicable(self, m, d, *, ctx):
            return True

        def propose(self, m, d, *, ctx):
            return Candidate(m, self.name)

    bandit = OperatorBandit([_Op("good", 1), _Op("bad", 0)], seed=0)
    for _ in range(60):
        op = bandit.select(1)[0]
        bandit.reward(op.name, 1.0 if op.name == "good" else 0.0, 1.0)
    ops = bandit.report()["operators"]
    assert ops["good"]["pulls"] > ops["bad"]["pulls"]  # the policy learned which operator helps


def test_population_improves_bad_seed():
    obj = nll_objective()
    seed_model = GaussianDistribution(0.0, 1.0)  # wrong mean and variance
    data = list(np.random.RandomState(3).normal(5.0, 1.0, 80))
    result = Population([seed_model], objective=obj, seed=0).run(data, generations=2)
    # anti-regression: the evolved champion is never worse than the seed
    assert obj.scalar(result.best_model, data) <= obj.scalar(seed_model, data)


def test_population_step_scores_verify_data_not_train():
    # Bug-1 regression: op.propose must fit on `data` (train); challenger_beats_champion and the
    # population's own fitness scoring must run on the disjoint `verify_data` -- never on `data` itself
    # (scoring a candidate on the same rows it was just fit on is a near-tautological "test").
    real_obj = nll_objective()
    scored_ids = []

    class _SpyObjective:
        name = "spy_nll"
        lower_is_better = True

        def pointwise(self, model, data):
            scored_ids.append(id(data))
            return real_obj.pointwise(model, data)

        def scalar(self, model, data):
            scored_ids.append(id(data))
            return real_obj.scalar(model, data)

    train = list(np.random.RandomState(11).normal(5.0, 1.0, 40))
    verify = list(np.random.RandomState(12).normal(5.0, 1.0, 40))
    seed_model = GaussianDistribution(0.0, 1.0)

    pop = Population([seed_model], objective=_SpyObjective(), seed=0)
    pop.step(train, verify)

    assert scored_ids, "expected the spy objective to be invoked at least once (seed scoring)"
    assert all(sid == id(verify) for sid in scored_ids)  # every score call used the SAME verify object
    assert id(train) not in scored_ids  # train (the fit data) must never be scored


def test_search_bandit_scores_on_held_out_split():
    # Bug-1 regression: search(method="bandit") must gate/score every candidate on the held-out val
    # split search() computes up front -- not on the unsplit rows / the train split it was fit on.
    from mixle.evolve.improve import _split

    obj = nll_objective()
    data = list(np.random.RandomState(4).normal(5.0, 1.0, 80))
    holdout, seed = 0.25, 0
    expected_train, expected_val = _split(data, holdout, seed)
    expected_train_sorted = sorted(expected_train)
    expected_val_sorted = sorted(expected_val)

    scored_batches = []

    class _SpyObjective:
        name = obj.name
        lower_is_better = obj.lower_is_better

        def pointwise(self, model, batch):
            scored_batches.append(sorted(batch))
            return obj.pointwise(model, batch)

        def scalar(self, model, batch):
            scored_batches.append(sorted(batch))
            return obj.scalar(model, batch)

    sp = Space({"mu": Real(-2.0, 8.0)})
    res = search(
        sp,
        data,
        objective=_SpyObjective(),
        build_fn=lambda cfg: GaussianDistribution(float(cfg["mu"]), 1.0),
        method="bandit",
        n_iter=2,
        seed=seed,
        holdout=holdout,
    )
    assert isinstance(res, SearchResult) and res.best_model is not None
    assert scored_batches, "expected the objective to be invoked for scoring"
    for batch in scored_batches:
        assert batch == expected_val_sorted, "every scoring call must use the held-out val split"
        assert batch != expected_train_sorted


def test_recompose_captures_bimodal_structure():
    rng = np.random.RandomState(0)
    data = list(rng.normal(-4.0, 0.5, 120)) + list(rng.normal(4.0, 0.5, 120))  # clearly bimodal
    champion = GaussianDistribution(0.0, 5.0)  # one wide Gaussian misses the modes
    obj = nll_objective()
    op = Recompose()
    assert op.applicable(champion, data, ctx={})
    cand = op.propose(champion, data, ctx={"seed": 0})
    assert obj.scalar(cand.model, data) < obj.scalar(champion, data)  # the 2-component mixture fits better
    verdict = challenger_beats_champion(champion, cand.model, data, objective=obj, nonnested=True)
    assert verdict.favored == "challenger"  # and it passes the verify gate


def test_recompose_registered_but_off_by_default():
    assert "recompose" in registered_operators()
    assert "recompose" not in {op.name for op in default_operators()}  # structural + expensive -> opt-in


def test_structural_genotype_distance():
    from mixle.evolve import model_signature, structural_distance
    from mixle.ops import mixture

    g = GaussianDistribution(0.0, 1.0)
    m2 = mixture([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)])
    m3 = mixture([GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)])
    assert model_signature(g) == ("GaussianDistribution", [])
    assert structural_distance(g, g) == 0.0
    assert structural_distance(m2, m3) < structural_distance(g, m3)  # 2-vs-3 comps closer than leaf-vs-3


def test_mutate_grows_structure_by_selection():
    from mixle.evolve import Mutate

    rng = np.random.RandomState(0)
    data = list(rng.normal(-4.0, 0.5, 120)) + list(rng.normal(4.0, 0.5, 120))
    champion = GaussianDistribution(0.0, 5.0)
    obj = nll_objective()
    op = Mutate()
    assert op.applicable(champion, data, ctx={})
    # structure search = mutate + select: over a few seeds, at least one mutation beats the single Gaussian
    best = min(obj.scalar(op.propose(champion, data, ctx={"seed": s}).model, data) for s in range(6))
    assert best < obj.scalar(champion, data)
    assert "mutate" in registered_operators() and "mutate" not in {o.name for o in default_operators()}


def test_search_bandit_method_runs():
    obj = nll_objective()
    data = list(np.random.RandomState(4).normal(5.0, 1.0, 80))
    sp = Space({"mu": Real(-2.0, 8.0)})
    res = search(
        sp,
        data,
        objective=obj,
        build_fn=lambda cfg: GaussianDistribution(float(cfg["mu"]), 1.0),
        method="bandit",
        n_iter=2,
        seed=0,
    )
    assert isinstance(res, SearchResult) and res.best_model is not None
