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


def test_integer_rejects_non_integer_bounds():
    # MXR-080-1775 regression: Integer.__post_init__ only compared int(lo) < int(hi), so a fractional
    # bound truncated silently -- Integer(0.5, 3) declared an integer dimension whose stored lo was
    # 0.5, making bounds()/sample()/encode() disagree about whether 0 was a legal level. Booleans got
    # through the same way (True is an int), so Integer(True, 5) silently meant Integer(1, 5).
    with pytest.raises(ValueError, match="must be an integer"):
        Integer(0.5, 3)
    with pytest.raises(ValueError, match="must be an integer"):
        Integer(1, 3.5)
    with pytest.raises(ValueError, match="must be an integer"):
        Integer(True, 5)
    with pytest.raises(ValueError, match="must be an integer"):
        Integer(1, True)
    with pytest.raises(ValueError, match="must be an integer"):
        Integer(np.float64(1.0), 5)
    # negative control: integer bounds -- including NumPy integer scalars -- keep working
    for lo, hi in ((1, 5), (np.int64(1), np.int32(5)), (-3, 3)):
        dim = Integer(lo, hi)
        rng = np.random.RandomState(0)
        assert all(int(lo) <= dim.sample(rng) <= int(hi) for _ in range(20))
        assert dim.bounds() == (float(lo) - 0.5, float(hi) + 0.5)


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
        build_fn=lambda cfg, train_data: GaussianDistribution(mu, float(cfg["sigma2"])),
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


def test_search_build_fn_receives_train_data_only_never_val():
    # Finding-2 regression: build_fn used to be called with ONLY `config` -- `_held_out_score` accepted
    # a `train` parameter but never actually passed it through, so a build_fn had no way to fit itself
    # to real training data. build_fn must now receive (config, train_data), and train_data must be
    # exactly the train split search() computed (never val, never the unsplit data) -- checked here for
    # all three backends (bo kept at n_iter=1 so it never needs the torch-only GP acquisition step).
    from mixle.evolve.improve import _split

    data = list(np.random.RandomState(5).normal(3.0, 1.0, 100))
    holdout, seed = 0.3, 1
    expected_train, _ = _split(data, holdout, seed)
    expected_train_sorted = sorted(expected_train)

    seen_train_data = []

    def build_fn(cfg, train_data):
        seen_train_data.append(sorted(train_data))
        return GaussianDistribution(float(cfg["mu"]), 1.0)

    sp = Space({"mu": Real(-2.0, 8.0)})

    for method, kwargs in (("bo", {}), ("evolutionary", {"mu": 2, "lam": 2}), ("bandit", {})):
        seen_train_data.clear()
        res = search(
            sp,
            data,
            objective=nll_objective(),
            build_fn=build_fn,
            method=method,
            n_iter=1,
            seed=seed,
            holdout=holdout,
            **kwargs,
        )
        assert isinstance(res, SearchResult)
        assert seen_train_data, f"method={method!r}: build_fn was never called"
        for got in seen_train_data:
            assert got == expected_train_sorted, f"method={method!r}: build_fn did not receive the train split"


def test_search_build_fn_ignoring_train_data_still_works():
    # negative control for Finding 2: a build_fn that doesn't need train_data (the shape of every
    # existing caller in this file) must keep working exactly as before -- the extra parameter is
    # purely additive, not a breaking requirement to actually use it.
    data = list(np.random.RandomState(1).normal(0.0, 2.0, 300))
    mu = float(np.mean(data))
    sp = Space({"sigma2": Real(0.5, 12.0)})
    res = search(
        sp,
        data,
        objective=nll_objective(),
        build_fn=lambda cfg, train_data: GaussianDistribution(mu, float(cfg["sigma2"])),
        method="evolutionary",
        n_iter=40,
        seed=2,
    )
    assert isinstance(res, SearchResult)
    assert 2.5 < res.best_config["sigma2"] < 6.0


def test_search_reports_search_failed_when_every_config_fails():
    # Finding-3 regression: when EVERY build/score attempt fails, search() used to return a
    # plausible-looking SearchResult (a real sampled best_config, best_model=None, best_score at the
    # internal 1e18 penalty sentinel) with nothing distinguishing it from a real search that merely
    # found a mediocre model. It must now say so explicitly. bo's n_iter=3 stays within its default
    # n_init (no acquisition steps), so this does not require the torch-only GP surrogate.
    def always_raises(cfg, train_data):
        raise RuntimeError("simulated: every config is unbuildable")

    sp = Space({"mu": Real(-1.0, 1.0)})
    data = list(range(20))
    obj = nll_objective()

    for method, n_iter in (("bo", 3), ("evolutionary", 2)):
        res = search(sp, data, objective=obj, build_fn=always_raises, method=method, n_iter=n_iter, seed=0)
        assert isinstance(res, SearchResult)
        assert res.best_model is None, f"method={method!r}"
        assert res.best_config is not None and len(res.best_config) > 0  # still a real sampled config
        assert res.search_failed is True, f"method={method!r}: search_failed must be True"
        assert res.n_evaluations > 0, f"method={method!r}: evaluations were attempted"
        assert res.n_successes == 0, f"method={method!r}: none of them succeeded"


def test_search_does_not_report_search_failed_on_a_normal_run():
    # negative control for Finding 3: an ordinary, fully-successful search must NOT be flagged failed.
    data = list(np.random.RandomState(1).normal(0.0, 2.0, 300))
    mu = float(np.mean(data))
    sp = Space({"sigma2": Real(0.5, 12.0)})
    res = search(
        sp,
        data,
        objective=nll_objective(),
        build_fn=lambda cfg, train_data: GaussianDistribution(mu, float(cfg["sigma2"])),
        method="evolutionary",
        n_iter=5,
        seed=2,
    )
    assert res.search_failed is False
    assert res.best_model is not None
    assert res.n_successes > 0
    assert res.n_successes == res.n_evaluations  # every attempt succeeded -- no exceptions in this build_fn


def test_search_n_iter_zero_rejected_not_silently_run():
    # Finding-4 regression: n_iter=0 used to still spend a real evaluation in both backends (bo forced
    # >=1 initial point via max(1, n_iter); evolutionary always evaluated its `mu` initial parents
    # regardless of n_iter). Zero budget must now be refused explicitly, before any evaluation, rather
    # than silently doing more work than requested.
    sp = Space({"mu": Real(-1.0, 1.0)})
    data = list(range(20))
    obj = nll_objective()
    calls = []

    def counting_build_fn(cfg, train_data):
        calls.append(cfg)
        return GaussianDistribution(float(cfg["mu"]), 1.0)

    for method in ("bo", "evolutionary"):
        calls.clear()
        with pytest.raises(ValueError, match="n_iter"):
            search(sp, data, objective=obj, build_fn=counting_build_fn, method=method, n_iter=0, seed=0)
        assert not calls, f"method={method!r}: n_iter=0 must not spend any evaluation"


def test_search_n_iter_positive_negative_control_exact_evaluation_count():
    # negative control for Finding 4: n_iter>=1 must still run (not over-rejected), spending EXACTLY
    # the documented number of evaluations -- proving the fix is a true zero-budget-only rejection.
    sp = Space({"mu": Real(-1.0, 1.0)})
    data = list(range(20))
    obj = nll_objective()

    calls_bo = []

    def build_fn_bo(cfg, train_data):
        calls_bo.append(cfg)
        return GaussianDistribution(float(cfg["mu"]), 1.0)

    res_bo = search(sp, data, objective=obj, build_fn=build_fn_bo, method="bo", n_iter=1, seed=0)
    assert len(calls_bo) == 1
    assert res_bo.n_evaluations == 1

    calls_evo = []

    def build_fn_evo(cfg, train_data):
        calls_evo.append(cfg)
        return GaussianDistribution(float(cfg["mu"]), 1.0)

    res_evo = search(
        sp, data, objective=obj, build_fn=build_fn_evo, method="evolutionary", n_iter=1, seed=0, mu=2, lam=2
    )
    # MXR-080-1769: n_iter is the TOTAL evaluation budget for the evolutionary backend too, not a
    # generation count. This used to spend mu + lam * n_iter == 4 evaluations under a budget of one.
    assert len(calls_evo) == 1
    assert res_evo.n_evaluations == 1


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
        build_fn=lambda cfg, train_data: GaussianDistribution(float(cfg["mu"]), 1.0),
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
        build_fn=lambda cfg, train_data: GaussianDistribution(float(cfg["mu"]), 1.0),
        method="bandit",
        n_iter=2,
        seed=0,
    )
    assert isinstance(res, SearchResult) and res.best_model is not None


def test_evolutionary_never_exceeds_its_total_evaluation_budget():
    # MXR-080-1769: the evolutionary backend treated n_iter as a generation count and spent
    # mu + lam * n_iter evaluations, so a stated budget of 1 bought 12 with mu=4, lam=8.
    sp = Space({"mu": Real(-1.0, 1.0)})
    data = list(range(20))
    obj = nll_objective()

    for n_iter in (1, 3, 5, 12, 40):
        calls = []

        def build_fn(cfg, train_data, _calls=calls):
            _calls.append(cfg)
            return GaussianDistribution(float(cfg["mu"]), 1.0)

        res = search(
            sp, data, objective=obj, build_fn=build_fn, method="evolutionary", n_iter=n_iter, seed=0, mu=4, lam=8
        )
        assert len(calls) == n_iter, f"n_iter={n_iter}: spent {len(calls)} evaluations"
        assert res.n_evaluations == n_iter


def test_evolutionary_rejects_invalid_population_controls():
    sp = Space({"mu": Real(-1.0, 1.0)})
    data = list(range(20))
    obj = nll_objective()

    def build_fn(cfg, train_data):
        return GaussianDistribution(float(cfg["mu"]), 1.0)

    for bad in (0, -2, 2.5):
        for kwargs in ({"mu": bad}, {"lam": bad}):
            with pytest.raises(ValueError, match="exact positive integer"):
                search(sp, data, objective=obj, build_fn=build_fn, method="evolutionary", n_iter=4, seed=0, **kwargs)
