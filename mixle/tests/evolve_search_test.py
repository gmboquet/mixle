"""mixle.evolve Phase 2-3: typed-space search + the meta-search that learns which operators help."""
import numpy as np

from mixle.evolve import (
    Categorical,
    Integer,
    OperatorBandit,
    Population,
    Real,
    Recompose,
    SearchResult,
    Space,
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


def test_search_evolutionary_finds_variance():
    data = list(np.random.RandomState(1).normal(0.0, 2.0, 300))     # true variance 4
    mu = float(np.mean(data))
    sp = Space({"sigma2": Real(0.5, 12.0)})
    res = search(sp, data, objective=nll_objective(),
                 build_fn=lambda cfg: GaussianDistribution(mu, float(cfg["sigma2"])),
                 method="evolutionary", n_iter=40, seed=2)
    assert isinstance(res, SearchResult)
    assert 2.5 < res.best_config["sigma2"] < 6.0                    # recovers the true ~4


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
    assert ops["good"]["pulls"] > ops["bad"]["pulls"]              # the policy learned which operator helps


def test_population_improves_bad_seed():
    obj = nll_objective()
    seed_model = GaussianDistribution(0.0, 1.0)                     # wrong mean and variance
    data = list(np.random.RandomState(3).normal(5.0, 1.0, 80))
    result = Population([seed_model], objective=obj, seed=0).run(data, generations=2)
    # anti-regression: the evolved champion is never worse than the seed
    assert obj.scalar(result.best_model, data) <= obj.scalar(seed_model, data)


def test_recompose_captures_bimodal_structure():
    rng = np.random.RandomState(0)
    data = list(rng.normal(-4.0, 0.5, 120)) + list(rng.normal(4.0, 0.5, 120))   # clearly bimodal
    champion = GaussianDistribution(0.0, 5.0)                                    # one wide Gaussian misses the modes
    obj = nll_objective()
    op = Recompose()
    assert op.applicable(champion, data, ctx={})
    cand = op.propose(champion, data, ctx={"seed": 0})
    assert obj.scalar(cand.model, data) < obj.scalar(champion, data)            # the 2-component mixture fits better
    verdict = challenger_beats_champion(champion, cand.model, data, objective=obj, nonnested=True)
    assert verdict.favored == "challenger"                                      # and it passes the verify gate


def test_recompose_registered_but_off_by_default():
    assert "recompose" in registered_operators()
    assert "recompose" not in {op.name for op in default_operators()}          # structural + expensive -> opt-in


def test_structural_genotype_distance():
    from mixle.evolve import model_signature, structural_distance
    from mixle.ops import mixture

    g = GaussianDistribution(0.0, 1.0)
    m2 = mixture([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)])
    m3 = mixture([GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)])
    assert model_signature(g) == ("GaussianDistribution", [])
    assert structural_distance(g, g) == 0.0
    assert structural_distance(m2, m3) < structural_distance(g, m3)            # 2-vs-3 comps closer than leaf-vs-3


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
    res = search(sp, data, objective=obj,
                 build_fn=lambda cfg: GaussianDistribution(float(cfg["mu"]), 1.0),
                 method="bandit", n_iter=2, seed=0)
    assert isinstance(res, SearchResult) and res.best_model is not None
