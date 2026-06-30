"""mixle.evolve Phase 2-3: typed-space search + the meta-search that learns which operators help."""
import numpy as np

from mixle.evolve import (
    Categorical,
    Integer,
    OperatorBandit,
    Population,
    Real,
    SearchResult,
    Space,
    nll_objective,
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


def test_search_bandit_method_runs():
    obj = nll_objective()
    data = list(np.random.RandomState(4).normal(5.0, 1.0, 80))
    sp = Space({"mu": Real(-2.0, 8.0)})
    res = search(sp, data, objective=obj,
                 build_fn=lambda cfg: GaussianDistribution(float(cfg["mu"]), 1.0),
                 method="bandit", n_iter=2, seed=0)
    assert isinstance(res, SearchResult) and res.best_model is not None
