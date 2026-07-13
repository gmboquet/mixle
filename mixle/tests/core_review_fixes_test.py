"""Regression tests for the 0.8.0 core-module review findings C-1..C-11 (audit/CODEBASE_REVIEW_LEDGER.md II.A).

Each test pins the corrected behavior of a verified defect: the quantize support bracket (C-1), k-best
Viterbi admissibility (C-2), the split-conformal small-n threshold and certificate coverage (C-3..C-5),
registry id uniqueness and tier ordering (C-6/C-7), and the relations/fault edge-case guards (C-8..C-11).
"""

import itertools
import tempfile

import numpy as np
import pytest

from mixle.relations import ViterbiPath, branch_and_bound_milp, cardinality_constrained_milp, tsp_held_karp


# --------------------------------------------------------------- C-1: quantize brackets the SPATIAL quantiles
def test_quantize_gaussian_preserves_mean_variance_and_both_tails():
    from mixle import ops
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    g = GaussianDistribution(mu=0.0, sigma2=1.0)
    q = ops.quantize(g, bits=6)
    vals = np.array(list(q.pmap.keys()), dtype=np.float64)
    probs = np.array([q.pmap[v] for v in q.pmap], dtype=np.float64)
    mean = float((vals * probs).sum())
    var = float((probs * (vals - mean) ** 2).sum())
    # the density_quantile bracket discarded the whole left half (support [0.026, 3.20], mean 0.795)
    assert vals.min() < 0.0 < vals.max()  # support spans both signs
    assert abs(mean) < 0.05
    assert var == pytest.approx(1.0, abs=0.1)


# --------------------------------------------------------------- C-2: ViterbiPath with positive log-densities
def _hmm_path_score(log_init, log_trans, log_obs, path):
    score = log_init[path[0]] + log_obs[0][path[0]]
    for t in range(1, len(log_obs)):
        score += log_trans[path[t - 1]][path[t]] + log_obs[t][path[t]]
    return float(score)


def test_viterbi_top1_matches_brute_force_when_emission_log_densities_are_positive():
    # continuous-emission log-densities are routinely > 0; the old zero heuristic was inadmissible there
    n_states, n_steps = 3, 4
    for seed in range(20):
        rng = np.random.RandomState(seed)
        log_init = np.log(rng.dirichlet(np.ones(n_states)))
        log_trans = np.log([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
        log_obs = rng.uniform(-1.0, 2.0, size=(n_steps, n_states))
        best = max(
            _hmm_path_score(log_init, log_trans, log_obs, path)
            for path in itertools.product(range(n_states), repeat=n_steps)
        )
        sol = ViterbiPath(log_init, log_trans, log_obs).solve()
        assert _hmm_path_score(log_init, log_trans, log_obs, sol.value) == pytest.approx(sol.objective, abs=1e-9)
        assert sol.objective == pytest.approx(best, abs=1e-9), f"suboptimal top-1 at seed {seed}"


def test_viterbi_enumeration_is_nonincreasing_and_k_limits():
    rng = np.random.RandomState(7)
    n_states, n_steps = 3, 4
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.log([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
    log_obs = rng.uniform(-1.0, 2.0, size=(n_steps, n_states))
    rel = ViterbiPath(log_init, log_trans, log_obs)
    scores = [sol.objective for sol in rel.enumerator()]
    assert len(scores) == n_states**n_steps  # k=None enumerates every path
    assert all(scores[i] >= scores[i + 1] - 1e-12 for i in range(len(scores) - 1))
    top = rel.top(5)
    assert len(top) == 5
    brute = sorted(
        (
            _hmm_path_score(log_init, log_trans, log_obs, path)
            for path in itertools.product(range(n_states), repeat=n_steps)
        ),
        reverse=True,
    )
    assert [sol.objective for sol in top] == pytest.approx(brute[:5], abs=1e-9)
    assert all(isinstance(sol.value, list) and len(sol.value) == n_steps for sol in top)


# --------------------------------------------------------------- C-3/C-4/C-5: scientist.study edge regimes
def _two_class_latents(n_per_class=20, dim=3, seed=0):
    rng = np.random.RandomState(seed)
    z = np.concatenate([rng.normal(0.0, 1.0, size=(n_per_class, dim)), rng.normal(5.0, 1.0, size=(n_per_class, dim))])
    y = ["a"] * n_per_class + ["b"] * n_per_class
    return z, y


def test_study_small_calibration_split_yields_infinite_qhat_not_undercoverage():
    from mixle.scientist import study

    z, y = _two_class_latents()
    # n_cal = 10 and alpha = 0.01: ceil((10+1) * 0.99) = 11 > 10, so no calibration score certifies the
    # level -- the finite-sample threshold is +inf (all labels / abstain), not the max score
    model = study(z, y, alpha=0.01, cal_frac=0.25, seed=0)
    assert model.provenance["n_cal"] == 10  # the regime is visible in provenance
    assert np.isinf(model.qhat)
    sets = model.prediction_sets(z[:5])
    assert all(s == model.classes for s in sets)  # every label returned: abstention, not silent under-coverage
    assert model.abstains(z[:5]).all()


def test_study_reachable_conformal_level_keeps_a_finite_threshold():
    from mixle.scientist import study

    z, y = _two_class_latents()
    model = study(z, y, alpha=0.1, cal_frac=0.25, seed=0)  # ceil(11 * 0.9) = 10 <= n_cal = 10
    assert np.isfinite(model.qhat)


def test_study_certificate_covers_every_class_head():
    from mixle.scientist import study

    z, y = _two_class_latents()
    model = study(z, y, alpha=0.1, cal_frac=0.25, seed=0)
    assert len(model.head) == len(model.classes) == 2
    block_names = [b.name for b in model.certificate.blocks]
    for c in model.classes:
        assert any(f"head[{c!r}]" in name for name in block_names), f"class {c!r} missing from the certificate"
    assert model.certificate.guarantee.name == "GLOBAL_UNIQUE"  # closed-form Gaussian heads, K of them
    assert len(model.certificate.gradient_blocks) == 0


def test_study_raises_clearly_when_a_class_misses_the_fit_split():
    from mixle.scientist import study

    rng = np.random.RandomState(0)
    z = np.concatenate([rng.normal(0.0, 1.0, size=(8, 2)), rng.normal(5.0, 1.0, size=(1, 2))])
    y = ["common"] * 8 + ["rare"] * 1
    n_cal = max(1, int(round(0.25 * len(z))))
    seed = next(s for s in range(100) if 8 in np.random.RandomState(s).permutation(len(z))[:n_cal])
    with pytest.raises(ValueError, match="'rare' has no examples in the fit split"):
        study(z, y, alpha=0.1, cal_frac=0.25, seed=seed)


# --------------------------------------------------------------- C-6/C-7: registry ids and tier ordering
def _json_task_model():
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
    from mixle.task.model import StructuredClassifierIO, TaskModel

    return TaskModel(
        model=CategoricalDistribution({"x": 0.5, "y": 0.5}),
        adapter=StructuredClassifierIO(field_keys=None, label_index=0, labels=["x", "y"]),
        payload="json",
    )


def test_registry_rejects_duplicate_entry_id_instead_of_overwriting():
    from mixle.registry import Registry

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(d)
        reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id="dup")
        with pytest.raises(ValueError, match="already has an entry 'dup'"):
            reg.register(_json_task_model(), capabilities=["cap"], cost=0.02, entry_id="dup")
        assert [e.entry_id for e in reg.find_for("cap")] == ["dup"]  # one index row, artifact intact


def test_registry_auto_ids_scan_past_taken_ones():
    from mixle.registry import Registry

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(d)
        reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id="entry_0001")
        # len()-based naming would now mint 'entry_0001' again and silently overwrite its artifact
        auto = reg.register(_json_task_model(), capabilities=["cap"], cost=0.02)
        ids = [e.entry_id for e in reg.find_for("cap")]
        assert auto.entry_id != "entry_0001"
        assert len(ids) == len(set(ids)) == 2
        reg.load(auto.entry_id)  # both artifacts reload
        reg.load("entry_0001")


def test_tier_stack_reordering_costs_override_yields_ascending_tiers():
    from mixle.registry import Registry

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(d)
        first = reg.register(_json_task_model(), capabilities=["cap"], cost=0.01)
        second = reg.register(_json_task_model(), capabilities=["cap"], cost=0.05)

        def frontier(texts):
            return ["x"] * len(texts)

        # positional override [0.09, 0.02, 1.0] swaps the pool's effective order (Router assumes ascending)
        stack = reg.tier_stack("cap", frontier=frontier, costs=[0.09, 0.02, 1.0])
        assert [cost for _name, _model, cost in stack] == [0.02, 0.09, 1.0]
        assert [name for name, _model, _cost in stack] == [second.entry_id, first.entry_id, "frontier"]
        assert stack[-1][1] is frontier
        # without an override the registered-cost order is unchanged
        stack = reg.tier_stack("cap", frontier=frontier)
        assert [name for name, _model, _cost in stack] == [first.entry_id, second.entry_id, "frontier"]


# --------------------------------------------------------------- C-8: tsp_held_karp infeasibility
def test_tsp_raises_clearly_when_no_hamiltonian_cycle_exists():
    inf = np.inf
    disconnected = np.array([[0.0, 1.0, inf], [1.0, 0.0, inf], [inf, inf, 0.0]])
    with pytest.raises(ValueError, match="no Hamiltonian cycle"):
        tsp_held_karp(disconnected)
    with pytest.raises(ValueError, match="no Hamiltonian cycle"):
        tsp_held_karp(np.array([[0.0, inf], [1.0, 0.0]]))
    cost, tour = tsp_held_karp(np.array([[0.0, 1.0, 4.0], [4.0, 0.0, 1.0], [1.0, 4.0, 0.0]]))  # feasible ring
    assert cost == pytest.approx(3.0)
    assert tour == [0, 1, 2]


# --------------------------------------------------------------- C-9: cardinality MILP bound validation
def test_cardinality_milp_rejects_nonfinite_bounds():
    with pytest.raises(ValueError, match="finite bounds"):
        cardinality_constrained_milp(np.array([1.0, 1.0]), None, None, 1, [(0.0, np.inf), (0.0, 1.0)], sense="min")
    res = cardinality_constrained_milp(np.array([-1.0, -2.0]), None, None, 1, [(0.0, 1.0), (0.0, 1.0)], sense="min")
    assert res is not None
    value, x = res
    assert value == pytest.approx(-2.0)
    assert x == pytest.approx([0.0, 1.0])


# --------------------------------------------------------------- C-10: route_past guards
def test_route_past_guards_empty_tiers_and_length_mismatch():
    from mixle.fault import route_past

    with pytest.raises(ValueError, match="at least one tier"):
        route_past([])
    with pytest.raises(ValueError, match="must match one-to-one"):
        route_past([lambda: 1, lambda: 2], names=["only_one"])
    ok = route_past([lambda: 1])
    assert (ok.value, ok.degraded) == (1, False)

    def boom():
        raise RuntimeError("tier down")

    routed = route_past([boom, lambda: 2], names=["primary", "backup"])
    assert (routed.value, routed.degraded, routed.mode) == (2, True, "model_error")


# --------------------------------------------------------------- C-11: MILP incumbent exactness
def test_milp_incumbent_integer_coordinates_are_exact_ints():
    # the LP optimum 0.9999994 is within tol of 1 and must be snapped, not stored raw
    res = branch_and_bound_milp(
        np.array([-1.0]), np.array([[2.0]]), np.array([1.9999988]), integer=[0], bounds=[(0.0, 10.0)], sense="min"
    )
    assert res is not None
    _value, x = res
    assert float(x[0]) == 1.0  # exact, so downstream int(x[0]) cannot truncate to 0


def test_milp_accepts_a_generator_for_integer_indices():
    c = np.array([-1.0, -1.0])
    a_ub = np.array([[3.0, 2.0]])
    b_ub = np.array([7.0])
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    from_list = branch_and_bound_milp(c, a_ub, b_ub, integer=[0, 1], bounds=bounds)
    from_gen = branch_and_bound_milp(c, a_ub, b_ub, integer=(i for i in range(2)), bounds=bounds)
    assert from_list is not None and from_gen is not None
    assert from_gen[0] == pytest.approx(from_list[0])
    assert from_gen[1] == pytest.approx(from_list[1])
    assert all(float(v).is_integer() for v in from_gen[1])
