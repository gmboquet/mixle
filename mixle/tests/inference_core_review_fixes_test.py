"""Regression tests for the inference-core review fixes (ledger I-1, I-3, I-6..I-9, I-11).

Each test pins a defect from audit/CODEBASE_REVIEW_LEDGER.md: the auto-schedule
objective scoring all-impossible rows as probability one (I-1), keyed (tied)
parameters silently untied by the posterior-transform strategies (I-3) and the
heterogeneous executor (I-7), HMM conditioning that crashed past the evidence
horizon and multiplied marginals while labeled exact (I-6), the no-op max-edge
clamp in spatial block folds (I-8), the discarded independent-composite fit on
the auto-structure path (I-9), and Vuong statistics exploding on numerically
indistinguishable models (I-11).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mixle.inference import optimize
from mixle.inference.condition import condition
from mixle.inference.cross_validation import spatial_block_kfold
from mixle.inference.em import PosteriorTransformEM, StandardEM
from mixle.inference.freeze_rollup import _combine
from mixle.inference.heterogeneous_executor import heterogeneous_em_step
from mixle.inference.model_comparison import vuong_test
from mixle.stats import seq_encode
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
from mixle.stats.latent.mixture import MixtureDistribution, MixtureEstimator
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator


# --------------------------------------------------------------- I-1: impossible rows in _combine
def test_combine_scores_all_impossible_rows_minus_inf() -> None:
    log_w = np.log(np.asarray([0.5, 0.5]))
    ll, _gamma = _combine(np.asarray([[-np.inf, -np.inf]]), log_w)
    assert ll[0] == -np.inf

    mix = MixtureDistribution([CategoricalDistribution({"a": 1.0}), CategoricalDistribution({"b": 1.0})], [0.5, 0.5])
    enc = mix.dist_to_encoder().seq_encode(["a", "b", "z"])
    comp = np.column_stack([np.asarray(c.seq_log_density(e)) for c, e in zip(mix.components, itertools.repeat(enc))])
    ll, _gamma = _combine(comp, np.log(np.asarray(mix.w)))
    np.testing.assert_allclose(ll, np.asarray(mix.seq_log_density(enc)))


def test_auto_schedule_never_returns_an_impossible_model() -> None:
    data = ["a"] * 60 + ["b"] * 39 + ["z"]
    est = MixtureEstimator([CategoricalEstimator(), CategoricalEstimator()])
    for seed in range(3):
        rng_full = np.random.RandomState(seed)
        rng_auto = np.random.RandomState(seed)
        full = optimize(data, est, max_its=8, rng=rng_full, out=None, schedule="full")
        auto = optimize(data, est, max_its=8, rng=rng_auto, out=None, schedule="auto")
        enc = auto.dist_to_encoder().seq_encode(data)
        ll_auto = float(np.sum(np.asarray(auto.seq_log_density(enc))))
        enc_f = full.dist_to_encoder().seq_encode(data)
        ll_full = float(np.sum(np.asarray(full.seq_log_density(enc_f))))
        assert np.isfinite(ll_auto), f"seed {seed}: auto schedule returned an impossible model"
        assert ll_auto >= ll_full - 2.0, (seed, ll_auto, ll_full)


# --------------------------------------------------------------- I-3: keyed tying in strategies
def _keyed_mixture_setup() -> tuple[list[float], MixtureEstimator, MixtureDistribution, object]:
    rng = np.random.RandomState(0)
    data = list(np.concatenate([rng.normal(-2.0, 1.0, size=80), rng.normal(2.0, 1.5, size=80)]))
    # one shared merge key pools the components' entire Gaussian sufficient statistics: after a
    # correct (tying-aware) M-step both components must estimate identical parameters
    est = MixtureEstimator([GaussianEstimator(keys="pooled"), GaussianEstimator(keys="pooled")])
    model = MixtureDistribution(
        [GaussianDistribution(mu=-1.0, sigma2=1.0), GaussianDistribution(mu=1.0, sigma2=2.0)], [0.5, 0.5]
    )
    enc = seq_encode(data, model=model)
    return data, est, model, enc


def test_posterior_transform_em_ties_keyed_parameters() -> None:
    _data, est, model, enc = _keyed_mixture_setup()
    std = StandardEM().step(enc, est, model).model
    soft = PosteriorTransformEM(temperature=1.0).step(enc, est, model).model
    assert soft.components[0].sigma2 == pytest.approx(soft.components[1].sigma2), "keyed stats must pool"
    assert soft.components[0].mu == pytest.approx(soft.components[1].mu)
    assert soft.components[0].sigma2 == pytest.approx(std.components[0].sigma2)
    assert soft.components[0].mu == pytest.approx(std.components[0].mu)


# --------------------------------------------------------------- I-7: keyed tying in the executor
def test_heterogeneous_executor_matches_serial_for_keyed_models() -> None:
    data, est, model, enc = _keyed_mixture_setup()
    serial = StandardEM().step(enc, est, model).model
    dist = heterogeneous_em_step(est, model, data, n_shards=1)
    assert dist.components[0].sigma2 == pytest.approx(dist.components[1].sigma2), "executor must run the tying pass"
    np.testing.assert_allclose([c.sigma2 for c in dist.components], [c.sigma2 for c in serial.components], rtol=1e-9)
    np.testing.assert_allclose([c.mu for c in dist.components], [c.mu for c in serial.components], rtol=1e-9)


# --------------------------------------------------------------- I-6: HMM conditioning
def _tiny_hmm() -> HiddenMarkovModelDistribution:
    emis = [CategoricalDistribution({"a": 0.8, "b": 0.2}), CategoricalDistribution({"a": 0.3, "b": 0.7})]
    return HiddenMarkovModelDistribution(topics=emis, w=[0.6, 0.4], transitions=[[0.9, 0.1], [0.2, 0.8]])


def _brute_conditional(hmm: HiddenMarkovModelDistribution, evidence: dict[int, str], query: dict[int, str]) -> float:
    horizon = max(max(evidence), max(query)) + 1
    emis, w, trans = hmm.topics, np.asarray(hmm.w), np.asarray(hmm.transitions)

    def total(obs: dict[int, str]) -> float:
        z = 0.0
        for path in itertools.product(range(2), repeat=horizon):
            p = w[path[0]] * np.prod([trans[path[t - 1], path[t]] for t in range(1, horizon)])
            for t, val in obs.items():
                p *= np.exp(emis[path[t]].log_density(val))
            z += p
        return float(np.log(z))

    return total({**evidence, **query}) - total(evidence)


def test_condition_hmm_queries_past_the_evidence_horizon() -> None:
    post = condition(_tiny_hmm(), {0: "a"}, method="exact")
    val = post.log_density({2: "b"})  # was: raw IndexError
    assert np.isfinite(val)
    assert val == pytest.approx(_brute_conditional(_tiny_hmm(), {0: "a"}, {2: "b"}), abs=1e-9)


def test_condition_hmm_multi_field_density_is_the_joint_not_product_of_marginals() -> None:
    hmm = _tiny_hmm()
    post = condition(hmm, {0: "a"}, method="exact")
    query = {1: "b", 2: "b"}
    got = post.log_density(query)
    want = _brute_conditional(hmm, {0: "a"}, query)
    assert got == pytest.approx(want, abs=1e-9)
    marginal_sum = post.log_density({1: "b"}) + post.log_density({2: "b"})
    assert abs(got - marginal_sum) > 1e-6, "joint must differ from the product of marginals here"


# --------------------------------------------------------------- I-8: spatial block max edge
def test_spatial_block_kfold_max_edge_points_join_the_last_cell() -> None:
    coords = np.asarray([[0.0, 0.0], [0.25, 0.25], [0.75, 0.75], [1.0, 1.0]])
    folds = spatial_block_kfold(coords, n_splits=2, n_side=2, seed=0)
    for train_idx, test_idx in folds:
        both = np.concatenate([train_idx, test_idx])
        assert sorted(both.tolist()) == [0, 1, 2, 3]
        # the exact-max-edge point (1,1) must fall in the same block as the interior (0.75,0.75)
        same_side = (2 in train_idx and 3 in train_idx) or (2 in test_idx and 3 in test_idx)
        assert same_side, (train_idx, test_idx)


# --------------------------------------------------------------- I-9: composite reuse contract
def test_maybe_structured_model_returns_the_fitted_composite_for_reuse() -> None:
    from mixle.inference.estimation import _maybe_structured_model

    rng = np.random.RandomState(1)
    rows = [(float(a), float(b)) for a, b in zip(rng.normal(size=60), rng.normal(size=60))]
    structured, composite = _maybe_structured_model(rows, 5, None, np.random.RandomState(0))
    if composite is not None:  # the BIC gate paid for the composite fit: it must be reusable
        assert np.isfinite(float(composite.log_density(rows[0])))
    assert structured is None or hasattr(structured, "log_density")


# --------------------------------------------------------------- I-11: Vuong degenerate variance
def test_vuong_flags_indistinguishable_models_instead_of_exploding() -> None:
    lla = np.random.RandomState(3).normal(loc=-1.0, size=200)
    res = vuong_test(lla, lla - 0.05)  # pointwise ratios exactly constant
    assert res["indistinguishable"] is True
    assert res["statistic"] == 0.0 and res["p_value"] == 1.0 and res["favored"] == "tie"

    res2 = vuong_test(lla, lla + np.random.RandomState(4).normal(scale=0.5, size=200))
    assert res2["indistinguishable"] is False
    assert np.isfinite(res2["statistic"]) and abs(res2["statistic"]) < 1e6
