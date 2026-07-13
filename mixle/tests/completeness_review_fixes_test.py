"""Regression tests for the completeness review fixes (ledger F-1, F-2, F-6, F-7, F-9, F-11, F-12).

Each test pins a hole from audit/CODEBASE_REVIEW_LEDGER.md Part V: fitted structured/IO
HMMs that could not be serialized because their TransitionOperator components were not
registered (F-1), an IOHMM you could fit and score but not decode, simulate, or save
(F-2), BetaBinomial opting out of the finite-support contract its Binomial sibling
honors (F-6), enumerator potholes in LogSeries/Skellam/DirichletMultinomial (F-7),
closed-form moment/entropy potholes across the univariate catalog (F-9), the advertised
``waic``/``loo`` names not being importable from ``mixle.ppl`` (F-11), and
ScheduledHMM breaking the prototype -> estimator convention (F-12).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats as sps

from mixle.stats import dump_models, load_models
from mixle.stats.latent.structured_hmm import DenseTransition, InputOutputHMM, StructuredHMM
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def _two_state_emissions() -> list[CategoricalDistribution]:
    return [CategoricalDistribution({"a": 0.8, "b": 0.2}), CategoricalDistribution({"a": 0.3, "b": 0.7})]


# ------------------------------------------------------------- F-1: TransitionOperator serialization
def test_structured_hmm_round_trips_through_dump_models() -> None:
    hmm = StructuredHMM(
        emissions=_two_state_emissions(),
        pi=[0.6, 0.4],
        transition=DenseTransition(np.asarray([[0.9, 0.1], [0.2, 0.8]])),
    )
    seqs = [["a", "b", "b"], ["b", "a"]]
    restored = load_models(dump_models(hmm))
    enc = hmm.dist_to_encoder().seq_encode(seqs)
    enc_r = restored.dist_to_encoder().seq_encode(seqs)
    np.testing.assert_allclose(np.asarray(hmm.seq_log_density(enc)), np.asarray(restored.seq_log_density(enc_r)))


def test_iohmm_round_trips_through_dump_models() -> None:
    trans = [
        DenseTransition(np.asarray([[0.9, 0.1], [0.2, 0.8]])),
        DenseTransition(np.asarray([[0.5, 0.5], [0.5, 0.5]])),
    ]
    io = InputOutputHMM(emissions=_two_state_emissions(), pi=[0.6, 0.4], transitions=trans)
    rows = [[("a", 0), ("b", 1), ("b", 0)], [("b", 1), ("a", 0)]]
    restored = load_models(dump_models(io))
    for row in rows:
        assert restored.log_density(row) == pytest.approx(io.log_density(row))


# ------------------------------------------------------------- F-2: IOHMM read-out parity
def _iohmm_and_row() -> tuple[InputOutputHMM, list[str], list[int]]:
    trans = [
        DenseTransition(np.asarray([[0.95, 0.05], [0.1, 0.9]])),
        DenseTransition(np.asarray([[0.3, 0.7], [0.6, 0.4]])),
    ]
    io = InputOutputHMM(emissions=_two_state_emissions(), pi=[0.6, 0.4], transitions=trans)
    return io, ["a", "b", "b", "a"], [0, 1, 1, 0]


def test_iohmm_viterbi_matches_brute_force() -> None:
    io, seq, inputs = _iohmm_and_row()
    a = [np.asarray([[0.95, 0.05], [0.1, 0.9]]), np.asarray([[0.3, 0.7], [0.6, 0.4]])]
    emis, pi = io.emissions, np.asarray([0.6, 0.4])

    def joint(path: tuple[int, ...]) -> float:
        # step t -> t+1 uses the transition inputs[t] selects (the class's documented convention)
        lp = np.log(pi[path[0]]) + emis[path[0]].log_density(seq[0])
        for t in range(1, len(seq)):
            lp += np.log(a[inputs[t - 1]][path[t - 1], path[t]]) + emis[path[t]].log_density(seq[t])
        return float(lp)

    best = max(itertools.product(range(2), repeat=len(seq)), key=joint)
    got = list(io.viterbi(seq, inputs=inputs))
    assert joint(tuple(got)) == pytest.approx(joint(best))

    gammas = np.asarray(io.state_posteriors(seq, inputs=inputs))
    assert gammas.shape == (len(seq), 2)
    np.testing.assert_allclose(gammas.sum(axis=1), np.ones(len(seq)))
    decoded = list(io.posterior_decode(seq, inputs=inputs))
    np.testing.assert_array_equal(decoded, np.argmax(gammas, axis=1))


def test_iohmm_sampler_draws_conditioned_on_inputs() -> None:
    io, _seq, inputs = _iohmm_and_row()
    draw = io.sampler(seed=5).sample(inputs)
    assert len(draw) == len(inputs)
    obs = [p[0] for p in draw]
    drawn_inputs = [p[1] for p in draw]
    assert set(obs) <= {"a", "b"}
    assert drawn_inputs == list(inputs), "the sampled record must carry the conditioning inputs"


# ------------------------------------------------------------- F-6: BetaBinomial finite-support contract
def test_beta_binomial_support_contract_matches_scipy() -> None:
    from mixle.stats.univariate.discrete.beta_binomial import BetaBinomialDistribution

    d = BetaBinomialDistribution(n=6, a=2.0, b=3.0)
    assert d.support_size() == 7

    pairs = list(itertools.islice(d.enumerator(), 7))
    masses = {int(v): float(np.exp(lp)) for v, lp in pairs}
    assert sum(masses.values()) == pytest.approx(1.0, abs=1e-9)
    probs = [float(np.exp(lp)) for _v, lp in pairs]
    assert probs == sorted(probs, reverse=True), "enumerator must be descending-probability"

    ref = sps.betabinom(n=6, a=2.0, b=3.0)
    for k in range(7):
        assert masses[k] == pytest.approx(float(ref.pmf(k)), rel=1e-9)
        assert d.cdf(k) == pytest.approx(float(ref.cdf(k)), rel=1e-9)
        assert d.quantile(d.cdf(k)) == k


# ------------------------------------------------------------- F-7: enumerator potholes
def test_logseries_and_skellam_enumerators_are_descending_and_consistent() -> None:
    from mixle.stats.univariate.discrete.logseries import LogSeriesDistribution
    from mixle.stats.univariate.discrete.skellam import SkellamDistribution

    for dist, ref_pmf in [
        (LogSeriesDistribution(p=0.4), lambda k: float(sps.logser(0.4).pmf(k))),
        (SkellamDistribution(mu1=2.0, mu2=1.0), lambda k: float(sps.skellam(2.0, 1.0).pmf(k))),
    ]:
        pairs = list(itertools.islice(dist.enumerator(), 25))
        probs = [float(np.exp(lp)) for _v, lp in pairs]
        assert probs == sorted(probs, reverse=True), type(dist).__name__
        assert sum(probs) == pytest.approx(1.0, abs=1e-4), type(dist).__name__
        for v, lp in pairs[:10]:
            assert float(np.exp(lp)) == pytest.approx(ref_pmf(int(v)), rel=1e-8), type(dist).__name__
        values = [v for v, _lp in pairs]
        assert len(values) == len(set(values)), "enumerator must not repeat values"


def test_dirichlet_multinomial_enumerator_masses_sum_to_one() -> None:
    from mixle.stats.multivariate.dirichlet_multinomial import DirichletMultinomialDistribution

    d = DirichletMultinomialDistribution(alpha=[1.5, 2.0, 1.0], n=4)
    pairs = list(d.enumerator())
    total = sum(float(np.exp(lp)) for _v, lp in pairs)
    assert total == pytest.approx(1.0, abs=1e-9)
    probs = [float(np.exp(lp)) for _v, lp in pairs]
    assert probs == sorted(probs, reverse=True)
    for v, lp in pairs[:5]:
        assert lp == pytest.approx(d.log_density(np.asarray(v)), rel=1e-9)


# ------------------------------------------------------------- F-9: closed-form moments/entropy
def test_new_moments_and_entropies_match_scipy() -> None:
    from mixle.stats.univariate.continuous.generalized_extreme_value import (
        GeneralizedExtremeValueDistribution,
    )
    from mixle.stats.univariate.continuous.generalized_pareto import GeneralizedParetoDistribution
    from mixle.stats.univariate.continuous.gumbel import GumbelDistribution
    from mixle.stats.univariate.continuous.half_normal import HalfNormalDistribution
    from mixle.stats.univariate.continuous.inverse_gamma import InverseGammaDistribution
    from mixle.stats.univariate.continuous.inverse_gaussian import InverseGaussianDistribution
    from mixle.stats.univariate.continuous.skew_normal import SkewNormalDistribution
    from mixle.stats.univariate.continuous.student_t import StudentTDistribution
    from mixle.stats.univariate.discrete.poisson import PoissonDistribution

    gum = GumbelDistribution(loc=1.5, scale=2.0)
    ref = sps.gumbel_r(loc=1.5, scale=2.0)
    assert gum.mean() == pytest.approx(float(ref.mean()), rel=1e-12)
    assert gum.variance() == pytest.approx(float(ref.var()), rel=1e-12)

    hn = HalfNormalDistribution(sigma=2.0)
    ref = sps.halfnorm(scale=2.0)
    assert hn.mean() == pytest.approx(float(ref.mean()), rel=1e-12)
    assert hn.variance() == pytest.approx(float(ref.var()), rel=1e-12)

    ig = InverseGammaDistribution(alpha=3.5, beta=2.0)
    ref = sps.invgamma(a=3.5, scale=2.0)
    assert ig.mean() == pytest.approx(float(ref.mean()), rel=1e-12)
    assert ig.variance() == pytest.approx(float(ref.var()), rel=1e-12)

    igauss = InverseGaussianDistribution(mu=1.4, lam=2.5)
    ref = sps.invgauss(mu=1.4 / 2.5, scale=2.5)
    assert igauss.mean() == pytest.approx(float(ref.mean()), rel=1e-12)
    assert igauss.variance() == pytest.approx(float(ref.var()), rel=1e-12)

    sn = SkewNormalDistribution(loc=0.5, scale=1.5, shape=2.0)
    ref = sps.skewnorm(2.0, loc=0.5, scale=1.5)
    assert sn.mean() == pytest.approx(float(ref.mean()), rel=1e-10)
    assert sn.variance() == pytest.approx(float(ref.var()), rel=1e-10)

    st = StudentTDistribution(df=5.0)
    assert st.entropy() == pytest.approx(float(sps.t(5.0).entropy()), rel=1e-10)

    poi = PoissonDistribution(lam=3.7)
    assert poi.entropy() == pytest.approx(float(sps.poisson(3.7).entropy()), rel=1e-6)

    gev = GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, shape=0.2)
    assert gev.entropy() == pytest.approx(float(sps.genextreme(-0.2).entropy()), rel=1e-10)

    gpd = GeneralizedParetoDistribution(scale=2.0, shape=0.3)
    assert gpd.entropy() == pytest.approx(float(sps.genpareto(0.3, scale=2.0).entropy()), rel=1e-10)


# ------------------------------------------------------------- F-11: ppl exports
def test_waic_and_loo_importable_from_ppl() -> None:
    from mixle.ppl import loo, waic

    assert callable(waic) and callable(loo)


# ------------------------------------------------------------- F-12: ScheduledHMM estimator()
def test_scheduled_hmm_prototype_yields_an_estimator() -> None:
    from mixle.stats.latent.scheduled_hidden_markov_model import (
        ByPosition,
        ScheduledHiddenMarkovModelDistribution,
    )

    proto = ScheduledHiddenMarkovModelDistribution(
        inits=np.asarray([[0.5, 0.5], [0.5, 0.5]]),
        transitions=np.asarray([[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]]),
        emissions=[[GaussianDistribution(mu=-1.0, sigma2=1.0), GaussianDistribution(mu=1.0, sigma2=1.0)]] * 2,
        schedule=ByPosition(cap=2),
    )
    est = proto.estimator()  # was: NotImplementedError
    assert est is not None and hasattr(est, "accumulator_factory")
