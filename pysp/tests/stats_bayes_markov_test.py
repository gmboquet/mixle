"""Conjugate-Bayesian parity and additivity tests for the folded pysp.stats Markov models.

Covers MarkovChainDistribution and HiddenMarkovModelDistribution:

* parity with the pysp.bstats reference for matched Dirichlet priors (and matched
  Gaussian/NormalGamma emission priors for the HMM) and identical sufficient statistics --
  conjugate posteriors, estimated init/transition probabilities, expected_log_density (scalar +
  seq), and model_log_density all match to < 1e-9;
* the closed-form posterior Dirichlet counts (prior alpha + observed counts) for MarkovChain;
* the additive guarantee that prior=None leaves the existing MLE path unchanged (expected_log_density
  falls back to log_density and the estimator carries no conjugate prior).
"""

import numpy as np
import pytest

import pysp.bstats.gaussian as bgauss
import pysp.bstats.hidden_markov as bhmm
import pysp.bstats.markov_chain as bmc
import pysp.stats.gaussian as sgauss
import pysp.stats.hidden_markov as shmm
import pysp.stats.markov_chain as smc
from pysp.bstats.dirichlet import DirichletDistribution as Dir
from pysp.bstats.normgamma import NormalGammaDistribution as bNG
from pysp.stats.normgamma import NormalGammaDistribution as sNG

TOL = 1e-9
S = 3
STATES = [0, 1, 2]

INIT_ALPHA = np.array([2.0, 1.5, 3.0])
ROW_ALPHAS = [np.array([1.0, 4.0, 2.0]), np.array([3.0, 1.0, 1.5]), np.array([2.0, 2.0, 2.0])]

INIT_COUNTS = np.array([5.0, 2.0, 3.0])
TRANS_COUNTS = np.array([[4.0, 2.0, 1.0], [0.0, 6.0, 3.0], [2.0, 1.0, 5.0]])
STATE_COUNTS = np.array([15.0, 15.0, 15.0])

W = np.array([0.2, 0.3, 0.5])
TRANS = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.33, 0.33, 0.34]])


def _maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


# -----------------------------------------------------------------------------------------
# MarkovChain
# -----------------------------------------------------------------------------------------
def _mc_pair():
    b_prior = (Dir(INIT_ALPHA), [Dir(a) for a in ROW_ALPHAS])
    s_prior = (STATES, Dir(INIT_ALPHA), [Dir(a) for a in ROW_ALPHAS])
    bd = bmc.MarkovChainDistribution(W, TRANS, prior=b_prior)
    sd = smc.MarkovChainDistribution(
        {i: W[i] for i in range(S)},
        {i: {j: TRANS[i, j] for j in range(S)} for i in range(S)},
        prior=s_prior,
    )
    return bd, sd


def _mc_estimates():
    bd, sd = _mc_pair()
    b_res = bd.estimator().estimate((INIT_COUNTS, TRANS_COUNTS, None))
    s_res = sd.estimator().estimate(
        None,
        (
            {i: INIT_COUNTS[i] for i in range(S)},
            {i: {j: TRANS_COUNTS[i, j] for j in range(S)} for i in range(S)},
            None,
        ),
    )
    return bd, sd, b_res, s_res


def test_markov_chain_estimated_probs_parity():
    _, _, b_res, s_res = _mc_estimates()
    assert _maxdiff(b_res.init_prob_vec, [s_res.init_prob_map[i] for i in range(S)]) < TOL
    assert _maxdiff(b_res.transition_mat, [[s_res.transition_map[i][j] for j in range(S)] for i in range(S)]) < TOL


def test_markov_chain_posterior_alpha_parity():
    _, _, b_res, s_res = _mc_estimates()
    assert _maxdiff(b_res.init_prior.get_parameters(), s_res.init_prior.get_parameters()) < TOL
    assert (
        _maxdiff(
            [rp.get_parameters() for rp in b_res.row_priors],
            [rp.get_parameters() for rp in s_res.row_priors],
        )
        < TOL
    )


def test_markov_chain_posterior_closed_form():
    _, _, _, s_res = _mc_estimates()
    assert _maxdiff(INIT_COUNTS + INIT_ALPHA, s_res.init_prior.get_parameters()) < TOL
    expected = [TRANS_COUNTS[i, :] + ROW_ALPHAS[i] for i in range(S)]
    assert _maxdiff(expected, [rp.get_parameters() for rp in s_res.row_priors]) < TOL


def test_markov_chain_expected_log_density_parity():
    bd, sd = _mc_pair()
    seqs = [[0, 1, 2, 2, 1, 0], [1, 1, 2], [0], [2, 0]]
    assert _maxdiff([bd.expected_log_density(x) for x in seqs], [sd.expected_log_density(x) for x in seqs]) < TOL
    b_seld = bd.seq_expected_log_density(bd.seq_encode(seqs))
    s_seld = sd.seq_expected_log_density(sd.dist_to_encoder().seq_encode(seqs))
    assert _maxdiff(b_seld, s_seld) < TOL


def test_markov_chain_model_log_density_parity():
    bd, sd, b_res, s_res = _mc_estimates()
    assert _maxdiff(bd.estimator().model_log_density(b_res), sd.estimator().model_log_density(s_res)) < TOL


def test_markov_chain_mle_unchanged_without_prior():
    sd = smc.MarkovChainDistribution({0: 0.2, 1: 0.3, 2: 0.5}, {0: {0: 0.5, 1: 0.3, 2: 0.2}})
    est = sd.estimator()
    assert est.has_conj_prior is False
    assert est.get_prior() is None
    # expected_log_density falls back to plug-in log_density byte-for-byte
    for x in ([0, 1, 2], [0], []):
        assert sd.expected_log_density(x) == sd.log_density(x)
    # MLE estimate matches the legacy estimate0 path exactly
    ss = (
        {0: 5.0, 1: 2.0, 2: 3.0},
        {0: {0: 4.0, 1: 2.0, 2: 1.0}, 1: {1: 6.0, 2: 3.0}, 2: {0: 2.0, 1: 1.0, 2: 5.0}},
        None,
    )
    res = est.estimate(None, ss)
    res0 = est.estimate0(None, ss)
    assert res.init_prob_map == res0.init_prob_map
    assert res.transition_map == res0.transition_map
    assert est.model_log_density(res) == 0.0


# -----------------------------------------------------------------------------------------
# Hidden Markov Model
# -----------------------------------------------------------------------------------------
MUS = [-2.0, 0.0, 3.0]
S2S = [1.0, 0.5, 2.0]
NG = [(0.0, 1.0, 2.0, 1.0), (1.0, 2.0, 3.0, 2.0), (-1.0, 0.5, 2.5, 1.5)]


def _hmm_pair():
    b_topics = [bgauss.GaussianDistribution(MUS[i], S2S[i], prior=bNG(*NG[i])) for i in range(S)]
    s_topics = [sgauss.GaussianDistribution(MUS[i], S2S[i], prior=sNG(*NG[i])) for i in range(S)]
    chain = (Dir(INIT_ALPHA), [Dir(a) for a in ROW_ALPHAS])
    bd = bhmm.HiddenMarkovModelDistribution(b_topics, W, TRANS, prior=chain)
    sd = shmm.HiddenMarkovModelDistribution(topics=s_topics, w=W, transitions=TRANS, prior=chain)
    return bd, sd, b_topics, s_topics


def _hmm_estimates():
    bd, sd, b_topics, s_topics = _hmm_pair()
    rng = np.random.RandomState(7)
    emit_data = [list(rng.normal(MUS[i], np.sqrt(S2S[i]), size=15)) for i in range(S)]

    b_topic_stats = []
    for i in range(S):
        acc = b_topics[i].estimator().accumulator_factory().make()
        enc = b_topics[i].seq_encode(emit_data[i])
        acc.seq_update(enc, np.ones(len(emit_data[i])), b_topics[i])
        b_topic_stats.append(acc.value())

    s_topic_stats = []
    for i in range(S):
        acc = s_topics[i].estimator().accumulator_factory().make()
        enc = s_topics[i].dist_to_encoder().seq_encode(emit_data[i])
        acc.seq_update(enc, np.ones(len(emit_data[i])), s_topics[i])
        s_topic_stats.append(acc.value())

    b_res = bd.estimator().estimate((INIT_COUNTS, TRANS_COUNTS, tuple(b_topic_stats), None))
    s_res = sd.estimator().estimate(None, (S, INIT_COUNTS, STATE_COUNTS, TRANS_COUNTS, tuple(s_topic_stats), None))
    return bd, sd, b_res, s_res


def test_hmm_estimated_probs_parity():
    _, _, b_res, s_res = _hmm_estimates()
    assert _maxdiff(b_res.w, s_res.w) < TOL
    assert _maxdiff(b_res.transitions, s_res.transitions) < TOL


def test_hmm_posterior_alpha_parity():
    _, _, b_res, s_res = _hmm_estimates()
    assert _maxdiff(b_res.init_prior.get_parameters(), s_res.init_prior.get_parameters()) < TOL
    assert (
        _maxdiff(
            [rp.get_parameters() for rp in b_res.row_priors],
            [rp.get_parameters() for rp in s_res.row_priors],
        )
        < TOL
    )


def test_hmm_posterior_closed_form():
    _, _, _, s_res = _hmm_estimates()
    assert _maxdiff(INIT_COUNTS + INIT_ALPHA, s_res.init_prior.get_parameters()) < TOL
    expected = [TRANS_COUNTS[i, :] + ROW_ALPHAS[i] for i in range(S)]
    assert _maxdiff(expected, [rp.get_parameters() for rp in s_res.row_priors]) < TOL


def test_hmm_expected_log_density_parity():
    bd, sd, _, _ = _hmm_pair()
    rng = np.random.RandomState(11)
    obs = [list(rng.normal(0, 1, size=n)) for n in [5, 3, 1, 4]]
    assert _maxdiff([bd.expected_log_density(x) for x in obs], [sd.expected_log_density(x) for x in obs]) < TOL
    b_seld = bd.seq_expected_log_density(bd.seq_encode(obs))
    s_seld = sd.seq_expected_log_density(sd.dist_to_encoder().seq_encode(obs))
    assert _maxdiff(b_seld, s_seld) < TOL


def test_hmm_model_log_density_parity():
    bd, sd, b_res, s_res = _hmm_estimates()
    assert _maxdiff(bd.estimator().model_log_density(b_res), sd.estimator().model_log_density(s_res)) < TOL


def test_hmm_expected_log_density_numba_path():
    _, sd, _, _ = _hmm_pair()
    sd.use_numba = True
    data = [[0.1, 1.5, -0.5], [0.2, 2.0], [1.0]]
    enc = sd.dist_to_encoder().seq_encode(data)
    seld = sd.seq_expected_log_density(enc)
    scalar = np.asarray([sd.expected_log_density(x) for x in data])
    assert _maxdiff(seld, scalar) < TOL


def test_hmm_mle_unchanged_without_prior():
    from pysp.stats.poisson import PoissonDistribution

    topics = [sgauss.GaussianDistribution(-1.0, 1.0), sgauss.GaussianDistribution(2.0, 1.0)]
    hd = shmm.HiddenMarkovModelDistribution(
        topics=topics, w=[0.5, 0.5], transitions=[[0.7, 0.3], [0.4, 0.6]], len_dist=PoissonDistribution(4.0)
    )
    est = hd.estimator()
    assert est.has_conj_prior is False
    assert est.get_prior() is None
    x = [0.1, 1.5, -0.5]
    assert hd.expected_log_density(x) == hd.log_density(x)
    enc = hd.dist_to_encoder().seq_encode([x, [0.2, 2.0]])
    assert _maxdiff(hd.seq_expected_log_density(enc), hd.seq_log_density(enc)) == 0.0
    # full EM roundtrip still runs and yields a plain (no-conjugate) model
    data = [hd.sampler(seed=2).sample_seq() for _ in range(50)]
    acc = est.accumulator_factory().make()
    enc2 = hd.dist_to_encoder().seq_encode(data)
    acc.seq_update(enc2, np.ones(len(data)), hd)
    res = est.estimate(None, acc.value())
    assert res.has_conj_prior is False
    assert est.model_log_density(res) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
