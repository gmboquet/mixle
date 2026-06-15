"""Conjugate-Bayesian closed-form and additivity tests for the folded pysp.stats Markov models.

Covers MarkovChainDistribution and HiddenMarkovModelDistribution:

* the closed-form posterior Dirichlet counts (prior alpha + observed counts) for both the initial
  and per-row transition priors;
* the Dirichlet-MAP estimated init/transition probabilities (boundary-clamped mode of the posterior);
* expected_log_density scalar-vs-seq self-consistency (including the HMM numba path);
* model_log_density self-consistency;
* the additive guarantee that prior=None leaves the existing MLE path unchanged (expected_log_density
  falls back to log_density and the estimator carries no conjugate prior).

The conjugate Dirichlet priors are the pysp.stats DirichletDistribution.
"""

import numpy as np
import pytest

import pysp.stats.graph.markov_chain as smc
import pysp.stats.latent.hidden_markov as shmm
import pysp.stats.leaf.gaussian as sgauss
from pysp.stats.bayes.dirichlet import DirichletDistribution as Dir
from pysp.stats.bayes.normgamma import NormalGammaDistribution as sNG

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


def _map_probs(counts, alpha):
    """Boundary-clamped Dirichlet MAP; posterior mean when degenerate (matches stats _map_probs)."""
    num = np.maximum(counts + alpha - 1.0, 0.0)
    tot = num.sum()
    if tot > 0:
        return num / tot
    cpp = counts + alpha
    return cpp / cpp.sum()


# -----------------------------------------------------------------------------------------
# MarkovChain
# -----------------------------------------------------------------------------------------
def _mc_dist():
    s_prior = (STATES, Dir(INIT_ALPHA), [Dir(a) for a in ROW_ALPHAS])
    return smc.MarkovChainDistribution(
        {i: W[i] for i in range(S)},
        {i: {j: TRANS[i, j] for j in range(S)} for i in range(S)},
        prior=s_prior,
    )


def _mc_estimate():
    sd = _mc_dist()
    s_res = sd.estimator().estimate(
        None,
        (
            {i: INIT_COUNTS[i] for i in range(S)},
            {i: {j: TRANS_COUNTS[i, j] for j in range(S)} for i in range(S)},
            None,
        ),
    )
    return sd, s_res


def test_markov_chain_estimated_probs_closed_form():
    _, s_res = _mc_estimate()
    # init probs are the Dirichlet MAP of (init counts + alpha)
    assert _maxdiff(_map_probs(INIT_COUNTS, INIT_ALPHA), [s_res.init_prob_map[i] for i in range(S)]) < TOL
    # each transition row is the Dirichlet MAP of (row counts + row alpha)
    expected_trans = [_map_probs(TRANS_COUNTS[i, :], ROW_ALPHAS[i]) for i in range(S)]
    got_trans = [[s_res.transition_map[i][j] for j in range(S)] for i in range(S)]
    assert _maxdiff(expected_trans, got_trans) < TOL


def test_markov_chain_posterior_closed_form():
    _, s_res = _mc_estimate()
    assert _maxdiff(INIT_COUNTS + INIT_ALPHA, s_res.init_prior.get_parameters()) < TOL
    expected = [TRANS_COUNTS[i, :] + ROW_ALPHAS[i] for i in range(S)]
    assert _maxdiff(expected, [rp.get_parameters() for rp in s_res.row_priors]) < TOL


def test_markov_chain_expected_log_density_seq_matches_scalar():
    sd = _mc_dist()
    seqs = [[0, 1, 2, 2, 1, 0], [1, 1, 2], [0], [2, 0]]
    scalar = [sd.expected_log_density(x) for x in seqs]
    s_seld = sd.seq_expected_log_density(sd.dist_to_encoder().seq_encode(seqs))
    assert _maxdiff(scalar, s_seld) < TOL


def test_markov_chain_model_log_density_finite():
    sd, s_res = _mc_estimate()
    mld = sd.estimator().model_log_density(s_res)
    assert np.isfinite(mld)


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
    s_topics = [sgauss.GaussianDistribution(MUS[i], S2S[i], prior=sNG(*NG[i])) for i in range(S)]
    chain = (Dir(INIT_ALPHA), [Dir(a) for a in ROW_ALPHAS])
    sd = shmm.HiddenMarkovModelDistribution(topics=s_topics, w=W, transitions=TRANS, prior=chain)
    return sd, s_topics


def _hmm_estimate():
    sd, s_topics = _hmm_pair()
    rng = np.random.RandomState(7)
    emit_data = [list(rng.normal(MUS[i], np.sqrt(S2S[i]), size=15)) for i in range(S)]

    s_topic_stats = []
    for i in range(S):
        acc = s_topics[i].estimator().accumulator_factory().make()
        enc = s_topics[i].dist_to_encoder().seq_encode(emit_data[i])
        acc.seq_update(enc, np.ones(len(emit_data[i])), s_topics[i])
        s_topic_stats.append(acc.value())

    s_res = sd.estimator().estimate(None, (S, INIT_COUNTS, STATE_COUNTS, TRANS_COUNTS, tuple(s_topic_stats), None))
    return sd, s_res


def test_hmm_estimated_probs_closed_form():
    _, s_res = _hmm_estimate()
    # initial weights are the Dirichlet MAP of (init counts + init alpha)
    assert _maxdiff(_map_probs(INIT_COUNTS, INIT_ALPHA), s_res.w) < TOL
    # each transition row is the Dirichlet MAP of (row counts + row alpha)
    expected_trans = np.array([_map_probs(TRANS_COUNTS[i, :], ROW_ALPHAS[i]) for i in range(S)])
    assert _maxdiff(expected_trans, s_res.transitions) < TOL


def test_hmm_posterior_closed_form():
    _, s_res = _hmm_estimate()
    assert _maxdiff(INIT_COUNTS + INIT_ALPHA, s_res.init_prior.get_parameters()) < TOL
    expected = [TRANS_COUNTS[i, :] + ROW_ALPHAS[i] for i in range(S)]
    assert _maxdiff(expected, [rp.get_parameters() for rp in s_res.row_priors]) < TOL


def test_hmm_expected_log_density_seq_matches_scalar():
    sd, _ = _hmm_pair()
    rng = np.random.RandomState(11)
    obs = [list(rng.normal(0, 1, size=n)) for n in [5, 3, 1, 4]]
    scalar = [sd.expected_log_density(x) for x in obs]
    s_seld = sd.seq_expected_log_density(sd.dist_to_encoder().seq_encode(obs))
    assert _maxdiff(scalar, s_seld) < TOL


def test_hmm_model_log_density_finite():
    sd, s_res = _hmm_estimate()
    mld = sd.estimator().model_log_density(s_res)
    assert np.isfinite(mld)


def test_hmm_expected_log_density_numba_path():
    sd, _ = _hmm_pair()
    sd.use_numba = True
    data = [[0.1, 1.5, -0.5], [0.2, 2.0], [1.0]]
    enc = sd.dist_to_encoder().seq_encode(data)
    seld = sd.seq_expected_log_density(enc)
    scalar = np.asarray([sd.expected_log_density(x) for x in data])
    assert _maxdiff(seld, scalar) < TOL


def test_hmm_mle_unchanged_without_prior():
    from pysp.stats.leaf.poisson import PoissonDistribution

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
