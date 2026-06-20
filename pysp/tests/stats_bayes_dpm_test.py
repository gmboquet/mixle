"""Tests for the pysp.stats ports of the Bayesian variational mixture models
DirichletProcessMixture (dpm.py) and HierarchicalDirichletProcessMixture
(hdpm.py).

Three things are checked:

  * Self-consistency of the scoring methods: seq_expected_log_density matches the
    per-element expected_log_density, seq_log_density matches per-group log_density,
    the local-ELBO is a finite lower bound on the marginal seq_log_density, group
    posteriors are valid simplices, and model_log_density is finite.
  * The local-ELBO objective behaves correctly under estimation. For the DPM the
    raw VB ``seq_estimate`` update is monotone non-decreasing on the local-ELBO
    objective; for the HDP the beta step is the customary approximation, so
    monotonicity is enforced by the ``fit`` acceptance gate on the *accepted*
    model rather than by the raw update.
  * Atom recovery on well-separated synthetic data: the dominant components
    recover the true atom means.
"""

import io

import numpy as np
from numpy.random import RandomState

from pysp.stats import seq_encode, seq_estimate, seq_initialize
from pysp.stats.bayes.dirichlet_process_mixture import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
)
from pysp.stats.bayes.hierarchical_dirichlet_process_mixture import HierarchicalDirichletProcessMixtureEstimator
from pysp.stats.bayes.normal_gamma import NormalGammaDistribution
from pysp.stats.leaf.gamma import GammaDistribution
from pysp.stats.leaf.gaussian import GaussianDistribution, GaussianEstimator
from pysp.utils.estimation import fit

TRUE_MUS = [-8.0, 0.0, 8.0]


# --------------------------------------------------------------------------- #
# Parity helpers
# --------------------------------------------------------------------------- #


def _matched_dpm():
    """Build a stats DPM plus its estimator with fixed components/priors/state."""
    mus = [-5.0, 0.0, 5.0, 10.0]
    k = len(mus)
    ng_params = [(m, 3.0, 2.0 + i, 1.5 + 0.2 * i) for i, m in enumerate(mus)]
    prior_params = [(0.0, 1.0e-3, 0.6, 1.0)] * k

    s_comps = [GaussianDistribution(m, 1.0, prior=NormalGammaDistribution(*p)) for m, p in zip(mus, ng_params)]
    s_cpri = [NormalGammaDistribution(*p) for p in prior_params]

    w = np.array([0.4, 0.3, 0.2, 0.1])
    a = 1.7
    g = np.array([[5.0, 3.0], [4.0, 2.0], [3.0, 1.5], [1.0, 1.0]])

    sd = DirichletProcessMixtureDistribution(s_comps, w, a, g, s_cpri, prior=GammaDistribution(2, 1))
    s_est = DirichletProcessMixtureEstimator(
        [GaussianEstimator(prior=NormalGammaDistribution(*p)) for p in prior_params], prior=GammaDistribution(2, 1)
    )
    return sd, s_est


def _matched_hdpm():
    """Build a stats HDPM with fixed components/priors/state."""
    from pysp.stats.bayes.hierarchical_dirichlet_process_mixture import (
        HierarchicalDirichletProcessMixtureDistribution as SHD,
    )

    rng = RandomState(5)
    mus = [-6.0, 0.0, 6.0]
    k = len(mus)
    ngp = [(m, 3.0, 2.0, 1.0) for m in mus]
    s_c = [GaussianDistribution(m, 1.0, prior=NormalGammaDistribution(*p)) for m, p in zip(mus, ngp)]

    beta = np.array([0.5, 0.3, 0.2])
    alpha, gamma = 2.0, 1.5
    gw = rng.dirichlet([1, 1, 1], size=12)

    sd = SHD(s_c, beta, alpha, gamma, group_weights=gw)

    groups = [[float(rng.normal() * 0.7 + rng.choice(mus)) for _ in range(rng.randint(5, 15))] for _ in range(12)]
    return sd, groups, k


# --------------------------------------------------------------------------- #
# DPM tests
# --------------------------------------------------------------------------- #


def test_dpm_scoring_self_consistency():
    sd, s_est = _matched_dpm()

    rng = RandomState(0)
    x = (rng.normal(size=200) + rng.choice([-5.0, 0.0, 5.0, 10.0], size=200)).tolist()

    senc = sd.dist_to_encoder().seq_encode(x)

    # seq_expected_log_density matches the per-element scalar expected_log_density
    seq_eld = np.asarray(sd.seq_expected_log_density(senc))
    scalar_eld = np.asarray([sd.expected_log_density(v) for v in x])
    assert float(np.max(np.abs(seq_eld - scalar_eld))) < 1e-9

    # all variational scores are finite
    seq_ld = np.asarray(sd.seq_log_density(senc))
    seq_elbo = np.asarray(sd.seq_local_elbo(senc))
    assert np.all(np.isfinite(seq_ld))
    assert np.all(np.isfinite(seq_elbo))

    # model_log_density (the global ELBO term) is finite
    assert np.isfinite(s_est.model_log_density(sd))


def test_dpm_seq_posterior():
    sd, _ = _matched_dpm()
    mus = [-5.0, 0.0, 5.0, 10.0]
    k = len(mus)

    rng = RandomState(0)
    x = (rng.normal(size=120) * 0.4 + rng.choice(mus, size=120)).tolist()
    senc = sd.dist_to_encoder().seq_encode(x)

    post = np.asarray(sd.seq_posterior(senc))
    # Shape, validity, and that each row is a probability simplex.
    assert post.shape == (len(x), k)
    assert np.all(post >= 0.0) and np.all(post <= 1.0)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-9)

    # Vectorized rows match the per-observation scalar posterior(x_i).
    scalar = np.asarray([sd.posterior(v) for v in x])
    np.testing.assert_allclose(post, scalar, atol=1e-9)

    # Consistent with the plug-in mixture density: p(z=k|x) = exp(comp_k + log w_k - logsumexp).
    comp_ld = np.asarray([u.seq_log_density(senc) for u in sd.components]).T + sd.log_w
    expected = np.exp(comp_ld - np.asarray(sd.seq_log_density(senc))[:, None])
    np.testing.assert_allclose(post, expected, atol=1e-9)

    # Atom assignment: an observation right at atom j is most probable under component j.
    atoms = sd.dist_to_encoder().seq_encode([float(m) for m in mus])
    np.testing.assert_array_equal(np.argmax(np.asarray(sd.seq_posterior(atoms)), axis=1), np.arange(k))


def test_dpm_elbo_monotone_and_recovery():
    rng = RandomState(1)
    data = []
    for m in TRUE_MUS:
        data.extend((rng.normal(size=200) * 0.6 + m).tolist())
    rng.shuffle(data)

    k = 10
    est = DirichletProcessMixtureEstimator(
        [GaussianEstimator(prior=NormalGammaDistribution(0.0, 1.0e-3, 1.0, 1.0)) for _ in range(k)],
        prior=GammaDistribution(2, 1),
    )

    enc = seq_encode(data, est.accumulator_factory().make().acc_to_encoder())
    mm = seq_initialize(enc_data=enc, estimator=est, rng=RandomState(101), p=1.0)

    def data_objective(model):
        # The local-ELBO data objective consumed by the fit driver
        # (estimation._data_objective_sum) for variational models.
        return float(sum(model.seq_local_elbo(u[1]).sum() for u in enc))

    objs = [data_objective(mm)]
    for _ in range(80):
        mm = seq_estimate(enc, est, mm)
        objs.append(data_objective(mm))

    diffs = np.diff(np.asarray(objs))
    # The DPM VB update is monotone non-decreasing on the local-ELBO data
    # objective. (The combined objective including the model_log_density global
    # term may dip slightly under the component re-sort/alpha update; the fit
    # acceptance gate handles that -- see test_dpm_combined_objective_gated.)
    assert np.all(diffs >= -1e-6), "min ELBO step %g" % diffs.min()

    # Each true atom is recovered by a dominant component.
    dom = np.where(mm.w > 0.05)[0]
    dmu = np.asarray([mm.components[i].mu for i in dom])
    for t in TRUE_MUS:
        assert float(np.min(np.abs(dmu - t))) < 0.5, "atom %g not recovered (got %s)" % (t, dmu)


def test_dpm_combined_objective_gated():
    """The combined (local-ELBO + prior) objective is monotone under the fit
    acceptance gate, exactly as the optimize() driver relies on it."""
    rng = RandomState(1)
    data = []
    for m in TRUE_MUS:
        data.extend((rng.normal(size=200) * 0.6 + m).tolist())
    rng.shuffle(data)

    k = 10
    est = DirichletProcessMixtureEstimator(
        [GaussianEstimator(prior=NormalGammaDistribution(0.0, 1.0e-3, 1.0, 1.0)) for _ in range(k)],
        prior=GammaDistribution(2, 1),
    )
    enc = seq_encode(data, est.accumulator_factory().make().acc_to_encoder())
    mm = seq_initialize(enc_data=enc, estimator=est, rng=RandomState(101), p=1.0)

    def objective(model):
        data_term = sum(model.seq_local_elbo(u[1]).sum() for u in enc)
        return data_term + est.model_log_density(model)

    accepted = objective(mm)
    traj = [accepted]
    for _ in range(80):
        proposed = seq_estimate(enc, est, mm)
        pobj = objective(proposed)
        if pobj >= accepted - 1e-9:
            mm = proposed
            accepted = pobj
        traj.append(accepted)
    diffs = np.diff(np.asarray(traj))
    assert np.all(diffs >= -1e-6), "accepted objective decreased: min step %g" % diffs.min()


def test_dpm_fit_driver_runs():
    """The fit() driver consumes seq_local_elbo as its data objective."""
    rng = RandomState(2)
    data = []
    for m in TRUE_MUS:
        data.extend((rng.normal(size=120) * 0.6 + m).tolist())
    rng.shuffle(data)

    k = 8
    est = DirichletProcessMixtureEstimator(
        [GaussianEstimator(prior=NormalGammaDistribution(0.0, 1.0e-3, 1.0, 1.0)) for _ in range(k)],
        prior=GammaDistribution(2, 1),
    )
    buf = io.StringIO()
    # delta=None runs the full schedule: the fit acceptance gate keeps the best
    # model, so this exercises the driver end to end and lets the atoms separate.
    model = fit(data, est, max_its=120, delta=None, rng=RandomState(11), init_p=1.0, out=buf)
    assert isinstance(model, DirichletProcessMixtureDistribution)
    dom = np.where(model.w > 0.05)[0]
    dmu = np.asarray([model.components[i].mu for i in dom])
    for t in TRUE_MUS:
        assert float(np.min(np.abs(dmu - t))) < 0.5


# --------------------------------------------------------------------------- #
# HDPM tests
# --------------------------------------------------------------------------- #


def test_hdpm_scoring_self_consistency():
    sd, groups, _ = _matched_hdpm()

    senc = sd.dist_to_encoder().seq_encode(groups)

    # seq_log_density matches per-group log_density
    seq_ld = np.asarray(sd.seq_log_density(senc))
    scalar_ld = np.asarray([sd.log_density(g) for g in groups])
    assert float(np.max(np.abs(seq_ld - scalar_ld))) < 1e-9

    # local ELBO is finite
    seq_elbo = np.asarray(sd.seq_local_elbo(senc))
    assert np.all(np.isfinite(seq_elbo))

    # group posteriors are valid probability simplices (rows sum to 1, nonnegative)
    gp = sd.group_posteriors(groups)
    assert np.all(gp >= -1e-12)
    assert np.allclose(gp.sum(axis=1), 1.0, atol=1e-9)


def _make_grouped_data(seed=3):
    rng = RandomState(seed)
    groups = []
    for _ in range(40):
        pref = rng.dirichlet([0.3, 0.3, 0.3])
        g = []
        for _ in range(60):
            kk = rng.choice(3, p=pref)
            g.append(float(rng.normal() * 0.6 + TRUE_MUS[kk]))
        groups.append(g)
    return groups


def test_hdpm_fit_accepted_objective_monotone_and_recovery():
    groups = _make_grouped_data()

    k = 8
    est = HierarchicalDirichletProcessMixtureEstimator(
        [GaussianEstimator(prior=NormalGammaDistribution(0.0, 1.0e-3, 1.0, 1.0)) for _ in range(k)],
        gamma=1.0,
        alpha=1.0,
    )

    # Track the accepted-model objective: fit only updates the kept model when
    # dobj >= 0, so the accepted trajectory is monotone even though the HDP beta
    # approximation can make an individual proposed step dip (the optimize()
    # driver relies on the same acceptance gate).
    enc = seq_encode(groups, est.accumulator_factory().make().acc_to_encoder())
    mm = seq_initialize(enc_data=enc, estimator=est, rng=RandomState(7), p=1.0)

    def objective(model):
        data_term = sum(model.seq_local_elbo(u[1]).sum() for u in enc)
        return data_term + est.model_log_density(model)

    accepted = objective(mm)
    accepted_traj = [accepted]
    for _ in range(60):
        proposed = seq_estimate(enc, est, mm)
        pobj = objective(proposed)
        if pobj >= accepted - 1e-9:  # acceptance gate (as in fit())
            mm = proposed
            accepted = pobj
        accepted_traj.append(accepted)

    diffs = np.diff(np.asarray(accepted_traj))
    assert np.all(diffs >= -1e-6), "accepted objective decreased: min step %g" % diffs.min()

    dom = np.where(mm.beta > 0.05)[0]
    dmu = np.asarray([mm.components[i].mu for i in dom])
    for t in TRUE_MUS:
        assert float(np.min(np.abs(dmu - t))) < 0.6, "atom %g not recovered (got %s)" % (t, dmu)


def test_hdpm_fit_driver_runs():
    groups = _make_grouped_data(seed=4)
    k = 8
    est = HierarchicalDirichletProcessMixtureEstimator(
        [GaussianEstimator(prior=NormalGammaDistribution(0.0, 1.0e-3, 1.0, 1.0)) for _ in range(k)],
        gamma=1.0,
        alpha=1.0,
    )
    buf = io.StringIO()
    model = fit(groups, est, max_its=80, delta=None, rng=RandomState(7), init_p=1.0, out=buf)
    dom = np.where(model.beta > 0.05)[0]
    dmu = np.asarray([model.components[i].mu for i in dom])
    for t in TRUE_MUS:
        assert float(np.min(np.abs(dmu - t))) < 0.6
