"""NonlinearEvidence: iterated-EKF assimilation of nonlinear forward models into the belief."""

import numpy as np
import pytest

from mixle.reason import Latent, NonlinearEvidence, reason
from mixle.reason.core import Evidence, _fd_jacobian


def _h(z):
    return np.array([z[0] ** 2 + z[1], np.sin(z[1])])


def _J(z):
    return np.array([[2.0 * z[0], 1.0], [0.0, np.cos(z[1])]])


def test_finite_difference_matches_analytic_jacobian():
    z = np.array([1.3, -0.4])
    assert np.allclose(_fd_jacobian(_h, z), _J(z), atol=1e-5)


def test_iekf_recovers_the_latent_through_a_nonlinear_forward():
    truth = np.array([1.5, 0.7])
    y = _h(truth)  # noise-free measurement, small R
    prior = Latent.gaussian([0.3, 0.0], np.eye(2) * 4.0)  # a prior mean far from the truth

    ans = reason(prior, [NonlinearEvidence(_h, y, 1e-4, jacobian=_J, iterations=6, name="sensor")])
    assert np.allclose(np.asarray(ans.belief.mean()).reshape(-1), truth, atol=0.05)
    assert ans.attribution()["sensor"] > 0.0  # nats removed are attributed

    # a single linearization from that far prior is measurably worse than the iterated one
    one = reason(prior, [NonlinearEvidence(_h, y, 1e-4, jacobian=_J, iterations=1)])
    err_one = np.linalg.norm(np.asarray(one.belief.mean()).reshape(-1) - truth)
    err_many = np.linalg.norm(np.asarray(ans.belief.mean()).reshape(-1) - truth)
    assert err_many <= err_one + 1e-9


def test_mixes_linear_and_nonlinear_evidence():
    truth = np.array([1.5, 0.7])
    prior = Latent.gaussian([0.0, 0.0], np.eye(2) * 4.0)
    ev = [
        Evidence(np.array([[1.0, 0.0]]), np.array([truth[0]]), 1e-3, name="linear-probe"),
        NonlinearEvidence(_h, _h(truth), 1e-3, iterations=4, name="nonlinear-sensor"),  # fd jacobian
    ]
    ans = reason(prior, ev)
    assert np.allclose(np.asarray(ans.belief.mean()).reshape(-1), truth, atol=0.08)
    att = ans.attribution()
    assert set(att) == {"linear-probe", "nonlinear-sensor"}
    assert all(v >= -1e-9 for v in att.values())


def _square(z):
    return np.array([z[0] ** 2])


def _square_jacobian(z):
    return np.array([[2.0 * z[0]]])


def test_evidence_order_changes_the_result_when_nonlinear_evidence_is_mixed_in():
    """Regression test: reason()'s docstring used to claim evidence order never affects the result,
    but that is only true for pure LinearGaussianEvidence. NonlinearEvidence linearizes at the
    CURRENT belief mean, so folding it in before vs. after other evidence changes its linearization
    point -- and therefore both the posterior and the per-source attribution.

    h(z) = z**2 has Jacobian 2z, which is exactly zero at the prior mean (0.0). Folded in FIRST, the
    linearized update is a total no-op (zero information gain) even though y=4.0 is informative;
    folded in AFTER linear evidence has moved the mean away from zero, the same evidence linearizes
    at a nonzero point and genuinely sharpens the belief."""
    prior = Latent.gaussian([0.0], [[4.0]])
    linear = Evidence(H=[[1.0]], y=[3.0], R=[[1.0]], name="linear")
    nonlinear = NonlinearEvidence(_square, [4.0], [[0.01]], jacobian=_square_jacobian, iterations=2, name="nonlinear")

    with pytest.warns(RuntimeWarning, match="ORDER-DEPENDENT"):
        nonlinear_first = reason(prior, [nonlinear, linear])
    with pytest.warns(RuntimeWarning, match="ORDER-DEPENDENT"):
        linear_first = reason(prior, [linear, nonlinear])

    # folded first, at the prior mean (0.0) where the Jacobian is exactly zero: no information at all
    assert nonlinear_first.attribution()["nonlinear"] == pytest.approx(0.0, abs=1e-9)
    # folded second, at a nonzero mean: the same evidence now genuinely sharpens the belief
    assert linear_first.attribution()["nonlinear"] > 1.0

    # the two posteriors disagree substantially -- not a rounding-level discrepancy
    cov_nonlinear_first = float(nonlinear_first.cov()[0, 0])
    cov_linear_first = float(linear_first.cov()[0, 0])
    assert cov_nonlinear_first / cov_linear_first > 50.0
    assert abs(float(nonlinear_first.mean[0]) - float(linear_first.mean[0])) > 0.1


def test_reason_does_not_warn_for_a_single_evidence_item_or_all_linear_evidence(recwarn):
    """The order-dependence warning is specifically about MIXING NonlinearEvidence with other
    evidence -- it must stay silent for the cases where order provably does not matter: a single
    evidence item (nothing to order), and any all-LinearGaussianEvidence sequence."""
    prior = Latent.gaussian([0.0], [[4.0]])
    reason(prior, [NonlinearEvidence(_square, [4.0], [[0.01]], jacobian=_square_jacobian)])
    reason(prior, [Evidence([[1.0]], [3.0], [[1.0]], "a"), Evidence([[1.0]], [2.0], [[1.0]], "b")])
    assert len(recwarn) == 0
