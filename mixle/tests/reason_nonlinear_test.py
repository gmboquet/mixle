"""NonlinearEvidence: iterated-EKF assimilation of nonlinear forward models into the belief."""

import numpy as np

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
