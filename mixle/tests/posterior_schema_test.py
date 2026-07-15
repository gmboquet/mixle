"""Semantic posterior schema: the validated replacement for hand-written convention bridges. The
headline test reproduces the real, hand-written G2->G6 bridge (SourcePosterior.to_doe_prior) exactly
via adapt()+join_independent(), and the rest pin down that the *validation* actually fires -- a
missing/renamed/mis-unit'd axis raises loudly instead of silently producing a wrong number, which is
the whole point (that silent mismatch is what motivated this)."""

import numpy as np
import pytest

from mixle.reason.posterior_protocol import Posterior
from mixle.reason.posterior_schema import (
    AxisSpec,
    PosteriorSchema,
    SchematizedPosterior,
    adapt,
    join_independent,
)


def test_adapt_plus_join_reproduces_the_handwritten_g2_to_g6_bridge():
    # G2's real (x, y, rate) posterior + a separately-estimated onset belief.
    mean3 = np.array([3.0, 7.0, 6.0])
    cov3 = np.array([[0.030, -0.007, 0.0010], [-0.007, 0.0036, 0.0005], [0.0010, 0.0005, 0.0200]])
    onset_mean, onset_sd = 0.5, 0.22

    # what SourcePosterior.to_doe_prior computes by hand (minus its 1e-6 PD jitter):
    rate = mean3[2]
    expected_mean = np.array([mean3[0], mean3[1], np.log(rate), onset_mean])
    expected_cov = np.zeros((4, 4))
    expected_cov[:2, :2] = cov3[:2, :2]
    expected_cov[2, 2] = cov3[2, 2] / rate**2
    expected_cov[:2, 2] = cov3[:2, 2] / rate
    expected_cov[2, :2] = expected_cov[:2, 2]
    expected_cov[3, 3] = onset_sd**2

    # the same thing via validated, composable operations:
    g2_schema = PosteriorSchema((AxisSpec("x", "m"), AxisSpec("y", "m"), AxisSpec("rate", "kg/s", "linear")))
    rate_logged = PosteriorSchema((AxisSpec("x", "m"), AxisSpec("y", "m"), AxisSpec("rate", "kg/s", "log")))
    logged = adapt(mean3, cov3, g2_schema, rate_logged)

    onset_schema = PosteriorSchema((AxisSpec("onset", "s"),))
    joined = join_independent(
        (logged.mean, logged.cov, rate_logged),
        (np.array([onset_mean]), np.array([[onset_sd**2]]), onset_schema),
    )

    np.testing.assert_allclose(joined.mean, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(joined.cov, expected_cov, atol=1e-12)
    assert joined.schema.names == ["x", "y", "rate", "onset"]


def test_adapt_raises_when_a_target_axis_is_absent_from_source():
    # the exact failure the old duck-typed IC-1 could not catch: a target needs an axis the source
    # posterior's covariance simply doesn't contain. adapt() must refuse, not invent it.
    src = PosteriorSchema((AxisSpec("x"), AxisSpec("y")))
    tgt = PosteriorSchema((AxisSpec("x"), AxisSpec("y"), AxisSpec("onset")))
    with pytest.raises(KeyError, match="onset"):
        adapt(np.zeros(2), np.eye(2), src, tgt)


def test_adapt_raises_on_unit_mismatch_for_a_matched_axis():
    src = PosteriorSchema((AxisSpec("depth", "m"),))
    tgt = PosteriorSchema((AxisSpec("depth", "ft"),))
    with pytest.raises(ValueError, match="unit mismatch"):
        adapt(np.array([1.0]), np.array([[1.0]]), src, tgt)


def test_schema_validate_catches_arity_mismatch():
    schema = PosteriorSchema((AxisSpec("x"), AxisSpec("y")))
    with pytest.raises(ValueError, match="schema declares 2 axes"):
        schema.validate(np.zeros(3), np.eye(3))


def test_adapt_reorders_and_marginalizes():
    src = PosteriorSchema((AxisSpec("a"), AxisSpec("b"), AxisSpec("c")))
    tgt = PosteriorSchema((AxisSpec("c"), AxisSpec("a")))  # reorder + drop b
    out = adapt(np.array([1.0, 2.0, 3.0]), np.diag([0.1, 0.2, 0.3]), src, tgt)
    np.testing.assert_allclose(out.mean, [3.0, 1.0])
    np.testing.assert_allclose(out.cov, np.diag([0.3, 0.1]))


def test_linear_log_roundtrip_is_exact_for_a_single_step():
    lin = PosteriorSchema((AxisSpec("rate", "kg/s", "linear"),))
    log = PosteriorSchema((AxisSpec("rate", "kg/s", "log"),))
    mean, cov = np.array([6.0]), np.array([[0.02]])
    there = adapt(mean, cov, lin, log)
    back = adapt(there.mean, there.cov, log, lin)
    np.testing.assert_allclose(back.mean, mean, rtol=1e-12)
    np.testing.assert_allclose(back.cov, cov, rtol=1e-12)  # reciprocal Jacobians at the same point


def test_log_transform_rejects_nonpositive_mean():
    lin = PosteriorSchema((AxisSpec("rate", "kg/s", "linear"),))
    log = PosteriorSchema((AxisSpec("rate", "kg/s", "log"),))
    with pytest.raises(ValueError, match="not positive"):
        adapt(np.array([-1.0]), np.array([[1.0]]), lin, log)


def test_schematized_posterior_satisfies_the_ic1_protocol():
    schema = PosteriorSchema((AxisSpec("x", "m"), AxisSpec("rate", "kg/s", "linear")))
    sp = SchematizedPosterior(np.array([3.0, 6.0]), np.diag([0.1, 0.2]), schema)
    assert isinstance(sp, Posterior)  # runtime_checkable structural check
    rng = np.random.default_rng(0)
    assert sp.samples(50, rng).shape == (50, 2)
    lo, hi = sp.credible_interval(0.9)
    assert np.all(lo < hi)
    dq = sp.derived_quantity(lambda draws: draws[:, 0] + draws[:, 1], 100, rng)
    assert dq.samples.shape == (100,)


def test_join_independent_rejects_duplicate_axis_names():
    s = PosteriorSchema((AxisSpec("x"),))
    with pytest.raises(ValueError, match="duplicate axis name"):
        join_independent((np.array([1.0]), np.array([[1.0]]), s), (np.array([2.0]), np.array([[1.0]]), s))
