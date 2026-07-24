"""Regression test: MultivariateStudentTDistribution must reject an invalid scale matrix at
construction, not silently accept it.

Before the fix, ``__init__`` only checked ``shape``'s dimensions; the only "validity" check was
buried in ``_safe_inverse_and_logdet``, which used ``np.linalg.slogdet``'s determinant SIGN as a
positive-definiteness proxy. That is not a valid test: a matrix can have a positive determinant
while being negative definite or indefinite (e.g. ``-I`` in an even dimension has determinant
``(-1)^p = +1`` while every eigenvalue is negative). So an invalid ``shape`` (non-symmetric,
non-PSD, or negative definite) was silently accepted, and scoring it produced a FINITE
``density``/``log_density`` at points near the location -- while ``sampler()`` on that very same
instance raised ``numpy.linalg.LinAlgError`` from ``np.linalg.cholesky(dist.shape)``, since
Cholesky correctly requires genuine positive-definiteness. Density scoring and sampling silently
disagreed about whether the distribution was even well-formed.

The fix validates ``shape`` in ``__init__`` -- symmetry via ``np.allclose(shape, shape.T)`` and
positive-definiteness via ``mixle.utils.vector.cholesky_logdet`` (Cholesky-based, not determinant
sign), matching the sibling idiom already used by ``StudentTCopulaDistribution``,
``GaussianCopulaDistribution``, and ``NormalWishartDistribution`` -- so an invalid scale matrix now
raises ``ValueError`` before a distribution object (and therefore a sampler) can ever exist.

Follow-up fix: that first pass rejected negative and indefinite matrices, but not an EXACTLY
SINGULAR one (a genuine zero eigenvalue, positive *semi*-definite but not positive *definite*).
``_safe_inverse_and_logdet``'s one-retry tiny-ridge fallback (meant to absorb float rounding on a
matrix that is PD in exact arithmetic, e.g. the EM estimator's own scatter matrix) cannot tell that
case apart from a shape that is genuinely, structurally singular -- their eigenvalues are identical
either way -- so it silently ridge-healed a singular ``shape`` enough to produce a finite
``inv_shape``/``log_det`` (and therefore a finite scored density), while ``self.shape`` itself (what
``sampler()`` runs ``np.linalg.cholesky`` on directly) was left untouched and still exactly singular,
so sampling raised ``LinAlgError``. ``__init__`` now checks ``cholesky_logdet(shape)`` directly on
the raw shape -- before ``_safe_inverse_and_logdet`` ever gets a chance to ridge through it -- so a
singular shape is rejected the same way a negative-definite one already was.
"""

import numpy as np
import pytest

from mixle.stats.multivariate.multivariate_student_t import MultivariateStudentTDistribution


def test_rejects_symmetric_indefinite_shape():
    # Symmetric, eigenvalues [1, -1] -- indefinite, not positive semi-definite.
    bad = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError):
        MultivariateStudentTDistribution(dof=5.0, loc=[0.0, 0.0], shape=bad)


def test_rejects_negative_definite_with_positive_determinant():
    # Symmetric, negative definite (eigenvalues [-1, -1]) -- but det(-I_2) = (-1)^2 = +1, so a
    # determinant-SIGN check (what _safe_inverse_and_logdet used pre-fix) is fooled into treating
    # this as valid. A correct PD test (Cholesky / eigenvalues) must still reject it.
    bad = -np.eye(2)
    assert np.linalg.det(bad) > 0.0
    with pytest.raises(ValueError):
        MultivariateStudentTDistribution(dof=5.0, loc=[0.0, 0.0], shape=bad)


def test_rejects_asymmetric_shape():
    # Off-diagonal entries disagree; determinant is positive and slogdet's sign check alone
    # would have waved this through too.
    bad = np.array([[2.0, 1.0], [0.0, 1.0]])
    assert np.linalg.det(bad) > 0.0
    with pytest.raises(ValueError):
        MultivariateStudentTDistribution(dof=5.0, loc=[0.0, 0.0], shape=bad)


def test_rejects_exactly_singular_shape():
    # Positive SEMI-definite but not positive DEFINITE: eigenvalues [1, 0], rank-deficient by
    # construction (not a float-rounding artifact -- 0.0 and 1.0 are both exactly representable).
    # A determinant-sign OR a naive "eigenvalue < -tol" check both wave this through (det == 0 is
    # not "negative", and the offending eigenvalue is 0, not negative), which is exactly why this
    # slipped past the first validation pass despite it having (re)used a correct Cholesky-based PD
    # test: the gap was _safe_inverse_and_logdet's ridge retry silently healing this specific case,
    # not the PD test itself.
    bad = np.diag([1.0, 0.0])
    assert np.linalg.det(bad) == 0.0
    assert np.linalg.matrix_rank(bad) == 1
    with pytest.raises(ValueError):
        MultivariateStudentTDistribution(dof=5.0, loc=[0.0, 0.0], shape=bad)


def test_singular_shape_previously_scored_finite_while_its_sampler_would_crash():
    """Documents the exact pre-fix failure mode for the singular case: this reconstructs, standalone,
    what ``_safe_inverse_and_logdet`` (imported directly, bypassing ``__init__``'s now-fixed gate)
    actually computed for a singular shape -- a FINITE log-determinant, because its ridge retry
    can't distinguish "singular by float rounding" from "genuinely, structurally singular" -- while
    a direct, unhealed Cholesky factorization of that same raw shape (what the sampler ran on
    ``self.shape`` before the fix) genuinely raises. That gap between "scores fine" and "sampling
    raises" on the SAME raw shape is the bug; ``__init__`` closing it is what the tests above check.
    """
    from mixle.stats.multivariate.multivariate_student_t import _safe_inverse_and_logdet

    bad = np.diag([1.0, 0.0])
    _, log_det = _safe_inverse_and_logdet(bad)  # the ridge retry heals this silently
    assert np.isfinite(log_det)

    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(bad)  # what the (pre-fix) sampler ran directly on the raw, unhealed shape


def test_invalid_shape_never_yields_finite_density_or_sampler_disagreement():
    """End-to-end guard against the exact reported asymmetry: for a battery of invalid scale
    matrices, construction must fail up front so neither ``density`` nor ``sampler`` can ever run
    on a bad instance (as opposed to density silently succeeding while sampling later raises).
    """
    invalid_shapes = [
        np.array([[1.0, 0.0], [0.0, -1.0]]),  # indefinite
        -np.eye(2),  # negative definite, positive determinant
        np.array([[2.0, 1.0], [0.0, 1.0]]),  # asymmetric
        np.array([[1.0, 2.0], [2.0, 1.0]]),  # symmetric, det=-3 < 0
        np.diag([1.0, 0.0]),  # exactly singular: positive semi-definite, not positive definite
        np.zeros((2, 2)),  # exactly singular, degenerate case: both eigenvalues 0
        np.array([[1.0, 1.0], [1.0, 1.0]]),  # rank-1, singular via a non-diagonal matrix
    ]
    for bad in invalid_shapes:
        with pytest.raises(ValueError):
            MultivariateStudentTDistribution(dof=5.0, loc=[0.0, 0.0], shape=bad)


def test_valid_shape_still_constructs_scores_and_samples():
    """No-regression control: a genuinely valid symmetric PD scale matrix must still construct,
    score a finite density, and sample without error (construction-time validation must not have
    become overly strict).
    """
    good = np.array([[2.0, 0.3], [0.3, 1.0]])
    dist = MultivariateStudentTDistribution(dof=5.0, loc=[0.0, 0.0], shape=good)

    d = dist.density([0.1, -0.2])
    ld = dist.log_density([0.1, -0.2])
    assert np.isfinite(d)
    assert np.isfinite(ld)

    sample = dist.sampler(seed=0).sample()
    assert sample.shape == (2,)
    assert np.all(np.isfinite(sample))


def test_condition_and_marginal_still_work_after_validation():
    """condition()/marginal() build new scale matrices internally (Schur complement / submatrix)
    and re-construct MultivariateStudentTDistribution instances; these legitimate internal paths
    must not be broken by the added validation.
    """
    dist = MultivariateStudentTDistribution(dof=5.0, loc=[0.0, 0.0, 0.0], shape=np.eye(3) * 2.0)

    cond = dist.condition({0: 0.5})
    assert np.isfinite(cond.density(cond.mu))

    marg = dist.marginal([0, 1])
    assert np.isfinite(marg.density(marg.mu))
