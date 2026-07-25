"""Theorem-matched receipts for finite-hypothesis PAC-Bayes certificates."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.pac_bayes import (
    PACBayesAssumptions,
    categorical_kl,
    certify_generalization,
    gaussian_kl,
    mcallester_bound,
)

pytestmark = pytest.mark.experimental


def _assumptions(**overrides):
    values = {
        "hypothesis_space_id": "bernoulli-grid-v1",
        "prior_id": "uniform-before-sample-v1",
        "sample_iid": True,
        "hypotheses_fixed_before_sample": True,
        "prior_fixed_before_sample": True,
        "loss_fixed_before_sample": True,
    }
    values.update(overrides)
    return PACBayesAssumptions(**values)


def _problem(seed, n, true_probability=0.65):
    predictions = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    observations = np.random.default_rng(seed).binomial(1, true_probability, size=n)
    losses = (predictions[:, None] - observations[None, :]) ** 2
    prior = np.full(predictions.size, 1.0 / predictions.size)
    empirical = losses.mean(axis=1)
    logits = np.log(prior) - n * empirical
    posterior = np.exp(logits - logits.max())
    posterior /= posterior.sum()
    true_losses = true_probability * (1.0 - predictions) ** 2 + (1.0 - true_probability) * predictions**2
    return losses, posterior, prior, true_losses


def test_gaussian_kl_is_only_a_validated_parameter_law_primitive():
    assert gaussian_kl(0.0, 1.0, 0.0, 1.0) == 0.0
    assert np.isclose(gaussian_kl(0.0, 1.0, 0.0, 4.0), 0.5 * (np.log(4.0) + 0.25 - 1.0))
    with pytest.raises(ValueError, match="positive"):
        gaussian_kl(0.0, 0.0, 0.0, 1.0)


def test_categorical_kl_is_over_hypotheses_and_decomposes_exactly():
    posterior = [0.7, 0.2, 0.1]
    prior = [0.4, 0.4, 0.2]
    kl, terms = categorical_kl(posterior, prior)
    assert np.isclose(kl, sum(terms))
    assert kl >= 0


def test_certificate_matches_finite_hypothesis_gibbs_risk_theorem():
    losses, posterior, prior, true_losses = _problem(seed=1, n=500)
    certificate = certify_generalization(
        losses,
        posterior,
        prior,
        assumptions=_assumptions(),
        delta=0.05,
    )
    true_gibbs_risk = float(posterior @ true_losses)
    assert true_gibbs_risk <= certificate.bound
    assert not certificate.vacuous
    assert certificate.theorem.startswith("McAllester bounded-loss PAC-Bayes")
    assert certificate.assumptions.prior_fixed_before_sample is True


def test_data_dependent_posterior_is_allowed_but_prior_must_be_precommitted():
    losses, posterior, prior, _ = _problem(seed=2, n=100)
    first = certify_generalization(losses, posterior, prior, assumptions=_assumptions())
    alternative_posterior = np.roll(posterior, 1)
    second = certify_generalization(losses, alternative_posterior, prior, assumptions=_assumptions())
    assert first.posterior != second.posterior
    assert first.prior == second.prior

    with pytest.raises(ValueError, match="prior_fixed_before_sample"):
        certify_generalization(
            losses,
            posterior,
            prior,
            assumptions=_assumptions(prior_fixed_before_sample=False),
        )


def test_bound_complexity_tightens_with_sample_size_for_fixed_risk_and_kl():
    small = mcallester_bound(0.2, 1.5, 50, delta=0.05)
    large = mcallester_bound(0.2, 1.5, 5_000, delta=0.05)
    assert large < small


def test_empirical_coverage_on_fixed_hypotheses_and_prior():
    delta = 0.1
    violations = 0
    replications = 100
    for seed in range(replications):
        losses, posterior, prior, true_losses = _problem(seed=seed, n=200)
        certificate = certify_generalization(
            losses,
            posterior,
            prior,
            assumptions=_assumptions(),
            delta=delta,
        )
        violations += float(posterior @ true_losses) > certificate.bound
    assert violations / replications <= delta


@pytest.mark.parametrize(
    "losses,posterior,prior,match",
    [
        ([[0.1, 1.1]], [1.0], [1.0], r"\[0, 1\]"),
        ([[0.1, np.nan]], [1.0], [1.0], "finite"),
        ([[0.1, 0.2]], [1.0, 0.0], [0.5, 0.5], "one entry"),
        ([[0.1, 0.2], [0.3, 0.4]], [0.5, 0.5], [1.0, 0.0], "strictly positive"),
    ],
)
def test_invalid_certificate_inputs_fail_closed(losses, posterior, prior, match):
    with pytest.raises(ValueError, match=match):
        certify_generalization(losses, posterior, prior, assumptions=_assumptions())


def test_assumptions_and_result_are_durable_and_deterministic():
    losses, posterior, prior, _ = _problem(seed=3, n=100)
    first = certify_generalization(losses, posterior, prior, assumptions=_assumptions()).as_dict()
    second = certify_generalization(losses, posterior, prior, assumptions=_assumptions()).as_dict()
    assert first == second
    assert first["assumptions"]["hypothesis_space_id"] == "bernoulli-grid-v1"
