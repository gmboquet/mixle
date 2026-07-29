"""Causal labels require explicit, auditable identification assumptions."""

import numpy as np
import pytest

from mixle.inference import (
    CausalIdentification,
    EventStudyIdentification,
    average_causal_effect,
    bn_do,
    hierarchical_event_study,
)
from mixle.inference.bayesian_network import (
    HeterogeneousBayesianNetwork,
    _LinearGaussianFactor,
    _MarginalFactor,
)
from mixle.stats import GaussianDistribution


def _network():
    return HeterogeneousBayesianNetwork(
        [
            _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
            _LinearGaussianFactor(1, [0], {}, np.array([2.0, 0.0]), 0.1),
        ]
    )


def _causal_receipt():
    return CausalIdentification.domain_asserted("protocol://pre-specified-dag")


def _event_receipt():
    return EventStudyIdentification(
        design_evidence=("study://matched-exposed-controls",),
        parallel_trends_evidence=("analysis://pre-period-placebo",),
        exchangeability=True,
        positivity=True,
        consistency=True,
        no_interference=True,
        no_anticipation=True,
        sensitivity_analysis="analysis://differential-drift",
    )


def test_graph_surgery_without_receipt_is_labeled_structural_not_causal():
    world = bn_do(_network(), {0: 1.0})
    assert not world.identified
    assert "not a causal estimate" in world.interpretation
    assert world.identification is None


def test_average_causal_effect_requires_an_identification_receipt():
    with pytest.raises(ValueError, match="CausalIdentification"):
        average_causal_effect(_network(), 0, 1.0, 0.0, 1, n=20)

    effect = average_causal_effect(_network(), 0, 1.0, 0.0, 1, identification=_causal_receipt(), n=2000, seed=3)
    assert effect == pytest.approx(2.0, abs=0.02)


def test_event_study_defaults_to_association_and_carries_identification_receipt():
    treated = np.array([0.8, 1.0, 1.2])
    control = np.array([0.2, 0.3, 0.4])
    variances = np.array([0.1, 0.1, 0.1])

    association = hierarchical_event_study(treated, variances, control, variances)
    assert not association.identified
    assert "association" in association.estimand
    assert association.identification is None

    identified = hierarchical_event_study(
        treated,
        variances,
        control,
        variances,
        identification=_event_receipt(),
    )
    assert identified.identified
    assert "treatment effect" in identified.estimand
    assert identified.identification["parallel_trends_evidence"] == ("analysis://pre-period-placebo",)


def test_event_identification_without_controls_or_complete_assumptions_is_rejected():
    with pytest.raises(ValueError, match="control group"):
        hierarchical_event_study(
            np.array([0.8]),
            np.array([0.1]),
            identification=_event_receipt(),
        )

    incomplete = EventStudyIdentification(
        design_evidence=("study://design",),
        parallel_trends_evidence=("analysis://placebo",),
        exchangeability=True,
        positivity=True,
        consistency=True,
        no_interference=False,
        no_anticipation=True,
    )
    with pytest.raises(ValueError, match="no interference"):
        hierarchical_event_study(
            np.array([0.8]),
            np.array([0.1]),
            np.array([0.2]),
            np.array([0.1]),
            identification=incomplete,
        )


@pytest.mark.parametrize(
    "args",
    [
        (np.array([np.nan]), np.array([0.1]), None, None),
        (np.array([0.1]), np.array([0.1]), np.array([]), np.array([])),
        (np.array([0.1]), np.array([0.1]), np.array([0.0]), None),
    ],
)
def test_event_study_rejects_malformed_inputs(args):
    with pytest.raises(ValueError):
        hierarchical_event_study(*args)
