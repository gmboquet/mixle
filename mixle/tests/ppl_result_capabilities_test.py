"""Capability-specific PPL fit-result contracts."""

from __future__ import annotations

import numpy as np

from mixle.ppl import (
    PointwiseLogLikelihood,
    Sampleable,
    Summarizable,
    result_capabilities,
)


class _SummaryOnly:
    def summary(self):
        return {"theta": {"mean": 1.0}}


class _PosteriorLike:
    def __init__(self):
        self.predictive = lambda n, rng: rng.normal(size=n)

    def summary(self):
        return {}

    def samples(self, param=None):
        return np.ones(3)

    def pointwise_log_likelihood(self, data):
        return np.zeros((3, len(data)))


def test_summary_only_result_does_not_claim_unimplemented_capabilities():
    result = _SummaryOnly()
    capabilities = result_capabilities(result)
    assert isinstance(result, Summarizable)
    assert not isinstance(result, Sampleable)
    assert not isinstance(result, PointwiseLogLikelihood)
    assert capabilities.summarizable is True
    assert capabilities.sampleable is False
    assert capabilities.predictive is False
    assert capabilities.pointwise_log_likelihood is False


def test_normalized_result_capabilities_report_each_independent_facet():
    capabilities = result_capabilities(_PosteriorLike())
    assert capabilities.summarizable is True
    assert capabilities.sampleable is True
    assert capabilities.predictive is True
    assert capabilities.pointwise_log_likelihood is True
