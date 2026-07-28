"""explain_margin() / explain_margin_mixture(): exact per-part attribution of a DECISION MARGIN --
log p(answer) - log p(runner_up) -- between two named hypotheses (workstream H1 of the 0.7.0
frontier-capability plan: the answer-with-receipts evidence ledger).

The ledger must sum to the margin at machine precision on every supported structure, and a deliberately
corrupted evidence field must be caught: its contribution is the one part that does not collapse to 0
when every other field is identical between the two candidate records.
"""

import numpy as np
import pytest

from mixle.inference import explain, explain_margin, explain_margin_mixture
from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork, _LinearGaussianFactor, _MarginalFactor
from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    MixtureDistribution,
)


def _composite():
    return CompositeDistribution(
        (
            CategoricalDistribution({"a": 0.9, "b": 0.1}),
            GaussianDistribution(0.0, 1.0),
            GaussianDistribution(5.0, 2.0),
        )
    )


class CompositeMarginTest:
    def test_margin_ledger_sums_exactly(self):
        comp = _composite()
        answer, runner_up = ("a", 0.2, 5.1), ("b", -0.3, 4.8)
        ex = explain_margin(comp, answer, runner_up)
        expected = comp.log_density(answer) - comp.log_density(runner_up)
        np.testing.assert_allclose(ex.total, expected, atol=1e-12)
        np.testing.assert_allclose(ex.ledger_sum(), ex.total, atol=1e-12)
        assert ex.is_exact()
        assert abs(ex.correction) < 1e-9  # Composite is purely additive -- no logsumexp anywhere

    def test_corrupted_single_field_is_the_only_nonzero_contribution(self):
        comp = _composite()
        answer = ("a", 0.1, 5.2)
        corrupted = ("a", 0.1, -50.0)  # only field[2] differs from `answer`
        ex = explain_margin(comp, answer, corrupted)
        by_name = dict(ex.parts)
        assert by_name["field[0]"] == 0.0  # identical category -> exactly 0, not merely small
        assert by_name["field[1]"] == 0.0  # identical Gaussian field -> exactly 0
        assert abs(by_name["field[2]"]) > 100.0  # the corrupted field is the visible anomaly
        assert ex.is_exact()


class BayesianNetworkMarginTest:
    def _net(self):
        return HeterogeneousBayesianNetwork(
            [
                _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
                _LinearGaussianFactor(1, [0], {}, np.array([2.0, 0.0]), 0.5),
            ]
        )

    def test_margin_ledger_sums_exactly(self):
        net = self._net()
        answer, runner_up = (0.5, 1.1), (0.4, 0.9)
        ex = explain_margin(net, answer, runner_up)
        expected = net.log_density(answer) - net.log_density(runner_up)
        np.testing.assert_allclose(ex.total, expected, atol=1e-12)
        assert ex.is_exact()
        assert abs(ex.correction) < 1e-9  # a BN factorization is purely additive -- no logsumexp

    def test_corrupted_parent_flips_the_dependent_edge_contribution(self):
        net = self._net()
        answer = (0.5, 1.0)  # child ~= 2*parent, on-model
        corrupted = (0.5, 20.0)  # only the child (an edge target) corrupted -> only the edge factor moves
        ex = explain_margin(net, answer, corrupted)
        by_name = dict((name.split("|")[0], v) for name, v in ex.parts)
        assert by_name["field[0]"] == 0.0  # the parent is identical between the two records
        assert abs(by_name["field[1]"]) > 50.0  # the corrupted child is the visible anomaly
        assert ex.is_exact()


class MixtureMarginTest:
    def _mix(self):
        c0 = CompositeDistribution((CategoricalDistribution({"a": 0.9, "b": 0.1}), GaussianDistribution(-3.0, 1.0)))
        c1 = CompositeDistribution((CategoricalDistribution({"a": 0.1, "b": 0.9}), GaussianDistribution(3.0, 1.0)))
        return c0, c1, MixtureDistribution([c0, c1], [0.5, 0.5])

    def test_margin_matches_manual_computation_and_needs_no_correction(self):
        c0, c1, mix = self._mix()
        x = ("b", 3.2)
        ex = explain_margin_mixture(mix, x, answer=1, runner_up=0)
        manual = (float(mix.log_w[1]) + c1.log_density(x)) - (float(mix.log_w[0]) + c0.log_density(x))
        np.testing.assert_allclose(ex.total, manual, atol=1e-12)
        # the logsumexp normalizer cancels exactly in a two-component margin -- up to floating-point noise
        assert abs(ex.correction) < 1e-9
        assert ex.is_exact()

    def test_margin_decomposes_into_prior_plus_per_field(self):
        _c0, _c1, mix = self._mix()
        x = ("b", 3.2)
        ex = explain_margin_mixture(mix, x, answer=1, runner_up=0)
        names = {name for name, _ in ex.parts}
        assert names == {"prior", "field[0]", "field[1]"}

    def test_explain_margin_rejects_a_mixture_pointing_to_explain_margin_mixture(self):
        _c0, _c1, mix = self._mix()
        with pytest.raises(TypeError, match="explain_margin_mixture"):
            explain_margin(mix, ("a", 0.0), ("b", 0.0))


class MixtureMarginHypothesisValidationTest:
    """MXR-080-1615: a hypothesis names a class exactly -- it is not rounded, wrapped, or self-compared."""

    def _mix(self):
        c0 = CompositeDistribution((CategoricalDistribution({"a": 0.9, "b": 0.1}), GaussianDistribution(-3.0, 1.0)))
        c1 = CompositeDistribution((CategoricalDistribution({"a": 0.1, "b": 0.9}), GaussianDistribution(3.0, 1.0)))
        return MixtureDistribution([c0, c1], [0.5, 0.5])

    def test_fractional_hypotheses_are_rejected_not_truncated(self):
        mix = self._mix()
        with pytest.raises(TypeError, match="exact integer component index"):
            explain_margin_mixture(mix, ("b", 3.2), answer=0.9, runner_up=1.9)

    def test_negative_hypothesis_does_not_wrap_to_the_last_component(self):
        mix = self._mix()
        with pytest.raises(IndexError, match="negative wrap-around"):
            explain_margin_mixture(mix, ("b", 3.2), answer=-1, runner_up=0)

    def test_out_of_range_hypothesis_is_rejected(self):
        mix = self._mix()
        with pytest.raises(IndexError, match="2-component mixture"):
            explain_margin_mixture(mix, ("b", 3.2), answer=2, runner_up=0)

    def test_boolean_hypotheses_are_rejected(self):
        mix = self._mix()
        with pytest.raises(TypeError, match="not a Boolean"):
            explain_margin_mixture(mix, ("b", 3.2), answer=True, runner_up=False)

    def test_identical_hypotheses_are_rejected_rather_than_reported_as_a_zero_margin(self):
        mix = self._mix()
        with pytest.raises(ValueError, match="two distinct"):
            explain_margin_mixture(mix, ("b", 3.2), answer=1, runner_up=1)

    def test_numpy_integer_hypotheses_are_accepted(self):
        mix = self._mix()
        ex = explain_margin_mixture(mix, ("b", 3.2), answer=np.int64(1), runner_up=np.int64(0))
        assert ex.is_exact()

    def test_mismatched_component_schemas_do_not_produce_a_truncated_field_ledger(self):
        """A truncating zip drops the trailing fields, leaving a margin ledger with a phantom correction.

        The mixture margin is documented to need no correction (the normalizer cancels), so a non-zero
        one here means evidence silently fell out of the per-field breakdown.
        """

        class _Field:
            def __init__(self, v):
                self.v = v

            def log_density(self, _x):
                return self.v

        class _Component:
            def __init__(self, dists, total):
                self.dists = dists
                self._total = total

            def log_density(self, _x):
                return self._total

        class _Mix:
            def __init__(self, components):
                self.components = components
                self.log_w = np.log([0.5, 0.5])

        answer_c = _Component([_Field(-0.4), _Field(-0.6)], -1.0)
        runner_c = _Component([_Field(-0.1), _Field(-0.2), _Field(-0.7)], -1.0)
        ex = explain_margin_mixture(_Mix([runner_c, answer_c]), (0.0, 0.0), answer=1, runner_up=0)
        assert {name for name, _ in ex.parts} == {"prior", "component_density"}
        assert abs(ex.correction) < 1e-9
        assert ex.is_exact()


class GenericModelSingleEvaluationTest:
    """MXR-080-1614: a generic scorer is evaluated exactly once per explanation."""

    class _CountingModel:
        def __init__(self):
            self.calls = 0

        def log_density(self, x):
            self.calls += 1
            return float(self.calls)

    def test_generic_fallback_scores_once_and_the_ledger_reconstructs_the_total(self):
        model = self._CountingModel()
        ex = explain(model, 0.0)
        assert model.calls == 1
        assert ex.total == 1.0
        assert ex.parts == [("model", 1.0)]
        assert ex.is_exact()


if __name__ == "__main__":
    import unittest

    unittest.main()
