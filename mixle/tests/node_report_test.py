"""Tests for the D1 node report protocol (mixle.inference.node_report)."""

import io
import math
import unittest

import numpy as np

from mixle.inference.em import run_em
from mixle.inference.estimation import optimize
from mixle.inference.node_report import (
    NodeReport,
    flat_report_table,
    node_report,
    root_em_report,
    walk_tree,
)
from mixle.stats import (
    BernoulliDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    CompositeEstimator,
    ExponentialDistribution,
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    OptionalDistribution,
    PoissonDistribution,
    SequenceDistribution,
    seq_encode,
)


def _representative_family_catalog():
    """A small, deliberately representative subset of stock families -- one leaf per common shape
    (continuous/discrete/count/binary), one wrapper (Optional), and one of each combinator
    (Composite/Mixture/Sequence) -- rather than this repo's full ~100-family catalog (see
    ``mixle.tests.sampler_seed_test``), which is unnecessarily large for exercising the GENERIC
    dispatcher this module adds (it is not per-family logic, so one representative of each structural
    shape is what actually exercises new code paths).
    """
    return {
        "GaussianDistribution": GaussianDistribution(0.0, 1.0),
        "ExponentialDistribution": ExponentialDistribution(1.5),
        "BernoulliDistribution": BernoulliDistribution(0.3),
        "PoissonDistribution": PoissonDistribution(3.0),
        "CategoricalDistribution": CategoricalDistribution({"a": 0.4, "b": 0.6}),
        "OptionalDistribution": OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.1),
        "CompositeDistribution": CompositeDistribution([GaussianDistribution(0.0, 1.0), PoissonDistribution(2.0)]),
        "MixtureDistribution": MixtureDistribution(
            [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)], [0.5, 0.5]
        ),
        "SequenceDistribution": SequenceDistribution(
            GaussianDistribution(0.0, 1.0), len_dist=CategoricalDistribution({3: 1.0})
        ),
    }


class NodeReportProtocolTestCase(unittest.TestCase):
    # -- 1. every stock family reports -----------------------------------------------------------
    def test_every_stock_family_reports(self):
        catalog = _representative_family_catalog()
        for name, dist in sorted(catalog.items()):
            with self.subTest(family=name):
                report = node_report(dist, seed=7)
                self.assertIsInstance(report, NodeReport)
                self.assertEqual(report.field_path, "root")
                self.assertEqual(report.node_type, type(dist).__name__)
                self.assertIn(
                    report.update_kind,
                    {"frozen", "em", "conjugate_closed_form", "gradient", "closed_form"},
                )
                self.assertTrue(math.isfinite(report.residual), "%s residual not finite" % name)
                self.assertGreaterEqual(report.param_count, 1)
                self.assertGreaterEqual(report.e_step_cost, 0.0)
                self.assertGreaterEqual(report.m_step_cost, 0.0)
                self.assertIn("finite_params", report.health)
                self.assertTrue(report.health["finite_params"], "%s reported non-finite params" % name)

    # -- 2. composed tree -> flat table -----------------------------------------------------------
    def test_composed_tree_flat_table(self):
        # A genuinely nested tree: a mixture of composites of (leaf, sequence-of-leaf).
        def make_component(mu):
            return CompositeDistribution(
                [
                    GaussianDistribution(mu, 1.0),
                    SequenceDistribution(PoissonDistribution(2.0), len_dist=CategoricalDistribution({3: 1.0})),
                ]
            )

        tree = MixtureDistribution([make_component(-2.0), make_component(2.0)], [0.5, 0.5])

        # Fit it so the report reflects a real (post-EM) model, not just an initial guess.
        data = tree.sampler(1).sample(500)
        # Build the sequence estimator FROM a SequenceDistribution that carries the real len_dist
        # (a fresh ``SequenceDistribution(PoissonDistribution(2.0))`` defaults ``len_dist`` to a
        # NullDistribution, whose estimator is a no-op NullEstimator -- the fitted model would then
        # keep a NullDistribution len_dist and be unsamplable; see SequenceEstimator.estimate).
        estimator = MixtureEstimator(
            [
                CompositeEstimator(
                    [
                        GaussianEstimator(),
                        SequenceDistribution(
                            PoissonDistribution(2.0), len_dist=CategoricalDistribution({3: 1.0})
                        ).estimator(),
                    ]
                )
            ]
            * 2
        )
        fitted = optimize(data, estimator, max_its=5, rng=np.random.RandomState(0), out=io.StringIO())

        nodes = walk_tree(fitted)
        table = flat_report_table(fitted, nobs=float(len(data)))

        # One row per node, same order and count as the raw walk.
        self.assertEqual(len(table), len(nodes))
        self.assertEqual([r.field_path for r in table], [path for path, _ in nodes])

        # Field paths are unique and every node in the nested structure is present.
        paths = {r.field_path for r in table}
        self.assertEqual(len(paths), len(table))
        self.assertTrue(any("MixtureDistribution.components[0]" in p for p in paths))
        self.assertTrue(any("MixtureDistribution.components[1]" in p for p in paths))
        self.assertTrue(any("CompositeDistribution.dists[0]" in p for p in paths))
        self.assertTrue(any("CompositeDistribution.dists[1]" in p for p in paths))
        self.assertTrue(any("SequenceDistribution.dist" in p for p in paths))

        # Sane, non-NaN values; residual/e_step_cost/m_step_cost/param_count are all non-negative.
        for row in table:
            self.assertTrue(math.isfinite(row.residual), row.field_path)
            self.assertGreaterEqual(row.e_step_cost, 0.0)
            self.assertGreaterEqual(row.m_step_cost, 0.0)
            self.assertGreaterEqual(row.param_count, 1)

        # A second table with the first as prev_table populates q_gain (diagnostic, per-node delta).
        table2 = flat_report_table(fitted, nobs=float(len(data)), prev_table=table)
        self.assertEqual(len(table2), len(table))
        # Same field paths -> every row's q_gain is computable (residual is finite before and after).
        for row in table2:
            self.assertIsNotNone(row.q_gain)
            self.assertTrue(math.isfinite(row.q_gain))

    # -- 3. Q-gain sanity check on the real, tracked EM objective ---------------------------------
    def test_root_em_objective_is_non_decreasing(self):
        """Neal-Hinton coordinate ascent: the ACTUAL tracked EM objective (here, observed
        log-likelihood via ``run_em``'s resolved objective) never decreases across a real EM step
        on a non-degenerate model/dataset. This is the one quantity this module does not
        approximate (see ``root_em_report``'s docstring) -- per-node ``NodeReport.q_gain`` is a
        diagnostic residual-delta decomposition and is NOT individually guaranteed non-negative
        (a mixture step can grow one component's spread while improving the joint objective), so we
        verify monotonicity on the real global objective instead of faking a per-node assertion.
        """
        truth = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        data = truth.sampler(3).sample(2000)
        estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        enc_data = seq_encode(data, model=truth)

        # Deliberately off-center initial model so the first EM step makes real progress.
        init_model = MixtureDistribution([GaussianDistribution(-1.0, 4.0), GaussianDistribution(1.0, 4.0)], [0.5, 0.5])

        model = init_model
        for _ in range(5):
            model, before, after = root_em_report(enc_data, estimator, model, max_its=1)
            self.assertGreaterEqual(after, before - 1e-9, "EM objective decreased: %r -> %r" % (before, after))

    def test_root_em_report_matches_run_em(self):
        """root_em_report's returned model matches a plain run_em call with the same inputs."""
        truth = GaussianDistribution(0.0, 1.0)
        data = truth.sampler(5).sample(300)
        estimator = GaussianEstimator()
        enc_data = seq_encode(data, model=truth)
        init_model = GaussianDistribution(1.0, 2.0)

        expected = run_em(enc_data, estimator, init_model, max_its=3, delta=None)
        got, _before, _after = root_em_report(enc_data, estimator, init_model, max_its=3)
        self.assertAlmostEqual(got.mu, expected.mu, places=10)
        self.assertAlmostEqual(got.sigma2, expected.sigma2, places=10)


if __name__ == "__main__":
    unittest.main()
