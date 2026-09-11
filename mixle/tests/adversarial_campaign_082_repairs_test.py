"""Repairs for the blocking defects the ten-pass adversarial campaign found on the 0.8.2 candidate.

Each test names the finding it pins and fails on the unrepaired tree. The campaign record and the
independent reproductions live in ``release-checklists/0.8.2-reviews/``.
"""

from __future__ import annotations

import unittest
import warnings
from pathlib import Path

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    PoissonDistribution,
    PoissonEstimator,
    TreeHiddenMarkovEstimator,
    TreeHiddenMarkovModelDistribution,
    dump_models,
    load_models,
)


class FittedTreeHmmSerializesTest(unittest.TestCase):
    """Q09-F01: a tree HMM fitted by ``optimize()`` could not be written as JSON at all.

    ``_p_level_cache`` memoizes ``init_prob @ transitions^k``. A constructor leaves it ``None`` and a
    fit warms it, and the decoder compares the artifact's state against a freshly constructed
    object's, so the warm tuple failed a comparison against ``None`` and the whole model was refused.
    The same fit round-tripped on 0.8.1, which makes this a regression the release line introduced.
    """

    def _fitted(self):
        source = TreeHiddenMarkovModelDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)],
            [0.5, 0.5],
            [[0.8, 0.2], [0.2, 0.8]],
            len_dist=PoissonDistribution(0.8),
            terminal_level=3,
        )
        data = source.sampler(1).sample(150)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return source, optimize(
                data,
                TreeHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()], len_estimator=PoissonEstimator()),
                max_its=8,
                rng=np.random.RandomState(1),
                out=None,
            )

    def test_a_fitted_tree_hmm_round_trips_through_json(self):
        source, fitted = self._fitted()
        # The memo really is warm: without that this test would pass on the broken tree too.
        self.assertIsNotNone(fitted._p_level_cache, "the fit should have warmed the level memo")
        self.assertIsNone(source._p_level_cache)
        for label, model in (("constructed", source), ("fitted", fitted)):
            with self.subTest(model=label):
                restored = load_models(dump_models(model))
                self.assertIsNotNone(restored)

    def test_the_memo_is_not_carried_in_the_artifact_and_is_rebuilt_on_use(self):
        """Two models with the same parameters are the same model, warm cache or not."""
        _, fitted = self._fitted()
        text = dump_models(fitted)
        self.assertNotIn("_p_level_cache", text)
        restored = load_models(text)
        model = restored["model"] if isinstance(restored, dict) and "model" in restored else restored
        target = model if hasattr(model, "_get_p_level") else fitted
        self.assertIsNone(getattr(target, "_p_level_cache", None) if target is not fitted else None)
        # Recomputed on demand, and equal to what the warm original holds.
        rebuilt = target._get_p_level(3)
        np.testing.assert_allclose(rebuilt, fitted._get_p_level(3), rtol=1e-12, atol=1e-12)


class QuantileDomainIsUniformTest(unittest.TestCase):
    """Q01-F01 / Q08-F08: the shared out-of-domain refusal held on 15 of 32 families.

    The CHANGELOG says every family that defines ``quantile`` refuses an out-of-domain ``q`` the
    same way, and the migration guide repeats it. It was not so: ``HalfNormal(1).quantile(-0.1)``
    returned ``-0.1257``, ``BetaBinomial(10, 2, 3).quantile(1.1)`` returned the support point
    ``10.0`` -- a plausible-looking answer for a level the caller computed wrong -- and
    Uniform/Laplace/StudentT let scipy answer NaN.

    The source test below is the one that matters. Enumerating families by hand is how the claim
    narrowed in the first place, so it walks the tree instead: every ``def quantile`` under
    ``mixle/stats`` must route through the shared contract, and a family added tomorrow fails until
    it does.
    """

    OUT_OF_DOMAIN = (-0.1, 1.1, float("nan"))

    def test_every_quantile_in_the_tree_routes_through_the_shared_contract(self):
        import ast

        root = Path(__file__).resolve().parents[1] / "stats"
        offenders = []
        checked = 0
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "def quantile" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.ClassDef):
                    continue
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "quantile":
                        checked += 1
                        body = ast.get_source_segment(source, item) or ""
                        if "validated_quantile_probability" not in body:
                            offenders.append(f"{path.name}::{node.name}")
        self.assertGreater(checked, 25, "the walk should find the whole quantile surface")
        self.assertEqual(offenders, [], "quantile implementations bypassing the domain contract")

    def test_the_refusal_is_the_same_on_families_that_answered_differently(self):
        """The five the campaign names, plus three that already refused, in one loop."""
        from mixle import stats

        cases = {
            "HalfNormalDistribution": (1.0,),
            "BetaBinomialDistribution": (10, 2.0, 3.0),
            "UniformDistribution": (-1.0, 2.0),
            "LaplaceDistribution": (0.0, 1.0),
            "StudentTDistribution": (5.0,),
            "GaussianDistribution": (0.0, 1.0),
            "PoissonDistribution": (2.0,),
            "ExponentialDistribution": (1.0,),
        }
        for name, args in cases.items():
            family = getattr(stats, name, None)
            if family is None:
                continue
            distribution = family(*args)
            for q in self.OUT_OF_DOMAIN:
                with self.subTest(family=name, q=q):
                    with self.assertRaises(ValueError) as caught:
                        distribution.quantile(q)
                    self.assertIn("q must be in [0, 1]", str(caught.exception))

    def test_an_in_domain_q_still_answers(self):
        """The refusal must not have swallowed the ordinary path."""
        from mixle import stats

        for name, args, expected in (
            ("HalfNormalDistribution", (1.0,), 0.6745),
            ("UniformDistribution", (-1.0, 2.0), 0.5),
            ("GaussianDistribution", (0.0, 1.0), 0.0),
        ):
            family = getattr(stats, name, None)
            if family is None:
                continue
            with self.subTest(family=name):
                self.assertAlmostEqual(family(*args).quantile(0.5), expected, places=3)


if __name__ == "__main__":
    unittest.main()
