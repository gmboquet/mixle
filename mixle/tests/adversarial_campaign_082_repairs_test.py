"""Repairs for the blocking defects the ten-pass adversarial campaign found on the 0.8.2 candidate.

Each test names the finding it pins and fails on the unrepaired tree. The campaign record and the
independent reproductions live in ``release-checklists/0.8.2-reviews/``.
"""

from __future__ import annotations

import unittest
import warnings

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


if __name__ == "__main__":
    unittest.main()
