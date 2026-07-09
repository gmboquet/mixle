"""Post-hoc relabeling of parallel mixture chains (label-switching identifiability)."""

import unittest
import warnings

import numpy as np

from mixle.ppl import Mix, Normal
from mixle.ppl.inference import _exchangeable_layout, _relabel_chain


class RelabelUnitTest(unittest.TestCase):
    def test_layout_detects_mixture_components(self):
        m = [Normal(0, 10, name=f"m{i}") for i in range(3)]
        model = Mix([Normal(mi, 1.0) for mi in m])
        from mixle.ppl.inference import _collect_composite

        slots, _ = _collect_composite(model)
        layout = _exchangeable_layout(slots)
        self.assertIsNotNone(layout)
        self.assertEqual(len(layout), 3)  # three exchangeable components

    def test_layout_none_for_non_mixture(self):
        slots, _ = __import__("mixle.ppl.inference", fromlist=["_collect_composite"])._collect_composite(
            Normal(Normal(0, 1, name="mu"), 1.0)
        )
        self.assertIsNone(_exchangeable_layout(slots))  # nothing exchangeable

    def test_relabel_sorts_components_per_draw(self):
        # two single-parameter components at columns 0 and 1; relabel must sort each draw ascending
        layout = [[0], [1]]
        u = np.array([[5.0, 1.0], [2.0, 8.0], [9.0, 3.0]])
        out = _relabel_chain(u, layout)
        np.testing.assert_allclose(out, [[1.0, 5.0], [2.0, 8.0], [3.0, 9.0]])


class RelabelParallelChainsTest(unittest.TestCase):
    def test_parallel_mixture_chains_agree_after_relabeling(self):
        rng = np.random.RandomState(2)
        data = np.concatenate([rng.normal(-4, 1.2, 120), rng.normal(0, 1.2, 120), rng.normal(4, 1.2, 120)])
        m = [Normal(0, 10, name=f"m{i}") for i in range(3)]
        model = Mix([Normal(mi, 1.2) for mi in m])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # chains=6 is kept (relabeling reliability benefits from more independent label
            # permutations); draws/burn trimmed -- verified stable (means within ~0.05 of the
            # true modes, rhat ~1.0-1.005) across 10 seeds at this smaller budget.
            fit = model.fit(
                data, how="nuts", draws=150, burn=200, chains=6, parallel=False, rng=np.random.RandomState(7)
            )
        s = fit.summary()
        means = sorted(s[f"m{i}"]["mean"] for i in range(3))
        # without relabeling these chains disagree (R-hat ~25, means smeared to ~0); after relabeling
        # the pooled posterior recovers the three true modes and R-hat collapses to ~1.
        self.assertLess(abs(means[0] - (-4)), 0.6)
        self.assertLess(abs(means[1] - 0), 0.6)
        self.assertLess(abs(means[2] - 4), 0.6)
        for r in (s.get("_rhat") or {}).values():
            self.assertLess(r, 1.1)


if __name__ == "__main__":
    unittest.main()
