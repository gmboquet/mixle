"""Experimental design for reasoning (mixle.reason.design): budgeted, multi-fidelity, non-myopic acquisition.

Batch acquisition must respect the budget, avoid redundant evidence (adaptive greedy re-scores against the
updated belief, unlike a naive top-k), and trade fidelity against cost.
"""

import unittest

import numpy as np

from mixle.reason import GaussianBelief, LinearGaussianEvidence, select_evidence_batch
from mixle.reason.store import CrossModalStore, _apply, _query_entropy

D = 4  # latent dimension


def _evidence(component: int, value: float, r: float) -> LinearGaussianEvidence:
    h = np.zeros((1, D))
    h[0, component] = 1.0
    return LinearGaussianEvidence(H=h, y=np.array([value]), R=np.array([[r]]))


def _store(payloads):
    # payload = (component, value); coarse = lossy (high R), fine = precise (low R)
    keys = np.random.RandomState(0).randn(len(payloads), 3)
    return CrossModalStore(
        keys,
        payloads,
        coarse=lambda p: _evidence(p[0], p[1], r=2.0),
        fine=lambda p: _evidence(p[0], p[1], r=0.05),
    )


def _prior():
    return GaussianBelief(np.zeros(D), np.eye(D) * 5.0)


class BudgetTest(unittest.TestCase):
    def test_respects_the_budget(self):
        store = _store([(i % D, float(i)) for i in range(12)])
        plan = select_evidence_batch(store, _prior(), budget=1.0, fine_cost=1.0, coarse_cost=0.2)
        self.assertLessEqual(plan.total_cost, 1.0 + 1e-9)
        self.assertGreater(plan.total_gain, 0.0)
        self.assertTrue(all(f in ("coarse", "fine") for _i, f, _g, _c in plan.items))


class NonMyopicTest(unittest.TestCase):
    def test_avoids_redundant_evidence(self):
        # three payloads all observing the SAME component with the same value -> redundant
        redundant = [(0, 3.0), (0, 3.0), (0, 3.0)] + [(1, -2.0), (2, 4.0)]
        store = _store(redundant)
        plan = select_evidence_batch(store, _prior(), budget=10.0, fine_cost=1.0, coarse_cost=1.0, max_items=3)
        picked_components = [store.payloads[i][0] for i in plan.indices]
        # adaptive greedy diversifies across components instead of picking the three identical ones
        self.assertEqual(len(set(picked_components)), len(picked_components))

    def test_beats_naive_topk_on_total_gain(self):
        # a mix of redundant and complementary evidence; adaptive batch should gain >= naive prior-ranked top-k
        payloads = [(0, 3.0), (0, 3.0), (1, -2.0), (2, 4.0), (3, 1.0)]
        store = _store(payloads)
        prior = _prior()
        k = 3
        adaptive = select_evidence_batch(store, prior, budget=100.0, fine_cost=1.0, coarse_cost=1.0, max_items=k)

        # naive: rank all items by their single-step gain against the PRIOR, take top-k, then assimilate
        before = _query_entropy(prior, None)
        ranked = sorted(
            range(len(payloads)),
            key=lambda i: before - _query_entropy(_apply(prior, store.fine(payloads[i])), None),
            reverse=True,
        )[:k]
        b = prior
        for i in ranked:
            b = _apply(b, store.fine(payloads[i]))
        naive_gain = before - _query_entropy(b, None)
        self.assertGreaterEqual(adaptive.total_gain, naive_gain - 1e-6)


class FidelityTest(unittest.TestCase):
    def test_prefers_cheap_coarse_under_a_tight_budget(self):
        # coarse is much cheaper; under a tight budget the plan should favour coarse to buy more information
        store = _store([(i % D, float(i)) for i in range(8)])
        plan = select_evidence_batch(store, _prior(), budget=0.8, fine_cost=1.0, coarse_cost=0.2)
        fidelities = [f for _i, f, _g, _c in plan.items]
        self.assertIn("coarse", fidelities)
        self.assertGreaterEqual(len(plan.items), 2)  # cheap fidelity let it acquire several items


if __name__ == "__main__":
    unittest.main()
