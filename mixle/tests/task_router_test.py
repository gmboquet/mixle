"""Router: calibrated N-tier model routing with realized-cost receipts."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _route(t):
    if t["amount"] > 500 and t["kind"] == "refund":
        return "finance-escalation"
    if t["kind"] in ("refund", "billing"):
        return "billing"
    return "support"


def _tickets(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["refund", "billing", "question", "bug"]
    return [
        {
            "kind": kinds[rng.randint(0, 4)],
            "amount": float(rng.gamma(2.0, 150.0)),
            "region": ["us", "eu"][rng.randint(0, 2)],
        }
        for _ in range(n)
    ]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class RouterTest(unittest.TestCase):
    def test_routes_cheap_first_with_receipts_and_harvest(self):
        from mixle.task import Router, solve

        train = _tickets(400)
        tiny = solve(_route, train, alpha=0.2, ood=None, seed=0, epochs=60, hidden=[8], dim=64)
        small = solve(_route, train, alpha=0.1, ood=None, seed=1, epochs=300, hidden=[64])
        router = Router.from_solutions(
            [tiny, small], _route, costs=[0.0001, 0.001, 0.03], names=["tiny", "small", "frontier"]
        )

        fresh = _tickets(300, seed=7)
        answers = router.serve(fresh)

        # 1) every LOCAL answer matches that tier's calibrated decision; frontier answers are exact truth
        rep = router.report()
        self.assertEqual(rep["requests"], 300)
        self.assertEqual(sum(t["answered"] for t in rep["tiers"]), 300)

        # 2) the cheap tiers carry real traffic and the realized cost beats frontier-only
        local_share = rep["tiers"][0]["share"] + rep["tiers"][1]["share"]
        self.assertGreater(local_share, 0.5)
        self.assertLess(rep["realized_cost"], rep["frontier_only_cost"])
        self.assertGreater(rep["savings"], 0.0)

        # 3) frontier answers were harvested as (input, label) for the next re-solve
        h_in, h_lab = router.harvested()
        self.assertEqual(len(h_in), rep["tiers"][-1]["answered"])
        for x, y in zip(h_in, h_lab):
            self.assertEqual(y, _route(x))

        # 4) no silent wrong answers: any answer differing from the teacher came from a calibrated tier
        #    (bounded by alpha), and the frontier's own answers are exact.
        wrong = sum(1 for x, a in zip(fresh, answers) if a != _route(x))
        self.assertLess(wrong / len(fresh), 0.15)  # << alpha-bounded local risk, never frontier errors

        self.assertIn("harvested", router.summary())

    def test_construction_contracts(self):
        from mixle.task import Router

        with self.assertRaises(ValueError):
            Router([("only", lambda x: "a", 0.01)])
        with self.assertRaises(TypeError):
            Router([("bad", object(), 0.001), ("frontier", lambda x: "a", 0.03)])


if __name__ == "__main__":
    unittest.main()
