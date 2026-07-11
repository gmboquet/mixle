"""The pitch, runnable: replace rigid code with tiny calibrated models — and show the receipts.

Classification: illustrative -- runs on small synthetic / stand-in data. It shows the
end-to-end workflow shape, not measured results on a real frontier-scale dataset. See
docs/example-execution-manifest.rst for which examples run on real public data.

A support-ops story in three acts, all synthetic and self-contained (no downloads, no API keys):

  1. ``solve``   — a 400-line-style if/elif ticket router becomes a ~65KB model with a conformal
                   answer-or-escalate gate; ``scorecard`` prints the measured receipts.
  2. ``Router``  — a tiny + small stack in front of the "frontier" (the rule, priced like an LLM);
                   the report shows realized $/request and savings, not projections.
  3. ``replace_extractor`` — the invoice regex scraper becomes a token-level tagger with a
                   required-field fallback.

Run it: ``python examples/win_demo_example.py`` (~1 minute on a laptop).
"""

from __future__ import annotations

import re

import numpy as np

from mixle.task import Router, replace_extractor, scorecard, solve


# --- the "rigid code" being replaced --------------------------------------------------------------------
def route_ticket(t: dict) -> str:
    if t["amount"] > 500 and t["kind"] == "refund":
        return "finance-escalation"
    if t["kind"] in ("refund", "billing"):
        return "billing"
    return "support"


def scrape_invoice(text: str) -> dict:
    m = re.search(r"order (\d+) .* total (\d+\.\d+)", text)
    return {"id": m.group(1), "amount": m.group(2)} if m else {}


def tickets(n: int, seed: int = 0) -> list[dict]:
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


def main() -> None:
    print("=" * 76)
    print("1) replace the ticket router with solve() — and show the receipts")
    print("=" * 76)
    sol = solve(route_ticket, tickets(400), alpha=0.1, seed=0, epochs=300)
    card = scorecard(
        sol, route_ticket, tickets(200, seed=9), student_cost=0.0001, teacher_cost=0.03, task="ticket routing"
    )
    print(card.table())

    print()
    print("=" * 76)
    print("2) a calibrated router: tiny -> small -> frontier, with realized economics")
    print("=" * 76)
    tiny = solve(route_ticket, tickets(400), alpha=0.2, ood=None, seed=0, epochs=60, hidden=[8], dim=64)
    small = solve(route_ticket, tickets(400), alpha=0.1, ood=None, seed=1, epochs=300)
    router = Router.from_solutions(
        [tiny, small], route_ticket, costs=[0.00005, 0.0005, 0.03], names=["tiny", "small", "frontier"]
    )
    router.serve(tickets(400, seed=11))
    print(router.summary())

    print()
    print("=" * 76)
    print("3) replace the invoice regex with a learned extractor (+ fallback)")
    print("=" * 76)
    rng = np.random.RandomState(0)
    texts = [
        f"order {rng.randint(100, 999)} placed by user{u} total {rng.randint(1, 99)}.{rng.randint(10, 99)}"
        for u in range(120)
    ]
    ex = replace_extractor(scrape_invoice, texts, ["id", "amount"], seed=0, epochs=40)
    demo = "order 314 placed by user7 total 15.99"
    print(f"holdout field-F1 vs the regex: {ex.holdout_f1:.3f}")
    print(f"extract({demo!r}) -> {ex(demo)}")
    print(f"report: {ex.report()}")

    print()
    print("every number above was measured in this run — rerun with different seeds and check.")


if __name__ == "__main__":
    main()
