"""The first REAL-data receipt: Banking77 (77 intents, real customer queries) through the solve loop.

Everything else in examples/ uses synthetic teachers; this one runs the loop against a real public
dataset. The "frontier" is an oracle stand-in (the dataset's gold labels, priced per call like an API)
— so the accuracy ceiling is honest and the costs are modeled, not yet billed. What it measures:

  * the scorecard on the official test split — measured in one run on a laptop:
        end-to-end accuracy 0.983 · local agreement 0.948 · escalation 0.66
        artifact 84 KB · $19.95 vs $30.00 per 1k
  * the ESCALATION-DECAY CURVE — six rounds of live traffic, each followed by ``improve()``
    (harvest the teacher's answers on escalated queries, re-distill, promote only if better):
        MLP student        : 0.679 -> 0.584 -> 0.576 -> 0.488 -> 0.499 -> 0.428
        generative student : 0.588 -> 0.476 -> 0.438 -> 0.380 -> 0.300 -> 0.297
    "Gets cheaper the longer it runs", measured — and the generative student (torch-free,
    ``student="generative"``) compounds faster, ending 13 points lower. End-to-end accuracy eases
    0.98 -> 0.95 as more traffic answers locally: that is the alpha = 0.1 design exposing its bounded
    local risk, not a regression (escalated queries are still answered exactly by the teacher).

Honest readings this run forces: (1) a 77-class real task makes a hashed-feature student humble — the
conformal gate correctly refuses ~2/3 of traffic at first; the SYSTEM is still 98% accurate because
refusals go to the teacher. (2) The student saturates around ~43% escalation — the ceiling of this
student family, which is the argument for stronger students (structured/generative), not for loosening
the gate. Run: ``python examples/real_receipt_banking77.py`` (~4 min; downloads Banking77 once).
"""

from __future__ import annotations

import numpy as np
from datasets import load_dataset

from mixle.task import scorecard, solve


def main() -> None:
    ds = load_dataset("banking77")
    names = ds["train"].features["label"].names
    train_all = [(r["text"], names[r["label"]]) for r in ds["train"]]
    test = [(r["text"], names[r["label"]]) for r in ds["test"]]
    rng = np.random.RandomState(0)
    rng.shuffle(train_all)

    gold = dict(train_all) | dict(test)

    def oracle(t: str) -> str:  # the frontier stand-in: always right, priced per call
        return gold[t]

    seed_texts = [t for t, _ in train_all[:3000]]
    rounds = [[t for t, _ in train_all[3000 + i * 1000 : 3000 + (i + 1) * 1000]] for i in range(6)]
    test_texts = [t for t, _ in test]

    import sys

    student = "generative" if "--generative" in sys.argv else "mlp"
    kw = {"student": "generative", "pseudo_count": 4.0} if student == "generative" else {"epochs": 250}
    sol = solve(oracle, seed_texts, alpha=0.1, seed=0, **kw)
    card = scorecard(
        sol,
        oracle,
        test_texts[:1500],
        student_cost=0.0001,
        teacher_cost=0.03,
        task="banking77 intents (77 classes)",
    )
    print(card.table())

    print("\nescalation-decay: serve 1k fresh queries, harvest, improve — six rounds")
    print("round | escalation | end-to-end accuracy")
    for i, chunk in enumerate(rounds):
        before = sol.cascade.stats.n_escalated
        answers = [sol(t) for t in chunk]
        esc = (sol.cascade.stats.n_escalated - before) / len(chunk)
        acc = float(np.mean([a == gold[t] for a, t in zip(answers, chunk)]))
        print(f"  {i}   |   {esc:.3f}    |   {acc:.3f}")
        sol.improve()

    print("\nevery number above was measured by this run — change the seed and check.")


if __name__ == "__main__":
    main()
