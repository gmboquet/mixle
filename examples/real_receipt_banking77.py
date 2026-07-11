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

from mixle.task import scorecard, solve


def run(
    *,
    n_seed: int = 3000,
    n_round: int = 1000,
    n_rounds: int = 6,
    n_test: int = 1500,
    student: str = "mlp",
    verbose: bool = True,
) -> dict:
    """Run the Banking77 solve loop and return its measured results.

    The defaults reproduce the headline run in the module docstring. The sizes are parameters so a fast
    bounded version can be gated in CI (see ``real_receipt_banking77_smoke_test``); ``main`` keeps the
    full defaults. Returns ``{"card", "accuracy", "rounds"}`` where ``rounds`` is a list of
    ``{"escalation", "accuracy"}`` per improve round.
    """
    from datasets import load_dataset

    ds = load_dataset("banking77")
    names = ds["train"].features["label"].names
    train_all = [(r["text"], names[r["label"]]) for r in ds["train"]]
    test = [(r["text"], names[r["label"]]) for r in ds["test"]]
    rng = np.random.RandomState(0)
    rng.shuffle(train_all)

    gold = dict(train_all) | dict(test)

    def oracle(t: str) -> str:  # the frontier stand-in: always right, priced per call
        return gold[t]

    seed_texts = [t for t, _ in train_all[:n_seed]]
    rounds = [[t for t, _ in train_all[n_seed + i * n_round : n_seed + (i + 1) * n_round]] for i in range(n_rounds)]
    test_texts = [t for t, _ in test]

    kw = {"student": "generative", "pseudo_count": 4.0} if student == "generative" else {"epochs": 250}
    sol = solve(oracle, seed_texts, alpha=0.1, seed=0, **kw)
    card = scorecard(
        sol,
        oracle,
        test_texts[:n_test],
        student_cost=0.0001,
        teacher_cost=0.03,
        task="banking77 intents (77 classes)",
    )
    if verbose:
        print(card.table())
        print("\nescalation-decay: serve fresh queries, harvest, improve — per round")
        print("round | escalation | end-to-end accuracy")

    round_results = []
    for i, chunk in enumerate(rounds):
        if not chunk:
            continue
        before = sol.cascade.stats.n_escalated
        answers = [sol(t) for t in chunk]
        esc = (sol.cascade.stats.n_escalated - before) / len(chunk)
        acc = float(np.mean([a == gold[t] for a, t in zip(answers, chunk)]))
        round_results.append({"escalation": esc, "accuracy": acc})
        if verbose:
            print(f"  {i}   |   {esc:.3f}    |   {acc:.3f}")
        sol.improve()

    if verbose:
        print("\nevery number above was measured by this run — change the seed and check.")
    return {"card": card, "metrics": card.as_dict(), "rounds": round_results}


def main() -> None:
    import sys

    run(student="generative" if "--generative" in sys.argv else "mlp")


if __name__ == "__main__":
    main()
