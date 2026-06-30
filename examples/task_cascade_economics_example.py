"""The whole money loop: distill a local model, gate it honestly, serve a cascade, and watch the cost fall.

GPU time is not free, so the question is always "what can I serve locally, and what must I pay the frontier
for?" This runs the full mixle.task spine end to end:

  1. distill a slow/expensive teacher into a tiny local classifier;
  2. calibrate it with conformal sets (honest answer-vs-escalate) + a generative density gate (escalate inputs
     it has never seen -- the p(x) a softmax cannot represent);
  3. serve a Cascade: answer locally when confident, escalate only the rest to the teacher;
  4. report realized dollars saved vs paying the frontier for every request;
  5. harvest the escalated items (free targeted labels) and re-distill -- the cascade gets cheaper with use.

Run: ``python task_cascade_economics_example.py``  (needs ``pip install "mixle[torch]"``).
"""

from __future__ import annotations

import numpy as np

from mixle.task import (
    CalibratedTaskModel,
    Cascade,
    CostModel,
    DensityGate,
    HashedNGram,
    distill,
)

SPAM = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
HAM = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
FILLER = ["the", "a", "today", "tomorrow", "please", "thanks", "we", "you"]


def corpus(seed: int, n_per_class: int = 150) -> list[str]:
    r = np.random.RandomState(seed)
    out = []
    for words in (SPAM, HAM):
        for _ in range(n_per_class):
            toks = list(r.choice(words, size=2)) + list(r.choice(FILLER, size=r.randint(3, 7)))
            r.shuffle(toks)
            out.append(" ".join(toks))
    r.shuffle(out)
    return out


def expensive_teacher(texts: list[str]) -> list[str]:
    """Stand-in for a frontier model / human: the ground truth, but it costs real money per call."""
    s = set(SPAM)
    return ["spam" if any(w in t.split() for w in s) else "ham" for t in texts]


def build_cascade(train, cal, cost):
    student = distill(expensive_teacher, train, n=4, dim=512, hidden=[64], epochs=250, seed=0, task="spam vs ham")
    gate = DensityGate(HashedNGram(n=3, dim=48, seed=1)).fit(train, n_components=3, seed=0)
    model = CalibratedTaskModel(student, alpha=0.1, density_gate=gate).calibrate(cal, expensive_teacher(cal))
    return Cascade(model, expensive_teacher, cost=cost)


def main() -> None:
    cost = CostModel(c_frontier=0.01, c_local=0.00001, c_label=0.01, train_cost=0.0)  # $/request
    train, cal, traffic = corpus(1), corpus(2), corpus(seed=900)

    print("round 1: distill + calibrate, then serve the cascade")
    casc = build_cascade(train, cal, cost)
    casc.serve(traffic)
    rep = casc.report()
    print(
        f"   served {rep['n_requests']} requests, escalated {rep['n_escalated']} "
        f"({rep['realized_escalation_rate']:.1%})"
    )
    print(
        f"   spent ${rep['realized_cost']:.2f} vs ${rep['frontier_only_cost']:.2f} frontier-only "
        f"-> saved ${rep['savings_vs_frontier']:.2f}"
    )

    print("\nharvest the escalated requests (free targeted labels) and re-distill")
    htexts, hlabels = casc.harvested()
    print(f"   harvested {len(htexts)} teacher-labeled examples from escalations")

    print("\nround 2: re-distill including the harvest, serve fresh traffic")
    casc2 = build_cascade(train + htexts, cal, cost)
    casc2.serve(corpus(seed=901))
    rep2 = casc2.report()
    print(
        f"   escalation {rep2['realized_escalation_rate']:.1%} "
        f"(was {rep['realized_escalation_rate']:.1%}) -> the cascade gets cheaper with use"
    )

    print("\nproject the cheapest route at 1,000,000 requests")
    plan = casc2.plan(volume=1_000_000, n_label=len(train))
    print(
        f"   recommended: {plan.route}  per-request ${plan.per_request:.5f}  "
        f"saves ${plan.savings_vs_frontier:,.0f} vs frontier-only"
    )


if __name__ == "__main__":
    main()
