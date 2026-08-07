"""The whole money loop: distill a local model, gate it honestly, serve a cascade, and MEASURE whether use made it cheaper.

GPU time is not free, so the question is always "what can I serve locally, and what must I pay the frontier
for?" This runs the full mixle.task spine end to end:

  1. distill a slow/expensive teacher into a tiny local classifier;
  2. calibrate it with conformal sets (honest answer-vs-escalate) + a generative density gate (escalate inputs
     it has never seen -- the p(x) a softmax cannot represent);
  3. serve a Cascade: answer locally when confident, escalate only the rest to the teacher;
  4. report realized dollars saved vs paying the frontier for every request;
  5. harvest the escalated items (free targeted labels), re-distill, and test "the cascade got
     cheaper with use" as a PAIRED comparison: both fixed cascades decide the same fresh
     requests, and the conclusion is gated on a 95% interval for the paired difference.
     (Comparing round 1's serving rate against round 2's would be invalid: round 2 is trained
     on escalations harvested FROM round 1's traffic, so those two rate estimates are coupled
     through the training data, and a two-independent-proportions interval does not apply.)

Scope: the ECONOMICS of the mixle.task loop. Start at ``task_distill_example.py`` for the plain
distill/save/reload story; ``task_llm_active_example.py`` covers the labeling-budget side.

Run: ``python examples/task_cascade_economics_example.py``  (needs ``pip install "mixle[torch]"``).
"""

from __future__ import annotations

import numpy as np

from mixle.task import (
    ESCALATE,
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

    print("\nround 2: re-distill including the harvest, then compare the two cascades HEAD-TO-HEAD")
    casc2 = build_cascade(train + htexts, cal, cost)
    # The ESTIMAND, stated before the comparison: the difference in escalation probability
    # between the two REALIZED cascades (round 1's and round 2's, both fully fitted above) over
    # the request distribution that corpus() draws from i.i.d. The comparison must be PAIRED on
    # fresh traffic: round 2 was trained on escalations harvested from round 1's served traffic,
    # so round 1's serving rate and round 2's rate are coupled through that shared sample and a
    # two-independent-proportions interval would rest on a false premise. Instead, both FIXED
    # cascades decide the same evaluation requests -- traffic no fitting step ever saw -- and,
    # conditional on the two fits, the per-request paired differences are i.i.d., giving a valid
    # normal-approximation 95% interval for the mean difference.
    evaluation = corpus(seed=901)
    escalated_1 = np.asarray([d is ESCALATE for d in casc.model.batch_decide(evaluation)], dtype=np.float64)
    escalated_2 = np.asarray([d is ESCALATE for d in casc2.model.batch_decide(evaluation)], dtype=np.float64)
    paired = escalated_1 - escalated_2
    difference = float(paired.mean())
    standard_error = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    low, high = difference - 1.96 * standard_error, difference + 1.96 * standard_error
    print(
        f"   on {len(evaluation)} fresh paired requests: round-1 escalates {escalated_1.mean():.1%}, "
        f"round-2 {escalated_2.mean():.1%}"
    )
    print(f"   paired difference {difference:+.1%}, 95% CI [{low:+.1%}, {high:+.1%}]")
    if low > 0.0:
        print("   -> the re-distilled cascade escalates less on identical traffic: it measurably")
        print("      got cheaper with use")
    else:
        print("   -> consistent with no change at this sample size; serve more traffic before")
        print("      claiming it got cheaper")

    print("\nserve round 2 on the fresh traffic and project the cheapest route at 1,000,000 requests")
    casc2.serve(evaluation)

    plan = casc2.plan(volume=1_000_000, n_label=len(train))
    print(
        f"   recommended: {plan.route}  per-request ${plan.per_request:.5f}  "
        f"saves ${plan.savings_vs_frontier:,.0f} vs frontier-only"
    )


if __name__ == "__main__":
    main()
