"""The whole money loop on SYNTHETIC STAND-IN DATA: distill, gate, serve, and MEASURE whether use made it cheaper.

Everything here is synthetic: the "frontier teacher" is a deterministic keyword rule and the
traffic is generated -- the point is the ACCOUNTING METHOD, not the specific dollar amounts.

GPU time is not free, so the question is always "what can I serve locally, and what must I pay the frontier
for?" This runs the full mixle.task spine end to end:

  1. distill a slow/expensive teacher into a tiny local classifier;
  2. calibrate it with conformal sets (honest answer-vs-escalate) + a generative density gate (escalate inputs
     it has never seen -- the p(x) a softmax cannot represent);
  3. serve a Cascade: answer locally when confident, escalate only the rest to the teacher;
  4. report realized dollars saved vs paying the frontier for every request;
  5. harvest the escalated items (free targeted labels), re-distill, and test "the cascade got
     cheaper with use" as a PAIRED comparison: both fixed cascades decide the same fresh
     requests, and the conclusion is gated on PER-STRATUM exact paired (McNemar) tests --
     the corpus fixes 150 rows per class, so the strata are the exchangeable units and a
     pooled "exact" test would overstate what the design supports (STAT-RR17-03); both
     strata must agree at the 5% level.
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
)

SPAM = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
HAM = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
FILLER = ["the", "a", "today", "tomorrow", "please", "thanks", "we", "you"]


def exact_paired_pvalue(first_only: int, second_only: int) -> float:
    """Exact two-sided paired (McNemar) p-value from the two discordant counts."""
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = max(first_only, second_only)
    from math import comb

    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


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


class CountingTeacher:
    """The paid teacher with an odometer: every label the pipeline buys is counted, no exceptions."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += len(texts)
        return expensive_teacher(texts)


def build_cascade(teacher, train, train_labels, cal, cal_labels, cost):
    """Distill/calibrate from ALREADY-PAID labels -- harvested labels are reused, never re-bought."""
    from mixle.task import distill_from_labels

    student = distill_from_labels(
        train, train_labels, n=4, dim=512, hidden=[64], epochs=250, seed=0, task="spam vs ham"
    )
    gate = DensityGate(HashedNGram(n=3, dim=48, seed=1)).fit(train, n_components=3, seed=0)
    model = CalibratedTaskModel(student, alpha=0.1, density_gate=gate).calibrate(cal, cal_labels)
    return Cascade(model, teacher, cost=cost)


def main() -> None:
    print("SYNTHETIC DEMO: keyword-rule teacher, generated traffic -- the accounting method is the point\n")
    cost = CostModel(c_frontier=0.01, c_local=0.00001, c_label=0.01, train_cost=0.0)  # $/request
    teacher = CountingTeacher()
    train, cal, traffic = corpus(1), corpus(2), corpus(seed=900)

    print("round 1: distill + calibrate, then serve the cascade")
    train_labels = teacher(train)  # 600 paid setup labels (train + cal), counted in the estimand
    cal_labels = teacher(cal)
    setup_calls_1 = teacher.calls
    casc = build_cascade(teacher, train, train_labels, cal, cal_labels, cost)
    casc.serve(traffic)
    rep = casc.report()
    print(
        f"   served {rep['n_requests']} requests, escalated {rep['n_escalated']} "
        f"({rep['realized_escalation_rate']:.1%})"
    )
    all_in_1 = rep["realized_cost"] + setup_calls_1 * cost.c_label
    print(
        f"   serving spent ${rep['realized_cost']:.3f}; ALL-IN round-1 cost including "
        f"{setup_calls_1} setup labels is ${all_in_1:.3f} vs ${rep['frontier_only_cost']:.2f} frontier-only"
    )
    if all_in_1 > rep["frontier_only_cost"]:
        print(
            f"   -> round 1 COSTS ${all_in_1 - rep['frontier_only_cost']:.3f} MORE than frontier-only: "
            "setup amortizes only over later traffic"
        )

    print("\nharvest the escalated requests (already-paid labels) and re-distill WITHOUT re-buying them")
    htexts, hlabels = casc.harvested()
    print(f"   harvested {len(htexts)} teacher-labeled examples from escalations (0 new calls)")

    print("\nround 2: re-distill REUSING the harvest, then compare the two cascades HEAD-TO-HEAD")
    calls_before_build2 = teacher.calls
    casc2 = build_cascade(teacher, train + htexts, train_labels + list(hlabels), cal, cal_labels, cost)
    print(f"   round-2 build bought {teacher.calls - calls_before_build2} new labels (harvest reused)")
    # The ESTIMAND, stated before the comparison: the difference in escalation probability
    # between the two REALIZED cascades over the request distribution corpus() draws from.
    # corpus() FIXES 150 rows per class (stratified, not i.i.d.), so the honest exact statement
    # is PER STRATUM: within each class the paired discordances are exchangeable under the null,
    # and each class gets its own exact McNemar test (STAT-RR17-03 -- the pooled test is not
    # exact for the overall mean under a fixed-stratum design without a stronger stratumwise
    # null). The conclusion requires BOTH strata to agree at the 5% level.
    evaluation = corpus(seed=901)
    truth = expensive_teacher(evaluation)
    escalated_1 = np.asarray([d is ESCALATE for d in casc.model.batch_decide(evaluation)], dtype=bool)
    escalated_2 = np.asarray([d is ESCALATE for d in casc2.model.batch_decide(evaluation)], dtype=bool)
    print(
        f"   on {len(evaluation)} fresh paired requests (150 per class, FIXED by design): "
        f"round-1 escalates {escalated_1.mean():.1%}, round-2 {escalated_2.mean():.1%}"
    )
    verdicts = []
    for label in ("spam", "ham"):
        stratum = np.asarray([t == label for t in truth])
        only_1 = int(np.sum(escalated_1 & ~escalated_2 & stratum))
        only_2 = int(np.sum(~escalated_1 & escalated_2 & stratum))
        p_exact = exact_paired_pvalue(only_1, only_2)
        verdicts.append(p_exact < 0.05 and only_1 > only_2)
        print(
            f"   {label}: discordant {only_1} round-1-only vs {only_2} round-2-only; "
            f"exact paired two-sided p = {p_exact:.4f}"
        )
    if all(verdicts):
        print("   -> BOTH strata improved (exact paired evidence at the 5% level in each):")
        print("      the re-distilled cascade escalates less on identical traffic")
    else:
        print("   -> the per-stratum exact evidence is inconclusive at the 5% level; serve more")
        print("      traffic before claiming it got cheaper (no pooled shortcut: the corpus is")
        print("      stratified by design, so the strata are the honest units)")

    print("\nserve round 2 on the fresh traffic and project the cheapest route at 1,000,000 requests")
    casc2.serve(evaluation)

    plan = casc2.plan(volume=1_000_000, n_label=len(train))
    print(
        f"   recommended: {plan.route}  per-request ${plan.per_request:.5f}  "
        f"saves ${plan.savings_vs_frontier:,.0f} vs frontier-only"
    )
    print("   (a POINT projection at this run's realized escalation rate; it assumes the served mix")
    print("    stays exchangeable with this traffic -- drift re-prices it, so re-measure before acting)")
    print(f"\ntotal paid teacher calls this whole demo: {teacher.calls}")


if __name__ == "__main__":
    main()
