"""LLM teacher + active labeling: spend a FIXED label budget on the least-certain examples, then serve locally.

The two differentiators in one run:

  * the teacher is an **LLM** (here a local ``CallableLLM``; swap in ``OpenAICompatLLM(base_url, model)``
    to use Ollama / vLLM / a hosted endpoint unchanged);
  * **active labeling** (DoE applied to the labeling decision) queries that LLM for the examples
    the current student is least sure about (margin acquisition). The comparison below is a SAME-TRAINING-BUDGET one -- both policies spend
    exactly the same number of paid training calls (the shared evaluation labels are paid too, and
    counted) -- and the measured quantity is held-out agreement with this synthetic deterministic
    teacher, decided by the EXACT paired test on the discordant pairs. (A labels-to-target
    curve -- how many calls each policy needs to REACH a fixed quality -- is a different experiment;
    the EIG/BALD curve in ``label_economics_demo.py`` is the acquisition-level version of it.)

Then the distilled student is wrapped in a calibrated cascade and the realized savings are reported.

Scope: the LABELING-BUDGET side of the mixle.task loop. ``task_distill_example.py`` is the plain
distill/save/reload entry point and ``task_cascade_economics_example.py`` prices the serving side;
``label_economics_demo.py`` explores the acquisition-level EIG/BALD side (note its headline
ratio compares one EIG run against unrelated random-seed runs -- a demonstration, not a paired
replicated estimand like the comparison here).

Run: ``python examples/task_llm_active_example.py``  (needs ``pip install "mixle[torch]"``).
"""

from __future__ import annotations

import numpy as np

from mixle.task import (
    CalibratedTaskModel,
    CallableLLM,
    Cascade,
    CostModel,
    active_distill,
    llm_labeler,
)

SPAM = ["free", "winner", "prize", "buy", "cheap", "offer", "click", "loan", "casino"]
HAM = ["meeting", "lunch", "project", "report", "schedule", "team", "review", "invoice"]
FILLER = ["the", "a", "today", "please", "thanks", "we", "you", "and", "to"]


def pool(seed, n_per_class=300):
    r = np.random.RandomState(seed)
    out = []
    for words in (SPAM, HAM):
        for _ in range(n_per_class):
            toks = list(r.choice(words, size=2)) + list(r.choice(FILLER, size=r.randint(3, 8)))
            r.shuffle(toks)
            out.append(" ".join(toks))
    r.shuffle(out)
    return out


def exact_paired_pvalue(first_only: int, second_only: int) -> float:
    """Exact two-sided paired (McNemar) p-value from the two discordant-pair counts.

    Under the null of equal policies, each discordant pair favors either side with probability
    one half, so the p-value is the two-sided binomial tail at 0.5 over the discordances. This
    is the example's CONCLUSION RULE: it stays valid at any discordance count, including the
    tiny ones where a Wald interval on the paired mean is unreliable.
    """
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = max(first_only, second_only)
    from math import comb

    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def local_llm(prompt, system=None):
    """Deterministic local teacher with the same callable shape as an LLM endpoint."""
    text = prompt.split("Text:", 1)[-1].lower()
    return "spam" if any(w in text.split() for w in SPAM) else "ham"


def main() -> None:
    # the teacher is an LLM, constrained to the label set
    teacher = llm_labeler(CallableLLM(local_llm), ["spam", "ham"], instruction="Classify the email as spam or ham.")
    recipe = {"n": 4, "dim": 512, "hidden": [64], "epochs": 200, "lr": 1e-2}

    p, val = pool(1), pool(seed=900)[:300]
    truth = teacher(val)  # 300 evaluation labels: PAID teacher calls, shared by both policies

    budget = 60
    print(f"training-label budget: {budget} LLM calls per policy (out of {len(p)} unlabeled)")
    print(f"shared evaluation labels: {len(val)} LLM calls (paid; score both students, train neither)")
    print(f"teacher calls for this comparison: {2 * budget + len(val)} (2 x {budget} training + {len(val)} evaluation)")
    active = active_distill(teacher, p, budget=budget, seed_size=20, rounds=4, acquisition="margin", recipe=recipe)
    rand = active_distill(teacher, p, budget=budget, seed_size=20, rounds=4, acquisition="random", recipe=recipe)
    # The ESTIMAND, stated before the numbers: each realized student's agreement with the
    # deterministic teacher over the synthetic pool() population, compared at the SAME
    # training-label budget, conditional on the two fitted students. Both students predict the
    # SAME 300 fresh validation rows. The decision rule is the EXACT paired (McNemar) test on
    # the discordant pairs -- a normal-approximation interval is unreliable exactly where this
    # comparison lives (an external review measured a Wald gate declaring superiority at four
    # discordant pairs, where the exact two-sided p-value is 0.125). "Agreement", not
    # "accuracy": the teacher is the reference, and the data are synthetic.
    active_hits = np.asarray([a == b for a, b in zip(active.model.batch(val), truth)], dtype=np.float64)
    random_hits = np.asarray([a == b for a, b in zip(rand.model.batch(val), truth)], dtype=np.float64)
    active_only = int(np.sum((active_hits == 1.0) & (random_hits == 0.0)))
    random_only = int(np.sum((random_hits == 1.0) & (active_hits == 0.0)))
    p_exact = exact_paired_pvalue(active_only, random_only)
    print(f"   active labeling : {active_hits.mean():.3f} held-out teacher agreement ({active.labels_used} labels)")
    print(f"   random labeling : {random_hits.mean():.3f} held-out teacher agreement ({rand.labels_used} labels)")
    print(
        f"   discordant pairs: {active_only} active-only vs {random_only} random-only; "
        f"exact paired two-sided p = {p_exact:.3f}"
    )
    if p_exact < 0.05:
        print("   -> active labeling beats random AT THE SAME TRAINING BUDGET on this population")
        print("      (exact paired evidence at the 5% level)")
    else:
        print("   -> the exact paired evidence is INCONCLUSIVE at the 5% level on this run: too few")
        print("      disagreements to distinguish the policies; no superiority claim is made")

    print("\nwrap the active student in a calibrated cascade and serve")
    cal = pool(seed=2)
    cal_labels = teacher(cal)  # calibration labels are PAID teacher calls too, and counted below
    model = CalibratedTaskModel(active.model, alpha=0.1).calibrate(cal, cal_labels)
    total_pre_serving = 2 * 60 + 300 + len(cal)
    print(f"   calibration bought {len(cal)} more teacher labels")
    print(
        f"   ALL-IN teacher calls before serving: {total_pre_serving} "
        f"(120 training + 300 evaluation + {len(cal)} calibration)"
    )
    casc = Cascade(model, teacher, cost=CostModel(c_frontier=0.01, c_local=0.00001))
    casc.serve(pool(seed=901))
    rep = casc.report()
    setup_cost = total_pre_serving * 0.01
    print(f"   served {rep['n_requests']} requests, escalated {rep['realized_escalation_rate']:.1%} to the LLM")
    print(
        f"   serving saved ${rep['savings_vs_frontier']:.2f} vs paying the LLM per request -- against "
        f"${setup_cost:.2f} of setup labels ({total_pre_serving} calls); setup amortizes only past that volume"
    )


if __name__ == "__main__":
    main()
