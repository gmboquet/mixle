"""LLM teacher + active labeling: pay a frontier model for the fewest labels, then serve locally for ~free.

The two differentiators in one run:

  * the teacher is an **LLM** (here a local ``CallableLLM``; swap in ``OpenAICompatLLM(base_url, model)``
    to use Ollama / vLLM / a hosted endpoint unchanged);
  * **active labeling** (DoE applied to the labeling decision) queries that LLM only for the most
    informative examples. The comparison below is a SAME-BUDGET one -- both policies spend exactly the
    same number of paid calls, and the measured quantity is held-out agreement with this synthetic
    deterministic teacher, compared as a paired difference with its uncertainty. (A labels-to-target
    curve -- how many calls each policy needs to REACH a fixed quality -- is a different experiment;
    the EIG/BALD curve in ``label_economics_demo.py`` is the acquisition-level version of it.)

Then the distilled student is wrapped in a calibrated cascade and the realized savings are reported.

Scope: the LABELING-BUDGET side of the mixle.task loop. ``task_distill_example.py`` is the plain
distill/save/reload entry point and ``task_cascade_economics_example.py`` prices the serving side;
``label_economics_demo.py`` makes the same "active beats random" point through the EIG/BALD
``acquire()`` API instead of a task-level teacher.

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


def local_llm(prompt, system=None):
    """Deterministic local teacher with the same callable shape as an LLM endpoint."""
    text = prompt.split("Text:", 1)[-1].lower()
    return "spam" if any(w in text.split() for w in SPAM) else "ham"


def main() -> None:
    # the teacher is an LLM, constrained to the label set
    teacher = llm_labeler(CallableLLM(local_llm), ["spam", "ham"], instruction="Classify the email as spam or ham.")
    recipe = {"n": 4, "dim": 512, "hidden": [64], "epochs": 200, "lr": 1e-2}

    p, val = pool(1), pool(seed=900)[:300]
    truth = teacher(val)

    budget = 60
    print(f"label budget: {budget} LLM calls per policy (out of {len(p)} unlabeled)")
    active = active_distill(teacher, p, budget=budget, seed_size=20, rounds=4, acquisition="margin", recipe=recipe)
    rand = active_distill(teacher, p, budget=budget, seed_size=20, rounds=4, acquisition="random", recipe=recipe)
    # The ESTIMAND, stated before the numbers: each realized student's agreement with the
    # deterministic teacher over the synthetic pool() population, compared at the SAME paid-call
    # budget. Both students predict the SAME 300 fresh validation rows (drawn independently of
    # both fits), so conditional on the two fits the per-row paired differences are i.i.d. and
    # give a valid 95% interval for the agreement difference. "Agreement", not "accuracy": the
    # teacher is the reference, and the data are synthetic.
    active_hits = np.asarray([a == b for a, b in zip(active.model.batch(val), truth)], dtype=np.float64)
    random_hits = np.asarray([a == b for a, b in zip(rand.model.batch(val), truth)], dtype=np.float64)
    paired = active_hits - random_hits
    difference = float(paired.mean())
    standard_error = float(paired.std(ddof=1) / np.sqrt(len(paired))) if len(paired) > 1 else float("inf")
    low, high = difference - 1.96 * standard_error, difference + 1.96 * standard_error
    print(f"   active labeling : {active_hits.mean():.3f} held-out teacher agreement ({active.labels_used} labels)")
    print(f"   random labeling : {random_hits.mean():.3f} held-out teacher agreement ({rand.labels_used} labels)")
    print(f"   paired difference {difference:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}] at the same budget")
    if low > 0.0:
        print("   -> active labeling measurably beats random AT THE SAME BUDGET on this population")
    else:
        print("   -> no measurable difference at this budget and sample size; active did not pay for")
        print("      itself here, and no fewer-calls claim is made")

    print("\nwrap the active student in a calibrated cascade and serve")
    cal = pool(seed=2)
    model = CalibratedTaskModel(active.model, alpha=0.1).calibrate(cal, teacher(cal))
    casc = Cascade(model, teacher, cost=CostModel(c_frontier=0.01, c_local=0.00001))
    casc.serve(pool(seed=901))
    rep = casc.report()
    print(f"   served {rep['n_requests']} requests, escalated {rep['realized_escalation_rate']:.1%} to the LLM")
    print(f"   saved ${rep['savings_vs_frontier']:.2f} vs paying the LLM for every request")


if __name__ == "__main__":
    main()
