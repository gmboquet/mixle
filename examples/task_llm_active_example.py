"""LLM teacher + active labeling: pay a frontier model for the fewest labels, then serve locally for ~free.

The two differentiators in one run:

  * the teacher is an **LLM** (here a local ``CallableLLM``; swap in ``OpenAICompatLLM(base_url, model)``
    to use Ollama / vLLM / a hosted endpoint unchanged);
  * **active labeling** (DoE applied to the labeling decision) queries that LLM only for the most informative
    examples, reaching the same student quality as random labeling for far fewer paid calls.

Then the distilled student is wrapped in a calibrated cascade and the realized savings are reported. Run:
``python task_llm_active_example.py``  (needs ``pip install "mixle[torch]"``).
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

    def acc(model):
        pred = model.batch(val)
        return float(np.mean([a == b for a, b in zip(pred, truth)]))

    budget = 60
    print(f"label budget: {budget} LLM calls (out of {len(p)} unlabeled)")
    active = active_distill(teacher, p, budget=budget, seed_size=20, rounds=4, acquisition="margin", recipe=recipe)
    rand = active_distill(teacher, p, budget=budget, seed_size=20, rounds=4, acquisition="random", recipe=recipe)
    print(f"   active labeling : {acc(active.model):.3f} accuracy with {active.labels_used} labels")
    print(f"   random labeling : {acc(rand.model):.3f} accuracy with {rand.labels_used} labels")

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
