"""Distill a slow teacher into a tiny local model, tune it with DoE, save it, and call it from a plain program.

The shape of the thing: you have a slow/expensive way to label text (here, a stand-in "teacher"); you want a
fast local model that does the same job. ``mixle.task`` distills the teacher into a small classifier, lets
``mixle.doe`` search for the cheapest recipe that still matches, saves a durable artifact, and hands you a
callable you load in any process and just call --- ``task(text) -> label`` --- no server, no GPU, no teacher.

Run: ``python task_distill_example.py``  (needs the torch extra: ``pip install "mixle[torch]"``).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from mixle.task import TaskModel, agreement, distill, tune_recipe


def make_corpus(n_per_class: int, seed: int) -> list[str]:
    rng = np.random.RandomState(seed)
    spam = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
    ham = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
    filler = ["the", "a", "today", "tomorrow", "please", "thanks", "we", "you"]
    texts: list[str] = []
    for words in (spam, ham):
        for _ in range(n_per_class):
            toks = list(rng.choice(words, size=2)) + list(rng.choice(filler, size=rng.randint(3, 7)))
            rng.shuffle(toks)
            texts.append(" ".join(toks))
    rng.shuffle(texts)
    return texts


def slow_teacher(texts: list[str]) -> list[str]:
    """Stand-in for an expensive labeler (a frontier LM, a human, a slow rule). The student never sees it again."""
    spam = {"free", "winner", "prize", "buy", "cheap", "offer", "click"}
    return ["spam" if any(w in t.split() for w in spam) else "ham" for t in texts]


def main() -> None:
    train, val, test = make_corpus(200, 1), make_corpus(80, 2), make_corpus(80, 3)

    print("1) distill the teacher into a tiny student")
    student = distill(slow_teacher, train, n=4, dim=512, hidden=[64], epochs=300, seed=0, task="spam vs ham")
    print(f"   train agreement with teacher: {student.meta['train_agreement']:.3f}")
    print(f"   held-out agreement:           {agreement(student, slow_teacher(test), test):.3f}")

    print("\n2) let mixle.doe search for a cheaper recipe that still matches")
    tuned = tune_recipe(slow_teacher, train, val, n_init=4, n_iter=6, cost_weight=0.5, seed=0)
    print(f"   best recipe: {tuned.recipe}")
    print(f"   agreement {tuned.agreement:.3f}   relative train cost {tuned.cost:.3f}")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spam_classifier")
        tuned.model.save(path)
        print(f"\n3) saved a durable artifact to {os.path.basename(path)}/  -- load it anywhere, no teacher needed")

        local = TaskModel.load(path)  # a plain program would do exactly this
        for text in ["free prize click now", "team meeting report tomorrow", "cheap offer today"]:
            print(f"   local_model({text!r}) -> {local(text)!r}")


if __name__ == "__main__":
    main()
