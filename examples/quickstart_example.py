"""The two front-door entry points: ``mixle.describe()`` and ``mixle.propose()``.

The package docstring's first line of guidance is "start with ``mixle.describe(x)`` to see what any
object can do"; the README's "Package highlights" pairs ``optimize(data)`` with ``mixle.propose(data)``
as the two ways to just hand mixle data and get a model back. This script is the runnable version of
both: the exact mixed-type records from the README Quickstart, described and fit; then a larger sample
from a known generative process (so the held-out comparison and dependency notes below have something
real to find) handed to ``propose()``, whose frontier search fits each candidate on a train split and
scores it on held-out data -- the ranking it prints is measured, not asserted.

Run: ``python examples/quickstart_example.py``
"""

from __future__ import annotations

import numpy as np

import mixle
from mixle.inference import optimize


def demo_describe_and_optimize():
    records = [  # your rows: a number, a category, a flag -- mixed, some missing
        (1.9, "paid", True),
        (0.4, "free", False),
        (2.1, "paid", True),
        (0.7, "free", False),
        (1.6, "paid", True),
        (0.3, "free", None),
    ]
    model = optimize(records, out=None)  # mixle works out the model and fits it (out=None: quiet)

    print("mixle.describe(fitted model):")
    print(" ", mixle.describe(model).replace("\n", "\n  "))
    print("\nscore an observation:", round(float(model.log_density(records[0])), 3))
    print("draw new ones:", model.sampler().sample(3))


def demo_propose():
    # A larger sample from a KNOWN generative process, so propose()'s dependency detection and
    # held-out ranking have real signal to report: `category` drives both `amount`'s mean and
    # `flag`'s rate, exactly the kind of cross-field dependency the frontier search should notice.
    rng = np.random.RandomState(0)
    n = 240
    category = rng.choice(["paid", "free"], size=n, p=[0.4, 0.6])
    amount = np.where(category == "paid", rng.normal(2.0, 0.5, n), rng.normal(0.5, 0.3, n))
    flag = np.where(category == "paid", rng.random(n) < 0.9, rng.random(n) < 0.05)
    records = list(zip((float(x) for x in amount), category.tolist(), flag.tolist()))

    result = mixle.propose(records, fit=True, seed=0)
    print("\nmixle.propose(records).explain():")
    print(" ", result.explain().replace("\n", "\n  "))
    print("\nheld-out-verified frontier (best first):")
    for c in result.frontier:
        if "heldout_mean_log_density" in c:
            print(f"  {c['name']:>12}: held-out mean log-density {c['heldout_mean_log_density']:.3f}")
        else:
            print(f"  {c['name']:>12}: {c.get('error') or c.get('skipped')}")


def main():
    print("# mixle quickstart: describe() and propose()\n")
    demo_describe_and_optimize()
    demo_propose()


if __name__ == "__main__":
    main()
