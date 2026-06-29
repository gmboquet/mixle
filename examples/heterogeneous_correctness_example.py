"""D3 — heterogeneous-record correctness: mixle fits every field; the nearest rival can't.

The differentiator is not "we compose distributions" (pomegranate/TFP do too) — it is that mixle's
estimator/encoder trees *mirror* the model tree, so **every field of a heterogeneous record is updated
correctly** during EM. Here each mixture component is a record ``(real ~ Normal, label ~ Categorical)``.
Two well-separated clusters differ in *both* fields; a correct fit must recover the Normal means AND the
per-cluster Categorical probabilities.

- **mixle**: `Mixture(Composite(Gaussian, Categorical))` — fit by EM, recovers both fields in every
  component.
- **pomegranate 1.1.2** (if installed): the documented heterogeneous path,
  `GeneralMixtureModel([IndependentComponents([Normal, Categorical]), ...])`, raises inside its own
  code (TypeError/IndexError) — it cannot fit a Normal-beside-Categorical mixture at all. We report the
  failure honestly rather than asserting a specific numeric misfit.

Run: ``python examples/heterogeneous_correctness_example.py``
"""

from __future__ import annotations

import importlib.util

import numpy as np

from mixle.inference import optimize
from mixle.stats import CategoricalDistribution as Cat
from mixle.stats import CompositeDistribution as Comp
from mixle.stats import GaussianDistribution as G
from mixle.stats import MixtureDistribution as Mix

TRUE = [
    {"mu": -3.0, "probs": [0.80, 0.10, 0.10]},
    {"mu": 3.0, "probs": [0.10, 0.10, 0.80]},
]


def make_data(n_per=600, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for cl in TRUE:
        for _ in range(n_per):
            rows.append((float(rng.normal(cl["mu"], 1.0)), int(rng.choice(3, p=cl["probs"]))))
    return rows


def fit_mixle(data):
    proto = Mix(
        [
            Comp((G(-2, 1), Cat({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}))),
            Comp((G(2, 1), Cat({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}))),
        ],
        [0.5, 0.5],
    )
    m = optimize(data, proto.estimator(), prev_estimate=proto, max_its=100, out=None)
    out = []
    for c in sorted(m.components, key=lambda c: c.dists[0].mu):
        out.append({"mu": float(c.dists[0].mu), "probs": [float(c.dists[1].pmap[k]) for k in (0, 1, 2)]})
    return out


def try_pomegranate(data):
    """Return (status, detail). status in {'ok','failed','absent'}."""
    if importlib.util.find_spec("pomegranate") is None:
        return "absent", "pomegranate not installed"
    try:
        import torch
        from pomegranate.distributions import Categorical as PC
        from pomegranate.distributions import IndependentComponents
        from pomegranate.distributions import Normal as PN
        from pomegranate.gmm import GeneralMixtureModel

        X = torch.tensor(np.asarray(data, dtype=float), dtype=torch.float64)
        comps = [IndependentComponents([PN(), PC(n_categories=[3])]) for _ in range(2)]
        GeneralMixtureModel(comps, max_iter=50).fit(X)
        return "ok", "fit succeeded"
    except Exception as e:  # noqa: BLE001  (the point is to report whatever it raises)
        return "failed", f"{type(e).__name__}: {e}"


def main():
    data = make_data()
    print("# Heterogeneous-record correctness: (real ~ Normal, label ~ Categorical), 2 clusters\n")
    print("true clusters:")
    for cl in TRUE:
        print(f"  mu={cl['mu']:+.1f}  cat.probs={cl['probs']}")

    print("\n## mixle — Mixture(Composite(Gaussian, Categorical)), EM")
    fit = fit_mixle(data)
    ok = True
    for i, c in enumerate(fit):
        print(f"  comp{i}: normal.mean={c['mu']:+.2f}  cat.probs={[round(p, 2) for p in c['probs']]}")
        ok = ok and abs(abs(c["mu"]) - 3.0) < 0.3 and max(c["probs"]) > 0.6
    print(f"  -> both fields recovered in every component: {ok}")

    print("\n## pomegranate 1.1.2 — GeneralMixtureModel([IndependentComponents([Normal, Categorical]), ...])")
    status, detail = try_pomegranate(data)
    if status == "absent":
        print("  (skipped — pomegranate not installed)")
    elif status == "ok":
        print(f"  fit succeeded: {detail}  (rerun mixle's recovery check against it)")
    else:
        print(f"  could NOT fit the heterogeneous mixture: {detail}")
        print("  -> the Normal-beside-Categorical path raises inside pomegranate; mixle fits it correctly.")

    print(
        "\nTakeaway: composing mixed-family distributions is not the differentiator — *correctly fitting "
        "every field of the composed record* is. mixle's estimator/encoder trees mirror the model, so the "
        "Categorical updates alongside the Normal; the nearest rival's heterogeneous path does not."
    )


if __name__ == "__main__":
    main()
