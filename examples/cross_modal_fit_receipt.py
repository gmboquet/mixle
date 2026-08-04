"""Vector-valued fields as ordinary Bayesian-network parents, with a machine-checkable acceptance record.

What it demonstrates: :func:`mixle.inference.learn_bayesian_network` over a record whose fields are a
mix of a category, TWO fixed-width vectors, and a scalar target. The structure search treats a vector
field as one parent (not as k separate columns), so the price factor it discovers is a
conditional-linear-Gaussian with vector coefficients. :func:`mixle.inference.certify` then reports how
the fit was solved, and the script ends by printing an ``ACCEPTANCE`` JSON line -- the recovered edge
set plus held-out correlation and RMSE -- and raising if the thresholds are not met, so the example
fails loudly rather than printing a bad number.

Takeaway: you do not flatten or one-hot a vector-valued feature to put it in a graphical model, and
"did the fit work?" can be a machine-readable assertion in the script itself rather than a human
eyeballing stdout. The readout at the bottom evaluates the factor's own closed-form linear map, so the
predicted price is the model's, not a reimplementation of it.

Scope, honestly: this does NOT claim cross-modal reasoning. The two vector fields are synthetic
Gaussian features, not encoded image/signal observations, and there is no alignment objective. Real
modality evidence requires typed raw inputs, pinned encoders, and ablations -- see
``heterogeneous_representation_example.py`` for the real encoder path.

Run: ``python examples/cross_modal_fit_receipt.py``
"""

from __future__ import annotations

import json

import numpy as np

from mixle.inference import certify, learn_bayesian_network


def make_records(n: int, seed: int) -> list[tuple]:
    """(plan, image_latent[3], signal_latent[3], price): price = 30*img[0] + 20*sig[0] + noise."""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        img = rng.normal(0.0, 1.0, 3)  # an image encoder's output (C2 modality leaf)
        sig = rng.normal(0.0, 1.0, 3)  # a signal encoder's output
        price = 30.0 * img[0] + 20.0 * sig[0] + 3.0 * rng.randn()
        plan = "pro" if img[0] + 0.5 * rng.randn() > 0 else "free"
        rows.append((plan, tuple(img), tuple(sig), float(price)))
    return rows


def main() -> None:
    train = make_records(300, 0)
    net = learn_bayesian_network(train, max_parents=2)

    print("=" * 70)
    print("SYNTHETIC MULTI-VECTOR FIT: categorical + vector A + vector B + price")
    print("=" * 70)
    print(f"model: {type(net).__name__}")
    for f in net.factors:
        parents = getattr(f, "parents", None)
        print(f"  field[{f.child}] <- {parents}   ({type(f).__name__})")

    cert = certify(net)
    print(f"\ncertificate: {cert.guarantee.name}")
    print(f"  {cert.why_not_adam().splitlines()[0]}")

    # predictive check: does the fitted graph predict held-out price from the two modality latents?
    test = make_records(300, 1)

    truth, pred = [], []
    pf = next(f for f in net.factors if f.child == 3)
    for record in test[:120]:
        truth.append(record[3])
        pred.append(_clg_mean(pf, record))
    r = float(np.corrcoef(truth, pred)[0, 1])
    rmse = float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(pred)) ** 2)))
    recovered_edges = set(pf.parents) == {1, 2}
    accepted = recovered_edges and r >= 0.95 and rmse <= 6.0
    print(f"\nheld-out price prediction: corr={r:.3f}, rmse={rmse:.2f}")
    print(
        "ACCEPTANCE "
        + json.dumps(
            {
                "artifact": "mixle.multi_vector_fit_acceptance/v1",
                "accepted": accepted,
                "held_out_correlation": r,
                "held_out_rmse": rmse,
                "price_parents": sorted(pf.parents),
                "required_price_parents": [1, 2],
            },
            sort_keys=True,
        )
    )
    if not accepted:
        raise RuntimeError("multi-vector acceptance failed: required edges or held-out thresholds were not met")
    print("the required vector-feature edges were recovered; the readout is the closed-form CLG mean.")


def _clg_mean(factor, record: tuple) -> float:
    """The conditional-linear-Gaussian mean the price factor learned, evaluated at a record's parents.

    Uses the factor's OWN design-row builder (which lays out vector parents per its vec_dims), so the
    readout is exactly the closed-form linear map the fit produced -- not a reconstruction."""
    from mixle.inference.bayesian_network import _design_row

    row = _design_row(factor.parents, [record[p] for p in factor.parents], factor.discrete, factor.vec_dims)
    return float(np.dot(row, factor.coef))


if __name__ == "__main__":
    main()
