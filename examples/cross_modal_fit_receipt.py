"""Flagship cross-modal fit receipt (C4): one typed graph over text + image + signal + a target.

Each record mixes modalities: a categorical plan, an IMAGE-encoder latent, a SIGNAL-encoder latent, and
a continuous price driven by both latents. mixle fits it into ONE heterogeneous Bayesian network where
the image and signal latents are vector nodes (C1) and the cross-modal edges (price <- image, price <-
signal) are recovered from data -- with an estimation certificate saying HOW every block was solved (all
closed-form / convex here, no gradient descent). Then it predicts held-out price and reports the fit.

Everything measured in-process; a few seconds, no GPU, no network.
"""

from __future__ import annotations

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
    print("CROSS-MODAL FIT: one graph over categorical + image + signal + price")
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
    print(f"\nheld-out price prediction: corr={r:.3f}, rmse={rmse:.2f}")
    print("the cross-modal edges were recovered from data; the readout is the closed-form CLG mean.")


def _clg_mean(factor, record: tuple) -> float:
    """The conditional-linear-Gaussian mean the price factor learned, evaluated at a record's parents.

    Uses the factor's OWN design-row builder (which lays out vector parents per its vec_dims), so the
    readout is exactly the closed-form linear map the fit produced -- not a reconstruction."""
    from mixle.inference.bayesian_network import _design_row

    row = _design_row(factor.parents, [record[p] for p in factor.parents], factor.discrete, factor.vec_dims)
    return float(np.dot(row, factor.coef))


if __name__ == "__main__":
    main()
