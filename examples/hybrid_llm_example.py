"""A hybrid model with a real job: anomaly detection on an event stream.

Each event in a log / activity stream has a TYPE (what happened) and a TIME (seconds since the previous
event). mixle models both at once as a *neural marked point process*: a causal Transformer predicts the
next event type from recent history, and a Gamma models the wait time. The two compose into a single
distribution and fit in one ``optimize`` call (the Gamma in closed form, the Transformer by gradient
descent) -- the language model and the classical timing law are the same kind of object -- so an event's
anomaly score is a single joint log-density that drops when an event is unusual
in *what* happened, in *when* it happened, or both. A sequence model alone would miss the timing; a timing
model alone would miss the content.

Self-contained (no downloads): a synthetic stream whose event types cycle and whose wait times scale with
the type. After fitting, a normal event scores high; corrupting either the type or the timing drops it.
"""
from __future__ import annotations

import numpy as np

from mixle.inference import optimize
from mixle.models import LM, StreamingTransformerLeaf
from mixle.stats import CompositeEstimator, GammaEstimator

K, B = 16, 16  # number of event types, history-window length


def synth_stream(n: int, rng: np.random.RandomState) -> list:
    """Events ((history window, next type), seconds since last): the type cycles, the wait scales with it."""
    out, t, hist = [], 0, [0.0] * B
    for _ in range(n):
        nxt = (t + 1) % K if rng.rand() < 0.97 else rng.randint(0, K)  # a near-deterministic cycle
        wait = float(rng.gamma(2.0, 0.3 + 0.25 * t))                   # timing depends on the event type
        out.append(((np.array(hist[-B:], dtype=float), nxt), wait))
        hist.append(float(nxt))
        t = nxt
    return out


def main() -> None:
    rng = np.random.RandomState(0)
    data = synth_stream(800, rng)

    # One optimize() call fits both: the Gamma in closed form, the Transformer by gradient descent.
    model = optimize(
        data,
        CompositeEstimator((
            StreamingTransformerLeaf(LM(vocab=K, d_model=96, n_layer=3, n_head=4, block=B).module).estimator(),
            GammaEstimator(),
        )),
        max_its=20,
    )

    # Joint anomaly score = transformer(type | history) + gamma(wait). Corrupt either channel, it drops.
    (hist, typ), wait = data[200]
    print("neural marked point process (Transformer 'what' x Gamma 'when'), one fit")
    for label, event in [
        ("normal event", ((hist, typ), wait)),
        ("anomalous timing (40x wait)", ((hist, typ), wait * 40.0)),
        ("anomalous next event", ((hist, (typ + 7) % K), wait)),
    ]:
        print(f"  {label:30s} joint log-density: {model.log_density(event):8.2f}")


if __name__ == "__main__":
    main()
