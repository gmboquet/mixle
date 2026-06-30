"""A hybrid model: a Transformer language model and a classical Gamma, composed and fit *together*.

Each event pairs text with a number -- here a short token context plus the next token (the "message"),
and a positive latency (how long the event took). One mixle model captures the whole event: a causal
Transformer scores the text, a Gamma models the latency, and a mixture puts a few latent event *types*
over the pair. EM trains every Transformer and every Gamma jointly -- the language model and the classical
density are the same kind of object (a distribution), so they fit through the same call, and the joint
log-density of an event is just transformer(text) + gamma(latency).

Self-contained (no downloads): it builds two synthetic event regimes, fits the hybrid mixture, and reports
the joint log-density it assigns to events. Swap GammaEstimator for any of ~90 families to change the
numeric side; swap the leaf for any other model to change the text side.
"""
from __future__ import annotations

import numpy as np

from mixle.inference import optimize
from mixle.models import LM, StreamingTransformerLeaf
from mixle.stats import CompositeEstimator, GammaEstimator, MixtureEstimator

VOCAB, BLOCK = 64, 16


def expert() -> CompositeEstimator:
    """One event 'type': a small causal Transformer over the text, x a Gamma over the latency."""
    lm = LM(vocab=VOCAB, d_model=64, n_layer=2, n_head=4, block=BLOCK)
    text = StreamingTransformerLeaf(lm.module).estimator()  # the LLM as a generative leaf
    return CompositeEstimator((text, GammaEstimator()))     # text x latency, one mixture component


def synth_events(parity: int, latency_scale: float, n: int, rng: np.random.RandomState) -> list:
    """Events of one regime: a token window of a given parity, plus a Gamma(2, scale) latency."""
    out = []
    for _ in range(n):
        ctx = ((rng.randint(0, VOCAB // 2, BLOCK) * 2 + parity) % VOCAB).astype(float)
        nxt = int((rng.randint(0, VOCAB // 2) * 2 + parity) % VOCAB)
        out.append(((ctx, nxt), float(rng.gamma(2.0, latency_scale))))
    return out


def main() -> None:
    rng = np.random.RandomState(0)
    events = synth_events(0, 0.5, 120, rng) + synth_events(1, 4.0, 120, rng)

    # One fit trains the Transformers and the Gammas together (responsibility-weighted EM).
    model = optimize(events, MixtureEstimator([expert(), expert()]), max_its=6)

    # The fitted hybrid scores each event jointly: transformer(text) + gamma(latency).
    ll = np.asarray(model.seq_log_density(model.dist_to_encoder().seq_encode(events)))
    print("hybrid (Transformer LM x Gamma) mixture fit by EM")
    print("  events: %d   mean joint log-density: %.2f   (all finite: %s)" % (len(ll), ll.mean(), np.isfinite(ll).all()))
    print("  mixing weights:", [round(float(w), 3) for w in model.w])


if __name__ == "__main__":
    main()
