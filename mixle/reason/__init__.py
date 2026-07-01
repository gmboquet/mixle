"""``mixle.reason`` -- the cross-modal scientific-reasoning front door.

A scientific question is a query on a joint posterior over a shared latent that every modality is
evidence about. This package wires that idea into one call:

    answer = reason(prior, [evidence_from_modality_1, evidence_from_modality_2, ...])
    answer.mean, answer.interval(0.9)      # a posterior, with honest error bars
    answer.attribution()                   # which modality sharpened the belief (nats)
    answer.predict(H, R).epistemic         # split a prediction's uncertainty (epi vs aleatoric)

The exact core here is linear-Gaussian: each modality contributes a linear-Gaussian observation
``y = H z + noise(R)`` and the beliefs fuse by exact Kalman assimilation (a product of experts).
Non-linear / learned encoders (Phase 3+, and application-specific forward models in the sibling
``mixle_pde`` package) plug in by *producing* such evidence -- a linearized ``(H, y, R)`` or a
Gaussian expert -- so the front door is stable while the encoders grow underneath it.

Design: notes/mixle-cross-modal-reasoning-design.md. Built on :mod:`mixle.inference.belief`
(the belief state) and :mod:`mixle.inference.uncertainty` (the epistemic/aleatoric split).
"""

from __future__ import annotations

from typing import Any

from mixle.inference.belief import BeliefState, GaussianBelief, as_belief
from mixle.reason.core import Evidence, Latent, LinearGaussianEvidence, ReasonedAnswer, reason

__all__ = [
    "reason",
    "Latent",
    "Evidence",
    "LinearGaussianEvidence",
    "ReasonedAnswer",
    "GaussianBelief",
    "BeliefState",
    "as_belief",
    "AmortizedEncoder",
]


def __getattr__(name: str) -> Any:
    # Lazy: defer importing the encoder module (and building its torch net) until first access, so
    # the exact linear-Gaussian core here does not construct a torch model just to be imported.
    if name == "AmortizedEncoder":
        from mixle.reason.encoder import AmortizedEncoder

        return AmortizedEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
