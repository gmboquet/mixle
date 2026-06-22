"""A single ``sample()`` entry point that draws from any samplable pysp object.

``sample(model, size)`` dispatches on the kind of ``model`` so distributions, conjugate posteriors,
relations, field posteriors and latent posteriors all read the same way::

    pysp.stats.sample(gauss, 100)                 # 100 iid observations
    pysp.stats.sample(posterior, 50)              # 50 parameter draws from a conjugate posterior
    pysp.stats.sample(assignment, 10, temperature=2.0)   # 10 Gibbs-weighted relation members
    pysp.stats.sample(field_post, 100)            # 100 joint field draws (dict per node)
    pysp.stats.sample(q_latent, 5)                # 5 latent-variable draws

A shared ``rng`` (a ``numpy.random.RandomState``) makes a whole pipeline reproducible: it is threaded
into the relation / field / latent draws directly, and for a distribution / conjugate posterior the
per-call stream is seeded from it, so one ``rng`` drives independent reproducible streams across many
``sample()`` calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pysp.arithmetic import maxrandint

__all__ = ["sample"]


def _resolve_rng(seed: int | None, rng: np.random.RandomState | None) -> np.random.RandomState:
    if rng is not None:
        return rng
    return np.random.RandomState(seed)


def sample(
    model: Any,
    size: int | None = None,
    *,
    seed: int | None = None,
    rng: np.random.RandomState | None = None,
    **kwargs: Any,
) -> Any:
    """Draw sample(s) from any samplable pysp object.

    Args:
        model: a distribution, conjugate posterior, :class:`~pysp.relations.Relation`,
            ``FieldPosterior`` or ``LatentPosterior``.
        size: ``None`` returns a single draw in the object's natural type; an int returns a collection
            (an array for homogeneous leaves, a list / dict-of-arrays for structured draws).
        seed: scalar seed for the draw (ignored if ``rng`` is given).
        rng: a shared ``RandomState`` for reproducible, composable streams; takes precedence over ``seed``.
        **kwargs: forwarded to the underlying sampler -- e.g. ``temperature`` / ``k`` / ``uniform`` for a
            relation, ``nodes`` for a field posterior, ``batched`` for a distribution.

    Returns:
        A single draw (``size=None``) or a collection of ``size`` draws.

    Raises:
        TypeError: if ``model`` is not a recognized samplable object.
    """
    # Relation -- a sampler under a Gibbs measure over its members (temperature/k/uniform are sampler args).
    from pysp.relations import Relation

    if isinstance(model, Relation):
        return model.sampler(seed=seed, rng=rng, **kwargs).sample(size)

    # FieldPosterior -- joint field/parameter draws (importing lazily; field.py needs torch).
    try:
        from pysp.ppl.field import FieldPosterior
    except Exception:  # pragma: no cover - torch optional
        FieldPosterior = ()  # type: ignore[assignment]
    if FieldPosterior and isinstance(model, FieldPosterior):
        return model.sample(1 if size is None else size, rng=_resolve_rng(seed, rng), **kwargs)

    # LatentPosterior -- latent-variable draws (one per call; loop for a collection).
    from pysp.sampling.latent_posterior import LatentPosterior

    if isinstance(model, LatentPosterior):
        r = _resolve_rng(seed, rng)
        return model.sample(r) if size is None else [model.sample(r) for _ in range(size)]

    # Distribution or ConjugatePosterior -- both expose .sampler(seed).sample(size).
    if hasattr(model, "sampler"):
        draw_seed = int(rng.randint(0, maxrandint)) if rng is not None else seed
        return model.sampler(seed=draw_seed).sample(size, **kwargs)

    raise TypeError(
        f"don't know how to sample from a {type(model).__name__}; expected a distribution, "
        "conjugate posterior, Relation, FieldPosterior, or LatentPosterior."
    )
