"""JIT-compile a model's whole-tree log-density to a single XLA program (A2).

mixle's per-distribution ``backend_seq_log_density`` is engine-neutral: a Composite/Mixture/... combines
its children's scores through ``ComputeEngine`` ops, so the *entire* model tree is one pure array
computation. Run it on the JAX engine under ``jax.jit`` and the whole tree lowers to a single compiled
XLA program -- the "compile the model once" trick that makes JAX-backed PPLs fast.

    score = jit_seq_log_density(model)     # compiles lazily, per data shape
    ll = score(data)                       # bit-identical to model.seq_log_density(encode(data)), faster

This is ideal for *repeated scoring of a fixed model* -- prediction, held-out / cross-validation
log-likelihoods, importance weights, bootstrap, large-batch scoring -- where the same compiled program
is reused across calls. The result is bit-identical to ``model.seq_log_density`` (verified) and, on a
large composite tree, several-fold faster than vectorized NumPy.

Scope note: this compiles with the model's parameters baked into the program (the structure lowers to
one XLA function). Reusing a *single* compiled program ACROSS EM iterations as the parameters update
needs the parameters threaded as traced inputs (a tree-level ``backend_log_density_from_params``); that
is the next increment and is not done here.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class JittedScorer:
    """A ``jax.jit``-compiled whole-tree log-density scorer for a fixed model.

    Calling the scorer encodes ``data``, runs the engine-neutral ``backend_seq_log_density`` over the
    whole model tree on the JAX engine under ``jax.jit``, and returns host log-densities. The compiled
    program is cached and reused across calls with the same data shape.
    """

    def __init__(self, model: Any, engine: Any = None) -> None:
        from mixle.engines.jax_engine import JaxEngine

        self.model = model
        self.engine = engine if engine is not None else JaxEngine()
        self._encoder = model.dist_to_encoder()
        self._fn = None  # the jitted callable (built lazily on first call)

    def _compile(self):
        import jax

        from mixle.stats.compute.backend import backend_seq_log_density

        model, engine = self.model, self.engine
        return jax.jit(lambda payload: backend_seq_log_density(model, payload, engine))

    def __call__(self, data: Any) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        enc = self._encoder.seq_encode(data)
        payload = getattr(enc, "engine_payload", enc)
        jpayload = jax.tree_util.tree_map(lambda a: jnp.asarray(a), payload)
        if self._fn is None:
            self._fn = self._compile()
        return np.asarray(self._fn(jpayload))


def jit_seq_log_density(model: Any, engine: Any = None) -> JittedScorer:
    """Return a :class:`JittedScorer`: the whole-tree log-density of ``model`` compiled to one XLA program.

    ``jit_seq_log_density(model)(data)`` is bit-identical to ``model.seq_log_density(encode(data))`` but
    runs as a single ``jax.jit`` XLA program over the entire composite tree -- fast for repeated scoring
    of a fixed model. Requires the JAX optional extra and that every leaf in ``model`` declares JAX
    support (raises ``EngineNotSupportedError``-style errors from the engine layer otherwise).
    """
    return JittedScorer(model, engine=engine)
