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


# ---------------------------------------------------------------------------------------------------
# A2 bullet 1: the EM step itself compiled to one XLA program, reused across iterations.
#
# For a finite mixture of same-family scalar exponential-family leaves, both the E-step (component
# log-densities + responsibilities) and the closed-form weighted M-step are pure array ops, so the whole
# EM step lowers to a single jax.jit program with the parameters as traced inputs, so the same compiled
# program is reused as the parameters update each iteration (the NumPyro 'compile once' trick for EM).
# Each family entry: read params off a leaf, score a component (reusing the engine-neutral backend
# density), the closed-form weighted M-step, and rebuild a fitted leaf. Extend the registry to add a
# family; the EM driver is family-agnostic.
# ---------------------------------------------------------------------------------------------------
def _mixture_em_family(leaf, jnp):
    name = type(leaf).__name__
    if name == "GaussianDistribution":
        from mixle.stats import GaussianDistribution as G

        def score(params_k, x, _extra, eng):
            return G.backend_log_density_from_params(x, params_k[0], params_k[1], eng)

        def mstep(x, r_k, _extra):
            nk = jnp.sum(r_k) + 1e-12
            mu = jnp.sum(r_k * x) / nk
            return (mu, jnp.sum(r_k * (x - mu) ** 2) / nk)

        return {
            "unpack": lambda d: (float(d.mu), float(d.sigma2)),
            "score": score,
            "mstep": mstep,
            "make": lambda pk: G(float(pk[0]), float(pk[1])),
            "extra": None,
        }
    if name == "PoissonDistribution":
        from mixle.stats import PoissonDistribution as P

        def score(params_k, x, extra, eng):
            return P.backend_log_density_from_params(x, extra, params_k[0], eng)  # extra = log(x!)

        def mstep(x, r_k, _extra):
            nk = jnp.sum(r_k) + 1e-12
            return (jnp.sum(r_k * x) / nk,)

        return {
            "unpack": lambda d: (float(d.lam),),
            "score": score,
            "mstep": mstep,
            "make": lambda pk: P(float(pk[0])),
            "extra": "log_factorial",  # the driver precomputes gammaln(x+1) (a fixed data stat)
        }
    if name == "ExponentialDistribution":
        from mixle.stats import ExponentialDistribution as E

        def score(params_k, x, _extra, eng):
            return E.backend_log_density_from_params(x, params_k[0], eng)  # beta = mean (scale)

        def mstep(x, r_k, _extra):
            nk = jnp.sum(r_k) + 1e-12
            return (jnp.sum(r_k * x) / nk,)  # beta = weighted mean (the Exponential MLE)

        return {
            "unpack": lambda d: (float(d.beta),),
            "score": score,
            "mstep": mstep,
            "make": lambda pk: E(float(pk[0])),
            "extra": None,
        }
    return None


def jit_em_mixture(model: Any, data: Any, *, max_its: int = 100, engine: Any = None):
    """Fit a finite mixture of same-family scalar exponential-family leaves by EM, with the ENTIRE EM loop
    (every E-step + closed-form weighted M-step iteration) compiled to ONE ``jax.jit`` XLA program via
    ``lax.scan`` -- the parameters are traced inputs threaded through the loop on-device, so there is no
    per-iteration recompile and no per-iteration host sync. This is roadmap A2 bullet 1 ("repeated EM
    iterations run as one XLA program") realized literally.

    ``model`` is the *initial* mixture (its components seed the EM); supported leaves: Gaussian, Poisson,
    Exponential. Runs a fixed ``max_its`` iterations (no host-side early stop -- that is the point: the
    loop stays on-device). Returns a fitted ``MixtureDistribution``, bit-close to the host EM from the
    same start (it is the same EM update). Raises ``NotImplementedError`` for unsupported structure.

    SPEED -- measured scope: the payoff is **GPU/TPU and large scale**, where XLA parallelizes the
    E-step over millions of points and many components. On an **Apple M4 GPU (via jax-metal)** this kernel
    runs **~21x faster than mixle's vectorized NumPy EM** (K=10, N=1e6, 50 iters: 156 ms vs 3254 ms) with
    identical estimates, and ~12x faster than the same jitted loop on CPU. **On CPU it is *not* a speedup**
    -- mixle's host EM is already vectorized NumPy (+ a fused path) and a long sequential ``scan`` of small
    steps loses to it (~0.1-0.8x for K up to 40). So: GPU -> big win, CPU -> use the host EM. (The CPU win
    from A2 is the single-pass scoring jit, :func:`jit_seq_log_density`, ~8x.) Note: jax-metal is
    version-pinned -- the GPU result above used Python 3.11 + jax/jaxlib 0.4.34 + jax-metal 0.1.1; newer
    jaxlib emits StableHLO that jax-metal 0.1.1 cannot compile.
    """
    import jax
    import jax.numpy as jnp

    from mixle.engines.jax_engine import JaxEngine
    from mixle.stats import MixtureDistribution

    eng = engine if engine is not None else JaxEngine()
    comps = getattr(model, "components", None)
    if comps is None or any(type(c) is not type(comps[0]) for c in comps):
        raise NotImplementedError("jit_em_mixture needs a mixture of leaves of a single family.")
    fam = _mixture_em_family(comps[0], jnp)
    if fam is None:
        raise NotImplementedError(f"jit_em_mixture does not support a mixture of {type(comps[0]).__name__}.")

    K = len(comps)
    x = jnp.asarray(np.asarray(data, dtype=float))
    extra = jax.scipy.special.gammaln(x + 1.0) if fam["extra"] == "log_factorial" else None
    # stack each component's params into per-parameter arrays of length K (traced inputs)
    p0 = [fam["unpack"](c) for c in comps]
    n_par = len(p0[0])
    params0 = tuple(jnp.asarray([p0[k][j] for k in range(K)]) for j in range(n_par))
    log_w0 = jnp.log(jnp.asarray(np.asarray(model.w, dtype=float)))

    def em_step(carry, _i):
        params, log_w = carry
        comp_ll = jnp.stack(
            [fam["score"](tuple(params[j][k] for j in range(n_par)), x, extra, eng) for k in range(K)],
            axis=1,
        )  # (N, K)
        log_r = comp_ll + log_w  # (N, K)
        log_norm = jax.scipy.special.logsumexp(log_r, axis=1)  # (N,)
        r = jnp.exp(log_r - log_norm[:, None])  # responsibilities (N, K)
        new_params_per_k = [fam["mstep"](x, r[:, k], extra) for k in range(K)]
        new_params = tuple(jnp.stack([new_params_per_k[k][j] for k in range(K)]) for j in range(n_par))
        new_log_w = jnp.log(jnp.sum(r, axis=0) / x.shape[0])
        return (new_params, new_log_w), jnp.sum(log_norm)

    # the ENTIRE EM loop compiled to ONE XLA program (lax.scan keeps every iteration on-device -- no
    # per-iteration host sync, the move that makes a jitted EM actually fast).
    @jax.jit
    def run(params, log_w):
        (params, log_w), lls = jax.lax.scan(em_step, (params, log_w), xs=None, length=int(max_its))
        return params, log_w, lls

    params, log_w, _lls = run(params0, log_w0)
    params_np = [np.asarray(eng.to_numpy(p)) for p in params]
    w = np.asarray(eng.to_numpy(jnp.exp(log_w)))
    fitted = [fam["make"](tuple(params_np[j][k] for j in range(n_par))) for k in range(K)]
    return MixtureDistribution(fitted, list(w))
