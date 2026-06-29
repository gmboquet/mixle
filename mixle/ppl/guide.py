"""User-declared structured variational inference: a ``Guide`` of per-latent variational factors.

This is the *general* counterpart to the packaged latent models (mixture/HMM/LDA): instead of calling
a built-in model, you (1) write the generative model from PPL primitives, sharing a latent's
``RandomVariable`` handle wherever it appears, (2) declare a :class:`Guide` naming each latent and the
variational factor that approximates it (the q-family), and (3) fit by mean-field VMP / coordinate-ascent
VI. The result is a :class:`StructuredVIPosterior` over the named latents, with a monotone ELBO.

The factorization is mean-field: each declared latent is an *independent* q-factor -- that independence
is the variational projection. The q-families are the conjugate exponential-family factors the VMP
engine carries:

    * ``'gaussian'`` -- a ``Normal`` latent (its mean),
    * ``'gamma'``    -- a precision / positive latent (a ``Gamma`` in a scale slot),
    * ``'dirichlet'``-- a simplex latent (a ``Dirichlet``).

Declare a family to make the projection explicit and have it *checked* against the model's conjugate
structure; omit it to take the conjugate default.

    mu  = Normal(0, 10)            # shared latent handle (its mean ~ Gaussian q)
    tau = Gamma(1, 1)              # precision latent      (~ Gamma q)
    post = structured_vi([(Normal(mu, tau), data)], Guide(mu=mu, tau=tau))
    post.mean("mu"); post.posterior("tau"); post.elbo

Latents shared across several observation factors combine their evidence (pass several
``(model, data)`` pairs). Coverage today is conjugate-exponential structured VI over
Gaussian/Gamma/Dirichlet factors, including hierarchies and shared latents. Non-conjugate q-families,
and the categorical-emission / admixture factors an LDA needs, are extensions to the underlying VMP
:class:`~mixle.ppl.vmp.Graph`, not to this surface -- a clear error is raised rather than a wrong answer.
"""

from __future__ import annotations

from typing import Any

from mixle.ppl.core import RandomVariable
from mixle.ppl.vmp import DirichletVNode, GammaVNode, GaussianVNode, Graph

# declared q-family name -> the VMP variational node type it must resolve to
_NODE_OF_FAMILY = {
    "gaussian": GaussianVNode,
    "normal": GaussianVNode,
    "gamma": GammaVNode,
    "precision": GammaVNode,
    "dirichlet": DirichletVNode,
    "simplex": DirichletVNode,
}
_FAMILY_OF_NODE = {GaussianVNode: "gaussian", GammaVNode: "gamma", DirichletVNode: "dirichlet"}


class Guide:
    """A declared mean-field variational approximation: named latents, each an independent q-factor.

    Build it with ``Guide(name=handle, ...)`` where each ``handle`` is the *same* ``RandomVariable``
    object used in the model (latents are matched by object identity). To pin and validate the
    variational family, pass ``name=(handle, 'gaussian'|'gamma'|'dirichlet')``; otherwise the conjugate
    factor implied by the latent's prior is used.
    """

    def __init__(self, **latents: Any) -> None:
        self._latents: dict[str, tuple[RandomVariable, str | None]] = {}
        for name, spec in latents.items():
            if isinstance(spec, tuple):
                handle, family = spec
                family = str(family).lower()
                if family not in _NODE_OF_FAMILY:
                    raise ValueError(
                        f"unknown variational family {family!r} for {name!r}; "
                        f"use one of {sorted(set(_NODE_OF_FAMILY))}."
                    )
            else:
                handle, family = spec, None
            if not isinstance(handle, RandomVariable):
                raise TypeError(f"guide latent {name!r} must be a RandomVariable handle, got {type(handle).__name__}.")
            self._latents[name] = (handle, family)

    def names(self) -> tuple[str, ...]:
        return tuple(self._latents)

    def __repr__(self) -> str:
        parts = [f"{n}={fam or 'auto'}" for n, (_, fam) in self._latents.items()]
        return f"Guide({', '.join(parts)})"


class StructuredVIPosterior:
    """Posterior from :func:`structured_vi`: per-latent variational factors + the ELBO trace.

    ``posterior(name)`` returns the factor's hyperparameters (e.g. ``{'mean','sd'}`` for a Gaussian,
    ``{'alpha','mean'}`` for a Dirichlet, ``{'shape','rate','mean'}`` for a Gamma); ``mean`` /
    ``samples`` give the posterior mean / exact draws of that latent. Names are the guide's keys (the
    latent handle itself is also accepted).
    """

    def __init__(self, gres, guide: Guide) -> None:
        self._g = gres
        self._guide = guide
        self.elbo = gres.elbo
        self.elbo_trace = gres.elbo_trace
        self.acceptance_rate = None
        self.predictive = None

    def _handle(self, name):
        if isinstance(name, str):
            if name not in self._guide._latents:
                raise KeyError(f"no guide latent named {name!r}; have {self._guide.names()}.")
            return self._guide._latents[name][0]
        return name  # a raw handle

    def posterior(self, name) -> dict:
        return self._g.posterior(self._handle(name))

    def mean(self, name):
        return self.posterior(name).get("mean")

    def samples(self, name, n: int = 4000, rng=None):
        return self._g.samples(self._handle(name), n=n, rng=rng)

    def summary(self) -> dict:
        out = {name: self._g.posterior(h) for name, (h, _) in self._guide._latents.items()}
        out["elbo"] = self.elbo
        out["iterations"] = int(self.elbo_trace.size)
        return out

    def __repr__(self) -> str:
        return f"StructuredVIPosterior({self._guide!r}, elbo={self.elbo:.4g})"


def structured_vi(observations, guide: Guide, *, max_its: int = 300, tol: float = 1e-8) -> StructuredVIPosterior:
    """Fit a structured model by mean-field VMP / coordinate-ascent VI under a declared :class:`Guide`.

    Args:
        observations: a list of ``(model, data)`` pairs -- each ``model`` is a PPL ``RandomVariable``
            observation factor, ``data`` its observed values. Factors that reuse the same latent handle
            share that latent (their evidence is combined). A single factor may be passed as
            ``(model, data)`` directly.
        guide: the :class:`Guide` naming the latents to approximate and (optionally) their q-families.
        max_its / tol: CAVI sweep budget and ELBO convergence tolerance.

    Returns:
        A :class:`StructuredVIPosterior` over the guide's latents (monotone ELBO).

    Raises:
        ValueError: if a guide latent is not actually an inferred latent of the model, or its declared
            q-family does not match the model's conjugate factor (the projection constraint is checked).
    """
    if isinstance(observations, tuple) and len(observations) == 2 and isinstance(observations[0], RandomVariable):
        observations = [observations]
    pairs = list(observations)
    if not pairs:
        raise ValueError("structured_vi needs at least one (model, data) observation factor.")

    g = Graph()
    for model, data in pairs:
        if not isinstance(model, RandomVariable):
            raise TypeError("each observation must be (model_RandomVariable, data).")
        g.observe(model, data)
    res = g.fit(max_its=max_its, tol=tol)

    # the projection constraint: every guide latent must be an inferred node, and (if a family was
    # declared) the model's conjugate factor for it must be that family.
    for name, (handle, family) in guide._latents.items():
        node = res._node_of.get(id(handle))
        if node is None:
            raise ValueError(
                f"guide latent {name!r} is not an inferred latent of the model -- check that this exact "
                "handle object appears in an observation factor (latents are matched by object identity)."
            )
        if family is not None and not isinstance(node, _NODE_OF_FAMILY[family]):
            got = _FAMILY_OF_NODE.get(type(node), type(node).__name__)
            raise ValueError(
                f"guide declares q({name})={family!r}, but the model's conjugate factor for it is {got!r}. "
                "Either drop the explicit family (use the conjugate default) or give the latent a prior "
                f"whose conjugate factor is {family!r}."
            )
    return StructuredVIPosterior(res, guide)


__all__ = ["Guide", "StructuredVIPosterior", "structured_vi"]
