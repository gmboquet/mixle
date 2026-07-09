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
Gaussian/Gamma/Dirichlet factors, including hierarchies and shared latents.

**Admixtures / LDA-class models** -- a *latent* per-token categorical that indexes shared topics, drawn
from a per-group Dirichlet -- are expressed through :func:`admixture` (LDA is its categorical-word-emission
special case), built from the same Dirichlet primitives by mean-field VI. Remaining gaps: non-conjugate
q-families, and latent-feature (IBP-style) factors -- those raise a clear error rather than a wrong answer.
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
        """Return the declared latent names in guide order."""
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
        """Return posterior parameters for a declared latent name or raw handle."""
        return self._g.posterior(self._handle(name))

    def mean(self, name):
        """Return the posterior mean for a declared latent."""
        return self.posterior(name).get("mean")

    def samples(self, name, n: int = 4000, rng=None):
        """Draw samples from the variational factor for a declared latent."""
        return self._g.samples(self._handle(name), n=n, rng=rng)

    def summary(self) -> dict:
        """Return per-latent posterior parameters and ELBO metadata."""
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


# ---------------------------------------------------------------------------------------------------
# The categorical-emission / admixture factor: LDA-class models expressed through this surface.
#
# An admixture adds a *latent* per-token categorical z (the topic assignment) drawn from a per-group
# Dirichlet theta_d, indexing a shared set of Dirichlet topics beta_k -- the structure the plain
# Dirichlet-Categorical factor lacks. Mean-field VI (Blei et al. 2003) fits it from the SAME Dirichlet
# primitives the rest of this surface uses: q(beta_k)=Dirichlet (the declared topic handles),
# q(theta_d)=Dirichlet (per document), q(z)=categorical responsibilities marginalized by coordinate
# ascent. No LDADistribution -- LDA is just an admixture with categorical word-emission.
# ---------------------------------------------------------------------------------------------------
def _dirichlet_expectation(alpha):
    from scipy.special import digamma

    return digamma(alpha) - digamma(alpha.sum(axis=-1, keepdims=True))


class AdmixturePosterior:
    """Posterior from :func:`admixture` (LDA-class mean-field VI). ``posterior(topic_handle)`` returns
    that topic's fitted Dirichlet ``{'alpha','mean'}``; ``topics()`` is ``E[beta]`` (K x V); ``doc_topics``
    is the per-document mixing ``E[theta_d]``; ``log_likelihood`` is the fitted-model corpus LL trace."""

    def __init__(self, lam, gamma, topic_handles, ll_trace):
        import numpy as _np

        self._lam = _np.asarray(lam)  # K x V topic Dirichlet posteriors
        self._gamma = _np.asarray(gamma)  # D x K doc-topic Dirichlet posteriors
        self._topics = list(topic_handles)
        self.log_likelihood = float(ll_trace[-1]) if len(ll_trace) else float("nan")
        self.log_likelihood_trace = _np.asarray(ll_trace)
        self.acceptance_rate = None

    def topics(self):
        """Return the posterior mean topic-word matrix ``E[beta]``."""
        return self._lam / self._lam.sum(axis=1, keepdims=True)

    def doc_topics(self, d=None):
        """Return posterior mean document-topic weights for one document or all documents."""
        m = self._gamma / self._gamma.sum(axis=1, keepdims=True)
        return m if d is None else m[d]

    def posterior(self, topic):
        """Return the fitted Dirichlet posterior for a topic handle or topic index."""
        k = self._topics.index(topic) if topic in self._topics else int(topic)
        a = self._lam[k]
        return {"alpha": a, "mean": a / a.sum()}

    def summary(self) -> dict:
        """Return corpus-level topic count, vocabulary size, and likelihood metadata."""
        return {"n_topics": self._lam.shape[0], "vocab_size": self._lam.shape[1], "log_likelihood": self.log_likelihood}


def admixture(docs, topics, *, alpha=1.0, max_its: int = 100, inner_its: int = 40, tol: float = 1e-5, seed: int = 0):
    """Fit an admixture (LDA-class) model by mean-field VI built from this surface's Dirichlet primitives.

    ``docs`` is a corpus -- a list of documents, each a sequence of integer word ids over the vocabulary.
    ``topics`` are the ``Dirichlet`` ``RandomVariable`` handles you declare as the per-topic variational
    factors q(beta_k); their prior alpha vectors set the vocabulary size V and the topic-word prior eta.
    ``alpha`` is the per-document topic Dirichlet prior. Returns an :class:`AdmixturePosterior`.

    This is "LDA via the guide": LDA = an admixture whose emission is Categorical over words. The same
    coordinate-ascent (q(theta_d), q(beta_k) Dirichlet + categorical responsibilities q(z)) fits any
    admixture over a categorical vocabulary -- no ``LDADistribution`` involved.
    """
    import numpy as np

    from mixle.ppl.core import RandomVariable

    if not topics or any(not isinstance(t, RandomVariable) or t._family.name != "Dirichlet" for t in topics):
        raise TypeError(
            "`topics` must be a non-empty list of Dirichlet RandomVariable handles (the q(beta_k) factors)."
        )
    eta = np.array([np.asarray(t._args[0], dtype=float) for t in topics])  # (K, V) per-topic word priors
    K, V = eta.shape
    docs = [np.asarray(d, dtype=int) for d in docs]
    if any(d.size and (d.max() >= V or d.min() < 0) for d in docs):
        raise ValueError(f"document word ids must be in [0, {V}) for topics of vocabulary size {V}.")

    rng = np.random.RandomState(seed)
    lam = rng.gamma(100.0, 1.0 / 100.0, (K, V)) + eta  # random init breaks the topic-label symmetry
    gamma = np.full((len(docs), K), float(alpha) + 1.0)
    ll_trace = []
    for _ in range(int(max_its)):
        elog_beta = _dirichlet_expectation(lam)  # (K, V)
        new_lam = eta.astype(float).copy()
        ll = 0.0
        beta_hat = lam / lam.sum(1, keepdims=True)
        for d, w in enumerate(docs):
            if w.size == 0:
                continue
            elog_beta_w = elog_beta[:, w]  # (K, n)
            g = gamma[d].copy()
            for _ in range(int(inner_its)):
                elog_theta = _dirichlet_expectation(g)  # (K,)
                from scipy.special import logsumexp

                log_phi = elog_theta[:, None] + elog_beta_w  # (K, n)
                log_phi -= logsumexp(log_phi, axis=0, keepdims=True)
                phi = np.exp(log_phi)  # (K, n)  responsibilities q(z)
                g_new = float(alpha) + phi.sum(axis=1)
                if np.mean(np.abs(g_new - g)) < tol:
                    g = g_new
                    break
                g = g_new
            gamma[d] = g
            np.add.at(new_lam, (slice(None), w), phi)  # lam[k, w_n] += phi[k, n]
            theta_hat = g / g.sum()
            ll += float(np.sum(np.log(theta_hat @ beta_hat[:, w] + 1e-300)))  # fitted-model doc LL
        lam = new_lam
        ll_trace.append(ll)
        if len(ll_trace) > 1 and abs(ll_trace[-1] - ll_trace[-2]) < 1e-6 * max(1.0, abs(ll_trace[-2])):
            break
    return AdmixturePosterior(lam, gamma, topics, ll_trace)


__all__ = ["AdmixturePosterior", "Guide", "StructuredVIPosterior", "admixture", "structured_vi"]
