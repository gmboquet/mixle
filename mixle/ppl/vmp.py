"""Variational Message Passing (VMP) engine for mixle.ppl.

A message-passing engine for conjugate-exponential (Gaussian-Gamma) models
(Winn & Bishop, 2005). Each unobserved node carries a variational factor q in an
exponential family, holds its *natural parameters*, and exchanges messages with the
factors it touches —

    node posterior natural params = prior natural params + sum of incoming factor messages

Coordinate ascent updates each node from the others' expected sufficient statistics; the
ELBO is computed each sweep and increases monotonically.

The graph is built from a model by **object identity**: the same ``RandomVariable`` handle
used in multiple positions becomes ONE node that combines messages from every factor
touching it (parameter tying / shared latents). Priors that are themselves handles become
parent nodes — hierarchies of any depth. Use :class:`Graph` directly for multi-factor
models, or ``fit(how="vmp")`` which auto-builds a single-factor graph.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import digamma, gammaln

from mixle.ppl.core import RandomVariable, free

_LOG2PI = math.log(2.0 * math.pi)


def _positive_scalar(value, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive scalar.") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{label} must be a finite positive scalar.")
    return out


def _positive_int(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _scalar_observations(data, label: str = "observations") -> np.ndarray:
    try:
        out = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric one-dimensional sequence.") from exc
    if out.ndim != 1 or out.size == 0 or not np.isfinite(out).all():
        raise ValueError(f"{label} must be a non-empty finite one-dimensional sequence.")
    return out.copy()


# ----------------------------------------------------------------- nodes & constants
class MeanConst:
    """Constant mean term used where the graph expects a Gaussian mean node."""

    def __init__(self, v):
        self.v = float(v)
        if not math.isfinite(self.v):
            raise ValueError("a fixed Gaussian mean must be finite.")

    def ex(self):
        """Return ``E[x]`` for the constant mean."""
        return self.v

    def ex2(self):
        """Return ``E[x^2]`` for the constant mean."""
        return self.v * self.v


class PrecConst:
    """Constant precision term used where the graph expects a precision node."""

    def __init__(self, v):
        self.v = _positive_scalar(v, "fixed precision")

    def et(self):
        """Return ``E[tau]`` for the constant precision."""
        return self.v

    def elogt(self):
        """Return ``E[log tau]`` for the constant precision."""
        return math.log(self.v)


class GaussianVNode:
    """Node q(x) = Normal(m, s2); prior Normal(prior_mean, 1/prior_prec) where prior_mean
    may be another node (hierarchy). ``inbox`` holds message thunks from the factors /
    children touching this node — the mechanism behind sharing."""

    is_gaussian = True

    def __init__(self, prior_mean, prior_prec):
        self.prior_mean = prior_mean  # MeanConst or GaussianVNode
        self.prior_prec = prior_prec  # PrecConst (link precision)
        self.inbox = []  # list of () -> (eta1, eta2)
        self.m, self.s2 = prior_mean.ex(), 1.0 / prior_prec.et()

    def ex(self):
        """Return the variational expectation ``E[x]``."""
        return self.m

    def ex2(self):
        """Return the second moment ``E[x^2]`` under ``q(x)``."""
        return self.m * self.m + self.s2

    def update(self):
        """Apply one coordinate-ascent natural-parameter update."""
        pt = self.prior_prec.et()
        e1 = pt * self.prior_mean.ex()
        e2 = -0.5 * pt
        for msg in self.inbox:
            a, b = msg()
            e1 += a
            e2 += b
        if not math.isfinite(e1) or not math.isfinite(e2) or e2 >= 0.0:
            raise FloatingPointError("Gaussian VMP update produced invalid natural parameters.")
        self.s2 = -0.5 / e2
        self.m = e1 * self.s2
        if not math.isfinite(self.m) or not math.isfinite(self.s2) or self.s2 <= 0.0:
            raise FloatingPointError("Gaussian VMP update produced an invalid variational factor.")

    def entropy(self):
        """Return the entropy of the Gaussian variational factor."""
        return 0.5 * (_LOG2PI + 1.0 + math.log(self.s2))

    def cross_prior(self):
        """Return ``E_q[log p(x | parent)]`` for the Gaussian prior factor."""
        pm, pt = self.prior_mean, self.prior_prec
        e_sq = self.ex2() - 2.0 * pm.ex() * self.ex() + pm.ex2()
        return 0.5 * (pt.elogt() - _LOG2PI) - 0.5 * pt.et() * e_sq


class GammaVNode:
    """Node q(t) = Gamma(a, b) over a precision."""

    is_gaussian = False

    def __init__(self, a0, b0):
        self.a0 = _positive_scalar(a0, "Gamma prior shape")
        self.b0 = _positive_scalar(b0, "Gamma prior rate")
        self.a, self.b = self.a0, self.b0
        self.inbox = []

    def et(self):
        """Return the expected precision ``E[tau]``."""
        return self.a / self.b

    def elogt(self):
        """Return the expected log precision ``E[log tau]``."""
        return digamma(self.a) - math.log(self.b)

    def update(self):
        """Apply one coordinate-ascent update from accumulated messages."""
        self.a = self.a0 + sum(msg()[0] for msg in self.inbox)
        self.b = self.b0 + sum(msg()[1] for msg in self.inbox)
        if not math.isfinite(self.a) or not math.isfinite(self.b) or self.a <= 0.0 or self.b <= 0.0:
            raise FloatingPointError("Gamma VMP update produced an invalid variational factor.")

    def entropy(self):
        """Return the entropy of the Gamma variational factor."""
        return self.a - math.log(self.b) + gammaln(self.a) + (1.0 - self.a) * digamma(self.a)

    def cross_prior(self):
        """Return ``E_q[log p(tau)]`` under the Gamma prior."""
        return self.a0 * math.log(self.b0) - gammaln(self.a0) + (self.a0 - 1.0) * self.elogt() - self.b0 * self.et()


class DirichletVNode:
    """Node q(pi) = Dirichlet(alpha) over a simplex (categorical probabilities)."""

    is_gaussian = False

    def __init__(self, alpha0):
        self.alpha0 = np.asarray(alpha0, dtype=float)
        if (
            self.alpha0.ndim != 1
            or self.alpha0.size == 0
            or not np.isfinite(self.alpha0).all()
            or np.any(self.alpha0 <= 0.0)
        ):
            raise ValueError("Dirichlet concentration must be a non-empty finite positive vector.")
        self.alpha = self.alpha0.copy()
        self.inbox = []

    def expected(self):  # E[pi]
        """Return the simplex mean ``E[pi]``."""
        return self.alpha / self.alpha.sum()

    def expected_log(self):  # E[log pi_k]
        """Return the vector ``E[log pi_k]``."""
        return digamma(self.alpha) - digamma(self.alpha.sum())

    def update(self):
        """Apply one coordinate-ascent update from categorical count messages."""
        total = self.alpha0.copy()
        for msg in self.inbox:
            total = total + msg()  # accumulate expected counts (sharing!)
        if not np.isfinite(total).all() or np.any(total <= 0.0):
            raise FloatingPointError("Dirichlet VMP update produced an invalid variational factor.")
        self.alpha = total

    def _log_beta(self, a):
        return float(np.sum(gammaln(a)) - gammaln(a.sum()))

    def entropy(self):
        """Return the entropy of the Dirichlet variational factor."""
        a, a0, K = self.alpha, self.alpha.sum(), self.alpha.size
        return self._log_beta(a) + (a0 - K) * digamma(a0) - float(np.sum((a - 1.0) * digamma(a)))

    def cross_prior(self):
        """Return ``E_q[log p(pi)]`` under the Dirichlet prior."""
        return -self._log_beta(self.alpha0) + float(np.sum((self.alpha0 - 1.0) * self.expected_log()))


class _CategoricalFactor:
    """Categorical likelihood over category counts with a Dirichlet (or constant) parameter."""

    def __init__(self, pi, counts):
        self.pi = pi
        self.counts = np.asarray(counts, dtype=float)

    def wire(self):
        if isinstance(self.pi, DirichletVNode):
            self.pi.inbox.append(lambda: self.counts)  # message = observed counts

    def elbo(self):
        if isinstance(self.pi, DirichletVNode):
            return float(np.dot(self.counts, self.pi.expected_log()))
        return float(np.dot(self.counts, np.log(np.asarray(self.pi, dtype=float))))


class _GraphFactor:
    """Gaussian likelihood y ~ Normal(mean, 1/prec) over a data plate."""

    def __init__(self, mean, prec, data):
        a = _scalar_observations(data)
        self.mean, self.prec = mean, prec
        self.N, self.sum, self.sumsq = float(a.size), float(a.sum()), float((a * a).sum())

    def ess(self):
        return self.sumsq - 2.0 * self.mean.ex() * self.sum + self.N * self.mean.ex2()

    def wire(self):
        if isinstance(self.mean, GaussianVNode):
            self.mean.inbox.append(lambda: (self.prec.et() * self.sum, -0.5 * self.prec.et() * self.N))
        if isinstance(self.prec, GammaVNode):
            self.prec.inbox.append(lambda: (0.5 * self.N, 0.5 * self.ess()))

    def elbo(self):
        return 0.5 * self.N * (self.prec.elogt() - _LOG2PI) - 0.5 * self.prec.et() * self.ess()


# --------------------------------------------------------------------------- results
class GraphResult:
    """Fitted VMP graph with posterior accessors for graph node handles."""

    def __init__(self, node_of, elbo_trace, *, converged: bool, termination_reason: str):
        # Snapshot every factor.  A Graph can be refit, but a previously returned
        # scientific result must remain an immutable record of that run.
        self._posterior_of = {}
        for key, node in node_of.items():
            if isinstance(node, GaussianVNode):
                self._posterior_of[key] = ("gaussian", float(node.m), float(node.s2))
            elif isinstance(node, DirichletVNode):
                self._posterior_of[key] = ("dirichlet", node.alpha.copy())
            else:
                self._posterior_of[key] = ("gamma", float(node.a), float(node.b))
        self.elbo_trace = np.asarray(elbo_trace, dtype=float).copy()
        self.elbo_trace.setflags(write=False)
        self.elbo = float(self.elbo_trace[-1])
        self.converged = require_exact_bool(converged, "converged")
        self.iterations = int(self.elbo_trace.size)
        self.termination_reason = str(termination_reason)
        self.objective_delta = (
            abs(float(self.elbo_trace[-1] - self.elbo_trace[-2])) if self.elbo_trace.size > 1 else math.inf
        )
        self.acceptance_rate = None

    def _node(self, rv):
        posterior = self._posterior_of.get(id(rv))
        if posterior is None:
            raise KeyError("variable is not a node in this graph.")
        return posterior

    def _family(self, rv):
        """Return the conjugate-family tag inferred for ``rv``, or None if it is not a node here.

        The snapshot above deliberately keeps tags rather than live ``*VNode`` objects, so callers
        that need to know which conjugate factor a latent got must ask for the tag instead of
        type-checking a node. ``mixle.ppl.guide`` uses this to validate a declared ``Guide`` family.
        """
        posterior = self._posterior_of.get(id(rv))
        return None if posterior is None else posterior[0]

    def posterior(self, rv):
        """Return posterior parameters for a latent handle in the fitted graph."""
        node = self._node(rv)
        if node[0] == "gaussian":
            return {"mean": node[1], "sd": math.sqrt(node[2])}
        if node[0] == "dirichlet":
            alpha = node[1].copy()
            return {"alpha": alpha, "mean": alpha / alpha.sum()}
        return {"shape": node[1], "rate": node[2], "mean": node[1] / node[2]}

    def samples(self, rv, n: int = 4000, rng=None):
        """Draw samples from the variational factor attached to ``rv``."""
        rng = rng or np.random.RandomState()
        node = self._node(rv)
        if node[0] == "gaussian":
            return rng.normal(node[1], math.sqrt(node[2]), n)
        if node[0] == "dirichlet":
            return rng.dirichlet(node[1], n)
        return rng.gamma(node[1], 1.0 / node[2], n)


# --------------------------------------------------------------------------- the graph
class Graph:
    """A VMP factor graph for arbitrary conjugate-Gaussian DAGs with shared variables.

        mu = Normal(0, 10)                       # one shared latent handle
        fit = (Graph()
               .observe(Normal(mu, 1.0), data_a) # factor A uses mu
               .observe(Normal(mu, 2.0), data_b) # factor B uses the SAME mu
               .fit())
        fit.posterior(mu)                        # evidence from A and B combined

    A prior that is itself a RandomVariable becomes a parent node (hierarchy, any depth).
    A Gamma in a scale slot is read as a prior on the precision (the conjugate choice).
    """

    def __init__(self):
        self._obs = []  # (model_rv, data)
        self._nodes = {}  # id(rv) -> node

    def observe(self, model, data) -> Graph:
        """Add an observed likelihood factor and return ``self`` for chaining."""
        try:
            frozen_data = tuple(data)
        except TypeError as exc:
            raise ValueError("graph observations must be an iterable of scalar values.") from exc
        self._obs.append((model, frozen_data))
        return self

    def _mean_of(self, spec):
        if not isinstance(spec, RandomVariable):
            return MeanConst(spec)
        if spec._family.name != "Normal":
            raise NotImplementedError("graph mean priors must be Normal.")
        if id(spec) in self._nodes:
            return self._nodes[id(spec)]  # SHARED instance -> one node
        mean_arg, scale_arg = spec._args
        prior_mean = self._mean_of(mean_arg) if isinstance(mean_arg, RandomVariable) else MeanConst(mean_arg)
        prior_sd = _positive_scalar(scale_arg, "Gaussian prior standard deviation")
        prior_prec = PrecConst(1.0 / prior_sd**2)
        node = GaussianVNode(prior_mean, prior_prec)
        self._nodes[id(spec)] = node
        if isinstance(prior_mean, GaussianVNode):  # child -> parent message (hierarchy)
            prior_mean.inbox.append(lambda c=node: (c.prior_prec.et() * c.ex(), -0.5 * c.prior_prec.et()))
        return node

    def _prec_of(self, spec):
        if not isinstance(spec, RandomVariable):
            sd = _positive_scalar(spec, "Gaussian observation standard deviation")
            return PrecConst(1.0 / sd**2)  # constant sd -> precision
        if spec._family.name != "Gamma":
            raise NotImplementedError("graph precision priors must be Gamma.")
        if id(spec) in self._nodes:
            return self._nodes[id(spec)]  # SHARED precision
        node = GammaVNode(spec._args[0], spec._args[1])
        self._nodes[id(spec)] = node
        return node

    def _pi_of(self, spec):
        if not isinstance(spec, RandomVariable):
            return np.asarray(spec, dtype=float)  # constant probability vector
        if spec._family.name != "Dirichlet":
            raise NotImplementedError("graph categorical priors must be Dirichlet.")
        if id(spec) in self._nodes:
            return self._nodes[id(spec)]  # SHARED simplex
        node = DirichletVNode(spec._args[0])
        self._nodes[id(spec)] = node
        return node

    def _make_factor(self, model, data):
        if not isinstance(model, RandomVariable) or model._kind != "sample":
            raise TypeError("graph observations require a sampled RandomVariable model.")
        fam = model._family.name
        if fam == "Normal":
            return _GraphFactor(self._mean_of(model._args[0]), self._prec_of(model._args[1]), data)
        if fam == "Categorical":
            pi = self._pi_of(model._args[0])
            K = pi.alpha0.size if isinstance(pi, DirichletVNode) else len(pi)
            raw = _scalar_observations(data, "categorical observations")
            if np.any(raw != np.floor(raw)) or np.any(raw < 0.0) or np.any(raw >= K):
                raise ValueError(f"categorical observations must be exact integer labels in [0, {K}).")
            counts = np.bincount(raw.astype(int), minlength=K).astype(float)
            return _CategoricalFactor(pi, counts)
        raise NotImplementedError(f"graph observations of family {fam} are not supported.")

    def fit(self, *, max_its: int = 300, tol: float = 1e-8) -> GraphResult:
        """Run coordinate-ascent VMP and return the fitted graph result."""
        max_its = _positive_int(max_its, "max_its")
        tol = _positive_scalar(tol, "tol")
        if not self._obs:
            raise ValueError("a VMP graph must contain at least one observation factor.")
        # Nodes and message inboxes are run-local.  Reusing a Graph starts from
        # the declared priors and cannot retain evidence from an earlier fit.
        self._nodes = {}
        factors = [self._make_factor(model, data) for model, data in self._obs]
        for f in factors:
            f.wire()
        by_type = lambda T: [n for n in self._nodes.values() if isinstance(n, T)]
        gaussian, gamma, dirichlet = by_type(GaussianVNode), by_type(GammaVNode), by_type(DirichletVNode)

        trace = []
        converged = False
        for _ in range(max_its):
            for n in gaussian:
                n.update()
            for n in gamma:
                n.update()
            for n in dirichlet:
                n.update()
            elbo = (
                sum(f.elbo() for f in factors)
                + sum(n.cross_prior() + n.entropy() for n in gaussian)
                + sum(n.cross_prior() + n.entropy() for n in gamma)
                + sum(n.cross_prior() + n.entropy() for n in dirichlet)
            )
            if not math.isfinite(elbo):
                raise FloatingPointError("VMP produced a non-finite ELBO.")
            trace.append(elbo)
            if len(trace) > 1 and abs(trace[-1] - trace[-2]) < tol:
                converged = True
                break
        reason = "tolerance" if converged else "max_iterations"
        return GraphResult(dict(self._nodes), trace, converged=converged, termination_reason=reason)


# ----------------------------------------------------- ppl entry point: fit(how="vmp")
class _VMPFit:
    """Result for ``fit(how="vmp")``: wraps a GraphResult, plus convenience views of the
    top observation's mean / precision nodes (q_mu, q_tau) and posterior-predictive."""

    def __init__(self, gres: GraphResult, mean_handle, prec_handle):
        self._g = gres
        self._mean_handle, self._prec_handle = mean_handle, prec_handle
        self.elbo_trace = gres.elbo_trace
        self.elbo = gres.elbo
        self.converged = gres.converged
        self.iterations = gres.iterations
        self.termination_reason = gres.termination_reason
        self.objective_delta = gres.objective_delta
        self.acceptance_rate = None
        self.q_mu = gres.posterior(mean_handle) if mean_handle is not None else None
        self.q_tau = gres.posterior(prec_handle) if prec_handle is not None else None
        self.predictive = None

    def posterior(self, handle):
        """Return posterior parameters for a latent handle in the wrapped graph."""
        return self._g.posterior(handle)

    def samples(self, param=None, n: int = 4000, rng=None):
        """Draw samples for the named mean, precision, standard deviation, or raw handle."""
        rng = rng or np.random.RandomState()
        if param in ("mu", "mean", 0) and self.q_mu is not None:
            return rng.normal(self.q_mu["mean"], self.q_mu["sd"], n)
        if param in ("tau", "precision") and self.q_tau is not None:
            return rng.gamma(self.q_tau["shape"], 1.0 / self.q_tau["rate"], n)
        if param == "sd" and self.q_tau is not None:
            return 1.0 / np.sqrt(rng.gamma(self.q_tau["shape"], 1.0 / self.q_tau["rate"], n))
        return self._g.samples(param, n=n, rng=rng)  # any node handle

    def summary(self) -> dict:
        """Return mean, precision, ELBO, and iteration metadata for the fit."""
        return {
            "q_mu": self.q_mu,
            "q_tau": self.q_tau,
            "elbo": self.elbo,
            "iterations": self.iterations,
            "converged": self.converged,
            "termination_reason": self.termination_reason,
            "objective_delta": self.objective_delta,
        }


def vmp_fit(rv: RandomVariable, data, *, max_its: int = 300, tol: float = 1e-8, rng=None) -> RandomVariable:
    """Auto-build a single-factor VMP graph for a nested Gaussian model and fit it.

    Handles ``Normal(mean, scale)`` where ``mean`` is a (possibly deeply nested) Normal
    prior chain and ``scale`` is a constant sd or a Gamma prior on the precision — e.g.
    ``Normal(Normal(0,10), Gamma(1,1))`` (unknown mean + precision) or
    ``Normal(Normal(Normal(0,100), 5), 1)`` (mean with a hyperprior). For multi-factor
    models or shared variables across datasets, use :class:`Graph` directly.
    """
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    if rv._kind != "sample" or rv._family.name != "Normal" or len(rv._args) != 2:
        raise NotImplementedError("vmp supports Normal(mean, scale) Gaussian models.")
    mean_spec, scale_spec = rv._args
    if any(a is free for a in rv._args):
        raise NotImplementedError(
            "vmp is closed-form variational message passing and needs a *prior* (or a fixed "
            "constant) on each parameter, not the point-estimate token `free`; give the slot a "
            "conjugate prior — Normal(Normal(0, 10), Gamma(1, 1)) for unknown mean+precision — "
            "or use how='vi' (general mean-field VB) or how='map'/'mcmc' for `free` parameters."
        )
    if not (isinstance(mean_spec, RandomVariable) or isinstance(scale_spec, RandomVariable)):
        raise NotImplementedError("nothing to infer: give the mean and/or scale a prior.")

    g = Graph().observe(rv, data)
    res = g.fit(max_its=max_its, tol=tol)

    mean_handle = mean_spec if isinstance(mean_spec, RandomVariable) else None
    mean_q = res.posterior(mean_handle) if mean_handle is not None else None
    mean_val = mean_q["mean"] if mean_q is not None else float(mean_spec)
    prec_handle = scale_spec if isinstance(scale_spec, RandomVariable) else None
    prec_q = res.posterior(prec_handle) if prec_handle is not None else None
    sd = (1.0 / math.sqrt(prec_q["mean"])) if prec_q is not None else float(scale_spec)

    fitted = GaussianDistribution(mu=mean_val, sigma2=sd * sd, name=rv._name)
    result = _VMPFit(res, mean_handle, prec_handle)

    def predictive(n, r):
        mu = r.normal(mean_q["mean"], mean_q["sd"], n) if mean_q is not None else np.full(n, mean_val)
        if prec_q is not None:
            tau = r.gamma(prec_q["shape"], 1.0 / prec_q["rate"], n)
            return mu + r.normal(0.0, 1.0, n) / np.sqrt(tau)
        return mu + r.normal(0.0, sd, n)

    result.predictive = predictive
    return RandomVariable._bound(fitted, name=rv._name, result=result)


# ============================================================================
# Discrete latents in-graph: Bayesian Gaussian mixture via VBEM (mean-field VMP)
# ============================================================================
#
# Per-datapoint categorical latent z_n with variational responsibilities r_nk; component
# means q(mu_k)=Normal, precisions q(tau_k)=Gamma, weights q(pi)=Dirichlet. Coordinate
# ascent between responsibilities and parameters (Bishop PRML 10.2).

from scipy.special import logsumexp as _logsumexp

from mixle.utils.exact import require_exact_bool


def _kmeanspp(x, k, rng):
    first = rng.integers(x.size) if callable(getattr(rng, "integers", None)) else rng.randint(x.size)
    chosen = [int(first)]
    d2 = (x - x[chosen[0]]) ** 2
    for _ in range(1, k):
        tot = d2.sum()
        p = d2 / tot if tot > 0 else np.ones(x.size) / x.size
        j = int(rng.choice(x.size, p=p))
        chosen.append(j)
        d2 = np.minimum(d2, (x - x[j]) ** 2)
    return chosen


class MixtureVMPResult:
    """Variational result for a scalar Gaussian mixture with discrete responsibilities."""

    def __init__(
        self,
        weights,
        comps,
        responsibilities,
        elbo_trace,
        normalizer_trace,
        *,
        converged: bool,
        termination_reason: str,
    ):
        self.weights = np.asarray(weights, dtype=float).copy()  # E[pi]
        self.components = [dict(c) for c in comps]
        self.responsibilities = np.asarray(responsibilities, dtype=float).copy()  # (N, K)
        self.objective_kind = "finite_mixture_elbo"
        self.elbo = float(elbo_trace[-1])
        self.elbo_trace = np.asarray(elbo_trace, dtype=float).copy()
        self.responsibility_normalizer_trace = np.asarray(normalizer_trace, dtype=float).copy()
        self.converged = require_exact_bool(converged, "converged")
        self.iterations = int(self.elbo_trace.size)
        self.termination_reason = str(termination_reason)
        self.objective_delta = (
            abs(float(self.elbo_trace[-1] - self.elbo_trace[-2])) if self.elbo_trace.size > 1 else math.inf
        )
        self.acceptance_rate = None
        self.predictive = None
        self.plugin_distribution = None

    def summary(self):
        """Return mixture weights, component summaries, and objective metadata."""
        return {
            "weights": self.weights,
            "components": self.components,
            "elbo": self.elbo,
            "objective_kind": self.objective_kind,
            "iterations": self.iterations,
            "converged": self.converged,
            "termination_reason": self.termination_reason,
            "objective_delta": self.objective_delta,
        }


def _mixture_vmp_elbo(x, r, m, s2, a, b, alpha, *, m0, s0, a0, b0, alpha0):
    """Full mean-field ELBO for the scalar Gaussian mixture VB helper."""
    K = alpha.size
    Etau = a / b
    Elogtau = digamma(a) - np.log(b)
    Elogpi = digamma(alpha) - digamma(alpha.sum())
    diff2 = (x[:, None] - m[None, :]) ** 2 + s2[None, :]

    likelihood = np.sum(r * (0.5 * (Elogtau[None, :] - _LOG2PI) - 0.5 * Etau[None, :] * diff2))
    allocation = np.sum(r * Elogpi[None, :])
    positive_r = r > 0.0
    entropy_z = -np.sum(r[positive_r] * np.log(r[positive_r]))

    prior_pi = gammaln(K * alpha0) - K * gammaln(alpha0) + (alpha0 - 1.0) * np.sum(Elogpi)
    log_beta_q = np.sum(gammaln(alpha)) - gammaln(alpha.sum())
    entropy_pi = log_beta_q + (alpha.sum() - K) * digamma(alpha.sum()) - np.sum((alpha - 1.0) * digamma(alpha))

    prior_mu = -0.5 * np.sum(_LOG2PI + math.log(s0 * s0) + ((m - m0) ** 2 + s2) / (s0 * s0))
    entropy_mu = 0.5 * np.sum(_LOG2PI + 1.0 + np.log(s2))

    prior_tau = np.sum(a0 * math.log(b0) - gammaln(a0) + (a0 - 1.0) * Elogtau - b0 * Etau)
    entropy_tau = np.sum(a - np.log(b) + gammaln(a) + (1.0 - a) * digamma(a))

    return float(
        likelihood + allocation + entropy_z + prior_pi + entropy_pi + prior_mu + entropy_mu + prior_tau + entropy_tau
    )


def mixture_vmp(data, K, *, max_its=300, tol=1e-7, rng=None, m0=None, s0=None, a0=1.0, b0=1.0, alpha0=1.0):
    """Bayesian scalar Gaussian mixture by variational message passing (VBEM)."""
    from mixle.stats.latent.mixture import MixtureDistribution
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    x = _scalar_observations(data)
    N = x.size
    K = _positive_int(K, "K")
    if K > N:
        raise ValueError("K cannot exceed the number of observations.")
    max_its = _positive_int(max_its, "max_its")
    tol = _positive_scalar(tol, "tol")
    a0 = _positive_scalar(a0, "a0")
    b0 = _positive_scalar(b0, "b0")
    alpha0 = _positive_scalar(alpha0, "alpha0")
    if rng is None:
        rng = np.random.RandomState(0)
    if not callable(getattr(rng, "choice", None)) or not (
        callable(getattr(rng, "randint", None)) or callable(getattr(rng, "integers", None))
    ):
        raise TypeError("rng must provide choice() and randint() or integers().")
    var = float(x.var()) or 1.0
    if m0 is None:
        m0 = float(x.mean())
    else:
        try:
            m0 = float(m0)
        except (TypeError, ValueError) as exc:
            raise ValueError("m0 must be a finite scalar.") from exc
        if not math.isfinite(m0):
            raise ValueError("m0 must be a finite scalar.")
    if s0 is None:
        s0 = 10.0 * math.sqrt(var)
    else:
        s0 = _positive_scalar(s0, "s0")

    m = x[_kmeanspp(x, K, rng)].astype(float)  # component means (q means)
    s2 = np.full(K, var)
    a = np.full(K, a0 + N / (2.0 * K))
    b = np.full(K, b0 + 0.5 * var * N / K)  # so E[tau] ~ 1/var initially
    alpha = np.full(K, alpha0 + N / K)

    trace = []
    normalizer_trace = []
    converged = False
    for _ in range(max_its):
        Elogpi = digamma(alpha) - digamma(alpha.sum())
        Etau, Elogtau = a / b, digamma(a) - np.log(b)
        diff2 = (x[:, None] - m[None, :]) ** 2 + s2[None, :]  # E[(x-mu)^2]
        log_rho = Elogpi[None, :] + 0.5 * Elogtau[None, :] - 0.5 * _LOG2PI - 0.5 * Etau[None, :] * diff2
        log_norm = _logsumexp(log_rho, axis=1, keepdims=True)
        r = np.exp(log_rho - log_norm)  # responsibilities
        Nk = r.sum(0)
        sumx = (r * x[:, None]).sum(0)

        prec = 1.0 / s0**2 + Nk * Etau  # q(mu_k)
        s2 = 1.0 / prec
        m = (m0 / s0**2 + Etau * sumx) * s2
        diff2 = (x[:, None] - m[None, :]) ** 2 + s2[None, :]
        a = a0 + 0.5 * Nk  # q(tau_k)
        b = b0 + 0.5 * (r * diff2).sum(0)
        alpha = alpha0 + Nk  # q(pi)

        # Complete the responsibility coordinate update after the parameter
        # update so every stored diagnostic describes this same iterate.
        Elogpi = digamma(alpha) - digamma(alpha.sum())
        Etau, Elogtau = a / b, digamma(a) - np.log(b)
        diff2 = (x[:, None] - m[None, :]) ** 2 + s2[None, :]
        log_rho = Elogpi[None, :] + 0.5 * Elogtau[None, :] - 0.5 * _LOG2PI - 0.5 * Etau[None, :] * diff2
        log_norm = _logsumexp(log_rho, axis=1, keepdims=True)
        r = np.exp(log_rho - log_norm)
        normalizer_trace.append(float(np.sum(log_norm)))
        elbo = _mixture_vmp_elbo(x, r, m, s2, a, b, alpha, m0=m0, s0=s0, a0=a0, b0=b0, alpha0=alpha0)
        if not math.isfinite(elbo) or not np.isfinite(r).all():
            raise FloatingPointError("mixture VMP produced non-finite variational state.")
        trace.append(elbo)
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < tol:
            converged = True
            break

    Etau = a / b
    weights = alpha / alpha.sum()
    sds = 1.0 / np.sqrt(Etau)
    comps = [
        {
            "mean": float(m[k]),
            "sd": float(sds[k]),
            "mean_sd": float(math.sqrt(s2[k])),
            "precision_shape": float(a[k]),
            "precision_rate": float(b[k]),
            "plugin_variance": float(b[k] / a[k]),
            "predictive_variance": float(s2[k] + b[k] / (a[k] - 1.0)) if a[k] > 1.0 else math.inf,
        }
        for k in range(K)
    ]
    fitted = MixtureDistribution([GaussianDistribution(float(m[k]), float(sds[k] ** 2)) for k in range(K)], w=weights)
    result = MixtureVMPResult(
        weights,
        comps,
        r,
        trace,
        normalizer_trace,
        converged=converged,
        termination_reason="tolerance" if converged else "max_iterations",
    )
    result.plugin_distribution = fitted

    # Posterior predictive: sample q(mu_k) and q(tau_k), then an observation.
    # The bound distribution above is deliberately the separately labelled
    # plug-in view needed by the existing distribution protocol.
    def predictive(n, rgen):
        n = _positive_int(n, "n")
        z = rgen.choice(K, size=n, p=weights)
        mu = rgen.normal(m[z], np.sqrt(s2[z]))
        tau = rgen.gamma(a[z], 1.0 / b[z])
        return mu + rgen.normal(0.0, 1.0, n) / np.sqrt(tau)

    result.predictive = predictive
    return RandomVariable._bound(fitted, result=result)
