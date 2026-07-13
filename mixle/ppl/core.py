"""Core of the mixle.ppl probabilistic-programming surface.

One immutable wrapper type, :class:`RandomVariable`, sits over mixle's existing
distribution / estimator / sampler objects. It adds *no* inference engine: every call
lowers (one routing site, :func:`lower`) to machinery that already exists and then
dispatches.

The core surface covers immutable random variables, ``free`` parameter holes,
fitting, and lowering into existing Mixle estimators and distributions. Unsupported
combinations should fail explicitly rather than returning a partially lowered model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.capability import supports
from mixle.inference.estimation import optimize
from mixle.ppl._result import PosteriorResult, Sampleable, Summarizable
from mixle.utils.aliasing import coalesce_alias

__all__ = [
    "RandomVariable",
    "free",
    "ordered",
    "lower",
    "register_family",
    "Family",
    "Constraint",
    "Event",
    "constrain",
    "eq",
    "equal",
    "ne",
    "increasing",
    "decreasing",
    "monotone",
    "convex",
    "concave",
    "lipschitz",
    "ode_residual",
]


# ----------------------------------------------------------------- fitter registry
# Pure ``how`` -> fitter dispatch for the inference (non-EM) paths. Each fitter has the uniform
# signature ``(rv: RandomVariable, data, **kw) -> RandomVariable`` and performs its own lazy import
# of the heavy inference module (the inference/vmp modules import ``core`` at module level, so the
# import must stay deferred to the call site to avoid a cycle). ``RandomVariable.fit`` derives
# ``valid_how`` from these keys plus the EM/auto entries it owns, and looks the fitter up here
# instead of walking an ``if how == ...`` ladder. Branches that need closure over the RV's local
# state (the ``vmp`` Mixture special case) are registered as small closures here too, so the whole
# pure-``how`` dispatch lives in one table.
_FITTERS: dict[str, Callable[..., RandomVariable]] = {}


def register_fitter(name: str) -> Callable[[Callable[..., RandomVariable]], Callable[..., RandomVariable]]:
    """Register ``fn`` as the fitter for ``how=name`` in :data:`_FITTERS`."""

    def deco(fn: Callable[..., RandomVariable]) -> Callable[..., RandomVariable]:
        _FITTERS[name] = fn
        return fn

    return deco


@register_fitter("map")
def _fit_map(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.map_fit(rv, data, **kw)


@register_fitter("laplace")
def _fit_laplace(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.laplace_fit(rv, data, **kw)


@register_fitter("mcmc")
def _fit_mcmc(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.mcmc_fit(rv, data, **kw)


@register_fitter("hmc")
def _fit_hmc(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.hmc_fit(rv, data, **kw)


@register_fitter("nuts")
def _fit_nuts(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.nuts_fit(rv, data, **kw)


@register_fitter("sample")
def _fit_sample(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.sample_fit(rv, data, **kw)


@register_fitter("ensemble")
def _fit_ensemble(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.ensemble_fit(rv, data, **kw)


@register_fitter("vi")
def _fit_vi(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.vi_fit(rv, data, **kw)


@register_fitter("conjugate")
def _fit_conjugate(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.conjugate_fit(rv, data, **kw)


@register_fitter("conjugate_mixture")
def _fit_conjugate_mixture(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.conjugate_mixture_fit(rv, data, **kw)


@register_fitter("hierarchical")
def _fit_hierarchical(rv, data, **kw):
    from mixle.ppl import inference as _inf

    return _inf.hierarchical_fit(rv, data, **kw)


@register_fitter("vmp")
def _fit_vmp(rv, data, **kw):
    # Entangled branch: a Mixture(...) RV reroutes to ``mixture_vmp`` using the RV's own family/args
    # (only the component COUNT is threaded; component-level priors/fixed params/constraints are NOT
    # applied -- use how='vi'/'mcmc' for those). Kept here as a closure over ``rv`` so the special
    # case travels with the rest of the pure-``how`` table rather than living in the ladder.
    from mixle.ppl import vmp as _vmp

    if isinstance(rv._family, CompositeFamily) and rv._family.name == "Mixture":
        comps = rv._args[0]
        return _vmp.mixture_vmp(data, len(comps), **kw)
    return _vmp.vmp_fit(rv, data, **kw)


# --------------------------------------------------------------------------- free
class _Free:
    """The ``free`` token: an argument slot to be estimated.

    Bare ``free`` is a scalar slot (identity ``arg is free`` is the test used during lowering).
    *Called*, it is a **vector/matrix parameter handle** you can both place in a slot and reference
    in constraints: ``free(dim)`` (a real vector), ``free(dim, name="mu")`` (named, for readout),
    ``free(dim, kind="ordered"|"simplex"|"cholesky")``, ``free(dim, support="positive")``. This
    subsumes the old ``param(...)`` helper — one token for "estimate this".
    """

    __slots__ = ()

    def __call__(self, dim: int, *, name=None, kind: str = "vector", support: str = "real"):
        return _param_handle(int(dim), name=name, kind=kind, support=support)

    def __mul__(self, other):  # free * Field -> an OLS regression coefficient
        if isinstance(other, Field):
            return _LinearPredictor([(self, other)])
        return NotImplemented

    __rmul__ = __mul__

    def __reduce__(self):  # preserve singleton identity across pickling
        return (_free_singleton, ())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "free"


free = _Free()


def _free_singleton():
    return free


def _is_free(x: Any) -> bool:
    return x is free


# --------------------------------------------------------------- covariates / GLM
class Field:
    """A named covariate (data column) for regression: ``a * Field("x") + b``."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __mul__(self, coef):  # Field * coef
        return _LinearPredictor([(coef, self)])

    __rmul__ = __mul__

    def __add__(self, other):
        return _LinearPredictor([(1.0, self)]).__add__(other)

    __radd__ = __add__

    def __repr__(self):
        return f"Field({self.name!r})"


def _combine_intercept(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if not isinstance(a, (_Free, RandomVariable)) and not isinstance(b, (_Free, RandomVariable)):
        return float(a) + float(b)
    raise ValueError("a linear predictor may have only one symbolic intercept.")


class Group:
    """A by-group random-effects term for mixed-effects models. ``Group("subject")`` is a
    random intercept (lme4's ``(1|subject)``); ``Group("subject", slopes=["x"])`` adds a
    correlated random slope on ``x`` (``(1 + x | subject)``)."""

    __slots__ = ("name", "slopes")

    def __init__(self, name: str, slopes=()):
        self.name = name
        self.slopes = tuple(s.name if isinstance(s, Field) else s for s in slopes)

    def _key(self):
        return (self.name, self.slopes)

    def __add__(self, other):
        return _LinearPredictor([], groups=[self._key()]).__add__(other)

    __radd__ = __add__

    def __repr__(self):
        return f"Group({self.name!r}, slopes={list(self.slopes)})"


class _LinearPredictor:
    """A linear predictor Σ coef_k · Field_k (+ intercept) (+ random intercepts by group).
    Coeffs are RVs (Gaussian priors), ``free`` (OLS), or constants."""

    __slots__ = ("terms", "intercept", "groups")

    def __init__(self, terms, intercept=None, groups=None):
        self.terms = list(terms)  # list of (coef, Field)
        self.intercept = intercept  # RandomVariable | free | float | None
        self.groups = list(groups or [])  # random-intercept group names

    def __add__(self, other):
        if isinstance(other, _LinearPredictor):
            return _LinearPredictor(
                self.terms + other.terms,
                _combine_intercept(self.intercept, other.intercept),
                self.groups + other.groups,
            )
        if isinstance(other, Field):
            return _LinearPredictor(self.terms + [(1.0, other)], self.intercept, self.groups)
        if isinstance(other, Group):
            return _LinearPredictor(self.terms, self.intercept, self.groups + [other._key()])
        return _LinearPredictor(self.terms, _combine_intercept(self.intercept, other), self.groups)

    __radd__ = __add__

    def __repr__(self):
        return f"_LinearPredictor({self.terms!r}, intercept={self.intercept!r}, groups={self.groups!r})"


class _NeuralPredictor:
    """Base for a neural predictor in a parameter slot -- the *nonlinear* sibling of :class:`_LinearPredictor`.

    Put one in an outer family's slot and the outer family sets the link, exactly as a linear predictor makes a
    GLM::

        Categorical(logits=Net(out=10))    # softmax link -> neural classification, p(y|x)
        Categorical(logits=Conv(out=10))    # ...over image covariates, with a conv net
        Normal(Net(out=1), free)            # identity link + learned noise -> neural regression

    Pure shape-data (no torch in user code); the torch module is built lazily at fit, with the input shape
    inferred from the covariates. Fit with the conditional verb, same as a GLM: ``.fit(y, given={"x": X})``.
    """

    __slots__ = ()

    def build(self, in_shape: Any) -> Any:  # in_shape: int (vector width) or (C, H, W) for images
        raise NotImplementedError


class Net(_NeuralPredictor):
    """An MLP predictor over vector covariates. ``Net(hidden=[256], out=10)`` is a one-hidden-layer ReLU net."""

    __slots__ = ("field", "hidden", "out")

    def __init__(self, field: Any = "x", *, hidden: Any = (64,), out: int = 1):
        self.field = field if isinstance(field, str) else getattr(field, "name", "x")
        self.hidden = tuple(int(h) for h in hidden)
        self.out = int(out)

    def build(self, in_shape: Any) -> Any:
        """Build a Torch MLP module for the inferred covariate shape."""
        import torch.nn as nn

        in_dim = int(in_shape if isinstance(in_shape, int) else int(np.prod(in_shape)))
        dims = [in_dim, *self.hidden, self.out]
        layers: list = [nn.Flatten()]  # accept vector or already-flat image covariates
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def __repr__(self) -> str:
        return f"Net(field={self.field!r}, hidden={self.hidden!r}, out={self.out})"


class Conv(_NeuralPredictor):
    """A conv-net predictor over image covariates ``(C, H, W)``. ``Conv(channels=[64,128,256], out=10)`` is a
    VGG-style stack (two 3x3 convs + BatchNorm + max-pool per channel stage), global-pooled into a linear head.
    Use it exactly like :class:`Net`: ``Categorical(logits=Conv(out=10)).fit(y, given={"x": images})``."""

    __slots__ = ("field", "channels", "out")

    def __init__(self, field: Any = "x", *, channels: Any = (64, 128, 256), out: int = 10):
        self.field = field if isinstance(field, str) else getattr(field, "name", "x")
        self.channels = tuple(int(c) for c in channels)
        self.out = int(out)

    def build(self, in_shape: Any) -> Any:
        """Build a Torch convolutional module for image-shaped covariates."""
        import torch.nn as nn

        c = int(in_shape[0])  # (C, H, W)
        layers: list = []
        prev = c
        for ch in self.channels:
            layers += [
                nn.Conv2d(prev, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = ch
        layers += [nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Dropout(0.3), nn.Linear(prev * 4, self.out)]
        return nn.Sequential(*layers)

    def __repr__(self) -> str:
        return f"Conv(field={self.field!r}, channels={self.channels!r}, out={self.out})"


class Transformer(_NeuralPredictor):
    """A causal decoder-only Transformer predictor over a ``(block,)`` context of token ids.

    ``Categorical(logits=Transformer(out=vocab, d_model=256, n_layer=6, n_head=8))`` is autoregressive
    next-token prediction ``p(token | context)`` -- it lowers to the same ``SoftmaxNeuralLeaf`` as ``Net``/``Conv``
    (cross-entropy = next-token NLL) and fits through the unchanged ``estimate()`` loop with
    ``.fit(next_tokens, given={"x": context_windows})``. The context width is the ``block`` inferred from the data.
    """

    __slots__ = ("field", "out", "d_model", "n_layer", "n_head", "embedding")

    def __init__(
        self,
        field: Any = "x",
        *,
        out: int,
        d_model: int = 128,
        n_layer: int = 3,
        n_head: int = 4,
        embedding: Any = None,
    ):
        self.field = field if isinstance(field, str) else getattr(field, "name", "x")
        self.out = int(out)
        self.d_model = int(d_model)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        # embedding=Embedding(...) ties one word embedding across every Transformer that references it (e.g. the
        # per-cluster language models of a Mix) -- they train the same token vectors jointly.
        self.embedding = embedding

    def build(self, in_shape: Any) -> Any:
        """Build a causal language-model module for the inferred token context width."""
        from mixle.models.transformer import build_causal_lm

        block = int(in_shape[0] if not isinstance(in_shape, int) else in_shape)
        return build_causal_lm(self.out, self.d_model, self.n_layer, self.n_head, block, embedding=self.embedding)

    def __repr__(self) -> str:
        emb = ", embedding=shared" if self.embedding is not None else ""
        return f"Transformer(out={self.out}, d_model={self.d_model}, n_layer={self.n_layer}, n_head={self.n_head}{emb})"


class _SimplexSpec:
    """A structural simplex-valued parameter of a combinator: mixture weights and an HMM
    initial distribution (``rows=1``, a single K-simplex) or an HMM transition matrix
    (``rows=K``, K independent simplex rows). ``alpha`` is the per-row Dirichlet concentration
    (a symmetric ``Dirichlet(1)`` for a ``free`` simplex). Inference expands it via the Gamma
    representation of the Dirichlet (one positive slot per entry, normalized per row)."""

    __slots__ = ("alpha", "rows", "name")

    def __init__(self, alpha, rows: int = 1, name: str | None = None):
        self.alpha = np.asarray(alpha, dtype=float)
        self.rows = int(rows)
        self.name = name


class _VectorSpec:
    """A vector-valued parameter of a combinator (e.g. an MVN mean): ``dim`` independent scalar
    slots of one ``support`` (``real``/``positive``/``unit``), assembled into a vector."""

    __slots__ = ("dim", "support", "name")

    def __init__(self, dim: int, support: str = "real", name: str | None = None):
        self.dim = int(dim)
        self.support = support
        self.name = name


class _OrderedSpec:
    """A strictly-increasing vector parameter (``v[0] < v[1] < ...``): one real base entry plus
    ``dim-1`` positive increments, assembled as a cumulative sum. Gives ordered means *by
    construction* (the standard mixture/HMM identifiability device) with no rejection."""

    __slots__ = ("dim", "name")

    def __init__(self, dim: int, name: str | None = None):
        self.dim = int(dim)
        self.name = name


class _Ordered:
    """The ``ordered`` token: an estimable vector parameter constrained to be increasing."""

    __slots__ = ()

    def __reduce__(self):
        return (_ordered_singleton, ())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "ordered"


ordered = _Ordered()


def _ordered_singleton():
    return ordered


class _CholeskySpec:
    """A covariance-matrix parameter (e.g. an MVN covariance): the ``dim*(dim+1)/2`` lower-
    triangular Cholesky entries (diagonal in log-space, off-diagonal on the real line). ``rebuild``
    forms ``L`` and returns ``Sigma = L Lᵀ`` — symmetric positive-definite by construction, so no
    explicit PSD constraint or Jacobian is needed (the prior, if any, lives on the entries)."""

    __slots__ = ("dim", "name")

    def __init__(self, dim: int, name: str | None = None):
        self.dim = int(dim)
        self.name = name


class Constraint:
    """A boolean relation over one or more random variables.

    Produced by comparisons on RVs — ``x > 0`` (RV vs constant), ``a < b`` (RV vs RV), or
    ``2 * a - b >= 1`` (linear/transformed expressions on either side) — and combined with
    ``&`` (and), ``|`` (or), ``~`` (not). A constraint over a single RV is consumed by
    ``rv.given(c)`` (truncation); a constraint over several RVs is consumed by
    ``constrain(c)`` (joint conditioning) or ``fit(..., constraints=c)`` (feasible region).

    ``leaves`` are the distinct leaf RVs the relation depends on; ``pred(env)`` evaluates the
    relation given ``env``, a dict mapping each leaf RV to its value(s).
    """

    __slots__ = ("leaves", "pred", "desc", "residual", "soft")

    def __init__(self, leaves, pred, desc, residual=None, soft=False):
        self.leaves = tuple(leaves)
        self.pred = pred  # env: {leaf_rv -> value(s)} -> bool mask
        self.desc = desc
        # Optional continuous violation r(env): a 1-D array that is 0 where the relation holds
        # (a hinge for inequalities, the signed gap for equalities). Enables the soft-penalty path
        # ``fit(..., penalty=w)`` so equality / convex / algebraic constraints can be honored by
        # gradient inference. ``None`` means penalty-mode is unavailable (e.g. a negated relation).
        self.residual = residual
        # ``soft``: a measure-zero relation (equality / ODE residual) that cannot be honored by
        # rejection, so ``fit`` auto-selects the soft-penalty path for it (no ``penalty=`` needed).
        self.soft = soft

    @property
    def rv(self):
        """The single RV this constraint restricts (back-compat for one-variable events)."""
        if len(self.leaves) != 1:
            raise AttributeError("constraint involves multiple RVs; use .leaves.")
        return self.leaves[0]

    def eval(self, env):
        """Evaluate the constraint predicate against an environment of RV values."""
        return self.pred(env)

    def contains(self, x):
        """Evaluate a single-variable constraint directly on that variable's value(s)."""
        if len(self.leaves) != 1:
            raise TypeError("contains(x) is only valid for a one-variable constraint; use eval(env).")
        return self.pred({self.leaves[0]: x})

    def _merge_leaves(self, other):
        seen, out = set(), []
        for lv in self.leaves + other.leaves:
            if id(lv) not in seen:
                seen.add(id(lv))
                out.append(lv)
        return tuple(out)

    def __and__(self, other):
        # AND must satisfy both, so the residual stacks both violations (all must reach 0).
        residual = _combine_residuals(self.residual, other.residual, "and")
        return Constraint(
            self._merge_leaves(other),
            lambda env: self.pred(env) & other.pred(env),
            f"({self.desc} & {other.desc})",
            residual,
            self.soft or other.soft,
        )

    def __or__(self, other):
        # OR is satisfied when either holds, so the residual is the smaller of the two magnitudes.
        residual = _combine_residuals(self.residual, other.residual, "or")
        return Constraint(
            self._merge_leaves(other),
            lambda env: self.pred(env) | other.pred(env),
            f"({self.desc} | {other.desc})",
            residual,
            self.soft or other.soft,
        )

    def __invert__(self):
        # Negation has no smooth penalty surface; only the hard (boolean) mode survives.
        return Constraint(self.leaves, lambda env: ~np.asarray(self.pred(env)), f"~{self.desc}", None)

    def __bool__(self):
        raise TypeError(
            "a Constraint has no truth value — Python chained comparisons (a < b < c) and "
            "`and`/`or` are not supported; combine with & | ~ instead, e.g. (a < b) & (b < c)."
        )

    def __repr__(self):
        return f"Constraint({self.desc})"


Event = Constraint  # back-compat alias


def _expr_leaves(rv) -> list:
    """The leaf (sample/bound) RVs an expression RV depends on, in left-to-right order."""
    if not isinstance(rv, RandomVariable):
        return []
    if rv._kind in ("apply", "pow", "select", "gather"):
        return _expr_leaves(rv._args[0])
    if rv._kind in ("sum", "prod"):
        out = _expr_leaves(rv._args[0])
        seen = {id(x) for x in out}
        for lv in _expr_leaves(rv._args[1]):
            if id(lv) not in seen:
                out.append(lv)
                seen.add(id(lv))
        return out
    return [rv]  # sample / bound / given: an atomic leaf


def _expr_has_gather(rv) -> bool:
    """True if a deterministic expression contains a data-indexed gather (theta[Field(...)])."""
    if not isinstance(rv, RandomVariable):
        return False
    if rv._kind == "gather":
        return True
    if rv._kind in ("apply", "pow", "select"):
        return _expr_has_gather(rv._args[0])
    if rv._kind in ("sum", "prod"):
        return _expr_has_gather(rv._args[0]) or _expr_has_gather(rv._args[1])
    return False


# Per-route caveats surfaced by RandomVariable.explain_fit. Kept here so the auto-selector and its explanation
# share one vocabulary about result type, diagnostics, and limitations.
_ROUTE_CAVEATS = {
    "conjugate": ["exact closed-form posterior; returns a ConjugatePosterior you can sample / mean / interval"],
    "conjugate_mixture": ["exact closed-form posterior over a mixture of conjugate priors"],
    "em": ["maximum-likelihood point estimate; no priors and no posterior uncertainty"],
    "map": [
        "MAP point estimate -- no posterior uncertainty",
        "uses analytic-gradient L-BFGS when torch is available, else a slower derivative-free optimizer",
        "for a posterior, pass how='laplace' (quick Gaussian approx) or how='mcmc'/'nuts'/'hmc'",
    ],
    "laplace": ["Gaussian posterior approximation at the MAP (inverse-Hessian covariance); local approximation"],
    "hierarchical": [
        "random-effects fit; non-Normal pairs use PQL, which is mildly biased for sparse/low-count groups"
    ],
    "lmm": ["linear mixed model by EM; exact for the Gaussian response"],
    "glmm": ["GLMM by penalized quasi-likelihood (PQL) -- mildly biased for sparse binary / low-count data"],
    "regression": ["GLM point estimate with a Laplace coefficient covariance; not a full posterior"],
    "indexed": [
        "per-observation fit; how='map' (default) gives point latents, how='mcmc' a full posterior over the vector"
    ],
    "state-space": ["bespoke Kalman/RTS + EM fitter for the composite family"],
    "mcmc": ["posterior samples via adaptive random-walk Metropolis"],
    "hmc": ["posterior samples via Hamiltonian Monte Carlo"],
    "nuts": ["posterior samples via the No-U-Turn Sampler"],
    "vi": ["variational (Gaussian) posterior approximation"],
}


def _eval_expr(rv, env):
    """Numerically evaluate an expression RV given ``env`` (leaf RV -> value)."""
    if not isinstance(rv, RandomVariable):
        return rv  # a constant
    if rv._kind == "apply":
        base, transform = rv._args
        return transform.forward(_eval_expr(base, env))
    if rv._kind == "sum":
        a, b = rv._args
        return _eval_expr(a, env) + _eval_expr(b, env)
    if rv._kind == "prod":
        a, b = rv._args
        return _eval_expr(a, env) * _eval_expr(b, env)
    if rv._kind == "pow":
        base, exponent = rv._args
        return _eval_expr(base, env) ** exponent
    if rv._kind == "select":
        base, index = rv._args
        return np.asarray(_eval_expr(base, env))[..., index]
    if rv._kind == "gather":
        base, field = rv._args
        idx = np.asarray(env[("field", field.name)])
        return np.asarray(_eval_expr(base, env))[..., idx]
    if rv not in env:
        raise KeyError(f"no value supplied for {rv!r} when evaluating a constraint.")
    return env[rv]


def _row_mask(mask) -> np.ndarray:
    """Reduce a constraint's per-sample mask to one boolean per row. A relation between scalars
    is already ``(n,)``; a whole-vector relation (``v > 0``) yields ``(n, d)`` and means *all
    entries* hold, so reduce the trailing axes with ``all``."""
    m = np.asarray(mask)
    if m.ndim > 1:
        m = m.all(axis=tuple(range(1, m.ndim)))
    return m


_CMP = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: np.isclose(a, b),
    "!=": lambda a, b: ~np.isclose(a, b),
}

# Continuous violation r(a, b) >= 0 (0 where the relation holds): a hinge for inequalities, the
# signed gap for equality. Used by the soft-penalty inference path (fit(..., penalty=w)).
_RESIDUAL = {
    ">": lambda a, b: np.maximum(0.0, b - a),
    ">=": lambda a, b: np.maximum(0.0, b - a),
    "<": lambda a, b: np.maximum(0.0, a - b),
    "<=": lambda a, b: np.maximum(0.0, a - b),
    "==": lambda a, b: a - b,
}


def _combine_residuals(ra, rb, mode):
    """Combine two constraint residual closures for ``&`` (stack) / ``|`` (min magnitude)."""
    if ra is None or rb is None:
        return None
    if mode == "and":
        return lambda env: np.concatenate([np.atleast_1d(ra(env)).ravel(), np.atleast_1d(rb(env)).ravel()])

    def or_residual(env):
        va = np.atleast_1d(ra(env)).ravel()
        vb = np.atleast_1d(rb(env)).ravel()
        mag_a = float(np.sqrt(np.sum(va * va)))
        mag_b = float(np.sqrt(np.sum(vb * vb)))
        return va if mag_a <= mag_b else vb

    return or_residual


def _make_constraint(lhs, op, rhs) -> Constraint:
    """Build a Constraint from ``lhs <op> rhs`` where each side is an RV/expression or constant."""
    leaves, seen = [], set()
    for side in (lhs, rhs):
        for lv in _expr_leaves(side):
            if id(lv) not in seen:
                seen.add(id(lv))
                leaves.append(lv)
    cmp = _CMP[op]

    def pred(env):
        return cmp(np.asarray(_eval_expr(lhs, env)), np.asarray(_eval_expr(rhs, env)))

    residual = None
    if op in _RESIDUAL:
        res_fn = _RESIDUAL[op]

        def residual(env):
            return np.asarray(res_fn(np.asarray(_eval_expr(lhs, env)), np.asarray(_eval_expr(rhs, env))))

    # equality is measure-zero -> mark soft so fit() auto-uses the penalty path (no rejection)
    return Constraint(leaves, pred, f"{_expr_desc(lhs)} {op} {_expr_desc(rhs)}", residual, soft=(op == "=="))


def eq(lhs, rhs) -> Constraint:
    """Build an equality relation ``lhs == rhs`` over RVs/expressions/constants.

    ``==`` is not overloaded on ``RandomVariable`` (RVs are used as dict keys by identity), so build
    equalities with this function or ``rv.eq(...)``. Equalities have measure zero and cannot be honored
    by rejection, so consume them with the soft-penalty inference path, e.g.
    ``model.fit(data, constraints=eq(a + b, 1.0), penalty=100.0)``.
    """
    return _make_constraint(lhs, "==", rhs)


equal = eq  # readable alias


def ne(lhs, rhs) -> Constraint:
    """Build an inequality relation ``lhs != rhs`` (boolean only; no smooth penalty surface)."""
    return _make_constraint(lhs, "!=", rhs)


class _Potential:
    """A custom additive log-factor on the joint: ``fn(*values)`` evaluated at the current values of
    ``vars`` and added to ``log p(data, theta)``. The PPL counterpart of Stan's ``target +=`` /
    NumPyro's ``factor`` -- an arbitrary log-weight the standard distribution slots can't express
    (a soft coupling, a custom log-prior, a regularizer)."""

    __slots__ = ("fn", "vars", "name")

    def __init__(self, fn, vars, name=None):
        if not callable(fn):
            raise TypeError("potential(fn, *vars): fn must be callable.")
        self.fn = fn
        self.vars = tuple(vars)
        self.name = name

    def __repr__(self) -> str:
        return f"_Potential({self.name or '<fn>'}, vars={[getattr(v, 'name', v) for v in self.vars]})"


def potential(fn, *vars, name=None) -> _Potential:
    """Add a custom log-factor ``fn(*values)`` to a model's joint log-density.

    ``vars`` are random-variable parameters of the model (named priors, or ``param(...)`` vector/matrix
    handles) -- exactly the references a :func:`constrain`/:func:`eq` constraint may use. At each
    inference evaluation they are resolved to their current values and passed to ``fn`` positionally;
    ``fn`` returns a scalar log-weight that is added to ``log p(data, theta)``. Use it for anything the
    distribution slots can't say directly: a soft coupling between two latents, a bespoke log-prior, a
    penalty/regularizer::

        a = Normal(0, 10, name="a"); b = Normal(0, 10, name="b")
        m = Normal(a, 1.0).fit(data, potentials=potential(lambda av, bv: -0.5 * (av - bv) ** 2, a, b))

    Pass one potential or a list via ``fit(..., potentials=...)``. Potentials route inference through the
    numerical target (``how`` in ``map`` / ``mcmc`` / ``hmc`` / ``nuts`` / ``ensemble``; ``auto`` picks
    ``map``); like constraints, every referenced variable must be a parameter of the fitted model.
    """
    return _Potential(fn, vars, name)


# ----------------------------------------------------- differential / shape constraints
# Constraints on the *shape* of a vector-valued RV / expression (a discretized function),
# expressed through finite differences: the first difference governs monotonicity / smoothness,
# the second difference governs curvature (convexity). Each carries a continuous residual so it
# works with the soft-penalty inference path as well as generative ``constrain(...)``.
def _diff(a, order: int = 1):
    return np.diff(np.asarray(a, dtype=float), n=order, axis=-1)


def _shape_constraint(v, test, violation, desc) -> Constraint:
    return Constraint(
        _expr_leaves(v),
        lambda env: np.all(test(_eval_expr(v, env)), axis=-1),
        desc,
        lambda env: np.asarray(violation(_eval_expr(v, env))).ravel(),
    )


def increasing(v, *, strict: bool = False) -> Constraint:
    """The entries of a vector RV/expression are non-decreasing (``strict`` -> strictly increasing)."""
    cmp = (lambda d: d > 0) if strict else (lambda d: d >= 0)
    return _shape_constraint(
        v, lambda x: cmp(_diff(x)), lambda x: np.maximum(0.0, -_diff(x)), f"increasing({_expr_desc(v)})"
    )


def decreasing(v, *, strict: bool = False) -> Constraint:
    """The entries of a vector RV/expression are non-increasing (``strict`` -> strictly decreasing)."""
    cmp = (lambda d: d < 0) if strict else (lambda d: d <= 0)
    return _shape_constraint(
        v, lambda x: cmp(_diff(x)), lambda x: np.maximum(0.0, _diff(x)), f"decreasing({_expr_desc(v)})"
    )


def monotone(v) -> Constraint:
    """The entries are monotone — non-decreasing *or* non-increasing."""
    return increasing(v) | decreasing(v)


def convex(v) -> Constraint:
    """The entries are convex: the second difference is non-negative everywhere."""
    return _shape_constraint(
        v, lambda x: _diff(x, 2) >= 0, lambda x: np.maximum(0.0, -_diff(x, 2)), f"convex({_expr_desc(v)})"
    )


def concave(v) -> Constraint:
    """The entries are concave: the second difference is non-positive everywhere."""
    return _shape_constraint(
        v, lambda x: _diff(x, 2) <= 0, lambda x: np.maximum(0.0, _diff(x, 2)), f"concave({_expr_desc(v)})"
    )


def lipschitz(v, bound: float) -> Constraint:
    """Bounded first difference: ``|v[i+1] - v[i]| <= bound`` (a discrete smoothness constraint)."""
    b = float(bound)
    return _shape_constraint(
        v,
        lambda x: np.abs(_diff(x)) <= b,
        lambda x: np.maximum(0.0, np.abs(_diff(x)) - b),
        f"lipschitz({_expr_desc(v)}, {b})",
    )


def ode_residual(v, f, dt: float = 1.0, *, tol: float = 1e-2) -> Constraint:
    """A differential-equation constraint: ``v`` (a function sampled on a uniform grid of spacing
    ``dt``) satisfies ``dv/dt = f(v)``. The signed residual ``diff(v)/dt - f(v[:-1])`` feeds the
    soft-penalty inference path — ``fit(..., constraints=ode_residual(y, f), penalty=w)`` fits a
    physics-informed curve. Like an equality it is measure-zero, so consume it with ``penalty=``
    rather than by rejection."""
    step = float(dt)

    def resid(env):
        y = np.asarray(_eval_expr(v, env), dtype=float)
        return np.diff(y, axis=-1) / step - np.asarray(f(y[..., :-1]), dtype=float)

    return Constraint(
        _expr_leaves(v),
        lambda env: np.all(np.abs(resid(env)) <= tol, axis=-1),
        f"ode_residual({_expr_desc(v)})",
        lambda env: np.asarray(resid(env)).ravel(),
        soft=True,  # an ODE residual is measure-zero -> always the penalty path
    )


def _expr_desc(rv) -> str:
    if not isinstance(rv, RandomVariable):
        return repr(rv)
    if rv._kind == "apply":
        return f"f({_expr_desc(rv._args[0])})"
    if rv._kind == "sum":
        return f"({_expr_desc(rv._args[0])} + {_expr_desc(rv._args[1])})"
    if rv._kind == "prod":
        return f"({_expr_desc(rv._args[0])} * {_expr_desc(rv._args[1])})"
    if rv._kind == "pow":
        return f"({_expr_desc(rv._args[0])} ** {rv._args[1]})"
    if rv._kind == "select":
        return f"{_expr_desc(rv._args[0])}[{rv._args[1]}]"
    if rv._kind == "gather":
        return f"{_expr_desc(rv._args[0])}[{rv._args[1].name}]"
    return rv._name or "rv"


def _convolve(da, db):
    """Closed-form distribution of da + db for independent operands, or None."""
    ta, tb = type(da).__name__, type(db).__name__
    if ta == tb == "GaussianDistribution":
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        return GaussianDistribution(da.mu + db.mu, da.sigma2 + db.sigma2)
    if ta == tb == "PoissonDistribution":
        from mixle.stats.univariate.discrete.poisson import PoissonDistribution

        return PoissonDistribution(da.lam + db.lam)
    if ta == tb == "GammaDistribution" and abs(da.theta - db.theta) < 1e-12:
        from mixle.stats.univariate.continuous.gamma import GammaDistribution

        return GammaDistribution(da.k + db.k, da.theta)  # same scale
    return None


# ------------------------------------------------------------------------- family
class Family:
    """Lowering recipe for one distribution family.

    Keeps the alias namespace and the engine objects in one place so the wrapper
    never hard-codes a distribution. ``to_dist`` maps user-facing (conventional)
    arguments to the underlying ``*Distribution`` kwargs; ``make_estimator`` builds
    the paired ``*Estimator`` for the all-``free`` case.
    """

    __slots__ = (
        "name",
        "dist_cls",
        "est_cls",
        "to_dist",
        "arity",
        "seed_at",
        "positive",
        "init_fit",
        "read",
        "support",
    )

    def __init__(
        self,
        name,
        dist_cls,
        est_cls,
        to_dist,
        arity,
        seed_at=None,
        positive=None,
        init_fit=None,
        read=None,
        support=None,
    ):
        self.name = name
        self.dist_cls = dist_cls
        self.est_cls = est_cls
        self.to_dist = to_dist
        self.arity = arity
        # seed_at(value, scale) -> dist kwargs: a concrete instance "located" at a data
        # point, used for k-means++-style initialization of latent composites.
        self.seed_at = seed_at
        # per-slot positivity (for unconstrained-space MCMC/MAP); default all real.
        self.positive = tuple(positive) if positive is not None else (False,) * arity
        # per-slot constraint/support for gradient & MCMC reparameterization:
        # 'real' (identity), 'positive' (log), or 'unit' (logit, for probabilities).
        # Defaults from `positive`; pass `support=` to mark unit-interval params.
        self.support = (
            tuple(support) if support is not None else tuple("positive" if p else "real" for p in self.positive)
        )
        # init_fit(data) -> a concrete Distribution to warm-start EM for families whose
        # MLE is sensitive to initialization (e.g. negative-binomial dispersion).
        self.init_fit = init_fit
        # read(dist) -> {conventional param name: value}: the inverse of construction, so
        # fitted params return in the *same* parameterization the user wrote (sd, not sigma2).
        self.read = read

    def make_dist(self, args: tuple[Any, ...], name: str | None):
        """Construct the concrete distribution for conventional PPL arguments."""
        kwargs = self.to_dist(*args)
        if name is not None:
            kwargs.setdefault("name", name)
        return self.dist_cls(**kwargs)

    def make_estimator(self, name: str | None, keys: str | None):
        """Construct the estimator associated with this family."""
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if keys is not None:
            kwargs["keys"] = keys
        try:
            return self.est_cls(**kwargs)
        except TypeError:
            # Estimator does not accept name/keys; fall back to the bare constructor.
            return self.est_cls()


class CompositeFamily:
    """Lowering recipe for a family whose arguments are themselves RandomVariables
    (mixtures, sequences, HMMs, ...). Kept out of :class:`Family` so the flat-family
    path stays trivial. ``dist_fn``/``est_fn`` receive the raw args plus a callback
    that lowers a child RandomVariable to a dist / estimator.
    """

    __slots__ = ("name", "dist_fn", "est_fn", "seed_fn", "read", "fit_fn")

    def __init__(self, name, dist_fn, est_fn, seed_fn=None, read=None, fit_fn=None):
        self.name = name
        self.dist_fn = dist_fn
        self.est_fn = est_fn
        # seed_fn(args, data, rng, seed_child) -> a concrete initial Distribution that
        # breaks EM symmetry (e.g. mixture components at distinct data points).
        self.seed_fn = seed_fn
        # read(dist, read_params) -> structured params in PPL vocabulary, recursing into
        # children via read_params (so the whole read surface is leak-free).
        self.read = read
        # fit_fn(rv, data, **kw) -> a fully fitted RandomVariable, for composites whose fitting is a
        # bespoke pipeline rather than EM over a single lowered distribution (state-space, PDE). When
        # set, RandomVariable.fit dispatches to it instead of the generic estimator path, so the
        # family owns its fitter and core needs no per-family branch. This is the extension point a
        # plugin (e.g. mixle-pde) uses to register a fittable composite without touching core.
        self.fit_fn = fit_fn


_FAMILIES: dict[str, Any] = {}
_DIST_TO_FAMILY: dict[type, Family] = {}  # reverse map for reading fitted params
_DIST_TO_COMPOSITE_READ: dict[type, Any] = {}  # composite dist type -> read(dist, read_params)


def register_family(
    name, dist_cls, est_cls, to_dist, arity, seed_at=None, positive=None, init_fit=None, read=None, support=None
) -> Family:
    """Register a flat PPL family and its distribution/estimator lowering rules."""
    fam = Family(
        name,
        dist_cls,
        est_cls,
        to_dist,
        arity,
        seed_at=seed_at,
        positive=positive,
        init_fit=init_fit,
        read=read,
        support=support,
    )
    _FAMILIES[name] = fam
    _DIST_TO_FAMILY[dist_cls] = fam
    return fam


def register_composite(name, dist_fn, est_fn, seed_fn=None, dist_cls=None, read=None, fit_fn=None) -> CompositeFamily:
    """Register a composite PPL family with custom lowering or fitting hooks."""
    fam = CompositeFamily(name, dist_fn, est_fn, seed_fn=seed_fn, read=read, fit_fn=fit_fn)
    _FAMILIES[name] = fam
    if dist_cls is not None and read is not None:
        _DIST_TO_COMPOSITE_READ[dist_cls] = read
    return fam


def _count_params(p) -> int:
    """Heuristic free-parameter count: numeric leaves in a params structure."""
    if isinstance(p, dict):
        return sum(_count_params(v) for v in p.values())
    if isinstance(p, (list, tuple)):
        return sum(_count_params(v) for v in p)
    arr = np.asarray(p)
    return int(arr.size) if arr.dtype.kind in "fiu" else 0


def compare(models, data, *, by: str = "aic"):
    """Compare fitted models on ``data``. Returns rows sorted best-first by ``by``
    ('aic' | 'bic' | 'loglik' | 'waic' | 'loo').

    ``'waic'`` and ``'loo'`` are the Bayesian predictive criteria (integrating over parameter
    uncertainty via the posterior draws of a Bayesian fit); ``'aic'``/``'bic'`` use the point estimate.
    Each row also reports ``elpd`` differences from the best model (``d_elpd``) for waic/loo.
    """
    rows = []
    for m in models:
        ll = m.log_likelihood(data)
        row = {"model": (m.name or type(m.dist).__name__), "loglik": ll, "aic": m.aic(data), "bic": m.bic(data)}
        if by in ("waic", "loo"):
            res = m.waic(data) if by == "waic" else m.loo(data)
            row[by] = res[by]
            row["elpd"] = res["elpd_waic" if by == "waic" else "elpd_loo"]
            row["se"] = res["se"]
            if by == "loo":
                row["khat_max"] = res["khat_max"]
        rows.append(row)
    keys = {
        "loglik": lambda r: -r["loglik"],
        "aic": lambda r: r["aic"],
        "bic": lambda r: r["bic"],
        "waic": lambda r: r["waic"],
        "loo": lambda r: r["loo"],
    }
    rows = sorted(rows, key=keys[by])
    if by in ("waic", "loo"):
        best = rows[0]["elpd"]
        for r in rows:
            r["d_elpd"] = r["elpd"] - best
    return rows


def read_params(dist):
    """Fitted parameters for any distribution in PPL (construction) vocabulary, recursing
    into composite children. Falls back to the raw distribution if unregistered."""
    fam = _DIST_TO_FAMILY.get(type(dist))
    if fam is not None and fam.read is not None:
        return fam.read(dist)
    creader = _DIST_TO_COMPOSITE_READ.get(type(dist))
    if creader is not None:
        return creader(dist, read_params)
    return dist


def seed_child(rv: RandomVariable, value: Any, scale: float, rng=None):
    """Build a concrete distribution for a child RV to break EM symmetry.

    Continuous families with a ``seed_at`` are located at the data ``value``; a Categorical
    is seeded with a random (non-degenerate) Dirichlet draw over its support. Returns None
    when the child can't be seeded (caller falls back to default init).
    """
    rng = rng or np.random.RandomState(0)
    if rv._kind == "bound":
        return rv._dist
    if rv._kind == "sample" and not isinstance(rv._family, CompositeFamily):
        fam = rv._family
        if fam.seed_at is not None:
            return fam.dist_cls(**fam.seed_at(value, scale))
        if fam.name == "Categorical":
            from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

            spec = rv._args[0]
            keys = list(spec.keys()) if isinstance(spec, dict) else list(range(len(spec)))
            w = rng.dirichlet(np.ones(len(keys)))  # random, valid (no zeros->inf)
            return CategoricalDistribution(pmap=dict(zip(keys, w)))
    return None


# ----------------------------------------------------------------- RandomVariable
class RandomVariable:
    """The single user-facing PPL type (immutable).

    Two states: ``sample`` (a symbolic draw: a family + argument expressions, some of
    which may be ``free``) and ``bound`` (wraps a concrete fitted distribution). The
    verb surface is fixed and state-independent; validity depends on
    state. Construct via the family functions in :mod:`mixle.ppl` or ``fit``.
    """

    __slots__ = ("_kind", "_family", "_args", "_name", "_keys", "_dist", "_result", "_cache", "_scope", "_reparam")

    @property
    def certificate(self):
        """The estimation certificate, when a fit attached one.

        Penalized fits downgrade the certificate because the optimum is for a surrogate objective, not
        an unpenalized likelihood.
        """
        return self._cache.get("certificate")

    def __init__(
        self,
        kind,
        *,
        family=None,
        args=(),
        name=None,
        keys=None,
        dist=None,
        result: PosteriorResult | None = None,
        scope="shared",
        reparam=None,
    ):
        # Private; use the classmethods / family functions. Treated as immutable.
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_args", tuple(args))
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_keys", keys)
        object.__setattr__(self, "_dist", dist)
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_scope", scope)  # 'shared' | 'grouped'
        object.__setattr__(self, "_reparam", reparam)  # None | 'loc_scale' (non-centered prior)

    def __setattr__(self, *a):  # enforce immutability
        raise AttributeError("RandomVariable is immutable")

    def __reduce__(self):
        # Picklable so models can cross a process boundary (parallel chains,
        # distributed fits). Families live in the module-level registry and are
        # restored by name; transient _result/_cache are dropped.
        fam_name = self._family.name if self._family is not None else None
        return (
            _rv_reconstruct,
            (self._kind, fam_name, self._args, self._name, self._keys, self._dist, self._scope, self._reparam),
        )

    # -- constructors -------------------------------------------------------
    @classmethod
    def _sample(cls, family_name, args, *, name=None, keys=None, scope="shared") -> RandomVariable:
        fam = _FAMILIES[family_name]
        return cls("sample", family=fam, args=args, name=name, keys=keys, scope=scope)

    def each(self, by: str | None = None) -> RandomVariable:
        """Mark this prior as per-group (a random effect / local latent). Used in a parameter slot:
        ``Normal(Normal(m, t).each(), s)`` is a hierarchical model.

        Two data layouts are supported:

        * **nested** -- ``each()`` with no argument: ``.fit(groups)`` where ``groups`` is a list of
          per-group observation lists (one list per group).
        * **indexed-flat** -- ``each(by="g")``: ``.fit(y, given={"g": labels})`` where ``y`` is one flat
          observation array and ``labels[i]`` is observation ``i``'s group. This is the varying-intercepts
          / 8-schools idiom; groups are taken in sorted order of the unique labels.
        """
        if self._kind != "sample":
            raise TypeError("each() applies to a distribution used as a prior.")
        rv = RandomVariable(
            "sample", family=self._family, args=self._args, name=self._name, keys=self._keys, scope="grouped"
        )
        if by is not None:
            rv._cache["group_by"] = str(by)  # read by fit() to reshape indexed-flat data into groups
        return rv

    def noncentered(self) -> RandomVariable:
        """Sample this location-scale prior in non-centered form (offset/multiplier).

        For ``mu = Normal(loc, scale)`` with a random ``scale`` (a hierarchical prior), the centered
        parameterization couples ``mu``'s range to ``scale`` and creates Neal's funnel -- a geometry
        HMC/NUTS samples badly. ``Normal(loc, scale).noncentered()`` instead samples a standard normal
        ``z`` and sets ``mu = loc + scale * z``, whose geometry is independent of ``scale``. Mathematically
        identical posterior, far better mixing (fewer divergences) when the data are weakly informative.
        Applies to ``Normal`` priors; a no-op marker on others.
        """
        if self._kind != "sample" or self._family is None or self._family.name != "Normal":
            raise TypeError("noncentered() applies to a Normal prior (a location-scale family).")
        return RandomVariable(
            "sample",
            family=self._family,
            args=self._args,
            name=self._name,
            keys=self._keys,
            scope=self._scope,
            reparam="loc_scale",
        )

    @property
    def scope(self) -> str:
        """Return the variable scope, such as scalar, grouped, or global."""
        return self._scope

    @classmethod
    def _bound(cls, dist, *, name=None, result: PosteriorResult | None = None) -> RandomVariable:
        return cls("bound", dist=dist, name=name or getattr(dist, "name", None), result=result)

    @classmethod
    def _apply(cls, base, transform) -> RandomVariable:
        # Apply node: a deterministic transform of one RV (algebra rung 1).
        return cls("apply", args=(base, transform))

    @classmethod
    def _sum(cls, a, b) -> RandomVariable:
        # Convolution node: the distribution of a + b for independent a, b.
        return cls("sum", args=(a, b))

    @classmethod
    def _prod(cls, a, b) -> RandomVariable:
        # Product expression node (a * b). Valid in constraint / solver expressions and as a
        # derived RV (sample/mean); not lowerable to a distribution (no tractable density).
        return cls("prod", args=(a, b))

    @classmethod
    def _pow(cls, base, exponent) -> RandomVariable:
        # Power expression node (base ** const). Same status as a product node.
        return cls("pow", args=(base, float(exponent)))

    @classmethod
    def _select(cls, base, index) -> RandomVariable:
        # Entry-selection node base[i]: picks component `index` of a vector-valued RV. Valid in
        # constraint / solver expressions and as a derived RV (sample/mean).
        return cls("select", args=(base, int(index)))

    @classmethod
    def _gather(cls, base, field) -> RandomVariable:
        # Data-indexed gather base[Field("g")]: picks, per observation i, entry ``g[i]`` of a latent
        # vector. Yields a per-observation value, so a model using it is fit by the per-observation
        # (indexed) target.
        return cls("gather", args=(base, field))

    def __getitem__(self, index) -> RandomVariable:
        if isinstance(index, Field):  # data-indexed latent: theta[Field("g")] -> per-observation gather
            return RandomVariable._gather(self, index)
        if not isinstance(index, int):
            raise TypeError("RandomVariable indexing takes an int entry (v[0]) or a Field (theta[Field('g')]).")
        return RandomVariable._select(self, index)

    # -- algebra (deterministic transforms + convolution) -------------------
    def _affine(self, loc, scale) -> RandomVariable:
        from mixle.stats.combinator.transform import AffineTransform

        return RandomVariable._apply(self, AffineTransform(loc=float(loc), scale=float(scale)))

    def __mul__(self, c):
        if isinstance(c, Field):  # coef * covariate -> regression term
            return _LinearPredictor([(self, c)])
        if isinstance(c, RandomVariable):  # product expression (constraints/solver; not a dist)
            return RandomVariable._prod(self, c)
        return self._affine(0.0, c)

    __rmul__ = __mul__

    def __pow__(self, p):
        if isinstance(p, RandomVariable):
            raise NotImplementedError("RV ** RV is not supported; the exponent must be constant.")
        return RandomVariable._pow(self, p)

    def __add__(self, c):
        if isinstance(c, _LinearPredictor):  # RV is an intercept
            return c.__add__(self)
        if isinstance(c, Field):
            return _LinearPredictor([(1.0, c)], self)
        if isinstance(c, Group):  # RV intercept + random group effects
            return _LinearPredictor([], self, [c._key()])
        if isinstance(c, RandomVariable):  # convolution of independent RVs
            return RandomVariable._sum(self, c)
        return self._affine(c, 1.0)

    __radd__ = __add__

    def __sub__(self, c):
        if isinstance(c, RandomVariable):
            return RandomVariable._sum(self, c._affine(0.0, -1.0))  # a + (-b)
        return self._affine(-float(c), 1.0)

    def __rsub__(self, c):
        return self._affine(c, -1.0) if not isinstance(c, RandomVariable) else c.__sub__(self)

    def __truediv__(self, c):
        if isinstance(c, RandomVariable):  # ratio expression: a / b = a * b**-1
            return RandomVariable._prod(self, RandomVariable._pow(c, -1.0))
        return self._affine(0.0, 1.0 / float(c))

    def __neg__(self):
        return self._affine(0.0, -1.0)

    def exp(self) -> RandomVariable:
        """Return the deterministic exponential transform of this random variable."""
        from mixle.stats.combinator.transform import ExpTransform

        return RandomVariable._apply(self, ExpTransform())

    def log(self) -> RandomVariable:
        """Return the deterministic logarithm transform of this random variable."""
        from mixle.stats.combinator.transform import LogTransform

        return RandomVariable._apply(self, LogTransform())

    # -- relations: comparisons build Constraints (RV vs constant / RV / linear expr) ----
    def __gt__(self, other):
        return _make_constraint(self, ">", other)

    def __ge__(self, other):
        return _make_constraint(self, ">=", other)

    def __lt__(self, other):
        return _make_constraint(self, "<", other)

    def __le__(self, other):
        return _make_constraint(self, "<=", other)

    # ``==`` / ``!=`` stay identity-based (RVs are dict keys), so equalities use explicit methods.
    def eq(self, other):
        """Build an equality constraint against a value or another expression."""
        return _make_constraint(self, "==", other)

    def ne(self, other):
        """Build an inequality constraint against a value or another expression."""
        return _make_constraint(self, "!=", other)

    def given(self, constraint) -> RandomVariable:
        """Condition this RV on a constraint over *itself* (e.g. ``x.given(x > 0)`` ->
        truncation). The result samples by rejection and scores with the renormalized
        density. For relations among *several* RVs (``a < b``) use ``constrain(...)``."""
        if not isinstance(constraint, Constraint):
            raise TypeError("given() expects a Constraint from a comparison, e.g. x > 0.")
        extra = [lv for lv in constraint.leaves if lv is not self]
        if extra:
            raise ValueError(
                "given() conditions an RV on a relation over itself only; this constraint also "
                "involves other RVs — use constrain(constraint) for a joint conditioning."
            )
        return RandomVariable("given", args=(self, constraint))

    # -- introspection ------------------------------------------------------
    @property
    def is_bound(self) -> bool:
        """Whether this variable has already been lowered to a fitted concrete object."""
        return self._kind == "bound"

    @property
    def has_free(self) -> bool:
        """Whether this sample expression contains one or more free parameters."""
        return self._kind == "sample" and any(_is_free(a) for a in self._args)

    @property
    def name(self) -> str | None:
        """Return the optional user-visible variable name."""
        return self._name

    @property
    def columns(self) -> list:
        """For a ``constrain(...)`` joint RV: the variable names, in sample-column order. A
        vector-valued variable expands to one name per entry (``v[0]``, ``v[1]``, ...)."""
        if self._kind != "joint":
            raise TypeError("columns is only defined for a constrain(...) RV.")
        leaves = self._args[0]
        names = []
        for i, lv in enumerate(leaves):
            base = lv._name or f"rv{i}"
            w = int(np.asarray(lv.sample(1, seed=0)).reshape(1, -1).shape[1])
            names.extend([base] if w == 1 else [f"{base}[{j}]" for j in range(w)])
        return names

    @property
    def dist(self):
        """The lowered concrete distribution — the full original mixle API (escape hatch)."""
        return lower(self, target="dist")

    @property
    def components(self):
        """Fitted sub-models of a composite (mixture components, HMM state emissions,
        sequence element) as RandomVariables — query each with the same verbs
        (``.params``, ``.sample``, ``.log_prob``). Raises for non-composite models.
        """
        d = lower(self, target="dist")
        if hasattr(d, "components"):
            children = list(d.components)
        elif hasattr(d, "topics"):
            children = list(d.topics)
        elif hasattr(d, "dist") and not hasattr(d, "mu"):  # SequenceDistribution.dist
            children = [d.dist]
        else:
            raise TypeError(f"{type(d).__name__} has no sub-models to expose as components.")
        return [RandomVariable._bound(c) for c in children]

    @property
    def params(self):
        """Fitted parameters in the *same* parameterization used to construct the model
        (e.g. ``{'mean': 5.0, 'sd': 2.0}`` for Normal — not the internal ``sigma2``).
        Falls back to ``.dist`` for families without a registered reader.
        """
        d = lower(self, target="dist")
        if d is None and self._result is not None and hasattr(self._result, "coefficients"):
            return self._result.coefficients  # regression: report coefficients
        return read_params(d)

    @property
    def result(self) -> PosteriorResult | None:
        """Inference metadata (EM history / MCMC chain) when present; else None."""
        return self._result

    # -- query verbs (valid once concrete) ----------------------------------
    def sample(self, n: int | None = None, seed: int | None = None, size: int | None = None):
        """Draw samples from the represented distribution or derived expression.

        ``n`` and ``size`` are aliases (``size`` matches the ``stats``-layer samplers); pass at
        most one. ``None`` returns a single draw.
        """
        n = coalesce_alias("n", n, "size", size, required=False, default=None)
        if self._kind == "joint":  # joint rejection sampling under a relation
            leaves, constraint = self._args
            rng = np.random.RandomState(seed)
            k = n if n is not None else 1
            kept = []
            have = 0
            while have < k:
                batch = max(k * 2, 1024)
                cols = {lv: np.asarray(lv.sample(batch, seed=int(rng.randint(1, 2**31))), dtype=float) for lv in leaves}
                mask = _row_mask(constraint.eval(cols))  # per-row; vector relations reduce over entries
                block = np.concatenate([cols[lv][mask].reshape(int(mask.sum()), -1) for lv in leaves], axis=1)
                kept.append(block)
                have += len(block)
            out = np.concatenate(kept, axis=0)[:k]
            return out if n is not None else out[0]
        if self._kind == "select":  # entry of a vector RV: sample base, take the component
            base, index = self._args
            k = n if n is not None else 1
            xs = np.asarray(base.sample(k, seed=seed), dtype=float)[..., index]
            return xs if n is not None else float(xs[0])
        if self._kind == "sum":  # convolution: sample operands and add
            rng = np.random.RandomState(seed)
            a, b = self._args
            k = n if n is not None else 1
            xs = np.asarray(a.sample(k, seed=int(rng.randint(1, 2**31))))
            ys = np.asarray(b.sample(k, seed=int(rng.randint(1, 2**31))))
            out = xs + ys
            return out if n is not None else float(out[0])
        if self._kind == "prod":  # product of independent RVs: sample operands and multiply
            rng = np.random.RandomState(seed)
            a, b = self._args
            k = n if n is not None else 1
            xs = np.asarray(a.sample(k, seed=int(rng.randint(1, 2**31))))
            ys = np.asarray(b.sample(k, seed=int(rng.randint(1, 2**31))))
            out = xs * ys
            return out if n is not None else float(out[0])
        if self._kind == "pow":  # power of an RV by a constant exponent
            base, exponent = self._args
            k = n if n is not None else 1
            xs = np.asarray(base.sample(k, seed=seed)) ** exponent
            return xs if n is not None else float(xs[0])
        if self._kind == "given":  # rejection sampling from the region
            base, event = self._args
            rng = np.random.RandomState(seed)
            k = n if n is not None else 1
            kept = []
            while sum(len(c) for c in kept) < k:
                draw = np.asarray(base.sample(max(k * 2, 1024), seed=int(rng.randint(1, 2**31))))
                kept.append(draw[event.contains(draw)])
            out = np.concatenate(kept)[:k]
            return out if n is not None else float(out[0])
        return lower(self, target="dist").sampler(seed=seed).sample(size=n)

    def _kde(self):
        kde = self._cache.get("_kde")
        if kde is None:
            from scipy.stats import gaussian_kde

            s = np.asarray(self.sample(40000, seed=12345))
            kde = gaussian_kde(s)
            self._cache["_kde"] = kde
        return kde

    def log_prob(self, x):
        """Evaluate the log probability or log density at ``x``."""
        if self._kind == "joint":  # joint density of independent leaves / Z
            leaves, constraint = self._args
            widths = [int(np.asarray(lv.sample(1, seed=0)).reshape(1, -1).shape[1]) for lv in leaves]
            if any(w > 1 for w in widths):
                raise NotImplementedError(
                    "joint log_prob over vector-valued variables is not supported yet; use .sample / .mean / .prob."
                )
            xa = np.atleast_2d(np.asarray(x, dtype=float))
            if xa.shape[1] != len(leaves):
                raise ValueError(f"expected {len(leaves)} columns, got shape {xa.shape}.")
            logZ = math.log(self.prob())
            env = {lv: xa[:, j] for j, lv in enumerate(leaves)}
            base_lp = sum(np.atleast_1d(lv.log_prob(xa[:, j])) for j, lv in enumerate(leaves))
            out = np.where(_row_mask(constraint.eval(env)), base_lp - logZ, -np.inf)
            return float(out[0]) if np.ndim(x) == 1 else out
        if self._kind == "sum":  # exact convolution if closed-form, else KDE
            a, b = self._args
            cd = None
            try:
                cd = _convolve(lower(a, target="dist"), lower(b, target="dist"))
            except Exception:  # noqa: BLE001
                cd = None
            if cd is not None:
                return RandomVariable._bound(cd).log_prob(x)
            xv = np.atleast_1d(np.asarray(x, dtype=float))
            lp = np.log(np.clip(self._kde()(xv), 1e-300, None))
            return float(lp[0]) if np.isscalar(x) else lp
        if self._kind == "given":
            base, event = self._args
            logZ = math.log(self.prob_of_event())
            xv = np.atleast_1d(np.asarray(x, dtype=float))
            base_lp = np.atleast_1d(base.log_prob(xv))
            out = np.where(event.contains(xv), base_lp - logZ, -np.inf)
            return float(out[0]) if np.isscalar(x) else out
        d = lower(self, target="dist")
        if np.isscalar(x):
            return float(d.log_density(x))
        data = list(x)
        enc = d.dist_to_encoder().seq_encode(data)
        return np.asarray(d.seq_log_density(enc))

    def log_density(self, x):
        """Alias of :meth:`log_prob` -- the ``mixle.stats`` density verb, so a random variable
        answers the same call a fitted distribution does."""
        return self.log_prob(x)

    def log_likelihood(self, data) -> float:
        """Total log-likelihood of ``data`` under the fitted model (sum of log_prob)."""
        return float(np.sum(self.log_prob(list(data))))

    def aic(self, data, k: int | None = None) -> float:
        """Akaike information criterion (lower is better). ``k`` defaults to a heuristic
        parameter count from ``.params``."""
        k = k if k is not None else _count_params(self.params)
        return 2.0 * k - 2.0 * self.log_likelihood(data)

    def bic(self, data, k: int | None = None) -> float:
        """Bayesian information criterion (lower is better)."""
        k = k if k is not None else _count_params(self.params)
        n = len(list(data))
        return k * math.log(n) - 2.0 * self.log_likelihood(data)

    def pointwise_log_likelihood(self, data) -> np.ndarray:
        """Return the ``(n_draws, n_obs)`` log-likelihood matrix used by WAIC / PSIS-LOO.

        For a Bayesian fit (``how='mcmc'|'hmc'|'ensemble'|'vi'``) each row is the log-likelihood of the
        data under one posterior draw; for a point-estimate fit it is a single row.
        """
        r = self._result
        if r is not None and hasattr(r, "pointwise_log_likelihood") and getattr(r, "build", None) is not None:
            return r.pointwise_log_likelihood(data)
        return np.asarray(self.log_prob(list(data)), dtype=float)[None, :]

    def waic(self, data) -> dict:
        """Widely Applicable Information Criterion from the posterior (lower ``waic`` is better).

        Returns ``{elpd_waic, p_waic, waic, se, n_draws, pointwise}``. Estimates out-of-sample
        predictive accuracy by integrating over parameter uncertainty -- the Bayesian analogue of
        ``aic``/``bic`` -- and falls back to a point estimate for non-Bayesian fits.
        """
        from mixle.ppl import diagnostics as _diag

        return _diag.waic(self.pointwise_log_likelihood(data))

    def loo(self, data) -> dict:
        """Pareto-Smoothed Importance-Sampling Leave-One-Out cross-validation (lower ``loo`` better).

        Returns ``{elpd_loo, p_loo, loo, se, khat_max, n_draws, pointwise}``. ``khat_max`` above ~0.7
        signals an unreliable estimate (refit with more posterior draws or prefer ``waic``).
        """
        from mixle.ppl import diagnostics as _diag

        return _diag.psis_loo(self.pointwise_log_likelihood(data))

    def summary(self):
        """Posterior summary of a Bayesian fit, or the fitted params for a point estimate.

        For ``how='mcmc'|'hmc'|'ensemble'|'vi'`` returns a per-parameter dict of
        ``{mean, std, q2.5, q97.5}`` (the 95% credible interval) plus ``_acceptance_rate`` and, for
        multi-chain runs, ``_rhat`` / ``_ess`` / ``_n_chains``. For ``map``/``em`` it returns ``.params``.
        """
        r = self._result
        if r is not None and supports(r, Summarizable):
            return r.summary()
        return self.params

    def mean(self, samples: int = 20000, seed: int = 0):
        """Expected value of the random variable (Monte-Carlo; works for any RV —
        concrete, transformed, convolved, or conditioned). For a joint ``constrain(...)``
        RV this is the per-variable mean vector."""
        s = np.asarray(self.sample(samples, seed=seed), dtype=float)
        return s.mean(axis=0) if self._kind == "joint" else float(np.mean(s))

    def var(self, samples: int = 20000, seed: int = 0):
        """Variance of the random variable (Monte-Carlo); per-variable for a joint RV."""
        s = np.asarray(self.sample(samples, seed=seed), dtype=float)
        return s.var(axis=0) if self._kind == "joint" else float(np.var(s))

    def prob(self, samples: int = 40000, seed: int = 999):
        """Probability that the relation holds (Monte-Carlo), for a ``constrain(...)`` RV."""
        if self._kind != "joint":
            raise TypeError("prob() is only defined for a constrain(...) RV.")
        p = self._cache.get("_pjoint")
        if p is None:
            leaves, constraint = self._args
            rng = np.random.RandomState(seed)
            cols = {lv: np.asarray(lv.sample(samples, seed=int(rng.randint(1, 2**31))), dtype=float) for lv in leaves}
            p = max(float(np.mean(_row_mask(constraint.eval(cols)))), 1e-9)
            self._cache["_pjoint"] = p
        return p

    def prob_of_event(self):
        """P(event) under the base distribution (Monte-Carlo), for a conditioned RV."""
        if self._kind != "given":
            raise TypeError("prob_of_event() is only defined for a .given(...) RV.")
        p = self._cache.get("_pevent")
        if p is None:
            base, event = self._args
            s = np.asarray(base.sample(40000, seed=999))
            p = max(float(np.mean(event.contains(s))), 1e-6)
            self._cache["_pevent"] = p
        return p

    def predict(self, n: int = 1, rng=None):
        """Posterior-predictive draws. For a Bayesian fit (conjugate/mcmc/hmc) this
        integrates over parameter uncertainty (draw params from the posterior, then
        data); for a point fit (EM/MAP) it is the plug-in predictive (sample from the
        fitted distribution).
        """
        import numpy as _np

        rng = rng or _np.random.RandomState()
        r = self._result
        pred = getattr(r, "predictive", None) if r is not None else None
        if pred is not None:
            return pred(n, rng)
        return self.sample(n)

    def posterior(self, x):
        """Posterior over a latent or a parameter.

        - ``posterior(data)`` -> latent-state posterior (the E-step; e.g. mixture
          responsibilities), routed to the lowered distribution's ``seq_posterior``.
        - ``posterior(handle | name | index)`` -> parameter posterior draws, when this
          RV was fit with ``how='mcmc'`` (read from ``.result``).
        """
        # Parameter posterior: a handle/name/index against an MCMC result.
        if (
            isinstance(x, (RandomVariable, str, int))
            and self._result is not None
            and supports(self._result, Sampleable)
        ):
            return self._result.samples(x)
        d = lower(self, target="dist")
        if not (hasattr(d, "seq_posterior") or hasattr(d, "posterior")):
            raise NotImplementedError(f"{type(d).__name__} exposes no posterior (no latent to infer).")
        if np.isscalar(x):
            return np.asarray(d.posterior(x))
        enc = d.dist_to_encoder().seq_encode(list(x))
        if hasattr(d, "seq_posterior"):
            return np.asarray(d.seq_posterior(enc))
        return np.asarray([d.posterior(xi) for xi in x])

    # -- resolve ------------------------------------------------------------
    def _has_priors(self) -> bool:
        # A prior is an RV in a *flat* family slot; composite children are sub-models.
        return (
            self._kind == "sample"
            and not isinstance(self._family, CompositeFamily)
            and any(isinstance(a, RandomVariable) for a in self._args)
        )

    def _has_struct_param(self) -> bool:
        # A structural vector/matrix parameter (a spec or a param(...) handle) anywhere in the tree
        # -> the model needs inference (map/mcmc/...), not the EM estimator that ignores it.
        if self._kind != "sample":
            return False
        stack = list(self._args)
        while stack:
            a = stack.pop()
            if isinstance(a, (_SimplexSpec, _VectorSpec, _CholeskySpec, _OrderedSpec)):
                return True
            if isinstance(a, RandomVariable):
                if a._kind == "param":
                    return True
                if a._kind == "sample":
                    stack.extend(a._args)
            elif isinstance(a, (list, tuple)):
                stack.extend(a)
        return False

    def _resolve_auto(self, *, has_constraints, has_potentials, grouped, partial_free, struct_param):
        """Resolve ``how='auto'`` to a concrete route + a one-line reason for the *flat* decision tree.

        Single source of the auto decision (``fit`` and :meth:`explain_fit` both call it, so the
        explanation can never drift from what actually runs). Does NOT cover the early structural
        short-circuits (gather / regression / state-space) -- those are handled by their callers.
        """
        if grouped:
            return "hierarchical", "a .each() group prior -> random-effects (hierarchical) fit"
        if has_constraints or has_potentials:
            return "map", "constraints/potentials need the numerical joint -> MAP (a point estimate)"
        if self._has_priors():
            from mixle.ppl import inference as _inf

            if _inf.conjugate_spec(self) is not None:
                return "conjugate", "a registered conjugate prior -> exact closed-form posterior"
            if _inf.conjugate_mixture_spec(self) is not None:
                return "conjugate_mixture", "a mixture of conjugate priors -> exact closed-form posterior"
            if _inf.stats_conjugate_supported(self):
                return "conjugate", "a closed-form conjugate exponential family -> exact posterior"
            return "map", "priors present but no registered closed form -> MAP (a point estimate)"
        if partial_free or struct_param:
            return "map", "a structural vector/matrix parameter or a fixed+free mix -> MAP"
        return "em", "all-free parameters, no priors -> maximum-likelihood EM"

    def _resolve_posterior_ladder(self):
        """Lowest-cost route that returns a *posterior* (uncertainty), not a point estimate -- the
        ``how='posterior'`` escalation ladder: conjugate (exact) -> Laplace (Gaussian at the MAP) -> MCMC.

        Unlike ``how='auto'`` (which stops at MAP for a non-conjugate prior and returns a point estimate),
        this always climbs to the lowest-cost route that yields posterior uncertainty, and reports which route.
        """
        from mixle.ppl import inference as _inf

        closed_form = (
            _inf.conjugate_spec(self) is not None
            or _inf.conjugate_mixture_spec(self) is not None
            or _inf.stats_conjugate_supported(self)
            or _inf._is_all_free_normal(self)
        )
        if closed_form:
            return "conjugate", "exact closed-form (conjugate) posterior -- the lowest-cost posterior route"
        flat = (
            self._kind == "sample"
            and not isinstance(self._family, CompositeFamily)
            and not any(isinstance(a, _LinearPredictor) for a in self._args)
        )
        if flat:
            return "laplace", "no closed form -> Laplace (Gaussian posterior at the MAP) -- the next rung up"
        return "mcmc", "structured/composite model -> MCMC for the posterior -- the general rung"

    def explain_fit(self, *, how="auto", constraints=None, potentials=None, **_) -> dict:
        """Report which inference route ``.fit(how=...)`` took (or would take, before fitting).

        Returns ``{'route', 'reason', 'caveats'}``. This is the inspection
        surface for Mixle's automatic cross-family inference selection:
        ``rv.explain_fit()`` answers how the expression will be fit, what result
        type it returns, and which diagnostics or limitations apply. The route
        mirrors :meth:`fit` exactly by sharing :meth:`_resolve_auto` for the flat
        tree and re-checking the same structural short-circuits.

        Called on a **bound** RV (the result of ``.fit(...)``), this reports what that fit actually
        did -- ``fit()`` stashes its own answer to this question, computed while the pre-fit expression
        still carried its priors, since a bound RV's ``_args`` is always empty and cannot be re-derived
        from. Raises if that record is unavailable (e.g. the model was reloaded from a saved artifact,
        which does not round-trip it) -- call ``explain_fit()`` on the pre-fit expression instead.
        """
        if self._kind == "bound":
            cached = self._cache.get("_fit_explanation")
            if cached is not None:
                return dict(cached)
            raise RuntimeError(
                "explain_fit() has no record of how this bound model was fit (e.g. it was reloaded "
                "from a saved artifact, or built directly rather than through .fit()). Call "
                "explain_fit() on the pre-fit expression instead, or re-fit for a fresh explanation."
            )
        if how == "posterior":
            route, reason = self._resolve_posterior_ladder()
        elif how != "auto":
            route, reason = how, f"explicit how={how!r}"
        elif self._kind == "sample" and any(_expr_has_gather(a) for a in self._args):
            route, reason = "indexed", "a data-indexed latent theta[Field(...)] -> per-observation MAP"
        elif self._kind == "sample" and any(isinstance(a, _LinearPredictor) for a in self._args):
            lp = next(a for a in self._args if isinstance(a, _LinearPredictor))
            if getattr(lp, "groups", None) and self._family.name != "Normal":
                route, reason = (
                    "glmm",
                    "a Group random effect + non-Normal response -> GLMM by penalized quasi-likelihood",
                )
            elif getattr(lp, "groups", None):
                route, reason = "lmm", "a Group random effect (Normal response) -> linear mixed model (EM)"
            else:
                route, reason = "regression", "a linear predictor over covariates -> GLM/regression"
        elif self._kind == "sample" and isinstance(self._family, CompositeFamily) and self._family.fit_fn is not None:
            route, reason = "state-space", "a composite family with a bespoke fitter (Kalman/RTS+EM, PDE)"
        else:
            grouped = self._kind == "sample" and any(
                isinstance(a, RandomVariable) and a._scope == "grouped" for a in self._args
            )
            flat = self._kind == "sample" and not isinstance(self._family, CompositeFamily)
            partial_free = (
                flat
                and not self._has_priors()
                and any(_is_free(a) for a in self._args)
                and not all(_is_free(a) for a in self._args)
            )
            route, reason = self._resolve_auto(
                has_constraints=constraints is not None,
                has_potentials=potentials is not None,
                grouped=grouped,
                partial_free=partial_free,
                struct_param=self._has_struct_param(),
            )
        caveats = list(_ROUTE_CAVEATS.get(route, []))
        # discoverability: an all-free Normal can get an exact Bayesian (Normal-Inverse-Gamma) posterior
        if (
            route == "em"
            and self._kind == "sample"
            and getattr(self._family, "name", None) == "Normal"
            and all(_is_free(a) for a in self._args)
        ):
            caveats.append(
                "for a Bayesian posterior over mean+variance, fit(how='conjugate') is exact (Normal-Inverse-Gamma)"
            )
        return {"route": route, "reason": reason, "caveats": caveats}

    def fit(
        self,
        data: Sequence[Any],
        *,
        how: str = "auto",
        max_its: int = 100,
        delta: float = 1e-8,
        backend: str = "local",
        num_workers: int | None = None,
        engine: Any = None,
        precision: Any = None,
        print_iter: int = 0,
        missing: str = "error",
        **kw,
    ) -> RandomVariable:
        """Estimate / infer parameters from ``data`` and return a bound RV.

        ``how``: ``'em'`` (EM/MLE, default for plain ``free`` models), ``'map'`` (maximize
        the joint with priors), ``'mcmc'`` (posterior samples over parameters with priors),
        ``'auto'`` picks ``map`` when the model has priors else ``em``. EM threads mixle's
        parallel/distributed backends (``backend='mp'|'mpi'|'dask'``).

        ``missing``: ``'error'`` (default) rejects non-finite entries; ``'marginalize'`` integrates a
        missing entry (``NaN`` in the data) out of the likelihood instead of imputing it -- each leaf is
        fit from its present rows only, so you get a well-defined mode/posterior over the present data (no
        fabricated values). Supported on the EM path (the default for ``free`` models, i.e. the posterior
        mode under flat priors); for ``how='map'/'mcmc'`` with missing data build the model with
        ``mixle.stats.marginalized()`` leaves directly.
        """
        if missing not in ("error", "marginalize"):
            raise ValueError(f"missing={missing!r}; choose 'error' or 'marginalize'.")
        # ``auto`` and ``em`` are resolved/handled by ``fit`` itself; every other ``how`` is a pure
        # dispatch into the fitter registry.
        valid_how = {"auto", "posterior", "em", *_FITTERS}
        if how not in valid_how:
            raise ValueError(f"unknown how={how!r}; choose from {sorted(valid_how)}.")
        if hasattr(data, "__len__") and len(data) == 0:
            raise ValueError("fit() received empty data.")

        # ``self`` (the pre-fit expression) still has its priors/args intact here, no matter which
        # branch below returns -- unlike the bound RV that comes back, whose _args is always empty. Stash
        # explain_fit()'s answer for the *originally requested* how onto the result so a bound RV's own
        # .explain_fit() reports what actually happened, instead of re-deriving from a structure that no
        # longer carries it (see the "bound" branch in explain_fit()).
        _original_how = how

        def _stash_explanation(rv):
            try:
                rv._cache["_fit_explanation"] = self.explain_fit(
                    how=_original_how, constraints=kw.get("constraints"), potentials=kw.get("potentials")
                )
            except Exception:  # noqa: BLE001 - best-effort; must never block a fit
                pass
            return rv

        if self._kind == "sample" and any(_expr_has_gather(a) for a in self._args):
            # A data-indexed latent (theta[Field("g")]) makes the parameter per-observation -> the
            # per-observation (indexed) target.
            from mixle.ppl import inference as _inf

            return _stash_explanation(_inf.indexed_fit(self, data, how=how, **kw))

        # regression / GLM: a linear predictor (covariates) in a parameter slot
        if self._kind == "sample" and any(isinstance(a, _LinearPredictor) for a in self._args):
            from mixle.ppl import regression as _reg

            return _stash_explanation(_reg.regression_fit(self, data, **kw))

        # neural conditional: a Net/Conv (nonlinear predictor) in a parameter slot -> a neural-headed leaf
        if self._kind == "sample" and any(isinstance(a, _NeuralPredictor) for a in self._args):
            from mixle.ppl import neural as _neu

            return _neu.neural_fit(self, data, **kw)

        # Composite families with a bespoke fitter (state-space Kalman/RTS+EM, PDE-constrained fields)
        # own their fit through the registered fit_fn hook -- no per-family branch in core. State-space
        # is registered in mixle.ppl.statespace; PDEStateSpace by the mixle-pde plugin.
        if self._kind == "sample" and isinstance(self._family, CompositeFamily) and self._family.fit_fn is not None:
            return _stash_explanation(self._family.fit_fn(self, data, **kw))

        # Indexed-flat hierarchical: Normal(Normal(m, t).each(by="g"), s).fit(y, given={"g": labels}).
        # Reshape the flat observation array into per-group lists (sorted unique labels) so the existing
        # nested grouped path handles it -- the model is identical, only the data layout differs.
        if self._kind == "sample":
            _gby = next(
                (
                    a._cache.get("group_by")
                    for a in self._args
                    if isinstance(a, RandomVariable) and a._scope == "grouped"
                ),
                None,
            )
            if _gby is not None:
                given = kw.pop("given", None)
                if not given or _gby not in given:
                    raise ValueError(f"each(by={_gby!r}) needs the group index: fit(..., given={{{_gby!r}: labels}}).")
                labels = np.asarray(given[_gby])
                if len(labels) != len(data):
                    raise ValueError(f"given[{_gby!r}] has length {len(labels)} but data has length {len(data)}.")
                if any(k != _gby for k in given):
                    raise NotImplementedError("indexed-flat hierarchical with extra covariates is not supported yet.")
                yarr = np.asarray(data, dtype=float)
                data = [yarr[labels == g].tolist() for g in sorted(set(labels.tolist()))]

        grouped = self._kind == "sample" and any(
            isinstance(a, RandomVariable) and a._scope == "grouped" for a in self._args
        )
        # partial-free: a flat model with some `free` slots and some fixed constants (no priors).
        # The all-free EM estimator can't hold params fixed, so fit only the free slots by
        # maximum likelihood (MAP with no prior term), with the fixed args held constant.
        flat = self._kind == "sample" and not isinstance(self._family, CompositeFamily)
        partial_free = (
            flat
            and not self._has_priors()
            and any(_is_free(a) for a in self._args)
            and not all(_is_free(a) for a in self._args)
        )
        has_constraints = kw.get("constraints") is not None
        has_potentials = kw.get("potentials") is not None
        struct_param = self._has_struct_param()
        if has_potentials and how in ("em", "conjugate", "conjugate_mixture", "vi", "vmp"):
            raise ValueError(
                f"how={how!r} cannot apply a custom potential; use 'map', 'mcmc', 'hmc', 'nuts', or "
                "'ensemble' (or how='auto')."
            )
        if has_constraints and how in ("em", "conjugate", "conjugate_mixture", "vi", "vmp"):
            raise ValueError(
                f"how={how!r} cannot honor inequality constraints; use 'map', 'mcmc', 'hmc', "
                "or 'ensemble' (or how='auto')."
            )
        if how == "posterior":
            # the escalation ladder: lowest-cost route that yields posterior uncertainty (conjugate ->
            # Laplace -> MCMC). Unlike 'auto', never returns a bare point estimate.
            how, _ = self._resolve_posterior_ladder()
        if how == "auto":
            how, _auto_reason = self._resolve_auto(
                has_constraints=has_constraints,
                has_potentials=has_potentials,
                grouped=grouped,
                partial_free=partial_free,
                struct_param=struct_param,
            )
            if how == "map" and "no registered closed form" in _auto_reason:
                # A prior is present but there is no closed-form posterior for this model, so auto returns a point
                # estimate. Warn so callers do not mistake it for posterior uncertainty.
                import warnings as _warnings

                _warnings.warn(
                    "how='auto' selected MAP -- a point estimate, not a posterior: a prior is present but "
                    "this model has no registered closed-form (conjugate) posterior. For posterior "
                    "uncertainty pass how='laplace' (local Gaussian at the MAP), 'vi', or 'mcmc'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif how == "em" and (partial_free or struct_param):
            how = "map"  # EM can't hold params fixed / infer a structural vector param
        # Pure ``how`` -> fitter dispatch (everything except the EM/MLE fall-through below). The
        # registry replaces the old ``if how == ...`` ladder; the ``vmp`` Mixture special case lives
        # inside its registered fitter (a closure over the RV's family/args).
        if missing == "marginalize" and how != "em":
            # the autograd-target fitters marginalize NaN observations (flat models); thread the flag in.
            if how in {"map", "mcmc", "hmc", "nuts", "vi", "ensemble", "sample"}:
                kw["missing"] = missing
            else:
                raise NotImplementedError(
                    f"missing='marginalize' is not wired for how={how!r} (closed-form/grouped path); use "
                    "how='em'/'map'/'mcmc'/'hmc'/'nuts'/'vi'/'ensemble', or build the model with "
                    "mixle.stats.marginalized() leaves."
                )
        fitter = _FITTERS.get(how)
        if fitter is not None:
            result = fitter(self, data, **kw)
            if has_constraints or has_potentials:
                # A penalized objective (soft constraints / residual factors / potentials) means the
                # optimum is for the surrogate, not the likelihood, so downgrade the certificate.
                try:
                    from mixle.inference.planning import certify as _certify

                    target = getattr(result, "_dist", None) or getattr(result, "dist", None) or result
                    why = "soft constraints" if has_constraints else "custom potential"
                    result._cache["certificate"] = _certify(target, penalized=why)
                except Exception:  # noqa: BLE001 - certification must never break a fit
                    pass
            return _stash_explanation(result)
        # EM / MLE path
        est = lower(self, target="estimator")
        if missing == "marginalize":
            import numpy as _np

            from mixle.stats.missing import marginalize_estimator_leaves

            est = marginalize_estimator_leaves(est, missing_value=_np.nan)
        # Warm-start finicky flat-family MLEs (e.g. negative-binomial) from a moment match.
        if (
            "prev_estimate" not in kw
            and missing != "marginalize"  # data-driven warm-start would choke on NaN; let EM seed plainly
            and self._kind == "sample"
            and not isinstance(self._family, CompositeFamily)
            and getattr(self._family, "init_fit", None) is not None
        ):
            seed = self._family.init_fit(data)
            if seed is not None:
                kw["prev_estimate"] = seed
        # Auto-seed latent composites (mixtures, ...) at distinct data points so EM
        # avoids the symmetric global-mean fixed point.
        if (
            "prev_estimate" not in kw
            and missing != "marginalize"
            and self._kind == "sample"
            and isinstance(self._family, CompositeFamily)
            and self._family.seed_fn is not None
        ):
            import numpy as _np

            rng = kw.get("rng") or _np.random.RandomState(0)  # fixed default: an un-seeded fit is deterministic
            seed = self._family.seed_fn(self._args, data, rng, seed_child)
            if seed is not None:
                kw["prev_estimate"] = seed
        import sys

        out = open("/dev/null", "w") if not print_iter else sys.stdout
        try:
            fitted = optimize(
                data,
                est,
                max_its=max_its,
                delta=delta,
                backend=backend,
                num_workers=num_workers,
                engine=engine,
                precision=precision,
                print_iter=max(print_iter, 1),
                out=out,
                **kw,
            )
        finally:
            if not print_iter:
                out.close()
        if missing == "marginalize":
            from mixle.stats.missing import unwrap_marginalized

            fitted = unwrap_marginalized(fitted)  # strip the Optional wrappers; recover the base model
        return _stash_explanation(RandomVariable._bound(fitted, name=self._name))

    def __repr__(self) -> str:
        if self._kind == "bound":
            return f"RV(bound={self._dist!r})"
        inner = ", ".join("free" if _is_free(a) else repr(a) for a in self._args)
        nm = f", name={self._name!r}" if self._name else ""
        return f"RV({self._family.name}({inner}){nm})"


def constrain(*constraints) -> RandomVariable:
    """A joint random variable formed by conditioning several RVs on a relation among them.

    ``constrain(a < b)`` is the pair ``(a, b)`` restricted to ``a < b``; pass several
    constraints (or combine with ``& | ~``) for richer regions, e.g.
    ``constrain(a < b, b < c)`` orders three variables. The result samples by joint rejection
    and answers ``.sample(n)`` (an ``(n, k)`` array, columns in ``.columns`` order),
    ``.mean()``/``.var()`` (per-variable), ``.prob()`` (probability the relation holds), and
    ``.log_prob(x)`` (renormalized joint density of the independent variables on the region).
    """
    if not constraints:
        raise ValueError("constrain() needs at least one constraint.")
    for c in constraints:
        if not isinstance(c, Constraint):
            raise TypeError("constrain() expects Constraints from comparisons, e.g. a < b.")
    combined = constraints[0]
    for c in constraints[1:]:
        combined = combined & c
    leaves = combined.leaves
    if len(leaves) < 1:
        raise ValueError("constraint references no random variables.")
    for lv in leaves:
        if lv._kind not in ("sample", "bound") or lv.has_free:
            raise ValueError(
                "constrain() variables must be concrete RVs (a distribution with fixed "
                "parameters), not models with `free` holes; fit those first."
            )
    return RandomVariable("joint", args=(leaves, combined), name=None)


def _param_handle(dim: int, *, name=None, kind: str = "vector", support: str = "real") -> RandomVariable:
    """Build a referenceable vector/matrix parameter handle (the result of calling ``free(...)``).

    ``kind``: ``vector`` (entries on ``support`` real/positive/unit), ``ordered`` (increasing),
    ``simplex`` (sums to 1), or ``cholesky`` (an SPD covariance). Placed in a constructor slot and
    referenced in constraints — ``m = free(3, name="mu"); MVN(3, mean=m).fit(X, constraints=increasing(m))``.
    The handle behaves like a vector RV in constraint expressions (``m[i]``, ``m[0] < m[1]``, ...).
    """
    dim = int(dim)
    if kind == "vector":
        spec = _VectorSpec(dim, support, name)
    elif kind == "ordered":
        spec = _OrderedSpec(dim, name)
    elif kind == "simplex":
        spec = _SimplexSpec(np.ones(dim), rows=1, name=name)
    elif kind == "cholesky":
        spec = _CholeskySpec(dim, name)
    else:
        raise ValueError(f"unknown free(...) kind {kind!r}; use vector/ordered/simplex/cholesky.")
    return RandomVariable("param", args=(spec,), name=name)


def _rv_reconstruct(kind, fam_name, args, name, keys, dist, scope, reparam=None):
    """Rebuild a RandomVariable from its picklable structural fields."""
    if kind == "bound":
        return RandomVariable._bound(dist, name=name)
    family = _FAMILIES[fam_name] if fam_name is not None else None
    return RandomVariable(kind, family=family, args=args, name=name, keys=keys, scope=scope, reparam=reparam)


# -------------------------------------------------------------------- the lowering
def lower(rv: RandomVariable, *, target: str = "dist"):
    """The one routing site: symbolic RandomVariable -> existing mixle object.

    ``target='dist'`` returns a concrete ``*Distribution`` (needs no ``free`` holes);
    ``target='estimator'`` returns a ``*Estimator``. Results are cached per random variable.
    """
    cache = rv._cache
    if target in cache:
        return cache[target]

    if rv._kind == "bound":
        if target == "dist":
            cache[target] = rv._dist
            return rv._dist
        raise ValueError("a bound RandomVariable has no estimator to lower to")

    if rv._kind == "apply":
        if target != "dist":
            raise NotImplementedError("fitting through an RV transform is a later slice.")
        from mixle.stats.combinator.transform import TransformDistribution

        base, transform = rv._args
        d = TransformDistribution(lower(base, target="dist"), transform)
        cache[target] = d
        return d

    if rv._kind != "sample":
        raise ValueError(f"cannot lower RandomVariable of kind {rv._kind!r}")

    fam = rv._family
    if isinstance(fam, CompositeFamily):
        if target == "dist":
            result = fam.dist_fn(rv._args, lambda c: lower(c, target="dist"))
        elif target == "estimator":
            result = fam.est_fn(rv._args, lambda c: lower(c, target="estimator"), rv._name, rv._keys)
        else:
            raise ValueError(f"unknown lowering target {target!r}")
        cache[target] = result
        return result

    if target == "estimator":
        if not all(_is_free(a) for a in rv._args):
            if any(_is_free(a) for a in rv._args):
                raise NotImplementedError(
                    f"{fam.name}: partial `free` (some args fixed) is a later slice; use all-free or all-fixed for now."
                )
            # No holes: estimator of a fully-specified model is its own estimator.
            est = lower(rv, target="dist").estimator()
        else:
            est = fam.make_estimator(rv._name, rv._keys)
        cache[target] = est
        return est

    if target == "dist":
        if any(_is_free(a) for a in rv._args):
            raise ValueError(f"{fam.name} has unresolved `free` parameters; call .fit(data) first.")
        if any(isinstance(a, RandomVariable) for a in rv._args):
            raise NotImplementedError("latent/random parameters (a distribution in a slot) land in build slice 5.")
        d = fam.make_dist(rv._args, rv._name)
        cache[target] = d
        return d

    raise ValueError(f"unknown lowering target {target!r}")
