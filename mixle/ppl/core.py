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

import inspect
import math
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.capability import supports
from mixle.inference.estimation import optimize
from mixle.ppl._result import PointwiseLogLikelihood, Sampleable, Summarizable
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
    "ProbabilityEstimate",
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
    if not isinstance(name, str) or not name:
        raise ValueError("fitter name must be a non-empty string.")

    def deco(fn: Callable[..., RandomVariable]) -> Callable[..., RandomVariable]:
        if not callable(fn):
            raise TypeError("registered fitter must be callable.")
        if name in _FITTERS:
            raise ValueError(f"fitter {name!r} is already registered; silent replacement is forbidden.")
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
    # The specialized mixture route implements one model only: scalar Gaussian
    # components with unknown mean/scale and inferred symmetric weights.  Reject
    # every other declaration instead of silently replacing its likelihood or
    # discarding fixed values and priors.
    from mixle.ppl import vmp as _vmp

    if isinstance(rv._family, CompositeFamily) and rv._family.name == "Mixture":
        comps, weights = rv._args
        supported = all(
            c._kind == "sample"
            and not isinstance(c._family, CompositeFamily)
            and c._family.name == "Normal"
            and len(c._args) == 2
            and all(a is free for a in c._args)
            for c in comps
        )
        if not supported or (weights is not None and weights is not free):
            raise NotImplementedError(
                "mixture VMP supports only Mix([Normal(free, free), ...], weights=None|free); "
                "use how='vi' or how='mcmc' for other families, fixed parameters, or declared priors."
            )
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
        return _param_handle(dim, name=name, kind=kind, support=support)

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
        if self.alpha.ndim != 1 or self.alpha.size == 0 or not np.isfinite(self.alpha).all():
            raise ValueError("simplex concentration must be a non-empty finite vector.")
        if np.any(self.alpha <= 0.0):
            raise ValueError("simplex concentration entries must be strictly positive.")
        self.rows = _exact_positive_int(rows, "simplex rows")
        self.name = name


class _VectorSpec:
    """A vector-valued parameter of a combinator (e.g. an MVN mean): ``dim`` independent scalar
    slots of one ``support`` (``real``/``positive``/``unit``), assembled into a vector."""

    __slots__ = ("dim", "support", "name")

    def __init__(self, dim: int, support: str = "real", name: str | None = None):
        self.dim = _exact_positive_int(dim, "vector dimension")
        if support not in {"real", "positive", "unit"}:
            raise ValueError(f"vector support must be real, positive, or unit, got {support!r}.")
        self.support = support
        self.name = name


class _OrderedSpec:
    """A strictly-increasing vector parameter (``v[0] < v[1] < ...``): one real base entry plus
    ``dim-1`` positive increments, assembled as a cumulative sum. Gives ordered means *by
    construction* (the standard mixture/HMM identifiability device) with no rejection."""

    __slots__ = ("dim", "name")

    def __init__(self, dim: int, name: str | None = None):
        self.dim = _exact_positive_int(dim, "ordered dimension")
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
        self.dim = _exact_positive_int(dim, "Cholesky dimension")
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

    __slots__ = ("leaves", "pred", "desc", "residual", "soft", "reduction")

    def __init__(self, leaves, pred, desc, residual=None, soft=False, reduction="all"):
        if reduction not in {"all", "any"}:
            raise ValueError("constraint reduction must be 'all' or 'any'.")
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
        self.reduction = reduction

    @property
    def rv(self):
        """The single RV this constraint restricts (back-compat for one-variable events)."""
        if len(self.leaves) != 1:
            raise AttributeError("constraint involves multiple RVs; use .leaves.")
        return self.leaves[0]

    def eval(self, env):
        """Evaluate the constraint predicate against an environment of RV values."""
        return self.pred(env)

    def eval_rows(self, env, rows: int | None = None) -> np.ndarray:
        """Evaluate and reduce to exactly one boolean per sampled row."""
        return _row_mask(self.pred(env), rows=rows, reduction=self.reduction)

    def with_reduction(self, reduction: str) -> Constraint:
        """Return the same event with vector entries reduced by ``all`` or ``any`` per row."""
        return Constraint(self.leaves, self.pred, self.desc, self.residual, self.soft, reduction)

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
            "all",
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
            "all",
        )

    def __invert__(self):
        # Negation has no smooth penalty surface; only the hard (boolean) mode survives.
        return Constraint(
            self.leaves, lambda env: ~np.asarray(self.pred(env)), f"~{self.desc}", None, False, self.reduction
        )

    def __bool__(self):
        raise TypeError(
            "a Constraint has no truth value — Python chained comparisons (a < b < c) and "
            "`and`/`or` are not supported; combine with & | ~ instead, e.g. (a < b) & (b < c)."
        )

    def __repr__(self):
        return f"Constraint({self.desc})"


Event = Constraint  # back-compat alias


class ProbabilityEstimate(float):
    """A Monte Carlo probability plus the experiment needed to interpret it.

    The numeric value remains backward-compatible with ordinary probability arithmetic. ``hits``,
    ``trials``, ``seed``, and a Wilson 95% interval distinguish zero observed hits from a mathematical
    proof that the event has probability zero.
    """

    def __new__(cls, hits: int, trials: int, seed: int):
        if not 0 <= hits <= trials or trials <= 0:
            raise ValueError("probability estimate requires 0 <= hits <= trials and trials > 0.")
        value = hits / trials
        obj = float.__new__(cls, value)
        obj.hits = int(hits)
        obj.trials = int(trials)
        obj.seed = int(seed)
        z2 = 1.959963984540054**2
        denom = 1.0 + z2 / trials
        center = (value + z2 / (2.0 * trials)) / denom
        radius = 1.959963984540054 * math.sqrt(value * (1.0 - value) / trials + z2 / (4.0 * trials**2)) / denom
        obj.lower = 0.0 if hits == 0 else max(0.0, center - radius)
        obj.upper = 1.0 if hits == trials else min(1.0, center + radius)
        return obj

    def as_dict(self) -> dict:
        """Return a serialization-friendly experiment receipt."""
        return {
            "estimate": float(self),
            "hits": self.hits,
            "trials": self.trials,
            "seed": self.seed,
            "interval_95": (self.lower, self.upper),
        }


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
    # STAT-RR18-03: these three previously carried NO caveats, so an accepted approximate fit
    # explained itself with an empty limitations list
    "sample": [
        "DYNAMICALLY selects a concrete sampler (ensemble or NUTS) at fit time -- this pre-fit "
        "explanation cannot name which; call explain_fit() on the FITTED model for the route "
        "that actually ran",
        "posterior samples, not an exact posterior: check R-hat/ESS/MCSE before promoting "
        "(posterior_summary refuses 'ok' without usable diagnostics)",
    ],
    "ensemble": [
        "affine-invariant ensemble sampler: approximate posterior samples; multimodal or "
        "high-dimensional targets can mix poorly -- check R-hat/ESS/MCSE before promoting"
    ],
    "vmp": [
        "variational message passing: a factorized (mean-field) approximation that "
        "systematically UNDERSTATES posterior variance; intervals from it are optimistic"
    ],
    "neural": [
        "neural-conditional fit: a point estimate of the network weights by stochastic "
        "gradient training -- NOT a MAP under a stated prior and NOT a posterior; no "
        "uncertainty over the network is quantified",
        "training is stochastic: two fits differ unless the torch/global seeds are pinned",
    ],
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


def _row_mask(mask, *, rows: int | None = None, reduction: str = "all") -> np.ndarray:
    """Validate and reduce a constraint mask to exactly one boolean per sampled row."""
    if reduction not in {"all", "any"}:
        raise ValueError("constraint reduction must be 'all' or 'any'.")
    m = np.asarray(mask)
    if m.dtype.kind != "b":
        raise TypeError("constraint predicates must return boolean masks.")
    if m.ndim == 0:
        if rows is None:
            return m.reshape(1)
        return np.full(rows, bool(m), dtype=bool)
    if rows is not None and m.shape[0] != rows:
        raise ValueError(f"constraint mask has {m.shape[0]} row(s), expected {rows}.")
    if m.ndim > 1:
        axes = tuple(range(1, m.ndim))
        m = m.all(axis=axes) if reduction == "all" else m.any(axis=axes)
    if m.ndim != 1:
        raise ValueError("constraint reduction did not produce one boolean per row.")
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


def _expressions_share_leaf(a, b) -> bool:
    """Whether two expressions reference at least one identical stochastic leaf."""
    left = {id(leaf) for leaf in _expr_leaves(a)}
    return any(id(leaf) in left for leaf in _expr_leaves(b))


def _declared_event_width(rv: RandomVariable) -> int:
    """Return a declared flat event width without sampling the distribution."""
    if rv._kind in {"apply", "pow"}:
        return _declared_event_width(rv._args[0])
    if rv._kind == "select":
        return 1
    if rv._kind in {"sum", "prod"}:
        left, right = (_declared_event_width(arg) for arg in rv._args)
        if left != right and left != 1 and right != 1:
            raise ValueError(f"incompatible derived event widths {left} and {right}.")
        return max(left, right)
    if rv._kind == "sample" and getattr(rv._family, "name", None) in {"MVN", "DiagGaussian"}:
        return int(rv._args[0])
    if rv._kind == "bound" and rv._dist is not None:
        mean = getattr(rv._dist, "mu", None)
        if mean is not None and np.ndim(mean) == 1:
            return int(np.asarray(mean).size)
    return 1


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
        "validator",
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
        validator=None,
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
        # Optional family-specific validation supplements the common finite/support contract below.
        # It receives the complete conventional argument tuple.
        self.validator = validator

    def validate_args(self, args: tuple[Any, ...]) -> None:
        """Validate fixed conventional parameters without rejecting symbolic parameter expressions."""
        if len(args) != self.arity:
            raise ValueError(f"{self.name} expects {self.arity} parameter(s), got {len(args)}.")
        structural = (_Free, RandomVariable, _SimplexSpec, _VectorSpec, _OrderedSpec, _CholeskySpec)
        for index, (arg, support) in enumerate(zip(args, self.support)):
            if arg is None:
                raise ValueError(f"{self.name} parameter {index} cannot be None.")
            if isinstance(arg, structural):
                continue
            values = list(arg.values()) if isinstance(arg, dict) else arg
            try:
                arr = np.asarray(values)
            except (TypeError, ValueError):
                continue  # neural/callable expressions are validated by their owning lowering route
            if arr.dtype.kind not in "fiu":
                continue
            if arr.dtype.kind == "b" or not np.isfinite(arr.astype(float)).all():
                raise ValueError(f"{self.name} parameter {index} must be finite numeric data.")
            numeric = arr.astype(float)
            if support == "positive" and np.any(numeric <= 0.0):
                raise ValueError(f"{self.name} parameter {index} must be strictly positive.")
            if support == "unit" and (np.any(numeric < 0.0) or np.any(numeric > 1.0)):
                raise ValueError(f"{self.name} parameter {index} must lie in [0, 1].")
            if support == "nonnegative" and np.any(numeric < 0.0):
                raise ValueError(f"{self.name} parameter {index} must be non-negative.")
            if support == "fixed_nonnegative_integer":
                if numeric.ndim != 0 or numeric < 0.0 or numeric != np.floor(numeric):
                    raise ValueError(f"{self.name} parameter {index} must be an exact non-negative integer.")
        if self.validator is not None:
            self.validator(args)

    def make_dist(self, args: tuple[Any, ...], name: str | None):
        """Construct the concrete distribution for conventional PPL arguments."""
        self.validate_args(args)
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
            signature = inspect.signature(self.est_cls)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"cannot inspect estimator constructor {self.est_cls!r}; "
                "register an estimator with an inspectable keyword signature."
            ) from exc
        parameters = signature.parameters.values()
        accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
        accepted_names = {
            p.name
            for p in parameters
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        call_kwargs = (
            kwargs if accepts_extra else {key: value for key, value in kwargs.items() if key in accepted_names}
        )
        # Do not catch TypeError here. A TypeError raised inside the constructor is a code/data defect,
        # not evidence that a different calling convention should be retried.
        return self.est_cls(**call_kwargs)


class CompositeFamily:
    """Lowering recipe for a family whose arguments are themselves RandomVariables
    (mixtures, sequences, HMMs, ...). Kept out of :class:`Family` so the flat-family
    path stays trivial. ``dist_fn``/``est_fn`` receive the raw args plus a callback
    that lowers a child RandomVariable to a dist / estimator.
    """

    __slots__ = ("name", "dist_fn", "est_fn", "seed_fn", "read", "fit_fn", "validator")

    def __init__(self, name, dist_fn, est_fn, seed_fn=None, read=None, fit_fn=None, validator=None):
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
        self.validator = validator

    def validate_args(self, args: tuple[Any, ...]) -> None:
        if self.validator is not None:
            self.validator(args)


_FAMILIES: dict[str, Any] = {}
_DIST_TO_FAMILY: dict[type, Family] = {}  # reverse map for reading fitted params
_DIST_TO_COMPOSITE_READ: dict[type, Any] = {}  # composite dist type -> read(dist, read_params)


def register_family(
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
    validator=None,
) -> Family:
    """Register a flat PPL family and its distribution/estimator lowering rules."""
    if not isinstance(name, str) or not name:
        raise ValueError("family name must be a non-empty string.")
    if name in _FAMILIES:
        raise ValueError(f"family {name!r} is already registered; silent replacement is forbidden.")
    if dist_cls in _DIST_TO_FAMILY:
        other = _DIST_TO_FAMILY[dist_cls]
        raise ValueError(f"distribution class {dist_cls.__name__} is already registered as family {other.name!r}.")
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
        validator=validator,
    )
    _FAMILIES[name] = fam
    _DIST_TO_FAMILY[dist_cls] = fam
    return fam


def register_composite(
    name, dist_fn, est_fn, seed_fn=None, dist_cls=None, read=None, fit_fn=None, validator=None
) -> CompositeFamily:
    """Register a composite PPL family with custom lowering or fitting hooks."""
    if not isinstance(name, str) or not name:
        raise ValueError("composite family name must be a non-empty string.")
    if name in _FAMILIES:
        raise ValueError(f"family {name!r} is already registered; silent replacement is forbidden.")
    if dist_cls is not None and dist_cls in _DIST_TO_COMPOSITE_READ:
        raise ValueError(f"composite distribution class {dist_cls.__name__} already has a registered parameter reader.")
    fam = CompositeFamily(name, dist_fn, est_fn, seed_fn=seed_fn, read=read, fit_fn=fit_fn, validator=validator)
    _FAMILIES[name] = fam
    if dist_cls is not None and read is not None:
        _DIST_TO_COMPOSITE_READ[dist_cls] = read
    return fam


def _structural_parameter_dimension(spec) -> int:
    """Return statistical degrees of freedom for one declared structural parameter."""
    if isinstance(spec, _SimplexSpec):
        return spec.rows * (len(spec.alpha) - 1)
    if isinstance(spec, (_VectorSpec, _OrderedSpec)):
        return spec.dim
    if isinstance(spec, _CholeskySpec):
        return spec.dim * (spec.dim + 1) // 2
    raise TypeError(f"unknown structural parameter {type(spec).__name__}.")


def _inferable_parameter_dimension(rv: RandomVariable) -> int | None:
    """Count declared inferable degrees of freedom without inspecting fitted numeric summaries.

    Fixed constructor values and derived result fields never contribute. Shared parameter/prior
    handles contribute once by identity, while separate uses of the bare ``free`` token are
    independent scalar declarations. ``None`` means the backend owns parameters that this structural
    model cannot count before execution (currently arbitrary neural predictors).
    """

    seen_handles: set[int] = set()

    def slot(value) -> int | None:
        if value is free:
            return 1
        if isinstance(value, (_SimplexSpec, _VectorSpec, _OrderedSpec, _CholeskySpec)):
            return _structural_parameter_dimension(value)
        if isinstance(value, _LinearPredictor):
            total = 0
            for coefficient, _field in value.terms:
                count = slot(coefficient)
                if count is None:
                    return None
                total += count
            if value.intercept is not None:
                count = slot(value.intercept)
                if count is None:
                    return None
                total += count
            for _group_name, slopes in value.groups:
                width = 1 + len(slopes)
                total += width * (width + 1) // 2
            return total
        if isinstance(value, _NeuralPredictor):
            return None
        if not isinstance(value, RandomVariable):
            return 0
        key = id(value)
        if key in seen_handles:
            return 0
        seen_handles.add(key)
        if value._kind == "param":
            return _structural_parameter_dimension(value._args[0])
        if value._kind in {"apply", "pow", "select", "gather"}:
            return slot(value._args[0])
        if value._kind in {"sum", "prod"}:
            left, right = slot(value._args[0]), slot(value._args[1])
            return None if left is None or right is None else left + right
        if value._kind == "sample":
            # In a flat family slot this RV is one latent parameter. Random hyperparameters are
            # additional latents and are counted recursively; fixed hyperparameters are not.
            total = 1
            for argument in value._args:
                if isinstance(argument, RandomVariable):
                    count = slot(argument)
                    if count is None:
                        return None
                    total += count
            return total
        return 0

    def model(node: RandomVariable) -> int | None:
        if node._kind in {"apply", "pow", "select", "gather"}:
            return model(node._args[0])
        if node._kind in {"sum", "prod"}:
            left = model(node._args[0]) if isinstance(node._args[0], RandomVariable) else 0
            right = model(node._args[1]) if isinstance(node._args[1], RandomVariable) else 0
            return None if left is None or right is None else left + right
        if node._kind != "sample":
            return slot(node)
        family = node._family
        if not isinstance(family, CompositeFamily):
            total = 0
            for argument in node._args:
                count = slot(argument)
                if count is None:
                    return None
                total += count
            return total

        name = family.name
        if name in {"Mixture", "SemiMix"}:
            components, weights = node._args
            total = 0
            for component in components:
                count = model(component)
                if count is None:
                    return None
                total += count
            if weights is None or weights is free:
                total += len(components) - 1
            elif isinstance(weights, (_SimplexSpec, RandomVariable)):
                count = slot(weights)
                if count is None:
                    return None
                total += count
            return total
        if name == "Sequence":
            return model(node._args[0])
        if name in {"MVN", "DiagGaussian"}:
            dim, mean, spread = node._args
            total = int(dim) if mean is None else slot(mean)
            if total is None:
                return None
            if spread is None:
                total += int(dim) * (int(dim) + 1) // 2 if name == "MVN" else int(dim)
            else:
                count = slot(spread)
                if count is None:
                    return None
                total += count
            return total
        if name == "Markov":
            components = node._args[0]
            total = 0
            for component in components:
                count = model(component)
                if count is None:
                    return None
                total += count
            k = len(components)
            transitions = node._args[1] if len(node._args) > 1 else None
            initial = node._args[2] if len(node._args) > 2 else None
            if not isinstance(transitions, np.ndarray):
                total += k * (k - 1)
            if not isinstance(initial, np.ndarray):
                total += k - 1
            return total
        if name == "LDA":
            # LDA declares only shape hyperparameters (num_topics, vocab_size, alpha) -- its actual
            # parameters, the topic-word simplices, are created by the estimator and appear in no
            # slot. The generic branch below therefore counted 0, which made a model with plenty to
            # infer look fully specified: .fit() took the no-op structural shortcut and rejected
            # max_its/rng as options a fully specified model cannot consume. Count the topics:
            # num_topics simplices over vocab_size words, each with vocab_size - 1 free entries.
            num_topics, vocab_size, alpha = node._args
            total = int(num_topics) * (int(vocab_size) - 1)
            count = slot(alpha)  # alpha is a fixed float by default, but may be declared inferable
            if count is None:
                return None
            return total + count
        # Generic composites recurse into declared child models and structural/free slots only.
        total = 0
        for argument in node._args:
            values = argument if isinstance(argument, (list, tuple)) else (argument,)
            for value in values:
                count = model(value) if isinstance(value, RandomVariable) and value._kind == "sample" else slot(value)
                if count is None:
                    return None
                total += count
        return total

    return model(rv)


def compare(models, data, *, by: str = "aic"):
    """Compare fitted models on ``data``. Returns rows sorted best-first by ``by``
    ('aic' | 'bic' | 'loglik' | 'waic' | 'loo').

    ``'waic'`` and ``'loo'`` are the Bayesian predictive criteria (integrating over parameter
    uncertainty via the posterior draws of a Bayesian fit); ``'aic'``/``'bic'`` use the point estimate.
    Each row also reports ``elpd`` differences from the best model (``d_elpd``) for waic/loo.
    """
    keys = {
        "loglik": lambda r: -r["loglik"],
        "aic": lambda r: r["aic"],
        "bic": lambda r: r["bic"],
        "waic": lambda r: r["waic"],
        "loo": lambda r: r["loo"],
    }
    if by not in keys:
        raise ValueError(f"by must be one of {sorted(keys)}, got {by!r}.")
    try:
        models = list(models)
    except TypeError as exc:
        raise TypeError("models must be an iterable of fitted RandomVariable objects.") from exc
    if not models:
        raise ValueError("compare() needs at least one fitted model.")
    if any(not isinstance(model, RandomVariable) or not model.is_bound for model in models):
        raise TypeError("compare() models must be fitted RandomVariable objects.")
    try:
        data = list(data)
    except TypeError as exc:
        raise TypeError("compare() data must be an iterable of observations.") from exc
    if not data:
        raise ValueError("compare() data must not be empty.")

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
    rows = sorted(rows, key=keys[by])
    if by in ("waic", "loo"):
        best = rows[0]["elpd"]
        for r in rows:
            r["d_elpd"] = r["elpd"] - best
    return rows


def _indexed_group_layout(labels, expected_rows: int) -> tuple[np.ndarray, tuple, dict]:
    """Validate group identities and return stable first-seen integer indices."""
    raw = np.asarray(labels, dtype=object)
    if raw.ndim != 1 or raw.size != expected_rows:
        raise ValueError(f"group labels must be one-dimensional with {expected_rows} rows; got shape {raw.shape}.")
    ordered = []
    mapping = {}
    indices = np.empty(expected_rows, dtype=int)
    label_type = None
    for row, value in enumerate(raw.tolist()):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or (isinstance(value, Real) and not math.isfinite(float(value))):
            raise ValueError(f"group label at row {row} must be finite and non-null.")
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"group label at row {row} is not hashable.") from exc
        current_type = type(value)
        if label_type is None:
            label_type = current_type
        elif current_type is not label_type:
            raise TypeError(
                f"group labels must have one homogeneous type; saw {label_type.__name__} and {current_type.__name__}."
            )
        if value not in mapping:
            mapping[value] = len(ordered)
            ordered.append(value)
        indices[row] = mapping[value]
    if not ordered:
        raise ValueError("group labels must contain at least one group.")
    return indices, tuple(ordered), dict(mapping)


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

    __slots__ = (
        "_kind",
        "_family",
        "_args",
        "_name",
        "_keys",
        "_dist",
        "_result",
        "_cache",
        "_scope",
        "_reparam",
        "_group_by",
    )

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
        result: Any | None = None,
        scope="shared",
        reparam=None,
        group_by=None,
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
        object.__setattr__(self, "_group_by", group_by)

    def __setattr__(self, *a):  # enforce immutability
        raise AttributeError("RandomVariable is immutable")

    def __reduce__(self):
        # Versioned artifact state lets fitted models and structural grouping cross process/storage
        # boundaries without silently degrading into unfitted or ungrouped objects. Runtime-only caches
        # (lowered encoders, KDEs, Monte Carlo estimates) remain deliberately excluded.
        fam_name = self._family.name if self._family is not None else None
        durable_cache = {
            key: self._cache[key]
            for key in (
                "certificate",
                "_fit_explanation",
                "_free_parameter_count",
                "_group_labels",
                "_group_index",
            )
            if key in self._cache
        }
        return (
            _rv_reconstruct_v2,
            (
                1,
                self._kind,
                fam_name,
                self._args,
                self._name,
                self._keys,
                self._dist,
                self._result,
                self._scope,
                self._reparam,
                self._group_by,
                durable_cache,
            ),
        )

    # -- constructors -------------------------------------------------------
    @classmethod
    def _sample(cls, family_name, args, *, name=None, keys=None, scope="shared") -> RandomVariable:
        fam = _FAMILIES[family_name]
        if isinstance(fam, (Family, CompositeFamily)):
            fam.validate_args(tuple(args))
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
        if by is not None and (not isinstance(by, str) or not by):
            raise ValueError("each(by=...) requires a non-empty string field name.")
        return RandomVariable(
            "sample",
            family=self._family,
            args=self._args,
            name=self._name,
            keys=self._keys,
            scope="grouped",
            group_by=by,
        )

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
            group_by=self._group_by,
        )

    @property
    def scope(self) -> str:
        """Return the variable scope, such as scalar, grouped, or global."""
        return self._scope

    @classmethod
    def _bound(cls, dist, *, name=None, result: Any | None = None) -> RandomVariable:
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
    def independent(self) -> RandomVariable:
        """Return an explicitly independent copy of one atomic random variable.

        Reusing the same object in an expression means reusing the same draw (``x - x == 0``).
        Use ``x.independent()`` when an identically distributed but independent draw is intended.
        """
        if self._kind == "sample":
            return RandomVariable(
                "sample",
                family=self._family,
                args=self._args,
                name=self._name,
                keys=self._keys,
                scope=self._scope,
                reparam=self._reparam,
                group_by=self._group_by,
            )
        if self._kind == "bound":
            return RandomVariable._bound(self._dist, name=self._name, result=self._result)
        raise TypeError("independent() applies to an atomic sample or bound random variable.")

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
        """Whether this parameter/expression graph contains one or more inferable unknowns."""
        if self._kind == "sample" and getattr(self._family, "name", None) in {"MVN", "DiagGaussian"}:
            # ``None`` is a concrete standard distribution for generative use but asks the paired
            # estimator to learn that slot when fit() is called. It is not an explicit free/prior hole,
            # so constraints over MVN(dim) remain valid generative constraints.
            return any(
                value is free
                or isinstance(value, (_SimplexSpec, _VectorSpec, _OrderedSpec, _CholeskySpec))
                or isinstance(value, RandomVariable)
                for value in self._args[1:]
            )
        dimension = _inferable_parameter_dimension(self)
        return dimension is None or dimension > 0

    @property
    def name(self) -> str | None:
        """Return the optional user-visible variable name."""
        return self._name

    @property
    def parameter_dimension(self) -> int | None:
        """Declared inferable degrees of freedom, or ``None`` when a backend owns an unknown-size model.

        A fitted artifact returns the count recorded before execution; an unfitted structural model is
        counted directly. Fixed constructor values and derived posterior summaries are excluded.
        """
        if self._kind == "bound":
            value = self._cache.get("_free_parameter_count")
            return None if value is None else int(value)
        return _inferable_parameter_dimension(self)

    @property
    def group_index(self) -> dict | None:
        """Stable label-to-posterior-index mapping for an indexed-group fit."""
        mapping = self._cache.get("_group_index")
        return None if mapping is None else dict(mapping)

    @property
    def group_labels(self) -> tuple | None:
        """Group labels in stable first-observation order for an indexed-group fit."""
        labels = self._cache.get("_group_labels")
        return None if labels is None else tuple(labels)

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
            w = _declared_event_width(lv)
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
    def result(self) -> Any | None:
        """Inference metadata (EM history / MCMC chain) when present; else None."""
        return self._result

    # -- query verbs (valid once concrete) ----------------------------------
    def sample(
        self,
        n: int | None = None,
        seed: int | None = None,
        size: int | None = None,
        *,
        max_attempts: int = 100,
    ):
        """Draw samples from the represented distribution or derived expression.

        ``n`` and ``size`` are aliases (``size`` matches the ``stats``-layer samplers); pass at
        most one. ``None`` returns a single draw.
        """
        n = coalesce_alias("n", n, "size", size, required=False, default=None)
        if n is not None:
            n = _exact_positive_int(n, "sample size")
        max_attempts = _exact_positive_int(max_attempts, "max_attempts")
        if self._kind == "joint":  # joint rejection sampling under a relation
            leaves, constraint = self._args
            if constraint.soft:
                raise ValueError("measure-zero/soft constraints cannot be sampled by rejection.")
            rng = np.random.RandomState(seed)
            k = n if n is not None else 1
            kept = []
            have = 0
            drawn = 0
            for _attempt in range(max_attempts):
                if have >= k:
                    break
                batch = max(k * 2, 1024)
                cols = {lv: np.asarray(lv.sample(batch, seed=int(rng.randint(1, 2**31))), dtype=float) for lv in leaves}
                mask = constraint.eval_rows(cols, rows=batch)
                block = np.concatenate([cols[lv].reshape(batch, -1)[mask] for lv in leaves], axis=1)
                kept.append(block)
                have += len(block)
                drawn += batch
            if have < k:
                raise RuntimeError(
                    "joint rejection sampling exhausted max_attempts "
                    f"(accepted={have}, drawn={drawn}, observed_rate={have / drawn if drawn else 0.0:.6g})."
                )
            out = np.concatenate(kept, axis=0)[:k]
            return out if n is not None else out[0]
        if self._kind in {"apply", "sum", "prod", "pow", "select"}:
            # Evaluate the expression as a DAG: each distinct leaf gets one draw per row and every
            # repeated reference reuses it. Distinct wrappers, including independent(), draw separately.
            leaves = _expr_leaves(self)
            rng = np.random.RandomState(seed)
            k = n if n is not None else 1
            env = {leaf: np.asarray(leaf.sample(k, seed=int(rng.randint(1, 2**31)))) for leaf in leaves}
            out = np.asarray(_eval_expr(self, env))
            if n is not None:
                return out
            first = out[0]
            return float(first) if np.ndim(first) == 0 else first
        if self._kind == "given":  # rejection sampling from the region
            base, event = self._args
            if event.soft:
                raise ValueError("measure-zero/soft events cannot be sampled by rejection.")
            rng = np.random.RandomState(seed)
            k = n if n is not None else 1
            kept = []
            accepted = 0
            drawn = 0
            for _attempt in range(max_attempts):
                if accepted >= k:
                    break
                batch = max(k * 2, 1024)
                draw = np.asarray(base.sample(batch, seed=int(rng.randint(1, 2**31))))
                mask = event.eval_rows({base: draw}, rows=batch)
                kept.append(draw[mask])
                accepted += int(mask.sum())
                drawn += batch
            if accepted < k:
                raise RuntimeError(
                    "conditional rejection sampling exhausted max_attempts "
                    f"(accepted={accepted}, drawn={drawn}, observed_rate={accepted / drawn if drawn else 0.0:.6g})."
                )
            out = np.concatenate(kept)[:k]
            if n is not None:
                return out
            first = out[0]
            return float(first) if np.ndim(first) == 0 else first
        return lower(self, target="dist").sampler(seed=seed).sample(size=n)

    def log_prob(self, x):
        """Evaluate the log probability or log density at ``x``."""
        if self._kind == "joint":  # joint density of independent leaves / Z
            leaves, constraint = self._args
            widths = [_declared_event_width(lv) for lv in leaves]
            if any(w > 1 for w in widths):
                raise NotImplementedError(
                    "joint log_prob over vector-valued variables is not supported yet; use .sample / .mean / .prob."
                )
            xa = np.atleast_2d(np.asarray(x, dtype=float))
            if xa.shape[1] != len(leaves):
                raise ValueError(f"expected {len(leaves)} columns, got shape {xa.shape}.")
            estimate = self.prob()
            if estimate.hits == 0:
                raise RuntimeError(
                    "the joint constraint had zero accepted Monte Carlo draws; "
                    f"P(constraint) is unresolved in [0, {estimate.upper:.6g}] at 95% confidence."
                )
            logZ = math.log(float(estimate))
            env = {lv: xa[:, j] for j, lv in enumerate(leaves)}
            base_lp = sum(np.atleast_1d(lv.log_prob(xa[:, j])) for j, lv in enumerate(leaves))
            out = np.where(constraint.eval_rows(env, rows=xa.shape[0]), base_lp - logZ, -np.inf)
            return float(out[0]) if np.ndim(x) == 1 else out
        if self._kind == "sum":  # exact convolution only; approximation requires a separate explicit API
            a, b = self._args
            if a is b:
                return a._affine(0.0, 2.0).log_prob(x)
            if _expressions_share_leaf(a, b):
                raise NotImplementedError(
                    "the density of a sum with shared stochastic leaves is not implemented; "
                    "sampling preserves the declared dependency graph."
                )
            cd = _convolve(lower(a, target="dist"), lower(b, target="dist"))
            if cd is not None:
                return RandomVariable._bound(cd).log_prob(x)
            raise NotImplementedError(
                "this independent sum has no registered exact convolution; "
                "Mixle does not silently substitute a KDE or change discrete/mixed measure semantics."
            )
        if self._kind in {"prod", "pow"}:
            raise NotImplementedError(
                f"exact log-density semantics are not registered for derived {self._kind!r} expressions; "
                "sampling remains available."
            )
        if self._kind == "given":
            base, event = self._args
            estimate = self.prob_of_event()
            if estimate.hits == 0:
                raise RuntimeError(
                    "the conditioning event had zero accepted Monte Carlo draws; "
                    f"P(event) is unresolved in [0, {estimate.upper:.6g}] at 95% confidence."
                )
            logZ = math.log(float(estimate))
            width = _declared_event_width(base)
            raw = np.asarray(x, dtype=float)
            single = np.isscalar(x) if width == 1 else raw.ndim == 1
            if width == 1:
                xv = np.atleast_1d(raw)
            else:
                xv = raw.reshape(1, -1) if raw.ndim == 1 else raw
                if xv.ndim != 2 or xv.shape[1] != width:
                    raise ValueError(
                        f"conditioned vector event expects shape ({width},) or (n, {width}); got {raw.shape}."
                    )
            base_lp = np.atleast_1d(base.log_prob(xv))
            mask = event.eval_rows({base: xv}, rows=xv.shape[0])
            out = np.where(mask, base_lp - logZ, -np.inf)
            return float(out[0]) if single else out
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
        """Akaike information criterion (lower is better).

        ``k`` defaults to the declared inferable dimension recorded by :meth:`fit`; fixed constructor
        values and derived summaries are never counted. Models without such a record must pass ``k``.
        """
        if k is None:
            k = self._cache.get("_free_parameter_count")
            if k is None:
                raise ValueError("AIC needs k= because this artifact has no recorded inferable parameter dimension.")
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 0:
            raise ValueError("AIC parameter count k must be an exact non-negative integer.")
        k = int(k)
        return 2.0 * k - 2.0 * self.log_likelihood(data)

    def bic(self, data, k: int | None = None) -> float:
        """Bayesian information criterion (lower is better)."""
        if k is None:
            k = self._cache.get("_free_parameter_count")
            if k is None:
                raise ValueError("BIC needs k= because this artifact has no recorded inferable parameter dimension.")
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 0:
            raise ValueError("BIC parameter count k must be an exact non-negative integer.")
        k = int(k)
        data = list(data)
        n = len(data)
        if n == 0:
            raise ValueError("BIC requires at least one observation.")
        return k * math.log(n) - 2.0 * self.log_likelihood(data)

    def pointwise_log_likelihood(self, data) -> np.ndarray:
        """Return the ``(n_draws, n_obs)`` log-likelihood matrix used by WAIC / PSIS-LOO.

        For a Bayesian fit (``how='mcmc'|'hmc'|'ensemble'|'vi'``) each row is the log-likelihood of the
        data under one posterior draw. Point estimates have no posterior-draw matrix; use
        :meth:`plugin_log_likelihood` for their ordinary per-observation plug-in score.
        """
        r = self._result
        if r is not None and supports(r, PointwiseLogLikelihood) and getattr(r, "build", None) is not None:
            matrix = np.asarray(r.pointwise_log_likelihood(data), dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] < 2:
                raise ValueError("posterior pointwise log likelihood must contain at least two draw rows.")
            return matrix
        raise NotImplementedError(
            "Bayesian pointwise log likelihood is unavailable for this point-estimate artifact; "
            "use plugin_log_likelihood(data), AIC, or BIC instead."
        )

    def plugin_log_likelihood(self, data) -> np.ndarray:
        """Per-observation log likelihood under the fitted point estimate (no posterior integration)."""
        values = np.asarray(self.log_prob(list(data)), dtype=float)
        if values.ndim != 1 or not values.size or np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("plug-in log likelihood must be a non-empty one-dimensional vector without NaN or +Inf.")
        return values

    def waic(self, data) -> dict:
        """Widely Applicable Information Criterion from the posterior (lower ``waic`` is better).

        Returns ``{elpd_waic, p_waic, waic, se, n_draws, pointwise}``. Estimates out-of-sample
        predictive accuracy by integrating over parameter uncertainty -- the Bayesian analogue of
        ``aic``/``bic``. Point-estimate fits are rejected.
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
        ``{mean, std, q2.5, q97.5, mcse}`` (the 95% credible interval plus each mean's own Monte
        Carlo standard error, STAT-RR19-11) plus ``_acceptance_rate`` and, for multi-chain runs,
        ``_rhat`` / ``_ess`` / ``_n_chains``. For ``map``/``em`` it returns ``.params``.
        """
        r = self._result
        if r is not None and supports(r, Summarizable):
            return r.summary()
        return self.params

    def mean(self, samples: int = 20000, seed: int = 0):
        """Expected value, using an analytic distribution moment before bounded Monte Carlo."""
        samples = _exact_positive_int(samples, "moment sample count")
        if samples > 1_000_000:
            raise ValueError("moment sample count must not exceed 1,000,000.")
        distribution = None
        if self._kind == "sum":
            a, b = self._args
            if a is b:
                return 2.0 * a.mean(samples=samples, seed=seed)
            if not _expressions_share_leaf(a, b):
                distribution = _convolve(lower(a, target="dist"), lower(b, target="dist"))
        elif self._kind in {"sample", "bound", "apply"}:
            distribution = lower(self, target="dist")
        analytic = getattr(distribution, "mean", None)
        if callable(analytic):
            value = np.asarray(analytic())
            return float(value) if value.ndim == 0 else value
        s = np.asarray(self.sample(samples, seed=seed), dtype=float)
        return s.mean(axis=0) if self._kind == "joint" else float(np.mean(s))

    def var(self, samples: int = 20000, seed: int = 0):
        """Variance, using an analytic distribution moment before bounded Monte Carlo."""
        samples = _exact_positive_int(samples, "moment sample count")
        if samples > 1_000_000:
            raise ValueError("moment sample count must not exceed 1,000,000.")
        distribution = None
        if self._kind == "sum":
            a, b = self._args
            if a is b:
                return 4.0 * a.var(samples=samples, seed=seed)
            if not _expressions_share_leaf(a, b):
                distribution = _convolve(lower(a, target="dist"), lower(b, target="dist"))
        elif self._kind in {"sample", "bound", "apply"}:
            distribution = lower(self, target="dist")
        analytic = getattr(distribution, "variance", None)
        if callable(analytic):
            value = np.asarray(analytic())
            return float(value) if value.ndim == 0 else value
        s = np.asarray(self.sample(samples, seed=seed), dtype=float)
        return s.var(axis=0) if self._kind == "joint" else float(np.var(s))

    def prob(self, samples: int = 40000, seed: int = 999):
        """Monte Carlo probability receipt for a ``constrain(...)`` relation."""
        if self._kind != "joint":
            raise TypeError("prob() is only defined for a constrain(...) RV.")
        samples = _exact_positive_int(samples, "probability sample count")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise ValueError("probability seed must be an exact integer.")
        seed = int(seed)
        key = ("_pjoint", samples, seed)
        estimate = self._cache.get(key)
        if estimate is None:
            leaves, constraint = self._args
            rng = np.random.RandomState(seed)
            cols = {lv: np.asarray(lv.sample(samples, seed=int(rng.randint(1, 2**31))), dtype=float) for lv in leaves}
            mask = constraint.eval_rows(cols, rows=samples)
            estimate = ProbabilityEstimate(int(mask.sum()), samples, seed)
            self._cache[key] = estimate
        return estimate

    def prob_of_event(self, samples: int = 40000, seed: int = 999):
        """Monte Carlo probability receipt for the event underlying a conditioned RV."""
        if self._kind != "given":
            raise TypeError("prob_of_event() is only defined for a .given(...) RV.")
        samples = _exact_positive_int(samples, "probability sample count")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise ValueError("probability seed must be an exact integer.")
        seed = int(seed)
        key = ("_pevent", samples, seed)
        estimate = self._cache.get(key)
        if estimate is None:
            base, event = self._args
            draws = np.asarray(base.sample(samples, seed=seed))
            mask = event.eval_rows({base: draws}, rows=samples)
            estimate = ProbabilityEstimate(int(mask.sum()), samples, seed)
            self._cache[key] = estimate
        return estimate

    def predict(self, n: int = 1, rng=None):
        """Posterior-predictive draws. For a Bayesian fit (conjugate/mcmc/hmc) this
        integrates over parameter uncertainty (draw params from the posterior, then
        data); for a point fit (EM/MAP) it is the plug-in predictive (sample from the
        fitted distribution).
        """
        import numpy as _np

        if n is None:
            n = 1
        n = _exact_positive_int(n, "predictive sample count")
        if rng is None:
            rng = _np.random.RandomState()
        elif isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_)):
            rng = _np.random.RandomState(int(rng))
        if not hasattr(rng, "randint") and not hasattr(rng, "integers"):
            raise TypeError("rng must be an integer seed, NumPy RandomState, or Generator.")
        r = self._result
        pred = getattr(r, "predictive", None) if r is not None else None
        if pred is not None:
            return pred(n, rng)
        seed = int(rng.randint(0, 2**31 - 1) if hasattr(rng, "randint") else rng.integers(0, 2**31 - 1))
        return self.sample(n, seed=seed)

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
        """Whether any slot in this expression's tree carries a prior -- an RV in a *flat* family slot.

        Recurses into composite children (``Mix(...)`` components, etc.): a composite's own immediate
        children are sub-models, not priors themselves, but a child that is itself a flat family with a
        prior nested in *its* slot (e.g. ``Mix([Bernoulli(Beta(1,1)), ...])``) carries a real prior the
        auto-router and ``explain_fit()`` must see. This used to stop at the first ``CompositeFamily``
        and report "no priors" even when a nested child slot held one, so ``explain_fit()`` could claim
        route="em" for an expression whose actual ``.fit()`` raises ``NotImplementedError``.
        """
        if self._kind != "sample":
            return False
        if not isinstance(self._family, CompositeFamily):
            return any(isinstance(a, RandomVariable) for a in self._args)
        stack = list(self._args)
        while stack:
            a = stack.pop()
            if isinstance(a, RandomVariable):
                if a._has_priors():
                    return True
            elif isinstance(a, (list, tuple)):
                stack.extend(a)
        return False

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

    def _resolve_posterior_ladder(self, *, grouped):
        """Lowest-cost route that returns a *posterior* (uncertainty), not a point estimate -- the
        ``how='posterior'`` escalation ladder: conjugate (exact) -> Laplace (Gaussian at the MAP) -> MCMC.

        Unlike ``how='auto'`` (which stops at MAP for a non-conjugate prior and returns a point estimate),
        this always climbs to the lowest-cost route that yields posterior uncertainty, and reports which route.

        ``grouped`` mirrors ``_resolve_auto``'s own first check: a ``.each()`` group prior needs the
        random-effects (hierarchical) fit for its group-level posterior, the same as the auto ladder --
        without this check here, a grouped model fell through to the flat conjugate/Laplace/MCMC checks
        below and was silently fit as if it had no group structure at all.
        """
        from mixle.ppl import inference as _inf

        if grouped:
            return "hierarchical", "a .each() group prior -> random-effects (hierarchical) fit"
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
        from. The record travels with the model through pickling, so a reloaded artifact still
        explains how it was fit. Raises only when no record exists at all -- a bound model built
        directly rather than through ``.fit()``, or one written before the record was carried --
        in which case call ``explain_fit()`` on the pre-fit expression instead.
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
            _grouped_for_posterior = self._kind == "sample" and any(
                isinstance(a, RandomVariable) and a._scope == "grouped" for a in self._args
            )
            route, reason = self._resolve_posterior_ladder(grouped=_grouped_for_posterior)
        elif how != "auto":
            # STAT-RR18-03: fit() rejects unknown routes; explaining one with a plausible-looking
            # empty-caveat answer presented a route that cannot run as an accepted plan
            if how not in _FITTERS and how not in _ROUTE_CAVEATS:
                raise ValueError(f"unknown fit route {how!r}: fit() would reject it, so there is nothing to explain")
            route, reason = how, f"explicit how={how!r}"
        elif self._kind == "sample" and any(isinstance(a, _NeuralPredictor) for a in self._args):
            # STAT-RR18-02: fit() dispatches a Net/Conv/Transformer parameter slot to the neural
            # route BEFORE the auto ladder, and explain_fit used to miss this branch -- reporting
            # "MAP point estimate" for a fit that runs stochastic network training and returns a
            # NeuralResult with no explain_fit of its own
            route, reason = (
                "neural",
                "a neural predictor (Net/Conv/Transformer) in a parameter slot -> neural-conditional fit",
            )
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
        if self._kind == "sample" and isinstance(self._family, (Family, CompositeFamily)):
            # Revalidate at execution because fixed arrays/lists may have been mutated after the
            # symbolic model was created. Construction, fitting, lowering, and reconstruction share
            # this one family contract.
            self._family.validate_args(self._args)
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
        _parameter_dimension = _inferable_parameter_dimension(self)
        _group_receipt = None

        def _stash_explanation(rv):
            if _parameter_dimension is not None:
                rv._cache["_free_parameter_count"] = int(_parameter_dimension)
            if _group_receipt is not None:
                labels, mapping = _group_receipt
                rv._cache["_group_labels"] = tuple(labels)
                rv._cache["_group_index"] = dict(mapping)
            try:
                rv._cache["_fit_explanation"] = self.explain_fit(
                    how=_original_how, constraints=kw.get("constraints"), potentials=kw.get("potentials")
                )
            except Exception:  # noqa: BLE001 - best-effort; must never block a fit
                pass
            return rv

        structural_shortcut = (
            self._kind == "sample"
            and _parameter_dimension == 0
            and not any(_expr_has_gather(argument) for argument in self._args)
            and not any(isinstance(argument, (_LinearPredictor, _NeuralPredictor)) for argument in self._args)
            and not (isinstance(self._family, CompositeFamily) and self._family.fit_fn is not None)
        )
        if structural_shortcut:
            if how not in {"auto", "em"}:
                raise ValueError(f"how={how!r} requested inference, but the model declares no inferable parameters.")
            if missing != "error":
                raise NotImplementedError("a fully specified model has no missing-data fitting route.")
            if backend != "local" or num_workers is not None or engine is not None or precision is not None:
                raise ValueError("a fully specified model performs no distributed or precision-planned fit.")
            if print_iter != 0:
                raise ValueError("a fully specified model performs no iterative fit and cannot consume print_iter.")
            if kw:
                raise ValueError(f"a fully specified model cannot consume fit option(s) {sorted(kw)}.")
            concrete = lower(self, target="dist")
            return _stash_explanation(RandomVariable._bound(concrete, name=self._name))

        if self._kind == "sample" and any(_expr_has_gather(a) for a in self._args):
            # A data-indexed latent (theta[Field("g")]) makes the parameter per-observation -> the
            # per-observation (indexed) target.
            from mixle.ppl import inference as _inf

            if missing != "error":
                raise NotImplementedError("indexed latent fitting does not support missing='marginalize'.")
            if backend != "local" or num_workers is not None or engine is not None or precision is not None:
                raise NotImplementedError("indexed latent fitting currently supports only local execution.")
            if print_iter != 0:
                raise NotImplementedError("indexed latent fitting does not implement print_iter.")
            allowed = {"given", "rng", "draws", "burn", "thin", "constraints", "penalty", "potentials"}
            unknown = sorted(set(kw) - allowed)
            if unknown:
                raise TypeError(f"unsupported indexed fit control(s): {', '.join(unknown)}")
            return _stash_explanation(_inf.indexed_fit(self, data, how=how, max_iter=max_its, tol=delta, **kw))

        # regression / GLM: a linear predictor (covariates) in a parameter slot
        if self._kind == "sample" and any(isinstance(a, _LinearPredictor) for a in self._args):
            from mixle.ppl import regression as _reg

            if missing != "error":
                raise NotImplementedError("regression fitting does not support missing='marginalize'.")
            if backend != "local" or num_workers is not None or engine is not None or precision is not None:
                raise NotImplementedError("regression fitting currently supports only local execution.")
            if print_iter != 0:
                raise NotImplementedError("regression fitting does not implement print_iter.")
            allowed = {"given", "inner_max_iter", "l2", "max_iter", "quantile", "tol"}
            unknown = sorted(set(kw) - allowed)
            if unknown:
                raise TypeError(f"unsupported regression fit control(s): {', '.join(unknown)}")
            regression_options = dict(kw)
            regression_options.setdefault("max_iter", max_its)
            regression_options.setdefault("tol", delta)
            return _stash_explanation(_reg.regression_fit(self, data, how=how, **regression_options))

        # neural conditional: a Net/Conv (nonlinear predictor) in a parameter slot -> a neural-headed leaf
        if self._kind == "sample" and any(isinstance(a, _NeuralPredictor) for a in self._args):
            from mixle.ppl import neural as _neu

            if how != "auto":
                raise NotImplementedError("neural conditional fitting currently supports how='auto' only.")
            if missing != "error":
                raise NotImplementedError("neural conditional fitting does not support missing='marginalize'.")
            if backend != "local" or num_workers is not None or engine is not None or precision is not None:
                raise NotImplementedError("neural conditional fitting currently supports only local execution.")
            if print_iter != 0:
                raise NotImplementedError("neural conditional fitting does not implement print_iter.")
            if delta != 1e-8:
                raise NotImplementedError("neural conditional fitting does not implement a delta tolerance.")
            allowed = {"batch_size", "device", "epochs", "ewc", "given", "init", "lr", "weights"}
            unknown = sorted(set(kw) - allowed)
            if unknown:
                raise TypeError(f"unsupported neural fit control(s): {', '.join(unknown)}")
            neural_options = dict(kw)
            neural_options.setdefault("epochs", max_its)
            result = _neu.neural_fit(self, data, **neural_options)
            result.fit_request = {
                "route": "neural",
                "how": "auto",
                "epochs": int(neural_options["epochs"]),
                "missing": missing,
                "backend": backend,
            }
            return result

        # Composite families with a bespoke fitter (state-space Kalman/RTS+EM, PDE-constrained fields)
        # own their fit through the registered fit_fn hook -- no per-family branch in core. State-space
        # is registered in mixle.ppl.statespace; PDEStateSpace by the mixle-pde plugin.
        if self._kind == "sample" and isinstance(self._family, CompositeFamily) and self._family.fit_fn is not None:
            bespoke_options = {
                "how": how,
                "max_its": max_its,
                "delta": delta,
                "missing": missing,
                "backend": backend,
                "num_workers": num_workers,
                "engine": engine,
                "precision": precision,
                "print_iter": print_iter,
                **kw,
            }
            return _stash_explanation(self._family.fit_fn(self, data, **bespoke_options))

        # Indexed-flat hierarchical: Normal(Normal(m, t).each(by="g"), s).fit(y, given={"g": labels}).
        # Reshape the flat observation array into per-group lists (sorted unique labels) so the existing
        # nested grouped path handles it -- the model is identical, only the data layout differs.
        if self._kind == "sample":
            _gby = next(
                (a._group_by for a in self._args if isinstance(a, RandomVariable) and a._scope == "grouped"),
                None,
            )
            if _gby is not None:
                given = kw.pop("given", None)
                if not given or _gby not in given:
                    raise ValueError(f"each(by={_gby!r}) needs the group index: fit(..., given={{{_gby!r}: labels}}).")
                data = list(data)
                group_ids, group_labels, group_index = _indexed_group_layout(given[_gby], len(data))
                if any(k != _gby for k in given):
                    raise NotImplementedError("indexed-flat hierarchical with extra covariates is not supported yet.")
                yarr = np.asarray(data, dtype=float)
                data = [yarr[group_ids == group].tolist() for group in range(len(group_labels))]
                _group_receipt = (group_labels, group_index)

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
            how, _ = self._resolve_posterior_ladder(grouped=grouped)
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
            if how in {"map", "mcmc", "hmc", "nuts", "laplace", "vi", "ensemble", "sample"}:
                kw["missing"] = missing
            else:
                raise NotImplementedError(
                    f"missing='marginalize' is not wired for how={how!r} (closed-form/grouped path); use "
                    "how='em'/'map'/'mcmc'/'hmc'/'nuts'/'vi'/'ensemble', or build the model with "
                    "mixle.stats.marginalized() leaves."
                )
        fitter = _FITTERS.get(how)
        if fitter is not None:
            if how in {"map", "laplace"}:
                if max_its != 100:
                    kw.setdefault("max_iter", max_its)
                if delta != 1e-8:
                    kw.setdefault("tol", delta)
            result = fitter(self, data, **kw)
            if has_constraints or has_potentials:
                # A penalized objective (soft constraints / residual factors / potentials) means the
                # optimum is for the surrogate, not the likelihood, so downgrade the certificate.
                try:
                    from mixle.inference.planning import certify as _certify

                    target = getattr(result, "_dist", None) or getattr(result, "dist", None) or result
                    why = "soft constraints" if has_constraints else "custom potential"
                    result._cache["certificate"] = _certify(target, data=data, penalized=why)
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


def _exact_positive_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an exact positive integer, got {value!r}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{label} must be an exact positive integer, got {value!r}.")
    return value


def _param_handle(dim: int, *, name=None, kind: str = "vector", support: str = "real") -> RandomVariable:
    """Build a referenceable vector/matrix parameter handle (the result of calling ``free(...)``).

    ``kind``: ``vector`` (entries on ``support`` real/positive/unit), ``ordered`` (increasing),
    ``simplex`` (sums to 1), or ``cholesky`` (an SPD covariance). Placed in a constructor slot and
    referenced in constraints — ``m = free(3, name="mu"); MVN(3, mean=m).fit(X, constraints=increasing(m))``.
    The handle behaves like a vector RV in constraint expressions (``m[i]``, ``m[0] < m[1]``, ...).
    """
    dim = _exact_positive_int(dim, "free dimension")
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
    """Rebuild the legacy, structure-only RandomVariable pickle format."""
    if kind == "bound":
        return RandomVariable._bound(dist, name=name)
    family = _FAMILIES[fam_name] if fam_name is not None else None
    if kind == "sample" and isinstance(family, (Family, CompositeFamily)):
        family.validate_args(tuple(args))
    return RandomVariable(kind, family=family, args=args, name=name, keys=keys, scope=scope, reparam=reparam)


def _rv_reconstruct_v2(
    version,
    kind,
    fam_name,
    args,
    name,
    keys,
    dist,
    result,
    scope,
    reparam,
    group_by,
    durable_cache,
):
    """Rebuild a versioned RandomVariable artifact with fitted and structural semantics intact."""
    if version != 1:
        raise ValueError(f"unsupported RandomVariable artifact version {version!r}.")
    if not isinstance(durable_cache, dict):
        raise TypeError("RandomVariable artifact cache must be a dictionary.")
    if kind == "bound":
        rv = RandomVariable._bound(dist, name=name, result=result)
    else:
        family = _FAMILIES[fam_name] if fam_name is not None else None
        if kind == "sample" and isinstance(family, (Family, CompositeFamily)):
            family.validate_args(tuple(args))
        rv = RandomVariable(
            kind,
            family=family,
            args=args,
            name=name,
            keys=keys,
            dist=dist,
            result=result,
            scope=scope,
            reparam=reparam,
            group_by=group_by,
        )
    rv._cache.update(durable_cache)
    return rv


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
        fam.validate_args(rv._args)
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
