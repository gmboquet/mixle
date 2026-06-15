"""Core of the pysp.ppl probabilistic-programming surface.

One immutable wrapper type, :class:`RandomVariable`, sits over pysparkplug's existing
distribution / estimator / sampler objects. It adds *no* inference engine: every call
lowers (one routing site, :func:`lower`) to machinery that already exists and then
dispatches. See notes/ppl-syntax-spec.md for the design charter and invariants.

Slice 1-2 of the build order: the wrapper + ``free`` holes + ``fit`` (EM core).
Algebra, conditioning, latent-arg Compound, and MCMC routing land in later slices.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from pysp.utils.estimation import optimize

__all__ = ["RandomVariable", "free", "lower", "register_family", "Family",
           "Constraint", "Event", "constrain"]


# --------------------------------------------------------------------------- free
class _Free:
    """The single ``free`` token: an argument slot to be estimated (MLE).

    In IR terms it is ``Var(prior=flat, scope=shared, policy=point)``. It is a
    singleton; identity (``arg is free``) is the test used during lowering.
    """

    __slots__ = ()

    def __mul__(self, other):       # free * Field -> an OLS regression coefficient
        if isinstance(other, Field):
            return _LinearPredictor([(self, other)])
        return NotImplemented
    __rmul__ = __mul__

    def __reduce__(self):           # preserve singleton identity across pickling
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

    def __mul__(self, coef):     # Field * coef
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
        self.terms = list(terms)         # list of (coef, Field)
        self.intercept = intercept       # RandomVariable | free | float | None
        self.groups = list(groups or [])  # random-intercept group names

    def __add__(self, other):
        if isinstance(other, _LinearPredictor):
            return _LinearPredictor(self.terms + other.terms,
                                    _combine_intercept(self.intercept, other.intercept),
                                    self.groups + other.groups)
        if isinstance(other, Field):
            return _LinearPredictor(self.terms + [(1.0, other)], self.intercept, self.groups)
        if isinstance(other, Group):
            return _LinearPredictor(self.terms, self.intercept, self.groups + [other._key()])
        return _LinearPredictor(self.terms, _combine_intercept(self.intercept, other), self.groups)
    __radd__ = __add__

    def __repr__(self):
        return (f"_LinearPredictor({self.terms!r}, intercept={self.intercept!r}, "
                f"groups={self.groups!r})")


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

    __slots__ = ("leaves", "pred", "desc")

    def __init__(self, leaves, pred, desc):
        self.leaves = tuple(leaves)
        self.pred = pred           # env: {leaf_rv -> value(s)} -> bool mask
        self.desc = desc

    @property
    def rv(self):
        """The single RV this constraint restricts (back-compat for one-variable events)."""
        if len(self.leaves) != 1:
            raise AttributeError("constraint involves multiple RVs; use .leaves.")
        return self.leaves[0]

    def eval(self, env):
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
        return Constraint(self._merge_leaves(other),
                          lambda env: self.pred(env) & other.pred(env),
                          f"({self.desc} & {other.desc})")

    def __or__(self, other):
        return Constraint(self._merge_leaves(other),
                          lambda env: self.pred(env) | other.pred(env),
                          f"({self.desc} | {other.desc})")

    def __invert__(self):
        return Constraint(self.leaves, lambda env: ~np.asarray(self.pred(env)), f"~{self.desc}")

    def __bool__(self):
        raise TypeError(
            "a Constraint has no truth value — Python chained comparisons (a < b < c) and "
            "`and`/`or` are not supported; combine with & | ~ instead, e.g. (a < b) & (b < c).")

    def __repr__(self):
        return f"Constraint({self.desc})"


Event = Constraint   # back-compat alias


def _expr_leaves(rv) -> list:
    """The leaf (sample/bound) RVs an expression RV depends on, in left-to-right order."""
    if not isinstance(rv, RandomVariable):
        return []
    if rv._kind in ("apply",):
        return _expr_leaves(rv._args[0])
    if rv._kind == "sum":
        out = _expr_leaves(rv._args[0])
        seen = {id(x) for x in out}
        for lv in _expr_leaves(rv._args[1]):
            if id(lv) not in seen:
                out.append(lv)
                seen.add(id(lv))
        return out
    return [rv]    # sample / bound / given: an atomic leaf


def _eval_expr(rv, env):
    """Numerically evaluate an expression RV given ``env`` (leaf RV -> value)."""
    if not isinstance(rv, RandomVariable):
        return rv      # a constant
    if rv._kind == "apply":
        base, transform = rv._args
        return transform.forward(_eval_expr(base, env))
    if rv._kind == "sum":
        a, b = rv._args
        return _eval_expr(a, env) + _eval_expr(b, env)
    if rv not in env:
        raise KeyError(f"no value supplied for {rv!r} when evaluating a constraint.")
    return env[rv]


_CMP = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}


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

    return Constraint(leaves, pred, f"{_expr_desc(lhs)} {op} {_expr_desc(rhs)}")


def _expr_desc(rv) -> str:
    if not isinstance(rv, RandomVariable):
        return repr(rv)
    if rv._kind == "apply":
        return f"f({_expr_desc(rv._args[0])})"
    if rv._kind == "sum":
        return f"({_expr_desc(rv._args[0])} + {_expr_desc(rv._args[1])})"
    return rv._name or "rv"


def _convolve(da, db):
    """Closed-form distribution of da + db for independent operands, or None."""
    ta, tb = type(da).__name__, type(db).__name__
    if ta == tb == "GaussianDistribution":
        from pysp.stats.leaf.gaussian import GaussianDistribution
        return GaussianDistribution(da.mu + db.mu, da.sigma2 + db.sigma2)
    if ta == tb == "PoissonDistribution":
        from pysp.stats.leaf.poisson import PoissonDistribution
        return PoissonDistribution(da.lam + db.lam)
    if ta == tb == "GammaDistribution" and abs(da.theta - db.theta) < 1e-12:
        from pysp.stats.leaf.gamma import GammaDistribution
        return GammaDistribution(da.k + db.k, da.theta)        # same scale
    return None


# ------------------------------------------------------------------------- family
class Family:
    """Lowering recipe for one distribution family.

    Keeps the alias namespace and the engine objects in one place so the wrapper
    never hard-codes a distribution. ``to_dist`` maps user-facing (conventional)
    arguments to the underlying ``*Distribution`` kwargs; ``make_estimator`` builds
    the paired ``*Estimator`` for the all-``free`` case.
    """

    __slots__ = ("name", "dist_cls", "est_cls", "to_dist", "arity", "seed_at", "positive",
                 "init_fit", "read", "support")

    def __init__(self, name, dist_cls, est_cls, to_dist, arity, seed_at=None, positive=None,
                 init_fit=None, read=None, support=None):
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
        self.support = (tuple(support) if support is not None
                        else tuple("positive" if p else "real" for p in self.positive))
        # init_fit(data) -> a concrete Distribution to warm-start EM for families whose
        # MLE is sensitive to initialization (e.g. negative-binomial dispersion).
        self.init_fit = init_fit
        # read(dist) -> {conventional param name: value}: the inverse of construction, so
        # fitted params come back in the *same* parameterization the user wrote (sd, not sigma2).
        self.read = read

    def make_dist(self, args: tuple[Any, ...], name: str | None):
        kwargs = self.to_dist(*args)
        if name is not None:
            kwargs.setdefault("name", name)
        return self.dist_cls(**kwargs)

    def make_estimator(self, name: str | None, keys: str | None):
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

    __slots__ = ("name", "dist_fn", "est_fn", "seed_fn", "read")

    def __init__(self, name, dist_fn, est_fn, seed_fn=None, read=None):
        self.name = name
        self.dist_fn = dist_fn
        self.est_fn = est_fn
        # seed_fn(args, data, rng, seed_child) -> a concrete initial Distribution that
        # breaks EM symmetry (e.g. mixture components at distinct data points).
        self.seed_fn = seed_fn
        # read(dist, read_params) -> structured params in PPL vocabulary, recursing into
        # children via read_params (so the whole read surface is leak-free).
        self.read = read


_FAMILIES: dict[str, Any] = {}
_DIST_TO_FAMILY: dict[type, Family] = {}        # reverse map for reading fitted params
_DIST_TO_COMPOSITE_READ: dict[type, Any] = {}     # composite dist type -> read(dist, read_params)


def register_family(name, dist_cls, est_cls, to_dist, arity, seed_at=None, positive=None,
                    init_fit=None, read=None, support=None) -> Family:
    fam = Family(name, dist_cls, est_cls, to_dist, arity, seed_at=seed_at, positive=positive,
                 init_fit=init_fit, read=read, support=support)
    _FAMILIES[name] = fam
    _DIST_TO_FAMILY[dist_cls] = fam
    return fam


def register_composite(name, dist_fn, est_fn, seed_fn=None, dist_cls=None, read=None) -> CompositeFamily:
    fam = CompositeFamily(name, dist_fn, est_fn, seed_fn=seed_fn, read=read)
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
    ('aic' | 'bic' | 'loglik')."""
    rows = []
    for m in models:
        ll = m.log_likelihood(data)
        rows.append({"model": (m.name or type(m.dist).__name__), "loglik": ll,
                     "aic": m.aic(data), "bic": m.bic(data)})
    keys = {"loglik": lambda r: -r["loglik"], "aic": lambda r: r["aic"], "bic": lambda r: r["bic"]}
    return sorted(rows, key=keys[by])


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
            from pysp.stats.leaf.categorical import CategoricalDistribution
            spec = rv._args[0]
            keys = list(spec.keys()) if isinstance(spec, dict) else list(range(len(spec)))
            w = rng.dirichlet(np.ones(len(keys)))            # random, valid (no zeros->inf)
            return CategoricalDistribution(pmap=dict(zip(keys, w)))
    return None


# ----------------------------------------------------------------- RandomVariable
class RandomVariable:
    """The single user-facing PPL type (immutable).

    Two states: ``sample`` (a symbolic draw: a family + argument expressions, some of
    which may be ``free``) and ``bound`` (wraps a concrete fitted distribution). The
    verb surface is fixed and state-independent (invariant I3); validity depends on
    state. Construct via the family functions in :mod:`pysp.ppl` or ``fit``.
    """

    __slots__ = ("_kind", "_family", "_args", "_name", "_keys", "_dist", "_result",
                 "_cache", "_scope")

    def __init__(self, kind, *, family=None, args=(), name=None, keys=None, dist=None,
                 result=None, scope="shared"):
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

    def __setattr__(self, *a):  # enforce immutability (I2)
        raise AttributeError("RandomVariable is immutable")

    def __reduce__(self):
        # Picklable so models can cross a process boundary (parallel chains,
        # distributed fits). Families live in the module-level registry and are
        # restored by name; transient _result/_cache are dropped.
        fam_name = self._family.name if self._family is not None else None
        return (_rv_reconstruct,
                (self._kind, fam_name, self._args, self._name, self._keys,
                 self._dist, self._scope))

    # -- constructors -------------------------------------------------------
    @classmethod
    def _sample(cls, family_name, args, *, name=None, keys=None, scope="shared") -> RandomVariable:
        fam = _FAMILIES[family_name]
        return cls("sample", family=fam, args=args, name=name, keys=keys, scope=scope)

    def each(self) -> RandomVariable:
        """Mark this prior as per-group (a random effect / local latent). Used in a
        parameter slot: ``Normal(Normal(m, t).each(), s)`` is a hierarchical model.
        """
        if self._kind != "sample":
            raise TypeError("each() applies to a distribution used as a prior.")
        return RandomVariable("sample", family=self._family, args=self._args,
                              name=self._name, keys=self._keys, scope="grouped")

    @property
    def scope(self) -> str:
        return self._scope

    @classmethod
    def _bound(cls, dist, *, name=None, result=None) -> RandomVariable:
        return cls("bound", dist=dist, name=name or getattr(dist, "name", None), result=result)

    @classmethod
    def _apply(cls, base, transform) -> RandomVariable:
        # Apply node: a deterministic transform of one RV (algebra rung 1).
        return cls("apply", args=(base, transform))

    @classmethod
    def _sum(cls, a, b) -> RandomVariable:
        # Convolution node: the distribution of a + b for independent a, b.
        return cls("sum", args=(a, b))

    # -- algebra (deterministic transforms + convolution) -------------------
    def _affine(self, loc, scale) -> RandomVariable:
        from pysp.stats.combinator.transform import AffineTransform
        return RandomVariable._apply(self, AffineTransform(loc=float(loc), scale=float(scale)))

    def __mul__(self, c):
        if isinstance(c, Field):                      # coef * covariate -> regression term
            return _LinearPredictor([(self, c)])
        if isinstance(c, RandomVariable):
            raise NotImplementedError("RV * RV (products) is not supported; use sums (+).")
        return self._affine(0.0, c)
    __rmul__ = __mul__

    def __add__(self, c):
        if isinstance(c, _LinearPredictor):           # RV is an intercept
            return c.__add__(self)
        if isinstance(c, Field):
            return _LinearPredictor([(1.0, c)], self)
        if isinstance(c, Group):                      # RV intercept + random group effects
            return _LinearPredictor([], self, [c._key()])
        if isinstance(c, RandomVariable):             # convolution of independent RVs
            return RandomVariable._sum(self, c)
        return self._affine(c, 1.0)
    __radd__ = __add__

    def __sub__(self, c):
        if isinstance(c, RandomVariable):
            return RandomVariable._sum(self, c._affine(0.0, -1.0))   # a + (-b)
        return self._affine(-float(c), 1.0)

    def __rsub__(self, c):
        return self._affine(c, -1.0) if not isinstance(c, RandomVariable) else c.__sub__(self)

    def __truediv__(self, c):
        if isinstance(c, RandomVariable):
            raise NotImplementedError("RV / RV (ratios) is not supported.")
        return self._affine(0.0, 1.0 / float(c))

    def __neg__(self):
        return self._affine(0.0, -1.0)

    def exp(self) -> RandomVariable:
        from pysp.stats.combinator.transform import ExpTransform
        return RandomVariable._apply(self, ExpTransform())

    def log(self) -> RandomVariable:
        from pysp.stats.combinator.transform import LogTransform
        return RandomVariable._apply(self, LogTransform())

    # -- relations: comparisons build Constraints (RV vs constant / RV / linear expr) ----
    def __gt__(self, other): return _make_constraint(self, ">", other)
    def __ge__(self, other): return _make_constraint(self, ">=", other)
    def __lt__(self, other): return _make_constraint(self, "<", other)
    def __le__(self, other): return _make_constraint(self, "<=", other)

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
                "involves other RVs — use constrain(constraint) for a joint conditioning.")
        return RandomVariable("given", args=(self, constraint))

    # -- introspection ------------------------------------------------------
    @property
    def is_bound(self) -> bool:
        return self._kind == "bound"

    @property
    def has_free(self) -> bool:
        return self._kind == "sample" and any(_is_free(a) for a in self._args)

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def columns(self) -> list:
        """For a ``constrain(...)`` joint RV: the variable names, in sample-column order."""
        if self._kind != "joint":
            raise TypeError("columns is only defined for a constrain(...) RV.")
        leaves = self._args[0]
        return [lv._name or f"rv{i}" for i, lv in enumerate(leaves)]

    @property
    def dist(self):
        """The lowered concrete distribution — the full original pysp API (escape hatch)."""
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
        elif hasattr(d, "dist") and not hasattr(d, "mu"):   # SequenceDistribution.dist
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
            return self._result.coefficients          # regression: report coefficients
        return read_params(d)

    @property
    def result(self):
        """Inference metadata (EM history / MCMC chain) when present; else None."""
        return self._result

    # -- query verbs (valid once concrete) ----------------------------------
    def sample(self, n: int | None = None, seed: int | None = None):
        if self._kind == "joint":                     # joint rejection sampling under a relation
            leaves, constraint = self._args
            rng = np.random.RandomState(seed)
            k = n if n is not None else 1
            kept = []
            have = 0
            while have < k:
                batch = max(k * 2, 1024)
                cols = {lv: np.asarray(lv.sample(batch, seed=int(rng.randint(1, 2 ** 31))),
                                      dtype=float) for lv in leaves}
                mask = np.asarray(constraint.eval(cols))
                block = np.stack([cols[lv][mask] for lv in leaves], axis=1)   # (m, K)
                kept.append(block)
                have += len(block)
            out = np.concatenate(kept, axis=0)[:k]
            return out if n is not None else out[0]
        if self._kind == "sum":                       # convolution: sample operands and add
            rng = np.random.RandomState(seed)
            a, b = self._args
            k = n if n is not None else 1
            xs = np.asarray(a.sample(k, seed=int(rng.randint(1, 2**31))))
            ys = np.asarray(b.sample(k, seed=int(rng.randint(1, 2**31))))
            out = xs + ys
            return out if n is not None else float(out[0])
        if self._kind == "given":                     # rejection sampling from the region
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
        if self._kind == "joint":                     # joint density of independent leaves / Z
            leaves, constraint = self._args
            xa = np.atleast_2d(np.asarray(x, dtype=float))
            if xa.shape[1] != len(leaves):
                raise ValueError(f"expected {len(leaves)} columns, got shape {xa.shape}.")
            logZ = math.log(self.prob())
            env = {lv: xa[:, j] for j, lv in enumerate(leaves)}
            base_lp = sum(np.atleast_1d(lv.log_prob(xa[:, j])) for j, lv in enumerate(leaves))
            out = np.where(np.asarray(constraint.eval(env)), base_lp - logZ, -np.inf)
            return float(out[0]) if np.ndim(x) == 1 else out
        if self._kind == "sum":                       # exact convolution if closed-form, else KDE
            a, b = self._args
            cd = None
            try:
                cd = _convolve(lower(a, target="dist"), lower(b, target="dist"))
            except Exception:
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
            cols = {lv: np.asarray(lv.sample(samples, seed=int(rng.randint(1, 2 ** 31))),
                                  dtype=float) for lv in leaves}
            p = max(float(np.mean(np.asarray(constraint.eval(cols)))), 1e-9)
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
        if isinstance(x, (RandomVariable, str, int)) and self._result is not None \
                and hasattr(self._result, "samples"):
            return self._result.samples(x)
        d = lower(self, target="dist")
        if not (hasattr(d, "seq_posterior") or hasattr(d, "posterior")):
            raise NotImplementedError(
                f"{type(d).__name__} exposes no posterior (no latent to infer)."
            )
        if np.isscalar(x):
            return np.asarray(d.posterior(x))
        enc = d.dist_to_encoder().seq_encode(list(x))
        if hasattr(d, "seq_posterior"):
            return np.asarray(d.seq_posterior(enc))
        return np.asarray([d.posterior(xi) for xi in x])

    # -- resolve ------------------------------------------------------------
    def _has_priors(self) -> bool:
        # A prior is an RV in a *flat* family slot; composite children are sub-models.
        return (self._kind == "sample"
                and not isinstance(self._family, CompositeFamily)
                and any(isinstance(a, RandomVariable) for a in self._args))

    def fit(self, data: Sequence[Any], *, how: str = "auto", max_its: int = 100,
            delta: float = 1e-8, backend: str = "local", num_workers: int | None = None,
            engine: Any = None, precision: Any = None, print_iter: int = 0,
            **kw) -> RandomVariable:
        """Estimate / infer parameters from ``data`` and return a bound RV.

        ``how``: ``'em'`` (EM/MLE, default for plain ``free`` models), ``'map'`` (maximize
        the joint with priors), ``'mcmc'`` (posterior samples over parameters with priors),
        ``'auto'`` picks ``map`` when the model has priors else ``em``. EM threads pysp's
        parallel/distributed backends (``backend='mp'|'mpi'|'dask'``).
        """
        valid_how = {"auto", "em", "map", "mcmc", "hmc", "ensemble", "vi", "vmp", "conjugate",
                     "conjugate_mixture", "hierarchical"}
        if how not in valid_how:
            raise ValueError(f"unknown how={how!r}; choose from {sorted(valid_how)}.")
        if hasattr(data, "__len__") and len(data) == 0:
            raise ValueError("fit() received empty data.")

        # regression / GLM: a linear predictor (covariates) in a parameter slot
        if self._kind == "sample" and any(isinstance(a, _LinearPredictor) for a in self._args):
            from pysp.ppl import regression as _reg
            return _reg.regression_fit(self, data, **kw)

        # state-space time-series models (Kalman/RTS + EM)
        if self._kind == "sample" and isinstance(self._family, CompositeFamily) \
                and self._family.name == "StateSpace":
            from pysp.ppl import statespace as _ss
            return _ss.statespace_fit(self, data, **kw)

        grouped = self._kind == "sample" and any(
            isinstance(a, RandomVariable) and a._scope == "grouped" for a in self._args)
        # partial-free: a flat model with some `free` slots and some fixed constants (no priors).
        # The all-free EM estimator can't hold params fixed, so fit only the free slots by
        # maximum likelihood (MAP with no prior term), with the fixed args held constant.
        flat = self._kind == "sample" and not isinstance(self._family, CompositeFamily)
        partial_free = (flat and not self._has_priors()
                        and any(_is_free(a) for a in self._args)
                        and not all(_is_free(a) for a in self._args))
        has_constraints = kw.get("constraints") is not None
        if has_constraints and how in ("em", "conjugate", "conjugate_mixture", "vi", "vmp"):
            raise ValueError(
                f"how={how!r} cannot honor inequality constraints; use 'map', 'mcmc', 'hmc', "
                "or 'ensemble' (or how='auto').")
        if how == "auto":
            if grouped:
                how = "hierarchical"
            elif has_constraints:
                how = "map"      # constraints truncate the region; the conjugate paths can't
            elif self._has_priors():
                from pysp.ppl import inference as _inf
                if _inf.conjugate_spec(self) is not None:
                    how = "conjugate"
                elif _inf.conjugate_mixture_spec(self) is not None:
                    how = "conjugate_mixture"
                else:
                    how = "map"
            elif partial_free:
                how = "map"
            else:
                how = "em"
        elif how == "em" and partial_free:
            how = "map"   # EM cannot hold some params fixed; MLE the free slots instead
        if how in ("map", "mcmc", "hmc", "ensemble", "vi", "vmp", "conjugate",
                   "conjugate_mixture", "hierarchical"):
            from pysp.ppl import inference as _inf
            if how == "conjugate_mixture":
                return _inf.conjugate_mixture_fit(self, data, **kw)
            if how == "mcmc":
                return _inf.mcmc_fit(self, data, **kw)
            if how == "hmc":
                return _inf.hmc_fit(self, data, **kw)
            if how == "ensemble":
                return _inf.ensemble_fit(self, data, **kw)
            if how == "vi":
                return _inf.vi_fit(self, data, **kw)
            if how == "vmp":
                from pysp.ppl import vmp as _vmp
                if isinstance(self._family, CompositeFamily) and self._family.name == "Mixture":
                    comps = self._args[0]
                    return _vmp.mixture_vmp(data, len(comps), **kw)
                return _vmp.vmp_fit(self, data, **kw)
            if how == "conjugate":
                return _inf.conjugate_fit(self, data, **kw)
            if how == "hierarchical":
                return _inf.hierarchical_fit(self, data, **kw)
            return _inf.map_fit(self, data, **kw)
        # EM / MLE path
        est = lower(self, target="estimator")
        # Warm-start finicky flat-family MLEs (e.g. negative-binomial) from a moment match.
        if "prev_estimate" not in kw and self._kind == "sample" \
                and not isinstance(self._family, CompositeFamily) \
                and getattr(self._family, "init_fit", None) is not None:
            seed = self._family.init_fit(data)
            if seed is not None:
                kw["prev_estimate"] = seed
        # Auto-seed latent composites (mixtures, ...) at distinct data points so EM
        # escapes the symmetric global-mean fixed point — "it just works".
        if "prev_estimate" not in kw and self._kind == "sample" \
                and isinstance(self._family, CompositeFamily) and self._family.seed_fn is not None:
            import numpy as _np
            rng = kw.get("rng") or _np.random.RandomState()
            seed = self._family.seed_fn(self._args, data, rng, seed_child)
            if seed is not None:
                kw["prev_estimate"] = seed
        import sys
        out = open("/dev/null", "w") if not print_iter else sys.stdout
        try:
            fitted = optimize(
                data, est, max_its=max_its, delta=delta,
                backend=backend, num_workers=num_workers,
                engine=engine, precision=precision,
                print_iter=max(print_iter, 1), out=out, **kw,
            )
        finally:
            if not print_iter:
                out.close()
        return RandomVariable._bound(fitted, name=self._name)

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
                "parameters), not models with `free` holes; fit those first.")
    return RandomVariable("joint", args=(leaves, combined), name=None)


def _rv_reconstruct(kind, fam_name, args, name, keys, dist, scope):
    """Rebuild a RandomVariable from its picklable structural fields."""
    if kind == "bound":
        return RandomVariable._bound(dist, name=name)
    family = _FAMILIES[fam_name] if fam_name is not None else None
    return RandomVariable(kind, family=family, args=args, name=name, keys=keys, scope=scope)


# -------------------------------------------------------------------- the lowering
def lower(rv: RandomVariable, *, target: str = "dist"):
    """The one routing site: symbolic RandomVariable -> existing pysp object.

    ``target='dist'`` returns a concrete ``*Distribution`` (needs no ``free`` holes);
    ``target='estimator'`` returns a ``*Estimator``. Results are cached per RV (I7).
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
        from pysp.stats.combinator.transform import TransformDistribution
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
            result = fam.est_fn(rv._args, lambda c: lower(c, target="estimator"),
                                rv._name, rv._keys)
        else:
            raise ValueError(f"unknown lowering target {target!r}")
        cache[target] = result
        return result

    if target == "estimator":
        if not all(_is_free(a) for a in rv._args):
            if any(_is_free(a) for a in rv._args):
                raise NotImplementedError(
                    f"{fam.name}: partial `free` (some args fixed) is a later slice; "
                    "use all-free or all-fixed for now."
                )
            # No holes: estimator of a fully-specified model is its own estimator.
            est = lower(rv, target="dist").estimator()
        else:
            est = fam.make_estimator(rv._name, rv._keys)
        cache[target] = est
        return est

    if target == "dist":
        if any(_is_free(a) for a in rv._args):
            raise ValueError(
                f"{fam.name} has unresolved `free` parameters; call .fit(data) first."
            )
        if any(isinstance(a, RandomVariable) for a in rv._args):
            raise NotImplementedError(
                "latent/random parameters (a distribution in a slot) land in build slice 5."
            )
        d = fam.make_dist(rv._args, rv._name)
        cache[target] = d
        return d

    raise ValueError(f"unknown lowering target {target!r}")
