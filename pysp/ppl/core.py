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
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from pysp.utils.estimation import optimize

__all__ = ["RandomVariable", "free", "lower", "register_family", "Family"]


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

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "free"


free = _Free()


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


class _LinearPredictor:
    """A linear predictor Σ coef_k · Field_k (+ intercept). Coeffs are RVs (Gaussian
    priors), ``free`` (OLS), or constants. Built by operator overloading."""

    __slots__ = ("terms", "intercept")

    def __init__(self, terms, intercept=None):
        self.terms = list(terms)         # list of (coef, Field)
        self.intercept = intercept       # RandomVariable | free | float | None

    def __add__(self, other):
        if isinstance(other, _LinearPredictor):
            return _LinearPredictor(self.terms + other.terms,
                                    _combine_intercept(self.intercept, other.intercept))
        if isinstance(other, Field):
            return _LinearPredictor(self.terms + [(1.0, other)], self.intercept)
        return _LinearPredictor(self.terms, _combine_intercept(self.intercept, other))
    __radd__ = __add__

    def __repr__(self):
        return f"_LinearPredictor({self.terms!r}, intercept={self.intercept!r})"


class Event:
    """A predicate over an RV's value, produced by comparisons (``x > 0``) and consumed by
    ``rv.given(event)``. Combine with ``&``."""

    __slots__ = ("rv", "pred", "desc")

    def __init__(self, rv, pred, desc):
        self.rv = rv
        self.pred = pred           # value(s) -> bool mask
        self.desc = desc

    def contains(self, x):
        return self.pred(x)

    def __and__(self, other):
        return Event(self.rv, lambda x: self.pred(x) & other.pred(x), f"({self.desc} & {other.desc})")

    def __repr__(self):
        return f"Event({self.desc})"


def _convolve(da, db):
    """Closed-form distribution of da + db for independent operands, or None."""
    ta, tb = type(da).__name__, type(db).__name__
    if ta == tb == "GaussianDistribution":
        from pysp.stats.gaussian import GaussianDistribution
        return GaussianDistribution(da.mu + db.mu, da.sigma2 + db.sigma2)
    if ta == tb == "PoissonDistribution":
        from pysp.stats.poisson import PoissonDistribution
        return PoissonDistribution(da.lam + db.lam)
    if ta == tb == "GammaDistribution" and abs(da.theta - db.theta) < 1e-12:
        from pysp.stats.gamma import GammaDistribution
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
                 "init_fit", "read")

    def __init__(self, name, dist_cls, est_cls, to_dist, arity, seed_at=None, positive=None,
                 init_fit=None, read=None):
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
        # init_fit(data) -> a concrete Distribution to warm-start EM for families whose
        # MLE is sensitive to initialization (e.g. negative-binomial dispersion).
        self.init_fit = init_fit
        # read(dist) -> {conventional param name: value}: the inverse of construction, so
        # fitted params come back in the *same* parameterization the user wrote (sd, not sigma2).
        self.read = read

    def make_dist(self, args: Tuple[Any, ...], name: Optional[str]):
        kwargs = self.to_dist(*args)
        if name is not None:
            kwargs.setdefault("name", name)
        return self.dist_cls(**kwargs)

    def make_estimator(self, name: Optional[str], keys: Optional[str]):
        kwargs: Dict[str, Any] = {}
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


_FAMILIES: Dict[str, Any] = {}
_DIST_TO_FAMILY: Dict[type, "Family"] = {}        # reverse map for reading fitted params
_DIST_TO_COMPOSITE_READ: Dict[type, Any] = {}     # composite dist type -> read(dist, read_params)


def register_family(name, dist_cls, est_cls, to_dist, arity, seed_at=None, positive=None,
                    init_fit=None, read=None) -> Family:
    fam = Family(name, dist_cls, est_cls, to_dist, arity, seed_at=seed_at, positive=positive,
                 init_fit=init_fit, read=read)
    _FAMILIES[name] = fam
    _DIST_TO_FAMILY[dist_cls] = fam
    return fam


def register_composite(name, dist_fn, est_fn, seed_fn=None, dist_cls=None, read=None) -> CompositeFamily:
    fam = CompositeFamily(name, dist_fn, est_fn, seed_fn=seed_fn, read=read)
    _FAMILIES[name] = fam
    if dist_cls is not None and read is not None:
        _DIST_TO_COMPOSITE_READ[dist_cls] = read
    return fam


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


def seed_child(rv: "RandomVariable", value: Any, scale: float):
    """Build a concrete distribution for a child RV located at a data ``value``.

    Bound children are used as-is; flat ``free`` children use their family ``seed_at``.
    Returns None when the child can't be seeded (caller falls back to default init).
    """
    if rv._kind == "bound":
        return rv._dist
    if rv._kind == "sample" and not isinstance(rv._family, CompositeFamily):
        fam = rv._family
        if fam.seed_at is not None:
            return fam.dist_cls(**fam.seed_at(value, scale))
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

    # -- constructors -------------------------------------------------------
    @classmethod
    def _sample(cls, family_name, args, *, name=None, keys=None, scope="shared") -> "RandomVariable":
        fam = _FAMILIES[family_name]
        return cls("sample", family=fam, args=args, name=name, keys=keys, scope=scope)

    def each(self) -> "RandomVariable":
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
    def _bound(cls, dist, *, name=None, result=None) -> "RandomVariable":
        return cls("bound", dist=dist, name=name or getattr(dist, "name", None), result=result)

    @classmethod
    def _apply(cls, base, transform) -> "RandomVariable":
        # Apply node: a deterministic transform of one RV (algebra rung 1).
        return cls("apply", args=(base, transform))

    @classmethod
    def _sum(cls, a, b) -> "RandomVariable":
        # Convolution node: the distribution of a + b for independent a, b.
        return cls("sum", args=(a, b))

    # -- algebra (deterministic transforms + convolution) -------------------
    def _affine(self, loc, scale) -> "RandomVariable":
        from pysp.stats.transform import AffineTransform
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

    def exp(self) -> "RandomVariable":
        from pysp.stats.transform import ExpTransform
        return RandomVariable._apply(self, ExpTransform())

    def log(self) -> "RandomVariable":
        from pysp.stats.transform import LogTransform
        return RandomVariable._apply(self, LogTransform())

    # -- conditioning: comparisons build Events; .given restricts the RV -----
    def __gt__(self, v): return Event(self, lambda x: np.asarray(x) > v, f"> {v}")
    def __ge__(self, v): return Event(self, lambda x: np.asarray(x) >= v, f">= {v}")
    def __lt__(self, v): return Event(self, lambda x: np.asarray(x) < v, f"< {v}")
    def __le__(self, v): return Event(self, lambda x: np.asarray(x) <= v, f"<= {v}")

    def given(self, event) -> "RandomVariable":
        """Condition this RV on an event (e.g. ``x.given(x > 0)`` -> truncation). The
        result samples by rejection and scores with the renormalized density."""
        if not isinstance(event, Event):
            raise TypeError("given() expects an Event from a comparison, e.g. x > 0.")
        return RandomVariable("given", args=(self, event))

    # -- introspection ------------------------------------------------------
    @property
    def is_bound(self) -> bool:
        return self._kind == "bound"

    @property
    def has_free(self) -> bool:
        return self._kind == "sample" and any(_is_free(a) for a in self._args)

    @property
    def name(self) -> Optional[str]:
        return self._name

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
    def sample(self, n: Optional[int] = None, seed: Optional[int] = None):
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

    def mean(self, samples: int = 20000, seed: int = 0):
        """Expected value of the random variable (Monte-Carlo; works for any RV —
        concrete, transformed, convolved, or conditioned)."""
        return float(np.mean(np.asarray(self.sample(samples, seed=seed), dtype=float)))

    def var(self, samples: int = 20000, seed: int = 0):
        """Variance of the random variable (Monte-Carlo)."""
        return float(np.var(np.asarray(self.sample(samples, seed=seed), dtype=float)))

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
            delta: float = 1e-8, backend: str = "local", num_workers: Optional[int] = None,
            print_iter: int = 0, **kw) -> "RandomVariable":
        """Estimate / infer parameters from ``data`` and return a bound RV.

        ``how``: ``'em'`` (EM/MLE, default for plain ``free`` models), ``'map'`` (maximize
        the joint with priors), ``'mcmc'`` (posterior samples over parameters with priors),
        ``'auto'`` picks ``map`` when the model has priors else ``em``. EM threads pysp's
        parallel/distributed backends (``backend='mp'|'mpi'|'dask'``).
        """
        # regression / GLM: a linear predictor (covariates) in a parameter slot
        if self._kind == "sample" and any(isinstance(a, _LinearPredictor) for a in self._args):
            from pysp.ppl import regression as _reg
            return _reg.regression_fit(self, data, **kw)

        grouped = self._kind == "sample" and any(
            isinstance(a, RandomVariable) and a._scope == "grouped" for a in self._args)
        if how == "auto":
            if grouped:
                how = "hierarchical"
            elif self._has_priors():
                from pysp.ppl import inference as _inf
                how = "conjugate" if _inf.conjugate_spec(self) is not None else "map"
            else:
                how = "em"
        if how in ("map", "mcmc", "hmc", "vi", "vmp", "conjugate", "hierarchical"):
            from pysp.ppl import inference as _inf
            if how == "mcmc":
                return _inf.mcmc_fit(self, data, **kw)
            if how == "hmc":
                return _inf.hmc_fit(self, data, **kw)
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


# -------------------------------------------------------------------- the lowering
def lower(rv: "RandomVariable", *, target: str = "dist"):
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
        from pysp.stats.transform import TransformDistribution
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
