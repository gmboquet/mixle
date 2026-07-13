"""Bayesian inference for mixle.ppl: parameter MCMC, MAP, VI, and closed-form conjugate updates.

A model whose parameter slots hold *distributions* (priors) or ``free`` defines a joint
``log p(data | theta) + log p(theta)``. MAP (maximize), MCMC/HMC/NUTS (sample), and VI all run on
the exact same target, scored with the existing vectorized ``seq_log_density`` and mixle's
``mixle.inference.mcmc`` kernels — no new inference engine.

What this module supports:
  * Flat ``Sample`` models (``Normal(Normal(0, 10), free)``) and *composite* models (mixtures,
    sequences): composites collect their leaf ``free``/prior parameters across the tree and rebuild
    a concrete model per evaluation (``_collect_composite``).
  * Hierarchical priors -- a prior whose own hyperparameter is another random variable
    (``Normal(0, tau)`` with ``tau`` estimated), scored differentiably via parent-slot substitution.
  * Non-centered reparameterization (``.noncentered()``) for funnel-prone location-scale priors.
  * Grouped / random-intercept ("plate") models -- ``Normal(Normal(mu, tau).each(), sigma)`` sampled
    jointly over the hyperparameters and every per-group latent.

Identifiability is handled automatically: mixture chains are relabeled (sorted by a leading
parameter) before pooling, so label-switching no longer needs a user-supplied ordering constraint.

Section map (banners below): core slot/target construction; init + result assembly (finalize,
convergence diagnostics, relabeling); parallel chains; inequality/region constraints; hierarchical
& grouped models; sampler setup and the ``how=`` drivers; closed-form conjugate / conjugate-mixture
/ hierarchical-EM updates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln as _gammaln

from mixle.ppl.core import (
    CompositeFamily,
    Constraint,
    RandomVariable,
    _CholeskySpec,
    _eval_expr,
    _expr_leaves,
    _OrderedSpec,
    _Potential,
    _row_mask,
    _SimplexSpec,
    _VectorSpec,
    free,
    lower,
)

# Deterministic-expression node kinds: an arg slot that is an arithmetic function of other latents
# (e.g. ``Normal(a + b, sigma)``) rather than a single prior/free. Its leaves are sampled; the slot's
# value is recomputed from them by ``_eval_expr`` at each evaluation.
_DET_EXPR_KINDS = frozenset({"sum", "prod", "pow", "apply", "select"})


def _is_det_expr(a: Any) -> bool:
    return isinstance(a, RandomVariable) and a._kind in _DET_EXPR_KINDS


_NEG_INF = -1e300


@dataclass
class _Slot:
    index: int  # position in the family's argument tuple
    prior: Any  # a concrete prior distribution, or None for a flat `free` slot
    positive: bool  # sampled in log-space when True (kept for back-compat; see `support`)
    name: str | None  # parameter name (prior's name, else "argN")
    handle: Any  # the prior RandomVariable (for .posterior(handle)), or None
    support: str = "real"  # 'real' | 'positive' (log) | 'unit' (logit) reparameterization
    group: int | None = None  # index of the exchangeable mixture component this slot belongs to
    role: str = "param"  # 'param' (a component's parameter) | 'weight' (its mixture weight)
    parent_args: dict[int, int] | None = None  # hierarchical prior: {prior-arg position -> child slot index}
    reparam: str | None = None  # 'loc_scale' -> sample z~N(0,1), value = loc + scale*z (non-centered)


def _to_value(support: str, u: float):
    """Map an unconstrained scalar to the constrained parameter, with the log|Jacobian|."""
    if support == "positive":
        return math.exp(u), float(u)
    if support == "unit":
        v = 1.0 / (1.0 + math.exp(-u))
        return v, math.log(v) + math.log1p(-v)  # d sigmoid/du = v(1-v)
    return float(u), 0.0


def _to_u(support: str, val: float) -> float:
    """Inverse of :func:`_to_value` (constrained value -> unconstrained)."""
    if support == "positive":
        return math.log(max(float(val), 1e-12))
    if support == "unit":
        p = min(max(float(val), 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))
    return float(val)


def _loc_scale(s: _Slot, vals: dict):
    """Resolve the (loc, scale) of a non-centered Normal slot from constants / hyperparameter slots.

    The non-centered transform value = loc + scale*z is a parameter-space map like ``_to_value`` above,
    so it lives with the other reparameterizations; ``loc``/``scale`` come from constants or, when the
    prior is hierarchical, the already-evaluated hyperparameter slots in ``vals``."""
    pa = s.parent_args or {}
    loc = vals[pa[0]] if 0 in pa else float(s.handle._args[0])
    scale = vals[pa[1]] if 1 in pa else float(s.handle._args[1])
    return loc, scale


class Posterior:
    """Parameter-posterior result attached to a fitted RV's ``.result``.

    Holds value-space draws and the raw ``MCMCResult``. Look up a parameter by its
    RandomVariable handle, its name, or its slot index.
    """

    def __init__(self, slots: list[_Slot], value_samples: np.ndarray, raw: Any):
        self._slots = slots
        self._samples = value_samples  # (n_draws, n_params), value space
        self.raw = raw
        self.acceptance_rate = getattr(raw, "acceptance_rate", None)
        self.predictive = None  # set by the fitter (posterior predictive)
        self.build = None  # set by the fitter: vals-dict -> concrete dist
        self.rhat = None  # {param: Gelman-Rubin R-hat} (multi-chain)
        self.ess = None  # combined effective sample size (multi-chain)
        self.n_chains = 1
        self.split_rhat = None  # {param: rank-normalized split-R-hat} (Vehtari 2021)
        self.bulk_ess = None  # {param: bulk effective sample size}
        self.tail_ess = None  # {param: tail effective sample size}
        self.num_divergences = 0  # NUTS divergent transitions (post-warmup, summed over chains)

    def pointwise_log_likelihood(self, data) -> np.ndarray:
        """Return the ``(n_draws, n_obs)`` log-likelihood of ``data`` under each posterior draw.

        This is the input to the predictive model-comparison diagnostics (WAIC, PSIS-LOO).
        """
        if self.build is None:
            raise ValueError("this posterior cannot recompute the pointwise log-likelihood.")
        data = list(data)
        rows = []
        for j in range(self._samples.shape[0]):
            d = self.build({s.index: float(self._samples[j, k]) for k, s in enumerate(self._slots)})
            enc = d.dist_to_encoder().seq_encode(data)
            rows.append(np.asarray(d.seq_log_density(enc), dtype=float))
        return np.asarray(rows)

    def _col(self, param) -> int:
        for k, s in enumerate(self._slots):
            if param is s.handle or param == s.name or param == s.index:
                return k
        raise KeyError(f"no sampled parameter matching {param!r}")

    def samples(self, param=None) -> np.ndarray:
        """Return posterior samples for all parameters or one selected parameter."""
        if param is None:
            return self._samples
        return self._samples[:, self._col(param)]

    def mean(self, param=None):
        """Return posterior means for all parameters or one selected parameter."""
        return self.samples(param).mean(axis=0)

    def summary(self) -> dict:
        """Return parameter summaries and any available convergence diagnostics."""
        out = {}
        for k, s in enumerate(self._slots):
            col = self._samples[:, k]
            row = {
                "mean": float(col.mean()),
                "std": float(col.std()),
                "q2.5": float(np.percentile(col, 2.5)),
                "q97.5": float(np.percentile(col, 97.5)),
            }
            # Fold the per-parameter convergence diagnostics into the same row (ArviZ-style one table),
            # when a multi-chain fit produced them. The aggregate ``_rhat``/``_ess`` keys below stay for
            # back-compat.
            if isinstance(self.rhat, dict) and s.name in self.rhat:
                row["r_hat"] = float(self.rhat[s.name])
            if isinstance(self.split_rhat, dict) and s.name in self.split_rhat:
                row["split_r_hat"] = float(self.split_rhat[s.name])
            if isinstance(self.bulk_ess, dict) and s.name in self.bulk_ess:
                row["ess_bulk"] = float(self.bulk_ess[s.name])
            if isinstance(self.tail_ess, dict) and s.name in self.tail_ess:
                row["ess_tail"] = float(self.tail_ess[s.name])
            out[s.name] = row
        out["_acceptance_rate"] = self.acceptance_rate
        if self.rhat is not None:
            out["_rhat"] = self.rhat
            out["_ess"] = self.ess
            out["_n_chains"] = self.n_chains
        if self.split_rhat is not None:
            out["_split_rhat"] = self.split_rhat
            out["_bulk_ess"] = self.bulk_ess
            out["_tail_ess"] = self.tail_ess
        if self.num_divergences:
            out["_num_divergences"] = self.num_divergences
        return out


# --------------------------------------------------------------------------- core
def _require_flat(rv: RandomVariable):
    # Composites are handled by _composite_target_parts before this is reached; this guards
    # the remaining non-`sample` kinds (apply/sum/given/joint), which have no parameters to fit.
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        raise NotImplementedError(
            f"parameter MCMC/MAP needs a `sample` model (a family with free/prior slots); got kind {rv._kind!r}."
        )
    return rv._family


def _slots_of(rv: RandomVariable, fam, extra_latents=()) -> tuple[list[_Slot], dict[int, tuple]]:
    """Collect the latent slots of a flat model, plus deterministic-expression bindings.

    Returns ``(slots, det_bindings)``. ``det_bindings`` maps a family arg position whose argument is a
    *deterministic expression* over latents (``Normal(a + b, sigma)``) to ``(expr, [(leaf, slot_index)])``:
    the leaves are sampled as ordinary slots and the arg's value is recomputed from them via
    ``_eval_expr`` at build time. Positions absent from ``det_bindings`` are ordinary prior/free slots.

    ``extra_latents`` are random variables that participate in the joint only through a custom
    :func:`~mixle.ppl.core.potential` (an auxiliary latent the likelihood does not touch). Each that is
    not already a model parameter is added as a sampled slot (real support) so the potential can
    reference it.
    """
    slots: list[_Slot] = []
    nested = [len(rv._args)]  # synthetic, build-ignored indices for hierarchical hyperparameters
    leaf_slot: dict[int, int] = {}  # id(leaf RV) -> slot index, dedups a latent shared across expressions
    det_bindings: dict[int, tuple] = {}

    def add_prior(handle: RandomVariable, index: int, support: str) -> None:
        """Add a slot for prior RV ``handle`` at ``index``; recurse into any random hyperparameters.

        A flat prior (constant hyperparameters) lowers to a fixed distribution. A *hierarchical* prior
        -- one whose own parameter is another random variable, e.g. ``Normal(0, tau)`` with ``tau``
        estimated -- instead gets each random hyperparameter its own (build-ignored) slot, and records
        the ``prior-arg -> child-slot`` map so the prior log-density is scored as a function of them.
        """
        pfam = handle._family
        parent_args: dict[int, int] = {}
        for j, arg in enumerate(handle._args):
            if isinstance(arg, RandomVariable):
                child = nested[0]
                nested[0] += 1
                parent_args[j] = child
                add_prior(arg, child, pfam.support[j])  # hyperparameter sits on the parent family's support
        nm = handle.name or f"arg{index}"
        if handle._reparam == "loc_scale":
            # Non-centered: sample z ~ N(0,1) (real) and set value = loc + scale*z. The slot is real
            # regardless of the model's support, and carries parent_args so loc/scale resolve at eval.
            slots.append(_Slot(index, None, False, nm, handle, "real", parent_args=parent_args, reparam="loc_scale"))
        elif parent_args:
            slots.append(_Slot(index, None, support == "positive", nm, handle, support, parent_args=parent_args))
        else:
            slots.append(_Slot(index, lower(handle, target="dist"), support == "positive", nm, handle, support))

    def add_leaf(leaf: Any, support: str) -> int:
        """Register one latent leaf of a deterministic expression (deduped by identity); return its slot index."""
        key = id(leaf)
        if key in leaf_slot:
            return leaf_slot[key]
        idx = nested[0]
        nested[0] += 1
        if leaf is free:
            slots.append(_Slot(idx, None, False, f"arg{idx}", None, support))
        elif isinstance(leaf, RandomVariable) and leaf._kind == "sample":
            add_prior(leaf, idx, support)  # a prior leaf (possibly itself hierarchical)
        else:
            slots.append(_Slot(idx, None, False, getattr(leaf, "name", None) or f"arg{idx}", leaf, support))
        leaf_slot[key] = idx
        return idx

    for i, a in enumerate(rv._args):
        if _is_det_expr(a):
            binding = [(leaf, add_leaf(leaf, fam.support[i])) for leaf in _expr_leaves(a)]
            det_bindings[i] = (a, binding)
        elif isinstance(a, RandomVariable):
            add_prior(a, i, fam.support[i])
        elif a is free:
            slots.append(_Slot(i, None, fam.positive[i], f"arg{i}", None, fam.support[i]))
    # Auxiliary latents that enter only through a custom potential: add any not already a slot handle.
    present = {id(s.handle) for s in slots if s.handle is not None}
    for lv in extra_latents:
        if id(lv) not in present and id(lv) not in leaf_slot:
            add_leaf(lv, "real")
    if not slots:
        raise ValueError("model has no `free`/prior parameters to infer.")
    return slots, det_bindings


def _encoder_for(fam):
    # A valid probe distribution to obtain the data encoder. Use support-aware defaults
    # (0 for real, 1 for positive, 0.5 for unit) so bounded families (Bernoulli/Beta/...) are
    # constructed with in-range params; fall back to all-ones if a family needs it.
    if fam.seed_at:
        kwargs = fam.seed_at(0.0, 1.0)
    else:
        defaults = {"real": 0.0, "positive": 1.0, "unit": 0.5}
        try:
            kwargs = fam.to_dist(*[defaults[s] for s in fam.support])
        except Exception:  # noqa: BLE001
            kwargs = fam.to_dist(*([1.0] * fam.arity))
    return fam.dist_cls(**kwargs).dist_to_encoder()


def _is_dirichlet_rv(a) -> bool:
    return (
        isinstance(a, RandomVariable)
        and a._kind == "sample"
        and not isinstance(a._family, CompositeFamily)
        and a._family.name == "Dirichlet"
    )


def _spec_of(a):
    """The structural spec carried by an arg: a bare spec, or the spec inside a ``param(...)``
    handle (an RV of kind ``param``). ``None`` if the arg is not a structural parameter."""
    if isinstance(a, (_SimplexSpec, _VectorSpec, _CholeskySpec, _OrderedSpec)):
        return a
    if isinstance(a, RandomVariable) and a._kind == "param":
        return a._args[0]
    return None


def _handle_of(a):
    """The referenceable handle for a structural parameter (a ``param(...)`` RV), else ``None``."""
    return a if (isinstance(a, RandomVariable) and a._kind == "param") else None


def _is_struct_spec(a) -> bool:
    return _spec_of(a) is not None


def _struct_spec_for(node, a):
    """The structural-parameter spec for a composite argument ``a``, or ``None``. Handles an
    explicit spec or a ``param(...)`` handle (mixture weights, HMM transition / initial, MVN
    mean / covariance), a bare ``Dirichlet(alpha)`` prior (one simplex), and a bare ``free``
    simplex sized to the component count (mixture weights)."""
    s = _spec_of(a)
    if s is not None:
        return s
    if _is_dirichlet_rv(a):
        return _SimplexSpec(np.asarray(a._args[0], dtype=float), rows=1, name=a.name)
    if a is free:
        for other in node._args:
            if isinstance(other, (list, tuple)):
                return _SimplexSpec(np.ones(len(other), dtype=float), rows=1)
    return None


def _spec_slot_defs(spec):
    """The scalar slots a structural spec expands to: a list of (prior, support, name)."""
    from mixle.stats.univariate.continuous.gamma import GammaDistribution

    if isinstance(spec, _SimplexSpec):  # Gamma representation of the Dirichlet, one slot per entry
        base, kk = spec.name or "w", len(spec.alpha)
        return [
            (
                GammaDistribution(k=float(spec.alpha[j]), theta=1.0),
                "positive",
                f"{base}{j}" if spec.rows == 1 else f"{base}{r}_{j}",
            )
            for r in range(spec.rows)
            for j in range(kk)
        ]
    if isinstance(spec, _VectorSpec):  # independent entries on one support
        base = spec.name or "v"
        return [(None, spec.support, f"{base}{i}") for i in range(spec.dim)]
    if isinstance(spec, _OrderedSpec):  # real base + positive increments -> increasing vector
        base = spec.name or "o"
        return [(None, "real" if i == 0 else "positive", f"{base}{i}") for i in range(spec.dim)]
    if isinstance(spec, _CholeskySpec):  # lower-triangular Cholesky entries (diag positive)
        base = spec.name or "L"
        return [
            (None, "positive" if i == j else "real", f"{base}{i}_{j}") for i in range(spec.dim) for j in range(i + 1)
        ]
    raise TypeError(f"unknown structural spec {type(spec).__name__}")


def _spec_assemble(spec, values):
    """Assemble a structural spec's scalar values into its vector/matrix parameter value."""
    g = np.asarray(values, dtype=float)
    if isinstance(spec, _SimplexSpec):
        m = g.reshape(spec.rows, len(spec.alpha))
        s = m.sum(axis=1, keepdims=True)
        w = m / np.where(s > 0, s, 1.0)
        return w[0] if spec.rows == 1 else w
    if isinstance(spec, _VectorSpec):
        return g
    if isinstance(spec, _OrderedSpec):  # cumulative sum of positive increments -> strictly increasing
        return g[0] + np.concatenate([[0.0], np.cumsum(g[1:])])
    if isinstance(spec, _CholeskySpec):
        d = spec.dim
        L = np.zeros((d, d))
        L[np.tril_indices(d)] = g  # row-major lower-triangular fill matches the slot order above
        return L @ L.T
    raise TypeError(f"unknown structural spec {type(spec).__name__}")


def _collect_composite(rv: RandomVariable):
    """Walk a composite model, collect every free/prior parameter as a slot, and return
    ``(slots, rebuild)`` where ``rebuild(vals)`` reconstructs a fully-concrete RV.

    Parameters come in two shapes. *Leaf* free/prior args (a component mean, a rate) are scalar
    slots. *Simplex* args of a combinator — mixture weights / a transition row, given as a
    ``Dirichlet(alpha)`` prior or ``free`` — are expanded via the Gamma representation of the
    Dirichlet: K positive slots ``g_k ~ Gamma(alpha_k, 1)`` that ``rebuild`` normalizes to
    ``w = g / sum(g)`` (so ``w ~ Dirichlet(alpha)`` exactly, with no simplex Jacobian needed).
    Child models (an RV, or a list of RVs) are recursed into; other args (lengths, fixed
    weights) are kept. ``collect`` and ``rebuild`` traverse identically, so the k-th parameter
    encountered is ``slots[k]``.

    Auto-generated slot names inside a multi-component list are qualified by the component path
    (``comp0.arg0``, ``comp1.arg0``, ...): without the qualifier a K-component mixture of
    same-shaped components yields K colliding ``arg{i}`` names, and ``summary()`` / ``samples()``
    / R-hat (all keyed by name) silently drop all but one. User-chosen names and any names in a
    single-component list are kept as-is."""
    slots: list[_Slot] = []

    def collect(node: RandomVariable, prefix: str = ""):
        fam = node._family
        if isinstance(fam, CompositeFamily):
            # Components of a finite mixture are exchangeable -> tag their slots with the component
            # index so parallel chains can be relabeled (label-switching) before pooling / R-hat.
            exch = fam.name in ("Mixture", "SemiMix")
            for a in node._args:
                if isinstance(a, (list, tuple)):
                    multi = sum(1 for c in a if isinstance(c, RandomVariable)) > 1
                    for gi, c in enumerate(a):
                        if isinstance(c, RandomVariable):
                            before = len(slots)
                            collect(c, f"{prefix}comp{gi}." if multi else prefix)
                            if exch:  # tag only the outermost exchangeable level (nested groups keep theirs)
                                for s in slots[before:]:
                                    if s.group is None:
                                        s.group = gi
                    continue
                spec = _struct_spec_for(node, a)
                if spec is not None:  # structural vector/matrix parameter -> scalar slots
                    handle = _handle_of(a)  # a param(...) handle is referenceable in constraints
                    named = bool(getattr(spec, "name", None))  # user-chosen spec names stay unqualified
                    for gi, (prior, support, nm) in enumerate(_spec_slot_defs(spec)):
                        nm = nm if named else f"{prefix}{nm}"
                        s = _Slot(len(slots), prior, support == "positive", nm, handle, support)
                        if exch:  # mixture weights: weight j pairs with component j -> permute together
                            s.group, s.role = gi, "weight"
                        slots.append(s)
                elif isinstance(a, RandomVariable):
                    collect(a, prefix)  # child model
            return
        for i, a in enumerate(node._args):
            if _spec_of(a) is not None:  # a vector/matrix leaf parameter (Dirichlet alpha, Categorical probs)
                handle = _handle_of(a)
                spec = _spec_of(a)
                named = bool(getattr(spec, "name", None))
                for prior, support, nm in _spec_slot_defs(spec):
                    nm = nm if named else f"{prefix}{nm}"
                    slots.append(_Slot(len(slots), prior, support == "positive", nm, handle, support))
            elif isinstance(a, RandomVariable):
                nm = a.name or f"{prefix}arg{i}"
                slots.append(_Slot(len(slots), lower(a, target="dist"), fam.positive[i], nm, a, fam.support[i]))
            elif a is free:
                slots.append(_Slot(len(slots), None, fam.positive[i], f"{prefix}arg{i}", None, fam.support[i]))

    collect(rv)
    if not slots:
        raise ValueError("model has no `free`/prior parameters to infer.")

    def rebuild(vals):
        counter = [0]

        def take(n):
            out = [vals[counter[0] + j] for j in range(n)]
            counter[0] += n
            return out

        def build_node(node: RandomVariable):
            fam = node._family
            if isinstance(fam, CompositeFamily):
                new_args = []
                for a in node._args:
                    if isinstance(a, (list, tuple)):
                        new_args.append([build_node(c) if isinstance(c, RandomVariable) else c for c in a])
                        continue
                    spec = _struct_spec_for(node, a)
                    if spec is not None:  # assemble scalar draws into the vector/matrix value
                        n_slots = len(_spec_slot_defs(spec))
                        new_args.append(_spec_assemble(spec, take(n_slots)))
                    elif isinstance(a, RandomVariable):
                        new_args.append(build_node(a))
                    else:
                        new_args.append(a)
                return RandomVariable._sample(
                    fam.name, tuple(new_args), name=node._name, keys=node._keys, scope=node._scope
                )
            new_args = []
            for a in node._args:
                spec = _spec_of(a)
                if spec is not None:
                    new_args.append(_spec_assemble(spec, take(len(_spec_slot_defs(spec)))))
                elif a is free or isinstance(a, RandomVariable):
                    new_args.append(take(1)[0])
                else:
                    new_args.append(a)
            return RandomVariable._sample(
                fam.name, tuple(new_args), name=node._name, keys=node._keys, scope=node._scope
            )

        return build_node(rv)

    return slots, rebuild


def _composite_target_parts(rv: RandomVariable, data):
    """``_target_parts`` for a composite model (mixtures, sequences): parameters are the leaf
    ``free``/prior slots collected across the tree; ``build`` rebuilds + lowers the composite."""
    slots, rebuild = _collect_composite(rv)
    try:
        arr = np.asarray(data, dtype=float)
        _fin = arr[np.isfinite(arr)]  # ignore NaN (missing) when seeding the init point
        dmean = float(_fin.mean()) if _fin.size else 0.0
        dstd = float(_fin.std() or 1.0) if _fin.size else 1.0
    except (ValueError, TypeError):
        dmean, dstd = 0.0, 1.0  # non-scalar data (sequences/vectors): use neutral init

    def unpack(u):
        vals, logj = {}, 0.0
        for k, s in enumerate(slots):
            v, lj = _to_value(s.support, u[k])
            vals[s.index] = v
            logj += lj
        return vals, logj

    def build(vals):
        return lower(rebuild(vals), target="dist")

    return None, slots, build, unpack, (dmean, dstd)


def _potential_latents(potentials) -> tuple:
    """Unique RVs referenced by ``potentials`` (an auxiliary latent may appear only here)."""
    if potentials is None:
        return ()
    if isinstance(potentials, _Potential):
        potentials = [potentials]
    out, seen = [], set()
    for p in potentials:
        for v in p.vars:
            if id(v) not in seen:
                seen.add(id(v))
                out.append(v)
    return tuple(out)


def _target_parts(rv: RandomVariable, data, extra_latents=()):
    """Encoder-free pieces shared by the numerical and autograd targets:
    (fam, slots, build, unpack, (dmean, dstd)). Building the mixle encoder is deferred to
    callers that need it (the autograd path scores the raw data tensor and never does)."""
    if rv._kind == "sample" and (isinstance(rv._family, CompositeFamily) or any(_is_struct_spec(a) for a in rv._args)):
        # composites, and flat leaves with a vector/matrix parameter (Dirichlet alpha,
        # Categorical probs), use the general collect/rebuild target.
        if extra_latents:
            raise NotImplementedError("custom potentials are supported on flat models only (not composites) for now.")
        return _composite_target_parts(rv, data)
    fam = _require_flat(rv)
    slots, det_bindings = _slots_of(rv, fam, extra_latents)
    arr = np.asarray(data, dtype=float)
    _fin = arr[np.isfinite(arr)]  # ignore NaN (missing) when seeding the init point
    dmean = float(_fin.mean()) if _fin.size else 0.0
    dstd = float(_fin.std() or 1.0) if _fin.size else 1.0

    def unpack(u):
        vals, logj = {}, 0.0
        for k, s in enumerate(slots):
            v, lj = _to_value(s.support, u[k])
            if s.reparam == "loc_scale":  # v was z ~ N(0,1); the parameter is loc + scale*z
                loc, scale = _loc_scale(s, vals)
                v = loc + scale * v
            vals[s.index] = v
            logj += lj
        return vals, logj

    def build(vals):
        args = []
        for i in range(len(rv._args)):
            if i in det_bindings:  # arg is a deterministic function of latents -> recompute from them
                expr, binding = det_bindings[i]
                args.append(_eval_expr(expr, {leaf: vals[sidx] for leaf, sidx in binding}))
            else:
                args.append(vals.get(i, rv._args[i]))
        return fam.make_dist(tuple(args), rv._name)

    return fam, slots, build, unpack, (dmean, dstd)


def _build_target(rv: RandomVariable, data, extra_latents=(), jacobian: bool = True):
    """Return (log_target(u), slots, fam, build, unpack, (dmean,dstd)) for unconstrained u.

    ``jacobian=True`` (samplers / VI / Laplace) includes the support-transform log|J|, so
    ``log_target`` is the joint density in the *unconstrained* space. ``jacobian=False`` builds
    the point-estimate objective: the joint scored in the constrained parameter space, so MAP
    maximizes ``ll + log prior`` there and reduces to the MLE under flat priors."""
    fam, slots, build, unpack, (dmean, dstd) = _target_parts(rv, data, extra_latents)
    if fam is not None:
        enc = _encoder_for(fam).seq_encode(list(data))
    else:  # composite: encode through a concrete instance built at the initial point
        v0, _ = unpack(_init_u(slots, dmean, dstd))
        enc = build(v0).dist_to_encoder().seq_encode(list(data))

    def log_target(u):
        vals, logj = unpack(u)
        try:
            d = build(vals)
            ll = float(np.sum(d.seq_log_density(enc)))
        except Exception:  # noqa: BLE001
            return _NEG_INF
        if not math.isfinite(ll):
            return _NEG_INF
        plp = 0.0
        for s in slots:
            if s.reparam == "loc_scale":  # prior is N(0,1) on the latent z = (value - loc) / scale
                loc, scale = _loc_scale(s, vals)
                z = (vals[s.index] - loc) / scale
                plp += -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
            elif s.parent_args:  # hierarchical prior: rebuild it with the sampled hyperparameter values
                pargs = list(s.handle._args)
                for j, child in s.parent_args.items():
                    pargs[j] = vals[child]
                plp += float(s.handle._family.make_dist(tuple(pargs), s.handle._name).log_density(vals[s.index]))
            elif s.prior is not None:
                plp += float(s.prior.log_density(vals[s.index]))
        return ll + plp + (logj if jacobian else 0.0)

    return log_target, slots, fam, build, unpack, (dmean, dstd)


# --------------------------------- unconstrained-space init & value reconstruction
def _init_u(slots, dmean, dstd) -> np.ndarray:
    u0 = []
    for s in slots:
        if s.reparam == "loc_scale":
            u0.append(0.0)  # a non-centered latent z ~ N(0, 1) starts at its mode
        elif s.support == "positive":
            u0.append(_to_u("positive", max(dstd, 1e-2)))
        elif s.support == "unit":
            u0.append(0.0)  # logit(0.5)
        else:
            u0.append(dmean)
    return np.asarray(u0, dtype=float)


def _init_scale(slots, dstd, n) -> np.ndarray:
    """Per-slot proposal scale ~ posterior width: a location (real) slot ~ dstd/sqrt(n);
    a transformed (positive/unit) slot ~ 1/sqrt(n). Adaptation then tunes the magnitude."""
    root = math.sqrt(max(n, 1))
    return np.asarray([max((dstd if s.support == "real" else 1.0) / root, 1e-3) for s in slots], dtype=float)


def _u_to_vals(slots, u) -> np.ndarray:
    """Map an (n, d) unconstrained sample array to constrained parameter values per slot.

    Applies each slot's support transform, and for a non-centered slot the deterministic
    ``value = loc + scale * z`` (resolving loc/scale from the already-computed hyperparameter columns --
    children precede parents in slot order, so they are ready)."""
    u = np.asarray(u, dtype=float).reshape(len(u), -1)
    vals = np.empty_like(u)
    idx_to_col = {s.index: k for k, s in enumerate(slots)}
    for k, s in enumerate(slots):
        if s.reparam == "loc_scale":
            pa = s.parent_args or {}
            loc = vals[:, idx_to_col[pa[0]]] if 0 in pa else float(s.handle._args[0])
            scale = vals[:, idx_to_col[pa[1]]] if 1 in pa else float(s.handle._args[1])
            vals[:, k] = loc + scale * u[:, k]
        elif s.support == "positive":
            vals[:, k] = np.exp(u[:, k])
        elif s.support == "unit":
            vals[:, k] = 1.0 / (1.0 + np.exp(-u[:, k]))
        else:
            vals[:, k] = u[:, k]
    return vals


# ------------------------------------ convergence diagnostics & mixture relabeling
def _gelman_rubin(chains_u: np.ndarray) -> np.ndarray:
    """Per-dimension Gelman-Rubin R-hat from (n_chains, n_draws, d) unconstrained samples."""
    m, n, _ = chains_u.shape
    if m < 2 or n < 2:
        return np.full(chains_u.shape[-1], np.nan)
    chain_means = chains_u.mean(axis=1)
    W = chains_u.var(axis=1, ddof=1).mean(axis=0)
    B = n * chain_means.var(axis=0, ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(np.maximum(var_hat / np.where(W > 0, W, 1.0), 0.0))


def _exchangeable_layout(slots) -> list[list[int]] | None:
    """Slot positions per exchangeable mixture component, ``[[comp0 slots], [comp1 slots], ...]``.

    Each inner list is that component's slots in canonical order (parameters as collected, then its
    weight if free); all components share the same layout. Returns None if there is no exchangeable
    structure with >= 2 identically-shaped components (nothing to relabel).
    """
    groups: dict[int, list[tuple[int, str]]] = {}
    for k, s in enumerate(slots):
        if s.group is not None:
            groups.setdefault(s.group, []).append((k, s.role))
    if len(groups) < 2:
        return None
    layout, param_counts = [], set()
    for g in sorted(groups):
        params = [k for k, r in groups[g] if r == "param"]
        weights = [k for k, r in groups[g] if r == "weight"]
        layout.append(params + weights)
        param_counts.add(len(params))
    if len({len(comp) for comp in layout}) != 1 or param_counts == {0} or len(param_counts) != 1:
        return None  # heterogeneous components, or a group with no parameter to sort on
    return layout


def _relabel_chain(u_chain: np.ndarray, layout: list[list[int]]) -> np.ndarray:
    """Relabel a chain's draws so exchangeable components are sorted by their first parameter.

    Resolves label-switching: each draw is permuted so component blocks are ordered by their leading
    (location) parameter. Applied per chain before pooling, it makes independent parallel chains agree
    on a labeling -- correct pooled posterior and meaningful R-hat -- without constraining the sampler.
    """
    first = [comp[0] for comp in layout]  # leading parameter of each component = sort key
    out = u_chain.copy()
    order = np.argsort(u_chain[:, first], axis=1)  # (n_draws, n_components) per-draw rank of components
    for i, tgt in enumerate(layout):  # component sorted into rank i pulls from the rank-i source block
        src_rank = order[:, i]
        for col, tgt_slot in enumerate(tgt):
            src_slots = np.array([layout[g][col] for g in range(len(layout))])
            out[:, tgt_slot] = u_chain[np.arange(u_chain.shape[0]), src_slots[src_rank]]
    return out


def _attach_convergence(post, slots, arr, results) -> None:
    """Attach rank-normalized split-R-hat / bulk-ESS / tail-ESS (per parameter) and the summed NUTS
    divergence count. ``arr`` is the relabeled unconstrained draws, shape ``(n_chains, n_draws, d)``."""
    from mixle.ppl.diagnostics import bulk_ess, split_rhat, tail_ess

    post.split_rhat = {s.name: float(split_rhat(arr[:, :, k])) for k, s in enumerate(slots)}
    post.bulk_ess = {s.name: float(bulk_ess(arr[:, :, k])) for k, s in enumerate(slots)}
    post.tail_ess = {s.name: float(tail_ess(arr[:, :, k])) for k, s in enumerate(slots)}
    post.num_divergences = int(sum(int(np.sum(getattr(r, "divergences", np.zeros(0)))) for r in results))


# ------------------------------------------------------------------- result assembly
def _finalize(rv, slots, res, build) -> RandomVariable:
    """Convert one chain's unconstrained samples to value space, relabel mixtures, attach the
    convergence diagnostics, and build the posterior-mean distribution. Shared by RW-MCMC and HMC."""
    u = np.asarray(res.samples, dtype=float).reshape(len(res.samples), -1)
    layout = _exchangeable_layout(slots)  # resolve within-chain mixture label-switching too
    if layout is not None:
        u = _relabel_chain(u, layout)
    vals = _u_to_vals(slots, u)
    mean_vals = {s.index: float(vals[:, k].mean()) for k, s in enumerate(slots)}
    post = Posterior(slots, vals, res)
    # split-R-hat / bulk-/tail-ESS work on a single chain (split into halves) + count its divergences
    _attach_convergence(post, slots, u[None, :, :], [res])

    def predictive(n, rng):
        idx = rng.randint(len(vals), size=n)
        out = []
        for j in idx:
            d = build({s.index: float(vals[j, k]) for k, s in enumerate(slots)})
            out.append(d.sampler(seed=int(rng.randint(1, 2**31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    post.build = build
    return RandomVariable._bound(build(mean_vals), name=rv._name, result=post)


# ---------------------------------------- parallel chains (workers + orchestration)
def _mcmc_worker(seed, rv, data, kw):
    """Module-level (picklable) single RW-Metropolis chain (parallel path: no constraints)."""
    from mixle.inference.mcmc import AdaptiveRandomWalkProposal, metropolis_hastings

    log_target, _grad, slots, _build, dmean, dstd, _f = _prepare_target(rv, data, None, None, want_grad=False)
    u0 = _init_u(slots, dmean, dstd)
    scale = kw.get("scale")
    init_scale = (scale * np.ones(len(u0))) if scale is not None else _init_scale(slots, dstd, len(data))
    return metropolis_hastings(
        log_target,
        u0,
        AdaptiveRandomWalkProposal(init_scale.copy()),
        num_samples=kw["draws"],
        burn_in=kw["burn"],
        thin=kw["thin"],
        rng=np.random.RandomState(seed),
    )


def _hmc_worker(seed, rv, data, kw):
    """Module-level (picklable) single HMC chain (parallel path: no constraints)."""
    from mixle.inference.mcmc import hamiltonian_monte_carlo

    log_target, grad, slots, _build, dmean, dstd, _f = _prepare_target(rv, data, None, None, want_grad=True)
    u0 = _init_u(slots, dmean, dstd)
    mass = 1.0 / (_init_scale(slots, dstd, len(data)) ** 2)
    step_size = kw["step_size"] if kw["step_size"] is not None else 2.5 / kw["num_steps"]
    return hamiltonian_monte_carlo(
        log_target,
        grad,
        u0,
        num_samples=kw["draws"],
        step_size=step_size,
        num_steps=kw["num_steps"],
        mass=mass,
        burn_in=kw["burn"],
        thin=kw["thin"],
        rng=np.random.RandomState(seed),
    )


def _nuts_worker(seed, rv, data, kw):
    """Module-level (picklable) single NUTS chain (parallel path: no constraints)."""
    from mixle.inference.mcmc import nuts

    log_target, grad, slots, _build, dmean, dstd, _f = _prepare_target(rv, data, None, None, want_grad=True)
    u0 = _init_u(slots, dmean, dstd)
    mass = 1.0 / (_init_scale(slots, dstd, len(data)) ** 2)
    return nuts(
        log_target,
        grad,
        u0,
        num_samples=kw["draws"],
        warmup=kw["burn"],
        mass=mass,
        target_accept=kw["target_accept"],
        max_tree_depth=kw["max_tree_depth"],
        thin=kw["thin"],
        rng=np.random.RandomState(seed),
    )


def _ensemble_p0(slots, dmean, dstd, n_data, walkers, rng):
    """Dispersed initial ensemble (walkers, d): walker 0 at the data-informed point, the rest
    jittered by the prior/posterior width so the stretch move starts spread out."""
    d = len(slots)
    u0 = _init_u(slots, dmean, dstd)
    spread = _init_scale(slots, dstd, n_data) * math.sqrt(n_data)
    p0 = u0[None, :] + 0.1 * spread[None, :] * rng.standard_normal((walkers, d))
    p0[0] = u0
    return p0


def _ensemble_worker(seed, rv, data, kw):
    """Module-level (picklable) single ensemble run (parallel path: no constraints)."""
    from mixle.inference.mcmc import affine_invariant_ensemble

    log_target, _grad, slots, _build, dmean, dstd, _f = _prepare_target(
        rv, data, None, None, want_grad=False, numpy_only=True
    )
    rng = np.random.RandomState(seed)
    p0 = _ensemble_p0(slots, dmean, dstd, len(data), kw["walkers"], rng)
    return affine_invariant_ensemble(
        log_target, p0, num_samples=kw["draws"], burn_in=kw["burn"], thin=kw["thin"], rng=rng
    )


def _run_chains(run_one, worker, worker_args, chains: int, parallel, rng):
    """Run ``chains`` independent chains.

    ``parallel`` selects the backend: ``False``/``None`` -> sequential; ``True`` or
    ``"process"`` -> a process pool (genuine parallelism — the model pickles cleanly,
    so each worker rebuilds its own Torch target and they run on separate cores);
    ``"thread"`` -> a thread pool (rarely a win: the Torch path is GIL-bound).
    """
    seeds = [int(rng.randint(1, 2**31)) for _ in range(chains)]
    mode = "process" if parallel is True else (parallel or "off")
    if chains > 1 and mode == "process":
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=chains) as ex:
            futs = [ex.submit(worker, s, *worker_args) for s in seeds]
            return [f.result() for f in futs]
    if chains > 1 and mode == "thread":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=chains) as ex:
            return list(ex.map(run_one, seeds))
    return [run_one(s) for s in seeds]


def _finalize_chains(rv, slots, results, build) -> RandomVariable:
    """Combine multiple chains: pool value-space draws, attach R-hat and combined ESS.

    For mixtures the chains are first relabeled (label-switching): independent chains may settle on
    different component orderings, which would smear the pooled posterior and blow up R-hat. Sorting
    each chain's components by their leading parameter aligns them so the pooled summary is correct.
    """
    us = [np.asarray(r.samples, dtype=float).reshape(len(r.samples), -1) for r in results]
    layout = _exchangeable_layout(slots)
    if layout is not None:
        us = [_relabel_chain(u, layout) for u in us]
    n = min(len(u) for u in us)
    rhat = _gelman_rubin(np.stack([u[:n] for u in us], axis=0))
    vals = _u_to_vals(slots, np.concatenate(us, axis=0))
    mean_vals = {s.index: float(vals[:, k].mean()) for k, s in enumerate(slots)}
    post = Posterior(slots, vals, results[0])
    post.n_chains = len(results)
    post.rhat = {s.name: float(rhat[k]) for k, s in enumerate(slots)}
    try:
        post.ess = float(sum(np.atleast_1d(r.effective_sample_size()).min() for r in results))
    except Exception:  # noqa: BLE001
        post.ess = None
    _attach_convergence(post, slots, np.stack([u[:n] for u in us], axis=0), results)

    def predictive(n_, rng_):
        idx = rng_.randint(len(vals), size=n_)
        out = []
        for j in idx:
            d = build({s.index: float(vals[j, k]) for k, s in enumerate(slots)})
            out.append(d.sampler(seed=int(rng_.randint(1, 2**31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    post.build = build
    return RandomVariable._bound(build(mean_vals), name=rv._name, result=post)


# ----------------------------------------------------- inequality / region constraints
def _vals_from_u(slots, u) -> dict:
    """Constrained parameter values keyed by slot index, from one unconstrained vector ``u``.

    The exp / logit links are clamped so a wide derivative-free or penalized excursion in ``u`` cannot
    overflow (it saturates the constrained value instead of raising).
    """
    vals = {}
    for k, s in enumerate(slots):
        uk = float(u[k])
        if s.support == "positive":
            vals[s.index] = math.exp(min(uk, 700.0))
        elif s.support == "unit":
            vals[s.index] = 1.0 / (1.0 + math.exp(-max(min(uk, 700.0), -700.0)))
        else:
            vals[s.index] = uk
    return vals


def _handle_groups(slots):
    """Map ``id(handle) -> (handle, [slot indices in spec order])`` for every constrained handle.
    A scalar prior RV owns one slot; a ``param(...)`` vector/matrix handle owns several."""
    groups: dict = {}
    for s in slots:
        if s.handle is not None:
            groups.setdefault(id(s.handle), (s.handle, []))[1].append(s.index)
    return groups


def _check_constraint_handles(constraints, groups):
    for c in constraints:
        for lv in c.leaves:
            if id(lv) not in groups:
                raise ValueError(
                    "a constraint references an RV that is not a parameter of this model; constrain "
                    "a scalar prior (give the slot a prior) or a param(...) vector/matrix handle that "
                    "is also passed to the model."
                )


def _constraint_env(constraints, groups, vals):
    """Resolve each constrained handle to its value: a scalar prior -> its slot value; a
    ``param(...)`` handle -> its slots assembled into the vector/matrix via the bijector."""
    env = {}
    for c in constraints:
        for lv in c.leaves:
            if lv in env:
                continue
            handle, idxs = groups[id(lv)]
            spec = _spec_of(handle)
            env[lv] = _spec_assemble(spec, [vals[i] for i in idxs]) if spec is not None else vals[idxs[0]]
    return env


def _feasibility(constraints, slots):
    """Compile ``constraints`` (a Constraint or list) into ``feasible(u) -> bool`` over the
    model's parameter slots. Constraint variables must be the model's prior RVs (slot handles)."""
    if constraints is None:
        return None
    if isinstance(constraints, Constraint):
        constraints = [constraints]
    constraints = list(constraints)
    if not constraints:
        return None
    groups = _handle_groups(slots)
    _check_constraint_handles(constraints, groups)

    def feasible(u):
        env = _constraint_env(constraints, groups, _vals_from_u(slots, u))
        return all(bool(np.all(_row_mask(c.eval(env)))) for c in constraints)

    return feasible


def _project_init(u0, feasible, rng):
    """Nudge an initial point into the feasible region by growing random jitter."""
    if feasible(u0):
        return u0
    base = np.maximum(np.abs(u0), 1.0)
    for t in range(20000):
        cand = u0 + (0.1 + 0.02 * t) * base * rng.standard_normal(len(u0))
        if feasible(cand):
            return cand
    raise ValueError(
        "could not find a parameter point satisfying the constraints; "
        "check that the region is non-empty and consistent with the supports."
    )


def _constrain_target(log_target, feasible):
    """Wrap a log-target so it is -inf outside the feasible region (samplers reject it; the
    region is a hard truncation of the joint posterior)."""
    if feasible is None:
        return log_target

    def clt(u):
        return log_target(u) if feasible(u) else -np.inf

    return clt


_DEFAULT_PENALTY = 1000.0  # tight enough that an auto-penalized equality is effectively enforced


def _auto_penalty(constraints, penalty):
    """Effective penalty weight: the user's ``penalty`` if given, else a default when any
    constraint is *soft* (equality / ODE residual — measure-zero, so rejection can't honor it),
    else ``None`` (hard inequalities go to the feasible-region path). Lets ``fit(constraints=...)``
    just work without the user choosing rejection vs penalty."""
    if penalty is not None:
        return penalty
    if constraints is None:
        return None
    cs = [constraints] if isinstance(constraints, Constraint) else list(constraints)
    return _DEFAULT_PENALTY if any(getattr(c, "soft", False) for c in cs) else None


def _soft_penalty(constraints, slots, weight):
    """Compile ``constraints`` into a smooth penalty ``penalty(u) -> float`` (<= 0) over the model's
    parameter slots, using each constraint's continuous ``residual``.

    The penalty ``-0.5 * weight * sum(residual^2)`` is added to the joint log-target so gradient /
    MCMC inference can honor equality, convex, and algebraic relations (which hard rejection cannot).
    ``weight`` plays the role of an inverse tolerance: larger weights enforce the relation more tightly.
    """
    if constraints is None or weight is None:
        return None
    if isinstance(constraints, Constraint):
        constraints = [constraints]
    constraints = list(constraints)
    if not constraints:
        return None
    for c in constraints:
        if c.residual is None:
            raise ValueError(
                "a constraint has no smooth penalty surface (e.g. a negated/!= relation) and cannot be "
                "used with penalty=...; drop penalty to enforce it by rejection instead."
            )
    groups = _handle_groups(slots)
    _check_constraint_handles(constraints, groups)
    w = float(weight)
    if w <= 0.0:
        raise ValueError("penalty weight must be positive.")

    def penalty(u):
        env = _constraint_env(constraints, groups, _vals_from_u(slots, u))
        total = 0.0
        for c in constraints:
            r = np.atleast_1d(np.asarray(c.residual(env), dtype=float)).ravel()
            total += float(np.sum(r * r))
        return -0.5 * w * total

    return penalty


def _potential_term(potentials, slots):
    """Compile custom potentials into an additive ``term(u) -> float`` over the model's parameter slots.

    Each :class:`~mixle.ppl.core._Potential` resolves its ``vars`` (named prior RVs or ``param(...)``
    handles, exactly like constraint leaves) to their current values and adds ``fn(*values)`` to the
    joint log-target -- the same wrapping shape as :func:`_soft_penalty`, so the late-bound numerical
    gradient picks it up too. Returns ``None`` when there are no potentials.
    """
    if potentials is None:
        return None
    if isinstance(potentials, _Potential):
        potentials = [potentials]
    potentials = list(potentials)
    if not potentials:
        return None
    groups = _handle_groups(slots)
    for p in potentials:
        for v in p.vars:
            if id(v) not in groups:
                raise ValueError(
                    "a potential references an RV that is not a parameter of this model; reference a "
                    "scalar prior (a slot with a prior) or a param(...) handle that is also passed to the "
                    "model -- as with constraints."
                )

    def term(u):
        vals = _vals_from_u(slots, u)
        total = 0.0
        for p in potentials:
            args = []
            for v in p.vars:
                handle, idxs = groups[id(v)]
                spec = _spec_of(handle)
                args.append(_spec_assemble(spec, [vals[i] for i in idxs]) if spec is not None else vals[idxs[0]])
            total += float(p.fn(*args))
        return total

    return term


def _penalize_target(log_target, penalty):
    """Add a smooth penalty term to a log-target (soft constraints)."""
    if penalty is None:
        return log_target

    def plt(u):
        return log_target(u) + penalty(u)

    return plt


# -------------------------- hierarchical priors & grouped (random-intercept) models
def _is_grouped(rv: RandomVariable) -> bool:
    """A random-intercept model: a flat family whose slot-0 prior is a ``.each()`` group prior."""
    return (
        rv._kind == "sample"
        and not isinstance(rv._family, CompositeFamily)
        and len(rv._args) >= 1
        and isinstance(rv._args[0], RandomVariable)
        and rv._args[0]._scope == "grouped"
    )


# Backend-parameterized log-densities (one body, evaluated with numpy for the scalar target and torch
# for the gradient, so the two can never drift). ``xp`` is numpy or torch. Distribution parameters may
# be python-float constants or tensors; ``_xlog`` logs either, so a constant never reaches torch.log.
def _xlog(v, xp):
    if xp is np:
        return np.log(v)
    return xp.log(v) if xp.is_tensor(v) else math.log(v)


def _lp_normal(x, m, s, xp):
    return -0.9189385332046727 - _xlog(s, xp) - 0.5 * ((x - m) / s) ** 2


def _lp_gamma(x, k, theta, xp):
    # k, theta are constants here -> their log / gammaln terms are constants computed in numpy
    return (k - 1.0) * _xlog(x, xp) - x / theta - (k * math.log(theta) + float(_gammaln(k)))


def _lp_halfnormal(x, s, xp):
    # log sqrt(2/pi) = -0.22579...: the folded-Normal doubling (+log 2) net of the Normal constant.
    return -0.2257913526447274 - _xlog(s, xp) - 0.5 * (x / s) ** 2


def _lp_exponential(x, rate, xp):
    return math.log(rate) - rate * x


_HYPER_LP = {  # prior family name -> log-density(value, const-args, xp)
    "Normal": lambda v, a, xp: _lp_normal(v, a[0], a[1], xp),
    "Gamma": lambda v, a, xp: _lp_gamma(v, a[0], a[1], xp),
    "HalfNormal": lambda v, a, xp: _lp_halfnormal(v, a[0], xp),
    "Exponential": lambda v, a, xp: _lp_exponential(v, a[0], xp),
    "InverseGamma": lambda v, a, xp: _lp_gamma(1.0 / v, a[0], 1.0 / a[1], xp) - 2.0 * _xlog(v, xp),
}


def _grouped_target(rv: RandomVariable, data, want_grad: bool):
    """Build the joint NUTS/HMC/ensemble target for a Normal-Normal random-intercept model.

    ``y_gi ~ Normal(theta_g, sigma)``, ``theta_g ~ Normal(mu, tau)``, with ``mu``/``tau``/``sigma`` each a
    constant, ``free``, or a (flat) prior, and the per-group ``theta`` sampled centered or, if the group
    prior was marked ``.noncentered()``, as ``mu + tau*z``. Returns ``(log_target, grad, slots, build,
    dmean, dstd)`` compatible with the existing finalize/Posterior path -- so the per-group latents and
    hyperparameters all appear in the posterior summary.
    """
    if rv._family.name != "Normal":
        raise NotImplementedError("grouped NUTS currently supports a Normal likelihood.")
    prior = rv._args[0]
    if prior._family.name != "Normal":
        raise NotImplementedError("grouped NUTS currently supports a Normal group prior.")
    noncentered = prior._reparam == "loc_scale"
    groups = [np.asarray(g, dtype=float) for g in data]
    G = len(groups)
    if G < 1:
        raise ValueError("grouped model needs at least one group of observations.")
    counts = np.array([len(g) for g in groups], dtype=float)
    sums = np.array([g.sum() for g in groups])
    sumsq = np.array([float(np.sum(g * g)) for g in groups])
    flat = np.concatenate(groups) if any(counts) else np.zeros(1)
    dmean, dstd = float(flat.mean()), float(flat.std() or 1.0)

    # Hyperparameters mu (slot 0 of prior), tau (slot 1 of prior), sigma (slot 1 of rv).
    specs = [("mu", prior._args[0], "real"), ("tau", prior._args[1], "positive"), ("sigma", rv._args[1], "positive")]
    slots: list[_Slot] = []
    hyper_cols: dict[str, int] = {}
    col = 0
    for nm, arg, support in specs:
        if isinstance(arg, RandomVariable) or arg is free:
            prior_dist = lower(arg, target="dist") if isinstance(arg, RandomVariable) else None
            slots.append(
                _Slot(
                    col,
                    prior_dist,
                    support == "positive",
                    arg.name if isinstance(arg, RandomVariable) and arg.name else nm,
                    arg if isinstance(arg, RandomVariable) else None,
                    support,
                )
            )
            hyper_cols[nm] = col
            col += 1
        else:
            hyper_cols[nm] = -1  # constant
    n_hyper = col
    const = {nm: float(arg) for (nm, arg, _support) in specs if hyper_cols[nm] < 0}
    # per-group latent slots; non-centered ones reconstruct theta = loc + scale*z via the shared mu/tau
    pa = {}
    if hyper_cols["mu"] >= 0:
        pa[0] = hyper_cols["mu"]
    if hyper_cols["tau"] >= 0:
        pa[1] = hyper_cols["tau"]
    for g in range(G):
        if noncentered:
            slots.append(
                _Slot(
                    n_hyper + g,
                    None,
                    False,
                    f"{prior.name or 'theta'}[{g}]",
                    prior,
                    "real",
                    parent_args=dict(pa),
                    reparam="loc_scale",
                )
            )
        else:
            slots.append(_Slot(n_hyper + g, None, False, f"{prior.name or 'theta'}[{g}]", None, "real"))

    def _eval(u, xp):
        # hyperparameters (unconstrained -> constrained, with Jacobian for the positive ones)
        logj = u[0] * 0.0
        h = {}
        for nm, _arg, support in specs:
            c = hyper_cols[nm]
            if c < 0:
                h[nm] = const[nm]
            elif support == "positive":
                h[nm] = xp.exp(u[c])
                logj = logj + u[c]
            else:
                h[nm] = u[c]
        mu, tau, sigma = h["mu"], h["tau"], h["sigma"]
        lp = logj
        # hyperpriors
        for nm, arg, _support in specs:
            if isinstance(arg, RandomVariable):
                lp = lp + _HYPER_LP[arg._family.name](h[nm], [float(z) for z in arg._args], xp)
        # per-group latents + likelihood (vectorized over groups)
        z = u[n_hyper : n_hyper + G]
        theta = mu + tau * z if noncentered else z
        cnt = xp.asarray(counts) if xp is np else z.new_tensor(counts)
        sm = xp.asarray(sums) if xp is np else z.new_tensor(sums)
        ssq = xp.asarray(sumsq) if xp is np else z.new_tensor(sumsq)
        # sum_i (y - theta)^2 = sumsq - 2 theta sum + count theta^2
        sse = ssq - 2.0 * theta * sm + cnt * theta * theta
        ll = xp.sum(-0.5 * cnt * 1.8378770664093453 - cnt * _xlog(sigma, xp) - 0.5 * sse / (sigma * sigma))
        prior_z = xp.sum(_lp_normal(z, 0.0, 1.0, xp)) if noncentered else xp.sum(_lp_normal(theta, mu, tau, xp))
        return lp + ll + prior_z

    def log_target(u):
        v = float(_eval(np.asarray(u, dtype=float), np))
        return v if math.isfinite(v) else _NEG_INF

    grad = None
    if want_grad:
        try:
            import torch

            def grad(u):  # noqa: F811
                t = torch.tensor(np.asarray(u, dtype=float), dtype=torch.float64, requires_grad=True)
                val = _eval(t, torch)
                (g,) = torch.autograd.grad(val, t)
                return g.detach().numpy()
        except Exception:  # noqa: BLE001
            grad = None

    def build(vals):
        mu = vals[hyper_cols["mu"]] if hyper_cols["mu"] >= 0 else const["mu"]
        tau = vals[hyper_cols["tau"]] if hyper_cols["tau"] >= 0 else const["tau"]
        return prior._family.make_dist((float(mu), float(tau)), prior._name)

    return log_target, grad, slots, build, dmean, dstd


# ----------------------------- sampler setup + the general how= drivers (MCMC, MAP, VI)
def _prepare_target(rv, data, constraints, penalty, *, want_grad, numpy_only=False, missing="error", potentials=None):
    """Shared sampler setup: build the joint log-target (analytic-Torch when available, else the
    numeric encoder target), optionally its gradient, then layer on constraints/penalty.

    Returns ``(log_target, grad, slots, build, dmean, dstd, feasible)``. The gradient closure binds
    ``log_target`` late, so a soft penalty added afterwards also enters the gradient. ``numpy_only``
    forces the encoder target (the ensemble sampler wants the fast scalar NumPy eval); ``want_grad``
    is for HMC/NUTS. Used by every sampler fit and its parallel worker — one place, no duplication."""
    from mixle.ppl import autograd as _ag

    eff = _auto_penalty(constraints, penalty)
    if _is_grouped(rv):  # random-intercept / plate model: per-group latents + shared hyperparameters
        log_target, grad, slots, build, dmean, dstd = _grouped_target(rv, data, want_grad=want_grad)
        soft = _soft_penalty(constraints, slots, eff)
        feasible = None if soft is not None else _feasibility(constraints, slots)
        log_target = _constrain_target(log_target, feasible)
        log_target = _penalize_target(log_target, soft)
        log_target = _penalize_target(log_target, _potential_term(potentials, slots))
        return log_target, grad, slots, build, dmean, dstd, feasible
    # A custom potential is scored on the numerical target (no autograd graph for an arbitrary fn), so
    # force the encoder target when one is present -- exactly as a soft penalty does.
    ag = (
        None
        if (numpy_only or eff is not None or potentials is not None)
        else _ag.grad_target(rv, data, missing=missing)
    )
    if ag is None and missing == "marginalize":
        raise NotImplementedError(
            "missing='marginalize' on this path needs the Torch autograd target (flat model, no "
            "constraints). For this model build it with mixle.stats.marginalized() leaves instead."
        )
    if ag is not None:
        slots, build, dmean, dstd = ag.slots, ag.build, ag.dmean, ag.dstd
        log_target = ag.log_target
        grad = ag.grad if want_grad else None
    else:
        log_target, slots, _fam, build, _unpack, (dmean, dstd) = _build_target(rv, data, _potential_latents(potentials))
        grad = None
        if want_grad:
            eps = 1e-5 * np.maximum(np.abs(_init_u(slots, dmean, dstd)), 1.0)

            def grad(u):  # late binding of log_target -> the penalty (added below) enters the gradient
                u = np.asarray(u, dtype=float)
                g = np.empty(len(u))
                for i in range(len(u)):
                    up = u.copy()
                    up[i] += eps[i]
                    um = u.copy()
                    um[i] -= eps[i]
                    g[i] = (log_target(up) - log_target(um)) / (2.0 * eps[i])
                return g

    soft = _soft_penalty(constraints, slots, eff)
    feasible = None if soft is not None else _feasibility(constraints, slots)
    log_target = _constrain_target(log_target, feasible)
    log_target = _penalize_target(log_target, soft)
    log_target = _penalize_target(log_target, _potential_term(potentials, slots))
    return log_target, grad, slots, build, dmean, dstd, feasible


def ensemble_fit(
    rv: RandomVariable,
    data,
    *,
    draws: int = 1500,
    burn: int = 500,
    thin: int = 1,
    walkers: int | None = None,
    constraints=None,
    penalty=None,
    potentials=None,
    rng=None,
    chains: int = 1,
    parallel: bool = False,
    missing: str = "error",
) -> RandomVariable:
    """Affine-invariant ensemble MCMC (Goodman & Weare stretch move).

    A population of walkers samples jointly with no per-dimension step tuning; it is invariant
    to affine rescalings, so it mixes well on correlated / poorly-scaled posteriors and gives
    very high ESS/sec on low/medium-dimensional models (no JIT-compile latency). Each ``draws``
    sweep contributes all ``walkers`` states, so the pooled posterior has ``draws*walkers``
    near-independent samples. Uses the fast NumPy scalar log-target (one eval per proposal).
    ``chains>1`` runs independent ensembles for Gelman-Rubin R-hat / pooled ESS (``parallel``
    spreads them over a process pool)."""
    from mixle.inference.mcmc import affine_invariant_ensemble

    if rng is None:
        rng = np.random.RandomState()
    log_target, _grad, slots, build, dmean, dstd, feasible = _prepare_target(
        rv, data, constraints, penalty, want_grad=False, numpy_only=True, potentials=potentials
    )
    d = len(slots)
    if walkers is None:
        walkers = max(2 * (d + 1), 8)
    if walkers % 2:
        walkers += 1
    if constraints is not None or potentials is not None:
        parallel = False  # process workers rebuild the target without the constraint/penalty/potential closure

    def run_one(seed):
        crng = np.random.RandomState(seed)
        p0 = _ensemble_p0(slots, dmean, dstd, len(data), walkers, crng)
        if feasible is not None:  # every walker must start feasible (finite log-target)
            p0[0] = _project_init(p0[0], feasible, crng)
            for k in range(walkers):
                if not feasible(p0[k]):
                    p0[k] = _project_init(p0[0], feasible, crng)
        return affine_invariant_ensemble(log_target, p0, num_samples=draws, burn_in=burn, thin=thin, rng=crng)

    if chains == 1:
        return _finalize(rv, slots, run_one(int(rng.randint(1, 2**31))), build)
    kw = {"draws": draws, "burn": burn, "thin": thin, "walkers": walkers}
    results = _run_chains(run_one, _ensemble_worker, (rv, data, kw), chains, parallel, rng)
    return _finalize_chains(rv, slots, results, build)


def mcmc_fit(
    rv: RandomVariable,
    data,
    *,
    draws: int = 2000,
    burn: int = 1000,
    thin: int = 1,
    scale: float | None = None,
    rng=None,
    chains: int = 1,
    parallel: bool = False,
    constraints=None,
    penalty=None,
    potentials=None,
    missing: str = "error",
) -> RandomVariable:
    """Fit a PPL model with adaptive random-walk Metropolis-Hastings."""
    from mixle.inference.mcmc import AdaptiveRandomWalkProposal, metropolis_hastings

    if rng is None:
        rng = np.random.RandomState()
    log_target, _grad, slots, build, dmean, dstd, feasible = _prepare_target(
        rv, data, constraints, penalty, want_grad=False, missing=missing, potentials=potentials
    )
    u0 = _init_u(slots, dmean, dstd)
    if feasible is not None:
        u0 = _project_init(u0, feasible, rng)
    if constraints is not None or potentials is not None:
        parallel = False  # constraint/penalty/potential closures do not survive process pickling
    init_scale = (scale * np.ones(len(u0))) if scale is not None else _init_scale(slots, dstd, len(data))

    def run_one(seed):
        proposal = AdaptiveRandomWalkProposal(init_scale.copy())  # per-chain adaptive state
        return metropolis_hastings(
            log_target, u0, proposal, num_samples=draws, burn_in=burn, thin=thin, rng=np.random.RandomState(seed)
        )

    if chains == 1:
        return _finalize(rv, slots, run_one(int(rng.randint(1, 2**31))), build)
    kw = {"draws": draws, "burn": burn, "thin": thin, "scale": scale}
    results = _run_chains(run_one, _mcmc_worker, (rv, data, kw), chains, parallel, rng)
    return _finalize_chains(rv, slots, results, build)


def hmc_fit(
    rv: RandomVariable,
    data,
    *,
    draws: int = 1000,
    burn: int = 500,
    step_size: float | None = None,
    num_steps: int = 15,
    thin: int = 1,
    rng=None,
    chains: int = 1,
    parallel: bool = False,
    constraints=None,
    penalty=None,
    potentials=None,
    missing: str = "error",
) -> RandomVariable:
    """Hamiltonian Monte Carlo over the parameter posterior.

    Uses mixle's ``hamiltonian_monte_carlo`` with a numerical gradient of the joint
    log-target and a diagonal mass matrix preconditioned to the data-informed posterior
    scale, so trajectories are well-conditioned without manual tuning. Inequality
    ``constraints`` truncate the posterior (trajectories leaving the region are rejected);
    for hard constraints ``how='ensemble'`` or ``'mcmc'`` usually mixes better.
    """
    from mixle.inference.mcmc import hamiltonian_monte_carlo

    if rng is None:
        rng = np.random.RandomState()
    log_target, grad, slots, build, dmean, dstd, feasible = _prepare_target(
        rv, data, constraints, penalty, want_grad=True, missing=missing, potentials=potentials
    )
    u0 = _init_u(slots, dmean, dstd)
    if feasible is not None:
        u0 = _project_init(u0, feasible, rng)
    if constraints is not None or potentials is not None:
        parallel = False  # constraint/penalty/potential closures do not survive process pickling
    mass = 1.0 / (_init_scale(slots, dstd, len(data)) ** 2)  # precondition: M ~ inverse posterior cov
    if step_size is None:
        step_size = 2.5 / num_steps  # tuned: acc~0.98, near-max ESS (preconditioned)

    def run_one(seed):
        return hamiltonian_monte_carlo(
            log_target,
            grad,
            u0,
            num_samples=draws,
            step_size=step_size,
            num_steps=num_steps,
            mass=mass,
            burn_in=burn,
            thin=thin,
            rng=np.random.RandomState(seed),
        )

    if chains == 1:
        return _finalize(rv, slots, run_one(int(rng.randint(1, 2**31))), build)
    kw = {"draws": draws, "burn": burn, "thin": thin, "step_size": step_size, "num_steps": num_steps}
    results = _run_chains(run_one, _hmc_worker, (rv, data, kw), chains, parallel, rng)
    return _finalize_chains(rv, slots, results, build)


def nuts_fit(
    rv: RandomVariable,
    data,
    *,
    draws: int = 1000,
    burn: int = 1000,
    thin: int = 1,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    rng=None,
    chains: int = 1,
    parallel: bool = False,
    constraints=None,
    penalty=None,
    potentials=None,
    missing: str = "error",
) -> RandomVariable:
    """No-U-Turn Sampler over the parameter posterior — auto-tuned HMC (trajectory length +
    dual-averaging step size), the default choice for correlated / higher-dimensional posteriors.

    Uses the analytic Torch gradient when available (else a numeric gradient), preconditioned by a
    diagonal mass matrix at the data-informed scale. ``warmup`` (= ``burn``) adapts the step size
    to ``target_accept``. Inequality ``constraints`` truncate the posterior; ``penalty`` adds a
    smooth penalty (which enters the gradient via the numeric-gradient path)."""
    from mixle.inference.mcmc import nuts

    if rng is None:
        rng = np.random.RandomState()
    log_target, grad, slots, build, dmean, dstd, feasible = _prepare_target(
        rv, data, constraints, penalty, want_grad=True, missing=missing, potentials=potentials
    )
    u0 = _init_u(slots, dmean, dstd)
    if feasible is not None:
        u0 = _project_init(u0, feasible, rng)
    if constraints is not None or potentials is not None:
        parallel = False  # constraint/penalty/potential closures do not survive process pickling
    mass = 1.0 / (_init_scale(slots, dstd, len(data)) ** 2)  # precondition ~ inverse posterior cov

    def run_one(seed):
        return nuts(
            log_target,
            grad,
            u0,
            num_samples=draws,
            warmup=burn,
            mass=mass,
            target_accept=target_accept,
            max_tree_depth=max_tree_depth,
            thin=thin,
            rng=np.random.RandomState(seed),
        )

    if chains == 1:
        return _finalize(rv, slots, run_one(int(rng.randint(1, 2**31))), build)
    kw = {"draws": draws, "burn": burn, "thin": thin, "target_accept": target_accept, "max_tree_depth": max_tree_depth}
    results = _run_chains(run_one, _nuts_worker, (rv, data, kw), chains, parallel, rng)
    return _finalize_chains(rv, slots, results, build)


def sample_fit(rv: RandomVariable, data, **kw) -> RandomVariable:
    """Draw the parameter posterior with an automatically chosen sampler.

    ``how='sample'`` selects among MCMC, HMC, NUTS, and ensemble samplers.
    Low- and medium-dimensional models use the affine-invariant ensemble
    sampler; larger models use NUTS for correlated, higher-dimensional
    posteriors. Common controls include ``draws``, ``burn``, ``thin``,
    ``rng``, ``chains``, ``parallel``, ``constraints``, and ``penalty``.
    """
    d = len(_target_parts(rv, data)[1])
    return ensemble_fit(rv, data, **kw) if d <= 12 else nuts_fit(rv, data, **kw)


def map_fit(
    rv: RandomVariable, data, *, rng=None, constraints=None, penalty=None, potentials=None, missing="error"
) -> RandomVariable:
    """Fit a PPL model by maximum a posteriori optimization.

    The point estimate maximizes ``log p(data | theta) + log p(theta)`` in the *constrained*
    parameter space -- no support-transform log-Jacobian -- so with flat priors MAP coincides
    with the MLE (e.g. the ``sqrt(S/n)`` sd for a Normal). The samplers and VI/Laplace keep the
    Jacobian: they need the unconstrained-space density.
    """
    from scipy.optimize import minimize

    from mixle.ppl import autograd as _ag

    g = None if potentials is not None else _ag.grad_target(rv, data, missing=missing, jacobian=False)
    if g is None and constraints is None and penalty is None and potentials is None and not _ag.torch_available():
        # Without Torch, the analytic-gradient L-BFGS path is unavailable and MAP uses a derivative-free
        # optimizer. Warn once rather than letting callers infer the route from timing.
        import warnings as _warnings

        _warnings.warn(
            "MAP is using a derivative-free optimizer because PyTorch is not installed; install torch for "
            "faster, more accurate analytic-gradient MAP (pip install 'mixle[torch]').",
            RuntimeWarning,
            stacklevel=2,
        )
    if g is None and missing == "marginalize" and potentials is None:
        raise NotImplementedError(
            "missing='marginalize' needs the Torch autograd target (flat model, no constraints); for this "
            "model build it with mixle.stats.marginalized() leaves."
        )
    if g is not None and constraints is None and penalty is None:
        # analytic-gradient MAP: L-BFGS on the joint posterior (fast, scales with #params)
        u0 = _init_u(g.slots, g.dmean, g.dstd)

        def neg(u):
            v, gr = g.value_and_grad(u)
            return -v, -gr

        res = minimize(neg, u0, jac=True, method="L-BFGS-B", options={"maxiter": 1000})
        vals, _ = g.unpack(res.x)
        return RandomVariable._bound(g.build(vals), name=rv._name)

    # derivative-free path (no Torch, an unsupported family, or a constrained / penalized region)
    log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(
        rv, data, _potential_latents(potentials), jacobian=False
    )
    soft = _soft_penalty(constraints, slots, _auto_penalty(constraints, penalty))
    # penalty=... enforces the constraints softly (a log-joint term), so it replaces hard rejection.
    feasible = None if soft is not None else _feasibility(constraints, slots)
    log_target = _penalize_target(log_target, soft)
    log_target = _penalize_target(log_target, _potential_term(potentials, slots))
    u0 = _init_u(slots, dmean, dstd)
    if feasible is not None:
        u0 = _project_init(u0, feasible, np.random.RandomState() if rng is None else rng)

    def objective(u):
        if feasible is not None and not feasible(u):
            return 1e18  # keep the constrained MAP inside the feasible region
        return -log_target(u)

    res = minimize(objective, u0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    vals, _ = unpack(res.x)
    return RandomVariable._bound(build(vals), name=rv._name)


# ----------------------------------------------------- Laplace: a local Gaussian posterior at the MAP
class _LaplaceRaw:
    """Minimal raw-result holder so a Laplace fit flows through the shared :func:`_finalize` path."""

    def __init__(self, samples):
        self.samples = samples  # (draws, d) unconstrained
        self.acceptance_rate = None
        self.num_divergences = 0


def _fd_hessian(f, x, eps=1.0e-4):
    """Symmetric finite-difference Hessian of scalar ``f`` at ``x`` (small parameter vectors only)."""
    x = np.asarray(x, dtype=float)
    d = x.size
    h = np.maximum(np.abs(x), 1.0) * eps
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += h[i]
            xpp[j] += h[j]
            xpm[i] += h[i]
            xpm[j] -= h[j]
            xmp[i] -= h[i]
            xmp[j] += h[j]
            xmm[i] -= h[i]
            xmm[j] -= h[j]
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h[i] * h[j])
    return H


def _psd_cov(prec):
    """Invert a (negative-log-posterior) precision into a PSD covariance, robust to finite-diff noise."""
    d = prec.shape[0]
    prec = 0.5 * (prec + prec.T) + 1.0e-8 * np.eye(d)
    try:
        cov = np.linalg.inv(prec)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(prec)
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)  # clip to PSD so sampling is well-defined
    w = np.clip(w, 1.0e-12, None)
    return (V * w) @ V.T


def laplace_fit(rv: RandomVariable, data, *, rng=None, draws: int = 2000, missing="error", **_) -> RandomVariable:
    """Gaussian (Laplace) posterior approximation at the MAP.

    Finds the posterior mode, takes the Gaussian whose precision is the Hessian of the negative joint
    log-density there, and draws from it in the unconstrained space (so positivity/simplex supports are
    respected after the link transform). Returns a full :class:`Posterior` (summary / intervals /
    predictive), so ``how='laplace'`` gives credible intervals where ``how='map'`` gives only a point.
    """
    from scipy.optimize import minimize

    rng = np.random.RandomState() if rng is None else rng
    log_target, slots, _fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
    neg = lambda u: -log_target(u)  # noqa: E731
    u0 = _init_u(slots, dmean, dstd)
    res = minimize(neg, u0, method="L-BFGS-B", options={"maxiter": 2000})
    u_star = np.asarray(res.x, dtype=float)
    cov = _psd_cov(_fd_hessian(neg, u_star))
    Z = rng.multivariate_normal(u_star, cov, size=int(draws))
    return _finalize(rv, slots, _LaplaceRaw(Z), build)


class _VIResult:
    """Lightweight raw-result holder for a variational fit (mirrors MCMCResult's role)."""

    def __init__(self, elbo, mean, std, objective_kind="kl_elbo", alpha=1.0, family="meanfield", batch_size=None):
        self.elbo = float(elbo)
        self.objective = float(elbo)  # alias; for alpha != 1 this is the tilted Renyi bound, not the ELBO
        self.objective_kind = objective_kind
        self.alpha = float(alpha)
        self.family = family
        self.batch_size = None if batch_size is None else int(batch_size)
        self.variational_mean = mean
        self.variational_std = std
        self.acceptance_rate = None


def vi_fit(
    rv: RandomVariable,
    data,
    *,
    samples: int = 4000,
    mc: int = 16,
    max_iter: int = 4000,
    steps: int = 600,
    lr: float = 0.05,
    batch_size: int | None = None,
    family: str = "meanfield",
    alpha: float = 1.0,
    rng=None,
    seed: int | None = None,
    missing: str = "error",
) -> RandomVariable:
    """Variational Bayes (ADVI) — a Gaussian variational posterior fit by reparameterized-MC Adam.

    ``family='meanfield'`` (diagonal q, default) or ``'fullrank'`` (full covariance via a Cholesky
    factor, capturing posterior correlations). ``alpha`` selects the tilted Renyi objective:
    ``alpha=1`` is the KL-ELBO (default), ``alpha=0`` the importance-weighted (IWAE) bound, and
    ``alpha<1`` is mass-covering (widens the often-too-narrow KL fit). ``batch_size`` subsamples the
    data per step (SGVB). Without Torch it falls back to derivative-free mean-field ELBO. Works for
    *non-conjugate* priors; returns a variational Posterior with draws and posterior-predictive.
    ``seed`` makes the fit deterministic when ``rng`` is not given (``fit(how='vi', seed=...)``,
    matching the samplers' seeding).
    """
    from mixle.ppl import autograd as _ag

    if rng is None:
        rng = np.random.RandomState(seed)

    ag = _ag.grad_target(rv, data, missing=missing)
    if ag is None and missing == "marginalize":
        raise NotImplementedError(
            "missing='marginalize' needs the Torch autograd target (flat model); for this model build it "
            "with mixle.stats.marginalized() leaves."
        )
    if ag is not None:
        slots, build = ag.slots, ag.build
        u0 = _init_u(slots, ag.dmean, ag.dstd)
        s0 = _init_scale(slots, ag.dstd, len(data))
        vals, mean, std, objective = ag.advi(
            u0,
            s0,
            samples=samples,
            mc=mc,
            steps=steps,
            lr=lr,
            rng=rng,
            batch_size=batch_size,
            family=family,
            alpha=alpha,
        )
        objective_kind = "kl_elbo" if alpha == 1.0 else "renyi_tilted"
    else:
        from scipy.optimize import minimize

        log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
        d = len(slots)
        u0 = _init_u(slots, dmean, dstd)
        s0 = _init_scale(slots, dstd, len(data))
        eps = rng.standard_normal((mc, d))  # common random numbers
        half_entropy_const = 0.5 * d * (1.0 + math.log(2.0 * math.pi))

        def neg_elbo(phi):
            mean, log_std = phi[:d], phi[d:]
            std = np.exp(log_std)
            U = mean + std * eps
            ll = float(np.mean([log_target(U[i]) for i in range(mc)]))
            return -(ll + float(np.sum(log_std)) + half_entropy_const)

        res = minimize(
            neg_elbo,
            np.concatenate([u0, np.log(s0)]),
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-5},
        )
        mean, std = res.x[:d], np.exp(res.x[d:])
        Z = rng.standard_normal((samples, d))
        U = mean + std * Z
        # map unconstrained samples back per slot support (exp/sigmoid/identity); the old hand-rolled
        # branch only handled positive support, passing unit-support (Beta/Bernoulli) values through unbounded
        vals = _u_to_vals(slots, U)
        objective = -float(res.fun)  # neg_elbo was minimized; the ELBO is its negation
        objective_kind = "kl_elbo_common_random"

    mean_vals = {s.index: float(vals[:, k].mean()) for k, s in enumerate(slots)}
    post = Posterior(
        slots,
        vals,
        _VIResult(
            objective, mean, std, objective_kind=objective_kind, alpha=alpha, family=family, batch_size=batch_size
        ),
    )

    def predictive(n, r):
        idx = r.randint(len(vals), size=n)
        out = []
        for j in idx:
            dd = build({s.index: float(vals[j, k]) for k, s in enumerate(slots)})
            out.append(dd.sampler(seed=int(r.randint(1, 2**31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    post.build = build
    return RandomVariable._bound(build(mean_vals), name=rv._name, result=post)


# ---------------------------------------------------- closed-form conjugate Bayes
class ConjugatePosterior:
    """Exact closed-form posterior over a conjugate parameter.

    ``post`` maps parameter name -> {mean, sample(n, rng), name, hyper}. This is the
    ideal case VB approximates: exact, instant, no iteration.
    """

    def __init__(self, post: dict):
        self.post = post
        self.acceptance_rate = None
        self.predictive = None

    def _entry(self, param):
        for nm, e in self.post.items():
            if param == nm or param == e["index"] or param is e["handle"]:
                return e
        raise KeyError(f"no conjugate parameter matching {param!r}")

    def samples(self, param=None, n: int = 4000, rng=None):
        """Draw samples from the exact conjugate posterior."""
        rng = rng or np.random.RandomState()
        if param is None:
            return {nm: e["sample"](n, rng) for nm, e in self.post.items()}
        return self._entry(param)["sample"](n, rng)

    def mean(self, param=None):
        """Return posterior means for all conjugate parameters or one parameter."""
        if param is None:
            return {nm: e["mean"] for nm, e in self.post.items()}
        return self._entry(param)["mean"]

    def summary(self) -> dict:
        """Return conjugate posterior family, hyperparameters, and posterior means."""
        return {nm: {"mean": e["mean"], "posterior": e["name"], "hyper": e["hyper"]} for nm, e in self.post.items()}


def _conj_normal_mean(prior_args, fixed, stats, handle, index):
    m0, s0 = float(prior_args[0]), float(prior_args[1])  # prior mean, sd
    sigma2 = float(fixed[1]) ** 2  # known variance (slot 1)
    n, sx = stats["n"], stats["sum"]
    prec = 1.0 / s0**2 + n / sigma2
    pm = (m0 / s0**2 + sx / sigma2) / prec
    pv = 1.0 / prec
    return {
        "index": index,
        "handle": handle,
        "name": "Normal",
        "mean": pm,
        "hyper": {"mean": pm, "sd": math.sqrt(pv)},
        "sample": lambda k, rng: rng.normal(pm, math.sqrt(pv), k),
    }


def _conj_poisson_gamma(prior_args, fixed, stats, handle, index):
    a, b = float(prior_args[0]), float(prior_args[1])  # Gamma(shape, rate) prior
    n, sx = stats["n"], stats["sum"]
    A, B = a + sx, b + n
    return {
        "index": index,
        "handle": handle,
        "name": "Gamma",
        "mean": A / B,
        "hyper": {"shape": A, "rate": B},
        "sample": lambda k, rng: rng.gamma(A, 1.0 / B, k),
    }


def _conj_exponential_gamma(prior_args, fixed, stats, handle, index):
    a, b = float(prior_args[0]), float(prior_args[1])  # Gamma prior on rate
    n, sx = stats["n"], stats["sum"]
    A, B = a + n, b + sx
    return {
        "index": index,
        "handle": handle,
        "name": "Gamma",
        "mean": A / B,
        "hyper": {"shape": A, "rate": B},
        "sample": lambda k, rng: rng.gamma(A, 1.0 / B, k),
    }


def _conj_bernoulli_beta(prior_args, fixed, stats, handle, index):
    a, b = float(prior_args[0]), float(prior_args[1])
    n, sx = stats["n"], stats["sum"]
    A, B = a + sx, b + n - sx
    return {
        "index": index,
        "handle": handle,
        "name": "Beta",
        "mean": A / (A + B),
        "hyper": {"a": A, "b": B},
        "sample": lambda k, rng: rng.beta(A, B, k),
    }


def _conj_categorical_dirichlet(prior_args, fixed, stats, handle, index):
    # Categorical over K integer categories 0..K-1 with probs ~ Dirichlet(alpha); posterior is
    # Dirichlet(alpha + counts). Returns the posterior-mean probability vector (not a scalar).
    alpha = np.asarray(prior_args[0], dtype=float).reshape(-1)
    K = alpha.size
    labels = np.asarray(stats["data"]).round().astype(int)
    if labels.size and (labels.min() < 0 or labels.max() >= K):
        raise ValueError(
            f"Categorical-Dirichlet conjugacy expects integer categories in [0, {K}); got values outside that "
            f"range. Build the prior with the right dimension, e.g. Categorical(Dirichlet(np.ones(K)))."
        )
    counts = np.bincount(labels, minlength=K)[:K].astype(float)
    post = alpha + counts
    return {
        "index": index,
        "handle": handle,
        "name": "Dirichlet",
        "mean": post / post.sum(),  # the K-vector of posterior-mean category probabilities
        "hyper": {"alpha": post},
        "sample": lambda k, rng: rng.dirichlet(post, k),
    }


def _conj_negbinomial_beta(prior_args, fixed, stats, handle, index):
    # NegativeBinomial(r known, p) with p ~ Beta(a, b); x = failure counts. Likelihood ~ p^(r n)(1-p)^(sum x)
    # -> posterior p ~ Beta(a + r*n, b + sum_x).
    a, b = float(prior_args[0]), float(prior_args[1])
    r = float(fixed[0])  # known successes (slot 0)
    n, sx = stats["n"], stats["sum"]
    A, B = a + r * n, b + sx
    return {
        "index": index,
        "handle": handle,
        "name": "Beta",
        "mean": A / (A + B),
        "hyper": {"a": A, "b": B},
        "sample": lambda k, rng: rng.beta(A, B, k),
    }


def _conj_gamma_rate(prior_args, fixed, stats, handle, index):
    # Gamma(shape=k known, rate=r) with r ~ Gamma(a, b) [shape a, rate b]; sufficient stat = sum_x.
    # Posterior rate ~ Gamma(a + n*k, b + sum_x).
    a, b = float(prior_args[0]), float(prior_args[1])
    k = float(fixed[0])  # known shape (slot 0)
    n, sx = stats["n"], stats["sum"]
    A, B = a + n * k, b + sx
    return {
        "index": index,
        "handle": handle,
        "name": "Gamma",
        "mean": A / B,  # posterior mean of the rate
        "hyper": {"a": A, "b": B},
        "sample": lambda kk, rng: rng.gamma(A, 1.0 / B, kk),
    }


def _conj_binomial_beta(prior_args, fixed, stats, handle, index):
    # Binomial(n, p) with p ~ Beta(a, b); n known (fixed slot 0). successes = sum_x,
    # failures = n*N - sum_x -> posterior Beta(a + successes, b + failures).
    a, b = float(prior_args[0]), float(prior_args[1])
    n_trials = float(fixed[0])
    N, sx = stats["n"], stats["sum"]
    A, B = a + sx, b + n_trials * N - sx
    return {
        "index": index,
        "handle": handle,
        "name": "Beta",
        "mean": A / (A + B),
        "hyper": {"a": A, "b": B},
        "sample": lambda k, rng: rng.beta(A, B, k),
    }


def _conj_geometric_beta(prior_args, fixed, stats, handle, index):
    # Geometric(p) on k>=1 with p ~ Beta(a, b): likelihood ∝ p^N (1-p)^(sum_x - N)
    # -> posterior Beta(a + N, b + sum_x - N).
    a, b = float(prior_args[0]), float(prior_args[1])
    N, sx = stats["n"], stats["sum"]
    A, B = a + N, b + sx - N
    return {
        "index": index,
        "handle": handle,
        "name": "Beta",
        "mean": A / (A + B),
        "hyper": {"a": A, "b": B},
        "sample": lambda k, rng: rng.beta(A, B, k),
    }


# (likelihood family, slot index, prior family) -> closed-form posterior builder
_CONJUGATE = {
    ("Normal", 0, "Normal"): _conj_normal_mean,  # unknown mean, known variance
    ("Poisson", 0, "Gamma"): _conj_poisson_gamma,
    ("Exponential", 0, "Gamma"): _conj_exponential_gamma,
    ("Bernoulli", 0, "Beta"): _conj_bernoulli_beta,
    ("Binomial", 1, "Beta"): _conj_binomial_beta,
    ("Geometric", 0, "Beta"): _conj_geometric_beta,
    ("Categorical", 0, "Dirichlet"): _conj_categorical_dirichlet,  # K-category counts -> Dirichlet posterior
    ("Gamma", 1, "Gamma"): _conj_gamma_rate,  # known shape, Gamma prior on the rate
    ("NegativeBinomial", 1, "Beta"): _conj_negbinomial_beta,  # known r, Beta prior on success prob
}


def conjugate_spec(rv: RandomVariable):
    """Return (builder, prior_slot_index, prior_rv) if exactly one slot is a conjugate
    prior and every other slot is a fixed constant; else None.
    """
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        return None
    prior_slots = [(i, a) for i, a in enumerate(rv._args) if isinstance(a, RandomVariable)]
    if len(prior_slots) != 1:
        return None
    if any(a is free for a in rv._args):
        return None  # other params must be known for textbook conjugacy
    i, prior_rv = prior_slots[0]
    if prior_rv._kind != "sample" or isinstance(prior_rv._family, CompositeFamily):
        return None
    key = (rv._family.name, i, prior_rv._family.name)
    builder = _CONJUGATE.get(key)
    if builder is None:
        return None
    return builder, i, prior_rv


def _is_all_free_normal(rv: RandomVariable) -> bool:
    """Normal(free, free) -- both mean and variance unknown, the Normal-Inverse-Gamma conjugate case."""
    return (
        rv._kind == "sample"
        and not isinstance(rv._family, CompositeFamily)
        and rv._family.name == "Normal"
        and len(rv._args) == 2
        and all(a is free for a in rv._args)
    )


def _nig_conjugate_fit(rv, data, *, mu0=0.0, kappa=0.0, alpha=1.0, beta=0.0) -> RandomVariable:
    """Closed-form Normal-Inverse-Gamma (NormalGamma) posterior for ``Normal(free, free)`` -- the most
    common Bayesian model: unknown mean AND variance, jointly conjugate.

    Prior ``(mu, tau) ~ NormalGamma(mu0, kappa, alpha, beta)`` (default weakly-informative); the posterior
    is exact in one pass. Returns a Gaussian at the posterior mean with a :class:`ConjugatePosterior`
    exposing the joint posterior over ``mu`` and ``sigma`` (sampled from the NIG).
    """
    from mixle.stats import GaussianDistribution
    from mixle.stats.bayes.normal_gamma import NormalGammaDistribution

    arr = np.asarray(data, dtype=float)
    n = float(arr.size)
    xbar = float(arr.mean()) if n else 0.0
    S = float(((arr - xbar) ** 2).sum())
    lam_n = kappa + n
    mu_n = (kappa * mu0 + n * xbar) / lam_n if lam_n > 0 else xbar
    a_n = alpha + n / 2.0
    b_n = beta + 0.5 * S + (0.5 * kappa * n / lam_n * (xbar - mu0) ** 2 if lam_n > 0 else 0.0)
    nig = NormalGammaDistribution(mu_n, lam_n, a_n, b_n)

    def _draw(k, rng):
        return np.asarray(nig.sampler(seed=int(rng.randint(1, 2**31))).sample(k), dtype=float).reshape(k, 2)

    mean_sigma2 = b_n / (a_n - 1.0) if a_n > 1.0 else b_n / a_n  # E[sigma^2] under the inverse-gamma marginal
    # E[sigma] under sigma^2 ~ InvGamma(a_n, b_n) is sqrt(b_n) * Gamma(a_n - 1/2) / Gamma(a_n)
    # (finite for a_n > 1/2); sqrt(E[sigma^2]) overstates it (Jensen).
    mean_sigma = (
        math.sqrt(b_n) * math.exp(math.lgamma(a_n - 0.5) - math.lgamma(a_n))
        if (a_n > 0.5 and b_n > 0.0)
        else float(np.sqrt(mean_sigma2))
    )
    entries = {
        "mu": {
            "index": 0,
            "handle": None,
            "name": "StudentT",
            "mean": mu_n,
            "hyper": {"mu_n": mu_n, "kappa_n": lam_n, "df": 2.0 * a_n},
            "sample": lambda k, rng: _draw(k, rng)[:, 0],
        },
        "sigma": {
            "index": 1,
            "handle": None,
            "name": "sqrt-InverseGamma",
            "mean": mean_sigma,
            "hyper": {"alpha_n": a_n, "beta_n": b_n},
            "sample": lambda k, rng: 1.0 / np.sqrt(_draw(k, rng)[:, 1]),
        },
    }
    fitted = GaussianDistribution(mu_n, mean_sigma2)
    cpost = ConjugatePosterior(entries)

    def predictive(k, rng):  # posterior predictive: draw (mu, tau) then x ~ Normal(mu, 1/tau)
        draws = _draw(k, rng)
        return np.array([rng.normal(mu, 1.0 / np.sqrt(tau)) for mu, tau in draws])

    cpost.predictive = predictive
    return RandomVariable._bound(fitted, name=rv._name, result=cpost)


# ------------------------------------------------ general exp-family conjugate bridge
# Rather than hand-code each posterior, delegate to mixle.stats.bayes.conjugate.conjugate_posterior --
# the exp-family-map-derived machinery that already gives closed-form posteriors for ~19 likelihood
# families (Bernoulli/Binomial/Geometric/Poisson/Exponential/Categorical/Gaussian/Gamma/InverseGamma/
# InverseGaussian/Pareto/NegativeBinomial/VonMises/Rayleigh/HalfNormal/...). The PPL side only has to
# (a) recognize the "one prior slot, the rest known" shape, (b) translate the prior RV's params into the
# family's hyperparameter dict, and (c) wrap the returned posterior. New conjugate families become
# available in how='auto' with no new posterior math here -- the derivation lives in the exp-family map.

# prior RV family -> its natural-hyperparameter dict (keyed as the stats builders read them)
_PRIOR_DICT_BUILDERS = {
    "Beta": lambda a: {"a": float(a[0]), "b": float(a[1])},
    "Gamma": lambda a: {"shape": float(a[0]), "rate": float(a[1])},
    "InverseGamma": lambda a: {"a": float(a[0]), "b": float(a[1])},
}


# Soundness guard for the bridge: a likelihood is only a conjugate pair with the *right* prior family.
# This maps each single-target conjugate likelihood (the bridge handles these) to the PPL prior family
# its conjugate posterior expects, so e.g. Normal(Beta, sd) (a Beta on a Gaussian mean -- NOT conjugate)
# is rejected here rather than routed to conjugate and crashing. Multi-parameter likelihoods
# (Gaussian/MVN/LogGaussian: joint NIG/NIW) are deliberately absent -- those are the hand table / NIG path.
_EXPECTED_PRIOR_FAMILY = {
    "BernoulliDistribution": "Beta",
    "BinomialDistribution": "Beta",
    "GeometricDistribution": "Beta",
    "NegativeBinomialDistribution": "Beta",
    "PoissonDistribution": "Gamma",
    "ExponentialDistribution": "Gamma",
    "GammaDistribution": "Gamma",
    "InverseGammaDistribution": "Gamma",
    "InverseGaussianDistribution": "Gamma",
    "ParetoDistribution": "Gamma",
    "CategoricalDistribution": "Dirichlet",
    "IntegerCategoricalDistribution": "Dirichlet",
}


def _prior_to_dict(prior_rv):
    builder = _PRIOR_DICT_BUILDERS.get(prior_rv._family.name)
    try:
        return builder(prior_rv._args) if builder is not None else None
    except Exception:  # noqa: BLE001
        return None


def _prior_probe_value(prior_rv):
    """A valid value for the likelihood's target slot (only the dist *type* + known params matter to
    conjugate_posterior; this just has to let make_dist succeed). A draw from the prior is in-domain."""
    try:
        d = prior_rv._family.make_dist(tuple(prior_rv._args), prior_rv._name)
        return float(np.ravel(d.sampler(seed=0).sample(1))[0])
    except Exception:  # noqa: BLE001
        return 1.0


def _stats_conjugate_probe(rv):
    """Return (slot_index, prior_rv, stats_likelihood) when ``rv`` is a single-prior-slot conjugate the
    stats machinery recognizes, with a known prior->hyperparameter mapping; else None. No data needed."""
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        return None
    prior_slots = [(i, a) for i, a in enumerate(rv._args) if isinstance(a, RandomVariable)]
    if len(prior_slots) != 1 or any(a is free for a in rv._args):
        return None
    idx, prior_rv = prior_slots[0]
    if prior_rv._kind != "sample" or isinstance(prior_rv._family, CompositeFamily):
        return None
    if _prior_to_dict(prior_rv) is None:
        return None
    try:
        probe_value = _prior_probe_value(prior_rv)
        probe_args = tuple(probe_value if i == idx else rv._args[i] for i in range(len(rv._args)))
        stats_dist = rv._family.make_dist(probe_args, rv._name)
    except Exception:  # noqa: BLE001
        return None
    from mixle.stats.bayes.conjugate import is_conjugate_family

    # sound only when the likelihood is a conjugate family AND the prior is the family its conjugate
    # posterior expects (a single-target pair the bridge actually handles)
    expected = _EXPECTED_PRIOR_FAMILY.get(type(stats_dist).__name__)
    if expected is None or prior_rv._family.name != expected or not is_conjugate_family(stats_dist):
        return None
    return (idx, prior_rv, stats_dist)


def stats_conjugate_supported(rv) -> bool:
    """True if the general exp-family conjugate bridge can fit ``rv`` (used by how='auto' routing)."""
    return _stats_conjugate_probe(rv) is not None


def _stats_conjugate_fit(rv: RandomVariable, data, *, prior_override=None):
    probe = _stats_conjugate_probe(rv)
    if probe is None:
        return None
    idx, prior_rv, stats_dist = probe
    from mixle.stats.bayes.conjugate import conjugate_posterior

    prior_dict = prior_override if prior_override else _prior_to_dict(prior_rv)
    arr = np.asarray(data, dtype=float)
    try:
        sp = conjugate_posterior(stats_dist, arr, prior=prior_dict)
        mean_dict = sp.mean()
    except Exception:  # noqa: BLE001
        return None
    if len(mean_dict) != 1:  # single-target conjugates only (joint NIG / NIW handled elsewhere)
        return None
    key = next(iter(mean_dict))
    name = prior_rv.name or f"arg{idx}"
    entry = {
        "index": idx,
        "handle": prior_rv,
        "name": getattr(sp, "family", "conjugate"),
        "mean": float(np.ravel(mean_dict[key])[0]),
        "hyper": sp.hyper() if hasattr(sp, "hyper") else {},
        "sample": lambda k, rng: np.ravel(sp.sample(int(k), rng)[key]),
    }
    cpost = ConjugatePosterior({name: entry})

    def predictive(n, rng):
        vals = np.ravel(sp.sample(int(n), rng)[key])
        out = []
        for v in vals:
            args = [v if i == idx else rv._args[i] for i in range(len(rv._args))]
            d = rv._family.make_dist(tuple(args), rv._name)
            out.append(d.sampler(seed=int(rng.randint(1, 2**31))).sample())
        return np.asarray(out)

    cpost.predictive = predictive
    return RandomVariable._bound(sp.point_estimate(), name=rv._name, result=cpost)


def conjugate_fit(rv: RandomVariable, data, *, prior=None, **_) -> RandomVariable:
    """Fit a registered conjugate PPL model and attach its exact posterior."""
    if _is_all_free_normal(rv):  # Normal(free, free) -> Normal-Inverse-Gamma (mean + variance unknown)
        return _nig_conjugate_fit(rv, data, **(prior or {}))
    spec = conjugate_spec(rv)
    if spec is None:
        # not in the hand-written table -> try the general exp-family bridge (delegates to the
        # exp-family-map-derived stats conjugate machinery, covering many more families)
        bridged = _stats_conjugate_fit(rv, data, prior_override=prior)
        if bridged is not None:
            return bridged
        raise NotImplementedError("model is not a registered conjugate pair.")
    builder, idx, prior_rv = spec
    fam = rv._family
    arr = np.asarray(data, dtype=float)
    # n/sum/sum2 serve the continuous scalar pairs; `data` lets a discrete builder (Categorical-Dirichlet)
    # take per-category counts. A vector posterior parameter (a Dirichlet probs slot) is supported below.
    stats = {"n": float(arr.size), "sum": float(arr.sum()), "sum2": float((arr * arr).sum()), "data": arr}
    fixed = {i: rv._args[i] for i in range(len(rv._args)) if i != idx}
    entry = builder(prior_rv._args, fixed, stats, prior_rv, idx)
    name = prior_rv.name or f"arg{idx}"
    # build the fitted likelihood at the posterior-mean parameter
    full = [entry["mean"] if i == idx else rv._args[i] for i in range(len(rv._args))]
    fitted = fam.make_dist(tuple(full), rv._name)
    cpost = ConjugatePosterior({name: entry})

    def predictive(n, rng):
        pvals = np.atleast_1d(entry["sample"](n, rng))
        out = []
        for v in pvals:
            # v is a scalar for scalar conjugates, a probability vector for Categorical-Dirichlet
            args = [v if i == idx else rv._args[i] for i in range(len(rv._args))]
            d = fam.make_dist(tuple(args), rv._name)
            out.append(d.sampler(seed=int(rng.randint(1, 2**31))).sample())
        return np.asarray(out)

    cpost.predictive = predictive
    return RandomVariable._bound(fitted, name=rv._name, result=cpost)


# ------------------------------------------------ mixtures of conjugate priors (exact)
def _logbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# (likelihood, slot, prior) -> log marginal likelihood of the data under one prior
# component, up to an additive component-INDEPENDENT constant (enough to reweight a
# mixture-of-conjugate-priors exactly). Same keys as _CONJUGATE.
def _logm_normal_mean(pa, fixed, stats):
    m0, s0 = float(pa[0]), float(pa[1])
    sigma2 = float(fixed[1]) ** 2
    n, sx = stats["n"], stats["sum"]
    prec0 = 1.0 / s0**2
    precP = prec0 + n / sigma2
    bb = m0 * prec0 + sx / sigma2
    return 0.5 * math.log(prec0 / precP) + 0.5 * (bb * bb / precP - m0 * m0 * prec0)


def _logm_poisson_gamma(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    n, sx = stats["n"], stats["sum"]
    return math.lgamma(a + sx) - math.lgamma(a) + a * math.log(b) - (a + sx) * math.log(b + n)


def _logm_exponential_gamma(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    n, sx = stats["n"], stats["sum"]
    return math.lgamma(a + n) - math.lgamma(a) + a * math.log(b) - (a + n) * math.log(b + sx)


def _logm_bernoulli_beta(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    n, sx = stats["n"], stats["sum"]
    return _logbeta(a + sx, b + n - sx) - _logbeta(a, b)


def _logm_binomial_beta(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    n_tr, N, sx = float(fixed[0]), stats["n"], stats["sum"]
    return _logbeta(a + sx, b + n_tr * N - sx) - _logbeta(a, b)


def _logm_geometric_beta(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    N, sx = stats["n"], stats["sum"]
    return _logbeta(a + N, b + sx - N) - _logbeta(a, b)


_CONJ_LOGM = {
    ("Normal", 0, "Normal"): _logm_normal_mean,
    ("Poisson", 0, "Gamma"): _logm_poisson_gamma,
    ("Exponential", 0, "Gamma"): _logm_exponential_gamma,
    ("Bernoulli", 0, "Beta"): _logm_bernoulli_beta,
    ("Binomial", 1, "Beta"): _logm_binomial_beta,
    ("Geometric", 0, "Beta"): _logm_geometric_beta,
}


class ConjugateMixturePosterior:
    """Exact posterior for a mixture-of-conjugate-priors model.

    The posterior is again a mixture: the per-component conjugate posteriors with weights
    reweighted by each component's marginal likelihood, ``w'_k ∝ w_k · m_k``. Sampling draws
    a component by ``w'`` then samples that component's conjugate posterior.
    """

    def __init__(self, entries, weights, param_name):
        self.entries = entries  # list of per-component conjugate posterior dicts
        self.weights = np.asarray(weights, dtype=float)  # posterior mixing weights w'
        self.param_name = param_name
        self.acceptance_rate = None
        self.predictive = None

    def mean(self, param=None):
        """Return the posterior mixture mean for the conjugate parameter."""
        return float(np.sum(self.weights * np.array([e["mean"] for e in self.entries])))

    def samples(self, param=None, n: int = 4000, rng=None):
        """Draw posterior samples by first sampling a conjugate prior component."""
        rng = rng or np.random.RandomState()
        comp = rng.choice(len(self.entries), size=n, p=self.weights)
        out = np.empty(n)
        for k, e in enumerate(self.entries):
            m = comp == k
            cnt = int(m.sum())
            if cnt:
                out[m] = np.atleast_1d(e["sample"](cnt, rng))
        return out

    def summary(self) -> dict:
        """Return posterior mixture weights, component summaries, and mean."""
        return {
            "posterior": "mixture",
            "weights": self.weights.tolist(),
            "components": [{"mean": e["mean"], "hyper": e["hyper"]} for e in self.entries],
            "mean": self.mean(),
        }


def conjugate_mixture_spec(rv: RandomVariable):
    """Return (builder, logm, slot_index, component_rvs, prior_weights) when exactly one slot
    is a ``Mix`` of conjugate priors (all forming the same registered conjugate pair) and every
    other slot is a fixed constant; else None."""
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        return None
    if any(a is free for a in rv._args):
        return None
    mix_slots = [
        (i, a)
        for i, a in enumerate(rv._args)
        if isinstance(a, RandomVariable) and isinstance(a._family, CompositeFamily) and a._family.name == "Mixture"
    ]
    other_rv = [
        a
        for a in rv._args
        if isinstance(a, RandomVariable)
        and not (isinstance(a._family, CompositeFamily) and a._family.name == "Mixture")
    ]
    if len(mix_slots) != 1 or other_rv:
        return None
    i, mix = mix_slots[0]
    comps, weights = mix._args
    comps = list(comps)
    if not comps:
        return None
    fam_names = {c._family.name for c in comps if c._kind == "sample" and not isinstance(c._family, CompositeFamily)}
    if len(fam_names) != 1:
        return None  # all components must be the same flat conjugate prior family
    key = (rv._family.name, i, next(iter(fam_names)))
    if key not in _CONJUGATE or key not in _CONJ_LOGM:
        return None
    w = np.ones(len(comps)) / len(comps) if weights is None else np.asarray(weights, dtype=float)
    return _CONJUGATE[key], _CONJ_LOGM[key], i, comps, w / w.sum()


def conjugate_mixture_fit(rv: RandomVariable, data) -> RandomVariable:
    """Fit a likelihood with a mixture of conjugate priors exactly."""
    spec = conjugate_mixture_spec(rv)
    if spec is None:
        raise NotImplementedError("model is not a mixture of registered conjugate priors.")
    builder, logm, idx, comps, w = spec
    fam = rv._family
    arr = np.asarray(data, dtype=float)
    stats = {"n": float(arr.size), "sum": float(arr.sum()), "sum2": float((arr * arr).sum())}
    fixed = {j: rv._args[j] for j in range(len(rv._args)) if j != idx}

    entries = [builder(c._args, fixed, stats, c, idx) for c in comps]
    logw = np.log(w) + np.array([logm(c._args, fixed, stats) for c in comps])
    logw -= logw.max()
    post_w = np.exp(logw)
    post_w /= post_w.sum()

    post = ConjugateMixturePosterior(entries, post_w, comps[0].name or f"arg{idx}")
    pmean = post.mean()
    full = [pmean if j == idx else rv._args[j] for j in range(len(rv._args))]
    fitted = fam.make_dist(tuple(full), rv._name)

    def predictive(n, rng):
        pvals = np.atleast_1d(post.samples(n=n, rng=rng))
        out = []
        for v in pvals:
            args = [float(v) if j == idx else rv._args[j] for j in range(len(rv._args))]
            out.append(fam.make_dist(tuple(args), rv._name).sampler(seed=int(rng.randint(1, 2**31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    return RandomVariable._bound(fitted, name=rv._name, result=post)


# ------------------------------------------- hierarchical random effects (conjugate EM)
class HierarchicalPosterior:
    """Per-group posteriors q(mu_i) = Normal(group_means[i], group_vars[i]) plus the
    fitted hyperparameters of a Normal-Normal random-effects model.
    """

    def __init__(self, group_means, group_vars, hyper):
        self.group_means = np.asarray(group_means)
        self.group_vars = np.asarray(group_vars)
        self.hyper = hyper  # {'m':..., 'tau':..., 'sigma':...}
        self.acceptance_rate = None

    def samples(self, param=None):
        """Return the per-group posterior means for the random effects."""
        # per-group posterior mean (the random effects)
        return self.group_means

    def summary(self) -> dict:
        """Return fitted hyperparameters and per-group posterior means."""
        return {"hyper": self.hyper, "n_groups": int(self.group_means.size), "group_means": self.group_means}


def _group_stats(data):
    groups = [np.asarray(g, dtype=float).reshape(-1) for g in data]
    n_i = np.array([g.size for g in groups], dtype=float)
    sum_i = np.array([g.sum() for g in groups], dtype=float)
    sumsq_i = np.array([float((g * g).sum()) for g in groups], dtype=float)
    return n_i, sum_i, sumsq_i


def _hier_normal_normal(rv, n_i, sum_i, sumsq_i, max_its, tol):
    """mu_i ~ Normal(m, tau^2); y_ij ~ Normal(mu_i, sigma^2). Exact conjugate EM."""
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    N = float(n_i.sum())
    gbar = sum_i / np.maximum(n_i, 1.0)
    m, tau2 = float(gbar.mean()), float(gbar.var()) or 1.0
    sigma_arg = rv._args[1]
    sigma_fixed = not (sigma_arg is free or isinstance(sigma_arg, RandomVariable))
    sigma2 = float(sigma_arg) ** 2 if sigma_fixed else max(float((sumsq_i.sum() / N) - (sum_i.sum() / N) ** 2), 1e-3)
    prev = None
    for _ in range(max_its):
        v_i = 1.0 / (1.0 / tau2 + n_i / sigma2)
        mhat = (m / tau2 + sum_i / sigma2) * v_i
        m = float(mhat.mean())
        tau2 = max(float(np.mean(mhat**2 + v_i) - m**2), 1e-8)
        if not sigma_fixed:
            resid = sumsq_i - 2.0 * mhat * sum_i + n_i * (mhat**2 + v_i)
            sigma2 = max(float(resid.sum() / N), 1e-8)
        cur = (m, tau2, sigma2)
        if prev is not None and max(abs(a - b) for a, b in zip(cur, prev)) < tol:
            break
        prev = cur
    pop = GaussianDistribution(mu=m, sigma2=tau2, name=rv._name)
    hyper = {"m": m, "tau": math.sqrt(tau2), "sigma": math.sqrt(sigma2)}
    return pop, mhat, v_i, hyper


def _hier_gamma_poisson(rv, n_i, sum_i, sumsq_i, max_its, tol):
    """lambda_i ~ Gamma(a, b); y_ij ~ Poisson(lambda_i). Conjugate E-step +
    moment-matched population M-step (law of total variance)."""
    from mixle.stats.univariate.continuous.gamma import GammaDistribution

    gm = sum_i / np.maximum(n_i, 1.0)
    m = float(gm.mean())
    v = float(gm.var()) or m
    b = m / max(v, 1e-6)
    a = m * b
    prev = None
    for _ in range(max_its):
        A, B = a + sum_i, b + n_i  # posterior Gamma(A_i, B_i) per group
        Elam, Vlam = A / B, A / (B * B)
        m = float(Elam.mean())
        v = float(np.var(Elam) + Vlam.mean())  # total variance
        b = m / max(v, 1e-8)
        a = m * b
        cur = (a, b)
        if prev is not None and max(abs(x - y) for x, y in zip(cur, prev)) < tol:
            break
        prev = cur
    pop = GammaDistribution(k=a, theta=1.0 / b, name=rv._name)  # population over rates
    hyper = {"shape": a, "rate": b, "mean": a / b}
    return pop, Elam, Vlam, hyper


def _hier_beta_bernoulli(rv, n_i, sum_i, sumsq_i, max_its, tol):
    """p_i ~ Beta(a, b); y_ij ~ Bernoulli(p_i). Conjugate E-step + moment-matched M-step."""
    from mixle.stats.univariate.continuous.beta import BetaDistribution

    gp = sum_i / np.maximum(n_i, 1.0)
    m = float(gp.mean())
    v = float(gp.var()) or (m * (1 - m))
    s = max(m * (1 - m) / max(v, 1e-6) - 1, 1e-3)
    a = m * s
    b = (1 - m) * s
    prev = None
    for _ in range(max_its):
        A, B = a + sum_i, b + (n_i - sum_i)  # posterior Beta(A_i, B_i)
        Ep = A / (A + B)
        Vp = A * B / ((A + B) ** 2 * (A + B + 1))
        m = float(Ep.mean())
        v = float(np.var(Ep) + Vp.mean())
        s = max(m * (1 - m) / max(v, 1e-8) - 1, 1e-3)
        a = m * s
        b = (1 - m) * s
        cur = (a, b)
        if prev is not None and max(abs(x - y) for x, y in zip(cur, prev)) < tol:
            break
        prev = cur
    pop = BetaDistribution(a, b, name=rv._name)
    hyper = {"a": a, "b": b, "mean": a / (a + b)}
    return pop, Ep, Vp, hyper


# (likelihood family, prior family) -> hierarchical conjugate EM
_HIERARCHICAL = {
    ("Normal", "Normal"): _hier_normal_normal,
    ("Poisson", "Gamma"): _hier_gamma_poisson,
    ("Bernoulli", "Beta"): _hier_beta_bernoulli,
}


def hierarchical_fit(rv: RandomVariable, data, *, max_its: int = 300, tol: float = 1e-8) -> RandomVariable:
    """Conjugate hierarchical (random-effects) EM, dispatched by conjugate pair.

    Supports Normal-Normal (exact), Gamma-Poisson, and Beta-Bernoulli. ``data`` is a list
    of groups; returns the fitted population distribution plus per-group posteriors.
    """
    fam = rv._family
    prior = rv._args[0]
    if not (isinstance(prior, RandomVariable) and prior._scope == "grouped"):
        raise NotImplementedError("expected a .each() group prior in slot 0.")
    key = (fam.name, prior._family.name)
    impl = _HIERARCHICAL.get(key)
    if impl is None:
        raise NotImplementedError(f"hierarchical pair {key} not supported; have {sorted(_HIERARCHICAL)}.")
    n_i, sum_i, sumsq_i = _group_stats(data)
    pop, group_means, group_vars, hyper = impl(rv, n_i, sum_i, sumsq_i, max_its, tol)
    post = HierarchicalPosterior(group_means, group_vars, hyper)
    return RandomVariable._bound(pop, name=rv._name, result=post)


# ----------------------------- data-indexed latent vectors: theta[Field("g")] (per-observation target)
class _NPShim:
    """numpy stand-in for the ``t`` module the autograd ``_scorers`` ``prep`` callbacks expect."""

    @staticmethod
    def lgamma(x):
        from scipy.special import gammaln

        return gammaln(np.asarray(x, dtype=float))

    log = staticmethod(np.log)
    log1p = staticmethod(np.log1p)


class IndexedPosterior:
    """Result of an indexed-latent fit: the fitted latent vector(s) and scalar parameters.

    ``latents`` maps each ``free(K)`` vector's name to its fitted ``K``-vector; ``group_means`` aliases
    the single-vector case (mirrors :class:`HierarchicalPosterior`).
    """

    def __init__(self, latents: dict, scalars: dict):
        self.latents = {k: np.asarray(v) for k, v in latents.items()}
        self.scalars = scalars
        vecs = list(self.latents.values())
        self.group_means = vecs[0] if len(vecs) == 1 else None
        self.acceptance_rate = None

    def samples(self, param=None):
        """Return the fitted latent vector for the indexed-latent result."""
        return self.group_means

    def summary(self) -> dict:
        """Return fitted latent vectors and scalar parameter estimates."""
        out = {"latents": self.latents, "scalars": self.scalars}
        if self.group_means is not None:
            out["group_means"] = self.group_means
            out["n_groups"] = int(self.group_means.size)
        return out


def _rep_eval(node, means: dict) -> float:
    """A representative scalar value for a slot expression (gather -> the latent vector's mean), used
    only to build a stand-in 'population' distribution for the bound result."""
    if not isinstance(node, RandomVariable):
        return float(node)
    if node._kind == "gather":
        return float(means[id(node._args[0])])
    if node._kind == "sum":
        return _rep_eval(node._args[0], means) + _rep_eval(node._args[1], means)
    if node._kind == "prod":
        return _rep_eval(node._args[0], means) * _rep_eval(node._args[1], means)
    if node._kind == "pow":
        return _rep_eval(node._args[0], means) ** node._args[1]
    if node._kind == "apply":
        x = _rep_eval(node._args[0], means)
        tr = node._args[1]
        nm = type(tr).__name__
        if nm == "ExpTransform":
            return float(np.exp(x))
        if nm == "LogTransform":
            return float(np.log(x))
        return tr.loc + tr.scale * x  # AffineTransform
    if node._kind == "param":
        return float(means.get(id(node), 0.0))
    return 0.0


def _indexed_target(rv: RandomVariable, data, given: dict):
    """Per-observation numerical target for a flat model with a data-indexed latent vector.

    Each observation ``i`` is scored against its own parameters, where a gathered latent contributes
    ``theta[g[i]]``. Latent vectors are ``free(K)`` handles (entries on the vector's support); other slots
    are ``free`` tokens or constants (scalar priors / hierarchical priors on the vector are a later step).
    Reuses the autograd ``_scorers`` with the numpy engine. Returns ``(log_target, slots, extract, rep)``.
    """
    from mixle.engines import NumpyEngine
    from mixle.ppl.autograd import _scorers
    from mixle.ppl.core import _eval_expr, _expr_leaves

    fam = rv._family
    if isinstance(fam, CompositeFamily):
        raise NotImplementedError("data-indexed latents are supported on flat families only.")
    scorers = _scorers()
    if fam.name not in scorers:
        raise NotImplementedError(f"data-indexed fitting needs a Torch/numpy scorer for {fam.name!r}.")
    eng = NumpyEngine()
    given = given or {}
    y = np.asarray(data, dtype=float).reshape(-1)

    # discover latent vectors (the bases of gather nodes) and validate their index covariates
    vec_handles: dict = {}  # id -> (handle, spec)
    field_names: set = set()

    def _scan(node):
        if not isinstance(node, RandomVariable):
            return
        if node._kind == "gather":
            base, field = node._args
            spec = _spec_of(base)
            if not isinstance(spec, _VectorSpec):
                raise NotImplementedError("a data-indexed gather requires a free(K) vector latent.")
            if field.name not in given:
                raise ValueError(f"data-indexed latent needs the index covariate: given={{{field.name!r}: labels}}.")
            if len(np.asarray(given[field.name]).reshape(-1)) != len(y):
                raise ValueError(f"given[{field.name!r}] length must match the data ({len(y)}).")
            vec_handles[id(base)] = (base, spec)
            field_names.add(field.name)
            return
        for a in node._args:
            _scan(a)

    for a in rv._args:
        _scan(a)
    if not vec_handles:
        raise NotImplementedError("no data-indexed latent (theta[Field(...)]) found in the model.")

    # only free tokens / constants are allowed in the non-gather slots for now
    for a in rv._args:
        if isinstance(a, RandomVariable):
            for leaf in _expr_leaves(a):
                if id(leaf) not in vec_handles and isinstance(leaf, RandomVariable) and leaf._kind == "sample":
                    raise NotImplementedError(
                        "scalar priors combined with a data-indexed latent are a later step; "
                        "use free / constant scalar slots for now."
                    )

    # slots: vector entries, then free-token positional slots
    slots: list[_Slot] = []
    meta: list = []  # parallel: ('vec', id(handle), j) | ('free', position)
    vec_cols: dict = {}
    for hid, (h, spec) in vec_handles.items():
        cols = []
        for j in range(spec.dim):
            slots.append(
                _Slot(len(slots), None, spec.support == "positive", f"{h.name or 'theta'}[{j}]", None, spec.support)
            )
            meta.append(("vec", hid, j))
            cols.append(len(slots) - 1)
        vec_cols[hid] = cols
    for i, a in enumerate(rv._args):
        if a is free:
            slots.append(_Slot(len(slots), None, fam.positive[i], f"arg{i}", None, fam.support[i]))
            meta.append(("free", i))

    prep, apply = scorers[fam.name]
    data_terms = prep(y, _NPShim)
    fin = y[np.isfinite(y)]
    dmean = float(fin.mean()) if fin.size else 0.0
    dstd = float(fin.std() or 1.0) if fin.size else 1.0
    fields = {nm: np.asarray(given[nm]).reshape(-1).astype(int) for nm in field_names}

    def _unpack(u):
        theta = {hid: np.empty(spec.dim) for hid, (h, spec) in vec_handles.items()}
        pos: dict = {}
        logj = 0.0
        for k, s in enumerate(slots):
            v, lj = _to_value(s.support, u[k])
            logj += lj
            tag = meta[k]
            if tag[0] == "vec":
                theta[tag[1]][tag[2]] = v
            else:
                pos[tag[1]] = v
        return theta, pos, logj

    def _args_for(theta, pos, evaluator):
        env = {h: theta[hid] for hid, (h, _s) in vec_handles.items()}
        env.update({("field", nm): fields[nm] for nm in field_names})
        out = []
        for i, a in enumerate(rv._args):
            if a is free:
                out.append(pos[i])
            elif isinstance(a, RandomVariable):
                out.append(evaluator(a, env))
            else:
                out.append(float(a))
        return out

    def log_target(u):
        theta, pos, logj = _unpack(u)
        args = _args_for(theta, pos, _eval_expr)
        lp = apply(args, data_terms, y, eng)
        rv_sum = float(np.sum(lp))
        return rv_sum + logj if math.isfinite(rv_sum) else _NEG_INF

    def extract(u):
        theta, pos, _ = _unpack(u)
        latents = {(h.name or f"theta{n}"): theta[hid] for n, (hid, (h, _s)) in enumerate(vec_handles.items())}
        scalars = {f"arg{i}": pos[i] for i in pos}
        return latents, scalars

    def rep(u):
        theta, pos, _ = _unpack(u)
        means = {hid: float(np.mean(theta[hid])) for hid in vec_handles}
        out = []
        for i, a in enumerate(rv._args):
            out.append(pos[i] if a is free else (_rep_eval(a, means) if isinstance(a, RandomVariable) else float(a)))
        return out

    return log_target, slots, extract, rep, (dmean, dstd)


def indexed_fit(
    rv: RandomVariable, data, *, given=None, how="map", rng=None, draws=2000, burn=1000, thin=1, **_
) -> RandomVariable:
    """Fit a flat model with a data-indexed latent vector (``theta[Field("g")]``).

    ``how='map'`` (the default) maximizes the per-observation joint and returns the fitted vectors on
    ``.result`` (``latents`` / ``group_means``). ``how='mcmc'`` samples the joint by adaptive random-walk
    Metropolis and returns a full :class:`Posterior` (per-latent summary / credible intervals). Other
    ``how`` values (hmc/nuts) are not yet wired for the indexed target.
    """
    log_target, slots, extract, rep, (dmean, dstd) = _indexed_target(rv, data, given)

    if how == "mcmc":
        from mixle.inference.mcmc import AdaptiveRandomWalkProposal, metropolis_hastings

        rng = np.random.RandomState() if rng is None else rng
        u0 = _init_u(slots, dmean, dstd)
        res = metropolis_hastings(
            log_target,
            u0,
            AdaptiveRandomWalkProposal(_init_scale(slots, dstd, len(data)).copy()),
            num_samples=draws,
            burn_in=burn,
            thin=thin,
            rng=rng,
        )
        u = np.asarray(res.samples, dtype=float).reshape(len(res.samples), -1)
        vals = _u_to_vals(slots, u)
        post = Posterior(slots, vals, res)
        _attach_convergence(post, slots, u[None, :, :], [res])
        pop = rv._family.make_dist(tuple(rep(u.mean(axis=0))), rv._name)  # bound at the posterior mean
        return RandomVariable._bound(pop, name=rv._name, result=post)

    if how not in ("map", "auto"):
        raise NotImplementedError(f"data-indexed latents support how='map'/'mcmc' (got {how!r}).")

    from scipy.optimize import minimize

    u0 = _init_u(slots, dmean, dstd)
    res = minimize(lambda u: -log_target(u), u0, method="L-BFGS-B", options={"maxiter": 2000})
    latents, scalars = extract(res.x)
    pop = rv._family.make_dist(tuple(rep(res.x)), rv._name)
    return RandomVariable._bound(pop, name=rv._name, result=IndexedPosterior(latents, scalars))
