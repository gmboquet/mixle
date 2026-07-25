"""Neural *densities* for ``mixle.ppl`` -- flexible ``p(x)`` and ``p(y | x)`` as first-class PPL constructors.

The nonlinear sibling of the plain distribution constructors. :mod:`mixle.ppl.neural` puts a neural *predictor*
(``Net``/``Conv``/``Transformer``) into an outer family's slot, so the outer family still fixes the likelihood
shape (a Gaussian mean, softmax logits). These constructors instead make the neural model *be* the whole density:

    Flow(dim=2).fit(x)                          # exact p(x) via a normalizing flow
    VAE(dim=8, latent=2).fit(x)                 # latent-variable p(x) (ELBO)
    DiscreteAR(dim=5, cats=4).fit(x)            # exact p(x) over discrete vectors
    EBM(dim=2).fit(x)                           # energy-based p(x) (NCE-trained, approximately normalized)
    MDN(x_dim=1, y_dim=1).fit(y, given={"x": X})       # multimodal p(y|x)
    CondFlow(x_dim=1, y_dim=2).fit(y, given={"x": X})  # exact conditional p(y|x)

Each lowers to the composable :class:`~mixle.models.neural_density.NeuralDensity` /
:class:`~mixle.models.mixture_density.NeuralConditionalDensity` leaf and fits through the same
``optimize`` EM loop -- no loss function, no training loop in user code. A fitted model is a bound
``RandomVariable`` whose ``.dist`` is the leaf, so it drops into a ``Mix``/composite like any distribution.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.ppl.core import RandomVariable, register_composite

# builder name -> (module factory, is-conditional). The spec stores the name + kwargs (pickle-safe: no closures).
_BUILDERS: dict[str, tuple[str, bool]] = {
    "coupling_flow": ("build_coupling_flow", False),
    "maf": ("build_maf", False),
    "vae": ("build_vae", False),
    "ar_categorical": ("build_autoregressive_categorical", False),
    "mdn": ("build_mdn", True),
    "conditional_flow": ("build_conditional_flow", True),
    "conditional_ar_categorical": ("build_conditional_autoregressive_categorical", True),
    "energy": ("build_energy_net", False),  # its own EnergyModel leaf (NCE), not a NeuralDensity module
}


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return result


def _matrix(
    value: Any,
    *,
    name: str,
    width: int,
    discrete: bool = False,
    categories: int | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular matrix with width {width}") from error
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != width:
        raise ValueError(f"{name} must have exact shape (rows, {width}) with at least one row")
    if discrete:
        if raw.dtype.kind not in {"i", "u"}:
            raise ValueError(f"{name} must contain integer category indices without float coercion")
        result = raw.astype(np.int64, copy=False)
        if np.any(result < 0) or categories is None or np.any(result >= categories):
            raise ValueError(f"{name} category indices must lie in [0, {categories})")
        return result
    try:
        result = raw.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite numeric matrix") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


class _DensitySpec:
    """Config for a neural-density module: which builder + its kwargs + fit hyperparameters. Pickle-safe."""

    __slots__ = ("kind", "params", "conditional", "field", "m_steps", "lr", "extra")

    def __init__(
        self,
        kind: str,
        params: dict,
        *,
        field: str = "x",
        m_steps: int = 80,
        lr: float = 5e-3,
        extra: dict | None = None,
    ):
        if kind not in _BUILDERS:
            raise ValueError(f"unknown neural density kind {kind!r}")
        if not isinstance(field, str) or not field:
            raise ValueError("conditional density field must be a non-empty string")
        self.kind = kind
        self.params = dict(params)
        self.conditional = _BUILDERS[kind][1]
        self.field = field
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        self.extra = dict(extra or {})  # leaf-specific extras (e.g. the EBM's noise_ratio)
        for name in ("dim", "x_dim", "y_dim", "hidden", "layers", "blocks", "latent", "eval_samples", "k"):
            if name in self.params:
                minimum = 2 if kind == "conditional_flow" and name == "y_dim" else 1
                self.params[name] = _positive_int(self.params[name], name, minimum=minimum)
        if "n_categories" in self.params:
            self.params["n_categories"] = _positive_int(
                self.params["n_categories"], "category count", minimum=2
            )
        if "noise_ratio" in self.extra:
            self.extra["noise_ratio"] = _positive_int(self.extra["noise_ratio"], "noise_ratio")

    @property
    def discrete(self) -> bool:
        return self.kind in {"ar_categorical", "conditional_ar_categorical"}

    @property
    def observation_dim(self) -> int:
        return int(self.params["y_dim"] if self.conditional else self.params["dim"])

    @property
    def categories(self) -> int | None:
        value = self.params.get("n_categories")
        return None if value is None else int(value)

    def build_module(self) -> Any:
        from mixle.models import (
            build_autoregressive_categorical,
            build_conditional_autoregressive_categorical,
            build_conditional_flow,
            build_coupling_flow,
            build_energy_net,
            build_maf,
            build_mdn,
            build_vae,
        )

        fn = {
            "build_coupling_flow": build_coupling_flow,
            "build_maf": build_maf,
            "build_vae": build_vae,
            "build_autoregressive_categorical": build_autoregressive_categorical,
            "build_mdn": build_mdn,
            "build_conditional_flow": build_conditional_flow,
            "build_conditional_autoregressive_categorical": build_conditional_autoregressive_categorical,
            "build_energy_net": build_energy_net,
        }[_BUILDERS[self.kind][0]]
        return fn(**self.params)

    def make_leaf(self) -> Any:
        if self.kind == "energy":
            from mixle.models.energy import EnergyModel

            return EnergyModel(self.build_module(), m_steps=self.m_steps, lr=self.lr, **self.extra)
        from mixle.models.mixture_density import NeuralConditionalDensity
        from mixle.models.neural_density import NeuralDensity

        cls = NeuralConditionalDensity if self.conditional else NeuralDensity
        return cls(self.build_module(), m_steps=self.m_steps, lr=self.lr)


# --- composite lowering: a density RV lowers to its leaf / estimator, so it composes inside a Mix/composite ----


def _density_dist(args: tuple, lower_child: Any) -> Any:
    return args[0].make_leaf()


def _density_est(args: tuple, lower_child: Any, name: Any, keys: Any) -> Any:
    return args[0].make_leaf().estimator()


def _density_fit(rv: RandomVariable, data: Any, **kw: Any) -> RandomVariable:
    """Bespoke fitter (the ``CompositeFamily.fit_fn`` hook): build the leaf, run EM, return a bound RV.

    ``its`` (default 8) is the number of warm-started M-steps; each M-step is ``spec.m_steps`` gradient steps.
    Conditional densities need the covariates: ``.fit(y, given={"x": X})``.
    """
    spec: _DensitySpec = rv._args[0]
    supported = {
        "backend",
        "delta",
        "engine",
        "given",
        "its",
        "max_its",
        "missing",
        "num_workers",
        "precision",
        "print_iter",
        "rng",
        "seed",
    }
    unknown = sorted(set(kw) - supported)
    if unknown:
        raise TypeError(f"unsupported neural-density fit control(s): {', '.join(unknown)}")
    if kw.get("missing", "error") != "error":
        raise NotImplementedError("neural-density fitting does not support missing='marginalize'")
    if "its" in kw and kw.get("max_its", 100) != 100:
        raise ValueError("its and max_its are aliases; pass only one")
    max_its = _positive_int(kw.get("its", kw.get("max_its", 8)), "density fit iterations")
    delta = _nonnegative_finite(kw.get("delta", 1e-8), "delta")
    if kw.get("seed") is not None and kw.get("rng") is not None:
        raise ValueError("pass at most one of seed and rng")
    seed = kw.get("seed")
    rng = kw.get("rng")
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        seed = int(seed)
        if seed < 0 or seed >= 2**32:
            raise ValueError("seed must lie in [0, 2**32)")
        torch_seed = seed
    elif rng is not None:
        if not isinstance(rng, np.random.RandomState):
            raise TypeError("rng must be a numpy.random.RandomState")
        torch_seed = int(rng.randint(0, 2**31))
    else:
        torch_seed = 0

    ys = _matrix(
        data,
        name="density observations",
        width=spec.observation_dim,
        discrete=spec.discrete,
        categories=spec.categories,
    )
    if not spec.conditional:
        rows = list(ys)
    else:
        given = kw.get("given")
        if not isinstance(given, dict) or spec.field not in given:
            raise ValueError(f"conditional density fit needs covariates: .fit(y, given={{{spec.field!r}: X}})")
        if set(given) != {spec.field}:
            raise ValueError(f"conditional density given data must contain only field {spec.field!r}")
        xs = _matrix(
            given[spec.field],
            name=f"given[{spec.field!r}]",
            width=int(spec.params["x_dim"]),
        )
        if xs.shape[0] != ys.shape[0]:
            raise ValueError(
                f"given[{spec.field!r}] has {xs.shape[0]} rows but observations have {ys.shape[0]}"
            )
        rows = list(zip(xs, ys))

    import torch

    from mixle.inference import optimize

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch_seed)
        leaf = spec.make_leaf()
        fitted = optimize(
            rows,
            leaf.estimator(),
            prev_estimate=leaf,
            max_its=max_its,
            delta=delta,
            out=None,
            backend=kw.get("backend", "local"),
            num_workers=kw.get("num_workers"),
            engine=kw.get("engine"),
            precision=kw.get("precision"),
            print_iter=kw.get("print_iter", 0),
            rng=rng,
            seed=seed,
        )
    return RandomVariable._bound(fitted, name=rv._name)


register_composite("NeuralDensity", _density_dist, _density_est, fit_fn=_density_fit)
register_composite("NeuralConditionalDensity", _density_dist, _density_est, fit_fn=_density_fit)


def _rv(spec: _DensitySpec) -> RandomVariable:
    fam = "NeuralConditionalDensity" if spec.conditional else "NeuralDensity"
    return RandomVariable._sample(fam, args=(spec,))


# --- the constructors: unconditional p(x) --------------------------------------------------------------------


def Flow(dim: int, *, hidden: int = 32, layers: int = 4, m_steps: int = 80, lr: float = 5e-3) -> RandomVariable:
    """An exact ``p(x)`` over ``R^dim`` via a RealNVP coupling flow. Fit with ``.fit(x)``."""
    return _rv(_DensitySpec("coupling_flow", {"dim": dim, "hidden": hidden, "layers": layers}, m_steps=m_steps, lr=lr))


def MAF(dim: int, *, hidden: int = 64, blocks: int = 3, m_steps: int = 80, lr: float = 5e-3) -> RandomVariable:
    """An exact ``p(x)`` over ``R^dim`` via a masked autoregressive flow (richer dependence). Fit with ``.fit(x)``."""
    return _rv(_DensitySpec("maf", {"dim": dim, "hidden": hidden, "blocks": blocks}, m_steps=m_steps, lr=lr))


def VAE(
    dim: int,
    *,
    latent: int = 2,
    hidden: int = 32,
    eval_samples: int = 16,
    m_steps: int = 120,
    lr: float = 5e-3,
) -> RandomVariable:
    """A latent-variable ``p(x)`` via a VAE whose score is a reproducible ELBO estimate; fit with ``.fit(x)``."""
    return _rv(
        _DensitySpec(
            "vae",
            {"dim": dim, "latent": latent, "hidden": hidden, "eval_samples": eval_samples},
            m_steps=m_steps,
            lr=lr,
        )
    )


def DiscreteAR(dim: int, cats: int, *, hidden: int = 64, m_steps: int = 100, lr: float = 5e-3) -> RandomVariable:
    """An exact ``p(x)`` over **discrete** vectors ``x in {0..cats-1}^dim`` (autoregressive). Fit with ``.fit(x)``."""
    return _rv(_DensitySpec("ar_categorical", {"dim": dim, "n_categories": cats}, m_steps=m_steps, lr=lr))


def EBM(
    dim: int, *, hidden: int = 64, layers: int = 3, noise_ratio: int = 2, m_steps: int = 250, lr: float = 5e-3
) -> RandomVariable:
    """An energy-based ``p(x) ∝ exp(-E(x))`` over ``R^dim``, trained by NCE (approximately normalized). Fit ``.fit(x)``."""
    return _rv(
        _DensitySpec(
            "energy",
            {"dim": dim, "hidden": hidden, "layers": layers},
            m_steps=m_steps,
            lr=lr,
            extra={"noise_ratio": int(noise_ratio)},
        )
    )


# --- the constructors: conditional p(y|x) --------------------------------------------------------------------


def MDN(
    x_dim: int, y_dim: int, *, k: int = 5, hidden: int = 32, field: str = "x", m_steps: int = 120, lr: float = 5e-3
) -> RandomVariable:
    """A multimodal, heteroscedastic ``p(y | x)`` via a mixture density network. Fit ``.fit(y, given={"x": X})``."""
    return _rv(
        _DensitySpec(
            "mdn", {"x_dim": x_dim, "y_dim": y_dim, "k": k, "hidden": hidden}, field=field, m_steps=m_steps, lr=lr
        )
    )


def CondFlow(
    x_dim: int, y_dim: int, *, hidden: int = 32, layers: int = 4, field: str = "x", m_steps: int = 100, lr: float = 5e-3
) -> RandomVariable:
    """An exact conditional ``p(y | x)`` via a conditional coupling flow (needs ``y_dim >= 2``). Fit with covariates."""
    return _rv(
        _DensitySpec(
            "conditional_flow",
            {"x_dim": x_dim, "y_dim": y_dim, "hidden": hidden, "layers": layers},
            field=field,
            m_steps=m_steps,
            lr=lr,
        )
    )


def CondDiscreteAR(
    x_dim: int, y_dim: int, cats: int, *, hidden: int = 64, field: str = "x", m_steps: int = 120, lr: float = 5e-3
) -> RandomVariable:
    """An exact conditional ``p(y | x)`` over **discrete** ``y`` (autoregressive, conditioned on ``x``). Fit w/ covariates."""
    return _rv(
        _DensitySpec(
            "conditional_ar_categorical",
            {"x_dim": x_dim, "y_dim": y_dim, "n_categories": cats},
            field=field,
            m_steps=m_steps,
            lr=lr,
        )
    )
