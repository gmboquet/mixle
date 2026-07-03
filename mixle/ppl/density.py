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
        self.kind = kind
        self.params = dict(params)
        self.conditional = _BUILDERS[kind][1]
        self.field = field
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.extra = dict(extra or {})  # leaf-specific extras (e.g. the EBM's noise_ratio)

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
    from mixle.inference import optimize

    spec: _DensitySpec = rv._args[0]
    its = int(kw.get("its", 8))
    leaf = spec.make_leaf()

    if not spec.conditional:
        rows = [np.atleast_1d(np.asarray(v, dtype=float)) for v in data]
        fitted = optimize(rows, leaf.estimator(), prev_estimate=leaf, max_its=its, out=None)
        return RandomVariable._bound(fitted, name=rv._name)

    given = kw.get("given") or {}
    if spec.field not in given:
        raise ValueError(f"conditional density fit needs covariates: .fit(y, given={{{spec.field!r}: X}})")
    xs = [np.atleast_1d(np.asarray(x, dtype=float)) for x in np.asarray(given[spec.field], dtype=float)]
    ys = [np.atleast_1d(np.asarray(v, dtype=float)) for v in data]
    if len(xs) != len(ys):
        raise ValueError(f"given[{spec.field!r}] has length {len(xs)} but data has length {len(ys)}.")
    fitted = optimize(list(zip(xs, ys)), leaf.estimator(), prev_estimate=leaf, max_its=its, out=None)
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


def VAE(dim: int, *, latent: int = 2, hidden: int = 32, m_steps: int = 120, lr: float = 5e-3) -> RandomVariable:
    """A latent-variable ``p(x)`` over ``R^dim`` via a VAE. ``log_density`` is the ELBO (a lower bound); fit ``.fit(x)``."""
    return _rv(_DensitySpec("vae", {"dim": dim, "latent": latent, "hidden": hidden}, m_steps=m_steps, lr=lr))


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
