"""``PINNRegression`` -- a physics-informed neural network as a Mixle conditional-density model.

A :class:`~mixle.models.neural_leaf.NeuralGaussian` fits ``p(y | x) = N(y; module(x), noise^2 I)`` from labeled
``(x, y)`` pairs alone. ``PINNRegression`` is the same model plus a **residual penalty**: at every M-step it also
draws unlabeled collocation points from a box domain, evaluates a caller-supplied PDE/ODE residual on the
module's output via autograd, and adds ``residual_weight * mean(residual**2)`` to the training loss -- the
standard physics-informed-neural-network (PINN) loss, ``L = L_data + w * L_physics``.

This makes the labeled-data term double as boundary/initial conditions (or scattered measurements) and the
residual term enforce the governing equation *between* them, so the fitted module honors the physics even where
it never saw labeled data -- the whole point of a PINN over plain regression. With zero labeled data
(``suff_stat`` empty) the model still trains: pure PDE-residual fitting, a boundary-value/collocation solver.

The reported density (:meth:`log_density`/:meth:`seq_log_density`, inherited unchanged from ``NeuralGaussian``)
is the data-fit Gaussian NLL only -- the model never claims the residual penalty as part of its probability
model. :func:`mixle.inference.planning.certify` already caps a bare gradient-fit model like this at
``STATIONARY`` (no global-optimum claim), so ``penalized=`` adds nothing for a standalone fit; pass
``certify(structure, penalized="PINN residual")`` when this model is composed as one block of a larger
structure that otherwise contains closed-form EM blocks, so the composite
certificate records the residual-penalized training step as a gradient-based
block (mirroring how :func:`mixle.ppl.core.ode_residual`'s soft-constraint fits
are certified).

Requires torch. ``residual_fn(module, collocation_points) -> tensor`` computes the residual using
``torch.autograd.grad`` on the module's output w.r.t. ``collocation_points`` (which arrive with
``requires_grad_(True)`` already set) -- ordinary PINN practice, e.g. for a 1-D heat equation
``u_t = alpha * u_xx`` over inputs ``(t, x)``::

    def heat_residual(module, coll):
        u = module(coll)
        grads = torch.autograd.grad(u, coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t, u_x = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x, coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        return u_t - ALPHA * u_xx

    model = PINNRegression(make_mlp(2, [32, 32], 1), heat_residual, domain=([0.0, -1.0], [1.0, 1.0]))
"""

from __future__ import annotations

import base64
import pickle
from typing import Any

import numpy as np

from mixle.models._neural_serial import decode_module, encode_module
from mixle.models.neural_leaf import (
    NeuralGaussian,
    NeuralGaussianAccumulatorFactory,
    NeuralGaussianEncoder,
    NeuralGaussianEstimator,
    _resolve_device,
    _torch,
)


def _encode_residual_fn(fn: Any) -> str:
    """Base64-encode a residual function via pickle -- works for any module-level (non-lambda) callable."""
    try:
        return base64.b64encode(pickle.dumps(fn)).decode("ascii")
    except Exception as e:
        raise ValueError(
            "PINNRegression.to_dict() needs residual_fn to be pickle-able (a module-level function, not a "
            "lambda or closure); construct the model from a named function if you need serialization."
        ) from e


def _decode_residual_fn(payload: str) -> Any:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


class PINNRegression(NeuralGaussian):
    """``NeuralGaussian`` plus a PDE/ODE-residual penalty evaluated on sampled collocation points.

    ``domain`` is a ``(low, high)`` pair of per-dimension box bounds for collocation sampling; ``residual_fn``
    computes the physics residual (see module docstring); ``residual_weight`` scales the penalty relative to
    the data-fit NLL; ``n_collocation`` is how many collocation points are drawn fresh every M-step.
    """

    def __init__(
        self,
        module: Any,
        residual_fn: Any,
        domain: tuple[Any, Any],
        *,
        noise: float = 1.0,
        residual_weight: float = 1.0,
        n_collocation: int = 64,
        m_steps: int = 40,
        lr: float = 0.01,
        seed: int = 0,
        name: str | None = None,
        device: Any = None,
    ) -> None:
        super().__init__(module, noise=noise, m_steps=m_steps, lr=lr, name=name, device=device)
        self.residual_fn = residual_fn
        self.domain = (np.asarray(domain[0], dtype=float), np.asarray(domain[1], dtype=float))
        self.residual_weight = float(residual_weight)
        self.n_collocation = int(n_collocation)
        self.seed = int(seed)

    def __str__(self) -> str:
        return "PINNRegression(noise=%.3g, residual_weight=%.3g)" % (self.noise, self.residual_weight)

    def estimator(self, pseudo_count: float | None = None) -> PINNRegressionEstimator:
        """Return the estimator that combines weighted data fit with residual collocation penalties."""
        return PINNRegressionEstimator(
            self.module,
            self.residual_fn,
            self.domain,
            noise=self.noise,
            residual_weight=self.residual_weight,
            n_collocation=self.n_collocation,
            m_steps=self.m_steps,
            lr=self.lr,
            seed=self.seed,
            name=self.name,
            device=self.device,
        )

    def dist_to_encoder(self) -> NeuralGaussianEncoder:
        """Return the neural-Gaussian encoder for ``(x, y)`` observation pairs."""
        return NeuralGaussianEncoder()

    # --- serialization: same module-as-bytes pattern as NeuralGaussian, plus the residual_fn/domain/PINN
    # hyperparameters. residual_fn must be a module-level (picklable) callable -- see _encode_residual_fn. ---
    def __pysp_getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["module"] = encode_module(self.module)
        state["residual_fn"] = _encode_residual_fn(self.residual_fn)
        state["domain"] = (self.domain[0].tolist(), self.domain[1].tolist())
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.module = decode_module(state["module"])
        self.residual_fn = _decode_residual_fn(state["residual_fn"])
        self.domain = (np.asarray(state["domain"][0], dtype=float), np.asarray(state["domain"][1], dtype=float))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the module, residual function reference, domain, and PINN hyperparameters."""
        return {
            "noise": self.noise,
            "m_steps": self.m_steps,
            "lr": self.lr,
            "name": self.name,
            "device": self.device,
            "module": encode_module(self.module),
            "residual_fn": _encode_residual_fn(self.residual_fn),
            "domain": (self.domain[0].tolist(), self.domain[1].tolist()),
            "residual_weight": self.residual_weight,
            "n_collocation": self.n_collocation,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PINNRegression:
        """Rebuild a :class:`PINNRegression` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["module"]),
            _decode_residual_fn(payload["residual_fn"]),
            payload["domain"],
            noise=payload["noise"],
            residual_weight=payload["residual_weight"],
            n_collocation=payload["n_collocation"],
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            seed=payload["seed"],
            name=payload["name"],
            device=payload["device"],
        )


class PINNRegressionEstimator(NeuralGaussianEstimator):
    """EM estimator for :class:`PINNRegression`: the M-step adds a residual penalty on fresh collocation points
    to the same weighted-NLL gradient descent :class:`~mixle.models.neural_leaf.NeuralGaussianEstimator` runs.

    Collocation sampling is deterministic given ``seed`` (a private ``numpy.random.RandomState``, advanced
    once per M-step) -- refitting with the same seed draws the same collocation batches.
    """

    def __init__(
        self,
        module: Any,
        residual_fn: Any,
        domain: tuple[np.ndarray, np.ndarray],
        *,
        noise: float = 1.0,
        residual_weight: float = 1.0,
        n_collocation: int = 64,
        m_steps: int = 40,
        lr: float = 0.01,
        seed: int = 0,
        name: str | None = None,
        device: Any = None,
    ) -> None:
        super().__init__(module, noise, m_steps, lr, name, device)
        self.residual_fn = residual_fn
        self.domain = domain
        self.residual_weight = float(residual_weight)
        self.n_collocation = int(n_collocation)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)

    def accumulator_factory(self) -> NeuralGaussianAccumulatorFactory:
        """Return the neural-Gaussian accumulator factory for weighted observation pairs."""
        return NeuralGaussianAccumulatorFactory()

    def _sample_collocation(self, dev: Any, torch: Any) -> Any:
        low, high = self.domain
        u = self._rng.uniform(0.0, 1.0, size=(self.n_collocation, low.shape[0]))
        pts = low + u * (high - low)
        return torch.as_tensor(pts, dtype=torch.float32, device=dev).requires_grad_(True)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> PINNRegression:
        """Run the data-plus-residual M-step and return the updated PINN leaf."""
        torch = _torch()
        xs, ys, ws = suff_stat
        has_data = len(xs) > 0
        dev = _resolve_device(self.device, torch)
        self.module.to(dev)

        if has_data:
            xt = torch.as_tensor(np.array(xs), dtype=torch.float32, device=dev)
            yt = torch.as_tensor(np.array(ys), dtype=torch.float32, device=dev)
            wt = torch.as_tensor(np.array(ws), dtype=torch.float32, device=dev)
            wsum = float(wt.sum()) + 1e-8
            d = yt.shape[1]

        log_noise = torch.log(torch.tensor(float(self.noise), device=dev)).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam(list(self.module.parameters()) + [log_noise], lr=self.lr)
        for _ in range(self.m_steps):
            opt.zero_grad()
            loss = torch.zeros((), device=dev)
            if has_data:
                mean = self.module(xt)
                sig2 = torch.exp(2.0 * log_noise)
                nll = (
                    wt * (0.5 * ((yt - mean) ** 2).sum(1) / sig2 + 0.5 * d * torch.log(2.0 * np.pi * sig2))
                ).sum() / wsum
                loss = loss + nll
            coll = self._sample_collocation(dev, torch)
            residual = self.residual_fn(self.module, coll)
            loss = loss + self.residual_weight * (residual**2).mean()
            loss.backward()
            opt.step()
        if has_data:
            self.noise = float(torch.exp(log_noise).detach())  # warm-start noise for the next EM iteration
        return PINNRegression(
            self.module,
            self.residual_fn,
            self.domain,
            noise=self.noise,
            residual_weight=self.residual_weight,
            n_collocation=self.n_collocation,
            m_steps=self.m_steps,
            lr=self.lr,
            seed=self.seed,
            name=self.name,
            device=self.device,
        )


def _register_serializable() -> None:
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(PINNRegression)


_register_serializable()
