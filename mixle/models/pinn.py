"""``PINNRegression`` -- a physics-informed neural network as a Mixle conditional-density model.

A :class:`~mixle.models.neural_leaf.NeuralGaussian` fits ``p(y | x) = N(y; module(x), noise^2 I)`` from labeled
``(x, y)`` pairs alone. ``PINNRegression`` is the same model plus a **residual penalty**: at every M-step it also
draws unlabeled collocation points from a box domain, evaluates a caller-supplied PDE/ODE residual on the
module's output via autograd, and adds ``residual_weight * mean(residual**2)`` to the training loss -- the
standard physics-informed-neural-network (PINN) loss, ``L = L_data + w * L_physics``.

The problem declaration identifies the equation, coordinates, fields, residual components, boundary/initial
constraints, identifiability conditions, and whether constraints are enforced by observations or by the module
architecture. Observation-enforced problems require actual labeled constraint rows. Residual-only optimization
is accepted only when the declaration states that a constrained architecture enforces the named conditions;
an unconstrained residual fit is never represented as a solved problem.

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

    problem = PINNProblemSpec(
        equation="u_t - alpha*u_xx = 0",
        coordinates=("t", "x"),
        fields=("u",),
        residual_components=("heat",),
        constraints=("u(0, x) = initial(x)",),
        identifiability_conditions=("initial condition fixes the parabolic solution",),
        constraint_enforcement="observations",
        completeness_basis="initial-value problem on the declared box",
    )
    model = PINNRegression(
        make_mlp(2, [32, 32], 1),
        heat_residual,
        domain=([0.0, -1.0], [1.0, 1.0]),
        problem=problem,
    )
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mixle.models._neural_serial import (
    _environment,
    _require_trusted_deserialization,
    _serialization_error,
    decode_module,
    encode_module,
)
from mixle.models.neural_leaf import (
    NeuralGaussian,
    NeuralGaussianAccumulatorFactory,
    NeuralGaussianEncoder,
    NeuralGaussianEstimator,
    _place_module,
    _resolve_device,
    _torch,
)

MAX_PINN_RESIDUAL_BYTES = 16 * 1024 * 1024
_PINN_RESIDUAL_FORMAT = "python-callable-pickle/v1"
_RESIDUAL_FIELDS = frozenset(
    {
        "__pinn_residual__",
        "format",
        "environment_bound",
        "decoded_bytes",
        "sha256",
        "environment",
    }
)


@dataclass(frozen=True)
class PINNProblemSpec:
    """Explicit mathematical and enforcement schema for a PINN fit."""

    equation: str
    coordinates: tuple[str, ...]
    fields: tuple[str, ...]
    residual_components: tuple[str, ...]
    constraints: tuple[str, ...]
    identifiability_conditions: tuple[str, ...]
    constraint_enforcement: str
    completeness_basis: str

    def __post_init__(self) -> None:
        for name in (
            "equation",
            "completeness_basis",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "coordinates",
            "fields",
            "residual_components",
            "constraints",
            "identifiability_conditions",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{name} must be a non-empty tuple of non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if self.constraint_enforcement not in {"observations", "architecture"}:
            raise ValueError("constraint_enforcement must be 'observations' or 'architecture'")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible problem declaration."""
        payload = asdict(self)
        for name in (
            "coordinates",
            "fields",
            "residual_components",
            "constraints",
            "identifiability_conditions",
        ):
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> PINNProblemSpec:
        """Validate and restore a problem declaration."""
        if not isinstance(payload, Mapping):
            raise ValueError("PINN problem specification must be a mapping")
        expected = {
            "equation",
            "coordinates",
            "fields",
            "residual_components",
            "constraints",
            "identifiability_conditions",
            "constraint_enforcement",
            "completeness_basis",
        }
        if set(payload) != expected:
            raise ValueError(
                f"invalid PINN problem fields: missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        return cls(
            equation=payload["equation"],
            coordinates=tuple(payload["coordinates"]),
            fields=tuple(payload["fields"]),
            residual_components=tuple(payload["residual_components"]),
            constraints=tuple(payload["constraints"]),
            identifiability_conditions=tuple(payload["identifiability_conditions"]),
            constraint_enforcement=payload["constraint_enforcement"],
            completeness_basis=payload["completeness_basis"],
        )


@dataclass(frozen=True)
class PINNConstraintReceipt:
    """Durable declaration and measured outcome of a constraint-aware PINN fit."""

    status: str
    equation: str
    constraint_enforcement: str
    constraints: tuple[str, ...]
    identifiability_conditions: tuple[str, ...]
    completeness_basis: str
    labeled_constraints: int
    residual_loss_initial: float
    residual_loss_final: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible receipt."""
        payload = asdict(self)
        payload["constraints"] = list(self.constraints)
        payload["identifiability_conditions"] = list(self.identifiability_conditions)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> PINNConstraintReceipt:
        """Restore a receipt produced by :meth:`to_dict`."""
        if not isinstance(payload, Mapping):
            raise ValueError("PINN constraint receipt must be a mapping")
        expected = {
            "status",
            "equation",
            "constraint_enforcement",
            "constraints",
            "identifiability_conditions",
            "completeness_basis",
            "labeled_constraints",
            "residual_loss_initial",
            "residual_loss_final",
        }
        if set(payload) != expected:
            raise ValueError(
                f"invalid PINN receipt fields: missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        return cls(
            status=str(payload["status"]),
            equation=str(payload["equation"]),
            constraint_enforcement=str(payload["constraint_enforcement"]),
            constraints=tuple(payload["constraints"]),
            identifiability_conditions=tuple(payload["identifiability_conditions"]),
            completeness_basis=str(payload["completeness_basis"]),
            labeled_constraints=int(payload["labeled_constraints"]),
            residual_loss_initial=float(payload["residual_loss_initial"]),
            residual_loss_final=float(payload["residual_loss_final"]),
        )


def _encode_residual_fn(fn: Any) -> dict[str, Any]:
    """Return a bounded, environment-bound executable-callable envelope."""
    if not callable(fn):
        raise TypeError("residual_fn must be callable")
    try:
        data = pickle.dumps(fn, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise ValueError(
            "PINNRegression.to_dict() needs residual_fn to be pickle-able (a module-level function, not a "
            "lambda or closure); construct the model from a named function if you need serialization."
        ) from e
    if not data or len(data) > MAX_PINN_RESIDUAL_BYTES:
        raise ValueError(
            f"PINN residual artifact must contain 1 through {MAX_PINN_RESIDUAL_BYTES} decoded bytes"
        )
    return {
        "__pinn_residual__": base64.b64encode(data).decode("ascii"),
        "format": _PINN_RESIDUAL_FORMAT,
        "environment_bound": True,
        "decoded_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "environment": _environment(),
    }


def _decode_residual_fn(payload: Any) -> Any:
    """Decode a trusted residual envelope after strict bounds and integrity checks."""
    _require_trusted_deserialization()
    if not isinstance(payload, Mapping) or set(payload) != _RESIDUAL_FIELDS:
        raise _serialization_error("PINN residual payload has invalid fields")
    if payload["format"] != _PINN_RESIDUAL_FORMAT or payload["environment_bound"] is not True:
        raise _serialization_error("PINN residual payload has an unsupported format or trust policy")
    decoded_bytes = payload["decoded_bytes"]
    if type(decoded_bytes) is not int or not 0 < decoded_bytes <= MAX_PINN_RESIDUAL_BYTES:
        raise _serialization_error(
            f"PINN residual decoded_bytes must be an integer from 1 through {MAX_PINN_RESIDUAL_BYTES}"
        )
    encoded = payload["__pinn_residual__"]
    max_encoded_bytes = 4 * ((MAX_PINN_RESIDUAL_BYTES + 2) // 3)
    if not isinstance(encoded, str) or not encoded or len(encoded) > max_encoded_bytes:
        raise _serialization_error("PINN residual must be a bounded ASCII base64 string")
    try:
        data = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise _serialization_error("PINN residual is not strict ASCII base64") from exc
    if len(data) != decoded_bytes:
        raise _serialization_error("PINN residual decoded length does not match decoded_bytes")
    digest = payload["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise _serialization_error("PINN residual sha256 must be a 64-character hexadecimal digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise _serialization_error("PINN residual sha256 must be hexadecimal") from exc
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), digest.lower()):
        raise _serialization_error("PINN residual sha256 does not match its decoded bytes")
    environment = payload["environment"]
    if not isinstance(environment, Mapping) or dict(environment) != _environment():
        raise _serialization_error("PINN residual environment does not match the current runtime")
    try:
        residual_fn = pickle.loads(data)
    except Exception as exc:
        raise _serialization_error("PINN residual pickle could not be decoded") from exc
    if not callable(residual_fn):
        raise _serialization_error("PINN residual artifact did not contain a callable")
    return residual_fn


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
        problem: PINNProblemSpec,
        noise: float = 1.0,
        residual_weight: float = 1.0,
        n_collocation: int = 64,
        m_steps: int = 40,
        lr: float = 0.01,
        seed: int = 0,
        name: str | None = None,
        device: Any = None,
        constraint_receipt: PINNConstraintReceipt | None = None,
    ) -> None:
        if not isinstance(problem, PINNProblemSpec):
            raise TypeError("problem must be a PINNProblemSpec")
        if not callable(residual_fn):
            raise TypeError("residual_fn must be callable")
        low, high = _validated_domain(domain, len(problem.coordinates))
        noise = _finite_positive(noise, "noise")
        residual_weight = _finite_positive(residual_weight, "residual_weight")
        n_collocation = _positive_int(n_collocation, "n_collocation")
        m_steps = _positive_int(m_steps, "m_steps")
        lr = _finite_positive(lr, "lr")
        seed = _integer(seed, "seed")
        super().__init__(module, noise=noise, m_steps=m_steps, lr=lr, name=name, device=device)
        self.residual_fn = residual_fn
        self.domain = (low, high)
        self.problem = problem
        self.residual_weight = residual_weight
        self.n_collocation = n_collocation
        self.seed = seed
        self.constraint_receipt = constraint_receipt

    def __str__(self) -> str:
        return "PINNRegression(noise=%.3g, residual_weight=%.3g)" % (self.noise, self.residual_weight)

    def estimator(self, pseudo_count: float | None = None) -> PINNRegressionEstimator:
        """Return the estimator that combines weighted data fit with residual collocation penalties."""
        return PINNRegressionEstimator(
            self.module,
            self.residual_fn,
            self.domain,
            problem=self.problem,
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
        state["problem"] = self.problem.to_dict()
        state["constraint_receipt"] = (
            None if self.constraint_receipt is None else self.constraint_receipt.to_dict()
        )
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        restored = type(self)(
            decode_module(state["module"]),
            _decode_residual_fn(state["residual_fn"]),
            state["domain"],
            problem=PINNProblemSpec.from_dict(state["problem"]),
            noise=state["noise"],
            residual_weight=state["residual_weight"],
            n_collocation=state["n_collocation"],
            m_steps=state["m_steps"],
            lr=state["lr"],
            seed=state["seed"],
            name=state["name"],
            device=state["device"],
            constraint_receipt=(
                None
                if state.get("constraint_receipt") is None
                else PINNConstraintReceipt.from_dict(state["constraint_receipt"])
            ),
        )
        self.__dict__.update(restored.__dict__)

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
            "problem": self.problem.to_dict(),
            "residual_weight": self.residual_weight,
            "n_collocation": self.n_collocation,
            "seed": self.seed,
            "constraint_receipt": (
                None if self.constraint_receipt is None else self.constraint_receipt.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PINNRegression:
        """Rebuild a :class:`PINNRegression` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["module"]),
            _decode_residual_fn(payload["residual_fn"]),
            payload["domain"],
            problem=PINNProblemSpec.from_dict(payload["problem"]),
            noise=payload["noise"],
            residual_weight=payload["residual_weight"],
            n_collocation=payload["n_collocation"],
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            seed=payload["seed"],
            name=payload["name"],
            device=payload["device"],
            constraint_receipt=(
                None
                if payload.get("constraint_receipt") is None
                else PINNConstraintReceipt.from_dict(payload["constraint_receipt"])
            ),
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
        problem: PINNProblemSpec,
        noise: float = 1.0,
        residual_weight: float = 1.0,
        n_collocation: int = 64,
        m_steps: int = 40,
        lr: float = 0.01,
        seed: int = 0,
        name: str | None = None,
        device: Any = None,
    ) -> None:
        if not isinstance(problem, PINNProblemSpec):
            raise TypeError("problem must be a PINNProblemSpec")
        if not callable(residual_fn):
            raise TypeError("residual_fn must be callable")
        low, high = _validated_domain(domain, len(problem.coordinates))
        noise = _finite_positive(noise, "noise")
        residual_weight = _finite_positive(residual_weight, "residual_weight")
        n_collocation = _positive_int(n_collocation, "n_collocation")
        m_steps = _positive_int(m_steps, "m_steps")
        lr = _finite_positive(lr, "lr")
        seed = _integer(seed, "seed")
        super().__init__(module, noise, m_steps, lr, name, device)
        self.residual_fn = residual_fn
        self.domain = (low, high)
        self.problem = problem
        self.residual_weight = residual_weight
        self.n_collocation = n_collocation
        self.seed = seed
        self._rng = np.random.RandomState(self.seed)

    def accumulator_factory(self) -> NeuralGaussianAccumulatorFactory:
        """Return the neural-Gaussian accumulator factory for weighted observation pairs."""
        return NeuralGaussianAccumulatorFactory()

    def _sample_collocation(self, dev: Any, dtype: Any, torch: Any) -> Any:
        low, high = self.domain
        u = self._rng.uniform(0.0, 1.0, size=(self.n_collocation, low.shape[0]))
        pts = low + u * (high - low)
        return torch.as_tensor(pts, dtype=dtype, device=dev).requires_grad_(True)

    def _validated_data(self, suff_stat: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(suff_stat, tuple) or len(suff_stat) != 3:
            raise ValueError("PINN sufficient statistics must be an (x, y, weights) tuple")
        xs, ys, ws = (np.asarray(value) for value in suff_stat)
        if xs.ndim != 2 or xs.shape[1] != len(self.problem.coordinates):
            raise ValueError(
                f"PINN x must have shape (n, {len(self.problem.coordinates)}), got {xs.shape}"
            )
        if ys.ndim != 2 or ys.shape[1] != len(self.problem.fields):
            raise ValueError(f"PINN y must have shape (n, {len(self.problem.fields)}), got {ys.shape}")
        if ws.ndim != 1 or not (len(xs) == len(ys) == len(ws)):
            raise ValueError("PINN x, y, and weights must have aligned rows and one-dimensional weights")
        try:
            xs = xs.astype(float, copy=False)
            ys = ys.astype(float, copy=False)
            ws = ws.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("PINN observations and weights must be numeric") from exc
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            raise ValueError("PINN observations must contain only finite values")
        if not np.all(np.isfinite(ws)) or np.any(ws < 0.0):
            raise ValueError("PINN weights must contain only finite, non-negative values")
        if len(xs) and not np.any(ws > 0.0):
            raise ValueError("a non-empty PINN observation set must contain positive effective weight")
        if self.problem.constraint_enforcement == "observations" and not len(xs):
            raise ValueError(
                "the PINN problem declares observation-enforced constraints but no boundary, initial, "
                "or identifiability observations were supplied"
            )
        return xs, ys, ws

    def _validated_residual(self, residual: Any, torch: Any) -> Any:
        if not isinstance(residual, torch.Tensor):
            raise TypeError("residual_fn must return a torch.Tensor")
        if residual.ndim == 1 and len(self.problem.residual_components) == 1:
            residual = residual[:, None]
        expected = (self.n_collocation, len(self.problem.residual_components))
        if tuple(residual.shape) != expected:
            raise ValueError(f"residual_fn must return shape {expected}, got {tuple(residual.shape)}")
        if not residual.dtype.is_floating_point:
            raise TypeError("residual_fn must return a floating-point tensor")
        if not bool(torch.all(torch.isfinite(residual)).detach().cpu().item()):
            raise ValueError("residual_fn returned non-finite values")
        return residual

    def estimate(self, nobs: float | None, suff_stat: tuple) -> PINNRegression:
        """Run the data-plus-residual M-step and return the updated PINN leaf."""
        torch = _torch()
        xs, ys, ws = self._validated_data(suff_stat)
        has_data = len(xs) > 0
        dev = _resolve_device(self.device, torch)
        dtype = _place_module(self.module, dev, torch)

        if has_data:
            xt = torch.as_tensor(xs, dtype=dtype, device=dev)
            yt = torch.as_tensor(ys, dtype=dtype, device=dev)
            wt = torch.as_tensor(ws, dtype=dtype, device=dev)
            wsum = wt.sum()
            d = yt.shape[1]

        log_noise = (
            torch.log(torch.tensor(float(self.noise), dtype=dtype, device=dev))
            .clone()
            .detach()
            .requires_grad_(True)
        )
        opt = torch.optim.Adam(list(self.module.parameters()) + [log_noise], lr=self.lr)
        residual_losses: list[float] = []
        for _ in range(self.m_steps):
            opt.zero_grad()
            loss = torch.zeros((), dtype=dtype, device=dev)
            if has_data:
                mean = self.module(xt)
                if tuple(mean.shape) != tuple(yt.shape):
                    raise ValueError(
                        f"PINN module must return observation shape {tuple(yt.shape)}, got {tuple(mean.shape)}"
                    )
                if not bool(torch.all(torch.isfinite(mean)).detach().cpu().item()):
                    raise ValueError("PINN module returned non-finite observation predictions")
                sig2 = torch.exp(2.0 * log_noise)
                nll = (
                    wt * (0.5 * ((yt - mean) ** 2).sum(1) / sig2 + 0.5 * d * torch.log(2.0 * np.pi * sig2))
                ).sum() / wsum
                loss = loss + nll
            coll = self._sample_collocation(dev, dtype, torch)
            residual = self._validated_residual(self.residual_fn(self.module, coll), torch)
            residual_loss = (residual**2).mean()
            residual_losses.append(float(residual_loss.detach().cpu().item()))
            loss = loss + self.residual_weight * residual_loss
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise RuntimeError("PINN objective became non-finite")
            loss.backward()
            opt.step()
            if any(
                not bool(torch.all(torch.isfinite(parameter)).detach().cpu().item())
                for parameter in self.module.parameters()
            ) or not bool(torch.isfinite(log_noise).detach().cpu().item()):
                raise RuntimeError("PINN optimization produced non-finite parameters")
        if has_data:
            self.noise = float(torch.exp(log_noise).detach())  # warm-start noise for the next EM iteration
        receipt = PINNConstraintReceipt(
            status="declared-complete",
            equation=self.problem.equation,
            constraint_enforcement=self.problem.constraint_enforcement,
            constraints=self.problem.constraints,
            identifiability_conditions=self.problem.identifiability_conditions,
            completeness_basis=self.problem.completeness_basis,
            labeled_constraints=len(xs),
            residual_loss_initial=residual_losses[0],
            residual_loss_final=residual_losses[-1],
        )
        return PINNRegression(
            self.module,
            self.residual_fn,
            self.domain,
            problem=self.problem,
            noise=self.noise,
            residual_weight=self.residual_weight,
            n_collocation=self.n_collocation,
            m_steps=self.m_steps,
            lr=self.lr,
            seed=self.seed,
            name=self.name,
            device=self.device,
            constraint_receipt=receipt,
        )


def _register_serializable() -> None:
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(PINNRegression)


def _validated_domain(domain: Any, n_coordinates: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(domain, (tuple, list)) or len(domain) != 2:
        raise ValueError("domain must be a (low, high) pair")
    try:
        low = np.asarray(domain[0], dtype=float)
        high = np.asarray(domain[1], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("domain bounds must be numeric vectors") from exc
    expected = (n_coordinates,)
    if low.shape != expected or high.shape != expected:
        raise ValueError(f"domain bounds must both have shape {expected}, got {low.shape} and {high.shape}")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("domain bounds must be finite")
    if np.any(low >= high):
        raise ValueError("every domain lower bound must be strictly below its upper bound")
    return low, high


def _finite_positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


_register_serializable()
