"""A neural network as a mixle conditional-density leaf -- the bridge that makes nets *generative* components.

``NeuralGaussian(module)`` wraps a Torch module as a mixle distribution ``p(y | x) = N(y; module(x), noise^2 I)``
over observations ``(x, y)``. It implements the full ``SequenceEncodableProbabilityDistribution`` contract, so
it drops into ``MixtureDistribution`` / ``CompositeDistribution`` / HMM emissions like any leaf -- but its EM
**M-step is weighted-NLL gradient descent** on the module (warm-started across EM iterations => generalized EM).

A ``MixtureDistribution`` of ``NeuralGaussian`` components is therefore a **mixture of neural experts**: the E-step
computes responsibilities, the M-step trains each expert by responsibility-weighted regression. Combined with
the ``em`` move in :mod:`mixle.experimental.program`, the same model fits with EM where conjugate and gradient where neural::

    from mixle.stats import MixtureEstimator
    experts = MixtureEstimator([NeuralGaussian(mlp_a).estimator(), NeuralGaussian(mlp_b).estimator()])
    # ... run EM (estimate loop) -> each expert specializes, gated by the responsibilities.

Requires torch (the module). The leaf is conditional: ``sampler().sample_given(x)`` draws ``y``; ``sample()``
raises (there is no ``p(x)``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.models._neural_serial import check_finite, decode_module, encode_module
from mixle.models.grad_leaf import _module_mode, _resolve_device, _resolve_dtype
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _torch() -> Any:
    import torch

    return torch


def _place_module(module: Any, dev: Any, torch: Any) -> Any:
    """Move ``module`` to ``dev`` and return the float dtype its tensor math must use.

    An active torch engine's declared float dtype wins and the module is cast to match, so a
    ``TorchEngine(dtype=torch.float64)`` run evaluates the net in fp64 like the rest of the substrate
    math instead of the old hardcoded fp32. Without an engine policy the module is left untouched and
    the math follows the module's own parameter dtype (``float32`` for a stock torch module -- today's
    behavior). Following the parameters, not a literal, keeps E-step scoring consistent after an
    engine-driven M-step cast the module: the estimation loop activates the engine only around
    ``estimate()``, so the next scoring pass runs outside the engine context."""
    dtype = _resolve_dtype(torch)
    if dtype is not None:
        module.to(dev, dtype=dtype)
        return dtype
    module.to(dev)
    return next((p.dtype for p in module.parameters() if p.dtype.is_floating_point), torch.float32)


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite real number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _matrix(value: Any, name: str, *, allow_empty_rows: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and array.size:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a two-dimensional matrix with a non-empty feature axis")
    if not allow_empty_rows and array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return check_finite(array, name)


def _weights(value: Any, rows: int, name: str = "weights") -> np.ndarray:
    weights = np.asarray(value, dtype=float)
    if weights.ndim != 1 or weights.shape != (rows,):
        raise ValueError(f"{name} must be a one-dimensional array with exactly {rows} entries")
    check_finite(weights, name)
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return weights


def _validate_regression_batch(x: Any, y: Any, where: str) -> tuple[np.ndarray, np.ndarray]:
    x_matrix = _matrix(x, f"{where} (x)")
    y_matrix = _matrix(y, f"{where} (y)")
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError(
            f"{where} requires x and y to have the same number of rows; "
            f"got {x_matrix.shape[0]} and {y_matrix.shape[0]}"
        )
    return x_matrix, y_matrix


class NeuralGaussian(SequenceEncodableProbabilityDistribution):
    """``p(y | x) = N(y; module(x), noise^2 I)`` as a mixle leaf. Observation is the pair ``(x, y)``."""

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self,
        module: Any,
        noise: float = 1.0,
        m_steps: int = 40,
        lr: float = 0.01,
        name: str | None = None,
        device: Any = None,
    ) -> None:
        self.module = module
        self.noise = _positive_finite(noise, "noise")
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        self.name = name
        self.device = device  # None => CUDA if available, else CPU (see _resolve_device)

    def __str__(self) -> str:
        return "NeuralGaussian(noise=%.3g)" % self.noise

    def _forward(self, x: np.ndarray) -> np.ndarray:
        x = _matrix(x, "NeuralGaussian._forward (x)")
        torch = _torch()
        dev = _resolve_device(self.device, torch)
        dtype = _place_module(self.module, dev, torch)
        with _module_mode(self.module, train=False), torch.no_grad():
            mean = self.module(torch.as_tensor(x, dtype=dtype, device=dev))
        if not isinstance(mean, torch.Tensor):
            raise TypeError("NeuralGaussian module must return a torch.Tensor")
        mean_array = np.asarray(mean.detach().cpu().numpy())
        if mean_array.ndim != 2 or mean_array.shape[0] != x.shape[0] or mean_array.shape[1] == 0:
            raise ValueError(
                "NeuralGaussian module output must be a two-dimensional matrix with one row per input; "
                f"got input {x.shape} and output {mean_array.shape}"
            )
        return check_finite(mean_array, "NeuralGaussian module output")

    def log_density(self, xy: Any) -> float:
        """Return ``log p(y | x)`` for one encoded observation pair ``(x, y)``."""
        x, y = xy
        return float(self.seq_log_density((np.atleast_2d(x), np.atleast_2d(y)))[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return per-row Gaussian conditional log densities for encoded ``(x, y)`` arrays."""
        if not isinstance(enc, (tuple, list)) or len(enc) != 2:
            raise ValueError("NeuralGaussian.seq_log_density expects an (x, y) pair")
        x, y = _validate_regression_batch(enc[0], enc[1], "NeuralGaussian.seq_log_density")
        mean = self._forward(x)
        if mean.shape != y.shape:
            raise ValueError(
                "NeuralGaussian target shape must exactly match module output shape; "
                f"got target {y.shape} and output {mean.shape}"
            )
        d = y.shape[1]
        sq = ((y - mean) ** 2).sum(axis=1)
        return -0.5 * sq / (self.noise**2) - 0.5 * d * np.log(2.0 * np.pi * self.noise**2)

    # --- compute-engine backend (numpy + torch/GPU), SCORING: the module forward runs on the leaf's
    # own device (as always), the Gaussian residual math on the active engine — so a mixture of neural
    # experts computes its E-step responsibilities through engine=TorchEngine(...) like any other family.
    # The accumulator (the responsibility-weighted gradient M-step buffer) stays host-side by design. ---
    @classmethod
    def compute_capabilities(cls):
        """Declare engine-ready scoring support for NumPy and Torch execution backends."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    def backend_seq_log_density(self, enc: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded ``(x, y)`` pairs."""
        if not isinstance(enc, (tuple, list)) or len(enc) != 2:
            raise ValueError("NeuralGaussian.backend_seq_log_density expects an (x, y) pair")
        x, y = _validate_regression_batch(enc[0], enc[1], "NeuralGaussian.backend_seq_log_density")
        mean_array = self._forward(x)
        if mean_array.shape != y.shape:
            raise ValueError(
                "NeuralGaussian target shape must exactly match module output shape; "
                f"got target {y.shape} and output {mean_array.shape}"
            )
        mean = engine.asarray(mean_array)
        yy = engine.asarray(y)
        d = int(yy.shape[1])
        resid = yy - mean
        sq = engine.sum(resid * resid, axis=1)
        return -0.5 * sq / (self.noise**2) - 0.5 * d * float(np.log(2.0 * np.pi * self.noise**2))

    def sampler(self, seed: int | None = None) -> NeuralGaussianSampler:
        """Return a conditional sampler for drawing ``y`` given ``x``."""
        return NeuralGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralGaussianEstimator:
        """Return the generalized-EM estimator for responsibility-weighted neural regression."""
        return NeuralGaussianEstimator(self.module, self.noise, self.m_steps, self.lr, self.name, self.device)

    def dist_to_encoder(self) -> NeuralGaussianEncoder:
        """Return the encoder for batches of ``(x, y)`` observation pairs."""
        return NeuralGaussianEncoder()

    # --- serialization: persist hparams + the module (as portable bytes); registered below so a mixture holding
    # this leaf round-trips through to_dict/to_json/pickle as well. ---
    def __pysp_getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["module"] = encode_module(self.module)
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.module = decode_module(state["module"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize hyperparameters and module bytes for registry-based round trips."""
        return {
            "noise": self.noise,
            "m_steps": self.m_steps,
            "lr": self.lr,
            "name": self.name,
            "device": self.device,
            "module": encode_module(self.module),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NeuralGaussian:
        """Rebuild a :class:`NeuralGaussian` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["module"]),
            noise=payload["noise"],
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            name=payload["name"],
            device=payload["device"],
        )


class NeuralGaussianSampler(DistributionSampler):
    """Conditional sampler for :class:`NeuralGaussian`; draws responses given covariates."""

    def __init__(self, dist: NeuralGaussian, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because the leaf defines ``p(y | x)`` and has no marginal ``p(x)``."""
        raise NotImplementedError("NeuralGaussian is conditional p(y|x); use sampler().sample_given(x).")

    def sample_given(self, x: Any) -> np.ndarray:
        """Draw Gaussian responses from ``p(y | x)``, preserving an input batch axis when supplied."""
        x_array = np.asarray(x)
        batched = x_array.ndim == 2
        mean = self.dist._forward(x)
        sample = mean + self.dist.noise * self.rng.randn(*mean.shape)
        return sample if batched else sample[0]


class NeuralGaussianEncoder(DataSequenceEncoder):
    """Encode ``(x, y)`` observation pairs for vectorized neural-Gaussian scoring and fitting."""

    def __str__(self) -> str:
        return "NeuralGaussianEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralGaussianEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        """Convert a list of ``(x, y)`` pairs into batched feature and target arrays."""
        x = np.array([np.atleast_1d(np.asarray(xy[0], dtype=float)) for xy in data])
        y = np.array([np.atleast_1d(np.asarray(xy[1], dtype=float)) for xy in data])
        return (x, y)


class NeuralGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffer weighted ``(x, y)`` batches for the neural-Gaussian M-step."""

    def __init__(self) -> None:
        self.x: list = []
        self.y: list = []
        self.w: list = []

    # x/y/w hold contiguous (n_i, dim) batch arrays and concatenate once at value(), avoiding per-row ndarray
    # buffering during streamed updates.
    def update(self, xy: Any, weight: float, estimate: Any) -> None:
        """Add one weighted observation pair to the accumulator."""
        if not isinstance(xy, (tuple, list)) or len(xy) != 2:
            raise ValueError("NeuralGaussianAccumulator.update expects an (x, y) pair")
        x, y = _validate_regression_batch(xy[0], xy[1], "NeuralGaussianAccumulator.update")
        if x.shape[0] != 1:
            raise ValueError("NeuralGaussianAccumulator.update accepts exactly one observation")
        weights = _weights([weight], 1)
        self.x.append(x)
        self.y.append(y)
        self.w.append(weights)

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Add a batch of encoded observation pairs and responsibility weights."""
        if not isinstance(enc, (tuple, list)) or len(enc) != 2:
            raise ValueError("NeuralGaussianAccumulator.seq_update expects an (x, y) pair")
        x, y = _validate_regression_batch(enc[0], enc[1], "NeuralGaussianAccumulator.seq_update")
        self.x.append(x)
        self.y.append(y)
        self.w.append(_weights(weights, x.shape[0]))

    def initialize(self, xy: Any, weight: float, rng: Any) -> None:
        """Initialize from one observation using the ordinary update path."""
        self.update(xy, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        """Initialize from an encoded batch using the ordinary batch update path."""
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralGaussianAccumulator:
        """Merge the value tuple from another neural-Gaussian accumulator."""
        xo, yo, wo = other  # another accumulator's value(): contiguous (n, dim) arrays
        if len(xo):
            x, y = _validate_regression_batch(xo, yo, "NeuralGaussianAccumulator.combine")
            self.x.append(x)
            self.y.append(y)
            self.w.append(_weights(wo, x.shape[0]))
        elif len(yo) or len(wo):
            raise ValueError("an empty neural-Gaussian accumulator must have empty x, y, and weights")
        return self

    def value(self) -> tuple:
        """Return contiguous ``(x, y, weights)`` arrays for the M-step."""
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        y = np.concatenate(self.y, axis=0) if self.y else np.zeros((0, 0))
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, y, w)

    def from_value(self, value: tuple) -> NeuralGaussianAccumulator:
        """Restore accumulator buffers from a value tuple."""
        x, y, w = value
        self.x = []
        self.y = []
        self.w = []
        self.combine((x, y, w))
        return self

    def acc_to_encoder(self) -> NeuralGaussianEncoder:
        """Return the encoder expected by this accumulator."""
        return NeuralGaussianEncoder()


class NeuralGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for neural-Gaussian accumulators."""

    def make(self) -> NeuralGaussianAccumulator:
        """Create a fresh accumulator."""
        return NeuralGaussianAccumulator()


class NeuralGaussianEstimator(ParameterEstimator):
    """EM estimator for a :class:`NeuralGaussian`: the M-step is ``m_steps`` of weighted-NLL gradient on the module.

    The module is held (and warm-started) across EM iterations, so each M-step is a *partial* maximization
    (generalized EM). The accumulator buffers responsibility-weighted ``(x, y)`` observations.
    """

    def __init__(
        self,
        module: Any,
        noise: float = 1.0,
        m_steps: int = 40,
        lr: float = 0.01,
        name: str | None = None,
        device: Any = None,
    ) -> None:
        self.module = module
        self.noise = _positive_finite(noise, "noise")
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        self.name = name
        self.device = device

    def accumulator_factory(self) -> NeuralGaussianAccumulatorFactory:
        """Return an accumulator factory for weighted neural-regression batches."""
        return NeuralGaussianAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: tuple) -> NeuralGaussian:
        """Run the weighted neural-regression M-step and return the updated leaf."""
        torch = _torch()
        xs, ys, ws = suff_stat
        if len(xs) == 0:
            if len(ys) or len(ws):
                raise ValueError("empty neural-Gaussian sufficient statistics must have empty x, y, and weights")
            return NeuralGaussian(self.module, self.noise, self.m_steps, self.lr, self.name, self.device)
        xs, ys = _validate_regression_batch(xs, ys, "NeuralGaussianEstimator.estimate")
        ws = _weights(ws, xs.shape[0])
        if not np.any(ws > 0.0):
            raise ValueError("NeuralGaussianEstimator.estimate requires positive effective weight")
        dev = _resolve_device(self.device, torch)
        dtype = _place_module(self.module, dev, torch)
        xt = torch.as_tensor(xs, dtype=dtype, device=dev)
        yt = torch.as_tensor(ys, dtype=dtype, device=dev)
        wt = torch.as_tensor(ws, dtype=dtype, device=dev)
        wsum = wt.sum()
        log_noise = (
            torch.log(torch.tensor(float(self.noise), dtype=dtype, device=dev)).clone().detach().requires_grad_(True)
        )
        opt = torch.optim.Adam(list(self.module.parameters()) + [log_noise], lr=self.lr)
        d = yt.shape[1]
        with _module_mode(self.module, train=True):
            for _ in range(self.m_steps):
                opt.zero_grad()
                mean = self.module(xt)
                if not isinstance(mean, torch.Tensor) or tuple(mean.shape) != tuple(yt.shape):
                    actual_shape = getattr(mean, "shape", None)
                    raise ValueError(
                        "NeuralGaussian module output must exactly match target shape during estimation; "
                        f"got target {tuple(yt.shape)} and output {actual_shape}"
                    )
                if not bool(torch.isfinite(mean).all()):
                    raise ValueError("NeuralGaussian module produced non-finite output during estimation")
                sig2 = torch.exp(2.0 * log_noise)
                nll = (
                    wt * (0.5 * ((yt - mean) ** 2).sum(1) / sig2 + 0.5 * d * torch.log(2.0 * np.pi * sig2))
                ).sum() / wsum
                if not bool(torch.isfinite(nll)):
                    raise ValueError("NeuralGaussian estimation objective became non-finite")
                nll.backward()
                opt.step()
        self.noise = _positive_finite(
            float(torch.exp(log_noise).detach()), "fitted noise"
        )  # warm-start noise for the next EM iteration
        return NeuralGaussian(self.module, self.noise, self.m_steps, self.lr, self.name, self.device)


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(NeuralGaussian)


_register_serializable()


# --- back-compat aliases (the classes were renamed off the '...Leaf' suffix) ---
NeuralLeaf = NeuralGaussian
NeuralLeafEstimator = NeuralGaussianEstimator
