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
from mixle.models.grad_leaf import DataBufferAccumulatorFactory
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)


def _torch() -> Any:
    import torch

    return torch


def _resolve_device(device: Any, torch: Any) -> Any:
    """Where to run the module, in priority order:

    1. an explicit ``device=`` on the leaf (always wins);
    2. the device of the **active compute engine** -- so ``optimize(engine=TorchEngine(device="mps"))``
       (or ``"cuda"``) drives the leaf's M-step onto that device, matching mixle's engine philosophy
       (set the device once on the engine, the leaf follows);
    3. otherwise CUDA if available, else CPU -- the implicit default (note: not MPS, so existing local
       CPU behaviour and tests are unchanged; reach MPS explicitly or via the engine)."""
    if device is not None:
        return torch.device(device)
    from mixle.engines.base import active_engine

    eng_dev = getattr(active_engine(), "device", None)
    if eng_dev is not None and str(eng_dev) != "cpu":
        try:
            return torch.device(eng_dev)
        except (TypeError, RuntimeError):
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.noise = float(noise)
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.name = name
        self.device = device  # None => CUDA if available, else CPU (see _resolve_device)

    def __str__(self) -> str:
        return "NeuralGaussian(noise=%.3g)" % self.noise

    def _forward(self, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        dev = _resolve_device(self.device, torch)
        self.module.to(dev)
        with torch.no_grad():
            mean = self.module(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32, device=dev))
        return np.atleast_2d(mean.detach().cpu().numpy())

    def log_density(self, xy: Any) -> float:
        x, y = xy
        return float(self.seq_log_density((np.atleast_2d(x), np.atleast_2d(y)))[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        x, y = enc
        check_finite(np.atleast_2d(np.asarray(x, dtype=float)), "NeuralGaussian.seq_log_density (x)")
        mean = self._forward(x)
        y = check_finite(np.atleast_2d(np.asarray(y, dtype=float)), "NeuralGaussian.seq_log_density (y)")
        d = y.shape[1]
        sq = ((y - mean) ** 2).sum(axis=1)
        return -0.5 * sq / (self.noise**2) - 0.5 * d * np.log(2.0 * np.pi * self.noise**2)

    # --- compute-engine backend (numpy + torch/GPU), SCORING: the module forward runs on the leaf's
    # own device (as always), the Gaussian residual math on the active engine — so a mixture of neural
    # experts computes its E-step responsibilities through engine=TorchEngine(...) like any other family.
    # The accumulator (the responsibility-weighted gradient M-step buffer) stays host-side by design. ---
    @classmethod
    def compute_capabilities(cls):
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    def backend_seq_log_density(self, enc: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded ``(x, y)`` pairs."""
        x, y = enc
        mean = engine.asarray(self._forward(x))
        yy = engine.asarray(np.atleast_2d(np.asarray(y, dtype=float)))
        d = int(yy.shape[1])
        resid = yy - mean
        sq = engine.sum(resid * resid, axis=1)
        return -0.5 * sq / (self.noise**2) - 0.5 * d * float(np.log(2.0 * np.pi * self.noise**2))

    def sampler(self, seed: int | None = None) -> NeuralGaussianSampler:
        return NeuralGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralGaussianEstimator:
        return NeuralGaussianEstimator(self.module, self.noise, self.m_steps, self.lr, self.name, self.device)

    def dist_to_encoder(self) -> NeuralGaussianEncoder:
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
        return cls(
            decode_module(payload["module"]),
            noise=payload["noise"],
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            name=payload["name"],
            device=payload["device"],
        )


class NeuralGaussianSampler(DistributionSampler):
    def __init__(self, dist: NeuralGaussian, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        raise NotImplementedError("NeuralGaussian is conditional p(y|x); use sampler().sample_given(x).")

    def sample_given(self, x: Any) -> np.ndarray:
        mean = self.dist._forward(x)[0]
        return mean + self.dist.noise * self.rng.randn(*mean.shape)


class NeuralGaussianEncoder(DataSequenceEncoder):
    def __str__(self) -> str:
        return "NeuralGaussianEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralGaussianEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([np.atleast_1d(np.asarray(xy[0], dtype=float)) for xy in data])
        y = np.array([np.atleast_1d(np.asarray(xy[1], dtype=float)) for xy in data])
        return (x, y)


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
        self.noise = float(noise)
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.name = name
        self.device = device

    def accumulator_factory(self) -> DataBufferAccumulatorFactory:
        return DataBufferAccumulatorFactory(NeuralGaussianEncoder(), n_fields=2)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> NeuralGaussian:
        torch = _torch()
        xs, ys, ws = suff_stat
        if len(xs) == 0:
            return NeuralGaussian(self.module, self.noise, self.m_steps, self.lr, self.name, self.device)
        dev = _resolve_device(self.device, torch)
        self.module.to(dev)
        xs = np.asarray(xs, dtype=float).reshape(len(xs), -1)
        ys = np.asarray(ys, dtype=float).reshape(len(ys), -1)
        xt = torch.as_tensor(xs, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(ys, dtype=torch.float32, device=dev)
        wt = torch.as_tensor(np.array(ws), dtype=torch.float32, device=dev)
        wsum = float(wt.sum()) + 1e-8
        log_noise = torch.log(torch.tensor(float(self.noise), device=dev)).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam(list(self.module.parameters()) + [log_noise], lr=self.lr)
        d = yt.shape[1]
        for _ in range(self.m_steps):
            opt.zero_grad()
            mean = self.module(xt)
            sig2 = torch.exp(2.0 * log_noise)
            nll = (wt * (0.5 * ((yt - mean) ** 2).sum(1) / sig2 + 0.5 * d * torch.log(2.0 * np.pi * sig2))).sum() / wsum
            nll.backward()
            opt.step()
        self.noise = float(torch.exp(log_noise).detach())  # warm-start noise for the next EM iteration
        return NeuralGaussian(self.module, self.noise, self.m_steps, self.lr, self.name, self.device)


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover
        return
    register_serializable_class(NeuralGaussian)


_register_serializable()


# --- back-compat aliases (the classes were renamed off the '...Leaf' suffix) ---
NeuralLeaf = NeuralGaussian
NeuralLeafEstimator = NeuralGaussianEstimator
