"""A neural classifier as a Mixle conditional-density leaf: ``p(y | x) = softmax(module(x))``.

The discriminative sibling of :class:`~mixle.models.neural_leaf.NeuralGaussian`. ``NeuralCategorical(module)`` wraps
a Torch module that emits ``k`` logits as a mixle distribution over observations ``(x, y)`` with ``y`` an integer
class index. It implements the full ``SequenceEncodableProbabilityDistribution`` contract, so it drops into
``MixtureDistribution`` / ``CompositeDistribution`` / HMM emissions like any leaf -- and its EM **M-step is a
responsibility-weighted cross-entropy gradient step** on the module (warm-started across EM iterations =>
generalized EM). The model's ``seq_log_density`` IS ``-cross_entropy(module(x), y)``: the objective is the
leaf's log-density, never a user-supplied loss closure.

This is the leaf that the declarative ``Categorical(logits=Net(...))`` PPL slot lowers to, and the component
that makes a ``Mix([Categorical(logits=Net(...)), ...])`` a mixture of neural classifiers fit by ordinary EM.

Requires torch. The leaf is conditional: ``predict(x)`` and ``sampler().sample_given(x)`` work; ``sample()`` raises
because the model has no marginal ``p(x)``. This is the same conditional contract used by ``NeuralGaussian`` and
``RandomForestConditional``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.models._neural_serial import check_finite, decode_module, encode_module
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


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))


class NeuralCategorical(SequenceEncodableProbabilityDistribution):
    """``p(y | x) = softmax(module(x))`` as a mixle leaf. Observation is the pair ``(x, y)``, ``y`` an int class.

    ``batch_size`` (None = full batch) makes the M-step minibatch SGD over ``m_steps`` passes -- needed to train a
    real conv net on a large image set; ``device`` (e.g. ``"mps"``/``"cuda"``) runs it on the GPU.
    """

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self,
        module: Any,
        m_steps: int = 40,
        lr: float = 0.01,
        name: str | None = None,
        batch_size: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.name = name
        self.batch_size = None if batch_size is None else int(batch_size)
        self.device = device

    def __str__(self) -> str:
        return "NeuralCategorical()"

    def _logits(self, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        self.module.to(self.device)
        out = []
        with torch.no_grad():
            xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
            for k in range(0, xt.shape[0], 4096):  # chunked so a large image set fits in GPU memory
                out.append(self.module(xt[k : k + 4096].to(self.device)).detach().cpu().numpy())
        return np.atleast_2d(np.concatenate(out))

    def log_density(self, xy: Any) -> float:
        """Return ``log p(y | x)`` for one feature/class observation pair."""
        x, y = xy
        return float(self.seq_log_density((np.atleast_2d(x), np.array([int(y)])))[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return per-row categorical conditional log probabilities for encoded pairs."""
        x, y = enc
        check_finite(np.atleast_2d(np.asarray(x, dtype=float)), "NeuralCategorical.seq_log_density")
        logp = _log_softmax(self._logits(x))
        y = np.asarray(y, dtype=int)
        return logp[np.arange(len(y)), y]

    def predict(self, x: Any) -> np.ndarray:
        """Return maximum-probability class predictions for one or more inputs."""
        p = self._logits(x).argmax(axis=1)
        return int(p[0]) if np.ndim(x) == 1 else p

    def sampler(self, seed: int | None = None) -> NeuralCategoricalSampler:
        """Return a conditional sampler over labels given features."""
        return NeuralCategoricalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralCategoricalEstimator:
        """Return the generalized-EM estimator for weighted cross-entropy training."""
        return NeuralCategoricalEstimator(self.module, self.m_steps, self.lr, self.name, self.batch_size, self.device)

    def dist_to_encoder(self) -> NeuralCategoricalEncoder:
        """Return the encoder for ``(x, class)`` observation pairs."""
        return NeuralCategoricalEncoder()

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
            "m_steps": self.m_steps,
            "lr": self.lr,
            "name": self.name,
            "batch_size": self.batch_size,
            "device": self.device,
            "module": encode_module(self.module),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NeuralCategorical:
        """Rebuild a :class:`NeuralCategorical` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["module"]),
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            name=payload["name"],
            batch_size=payload["batch_size"],
            device=payload["device"],
        )


class NeuralCategoricalSampler(DistributionSampler):
    """Conditional sampler over class labels for :class:`NeuralCategorical`."""

    def __init__(self, dist: NeuralCategorical, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because the leaf defines ``p(y | x)`` and has no marginal ``p(x)``."""
        raise NotImplementedError("NeuralCategorical is conditional p(y|x); use sampler().sample_given(x).")

    def sample_given(self, x: Any) -> int:
        """Draw one class label from ``p(y | x)``."""
        p = np.exp(_log_softmax(self.dist._logits(x))[0])
        return int(self.rng.choice(len(p), p=p / p.sum()))


class NeuralCategoricalEncoder(DataSequenceEncoder):
    """Encode feature/class pairs for neural-categorical scoring and fitting."""

    def __str__(self) -> str:
        return "NeuralCategoricalEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralCategoricalEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        """Convert ``(x, class)`` pairs into batched feature and integer-label arrays."""
        x = np.array([np.atleast_1d(np.asarray(xy[0], dtype=float)) for xy in data])
        y = np.array([int(xy[1]) for xy in data], dtype=int)
        return (x, y)


class NeuralCategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffer weighted feature/class batches for the neural-categorical M-step."""

    def __init__(self) -> None:
        self.x: list = []
        self.y: list = []
        self.w: list = []

    # x/y/w hold contiguous batch arrays and concatenate once at value(), avoiding per-row ndarray buffering.
    # x batching is shape-preserving so conv/structured inputs survive; y stays an integer class index.
    def update(self, xy: Any, weight: float, estimate: Any) -> None:
        """Add one weighted feature/class pair to the accumulator."""
        self.x.append(np.atleast_1d(np.asarray(xy[0], dtype=float))[None, ...])
        self.y.append(np.asarray([int(xy[1])], dtype=int))
        self.w.append(np.asarray([float(weight)], dtype=float))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Add an encoded batch and responsibility weights to the accumulator."""
        x, y = enc
        xb = np.asarray(x, dtype=float)
        self.x.append(xb.reshape(xb.shape[0], 1) if xb.ndim == 1 else xb)
        self.y.append(np.asarray(y, dtype=int).ravel())
        self.w.append(np.asarray(weights, dtype=float).ravel())

    def initialize(self, xy: Any, weight: float, rng: Any) -> None:
        """Initialize from one observation using the ordinary update path."""
        self.update(xy, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        """Initialize from an encoded batch using the ordinary batch update path."""
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralCategoricalAccumulator:
        """Merge the value tuple from another categorical accumulator."""
        xo, yo, wo = other
        if len(xo):
            self.x.append(np.asarray(xo, dtype=float))
            self.y.append(np.asarray(yo, dtype=int).ravel())
            self.w.append(np.asarray(wo, dtype=float).ravel())
        return self

    def value(self) -> tuple:
        """Return contiguous ``(x, class, weights)`` arrays for the M-step."""
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        y = np.concatenate(self.y) if self.y else np.zeros((0,), dtype=int)
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, y, w)

    def from_value(self, value: tuple) -> NeuralCategoricalAccumulator:
        """Restore accumulator buffers from a value tuple."""
        x, y, w = value
        self.x = [np.asarray(x, dtype=float)] if len(x) else []
        self.y = [np.asarray(y, dtype=int).ravel()] if len(y) else []
        self.w = [np.asarray(w, dtype=float).ravel()] if len(w) else []
        return self

    def acc_to_encoder(self) -> NeuralCategoricalEncoder:
        """Return the encoder expected by this accumulator."""
        return NeuralCategoricalEncoder()


class NeuralCategoricalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for neural-categorical accumulators."""

    def make(self) -> NeuralCategoricalAccumulator:
        """Create a fresh accumulator."""
        return NeuralCategoricalAccumulator()


class NeuralCategoricalEstimator(ParameterEstimator):
    """EM estimator for a :class:`NeuralCategorical`: the M-step is ``m_steps`` of responsibility-weighted
    cross-entropy gradient on the module (the module is warm-started across EM iterations => generalized EM).

    The weighted CE is normalized by the responsibility mass ``sum(w)`` so the M-step is scale-invariant to the
    responsibility magnitude (the easy bug: an unnormalized weighted loss makes the step size track cluster size).
    """

    def __init__(
        self,
        module: Any,
        m_steps: int = 40,
        lr: float = 0.01,
        name: str | None = None,
        batch_size: int | None = None,
        device: str = "cpu",
        ewc: Any = None,
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.name = name
        self.batch_size = None if batch_size is None else int(batch_size)
        self.device = device
        # ewc = (anchor_params, fisher_diag, lambda): the EWC anti-forgetting penalty for continued pretraining
        self.ewc = ewc

    def accumulator_factory(self) -> NeuralCategoricalAccumulatorFactory:
        """Return an accumulator factory for weighted classification batches."""
        return NeuralCategoricalAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: tuple) -> NeuralCategorical:
        """Run the weighted cross-entropy M-step and return the updated leaf."""
        torch = _torch()
        xs, ys, ws = suff_stat
        out = NeuralCategorical(self.module, self.m_steps, self.lr, self.name, self.batch_size, self.device)
        if len(xs) == 0:
            return out
        dev = self.device
        self.module.to(dev)
        # data stays on CPU (a large image set won't fit on the GPU); each minibatch is moved to the device.
        # x arrives shape-preserving from the buffer so conv/structured inputs survive; the generic buffer
        # stores labels as a (n, 1) float64 column -- integral class indices cast to long exactly.
        xt = torch.as_tensor(np.array(xs), dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(ys).reshape(-1), dtype=torch.long)
        wt = torch.as_tensor(np.array(ws), dtype=torch.float32)
        n = xt.shape[0]
        bs = self.batch_size or n
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        ewc = None
        if self.ewc is not None:  # anchor + Fisher moved to the device once (continued-pretraining anti-forget)
            anchor, fisher, lam = self.ewc
            ewc = ([a.to(dev) for a in anchor], [f.to(dev) for f in fisher], float(lam))
        for _ in range(self.m_steps):  # m_steps passes over the data (full-batch when batch_size is None)
            perm = torch.randperm(n) if bs < n else torch.arange(n)
            for k in range(0, n, bs):
                idx = perm[k : k + bs]
                xb, yb, wb = xt[idx].to(dev), yt[idx].to(dev), wt[idx].to(dev)
                opt.zero_grad()
                # responsibility-weighted CE, normalized by the batch's responsibility mass (scale-invariant)
                loss = (wb * ce(self.module(xb), yb)).sum() / (wb.sum() + 1e-8)
                if ewc is not None:  # + lambda * sum_i F_i (theta_i - theta*_i)^2 -- pull the important weights back
                    anchor, fisher, lam = ewc
                    loss = loss + lam * sum(
                        (f * (p - a) ** 2).sum() for p, a, f in zip(self.module.parameters(), anchor, fisher)
                    )
                loss.backward()
                opt.step()
        return out


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(NeuralCategorical)


_register_serializable()


# --- back-compat aliases (the classes were renamed off the '...Leaf' suffix) ---
SoftmaxNeuralLeaf = NeuralCategorical
SoftmaxNeuralLeafEstimator = NeuralCategoricalEstimator
