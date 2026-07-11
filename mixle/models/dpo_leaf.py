"""Direct Preference Optimization (DPO) as a mixle leaf -- alignment as a supervised preference likelihood.

Observation = ``(x, chosen, rejected)``: a context and a preferred vs dispreferred action/completion. The leaf
carries a POLICY module and a FROZEN REFERENCE module; ``seq_log_density`` returns the DPO log-sigmoid reward

    log sigma( beta * [ (log pi(chosen|x) - log pi_ref(chosen|x)) - (log pi(rejected|x) - log pi_ref(rejected|x)) ] )

(higher = the policy prefers chosen over rejected, relative to the reference). The M-step gradient-steps the
policy; the reference stays frozen. **No reward model, no RL** -- the alignment stage of the LLM pipeline as a
likelihood, on the same substrate as pretrain/CPT/SFT.

This is the genuinely-new *paired* leaf the design flagged: it couples two forward passes plus a frozen
reference, so it does not reduce to a single ``Categorical`` (the ``log_density`` contract is over a *pair*, not
a single token). It composes through the same ``estimate()`` driver; the M-step owns the policy optimizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.models._neural_serial import decode_module, encode_module
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


def _logp_np(logits: np.ndarray, a: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return logp[np.arange(len(a)), a]


class DPOModel(SequenceEncodableProbabilityDistribution):
    """DPO over ``(x, chosen, rejected)`` preference triples. ``policy`` is trained, ``ref`` is frozen."""

    __pysp_serializable__ = True  # modules persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self, policy: Any, ref: Any, beta: float = 0.1, m_steps: int = 100, lr: float = 1e-3, device: str = "cpu"
    ) -> None:
        self.policy = policy
        self.ref = ref
        self.beta = float(beta)
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device

    def __str__(self) -> str:
        return "DPOModel(beta=%.3g)" % self.beta

    def _logits(self, module: Any, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        module.to(self.device)
        with torch.no_grad():
            return module(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32).to(self.device)).cpu().numpy()

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return per-row DPO preference log likelihoods for encoded triples."""
        x, ch, rj = enc
        ch = np.asarray(ch, dtype=int)
        rj = np.asarray(rj, dtype=int)
        lp_pol = self._logits(self.policy, x)
        lp_ref = self._logits(self.ref, x)
        margin = (_logp_np(lp_pol, ch) - _logp_np(lp_ref, ch)) - (_logp_np(lp_pol, rj) - _logp_np(lp_ref, rj))
        return -np.logaddexp(0.0, -self.beta * margin)  # log sigmoid(beta * margin)

    def log_density(self, xcr: Any) -> float:
        """Return the DPO log likelihood for one ``(x, chosen, rejected)`` triple."""
        x, ch, rj = xcr
        return float(self.seq_log_density((np.atleast_2d(x), [int(ch)], [int(rj)]))[0])

    def prefers(self, x: Any) -> np.ndarray:
        """The policy's argmax action at ``x`` -- what the aligned policy now picks."""
        return self._logits(self.policy, x).argmax(axis=1)

    def sampler(self, seed: int | None = None) -> DPOModelSampler:
        """Return the sampler for the preference-scoring leaf."""
        return DPOModelSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> DPOModelEstimator:
        """Return the DPO estimator that trains the policy while keeping the reference fixed."""
        return DPOModelEstimator(self.policy, self.ref, self.beta, self.m_steps, self.lr, self.device)

    def dist_to_encoder(self) -> DPOEncoder:
        """Return the encoder for preference triples."""
        return DPOEncoder()

    # --- serialization: persist hparams + both modules (as portable bytes); registered below so a mixture
    # holding this leaf round-trips through to_dict/to_json/pickle as well. ---
    def __pysp_getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["policy"] = encode_module(self.policy)
        state["ref"] = encode_module(self.ref)
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.policy = decode_module(state["policy"])
        self.ref = decode_module(state["ref"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy/reference modules and DPO hyperparameters."""
        return {
            "policy": encode_module(self.policy),
            "ref": encode_module(self.ref),
            "beta": self.beta,
            "m_steps": self.m_steps,
            "lr": self.lr,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DPOModel:
        """Rebuild a :class:`DPOModel` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["policy"]),
            decode_module(payload["ref"]),
            beta=payload["beta"],
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            device=payload["device"],
        )


class DPOModelSampler(DistributionSampler):
    """Sampler facade for DPO leaves, which score preference pairs rather than generating."""

    def __init__(self, dist: DPOModel, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because DPO is a preference-scoring likelihood, not a generator."""
        raise NotImplementedError("DPOModel scores preference pairs; it does not generate.")


class DPOEncoder(DataSequenceEncoder):
    """Encode ``(context, chosen, rejected)`` preference triples for DPO."""

    def __str__(self) -> str:
        return "DPOEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DPOEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert preference triples into batched contexts and integer action arrays."""
        x = np.array([np.atleast_1d(np.asarray(d[0], dtype=float)) for d in data])
        ch = np.array([int(d[1]) for d in data], dtype=int)
        rj = np.array([int(d[2]) for d in data], dtype=int)
        return (x, ch, rj)


class DPOAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffer weighted preference triples for the DPO M-step."""

    def __init__(self) -> None:
        self.x: list = []
        self.ch: list = []
        self.rj: list = []
        self.w: list = []  # per-pair weight (EM responsibility / streaming decay / sample weight)

    def update(self, xcr: Any, weight: float, estimate: Any) -> None:
        """Add one weighted preference triple to the accumulator."""
        self.x.append(np.atleast_1d(np.asarray(xcr[0], dtype=float)))
        self.ch.append(int(xcr[1]))
        self.rj.append(int(xcr[2]))
        self.w.append(float(weight))

    def seq_update(self, enc: Any, weights: Any, estimate: Any) -> None:
        """Add an encoded batch of preference triples and optional weights."""
        x, ch, rj = enc
        ws = np.asarray(weights, dtype=float).ravel() if weights is not None else np.ones(len(x))
        for i in range(len(x)):
            self.x.append(np.atleast_1d(x[i]))
            self.ch.append(int(ch[i]))
            self.rj.append(int(rj[i]))
            self.w.append(float(ws[i]))

    def initialize(self, xcr: Any, weight: float, rng: Any) -> None:
        """Initialize from one preference triple using the ordinary update path."""
        self.update(xcr, weight, None)

    def seq_initialize(self, enc: Any, weights: Any, rng: Any) -> None:
        """Initialize from an encoded batch using the ordinary batch update path."""
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> DPOAccumulator:
        """Merge the value tuple from another DPO accumulator."""
        xo, co, ro, wo = other
        self.x.extend(xo)
        self.ch.extend(co)
        self.rj.extend(ro)
        self.w.extend(wo)
        return self

    def value(self) -> tuple:
        """Return buffered contexts, chosen actions, rejected actions, and weights."""
        return (list(self.x), list(self.ch), list(self.rj), np.asarray(self.w, dtype=float))

    def from_value(self, v: tuple) -> DPOAccumulator:
        """Restore accumulator buffers from a value tuple."""
        self.x, self.ch, self.rj, self.w = list(v[0]), list(v[1]), list(v[2]), list(v[3])
        return self

    def acc_to_encoder(self) -> DPOEncoder:
        """Return the encoder expected by this accumulator."""
        return DPOEncoder()


class DPOAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for DPO accumulators."""

    def make(self) -> DPOAccumulator:
        """Create a fresh accumulator."""
        return DPOAccumulator()


class DPOModelEstimator(ParameterEstimator):
    """DPO M-step: ``m_steps`` of gradient on the POLICY minimizing ``-log sigmoid(beta * margin)``; ref frozen."""

    outer_objective_compatible = False

    def __init__(self, policy: Any, ref: Any, beta: float, m_steps: int, lr: float, device: str) -> None:
        self.policy = policy
        self.ref = ref
        self.beta = float(beta)
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device

    def accumulator_factory(self) -> DPOAccumulatorFactory:
        """Return an accumulator factory for weighted preference triples."""
        return DPOAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: tuple) -> DPOModel:
        """Run the weighted DPO M-step and return the updated policy leaf."""
        torch = _torch()
        xs, chs, rjs, ws = suff_stat
        out = DPOModel(self.policy, self.ref, self.beta, self.m_steps, self.lr, self.device)
        if len(xs) == 0:
            return out
        dev = self.device
        self.policy.to(dev)
        self.ref.to(dev)
        for p in self.ref.parameters():
            p.requires_grad_(False)  # frozen reference
        # the generic buffer stores every field as float batch arrays; restore shapes/dtypes at tensor prep
        xt = torch.as_tensor(np.asarray(xs, dtype=float).reshape(len(xs), -1), dtype=torch.float32).to(dev)
        ct = torch.as_tensor(np.asarray(chs).ravel().astype(int), dtype=torch.long).to(dev)
        rt = torch.as_tensor(np.asarray(rjs).ravel().astype(int), dtype=torch.long).to(dev)
        wt = torch.as_tensor(np.asarray(ws, dtype=float), dtype=torch.float32).to(dev)  # per-pair weight
        wsum = wt.sum().clamp(min=1e-8)
        ar = torch.arange(len(ct), device=dev)
        opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        with torch.no_grad():  # reference log-probs are constant -- compute once
            lr_all = torch.log_softmax(self.ref(xt), dim=1)
            lr_ch, lr_rj = lr_all[ar, ct], lr_all[ar, rt]
        for _ in range(self.m_steps):
            opt.zero_grad()
            lp = torch.log_softmax(self.policy(xt), dim=1)
            margin = (lp[ar, ct] - lr_ch) - (lp[ar, rt] - lr_rj)
            loss = -(wt * torch.nn.functional.logsigmoid(self.beta * margin)).sum() / wsum  # weighted DPO loss
            loss.backward()
            opt.step()
        return out


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(DPOModel)


_register_serializable()


# --- back-compat aliases (the classes were renamed off the '...Leaf' suffix) ---
DPOLeaf = DPOModel
DPOLeafEstimator = DPOModelEstimator
