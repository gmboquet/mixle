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
from mixle.models.grad_leaf import _module_mode
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
    logits = np.asarray(logits, dtype=float)
    actions = _exact_actions(a, "actions", len(logits))
    if logits.ndim != 2 or logits.shape[0] != len(actions) or logits.shape[1] < 2:
        raise ValueError("logits must have shape (n, actions) with at least two actions")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must contain only finite values")
    if np.any(actions >= logits.shape[1]):
        raise ValueError(f"actions must lie in [0, {logits.shape[1]})")
    m = logits.max(axis=1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return logp[np.arange(len(actions)), actions]


class DPOModel(SequenceEncodableProbabilityDistribution):
    """DPO over ``(x, chosen, rejected)`` preference triples. ``policy`` is trained, ``ref`` is frozen."""

    __pysp_serializable__ = True  # modules persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self, policy: Any, ref: Any, beta: float = 0.1, m_steps: int = 100, lr: float = 1e-3, device: str = "cpu"
    ) -> None:
        torch = _torch()
        _validate_independent_modules(policy, ref, torch)
        self.policy = policy
        self.ref = ref
        self.beta = _positive_finite(beta, "beta")
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        try:
            self.device = str(torch.device(device))
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid torch device {device!r}") from exc
        for parameter in self.ref.parameters():
            parameter.requires_grad_(False)

    def __str__(self) -> str:
        return "DPOModel(beta=%.3g)" % self.beta

    def _logits(self, module: Any, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        module.to(self.device)
        context = _contexts(x)
        dtype = next(
            (parameter.dtype for parameter in module.parameters() if parameter.dtype.is_floating_point),
            torch.float32,
        )
        with _module_mode(module, train=False), torch.no_grad():
            output = module(torch.as_tensor(context, dtype=dtype, device=self.device))
        logits = np.asarray(output.detach().cpu().numpy(), dtype=float)
        if logits.ndim != 2 or logits.shape[0] != len(context) or logits.shape[1] < 2:
            raise ValueError("DPO modules must return shape (n, actions) with at least two actions")
        if not np.all(np.isfinite(logits)):
            raise ValueError("DPO modules returned non-finite logits")
        return logits

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return per-row DPO preference log likelihoods for encoded triples."""
        x, ch, rj = _preference_batch(enc)
        lp_pol = self._logits(self.policy, x)
        lp_ref = self._logits(self.ref, x)
        if lp_ref.shape != lp_pol.shape:
            raise ValueError(
                f"policy and reference logits must have the same shape, got {lp_pol.shape} and {lp_ref.shape}"
            )
        if np.any(ch >= lp_pol.shape[1]) or np.any(rj >= lp_pol.shape[1]):
            raise ValueError(f"chosen and rejected actions must lie in [0, {lp_pol.shape[1]})")
        margin = (_logp_np(lp_pol, ch) - _logp_np(lp_ref, ch)) - (_logp_np(lp_pol, rj) - _logp_np(lp_ref, rj))
        result = -np.logaddexp(0.0, -self.beta * margin)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("DPO preference log likelihood became non-finite")
        return result  # log sigmoid(beta * margin)

    def log_density(self, xcr: Any) -> float:
        """Return the DPO log likelihood for one ``(x, chosen, rejected)`` triple."""
        x, ch, rj = xcr
        return float(self.seq_log_density((np.atleast_2d(x), [ch], [rj]))[0])

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
        restored = type(self)(
            decode_module(state["policy"]),
            decode_module(state["ref"]),
            beta=state["beta"],
            m_steps=state["m_steps"],
            lr=state["lr"],
            device=state["device"],
        )
        self.__dict__.update(restored.__dict__)

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
        if not isinstance(data, list):
            raise TypeError("DPOEncoder.seq_encode expects a list of preference triples")
        if not data:
            return (np.zeros((0, 0)), np.zeros(0, dtype=int), np.zeros(0, dtype=int))
        if any(not isinstance(row, (tuple, list)) or len(row) != 3 for row in data):
            raise ValueError("every DPO observation must be a (context, chosen, rejected) triple")
        return _preference_batch(
            (
                [row[0] for row in data],
                [row[1] for row in data],
                [row[2] for row in data],
            )
        )


class DPOAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffer weighted preference triples for the DPO M-step."""

    def __init__(self) -> None:
        self.x: list = []
        self.ch: list = []
        self.rj: list = []
        self.w: list = []  # per-pair weight (EM responsibility / streaming decay / sample weight)

    def update(self, xcr: Any, weight: float, estimate: Any) -> None:
        """Add one weighted preference triple to the accumulator."""
        x, chosen, rejected = _preference_batch(([xcr[0]], [xcr[1]], [xcr[2]]))
        validated_weight = _weights([weight], 1)
        self.x.append(x[0])
        self.ch.append(int(chosen[0]))
        self.rj.append(int(rejected[0]))
        self.w.append(float(validated_weight[0]))

    def seq_update(self, enc: Any, weights: Any, estimate: Any) -> None:
        """Add an encoded batch of preference triples and optional weights."""
        x, ch, rj = _preference_batch(enc)
        ws = np.ones(len(x)) if weights is None else _weights(weights, len(x))
        for i in range(len(x)):
            self.x.append(x[i])
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
        if not isinstance(other, tuple) or len(other) != 4:
            raise ValueError("DPO sufficient statistics must be an (x, chosen, rejected, weights) tuple")
        if len(other[0]) == 0:
            if any(len(field) != 0 for field in other[1:]):
                raise ValueError("empty DPO sufficient-statistic fields must all be empty")
            return self
        xo, co, ro = _preference_batch(other[:3])
        wo = _weights(other[3], len(xo))
        self.x.extend(xo)
        self.ch.extend(co.tolist())
        self.rj.extend(ro.tolist())
        self.w.extend(wo.tolist())
        return self

    def value(self) -> tuple:
        """Return buffered contexts, chosen actions, rejected actions, and weights."""
        return (list(self.x), list(self.ch), list(self.rj), np.asarray(self.w, dtype=float))

    def from_value(self, v: tuple) -> DPOAccumulator:
        """Restore accumulator buffers from a value tuple."""
        self.x, self.ch, self.rj, self.w = [], [], [], []
        self.combine(v)
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
        torch = _torch()
        if policy is not None or ref is not None:
            if policy is None or ref is None:
                raise ValueError("policy and ref must either both be modules or both be None")
            _validate_independent_modules(policy, ref, torch)
        self.policy = policy
        self.ref = ref
        self.beta = _positive_finite(beta, "beta")
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        try:
            self.device = str(torch.device(device))
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid torch device {device!r}") from exc

    def accumulator_factory(self) -> DPOAccumulatorFactory:
        """Return an accumulator factory for weighted preference triples."""
        return DPOAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: tuple) -> DPOModel:
        """Run the weighted DPO M-step and return the updated policy leaf."""
        torch = _torch()
        if self.policy is None or self.ref is None:
            raise ValueError("DPO estimation requires independent policy and reference modules")
        if not isinstance(suff_stat, tuple) or len(suff_stat) != 4:
            raise ValueError("DPO sufficient statistics must be an (x, chosen, rejected, weights) tuple")
        xs, chs, rjs = _preference_batch(suff_stat[:3])
        if len(xs) == 0:
            raise ValueError("DPO estimation requires at least one preference")
        ws = _weights(suff_stat[3], len(xs))
        dev = self.device
        self.policy.to(dev)
        self.ref.to(dev)
        for p in self.ref.parameters():
            p.requires_grad_(False)  # frozen reference
        dtype = next(
            (parameter.dtype for parameter in self.policy.parameters() if parameter.dtype.is_floating_point),
            torch.float32,
        )
        xt = torch.as_tensor(xs, dtype=dtype, device=dev)
        ct = torch.as_tensor(chs, dtype=torch.long, device=dev)
        rt = torch.as_tensor(rjs, dtype=torch.long, device=dev)
        wt = torch.as_tensor(ws, dtype=dtype, device=dev)
        wsum = wt.sum()
        ar = torch.arange(len(ct), device=dev)
        trainable = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("DPO policy must contain at least one trainable parameter")
        opt = torch.optim.Adam(trainable, lr=self.lr)
        with _module_mode(self.ref, train=False), torch.no_grad():  # reference log-probs are constant -- compute once
            lr_all = torch.log_softmax(self.ref(xt), dim=1)
            _validate_torch_logits(lr_all, len(xs), "reference", torch)
            if torch.any(ct >= lr_all.shape[1]) or torch.any(rt >= lr_all.shape[1]):
                raise ValueError(f"chosen and rejected actions must lie in [0, {lr_all.shape[1]})")
            lr_ch, lr_rj = lr_all[ar, ct], lr_all[ar, rt]
        with _module_mode(self.policy, train=True):
            for _ in range(self.m_steps):
                opt.zero_grad()
                lp = torch.log_softmax(self.policy(xt), dim=1)
                _validate_torch_logits(lp, len(xs), "policy", torch)
                if lp.shape != lr_all.shape:
                    raise ValueError(
                        f"policy and reference logits must have the same shape, got "
                        f"{tuple(lp.shape)} and {tuple(lr_all.shape)}"
                    )
                margin = (lp[ar, ct] - lr_ch) - (lp[ar, rt] - lr_rj)
                loss = -(wt * torch.nn.functional.logsigmoid(self.beta * margin)).sum() / wsum  # weighted DPO loss
                if not bool(torch.isfinite(loss).detach().cpu().item()):
                    raise RuntimeError("DPO objective became non-finite")
                loss.backward()
                opt.step()
                if any(
                    not bool(torch.all(torch.isfinite(parameter)).detach().cpu().item())
                    for parameter in trainable
                ):
                    raise RuntimeError("DPO optimization produced non-finite policy parameters")
        return DPOModel(self.policy, self.ref, self.beta, self.m_steps, self.lr, self.device)


def _validate_independent_modules(policy: Any, ref: Any, torch: Any) -> None:
    if not isinstance(policy, torch.nn.Module) or not isinstance(ref, torch.nn.Module):
        raise TypeError("policy and ref must be torch.nn.Module instances")
    if policy is ref:
        raise ValueError("policy and ref must be independently owned modules, not the same object")
    policy_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in policy.parameters()
        if parameter.numel()
    }
    ref_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in ref.parameters()
        if parameter.numel()
    }
    if policy_storage & ref_storage:
        raise ValueError("policy and ref must not share parameter storage")


def _contexts(value: Any) -> np.ndarray:
    try:
        contexts = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("DPO contexts must form a numeric two-dimensional matrix") from exc
    if contexts.ndim == 1:
        contexts = np.atleast_2d(contexts)
    if contexts.ndim != 2 or contexts.shape[0] == 0 or contexts.shape[1] == 0:
        raise ValueError("DPO contexts must have non-empty shape (n, features)")
    if not np.all(np.isfinite(contexts)):
        raise ValueError("DPO contexts must contain only finite values")
    return contexts


def _preference_batch(enc: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(enc, (tuple, list)) or len(enc) != 3:
        raise ValueError("DPO data must be an (x, chosen, rejected) triple")
    contexts = _contexts(enc[0])
    chosen = _exact_actions(enc[1], "chosen", len(contexts))
    rejected = _exact_actions(enc[2], "rejected", len(contexts))
    if np.any(chosen == rejected):
        raise ValueError("chosen and rejected actions must differ for every preference")
    return contexts, chosen, rejected


def _exact_actions(value: Any, name: str, n_rows: int) -> np.ndarray:
    actions = np.asarray(value)
    if actions.ndim != 1 or len(actions) != n_rows:
        raise ValueError(f"{name} must be a one-dimensional vector with {n_rows} rows")
    if actions.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} actions must be numeric integers")
    if actions.dtype.kind == "f" and (
        not np.all(np.isfinite(actions)) or not np.all(actions == np.round(actions))
    ):
        raise ValueError(f"{name} actions must be finite integer values")
    if np.any(actions < 0):
        raise ValueError(f"{name} actions must be non-negative")
    if np.any(actions > np.iinfo(np.intp).max):
        raise ValueError(f"{name} actions exceed the supported integer index range")
    return actions.astype(int, copy=False)


def _weights(value: Any, n_rows: int) -> np.ndarray:
    weights = np.asarray(value, dtype=float)
    if weights.ndim != 1 or len(weights) != n_rows:
        raise ValueError(f"DPO weights must be a one-dimensional vector with {n_rows} rows")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("DPO weights must contain only finite, strictly positive values")
    return weights


def _validate_torch_logits(logits: Any, n_rows: int, name: str, torch: Any) -> None:
    if logits.ndim != 2 or logits.shape[0] != n_rows or logits.shape[1] < 2:
        raise ValueError(f"{name} must return shape (n, actions) with at least two actions")
    if not bool(torch.all(torch.isfinite(logits)).detach().cpu().item()):
        raise ValueError(f"{name} returned non-finite logits")


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


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
