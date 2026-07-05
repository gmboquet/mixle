"""``NeuralConditionalDensity`` -- the adapter that turns ANY torch *conditional* density into a mixle leaf.

This is the conditional sibling of :class:`~mixle.models.neural_density.NeuralDensity`. Where that one wraps a
module exposing ``log_density(x) -> (n,)`` (an unconditional ``p(x)``), this wraps a module exposing
``log_density(x, y) -> (n,)`` (and ``sample_given(x) -> (n, d)``) and gives you a full five-piece mixle
``Distribution`` over the pair ``(x, y)`` -- so a *flexible conditional density* drops into a mixture of experts,
a composite field, or an HMM emission and is fit **jointly with classical families** by the same
responsibility-weighted-NLL EM M-step (warm-started across iterations, i.e. generalized EM).

Why it matters: :class:`~mixle.models.neural_leaf.NeuralGaussian` fixes the conditional law to a single Gaussian,
``p(y | x) = N(y; f(x), sigma^2 I)`` -- one mean per ``x``, unimodal and homoscedastic. Many real conditionals
are neither: an inverse problem has *several* valid ``y`` for one ``x``; measurement noise grows with ``x``.
:func:`build_mdn` is the ready instance -- a **mixture density network**, ``p(y | x) = sum_k pi_k(x) N(y; mu_k(x),
sigma_k(x)^2)`` -- whose entire mixture (weights, means, variances) is a function of ``x``, so it is multimodal
and heteroscedastic. Any other conditional density (a conditional flow, an autoregressive head) plugs in the same
way: give it ``log_density(x, y)`` and ``sample_given(x)``.
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


# --- module-class hoisting (see mixle.models.neural_density for the rationale): the wrapped nn.Module classes are
# reachable at MODULE level so a leaf -- and any mixture holding one -- pickles for distributed EM. Built on first
# use, resolved by name via __getattr__ so unpickling in a fresh interpreter works. ---

_MODULE_CLASS_CACHE: dict[str, Any] = {}
_MODULE_CLASS_FACTORIES: dict[str, Any] = {}


def _register_module_class(name: str, factory: Any) -> None:
    _MODULE_CLASS_FACTORIES[name] = factory


def _module_class(name: str) -> Any:
    cls = _MODULE_CLASS_CACHE.get(name)
    if cls is not None:
        return cls
    import torch
    import torch.nn as nn

    cls = _MODULE_CLASS_FACTORIES[name](torch, nn)
    cls.__module__ = __name__
    cls.__qualname__ = name
    cls.__name__ = name
    _MODULE_CLASS_CACHE[name] = cls
    return cls


def __getattr__(name: str) -> Any:  # PEP 562: lets ``pickle`` resolve the hoisted module classes by name
    if name in _MODULE_CLASS_FACTORIES:
        return _module_class(name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


class NeuralConditionalDensity(SequenceEncodableProbabilityDistribution):
    """Wrap a torch conditional-density ``module`` (``module.log_density(x, y) -> (n,)``) as a mixle leaf.

    Observations are pairs ``(x, y)``. The module must also expose ``sample_given(x) -> (n, d)`` to draw ``y``.
    """

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self, module: Any, *, m_steps: int = 60, lr: float = 5e-3, device: str = "cpu", name: str | None = None
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device
        self.name = name

    def __str__(self) -> str:
        return f"NeuralConditionalDensity({type(self.module).__name__})"

    def log_density(self, xy: Any) -> float:
        x, y = xy
        return float(self.seq_log_density(([np.atleast_1d(x)], [np.atleast_1d(y)]))[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        torch = _torch()
        xs, ys = enc
        xx = check_finite(np.atleast_2d(np.asarray(xs, dtype=float)), "NeuralConditionalDensity.seq_log_density (x)")
        yy = check_finite(np.atleast_2d(np.asarray(ys, dtype=float)), "NeuralConditionalDensity.seq_log_density (y)")
        self.module.to(self.device).eval()
        xt = torch.as_tensor(xx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(yy, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.module.log_density(xt, yt).cpu().numpy().reshape(-1)

    def sampler(self, seed: int | None = None) -> NeuralConditionalDensitySampler:
        return NeuralConditionalDensitySampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralConditionalDensityEstimator:
        return NeuralConditionalDensityEstimator(
            self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name
        )

    def dist_to_encoder(self) -> NeuralConditionalDensityEncoder:
        return NeuralConditionalDensityEncoder()

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
            "m_steps": self.m_steps,
            "lr": self.lr,
            "device": self.device,
            "name": self.name,
            "module": encode_module(self.module),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NeuralConditionalDensity:
        return cls(
            decode_module(payload["module"]),
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            device=payload["device"],
            name=payload["name"],
        )


class NeuralConditionalDensitySampler(DistributionSampler):
    def __init__(self, dist: NeuralConditionalDensity, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        raise NotImplementedError("NeuralConditionalDensity is conditional p(y|x); use sampler().sample_given(x).")

    def sample_given(self, x: Any) -> np.ndarray:
        torch = _torch()
        self.dist.module.to(self.dist.device).eval()
        torch.manual_seed(int(self.rng.randint(0, 2**31 - 1)))
        xt = torch.as_tensor(np.atleast_2d(np.asarray(x, dtype=float)), dtype=torch.float32, device=self.dist.device)
        with torch.no_grad():
            return self.dist.module.sample_given(xt).cpu().numpy()[0]


class NeuralConditionalDensityEncoder(DataSequenceEncoder):
    def __str__(self) -> str:
        return "NeuralConditionalDensityEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralConditionalDensityEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([np.atleast_1d(np.asarray(xy[0], dtype=float)) for xy in data])
        y = np.array([np.atleast_1d(np.asarray(xy[1], dtype=float)) for xy in data])
        return (x, y)


class NeuralConditionalDensityAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffers responsibility-weighted ``(x, y)`` pairs for the M-step (the weights are the E-step soft counts)."""

    def __init__(self) -> None:
        self.x: list = []
        self.y: list = []
        self.w: list = []

    # Contiguous batch arrays concatenated once at value() (shape-preserving) rather than one ndarray per row.
    def update(self, xy: Any, weight: float, estimate: Any) -> None:
        self.x.append(np.atleast_1d(np.asarray(xy[0], dtype=float))[None, ...])
        self.y.append(np.atleast_1d(np.asarray(xy[1], dtype=float))[None, ...])
        self.w.append(np.asarray([float(weight)], dtype=float))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        x, y = enc
        xb = np.asarray(x, dtype=float)
        yb = np.asarray(y, dtype=float)
        self.x.append(xb.reshape(xb.shape[0], 1) if xb.ndim == 1 else xb)
        self.y.append(yb.reshape(yb.shape[0], 1) if yb.ndim == 1 else yb)
        self.w.append(np.asarray(weights, dtype=float).ravel())

    def initialize(self, xy: Any, weight: float, rng: Any) -> None:
        self.update(xy, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralConditionalDensityAccumulator:
        xo, yo, wo = other
        if len(xo):
            self.x.append(np.asarray(xo, dtype=float))
            self.y.append(np.asarray(yo, dtype=float))
            self.w.append(np.asarray(wo, dtype=float).ravel())
        return self

    def value(self) -> tuple:
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        y = np.concatenate(self.y, axis=0) if self.y else np.zeros((0, 0))
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, y, w)

    def from_value(self, value: tuple) -> NeuralConditionalDensityAccumulator:
        x, y, w = value
        self.x = [np.asarray(x, dtype=float)] if len(x) else []
        self.y = [np.asarray(y, dtype=float)] if len(y) else []
        self.w = [np.asarray(w, dtype=float).ravel()] if len(w) else []
        return self

    def acc_to_encoder(self) -> NeuralConditionalDensityEncoder:
        return NeuralConditionalDensityEncoder()


class NeuralConditionalDensityAccumulatorFactory(StatisticAccumulatorFactory):
    def make(self) -> NeuralConditionalDensityAccumulator:
        return NeuralConditionalDensityAccumulator()


class NeuralConditionalDensityEstimator(ParameterEstimator):
    """M-step: responsibility-weighted MLE ``max sum_i w_i log p(y_i | x_i)`` by gradient ascent (warm-started)."""

    def __init__(
        self, module: Any, *, m_steps: int = 60, lr: float = 5e-3, device: str = "cpu", name: str | None = None
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device
        self.name = name

    def accumulator_factory(self) -> NeuralConditionalDensityAccumulatorFactory:
        return NeuralConditionalDensityAccumulatorFactory()

    def _make(self) -> NeuralConditionalDensity:
        return NeuralConditionalDensity(
            self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name
        )

    def estimate(self, nobs: float | None, suff_stat: tuple) -> NeuralConditionalDensity:
        torch = _torch()
        xs, ys, ws = suff_stat
        if len(xs) == 0:
            return self._make()
        x = torch.as_tensor(np.asarray(xs, dtype=float), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.asarray(ys, dtype=float), dtype=torch.float32, device=self.device)
        w = torch.as_tensor(np.asarray(ws, dtype=float), dtype=torch.float32, device=self.device)
        w = w / w.sum().clamp(min=1e-8)
        self.module.to(self.device).train()
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        for _ in range(self.m_steps):
            opt.zero_grad()
            loss = -(w * self.module.log_density(x, y)).sum()  # weighted negative conditional log-likelihood
            loss.backward()
            opt.step()
        return self._make()


# --- a ready conditional-density module to wrap: a mixture density network (Bishop 1994) --------------------


def build_mdn(x_dim: int, y_dim: int, *, k: int = 5, hidden: int = 32, layers: int = 2) -> Any:
    """A mixture density network: ``p(y | x) = sum_k pi_k(x) N(y; mu_k(x), diag sigma_k(x)^2)`` -- ready to wrap.

    A shared MLP body maps ``x`` to three heads -- mixing logits, component means, and (log) component scales --
    so the *entire* conditional law is a function of ``x``: multimodal (several ``mu_k``) and heteroscedastic
    (input-dependent ``sigma_k``). Exposes ``log_density(x, y)`` (a log-sum-exp over components) and
    ``sample_given(x)`` (pick a component by ``pi``, then a Gaussian), the contract a
    :class:`NeuralConditionalDensity` adapts.
    """
    return _module_class("MixtureDensityNetwork")(x_dim, y_dim, k, hidden, layers)


def _build_mdn_class(torch: Any, nn: Any) -> Any:
    class MixtureDensityNetwork(nn.Module):
        def __init__(self, x_dim: int, y_dim: int, k: int = 5, hidden: int = 32, layers: int = 2) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.k = int(k)
            self.y_dim = int(y_dim)
            self.hidden = int(hidden)
            self.n_layers = int(layers)
            body: list[nn.Module] = []
            d = self.x_dim
            for _ in range(self.n_layers):
                body += [nn.Linear(d, self.hidden), nn.Tanh()]
                d = self.hidden
            self.body = nn.Sequential(*body)
            self.head_logits = nn.Linear(d, self.k)
            self.head_mu = nn.Linear(d, self.k * self.y_dim)
            self.head_log_sigma = nn.Linear(d, self.k * self.y_dim)

        def _params(self, x: Any) -> tuple[Any, Any, Any]:
            h = self.body(x)
            log_pi = torch.log_softmax(self.head_logits(h), dim=-1)  # (n, k)
            mu = self.head_mu(h).view(-1, self.k, self.y_dim)  # (n, k, d)
            log_sigma = self.head_log_sigma(h).view(-1, self.k, self.y_dim).clamp(-7.0, 7.0)
            return log_pi, mu, log_sigma

        def log_density(self, x: Any, y: Any) -> Any:
            log_pi, mu, log_sigma = self._params(x)
            yb = y.unsqueeze(1)  # (n, 1, d)
            z = (yb - mu) * torch.exp(-log_sigma)  # (n, k, d)
            log_n = -0.5 * (z**2).sum(-1) - log_sigma.sum(-1) - 0.5 * self.y_dim * float(np.log(2.0 * np.pi))
            return torch.logsumexp(log_pi + log_n, dim=1)  # (n,)

        def sample_given(self, x: Any) -> Any:
            log_pi, mu, log_sigma = self._params(x)
            comp = torch.multinomial(torch.exp(log_pi), 1).squeeze(-1)  # (n,)
            idx = comp.view(-1, 1, 1).expand(-1, 1, self.y_dim)
            mu_c = mu.gather(1, idx).squeeze(1)  # (n, d)
            sig_c = torch.exp(log_sigma.gather(1, idx).squeeze(1))
            return mu_c + sig_c * torch.randn_like(mu_c)

    return MixtureDensityNetwork


_register_module_class("MixtureDensityNetwork", _build_mdn_class)


# --- the exact counterpart: a conditional normalizing flow -- exact p(y|x) with within-y structure -------------


def build_conditional_flow(x_dim: int, y_dim: int, *, hidden: int = 32, layers: int = 4) -> Any:
    """A conditional coupling flow: an **exact** ``p(y | x)`` whose transform of ``y`` is conditioned on ``x``.

    The exact-density counterpart to :func:`build_mdn`. Each affine-coupling layer's shift/scale networks take
    both the passed-through ``y`` coordinates *and* ``x``, so the whole invertible ``y``-transform bends with the
    input -- capturing *within-``y``* dependence (e.g. ``y2`` a nonlinear function of ``y1``) that a single-Gaussian
    :class:`~mixle.models.neural_leaf.NeuralGaussian` (isotropic mean-only) cannot, while keeping an exact log-density
    (so it composes honestly, unlike a bound). Needs ``y_dim >= 2`` for the coupling to be non-trivial. Exposes
    ``log_density(x, y)`` and ``sample_given(x)`` -- the contract a :class:`NeuralConditionalDensity` adapts.
    """
    return _module_class("ConditionalFlow")(x_dim, y_dim, hidden, layers)


def _build_conditional_flow_class(torch: Any, nn: Any) -> Any:
    class ConditionalFlow(nn.Module):
        def __init__(self, x_dim: int, y_dim: int, hidden: int = 32, layers: int = 4) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.y_dim = int(y_dim)
            self.hidden = int(hidden)
            self.layers = int(layers)
            masks = []
            for k in range(self.layers):
                m = torch.zeros(self.y_dim)
                m[k % self.y_dim :: 2] = 1.0  # alternating coordinate masks
                masks.append(m)
            self.register_buffer("masks", torch.stack(masks))

            def net() -> nn.Module:
                return nn.Sequential(
                    nn.Linear(self.y_dim + self.x_dim, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.y_dim)
                )

            self.s = nn.ModuleList([net() for _ in range(self.layers)])
            self.t = nn.ModuleList([net() for _ in range(self.layers)])

        def _normalize(self, x: Any, y: Any) -> tuple[Any, Any]:
            z = y
            logdet = torch.zeros(y.shape[0], device=y.device)
            for m, s_net, t_net in zip(self.masks, self.s, self.t):
                zm = z * m
                inp = torch.cat([zm, x], dim=1)  # the coupling is conditioned on x
                s = s_net(inp) * (1.0 - m)
                t = t_net(inp) * (1.0 - m)
                z = zm + (1.0 - m) * ((z - t) * torch.exp(-s))
                logdet = logdet - s.sum(1)
            return z, logdet

        def log_density(self, x: Any, y: Any) -> Any:
            z, logdet = self._normalize(x, y)
            base = -0.5 * (z**2).sum(1) - 0.5 * self.y_dim * float(np.log(2.0 * np.pi))
            return base + logdet

        def sample_given(self, x: Any) -> Any:
            y = torch.randn(x.shape[0], self.y_dim, device=x.device)
            for m, s_net, t_net in zip(reversed(self.masks), reversed(list(self.s)), reversed(list(self.t))):
                ym = y * m
                inp = torch.cat([ym, x], dim=1)
                s = s_net(inp) * (1.0 - m)
                t = t_net(inp) * (1.0 - m)
                y = ym + (1.0 - m) * (y * torch.exp(s) + t)  # inverse of _normalize
            return y

    return ConditionalFlow


_register_module_class("ConditionalFlow", _build_conditional_flow_class)


# --- the discrete conditional: an autoregressive categorical conditioned on x -- exact p(y|x) over discrete y ----


def build_conditional_autoregressive_categorical(x_dim: int, y_dim: int, n_categories: int, *, hidden: int = 64) -> Any:
    """An autoregressive categorical conditioned on ``x``: exact ``p(y | x)`` over **discrete** ``y in {0..C-1}^y_dim``.

    The conditional sibling of :func:`~mixle.models.neural_density.build_autoregressive_categorical` and the discrete
    counterpart to :func:`build_conditional_flow`. It factorizes ``p(y | x) = prod_i p(y_i | y_{<i}, x)`` with a
    MADE-masked net over ``y`` into which ``x`` is injected *unmasked* (degree 0, so every coordinate may depend on
    ``x``). Each per-coordinate softmax is exactly a conditional, so the density is **exactly normalized** and
    composes honestly. Exposes ``log_density(x, y)`` and ``sample_given(x)`` -- the contract a
    :class:`NeuralConditionalDensity` adapts.
    """
    return _module_class("ConditionalAutoregressiveCategorical")(x_dim, y_dim, n_categories, hidden)


def _build_cond_masked_linear_class(torch: Any, nn: Any) -> Any:
    class CondMaskedLinear(nn.Linear):
        def set_mask(self, mask: Any) -> None:
            self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))

        def forward(self, x: Any) -> Any:
            return nn.functional.linear(x, self.mask * self.weight, self.bias)

    return CondMaskedLinear


_register_module_class("CondMaskedLinear", _build_cond_masked_linear_class)


def _build_conditional_autoregressive_categorical_class(torch: Any, nn: Any) -> Any:
    MaskedLinear = _module_class("CondMaskedLinear")

    class ConditionalAutoregressiveCategorical(nn.Module):
        def __init__(self, x_dim: int, y_dim: int, n_categories: int, hidden: int = 64) -> None:
            super().__init__()
            self.X = int(x_dim)
            self.D = int(y_dim)
            self.C = int(n_categories)
            self.hidden = int(hidden)
            X, D, C, hid = self.X, self.D, self.C, self.hidden
            m_in = np.arange(1, D + 1)
            m_h = 1 + (np.arange(hid) % max(D - 1, 1))
            self.l1_y = MaskedLinear(D, hid)  # autoregressive path over y
            self.l1_x = nn.Linear(X, hid)  # x is conditioning context, available to ALL coordinates (unmasked)
            self.l2 = MaskedLinear(hid, hid)
            self.lout = MaskedLinear(hid, D * C)
            self.l1_y.set_mask((m_h[:, None] >= m_in[None, :]).astype(float))
            self.l2.set_mask((m_h[:, None] >= m_h[None, :]).astype(float))
            m_out = np.repeat(m_in, C)
            self.lout.set_mask((m_out[:, None] > m_h[None, :]).astype(float))  # strict: logits_i see y_{<i} (and x)
            self.act = nn.Tanh()

        def _logits(self, x: Any, y: Any) -> Any:
            h = self.act(self.l1_y(y) + self.l1_x(x))
            h = self.act(self.l2(h))
            return self.lout(h).view(-1, self.D, self.C)  # (n, D, C)

        def log_density(self, x: Any, y: Any) -> Any:
            log_p = torch.log_softmax(self._logits(x, y), dim=-1)  # (n, D, C)
            idx = y.long().clamp(0, self.C - 1).unsqueeze(-1)
            return log_p.gather(-1, idx).squeeze(-1).sum(1)  # sum_i log p(y_i | y_{<i}, x)

        def sample_given(self, x: Any) -> Any:
            y = torch.zeros(x.shape[0], self.D, device=x.device)
            for d in range(self.D):  # coordinate d depends on x (always) and already-filled y_{<d}
                probs = torch.softmax(self._logits(x, y)[:, d, :], dim=-1)
                y[:, d] = torch.multinomial(probs, 1).squeeze(-1).float()
            return y

    return ConditionalAutoregressiveCategorical


_register_module_class("ConditionalAutoregressiveCategorical", _build_conditional_autoregressive_categorical_class)


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover
        return
    register_serializable_class(NeuralConditionalDensity)


_register_serializable()
