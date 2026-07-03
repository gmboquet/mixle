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


class NeuralConditionalDensity(SequenceEncodableProbabilityDistribution):
    """Wrap a torch conditional-density ``module`` (``module.log_density(x, y) -> (n,)``) as a mixle leaf.

    Observations are pairs ``(x, y)``. The module must also expose ``sample_given(x) -> (n, d)`` to draw ``y``.
    """

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
        self.module.to(self.device).eval()
        xt = torch.as_tensor(np.atleast_2d(np.asarray(xs, dtype=float)), dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(np.atleast_2d(np.asarray(ys, dtype=float)), dtype=torch.float32, device=self.device)
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

    def update(self, xy: Any, weight: float, estimate: Any) -> None:
        self.x.append(np.atleast_1d(np.asarray(xy[0], dtype=float)))
        self.y.append(np.atleast_1d(np.asarray(xy[1], dtype=float)))
        self.w.append(float(weight))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        x, y = enc
        for i in range(len(x)):
            self.x.append(np.atleast_1d(x[i]))
            self.y.append(np.atleast_1d(y[i]))
            self.w.append(float(weights[i]))

    def initialize(self, xy: Any, weight: float, rng: Any) -> None:
        self.update(xy, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralConditionalDensityAccumulator:
        xo, yo, wo = other
        self.x.extend(xo)
        self.y.extend(yo)
        self.w.extend(wo)
        return self

    def value(self) -> tuple:
        return (list(self.x), list(self.y), list(self.w))

    def from_value(self, value: tuple) -> NeuralConditionalDensityAccumulator:
        self.x, self.y, self.w = list(value[0]), list(value[1]), list(value[2])
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
        if not xs:
            return self._make()
        x = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.stack(ys), dtype=torch.float32, device=self.device)
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
    import torch
    import torch.nn as nn

    class MixtureDensityNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = int(k)
            self.y_dim = int(y_dim)
            body: list[nn.Module] = []
            d = int(x_dim)
            for _ in range(int(layers)):
                body += [nn.Linear(d, hidden), nn.Tanh()]
                d = hidden
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

    return MixtureDensityNetwork()


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
    import torch
    import torch.nn as nn

    class ConditionalFlow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.y_dim = int(y_dim)
            masks = []
            for k in range(int(layers)):
                m = torch.zeros(self.y_dim)
                m[k % self.y_dim :: 2] = 1.0  # alternating coordinate masks
                masks.append(m)
            self.register_buffer("masks", torch.stack(masks))

            def net() -> nn.Module:
                return nn.Sequential(
                    nn.Linear(self.y_dim + self.x_dim, hidden), nn.Tanh(), nn.Linear(hidden, self.y_dim)
                )

            self.s = nn.ModuleList([net() for _ in range(int(layers))])
            self.t = nn.ModuleList([net() for _ in range(int(layers))])

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

    return ConditionalFlow()
