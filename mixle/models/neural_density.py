"""``NeuralDensity`` -- the adapter that turns ANY torch density module into a composable mixle distribution.

The point is not a specific architecture; it is the *wrapper*. ``NeuralGaussian`` already adapts a conditional net
(``p(y | x)``); this is its unconditional sibling: give it any torch module that exposes ``log_density(x) -> (n,)``
(and, to draw samples, ``sample(n) -> (n, d)``) and you get a full five-piece mixle ``Distribution`` -- so a
*flexible neural density* drops into a ``MixtureDistribution`` component, an HMM emission, or a
``CompositeDistribution`` field, and is fit **jointly with classical families** by EM. Its M-step is a
responsibility-weighted maximum-likelihood gradient ascent on the module, warm-started across EM iterations.

That is the thing no NN library offers: "a mixture of a normalizing flow and a Gamma", "an HMM whose emissions
are flows". Ready instances ship: :func:`build_coupling_flow` (a RealNVP-style flow, *exact*), :func:`build_maf` (a masked
autoregressive flow, *exact*, richer autoregressive dependence), :func:`build_vae` (a variational autoencoder, a
*latent-variable* density whose ``log_density`` is the ELBO lower bound) and :func:`build_autoregressive_categorical`
(an exact autoregressive density over **discrete** vectors) -- structurally different families, continuous and
discrete, behind one adapter. Any other density (a normalized energy model, ...) plugs in the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.models._neural_serial import check_finite, decode_module, encode_module
from mixle.models.grad_leaf import DataBufferAccumulatorFactory, GradEstimator, GradLeaf
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _torch() -> Any:
    import torch

    return torch


class NeuralDensity(GradLeaf):
    """Wrap a torch density ``module`` (``module.log_density(x) -> (n,)``) as a composable mixle distribution.

    A thin named subclass of :class:`~mixle.models.grad_leaf.GradLeaf` -- the generic bridge owns the
    manufactured contract (buffer accumulator, array encoder, gradient M-step, sampler); this class owns
    only its name, its JSON payload, and its ready-module builders below. ``loss``/``optimizer`` hooks
    pass through (see the grad_leaf module docstring for the control story).
    """

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self, module: Any, *, m_steps: int = 60, lr: float = 5e-3, device: str = "cpu", name: str | None = None
    ) -> None:
        super().__init__(module, m_steps=m_steps, lr=lr, device=device, name=name)

    def __str__(self) -> str:
        return f"NeuralDensity({type(self.module).__name__})"

    def log_density(self, x: Any) -> float:
        """Return ``log p(x)`` for one observation under the wrapped density module."""
        return float(self.seq_log_density(np.atleast_2d(np.asarray(x, dtype=float)))[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return per-row log densities for encoded observations."""
        torch = _torch()
        xx = check_finite(np.atleast_2d(np.asarray(x, dtype=float)), "NeuralDensity.seq_log_density")
        self.module.to(self.device).eval()
        xt = torch.as_tensor(xx, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.module.log_density(xt).cpu().numpy().reshape(-1)

    def sampler(self, seed: int | None = None) -> NeuralDensitySampler:
        """Return a sampler delegating to the wrapped module's ``sample`` method."""
        return NeuralDensitySampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralDensityEstimator:
        """Return the generalized-EM estimator for weighted neural-density training."""
        return NeuralDensityEstimator(self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name)

    def dist_to_encoder(self) -> NeuralDensityEncoder:
        """Return the encoder for vectorized neural-density scoring and fitting."""
        return NeuralDensityEncoder()

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
            "device": self.device,
            "name": self.name,
            "module": encode_module(self.module),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NeuralDensity:
        """Rebuild a :class:`NeuralDensity` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["module"]),
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            device=payload["device"],
            name=payload["name"],
        )


class NeuralDensitySampler(DistributionSampler):
    """Sampler for wrapped neural density modules exposing ``sample(n)``."""

    def __init__(self, dist: NeuralDensity, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw observations from the wrapped module's sampler."""
        torch = _torch()
        n = int(size or 1)
        self.dist.module.to(self.dist.device).eval()
        torch.manual_seed(int(self.rng.randint(0, 2**31 - 1)))
        with torch.no_grad():
            out = self.dist.module.sample(n).cpu().numpy()
        return out if (size is not None) else out[0]


class NeuralDensityEncoder(DataSequenceEncoder):
    """Encode observations for vectorized neural-density scoring and fitting."""

    def __str__(self) -> str:
        return "NeuralDensityEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralDensityEncoder)

    def seq_encode(self, data: list) -> np.ndarray:
        """Convert observations to a two-dimensional float array."""
        return np.array([np.atleast_1d(np.asarray(x, dtype=float)) for x in data])


class NeuralDensityAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffers the (responsibility-weighted) data for the M-step -- the weights are the E-step's soft counts."""

    def __init__(self) -> None:
        self.x: list = []
        self.w: list = []

    # Contiguous batch arrays concatenated once at value() (shape-preserving) rather than one ndarray per row.
    def update(self, x: Any, weight: float, estimate: Any) -> None:
        """Add one weighted observation to the accumulator."""
        self.x.append(np.atleast_1d(np.asarray(x, dtype=float))[None, ...])
        self.w.append(np.asarray([float(weight)], dtype=float))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Add an encoded batch and responsibility weights to the accumulator."""
        xb = np.asarray(enc, dtype=float)
        self.x.append(xb.reshape(xb.shape[0], 1) if xb.ndim == 1 else xb)
        self.w.append(np.asarray(weights, dtype=float).ravel())

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        """Initialize from one observation using the ordinary update path."""
        self.update(x, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        """Initialize from an encoded batch using the ordinary batch update path."""
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralDensityAccumulator:
        """Merge the value tuple from another neural-density accumulator."""
        xs, ws = other
        if len(xs):
            self.x.append(np.asarray(xs, dtype=float))
            self.w.append(np.asarray(ws, dtype=float).ravel())
        return self

    def value(self) -> tuple:
        """Return contiguous ``(x, weights)`` arrays for the M-step."""
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, w)

    def from_value(self, v: tuple) -> NeuralDensityAccumulator:
        """Restore accumulator buffers from a value tuple."""
        x, w = v
        self.x = [np.asarray(x, dtype=float)] if len(x) else []
        self.w = [np.asarray(w, dtype=float).ravel()] if len(w) else []
        return self

    def acc_to_encoder(self) -> NeuralDensityEncoder:
        """Return the encoder expected by this accumulator."""
        return NeuralDensityEncoder()


class NeuralDensityAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for neural-density accumulators."""

    def make(self) -> NeuralDensityAccumulator:
        """Create a fresh accumulator."""
        return NeuralDensityAccumulator()


class NeuralDensityEstimator(GradEstimator):
    """M-step: responsibility-weighted MLE -- ``max sum_i w_i log p(x_i)`` by gradient ascent on the module (warm)."""

    def __init__(
        self, module: Any, *, m_steps: int = 60, lr: float = 5e-3, device: str = "cpu", name: str | None = None
    ) -> None:
        super().__init__(module, m_steps=m_steps, lr=lr, device=device, name=name)

    def accumulator_factory(self) -> DataBufferAccumulatorFactory:
        """Return an accumulator factory for weighted neural-density batches."""
        return DataBufferAccumulatorFactory(NeuralDensityEncoder())

    def _leaf(self) -> NeuralDensity:
        return NeuralDensity(self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name)


# --- ready density modules to wrap ---------------------------------------------------------------------------
#
# The wrapped nn.Module classes must be reachable at MODULE level so a leaf (and any mixture holding one) pickles
# -- otherwise distributed EM and ``pickle`` fail on a function-local class. torch is optional at import, so each
# class is built on first use by ``_module_class`` and cached as a module global with a clean ``__qualname__``;
# pickle then reconstructs it by name. The ``build_*`` helpers construct instances (hparams stored on the
# instance so ``to_dict`` can rebuild it).

_MODULE_CLASS_CACHE: dict[str, Any] = {}
_MODULE_CLASS_FACTORIES: dict[str, Any] = {}


def _register_module_class(name: str, factory: Any) -> None:
    _MODULE_CLASS_FACTORIES[name] = factory


def _module_class(name: str) -> Any:
    """Return (building once, then caching) a module-level nn.Module class named ``name``.

    The registered factory receives ``(torch, nn)`` and returns the class. The class's ``__module__``/``__qualname__``
    are fixed to this module so ``pickle`` can look it up by name -- the whole point of hoisting it out of ``build_*``.
    Module ``__getattr__`` (below) builds on demand, so unpickling in a fresh interpreter resolves the class too.
    """
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


def _build_coupling_flow_class(torch: Any, nn: Any) -> Any:
    class CouplingFlow(nn.Module):
        def __init__(self, dim: int, hidden: int = 32, layers: int = 4) -> None:
            super().__init__()
            self.dim = int(dim)
            self.hidden = int(hidden)
            self.layers = int(layers)
            masks = []
            for k in range(self.layers):
                m = torch.zeros(self.dim)
                m[k % self.dim :: 2] = 1.0  # alternating coordinate masks
                masks.append(m)
            self.register_buffer("masks", torch.stack(masks))
            self.s = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.dim, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.dim))
                    for _ in range(self.layers)
                ]
            )
            self.t = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.dim, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.dim))
                    for _ in range(self.layers)
                ]
            )

        def _normalize(self, x: Any) -> tuple[Any, Any]:
            """x -> z (toward the base) and the accumulated log|det dz/dx|."""
            z = x
            logdet = torch.zeros(x.shape[0], device=x.device)
            for m, s_net, t_net in zip(self.masks, self.s, self.t):
                zm = z * m
                s = s_net(zm) * (1.0 - m)
                t = t_net(zm) * (1.0 - m)
                z = zm + (1.0 - m) * ((z - t) * torch.exp(-s))
                logdet = logdet - s.sum(dim=1)
            return z, logdet

        def log_density(self, x: Any) -> Any:
            z, logdet = self._normalize(x)
            base = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * float(np.log(2.0 * np.pi))
            return base + logdet

        def sample(self, n: int) -> Any:
            z = torch.randn(int(n), self.dim, device=self.masks.device)
            x = z
            for m, s_net, t_net in zip(reversed(self.masks), reversed(list(self.s)), reversed(list(self.t))):
                xm = x * m
                s = s_net(xm) * (1.0 - m)
                t = t_net(xm) * (1.0 - m)
                x = xm + (1.0 - m) * (x * torch.exp(s) + t)  # inverse of _normalize
            return x

    return CouplingFlow


def build_coupling_flow(dim: int, *, hidden: int = 32, layers: int = 4) -> Any:
    """A RealNVP coupling flow over ``R^dim`` with an exact ``log_density(x)`` and ``sample(n)`` -- ready to wrap.

    Alternating affine-coupling layers map data to a standard-normal base; ``log_density`` is the base log-prob
    plus the log-determinant of the (triangular) Jacobian. A minimal, correct instance of the density module a
    :class:`NeuralDensity` adapts -- swap in any other module with the same two methods.
    """
    return _module_class("CouplingFlow")(dim, hidden, layers)


_register_module_class("CouplingFlow", _build_coupling_flow_class)


# --- a second, structurally different instance: a variational autoencoder (a latent-variable density) ---------


def build_vae(dim: int, *, latent: int = 2, hidden: int = 32) -> Any:
    """Build a variational autoencoder over ``R^dim``.

    An amortized encoder ``q(z | x)`` and a decoder ``p(x | z)`` (diagonal-Gaussian, learned observation scale)
    are trained by the ELBO with the reparameterization trick. This is a different family from a flow: structure
    is represented through a low-dimensional latent rather than an invertible map, while the same
    :class:`NeuralDensity` adapter can still use it because it exposes the same two methods.

    ``log_density(x)`` returns the **ELBO**, a lower bound on ``log p(x)``, not the exact value. Compare VAE
    leaves with other bounded leaves whenever possible. Mixing a VAE with an exact-density leaf, such as a
    Gaussian or flow, compares a bound against an exact value and can under-weight the VAE.

    ``log_density`` is **deterministic**: it evaluates the ELBO at the encoder mean ``z = mu(x)`` (no ``randn``
    resample), so repeated scoring of the same ``x`` is bit-identical and an EM log-likelihood stays monotone.
    Training still uses the reparameterized sample (``training=True``) for an unbiased gradient.
    """
    return _module_class("VAE")(dim, latent, hidden)


def _build_vae_class(torch: Any, nn: Any) -> Any:
    class VAE(nn.Module):
        def __init__(self, dim: int, latent: int = 2, hidden: int = 32) -> None:
            super().__init__()
            self.dim = int(dim)
            self.latent = int(latent)
            self.hidden = int(hidden)
            self.enc = nn.Sequential(nn.Linear(self.dim, self.hidden), nn.Tanh())
            self.enc_mu = nn.Linear(self.hidden, self.latent)
            self.enc_log_var = nn.Linear(self.hidden, self.latent)
            self.dec = nn.Sequential(nn.Linear(self.latent, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.dim))
            self.log_obs_scale = nn.Parameter(torch.zeros(1))  # learned diagonal p(x|z) scale

        def _decode_logp(self, x: Any, z: Any) -> Any:
            recon = self.dec(z)
            inv_var = torch.exp(-2.0 * self.log_obs_scale)
            return (
                -0.5 * ((x - recon) ** 2).sum(1) * inv_var
                - self.dim * self.log_obs_scale
                - 0.5 * self.dim * float(np.log(2.0 * np.pi))
            )

        def log_density(self, x: Any) -> Any:
            h = self.enc(x)
            mu, log_var = self.enc_mu(h), self.enc_log_var(h).clamp(-8.0, 8.0)
            # Deterministic when scoring: z = mu (encoder mean) so log_density is a fixed function of x, giving a
            # monotone EM LL. During training a reparameterized draw keeps the gradient unbiased.
            if self.training:
                z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)  # reparameterization
            else:
                z = mu
            kl = 0.5 * (torch.exp(log_var) + mu**2 - 1.0 - log_var).sum(1)
            return self._decode_logp(x, z) - kl  # ELBO: E_q[log p(x|z)] - KL(q(z|x) || p(z))

        def sample(self, n: int) -> Any:
            z = torch.randn(int(n), self.latent, device=self.log_obs_scale.device)
            return self.dec(z) + torch.exp(self.log_obs_scale) * torch.randn(int(n), self.dim, device=z.device)

    return VAE


_register_module_class("VAE", _build_vae_class)


# --- a third instance: a masked autoregressive flow (MAF) -- exact multivariate p(x) --------------------------


def build_maf(dim: int, *, hidden: int = 64, blocks: int = 3) -> Any:
    """A masked autoregressive flow over ``R^dim`` -- an **exact** density that factorizes ``p(x)`` by the chain
    rule, each ``p(x_i | x_{<i})`` an affine map with autoregressive (MADE-masked) mean and log-scale.

    Unlike the coupling flow it conditions every coordinate on *all* earlier ones (a richer autoregressive
    dependence), and unlike the VAE its ``log_density`` is exact. It can therefore be compared directly with a
    Gaussian, a flow, or another exact-density leaf. Sampling is the sequential inverse (one coordinate at a
    time). Another ready module for :class:`NeuralDensity`; the adapter is unchanged.
    """
    return _module_class("MAF")(dim, hidden, blocks)


def _build_masked_linear_class(torch: Any, nn: Any) -> Any:
    class MaskedLinear(nn.Linear):
        def set_mask(self, mask: Any) -> None:
            self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))

        def forward(self, x: Any) -> Any:
            return nn.functional.linear(x, self.mask * self.weight, self.bias)

    return MaskedLinear


_register_module_class("MaskedLinear", _build_masked_linear_class)


def _build_made_class(torch: Any, nn: Any) -> Any:
    MaskedLinear = _module_class("MaskedLinear")

    class MADE(nn.Module):
        """Autoregressive net: outputs per-coordinate ``(mu, log_scale)`` depending only on earlier coordinates."""

        def __init__(self, dim: int, hidden: int = 64) -> None:
            super().__init__()
            self.D = int(dim)
            self.hidden = int(hidden)
            m_in = np.arange(1, self.D + 1)
            m_h = 1 + (np.arange(self.hidden) % max(self.D - 1, 1))
            self.l1 = MaskedLinear(self.D, self.hidden)
            self.l2 = MaskedLinear(self.hidden, self.hidden)
            self.lout = MaskedLinear(self.hidden, 2 * self.D)
            self.l1.set_mask((m_h[:, None] >= m_in[None, :]).astype(float))
            self.l2.set_mask((m_h[:, None] >= m_h[None, :]).astype(float))
            m_out = np.concatenate([m_in, m_in])  # mu block then log-scale block
            self.lout.set_mask((m_out[:, None] > m_h[None, :]).astype(float))  # strict: output_i sees x_{<i} only
            self.act = nn.Tanh()

        def forward(self, x: Any) -> tuple[Any, Any]:
            h = self.act(self.l2(self.act(self.l1(x))))
            out = self.lout(h)
            return out[:, : self.D], out[:, self.D :].clamp(-5.0, 5.0)

    return MADE


_register_module_class("MADE", _build_made_class)


def _build_maf_class(torch: Any, nn: Any) -> Any:
    MADE = _module_class("MADE")

    class MAF(nn.Module):
        def __init__(self, dim: int, hidden: int = 64, blocks: int = 3) -> None:
            super().__init__()
            self.D = int(dim)
            self.hidden = int(hidden)
            self.blocks = int(blocks)
            self.mades = nn.ModuleList([MADE(self.D, self.hidden) for _ in range(self.blocks)])

        def log_density(self, x: Any) -> Any:
            z = x
            logdet = torch.zeros(x.shape[0], device=x.device)
            for i, made in enumerate(self.mades):
                mu, log_scale = made(z)
                z = (z - mu) * torch.exp(-log_scale)  # x_i -> z_i, affine and autoregressive
                logdet = logdet - log_scale.sum(1)
                if i < len(self.mades) - 1:
                    z = z.flip(1)  # reverse the order between blocks so every coordinate leads somewhere
            base = -0.5 * (z**2).sum(1) - 0.5 * self.D * float(np.log(2.0 * np.pi))
            return base + logdet

        def sample(self, n: int) -> Any:
            z = torch.randn(int(n), self.D, device=next(self.parameters()).device)
            for i in reversed(range(len(self.mades))):
                if i < len(self.mades) - 1:
                    z = z.flip(1)  # undo the inter-block flip
                made = self.mades[i]
                x = torch.zeros_like(z)
                for d in range(self.D):  # sequential inverse: coordinate d needs x_{<d} already filled
                    mu, log_scale = made(x)
                    x[:, d] = z[:, d] * torch.exp(log_scale[:, d]) + mu[:, d]
                z = x
            return z

    return MAF


_register_module_class("MAF", _build_maf_class)


# --- a discrete instance: an autoregressive categorical density -- exact p(x) over DISCRETE vectors ------------


def build_autoregressive_categorical(dim: int, n_categories: int, *, hidden: int = 64) -> Any:
    """An autoregressive neural density over **discrete** vectors ``x in {0..C-1}^dim`` -- exact, normalized ``p(x)``.

    The continuous flows/VAE above model ``R^d``; heterogeneous data is also categorical. This factorizes
    ``p(x) = prod_i p(x_i | x_{<i})`` with a MADE-masked network whose per-coordinate softmax *is* each conditional,
    so the density is **exactly normalized** (sums to 1 over the finite space) and can be compared directly with
    count/categorical families. ``log_density`` sums the picked log-softmax logits; ``sample`` fills the vector
    one coordinate at a time. Another ready module for :class:`NeuralDensity`; the adapter is unchanged.
    """
    return _module_class("AutoregressiveCategorical")(dim, n_categories, hidden)


def _build_autoregressive_categorical_class(torch: Any, nn: Any) -> Any:
    MaskedLinear = _module_class("MaskedLinear")

    class AutoregressiveCategorical(nn.Module):
        def __init__(self, dim: int, n_categories: int, hidden: int = 64) -> None:
            super().__init__()
            self.D = int(dim)
            self.C = int(n_categories)
            self.hidden = int(hidden)
            D, C, hid = self.D, self.C, self.hidden
            m_in = np.arange(1, D + 1)
            m_h = 1 + (np.arange(hid) % max(D - 1, 1))
            self.l1 = MaskedLinear(D, hid)
            self.l2 = MaskedLinear(hid, hid)
            self.lout = MaskedLinear(hid, D * C)
            self.l1.set_mask((m_h[:, None] >= m_in[None, :]).astype(float))
            self.l2.set_mask((m_h[:, None] >= m_h[None, :]).astype(float))
            m_out = np.repeat(m_in, C)  # C logits per coordinate, all carrying that coordinate's degree
            self.lout.set_mask((m_out[:, None] > m_h[None, :]).astype(float))  # strict: logits_i see x_{<i} only
            self.act = nn.Tanh()

        def _logits(self, x: Any) -> Any:
            h = self.act(self.l2(self.act(self.l1(x))))
            return self.lout(h).view(-1, self.D, self.C)  # (n, D, C)

        def log_density(self, x: Any) -> Any:
            log_p = torch.log_softmax(self._logits(x), dim=-1)  # (n, D, C), each row a proper conditional
            idx = x.long().clamp(0, self.C - 1).unsqueeze(-1)  # (n, D, 1)
            return log_p.gather(-1, idx).squeeze(-1).sum(1)  # sum_i log p(x_i | x_{<i})

        def sample(self, n: int) -> Any:
            x = torch.zeros(int(n), self.D, device=next(self.parameters()).device)
            for d in range(self.D):  # coordinate d's conditional depends only on already-filled x_{<d}
                probs = torch.softmax(self._logits(x)[:, d, :], dim=-1)
                x[:, d] = torch.multinomial(probs, 1).squeeze(-1).float()
            return x

    return AutoregressiveCategorical


_register_module_class("AutoregressiveCategorical", _build_autoregressive_categorical_class)


# Register the leaf so ``to_json``/``from_json`` (and a mixture holding it) round-trip. The auto-walk in
# mixle.utils.serialization only covers mixle.stats/mixle.analysis, so mixle.models classes opt in explicitly here.
def _register_serializable() -> None:
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover - serialization support is optional at import  # noqa: BLE001
        return
    register_serializable_class(NeuralDensity)


_register_serializable()
