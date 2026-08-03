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

:func:`build_contrastive_projection` is the separate batch-objective/ranker for an InfoNCE projection between
two, typically frozen, embedding spaces. It deliberately does not implement ``log_density`` and cannot be wrapped
as a conditional distribution: cross-row negatives are not per-observation probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.models._neural_serial import check_finite, decode_module, encode_module
from mixle.models.grad_leaf import (
    GradEstimator,
    _call_sample_with_seed,
    _module_mode,
    _positive_float,
    _positive_int,
    _resolve_device,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.exact import require_exact_bool


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
        if not callable(getattr(module, "log_density", None)):
            raise TypeError(
                "NeuralConditionalDensity requires module.log_density(x, y) with one independent score per row."
            )
        self.module = module
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_float(lr, "lr")
        self.device = device
        self.name = name

    def __str__(self) -> str:
        return f"NeuralConditionalDensity({type(self.module).__name__})"

    def log_density(self, xy: Any) -> float:
        """Return ``log p(y | x)`` for one observation pair ``(x, y)``."""
        x, y = xy
        return float(self.seq_log_density(([np.atleast_1d(x)], [np.atleast_1d(y)]))[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return per-row conditional log densities for encoded ``(x, y)`` arrays."""
        torch = _torch()
        xs, ys = enc
        xx = check_finite(np.atleast_2d(np.asarray(xs, dtype=float)), "NeuralConditionalDensity.seq_log_density (x)")
        yy = check_finite(np.atleast_2d(np.asarray(ys, dtype=float)), "NeuralConditionalDensity.seq_log_density (y)")
        if xx.shape[0] != yy.shape[0]:
            raise ValueError("conditional density x and y batches must have identical row counts.")
        dev = _resolve_device(self.device, torch)
        self.module.to(dev)
        xt = torch.as_tensor(xx, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(yy, dtype=torch.float32, device=dev)
        with _module_mode(self.module, train=False), torch.no_grad():
            scores = self.module.log_density(xt, yt)
            if tuple(scores.shape) != (xt.shape[0],):
                raise ValueError(
                    f"conditional log_density must return one score per row; got shape {tuple(scores.shape)}."
                )
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("conditional log_density returned non-finite scores.")
            return scores.cpu().numpy()

    def sampler(self, seed: int | None = None) -> NeuralConditionalDensitySampler:
        """Return a conditional sampler for drawing ``y`` given ``x``."""
        return NeuralConditionalDensitySampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralConditionalDensityEstimator:
        """Return the generalized-EM estimator for weighted conditional-density training."""
        return NeuralConditionalDensityEstimator(
            self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name
        )

    def dist_to_encoder(self) -> NeuralConditionalDensityEncoder:
        """Return the encoder for ``(x, y)`` observation pairs."""
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
        """Serialize hyperparameters and module bytes for registry-based round trips."""
        return {
            "m_steps": self.m_steps,
            "lr": self.lr,
            "device": self.device,
            "name": self.name,
            "module": encode_module(self.module),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NeuralConditionalDensity:
        """Rebuild a :class:`NeuralConditionalDensity` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["module"]),
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            device=payload["device"],
            name=payload["name"],
        )


class NeuralConditionalDensitySampler(DistributionSampler):
    """Conditional sampler for modules exposing ``sample_given(x)``."""

    def __init__(self, dist: NeuralConditionalDensity, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because the leaf defines ``p(y | x)`` and has no marginal ``p(x)``."""
        raise NotImplementedError("NeuralConditionalDensity is conditional p(y|x); use sampler().sample_given(x).")

    def sample_given(self, x: Any) -> np.ndarray:
        """Draw one response from ``p(y | x)`` using the wrapped module."""
        torch = _torch()
        dev = _resolve_device(self.dist.device, torch)
        self.dist.module.to(dev)
        seed = int(self.rng.randint(0, 2**31 - 1))
        rows = check_finite(np.atleast_2d(np.asarray(x, dtype=float)), "sample_given")
        if rows.shape[0] != 1:
            raise ValueError("sample_given accepts one conditioning row; use sample_given_batch for a batch.")
        xt = torch.as_tensor(rows, dtype=torch.float32, device=dev)
        with _module_mode(self.dist.module, train=False), torch.no_grad():
            output = _call_sample_with_seed(self.dist.module.sample_given, xt, torch=torch, device=dev, seed=seed)
            if not isinstance(output, torch.Tensor) or output.ndim < 1 or output.shape[0] != 1:
                raise ValueError("conditional sample_given must return one output row per conditioning row.")
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("conditional sample_given returned non-finite output.")
            return output.cpu().numpy()[0]

    def sample_given_batch(self, x_batch: Any) -> np.ndarray:
        """One draw of ``y ~ p(y | x)`` for every row of ``x_batch`` (shape ``(n, x_dim)``), in one
        batched forward pass -- statistically identical to calling :meth:`sample_given` once per row
        (same model, same per-draw sampling procedure), just without paying framework/dispatch
        overhead per row. That per-call overhead dominates a Python loop of hundreds of individual
        ``sample_given`` calls, which is exactly the shape both a particle-walk step (many different
        x's, one draw each) and a per-point coverage check (repeat one x, many draws) reduce to --
        both call sites use this to speed up the same check/walk rather than shrink it. Repeat a row
        of ``x_batch`` to draw more than once from the same ``x``."""
        torch = _torch()
        dev = _resolve_device(self.dist.device, torch)
        self.dist.module.to(dev)
        seed = int(self.rng.randint(0, 2**31 - 1))
        rows = check_finite(np.atleast_2d(np.asarray(x_batch, dtype=float)), "sample_given_batch")
        if rows.shape[0] == 0:
            raise ValueError("sample_given_batch requires at least one conditioning row.")
        xt = torch.as_tensor(rows, dtype=torch.float32, device=dev)
        with _module_mode(self.dist.module, train=False), torch.no_grad():
            output = _call_sample_with_seed(self.dist.module.sample_given, xt, torch=torch, device=dev, seed=seed)
            if not isinstance(output, torch.Tensor) or output.ndim < 1 or output.shape[0] != len(rows):
                raise ValueError("conditional sample_given must return one output row per conditioning row.")
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("conditional sample_given returned non-finite output.")
            return output.cpu().numpy()  # (n, y_dim)


class NeuralConditionalDensityEncoder(DataSequenceEncoder):
    """Encode ``(x, y)`` pairs for vectorized conditional-density scoring and fitting."""

    def __str__(self) -> str:
        return "NeuralConditionalDensityEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralConditionalDensityEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        """Convert a list of ``(x, y)`` pairs into batched feature and target arrays."""
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
        """Add one weighted observation pair to the accumulator."""
        self.x.append(np.atleast_1d(np.asarray(xy[0], dtype=float))[None, ...])
        self.y.append(np.atleast_1d(np.asarray(xy[1], dtype=float))[None, ...])
        self.w.append(np.asarray([float(weight)], dtype=float))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Add a batch of encoded observation pairs and responsibility weights."""
        x, y = enc
        xb = np.asarray(x, dtype=float)
        yb = np.asarray(y, dtype=float)
        if xb.ndim == 0 or yb.ndim == 0:
            raise ValueError("conditional accumulator fields must have a row dimension.")
        self.x.append(xb.reshape(xb.shape[0], 1) if xb.ndim == 1 else xb)
        self.y.append(yb.reshape(yb.shape[0], 1) if yb.ndim == 1 else yb)
        wb = np.asarray(weights, dtype=float)
        if xb.shape[0] != yb.shape[0]:
            raise ValueError("conditional accumulator x and y fields must have identical row counts.")
        if wb.ndim != 1 or wb.shape[0] != xb.shape[0]:
            raise ValueError("conditional accumulator requires exactly one weight per row.")
        self.w.append(wb)

    def initialize(self, xy: Any, weight: float, rng: Any) -> None:
        """Initialize from one observation using the ordinary update path."""
        self.update(xy, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        """Initialize from an encoded batch using the ordinary batch update path."""
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralConditionalDensityAccumulator:
        """Merge the value tuple from another conditional-density accumulator."""
        xo, yo, wo = other
        if len(xo) or len(yo) or len(wo):
            self.seq_update((xo, yo), np.asarray(wo, dtype=float), None)
        return self

    def value(self) -> tuple:
        """Return contiguous ``(x, y, weights)`` arrays for the M-step."""
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        y = np.concatenate(self.y, axis=0) if self.y else np.zeros((0, 0))
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, y, w)

    def from_value(self, value: tuple) -> NeuralConditionalDensityAccumulator:
        """Restore accumulator buffers from a value tuple."""
        x, y, w = value
        self.x, self.y, self.w = [], [], []
        if len(x) or len(y) or len(w):
            self.seq_update((x, y), np.asarray(w, dtype=float), None)
        return self

    def acc_to_encoder(self) -> NeuralConditionalDensityEncoder:
        """Return the encoder expected by this accumulator."""
        return NeuralConditionalDensityEncoder()


class NeuralConditionalDensityAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for conditional-density accumulators."""

    def make(self) -> NeuralConditionalDensityAccumulator:
        """Create a fresh accumulator."""
        return NeuralConditionalDensityAccumulator()


class NeuralConditionalDensityEstimator(GradEstimator):
    """M-step: responsibility-weighted MLE ``max sum_i w_i log p(y_i | x_i)`` by gradient ascent (warm-started)."""

    def __init__(
        self, module: Any, *, m_steps: int = 60, lr: float = 5e-3, device: str = "cpu", name: str | None = None
    ) -> None:
        super().__init__(module, m_steps=m_steps, lr=lr, device=device, name=name)

    def accumulator_factory(self) -> NeuralConditionalDensityAccumulatorFactory:
        """Return an accumulator factory for weighted conditional-density batches."""
        return NeuralConditionalDensityAccumulatorFactory()

    def _leaf(self) -> NeuralConditionalDensity:
        return NeuralConditionalDensity(
            self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name
        )


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

        def sample_given(self, x: Any, *, generator: Any = None) -> Any:
            log_pi, mu, log_sigma = self._params(x)
            comp = torch.multinomial(torch.exp(log_pi), 1, generator=generator).squeeze(-1)  # (n,)
            idx = comp.view(-1, 1, 1).expand(-1, 1, self.y_dim)
            mu_c = mu.gather(1, idx).squeeze(1)  # (n, d)
            sig_c = torch.exp(log_sigma.gather(1, idx).squeeze(1))
            noise = torch.randn(mu_c.shape, dtype=mu_c.dtype, device=mu_c.device, generator=generator)
            return mu_c + sig_c * noise

    return MixtureDensityNetwork


_register_module_class("MixtureDensityNetwork", _build_mdn_class)


# --- the exact counterpart: a conditional normalizing flow -- exact p(y|x) with within-y structure -------------


def build_conditional_flow(x_dim: int, y_dim: int, *, hidden: int = 32, layers: int = 4) -> Any:
    """A conditional coupling flow: an **exact** ``p(y | x)`` whose transform of ``y`` is conditioned on ``x``.

    The exact-density counterpart to :func:`build_mdn`. Each affine-coupling layer's shift/scale networks take
    both the passed-through ``y`` coordinates *and* ``x``, so the whole invertible ``y``-transform bends with the
    input -- capturing *within-``y``* dependence (e.g. ``y2`` a nonlinear function of ``y1``) that a single-Gaussian
    :class:`~mixle.models.neural_leaf.NeuralGaussian` (isotropic mean-only) cannot, while keeping an exact
    log-density rather than a bound. Needs ``y_dim >= 2`` for the coupling to be non-trivial. Exposes
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
            if self.y_dim < 2:
                raise ValueError("conditional coupling flows require y_dim >= 2.")
            masks = []
            for k in range(self.layers):
                m = torch.zeros(self.y_dim)
                m[k % 2 :: 2] = 1.0  # alternate the two parity masks for every layer
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

        def sample_given(self, x: Any, *, generator: Any = None) -> Any:
            y = torch.randn(x.shape[0], self.y_dim, device=x.device, generator=generator)
            for m, s_net, t_net in zip(reversed(self.masks), reversed(list(self.s)), reversed(list(self.t))):
                ym = y * m
                inp = torch.cat([ym, x], dim=1)
                s = s_net(inp) * (1.0 - m)
                t = t_net(inp) * (1.0 - m)
                y = ym + (1.0 - m) * (y * torch.exp(s) + t)  # inverse of _normalize
            return y

    return ConditionalFlow


_register_module_class("ConditionalFlow", _build_conditional_flow_class)


# --- batch-contrastive projection/ranker (deliberately not a probability-distribution leaf) --------------------


def build_contrastive_projection(
    d_x: int,
    d_y: int,
    *,
    encoder_x: Any = None,
    encoder_y: Any = None,
    proj_dim: int | None = None,
    hidden: int = 64,
    freeze_encoders: bool = True,
    temperature: float = 0.07,
) -> Any:
    """Build a contrastive projection/ranker between two embedding spaces.

    This is the stage-1 multimodal pattern -- frozen encoder -> trainable projection -> frozen encoder -- stated
    with no domain nouns. ``encoder_x``/``encoder_y`` are any torch module mapping a raw item to a ``d_x``/``d_y``
    embedding; both default to ``nn.Identity()``, so ``x``/``y`` may already BE the embeddings (pass precomputed
    vectors straight in, no backbone required). Encoders are frozen by default (``freeze_encoders=True``): their
    parameters get ``requires_grad_(False)`` and the module is pinned in ``eval()`` regardless of the outer
    ``train()``/``eval()`` calls the M-step makes, so no dropout/batchnorm noise leaks into a "frozen" backbone
    and no gradient ever reaches it. The only trainable piece is a small projection head per side (``d_x`` /
    ``d_y`` -> ``hidden`` -> ``proj_dim``, default ``proj_dim = min(d_x, d_y)``) mapping BOTH embeddings into one
    shared, L2-normalized space -- the CLIP design (two projections into a shared space), not a single asymmetric
    ``x -> y`` regression -- so the same leaf answers "which y matches this x" and "which x matches this y".

    ``contrastive_loss`` is a symmetric weighted InfoNCE objective over a complete paired batch. Anchor weights
    and the two candidate/negative measures are explicit and independently validated, so responsibilities are
    never applied only after cross-row negative contributions have already been formed. ``score_pairs`` and
    ``similarity_matrix`` are batch-independent retrieval surfaces.

    This object intentionally has no ``log_density`` or ``sample_given`` method: a contrastive batch objective
    is not a normalized per-observation probability distribution and cannot be composed into mixture/HMM EM.
    """
    return _module_class("ContrastiveProjection")(
        d_x, d_y, encoder_x, encoder_y, proj_dim, hidden, freeze_encoders, temperature
    )


def build_projection_leaf(
    d_x: int,
    d_y: int,
    *,
    encoder_x: Any = None,
    encoder_y: Any = None,
    proj_dim: int | None = None,
    hidden: int = 64,
    freeze_encoders: bool = True,
    temperature: float = 0.07,
) -> Any:
    """Compatibility spelling for :func:`build_contrastive_projection`; the result is not a distribution leaf."""
    return build_contrastive_projection(
        d_x,
        d_y,
        encoder_x=encoder_x,
        encoder_y=encoder_y,
        proj_dim=proj_dim,
        hidden=hidden,
        freeze_encoders=freeze_encoders,
        temperature=temperature,
    )


def _build_contrastive_projection_class(torch: Any, nn: Any) -> Any:
    class ContrastiveProjection(nn.Module):
        def __init__(
            self,
            d_x: int,
            d_y: int,
            encoder_x: Any = None,
            encoder_y: Any = None,
            proj_dim: int | None = None,
            hidden: int = 64,
            freeze_encoders: bool = True,
            temperature: float = 0.07,
        ) -> None:
            super().__init__()
            self.d_x = _positive_int(d_x, "d_x")
            self.d_y = _positive_int(d_y, "d_y")
            self.proj_dim = _positive_int(proj_dim, "proj_dim") if proj_dim is not None else min(self.d_x, self.d_y)
            self.hidden = _positive_int(hidden, "hidden")
            if not isinstance(freeze_encoders, (bool, np.bool_)):
                raise TypeError("freeze_encoders must be Boolean.")
            self._freeze_encoders = require_exact_bool(freeze_encoders, "freeze_encoders")
            temperature = _positive_float(temperature, "temperature")
            self.encoder_x = encoder_x if encoder_x is not None else nn.Identity()
            self.encoder_y = encoder_y if encoder_y is not None else nn.Identity()
            if self._freeze_encoders:
                for p in self.encoder_x.parameters():
                    p.requires_grad_(False)
                for p in self.encoder_y.parameters():
                    p.requires_grad_(False)
                self.encoder_x.eval()
                self.encoder_y.eval()
            # the ONLY trainable pieces: two small projection heads into a shared space, plus a learned
            # (CLIP-style) log-temperature -- clamped at use so training cannot blow the softmax up/flat.
            self.proj_x = nn.Sequential(
                nn.Linear(self.d_x, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.proj_dim)
            )
            self.proj_y = nn.Sequential(
                nn.Linear(self.d_y, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.proj_dim)
            )
            self.log_tau = nn.Parameter(torch.log(torch.tensor(float(temperature))))

        def train(self, mode: bool = True) -> ContrastiveProjection:
            super().train(mode)
            if self._freeze_encoders:  # a frozen backbone stays in eval() no matter what the M-step requests
                self.encoder_x.eval()
                self.encoder_y.eval()
            return self

        def embed_x(self, x: Any) -> Any:
            """``x`` (raw item or precomputed embedding) through the frozen encoder then the trainable
            projection, L2-normalized -- the vector to compare against ``embed_y`` in the shared space."""
            return nn.functional.normalize(self.proj_x(self.encoder_x(x)), dim=-1)

        def embed_y(self, y: Any) -> Any:
            return nn.functional.normalize(self.proj_y(self.encoder_y(y)), dim=-1)

        def similarity_matrix(self, x: Any, y: Any) -> Any:
            """Return all pairwise projected cosine similarities, independent of unrelated batches."""
            px, py = self.embed_x(x), self.embed_y(y)
            tau = self.log_tau.exp().clamp(min=1e-2, max=100.0)
            return (px @ py.t()) / tau

        def score_pairs(self, x: Any, y: Any) -> Any:
            """Score aligned pairs without using any other row as a negative."""
            if x.shape[0] != y.shape[0]:
                raise ValueError("score_pairs requires aligned x and y row counts.")
            px, py = self.embed_x(x), self.embed_y(y)
            tau = self.log_tau.exp().clamp(min=1e-2, max=100.0)
            return torch.sum(px * py, dim=-1) / tau

        @staticmethod
        def _measure(value: Any, rows: int, reference: Any, name: str) -> Any:
            if value is None:
                result = torch.ones(rows, device=reference.device, dtype=reference.dtype)
            else:
                result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
            if tuple(result.shape) != (rows,):
                raise ValueError(f"{name} must have shape ({rows},).")
            if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
                raise ValueError(f"{name} must be finite and non-negative.")
            total = result.sum()
            if not bool(torch.isfinite(total)) or not bool(total > 0):
                raise ValueError(f"{name} must have positive finite total mass.")
            return result / total

        def contrastive_loss(
            self,
            x: Any,
            y: Any,
            *,
            anchor_weights: Any = None,
            x_candidate_weights: Any = None,
            y_candidate_weights: Any = None,
        ) -> Any:
            """Return symmetric weighted InfoNCE with explicit anchor and candidate measures."""
            if x.shape[0] != y.shape[0]:
                raise ValueError("contrastive_loss requires aligned x and y row counts.")
            rows = x.shape[0]
            if rows < 2:
                raise ValueError("contrastive_loss requires at least two paired rows.")
            logits = self.similarity_matrix(x, y)
            anchors = self._measure(anchor_weights, rows, logits, "anchor_weights")
            x_measure = self._measure(x_candidate_weights, rows, logits, "x_candidate_weights")
            y_measure = self._measure(y_candidate_weights, rows, logits, "y_candidate_weights")
            active = anchors > 0
            if bool(((x_measure <= 0) & active).any()) or bool(((y_measure <= 0) & active).any()):
                raise ValueError("every positive-weight anchor must have positive matched candidate mass.")
            log_x_measure = torch.where(x_measure > 0, torch.log(x_measure), torch.full_like(x_measure, -torch.inf))
            log_y_measure = torch.where(y_measure > 0, torch.log(y_measure), torch.full_like(y_measure, -torch.inf))
            diagonal = torch.arange(rows, device=logits.device)
            x_to_y = (
                logits[diagonal, diagonal] + log_y_measure - torch.logsumexp(logits + log_y_measure[None, :], dim=1)
            )
            y_to_x = (
                logits[diagonal, diagonal] + log_x_measure - torch.logsumexp(logits + log_x_measure[:, None], dim=0)
            )
            pair_log_score = 0.5 * (x_to_y + y_to_x)
            return -(anchors[active] * pair_log_score[active]).sum()

    return ContrastiveProjection


_register_module_class("ContrastiveProjection", _build_contrastive_projection_class)
# Preserve deserialization of pre-release artifacts while removing their invalid probability surface.
_register_module_class("ProjectionLeaf", _build_contrastive_projection_class)


# --- the discrete conditional: an autoregressive categorical conditioned on x -- exact p(y|x) over discrete y ----


def build_conditional_autoregressive_categorical(x_dim: int, y_dim: int, n_categories: int, *, hidden: int = 64) -> Any:
    """An autoregressive categorical conditioned on ``x``: exact ``p(y | x)`` over **discrete** ``y in {0..C-1}^y_dim``.

    The conditional sibling of :func:`~mixle.models.neural_density.build_autoregressive_categorical` and the discrete
    counterpart to :func:`build_conditional_flow`. It factorizes ``p(y | x) = prod_i p(y_i | y_{<i}, x)`` with a
    MADE-masked net over ``y`` into which ``x`` is injected *unmasked* (degree 0, so every coordinate may depend on
    ``x``). Each per-coordinate softmax is exactly a conditional, so the density is **exactly normalized** and
    comparable to other exact discrete conditional leaves. Exposes ``log_density(x, y)`` and ``sample_given(x)`` -- the contract a
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
            in_support = torch.isfinite(y) & (y == torch.round(y)) & (y >= 0) & (y < self.C)
            valid_rows = in_support.all(1)
            safe_y = torch.where(in_support, y, torch.zeros_like(y))
            log_p = torch.log_softmax(self._logits(x, safe_y), dim=-1)  # (n, D, C)
            idx = safe_y.long().unsqueeze(-1)
            score = log_p.gather(-1, idx).squeeze(-1).sum(1)
            return torch.where(valid_rows, score, torch.full_like(score, -torch.inf))

        def sample_given(self, x: Any, *, generator: Any = None) -> Any:
            y = torch.zeros(x.shape[0], self.D, device=x.device)
            for d in range(self.D):  # coordinate d depends on x (always) and already-filled y_{<d}
                probs = torch.softmax(self._logits(x, y)[:, d, :], dim=-1)
                y[:, d] = torch.multinomial(probs, 1, generator=generator).squeeze(-1).float()
            return y

    return ConditionalAutoregressiveCategorical


_register_module_class("ConditionalAutoregressiveCategorical", _build_conditional_autoregressive_categorical_class)


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(NeuralConditionalDensity)


_register_serializable()
