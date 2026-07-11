"""Constructible neural-density families -- ``VAE(dim=8, latent=2)`` instead of ``NeuralDensity(build_vae(...))``.

Each class here is a thin subclass of :class:`~mixle.models.neural_density.NeuralDensity` whose ``__init__`` builds
its torch module from readable hyperparameters. So a neural density is a *first-class distribution object* you drop
straight into a ``MixtureDistribution`` / composite / HMM emission -- no ``build_* `` + adapter double-wrap::

    from mixle.models import VAE
    from mixle.stats import GaussianDistribution, MixtureDistribution

    mix = MixtureDistribution([VAE(dim=8, latent=2), GaussianDistribution(...)], [0.5, 0.5])
    fitted = optimize(x, mix.estimator())          # one estimator() at the top -- fits the VAE jointly by EM

They fit through the same :class:`~mixle.models.neural_density.NeuralDensityEstimator`; construct that directly to
build an estimator tree without a ``dist.estimator()`` hop::

    from mixle.models import NeuralDensityEstimator, build_vae
    from mixle.stats import MixtureEstimator, GaussianEstimator

    est = MixtureEstimator([NeuralDensityEstimator(build_vae(8, latent=2)), GaussianEstimator()])

The classes live in this module (not ``neural_density``) on purpose: the wrapped ``nn.Module`` classes are already
resolved as ``neural_density.<Name>`` for pickle, so a distribution class of the same name there would shadow them.
"""

from __future__ import annotations

from typing import Any

from mixle.models._neural_serial import decode_module
from mixle.models.neural_density import (
    NeuralDensity,
    build_autoregressive_categorical,
    build_coupling_flow,
    build_maf,
    build_vae,
)


class _NeuralFamily(NeuralDensity):
    """Base for constructible neural-density families.

    Subclasses build their module in ``__init__``; serialization stays the module-bytes path inherited from
    ``NeuralDensity`` (the JSON/pickle round-trip goes through ``__new__`` + ``__pysp_setstate__``, never
    ``__init__``, so the hyperparameter signature is irrelevant there). ``from_dict`` is overridden to rebuild the
    subclass around the decoded module rather than re-run the hyperparameter ``__init__``.
    """

    def __str__(self) -> str:
        return f"{type(self).__name__}(dim={getattr(self.module, 'dim', '?')})"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> _NeuralFamily:
        obj = cls.__new__(cls)
        NeuralDensity.__init__(
            obj,
            decode_module(payload["module"]),
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            device=payload["device"],
            name=payload["name"],
        )
        return obj


class VAE(_NeuralFamily):
    """A latent-variable ``p(x)`` over ``R^dim`` via a variational autoencoder.

    ``log_density`` is the ELBO -- a lower bound on ``log p(x)`` -- evaluated deterministically at the encoder
    mean so an EM log-likelihood stays monotone. Compare it with other bounded leaves whenever possible; mixing it
    with an exact-density leaf, such as a Gaussian or flow, compares a bound against an exact value and can
    under-weight the VAE. See :func:`~mixle.models.neural_density.build_vae` for the full statement.
    """

    def __init__(
        self,
        dim: int,
        *,
        latent: int = 2,
        hidden: int = 32,
        m_steps: int = 120,
        lr: float = 5e-3,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        super().__init__(build_vae(dim, latent=latent, hidden=hidden), m_steps=m_steps, lr=lr, device=device, name=name)


class Flow(_NeuralFamily):
    """An **exact** ``p(x)`` over ``R^dim`` via a RealNVP coupling flow (invertible map to a standard-normal base)."""

    def __init__(
        self,
        dim: int,
        *,
        hidden: int = 32,
        layers: int = 4,
        m_steps: int = 80,
        lr: float = 5e-3,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        super().__init__(
            build_coupling_flow(dim, hidden=hidden, layers=layers), m_steps=m_steps, lr=lr, device=device, name=name
        )


class MAF(_NeuralFamily):
    """An **exact** ``p(x)`` over ``R^dim`` via a masked autoregressive flow (richer autoregressive dependence)."""

    def __init__(
        self,
        dim: int,
        *,
        hidden: int = 64,
        blocks: int = 3,
        m_steps: int = 80,
        lr: float = 5e-3,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        super().__init__(build_maf(dim, hidden=hidden, blocks=blocks), m_steps=m_steps, lr=lr, device=device, name=name)


class DiscreteAR(_NeuralFamily):
    """An **exact**, normalized ``p(x)`` over discrete vectors ``x in {0..cats-1}^dim`` (autoregressive, MADE-masked)."""

    def __init__(
        self,
        dim: int,
        cats: int,
        *,
        hidden: int = 64,
        m_steps: int = 100,
        lr: float = 5e-3,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        super().__init__(
            build_autoregressive_categorical(dim, cats, hidden=hidden),
            m_steps=m_steps,
            lr=lr,
            device=device,
            name=name,
        )


def _register_serializable() -> None:
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover - serialization support is optional at import  # noqa: BLE001
        return
    for cls in (VAE, Flow, MAF, DiscreteAR):
        register_serializable_class(cls)


_register_serializable()
