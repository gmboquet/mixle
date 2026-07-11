"""Machine-readable API maturity registry (worklist A1.2).

``docs/maturity.rst`` describes, in prose, how mature each public surface is. This is the machine-readable
mirror of that map: a single source of truth a tool (or the API manifest, worklist A1.6) can query to learn
the maturity tier of any dotted module name. The three tiers match the deprecation policy in
``docs/support-policy.rst`` (worklist A1.5):

* ``STABLE`` -- covered by the compatibility policy; changes follow the deprecation lifecycle.
* ``PROVISIONAL`` -- usable and tested, but signatures/defaults may still change within a minor release.
* ``EXPERIMENTAL`` -- no compatibility guarantee (only ``mixle.experimental``).

:func:`maturity_of` resolves a dotted name by longest matching prefix, so ``mixle.stats.latent.hidden_markov``
inherits ``mixle.stats``'s tier while ``mixle.inference.production`` overrides the ``mixle.inference`` tier.
Surfaces not listed default to :data:`DEFAULT_MATURITY` (``PROVISIONAL``) -- the conservative choice: a
surface makes no stability promise until it is explicitly recorded as stable.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Maturity", "MATURITY_REGISTRY", "DEFAULT_MATURITY", "maturity_of", "status_of"]


class Maturity(StrEnum):
    """API maturity tier. String-valued so it serializes as its name in JSON manifests."""

    STABLE = "stable"
    PROVISIONAL = "provisional"
    EXPERIMENTAL = "experimental"


# module prefix -> (tier, human status label mirrored from docs/maturity.rst). More specific prefixes win.
MATURITY_REGISTRY: dict[str, tuple[Maturity, str]] = {
    "mixle.stats": (Maturity.STABLE, "Stable core"),
    "mixle.inference": (Maturity.STABLE, "Stable core (optimize + direct estimation)"),
    "mixle.inference.production": (Maturity.PROVISIONAL, "Practical helpers, not a platform"),
    "mixle.enumeration": (Maturity.PROVISIONAL, "Usable, evolving"),
    "mixle.ops": (Maturity.PROVISIONAL, "Usable, evolving"),
    "mixle.ppl": (Maturity.PROVISIONAL, "Active development"),
    "mixle.process": (Maturity.PROVISIONAL, "Active development"),
    "mixle.models": (Maturity.PROVISIONAL, "Incubating applied helpers"),
    "mixle.task": (Maturity.PROVISIONAL, "Active application/research workflows"),
    "mixle.reason": (Maturity.PROVISIONAL, "Active application/research workflows"),
    "mixle.substrate": (Maturity.PROVISIONAL, "New local application runtime"),
    "mixle.pool": (Maturity.PROVISIONAL, "New local application runtime"),
    "mixle.telemetry": (Maturity.PROVISIONAL, "New local application runtime"),
    "mixle.scientist": (Maturity.PROVISIONAL, "Optional assembled workflow"),
    "mixle.doe": (Maturity.PROVISIONAL, "Active application/research workflows"),
    "mixle.evolve": (Maturity.PROVISIONAL, "Active application/research workflows"),
    "mixle.experimental": (Maturity.EXPERIMENTAL, "No compatibility guarantee"),
}

DEFAULT_MATURITY = Maturity.PROVISIONAL


def _resolve(name: str) -> tuple[Maturity, str] | None:
    """Return the registry entry for the longest prefix of ``name`` (dot-boundary aware), or None."""
    parts = name.split(".")
    for depth in range(len(parts), 0, -1):
        prefix = ".".join(parts[:depth])
        if prefix in MATURITY_REGISTRY:
            return MATURITY_REGISTRY[prefix]
    return None


def maturity_of(name: str) -> Maturity:
    """Return the :class:`Maturity` tier of a dotted module name (longest-prefix match; default provisional)."""
    entry = _resolve(name)
    return entry[0] if entry is not None else DEFAULT_MATURITY


def status_of(name: str) -> str:
    """Return the human status label for a dotted module name, mirroring ``docs/maturity.rst``."""
    entry = _resolve(name)
    return entry[1] if entry is not None else "Unclassified (provisional by default)"
