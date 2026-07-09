"""Package a fitted model or callable as a reusable skill.

``skill`` wraps a model, subgraph, or plain callable as a named
:class:`Skill` and stores it in a :class:`SkillRegistry`. Skills carry
description, tags, provenance, and any estimation certificate inherited from
the wrapped object, so reuse remains auditable.

``find(query)`` ranks skills by lexical overlap against the skill name,
description, and tags. :meth:`SkillRegistry.index` can also mirror skills into a
:class:`~mixle.substrate.Substrate` when a workflow wants retrieval over the
same capability catalog.

The wrapped callable is resolved from the object: explicit ``call=`` wins, then
``predict`` or ``__call__``, then ``sampler`` for generative models.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class Skill:
    """A named, described, provenanced callable derived from a fitted artifact (or a plain function)."""

    name: str
    call: Callable[..., Any]
    description: str = ""
    tags: tuple[str, ...] = ()
    certificate: Any | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    @property
    def guarantee(self) -> Any | None:
        """The inherited estimation guarantee, if this skill wraps a certified model."""
        return getattr(self.certificate, "guarantee", None)

    def _haystack(self) -> set[str]:
        return _tokens(self.name) | _tokens(self.description) | {t.lower() for t in self.tags}

    def score(self, query: str) -> float:
        """Lexical overlap of ``query`` with this skill's name/description/tags, in ``[0, 1]``."""
        q = _tokens(query)
        if not q:
            return 0.0
        return len(q & self._haystack()) / len(q)


def _resolve_call(obj: Any, call: Callable[..., Any] | None) -> Callable[..., Any]:
    """Pick the callable a skill exposes: explicit ``call`` > model verb > the object itself."""
    if call is not None:
        return call
    for attr in ("predict", "__call__"):
        fn = getattr(obj, attr, None)
        if callable(fn) and not isinstance(obj, type):
            # bare functions have __call__ too; only prefer it when obj isn't already the function
            if attr == "__call__" and callable(obj) and not hasattr(obj, "predict"):
                break
            return fn
    if hasattr(obj, "sampler"):
        return lambda n=1, seed=0: obj.sampler(seed=seed).sample(int(n))
    if callable(obj):
        return obj
    raise TypeError(f"cannot derive a callable from {type(obj).__name__}; pass call=")


class SkillRegistry:
    """A findable collection of skills -- register once, retrieve by name or by query."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def add(self, sk: Skill) -> Skill:
        """Register a skill and return it for chaining."""
        self._skills[sk.name] = sk
        return sk

    def get(self, name: str) -> Skill:
        """Return a registered skill by exact name."""
        return self._skills[name]

    def all(self) -> list[Skill]:
        """Return all registered skills in insertion order."""
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def find(self, query: str, k: int = 5) -> list[Skill]:
        """The best ``k`` skills for ``query`` by lexical overlap (highest score first, ties by name)."""
        scored = [(sk.score(query), sk) for sk in self._skills.values()]
        scored = [(s, sk) for s, sk in scored if s > 0.0]
        scored.sort(key=lambda t: (-t[0], t[1].name))
        return [sk for _, sk in scored[:k]]

    def best(self, query: str) -> Skill | None:
        """The single best-matching skill for ``query`` (or None if nothing overlaps)."""
        hits = self.find(query, k=1)
        return hits[0] if hits else None

    def index(self, substrate: Any) -> list[str]:
        """Mirror every skill into a :class:`~mixle.substrate.Substrate` for embedding-grade recall.

        The registry's own ``find`` is a lexical matcher; when a corpus grows past that, index the
        skills as ``artifact`` items and retrieve through the substrate instead. Returns the item ids."""
        ids: list[str] = []
        for sk in self._skills.values():
            text = f"{sk.name}: {sk.description}".strip()
            item_id = substrate.add(
                text=text,
                kind="artifact",
                payload={"skill": sk.name, "tags": list(sk.tags)},
                provenance={"origin": "skill", **sk.provenance},
            )
            ids.append(item_id)
        return ids


_DEFAULT_REGISTRY = SkillRegistry()


def default_registry() -> SkillRegistry:
    """The process-wide default registry :func:`skill` writes to when no ``registry`` is given."""
    return _DEFAULT_REGISTRY


def skill(
    name: str,
    obj: Any,
    *,
    description: str = "",
    tags: tuple[str, ...] = (),
    call: Callable[..., Any] | None = None,
    provenance: dict[str, Any] | None = None,
    registry: SkillRegistry | None = None,
) -> Skill:
    """Package ``obj`` (a fitted model, a :class:`~mixle.inference.CreatedModel`, or a function) as a
    reusable :class:`Skill` and register it (see module docstring).

    The estimation certificate is inherited from ``obj.certificate`` when present (so a certified model
    yields a certified skill). ``call`` overrides how the skill is invoked; otherwise a model verb
    (``predict`` / ``__call__`` / ``sampler``) is used. The skill is added to ``registry`` (or the
    process default) and returned.
    """
    cert = getattr(obj, "certificate", None)
    prov = dict(provenance or {})
    if hasattr(obj, "provenance") and isinstance(obj.provenance, dict):
        prov = {**obj.provenance, **prov}
    target = getattr(obj, "model", obj)  # a CreatedModel wraps the fitted model in .model
    sk = Skill(
        name=name,
        call=_resolve_call(target, call),
        description=description,
        tags=tuple(tags),
        certificate=cert,
        provenance=prov,
    )
    (registry if registry is not None else _DEFAULT_REGISTRY).add(sk)
    return sk
