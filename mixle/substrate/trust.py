"""``verify_lineage()`` -- check that an item's provenance chain is intact (trust / provenance, N1).

Every substrate item can point at the items it genuinely derives from -- ``derived_from`` is the typed
provenance/ancestry edge list (parent documents, the model that produced an artifact, the trace it came
from), kept separate from ``links``, the generic, untyped KG-relation surface (:mod:`~mixle.substrate.kg_rag`
and :mod:`~mixle.substrate.multihop`'s "related to" edges) that is NOT ancestry (MXR-080-0261; see
:class:`~mixle.substrate.core.SubstrateItem`). A citation or a merge is only trustworthy if that ancestry
chain actually resolves: a link to an item that no longer exists is a dangling provenance edge, and any
claim resting on it is unverifiable.

:func:`verify_lineage` walks an item's ancestry through ``derived_from`` alone and returns a three-state
:class:`LineageState` verdict, never a bare boolean (MXR-080-0260/0261):

* ``INTACT`` -- the full chain was traversed: every ancestor authorized, resolved, cycle-free, complete.
* ``BROKEN`` -- a confirmed defect *within the verified region*: a dangling edge, or a cycle (an ancestor
  that is, directly or transitively, its own ancestor).
* ``UNVERIFIED`` -- the chain could not be fully checked: ``max_depth`` was reached with further,
  unexplored edges beyond it (MXR-080-0260 -- an unvisited tail is never silently certified), or an
  ancestor lies outside the caller's authorized ``scope`` and could be neither confirmed present nor
  absent (MXR-080-0261).

Scope enforcement (MXR-080-0261) follows the same convention :mod:`~mixle.substrate.belief` uses for the
identical problem (see ``assimilate``/``_resolve``/``_launders``): ``scope`` doubles as the authorization
principal, and every ancestor at every hop is resolved through an :class:`~mixle.substrate.spaces.AccessPolicy`
-- never a raw, scope-blind lookup. An ancestor id that does not exist at all and one that exists but is
private to a scope the caller cannot read are deliberately indistinguishable in the report (both land in
``unverified``): telling them apart would let a caller use their own scoped audits as an oracle for
"does something exist in a scope I can't read," the same side channel MXR-080-0237 closed for semantic
search and MXR-080-0244 closed for belief evidence. A scoped audit can no longer be validated -- silently
promoted to INTACT -- using content it has no authorized view of.

:func:`audit_substrate` runs it over the whole store -- a knowledge-integrity sweep, the same "trust is
re-derivable, not asserted" discipline the factuality receipts apply to answers, applied here to the
knowledge itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.spaces import AccessPolicy
from mixle.utils.immutable import detach_receipt_container


class LineageState(StrEnum):
    """The three-state verdict of a lineage verification (MXR-080-0260/0261) -- a closed vocabulary,
    the same discipline :class:`~mixle.substrate.spaces.MergeStrategy` applies to merge strategies.
    Never a bare boolean: a bare ``intact`` cannot distinguish "checked everything, it's fine" from
    "stopped looking partway and don't actually know" -- exactly the overclaim these findings fix.

    INTACT: the full chain was traversed -- every ancestor authorized and resolved, no cycle, no
        depth-cap truncation, no dangling edge. The only state that certifies anything.
    BROKEN: a confirmed structural defect *within the verified region* -- a dangling edge (an ancestor
        id that resolves to nothing, where the caller had legitimate visibility to know that for a
        fact) or a cycle (an ancestor that is, directly or transitively, its own ancestor).
    UNVERIFIED: the chain could not be fully checked -- ``max_depth`` was reached with further,
        unexplored edges beyond it, or an ancestor lies outside the caller's authorized scope and
        could be neither confirmed present nor absent. Never silently promoted to INTACT.
    """

    INTACT = "intact"
    BROKEN = "broken"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class LineageReport:
    """Whether an item's provenance chain resolves end to end -- and exactly why, if it doesn't.

    ``state`` is the headline three-state verdict (see :class:`LineageState`); every other field names
    precisely which ancestor ids drove it, so a caller can always see WHY, not just THAT."""

    item_id: str
    state: LineageState
    n_links: int  # total derived_from edges examined (cycles/shared ancestors counted each time seen)
    dangling: list[str] = field(default_factory=list)  # ids CONFIRMED to resolve to nothing (broken)
    cycles: list[str] = field(default_factory=list)  # ids that are their own (in-progress) ancestor
    unverified: list[str] = field(default_factory=list)  # ids outside the caller's authorized scope
    truncated: list[str] = field(default_factory=list)  # ids at max_depth with further, unexplored edges
    depth: int = 0  # how many levels of ancestry resolved before a leaf or a stop
    visited: int = 0  # distinct, resolved-and-authorized ancestors reached (cycle-safe count)

    def __post_init__(self) -> None:
        # A receipt is a record. Detaching severs the caller's alias, so a mutation after
        # construction cannot rewrite evidence that was already recorded; `frozen=True` above
        # stops the field being rebound through the receipt itself. Containers keep their
        # concrete types -- see detach_receipt_container for why (MXR-080-1876).
        object.__setattr__(self, "dangling", detach_receipt_container(self.dangling))
        object.__setattr__(self, "cycles", detach_receipt_container(self.cycles))
        object.__setattr__(self, "unverified", detach_receipt_container(self.unverified))
        object.__setattr__(self, "truncated", detach_receipt_container(self.truncated))

    @property
    def intact(self) -> bool:
        """Read-only convenience view: ``True`` iff :attr:`state` is :attr:`LineageState.INTACT`.

        Never itself the source of truth -- ``state`` is (MXR-080-0260/0261): a bare boolean cannot
        distinguish a confirmed break from an open question, which is exactly what these findings
        fix. Kept only so a caller that wants a single coarse check still gets a CORRECT one: both
        ``BROKEN`` and ``UNVERIFIED`` now read ``False`` here, where the pre-fix code could read
        ``True`` for a truncated or cross-scope chain it had never actually verified.
        """
        return self.state is LineageState.INTACT

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable lineage report."""
        return {
            "item_id": self.item_id,
            "state": self.state.value,
            "n_links": self.n_links,
            "dangling": self.dangling,
            "cycles": self.cycles,
            "unverified": self.unverified,
            "truncated": self.truncated,
            "depth": self.depth,
            "visited": self.visited,
        }


def verify_lineage(
    substrate: Substrate,
    item_id: str,
    *,
    max_depth: int = 20,
    scope: str | None = None,
    policy: AccessPolicy | None = None,
) -> LineageReport:
    """Walk ``item_id``'s ancestry via ``derived_from`` ONLY, returning a three-state verdict (cycle-safe).

    Traverses exclusively ``derived_from`` (MXR-080-0261) -- never the generic, untyped ``links`` KG
    edges (see :class:`~mixle.substrate.core.SubstrateItem`) -- so a merely-related entity can no longer
    be certified as a derivation ancestor just by existing.

    ``scope``, when given, doubles as the authorization principal (the same convention
    :mod:`~mixle.substrate.belief` uses for the identical problem): every ancestor, at every hop, is
    resolved through ``policy`` (a fresh, home-scope-plus-PUBLIC-only
    :class:`~mixle.substrate.spaces.AccessPolicy` when omitted). An ancestor outside that authorized view
    can neither ground an ``INTACT`` verdict nor be silently skipped as though absent-therefore-fine --
    it is named in :attr:`LineageReport.unverified`. An ancestor id that does not exist at all and one
    that exists but is private to a scope ``scope`` cannot read are DELIBERATELY indistinguishable here:
    both land in ``unverified``, never ``dangling`` -- telling them apart would let a caller use their
    own scoped audits as an existence oracle for another scope's content, the same side channel
    MXR-080-0237 closed for semantic search and MXR-080-0244 closed for belief evidence. ``scope=None``
    (the default) is the one deliberate exception, matching :meth:`Substrate.all`/:meth:`Substrate.search`'s
    own convention: an unrestricted caller has no boundary to hide behind, so a missing ancestor is
    unambiguously reportable as :attr:`LineageState.BROKEN` (``dangling``) -- the legacy behavior.

    ``max_depth`` bounds pathological chains, but reaching it is no longer silent (MXR-080-0260): a node
    at the cap that still has its OWN unexplored ``derived_from`` entries is named in
    :attr:`LineageReport.truncated` and the verdict is :attr:`LineageState.UNVERIFIED`, never
    ``INTACT`` -- an unvisited tail is never certified. A node at the cap with no further edges (a
    genuine leaf) is not truncated.

    A back-edge into an ancestor already on the CURRENT root-to-node path is a genuine cycle (an item
    cannot legitimately derive, even transitively, from itself) and is named in
    :attr:`LineageReport.cycles`, forcing ``BROKEN``. Reaching an already-fully-resolved ancestor via a
    DIFFERENT branch (a shared ancestor / diamond) is ordinary DAG structure, not a cycle, and is not
    re-walked.

    Precedence when a walk has more than one kind of finding: ``BROKEN`` (a confirmed defect) outranks
    ``UNVERIFIED`` (an open question) outranks ``INTACT`` -- the legacy "one break anywhere fails the
    whole chain" discipline, extended to the two new states. A missing or unauthorized ROOT item is
    handled the same way: ``BROKEN`` when unrestricted (``scope=None``), ``UNVERIFIED`` when scoped.
    """
    policy = policy if policy is not None else AccessPolicy()

    def _authorized(node: SubstrateItem) -> bool:
        return scope is None or policy.can_read(scope, node.scope)

    root = substrate.get(item_id)
    if root is None or not _authorized(root):
        if scope is None:  # unrestricted caller: a missing root is an unambiguous, confirmed break
            return LineageReport(item_id=item_id, state=LineageState.BROKEN, n_links=0, dangling=[item_id])
        # scoped caller: missing vs. exists-but-inaccessible are indistinguishable (MXR-080-0261)
        return LineageReport(item_id=item_id, state=LineageState.UNVERIFIED, n_links=0, unverified=[item_id])

    n_links = 0
    dangling: list[str] = []
    cycles: list[str] = []
    unverified: list[str] = []
    truncated: list[str] = []
    resolved: set[str] = {item_id}  # authorized, resolved ancestors actually reached -> report.visited
    settled: set[str] = set()  # ids already recorded dangling/unverified once -- dedup, no re-fetch
    max_reached = 0

    # Explicit-stack DFS (bounded by max_depth, so no Python recursion-depth concern regardless of how
    # large a caller sets it). Each frame's `path` is the full root-to-node ancestry chain of ids, so a
    # back-edge into an id still ON that path is a genuine cycle, while an id already fully RESOLVED via
    # a different branch (a diamond / shared ancestor) is legitimate DAG structure, not a cycle.
    stack: list[tuple[SubstrateItem, int, tuple[str, ...]]] = [(root, 0, (item_id,))]
    while stack:
        node, depth, path = stack.pop()
        if depth >= max_depth:
            if node.derived_from:
                truncated.append(node.id)  # edges exist beyond the cap, never inspected (MXR-080-0260)
            continue
        for target_id in node.derived_from:
            n_links += 1
            if target_id in path:
                if target_id not in cycles:
                    cycles.append(target_id)
                continue
            if target_id in resolved or target_id in settled:
                continue  # already fully accounted for via another branch
            child = substrate.get(target_id)
            authorized = child is not None and _authorized(child)
            if not authorized:
                settled.add(target_id)
                if scope is None:
                    dangling.append(target_id)  # unrestricted: a confirmed, nameable break
                else:
                    unverified.append(target_id)  # scoped: missing/inaccessible, indistinguishable
                continue
            resolved.add(target_id)
            max_reached = max(max_reached, depth + 1)
            stack.append((child, depth + 1, (*path, target_id)))

    if dangling or cycles:
        state = LineageState.BROKEN
    elif truncated or unverified:
        state = LineageState.UNVERIFIED
    else:
        state = LineageState.INTACT

    return LineageReport(
        item_id=item_id,
        state=state,
        n_links=n_links,
        dangling=dangling,
        cycles=cycles,
        unverified=unverified,
        truncated=truncated,
        depth=max_reached,
        visited=len(resolved),
    )


def audit_substrate(
    substrate: Substrate, *, scope: str | None = None, policy: AccessPolicy | None = None
) -> dict[str, Any]:
    """A knowledge-integrity sweep: how many items are intact/broken/unverified, every one named.

    MXR-080-0261: ``scope``/``policy`` are threaded into every :func:`verify_lineage` call, so a scoped
    audit (``scope`` not ``None``) can only ever certify a chain using ancestors that SAME scope is
    authorized to read. Previously this filtered its own item list by ``scope`` but then handed each id
    to an unscoped ``verify_lineage``, which walked every scope regardless -- a scoped audit could be,
    and was, validated by (or leak the internal integrity of) items it had no business seeing at all.

    Returns ``{n_items, n_intact, n_broken, n_unverified, broken: [...], unverified: [...]}``. ``broken``
    names every item with a confirmed structural defect (dangling edge and/or cycle) within its
    authorized visibility; ``unverified`` names every item whose chain could not be fully checked
    (depth-cap truncation and/or an ancestor outside ``scope``'s authorized visibility) -- kept separate
    from ``broken`` because "could not verify" is never the same claim as "confirmed defective."
    """
    items = substrate.all(scope=scope)
    broken: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for it in items:
        report = verify_lineage(substrate, it.id, scope=scope, policy=policy)
        if report.state is LineageState.BROKEN:
            broken.append({"item_id": it.id, "dangling": report.dangling, "cycles": report.cycles})
        elif report.state is LineageState.UNVERIFIED:
            unverified.append({"item_id": it.id, "unverified": report.unverified, "truncated": report.truncated})
    return {
        "n_items": len(items),
        "n_intact": len(items) - len(broken) - len(unverified),
        "n_broken": len(broken),
        "n_unverified": len(unverified),
        "broken": broken,
        "unverified": unverified,
    }
