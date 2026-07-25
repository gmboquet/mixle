"""Ontology objects for typed constraints on knowledge graphs.

A knowledge graph without a schema will happily assert ``(paris, employs, france)``. An
:class:`Ontology` is the typed contract that rules such triples out *structurally*: a class hierarchy,
relation signatures (``employs: Organization -> Person``), per-relation axioms (functional, symmetric,
asymmetric, irreflexive), and disjoint-class declarations. :meth:`Ontology.check_triple` names every
violation of one assertion; :meth:`Ontology.check_graph` audits a whole triple set, including the
cross-triple axioms (a functional relation asserted with two different tails).

The same contract turns a fitted KG embedding into an ontology-constrained
distribution: :class:`OntologyConstrainedKG` wraps a
:class:`~mixle.stats.graphs.knowledge_graph.KnowledgeGraphDistribution` and masks the tail posterior to
range-conforming entities, renormalizing -- so the model literally cannot place probability on a triple
the ontology forbids. Constrained extraction or decoding can apply the same
mask before accepting generated triples.

Everything is symbolic and dependency-free: entities/relations are strings, entity types are supplied
as a ``{entity: class}`` map (the entity-linking output). Violations are named, never silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

AXIOMS = ("functional", "inverse_functional", "symmetric", "asymmetric", "transitive", "irreflexive")


@dataclass
class Ontology:
    """A typed schema over knowledge: class hierarchy + relation signatures + axioms + disjointness."""

    classes: dict[str, str | None] = field(default_factory=dict)  # class -> parent class (None = root)
    relations: dict[str, tuple[str, str]] = field(default_factory=dict)  # relation -> (domain, range)
    axioms: dict[str, set[str]] = field(default_factory=dict)  # relation -> subset of AXIOMS
    disjoint: list[tuple[str, str]] = field(default_factory=list)  # mutually-exclusive class pairs

    # -- construction (chainable) ------------------------------------------------------------------
    def add_class(self, name: str, parent: str | None = None) -> Ontology:
        """Add a NEW ontology class, optionally under a known parent.

        MXR-080-0297: a duplicate ``name`` is rejected rather than silently overwriting the
        existing class's parent -- the prior behavior let a second, differently-parented
        ``add_class(name, other_parent)`` retroactively change what every relation/disjointness
        declaration mentioning ``name`` means, with no trace that a redefinition happened. Use
        :meth:`replace_class` to deliberately reparent an existing class.

        Raises:
            ValueError: ``name`` is already registered, or ``parent`` is not a known class.
        """
        if name in self.classes:
            raise ValueError(
                f"class {name!r} is already registered (parent={self.classes[name]!r}); "
                f"use replace_class({name!r}, ...) to deliberately reparent it"
            )
        if parent is not None and parent not in self.classes:
            raise ValueError(f"unknown parent class {parent!r}")
        self.classes[name] = parent
        return self

    def replace_class(self, name: str, parent: str | None = None) -> Ontology:
        """Deliberately reparent an existing class.

        MXR-080-0297: unlike :meth:`add_class` (which rejects a duplicate name outright), this
        exists specifically to allow redefining an existing class's parent -- but a ``parent`` that
        IS ``name``, or that is already a descendant of ``name`` in the CURRENT hierarchy, is
        rejected here, before the mutation, rather than silently installed and merely limped around
        later by :meth:`is_a`'s cycle guard (a partial-answer, quietly-wrong result).

        Raises:
            KeyError: ``name`` is not already registered.
            ValueError: ``parent`` is not a known class, or reparenting would create a self-loop or
                a cycle.
        """
        if name not in self.classes:
            raise KeyError(f"class {name!r} is not registered; use add_class({name!r}, ...) instead")
        if parent is not None:
            if parent not in self.classes:
                raise ValueError(f"unknown parent class {parent!r}")
            if parent == name or self.is_a(parent, name):
                raise ValueError(
                    f"reparenting {name!r} under {parent!r} would create a cycle "
                    f"({parent!r} is {name!r} or already a descendant of it)"
                )
        self.classes[name] = parent
        return self

    def add_relation(self, name: str, domain: str, range_: str, *axioms: str) -> Ontology:
        """Add a NEW relation with domain, range, and optional ontology axioms.

        MXR-080-0297: a duplicate ``name`` is rejected rather than silently overwriting the
        existing relation's signature and axioms. Use :meth:`replace_relation` to deliberately
        redefine an existing one.

        Raises:
            ValueError: ``name`` is already registered, ``domain``/``range_`` is not a known class,
                or an axiom is not one of :data:`AXIOMS`.
        """
        if name in self.relations:
            raise ValueError(
                f"relation {name!r} is already registered (signature={self.relations[name]!r}); "
                f"use replace_relation({name!r}, ...) to deliberately redefine it"
            )
        for c in (domain, range_):
            if c not in self.classes:
                raise ValueError(f"unknown class {c!r}; add_class it first")
        bad = [a for a in axioms if a not in AXIOMS]
        if bad:
            raise ValueError(f"unknown axiom(s) {bad}; known: {AXIOMS}")
        self.relations[name] = (domain, range_)
        self.axioms[name] = set(axioms)
        return self

    def replace_relation(self, name: str, domain: str, range_: str, *axioms: str) -> Ontology:
        """Deliberately redefine an existing relation's domain, range, and/or axioms.

        MXR-080-0297: the explicit, differently-named counterpart to :meth:`add_relation`'s
        reject-on-duplicate.

        Raises:
            KeyError: ``name`` is not already registered.
            ValueError: ``domain``/``range_`` is not a known class, or an axiom is not one of
                :data:`AXIOMS`.
        """
        if name not in self.relations:
            raise KeyError(f"relation {name!r} is not registered; use add_relation({name!r}, ...) instead")
        for c in (domain, range_):
            if c not in self.classes:
                raise ValueError(f"unknown class {c!r}; add_class it first")
        bad = [a for a in axioms if a not in AXIOMS]
        if bad:
            raise ValueError(f"unknown axiom(s) {bad}; known: {AXIOMS}")
        self.relations[name] = (domain, range_)
        self.axioms[name] = set(axioms)
        return self

    def add_disjoint(self, a: str, b: str) -> Ontology:
        """Declare two classes mutually exclusive.

        MXR-080-0297: an unknown class reference, or declaring a class disjoint with ITSELF, is
        rejected rather than silently accepted -- ``add_disjoint(x, x)`` previously made
        :meth:`check_triple` flag every single instance of ``x`` as violating disjointness against
        itself, and an unknown class reference could never match any real entity, silently
        defeating the declaration instead of failing loudly at the typo.

        Raises:
            ValueError: ``a`` or ``b`` is not a known class, or ``a == b``.
        """
        for c in (a, b):
            if c not in self.classes:
                raise ValueError(f"unknown class {c!r}; add_class it first")
        if a == b:
            raise ValueError(f"a class cannot be declared disjoint with itself ({a!r})")
        self.disjoint.append((a, b))
        return self

    # -- the hierarchy -----------------------------------------------------------------------------
    def is_a(self, cls: str, ancestor: str) -> bool:
        """Whether ``cls`` is ``ancestor`` or a descendant of it (walks the parent chain).

        MXR-080-0297: construction (:meth:`add_class`/:meth:`replace_class`) now rejects every path
        that could install a cycle, so this walk should never see one. As a defense-in-depth check
        (e.g. against ``.classes`` being mutated directly, bypassing the API) this RAISES if a cycle
        is nonetheless encountered, rather than silently stopping and returning a partial, possibly
        wrong answer -- a malformed schema must never look like an ordinary "not an ancestor".
        """
        cur: str | None = cls
        seen: set[str] = set()
        while cur is not None:
            if cur == ancestor:
                return True
            if cur in seen:
                raise RuntimeError(
                    f"cycle detected in class hierarchy while walking from {cls!r} toward {ancestor!r} "
                    f"({cur!r} is its own ancestor); the schema is malformed -- this should be "
                    f"unreachable when classes are only ever added via add_class/replace_class"
                )
            seen.add(cur)
            cur = self.classes.get(cur)
        return False

    def _conforms(self, entity_cls: str | None, required: str) -> bool:
        return entity_cls is not None and self.is_a(entity_cls, required)

    def _pair_violated(self, cls: str, a: str, b: str) -> bool:
        """Whether ``cls`` is (transitively) both halves of the declared-disjoint pair ``(a, b)``."""
        return self.is_a(cls, a) and self.is_a(cls, b)

    def _violates_disjoint(self, cls: str) -> bool:
        """Whether ``cls`` is both halves of ANY declared-disjoint pair.

        Shared by :meth:`check_triple` and :class:`OntologyConstrainedKG`'s completion mask
        (MXR-080-0299) so the two can never independently drift on what "disjoint" means.
        """
        return any(self._pair_violated(cls, a, b) for a, b in self.disjoint)

    # -- checking one assertion ---------------------------------------------------------------------
    def check_triple(self, h: str, r: str, t: str, types: dict[str, str]) -> list[str]:
        """Every named violation of ``(h, r, t)`` given entity ``types`` ({} means unconstrained)."""
        out: list[str] = []
        sig = self.relations.get(r)
        if sig is None:
            return [f"unknown relation {r!r}"]
        domain, range_ = sig
        h_cls, t_cls = types.get(h), types.get(t)
        if h_cls is not None and not self._conforms(h_cls, domain):
            out.append(f"domain: {h!r} is {h_cls!r}, {r!r} requires {domain!r}")
        if t_cls is not None and not self._conforms(t_cls, range_):
            out.append(f"range: {t!r} is {t_cls!r}, {r!r} requires {range_!r}")
        ax = self.axioms.get(r, set())
        if "irreflexive" in ax and h == t:
            out.append(f"irreflexive: {r!r} cannot relate {h!r} to itself")
        for a, b in self.disjoint:
            for e, cls in ((h, h_cls), (t, t_cls)):
                if cls is not None and self._pair_violated(cls, a, b):
                    out.append(f"disjoint: {e!r} is both {a!r} and {b!r}")
        return out

    # -- auditing a graph (cross-triple axioms live here) --------------------------------------------
    def check_graph(self, triples: Any, types: dict[str, str]) -> dict[str, Any]:
        """Audit a triple set: per-triple violations plus the cross-triple axioms (functional/symmetric/
        asymmetric/transitive).

        Returns ``{consistent, n_triples, violations: [{triple, problems}]}`` -- every problem named.
        Every cross-triple (graph-level) violation also carries an ``"axiom"`` key naming which axiom
        it is (the per-triple violations sourced from :meth:`check_triple` do not, since one of those
        can bundle several different kinds of problem for a single triple) -- :meth:`filter_triples`
        relies on this to apply a deterministic policy per axiom kind (MXR-080-0298).
        """
        triple_list = [tuple(t) for t in triples]
        violations: list[dict[str, Any]] = []
        for tr in triple_list:
            probs = self.check_triple(*tr, types)
            if probs:
                violations.append({"triple": tr, "problems": probs})

        by_hr: dict[tuple[str, str], set[str]] = {}
        by_rt: dict[tuple[str, str], set[str]] = {}
        present = set(triple_list)
        for h, r, t in triple_list:
            by_hr.setdefault((h, r), set()).add(t)
            by_rt.setdefault((r, t), set()).add(h)
        for r, ax in self.axioms.items():
            if "functional" in ax:
                for (h, rr), tails in by_hr.items():
                    if rr == r and len(tails) > 1:
                        violations.append(
                            {
                                "triple": (h, r, "*"),
                                "axiom": "functional",
                                "problems": [f"functional: {r!r} has {sorted(tails)} tails for {h!r}"],
                            }
                        )
            if "inverse_functional" in ax:
                for (rr, t), heads in by_rt.items():
                    if rr == r and len(heads) > 1:
                        violations.append(
                            {
                                "triple": ("*", r, t),
                                "axiom": "inverse_functional",
                                "problems": [f"inverse_functional: {sorted(heads)} heads for {t!r}"],
                            }
                        )
            if "asymmetric" in ax:
                for h, rr, t in triple_list:
                    if rr == r and (t, r, h) in present and h != t:
                        violations.append(
                            {
                                "triple": (h, r, t),
                                "axiom": "asymmetric",
                                "problems": ["asymmetric: both directions asserted"],
                            }
                        )
            if "symmetric" in ax:
                for h, rr, t in triple_list:
                    if rr == r and h != t and (t, r, h) not in present:
                        violations.append(
                            {
                                "triple": (h, r, t),
                                "axiom": "symmetric",
                                "problems": [f"symmetric: {r!r} has {h!r}->{t!r} without the reverse {t!r}->{h!r}"],
                            }
                        )
            if "transitive" in ax:
                for h, rr, m in triple_list:
                    if rr != r:
                        continue
                    for m2, rr2, t in triple_list:
                        if rr2 == r and m2 == m and (h, r, t) not in present:
                            violations.append(
                                {
                                    "triple": (h, r, t),
                                    "axiom": "transitive",
                                    "problems": [
                                        f"transitive: {r!r} has {h!r}->{m!r}->{t!r} without the closure edge "
                                        f"{h!r}->{t!r}"
                                    ],
                                }
                            )
        return {"consistent": not violations, "n_triples": len(triple_list), "violations": violations}

    def _resolve_graph_conflicts(
        self, kept: list[tuple], types: dict[str, str]
    ) -> tuple[list[tuple], list[dict[str, Any]]]:
        """Deterministically resolve SET-level (cross-triple) axiom violations among per-triple-valid
        ``kept`` triples (MXR-080-0298).

        :meth:`check_triple` only ever sees one triple at a time, so it cannot catch a functional
        relation asserted with two different tails, two different heads asserted for one inverse-
        functional tail, both directions of an asymmetric relation, or an incomplete symmetric/
        transitive closure -- those are properties of the whole candidate SET. This re-audits
        ``kept`` with :meth:`check_graph` and, for every cross-triple violation, applies one fixed
        policy per axiom kind:

        * functional / inverse_functional / asymmetric (genuine VALUE conflicts: more than one
          admissible filler for one slot, or both directions of an asymmetric relation asserted at
          once): keep the FIRST triple by position in ``kept`` among the conflicting ones, reject
          the rest. (A caller that wants "highest-scored wins" gets it for free by passing ``kept``
          in descending-confidence order; :func:`constrained_decode` does exactly this for its
          aggregate, confidence-sorted fact set.)
        * symmetric (a triple's required reverse edge was never asserted): the ontology promises
          symmetric relations only ever appear in pairs, so a solitary direction cannot stand alone
          as validated -- it is rejected too.
        * transitive (the required closure edge was never asserted): the two premise triples are
          each individually valid and are NOT removed from ``kept`` -- only the missing closure fact
          itself is reported, since it names something that was never in the candidate set at all.

        Returns ``(new_kept, extra_rejected)``. Every triple removed from ``kept``, and every
        unconfirmed transitive closure, is named in ``extra_rejected`` with its reason.
        """
        if len(kept) < 2:
            return kept, []  # no cross-triple axiom can be violated by 0 or 1 triples
        report = self.check_graph(kept, types)
        if report["consistent"]:
            return kept, []

        order = {tr: i for i, tr in enumerate(kept)}
        kept_set = set(kept)
        losers: dict[tuple, set[str]] = {}
        missing: dict[tuple, set[str]] = {}

        for v in report["violations"]:
            axiom = v.get("axiom")
            if axiom is None:
                continue  # per-triple violation; filter_triples already excluded it from `kept`
            reason = v["problems"][0]
            h, r, t = v["triple"]
            if axiom == "functional":  # (h, r, "*"): every tail asserted for this (h, r)
                group = sorted((tr for tr in kept if tr[0] == h and tr[1] == r), key=order.get)
                for loser in group[1:]:
                    losers.setdefault(loser, set()).add(reason)
            elif axiom == "inverse_functional":  # ("*", r, t): every head asserted for this (r, t)
                group = sorted((tr for tr in kept if tr[1] == r and tr[2] == t), key=order.get)
                for loser in group[1:]:
                    losers.setdefault(loser, set()).add(reason)
            elif axiom == "asymmetric":  # (h, r, t) with (t, r, h) also present
                pair = [p for p in ((h, r, t), (t, r, h)) if p in kept_set]
                pair.sort(key=order.get)
                for loser in pair[1:]:
                    losers.setdefault(loser, set()).add(reason)
            elif axiom == "symmetric":  # (h, r, t) itself, missing its reverse
                if (h, r, t) in kept_set:
                    losers.setdefault((h, r, t), set()).add(reason)
            elif axiom == "transitive":  # (h, r, t) is the MISSING closure edge, not in kept
                missing.setdefault((h, r, t), set()).add(reason)

        if not losers and not missing:
            return kept, []
        new_kept = [tr for tr in kept if tr not in losers]
        extra_rejected = [{"triple": tr, "problems": sorted(reasons)} for tr, reasons in losers.items()]
        extra_rejected += [{"triple": tr, "problems": sorted(reasons)} for tr, reasons in missing.items()]
        return new_kept, extra_rejected

    def filter_triples(self, triples: Any, types: dict[str, str]) -> tuple[list[tuple], list[dict[str, Any]]]:
        """Split triples into (kept, rejected-with-reasons) -- the decode mask.

        MXR-080-0298: per-triple consistency (:meth:`check_triple`) alone is not enough -- it
        cannot see that a functional relation was asserted with two different tails, or any other
        cross-triple axiom violation, so ``kept`` could previously contain a set that is globally
        inconsistent even though every individual triple in it looks fine alone. After the
        per-triple pass, the accepted set is re-validated AS A GRAPH and any set-level conflict is
        resolved by :meth:`_resolve_graph_conflicts`'s deterministic policy; every triple that
        policy withholds is added to ``rejected`` with its reason, same as a per-triple violation.
        """
        kept: list[tuple] = []
        rejected: list[dict[str, Any]] = []
        for tr in (tuple(t) for t in triples):
            probs = self.check_triple(*tr, types)
            if probs:
                rejected.append({"triple": tr, "problems": probs})
            else:
                kept.append(tr)
        kept, extra_rejected = self._resolve_graph_conflicts(kept, types)
        rejected.extend(extra_rejected)
        return kept, rejected


class OntologyConstrainedKG:
    """A fitted KG embedding, typed by an ontology: probability mass only on schema-consistent triples.

    Wraps a :class:`~mixle.stats.graphs.knowledge_graph.KnowledgeGraphDistribution` (entities and
    relations as integer indices) together with the symbolic ontology and the index<->name maps. The
    tail posterior is masked to entities that satisfy the FULL triple contract for the completion --
    range, the head's domain, irreflexivity, and disjointness (MXR-080-0299) -- and renormalized, so
    completion can never propose an ontology-violating tail -- ``Graph(ontology)`` as a distribution.
    """

    def __init__(
        self,
        kg: Any,
        ontology: Ontology,
        *,
        entities: list[str],
        relations: list[str],
        types: dict[str, str],
    ) -> None:
        self.kg = kg
        self.ontology = ontology
        self.entities = list(entities)
        self.relations = list(relations)
        self.types = dict(types)
        self._eidx = {e: i for i, e in enumerate(self.entities)}
        self._ridx = {r: i for i, r in enumerate(self.relations)}

    def _candidate_mask(self, head: str, relation: str) -> np.ndarray:
        """Which entities are POSITIVELY confirmed admissible as the tail of ``(head, relation, ?)``
        -- the full triple contract, not just range (MXR-080-0299): range, ``head``'s domain,
        irreflexivity, and disjointness.

        Completion is the mirror image of :meth:`Ontology.check_triple`'s filtering: check_triple is
        permissive (an untyped entity is "unconstrained", not proven invalid, because its job is to
        catch CONFIRMED violations in triples someone else already vetted); a completion mask must be
        restrictive (an untyped candidate gets ZERO mass, not the benefit of the doubt, because here
        we are the ones proposing to assert it as fact).
        """
        sig = self.ontology.relations.get(relation)
        if sig is None:
            raise KeyError(f"unknown relation {relation!r}")
        domain, range_ = sig
        axioms = self.ontology.axioms.get(relation, set())
        n = len(self.entities)

        h_cls = self.types.get(head)
        if h_cls is None or not self.ontology.is_a(h_cls, domain) or self.ontology._violates_disjoint(h_cls):
            return np.zeros(n, dtype=bool)  # head itself is not a confirmed, disjoint-clean `domain`

        irreflexive = "irreflexive" in axioms
        ok = np.zeros(n, dtype=bool)
        for i, cand in enumerate(self.entities):
            if irreflexive and cand == head:
                continue  # MXR-080-0299: an irreflexive relation cannot complete to the head itself
            t_cls = self.types.get(cand)
            if t_cls is None or not self.ontology.is_a(t_cls, range_):
                continue
            if self.ontology._violates_disjoint(t_cls):
                continue
            ok[i] = True
        return ok

    def tail_posterior(self, head: str, relation: str) -> dict[str, float]:
        """``p(tail | head, relation)`` over ONLY entities satisfying the full triple contract for
        this completion (renormalized).

        MXR-080-0299: previously masked on the candidate tail's range class alone, so e.g. an
        irreflexive relation's self-completion competed for probability mass on equal footing with
        every genuinely valid tail -- for one such relation, self-completion was assigned the
        LARGEST probability of any candidate. The mask now also checks the head's domain,
        irreflexivity, and disjointness.
        """
        lp = self.kg.tail_log_posterior(self._eidx[head], self._ridx[relation])
        mask = self._candidate_mask(head, relation)
        if not mask.any():
            return {}
        p = np.exp(lp - lp.max())
        p[~mask] = 0.0
        total = p.sum()
        if total <= 0:
            return {}
        p /= total
        return {self.entities[i]: float(p[i]) for i in np.flatnonzero(mask)}

    def complete(self, head: str, relation: str) -> tuple[str, float] | None:
        """The most probable ontology-consistent tail (or None when the range admits no entity)."""
        post = self.tail_posterior(head, relation)
        if not post:
            return None
        best = max(post.items(), key=lambda kv: kv[1])
        return best


@dataclass
class ConstrainedDecode:
    """The result of ontology-constrained LLM decoding: what survived, what the schema rejected, and why."""

    facts: list[tuple[Any, float]]  # accepted (triple, confidence) pairs, best-first
    rejected: list[dict[str, Any]]  # ontology-violating triples with named reasons
    below_floor: list[tuple[Any, float]]  # consistent but under-confident facts (withheld, not asserted)
    n_samples: int

    def asserted(self) -> list[Any]:
        """Return facts that passed constraints and confidence floor."""
        return [t for t, _ in self.facts]

    def as_dict(self) -> dict[str, Any]:
        """Return constrained decoding results as JSON-compatible data."""
        return {
            "facts": [{"triple": list(t), "confidence": round(c, 4)} for t, c in self.facts],
            "rejected": self.rejected,
            "below_floor": [{"triple": list(t), "confidence": round(c, 4)} for t, c in self.below_floor],
            "n_samples": self.n_samples,
        }


def constrained_decode(
    llm: Any,
    prompt: str,
    ontology: Ontology,
    types: dict[str, str],
    *,
    n: int | None = None,
    floor: float = 0.5,
    calibrator: Any = None,
) -> ConstrainedDecode:
    """Decode only schema-consistent facts above a confidence floor.

    Samples ``llm`` (a :class:`~mixle.reason.graph_llm.GraphLLM`) ``n`` times, masks every sampled
    graph through :meth:`Ontology.filter_triples` (violating triples are rejected with named reasons),
    then marginalizes the constrained graphs into a
    :class:`~mixle.reason.graph_llm.GraphDistribution` and keeps only facts whose edge marginal clears
    ``floor`` -- the calibrated confidence floor (pass a fitted ``calibrator`` from
    :func:`~mixle.reason.graph_llm.fit_fact_calibrator` to apply the floor on calibrated truth
    probability rather than the raw marginal). Consistent-but-underconfident facts are reported as
    withheld, never silently dropped: the decode says what it refused to assert and why.

    MXR-080-0298: per-sample filtering only guarantees each INDIVIDUAL sampled graph is internally
    consistent; the published ``facts`` set is built by thresholding each triple's marginal
    independently, so two triples that never co-occurred in any single sample (e.g. two different
    tails of a functional relation, each confident in a disjoint subset of samples) could previously
    both clear the floor and be asserted together -- globally inconsistent while still labeled
    "ontology-constrained". ``facts`` is re-validated as a graph after confidence-sorting, so
    :meth:`Ontology.filter_triples`'s "first in the list wins" conflict policy resolves to
    "highest-confidence wins" at this aggregate stage; every casualty is named in ``rejected``.
    """
    import numpy as np  # noqa: F811 - local so the module stays import-light

    from mixle.reason.graph_llm import canonical_graph

    graphs = llm.sample_graphs(prompt, n)
    n_samples = len(graphs)
    constrained: list[frozenset] = []
    rejected_all: dict[tuple, dict[str, Any]] = {}
    for g in graphs:
        kept, rejected = ontology.filter_triples(g, types)
        for rj in rejected:
            rejected_all.setdefault(tuple(rj["triple"]), rj)
        constrained.append(canonical_graph(kept))

    dist = llm.distribution(prompt, graphs=constrained)
    marginals = dist.edge_marginals()

    def confidence(marg: float) -> float:
        if calibrator is None:
            return float(marg)
        out = calibrator.predict(np.asarray([marg]))
        return float(np.asarray(out).reshape(-1)[0])

    facts: list[tuple[Any, float]] = []
    withheld: list[tuple[Any, float]] = []
    for triple, marg in marginals.items():
        conf = confidence(float(marg))
        (facts if conf >= floor else withheld).append((triple, conf))
    facts.sort(key=lambda tc: -tc[1])
    withheld.sort(key=lambda tc: -tc[1])

    # MXR-080-0298: catch conflicts across the AGGREGATE fact set that no single sample exhibited.
    surviving, aggregate_conflicts = ontology._resolve_graph_conflicts([t for t, _ in facts], types)
    if aggregate_conflicts:
        surviving_set = set(surviving)
        facts = [(t, c) for t, c in facts if t in surviving_set]
        for entry in aggregate_conflicts:
            rejected_all.setdefault(tuple(entry["triple"]), entry)

    return ConstrainedDecode(
        facts=facts, rejected=list(rejected_all.values()), below_floor=withheld, n_samples=n_samples
    )
