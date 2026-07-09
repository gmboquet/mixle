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
        """Add an ontology class, optionally under a known parent."""
        if parent is not None and parent not in self.classes:
            raise ValueError(f"unknown parent class {parent!r}")
        self.classes[name] = parent
        return self

    def add_relation(self, name: str, domain: str, range_: str, *axioms: str) -> Ontology:
        """Add a relation with domain, range, and optional ontology axioms."""
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
        """Declare two classes mutually exclusive."""
        self.disjoint.append((a, b))
        return self

    # -- the hierarchy -----------------------------------------------------------------------------
    def is_a(self, cls: str, ancestor: str) -> bool:
        """Whether ``cls`` is ``ancestor`` or a descendant of it (walks the parent chain)."""
        cur: str | None = cls
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            if cur == ancestor:
                return True
            seen.add(cur)
            cur = self.classes.get(cur)
        return False

    def _conforms(self, entity_cls: str | None, required: str) -> bool:
        return entity_cls is not None and self.is_a(entity_cls, required)

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
                if cls is not None and self.is_a(cls, a) and self.is_a(cls, b):
                    out.append(f"disjoint: {e!r} is both {a!r} and {b!r}")
        return out

    # -- auditing a graph (cross-triple axioms live here) --------------------------------------------
    def check_graph(self, triples: Any, types: dict[str, str]) -> dict[str, Any]:
        """Audit a triple set: per-triple violations plus the cross-triple axioms (functional/asymmetric).

        Returns ``{consistent, n_triples, violations: [{triple, problems}]}`` -- every problem named."""
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
                                "problems": [f"functional: {r!r} has {sorted(tails)} tails for {h!r}"],
                            }
                        )
            if "inverse_functional" in ax:
                for (rr, t), heads in by_rt.items():
                    if rr == r and len(heads) > 1:
                        violations.append(
                            {
                                "triple": ("*", r, t),
                                "problems": [f"inverse_functional: {sorted(heads)} heads for {t!r}"],
                            }
                        )
            if "asymmetric" in ax:
                for h, rr, t in triple_list:
                    if rr == r and (t, r, h) in present and h != t:
                        violations.append({"triple": (h, r, t), "problems": ["asymmetric: both directions asserted"]})
        return {"consistent": not violations, "n_triples": len(triple_list), "violations": violations}

    def filter_triples(self, triples: Any, types: dict[str, str]) -> tuple[list[tuple], list[dict[str, Any]]]:
        """Split triples into (kept, rejected-with-reasons) by per-triple consistency -- the decode mask."""
        kept: list[tuple] = []
        rejected: list[dict[str, Any]] = []
        for tr in (tuple(t) for t in triples):
            probs = self.check_triple(*tr, types)
            if probs:
                rejected.append({"triple": tr, "problems": probs})
            else:
                kept.append(tr)
        return kept, rejected


class OntologyConstrainedKG:
    """A fitted KG embedding, typed by an ontology: probability mass only on schema-consistent triples.

    Wraps a :class:`~mixle.stats.graphs.knowledge_graph.KnowledgeGraphDistribution` (entities and
    relations as integer indices) together with the symbolic ontology and the index<->name maps. The
    tail posterior is masked to entities whose class conforms to the relation's range and renormalized,
    so completion can never propose an ontology-violating tail -- ``Graph(ontology)`` as a distribution.
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

    def _range_mask(self, relation: str) -> np.ndarray:
        sig = self.ontology.relations.get(relation)
        if sig is None:
            raise KeyError(f"unknown relation {relation!r}")
        _, range_ = sig
        ok = np.zeros(len(self.entities), dtype=bool)
        for i, e in enumerate(self.entities):
            cls = self.types.get(e)
            ok[i] = cls is not None and self.ontology.is_a(cls, range_)
        return ok

    def tail_posterior(self, head: str, relation: str) -> dict[str, float]:
        """``p(tail | head, relation)`` over ONLY the range-conforming entities (renormalized)."""
        lp = self.kg.tail_log_posterior(self._eidx[head], self._ridx[relation])
        mask = self._range_mask(relation)
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
    return ConstrainedDecode(
        facts=facts, rejected=list(rejected_all.values()), below_floor=withheld, n_samples=n_samples
    )
