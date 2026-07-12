"""The typed update graph as an EXECUTION planner, not just a semantic checker.

``plan_execution`` compiles a model/estimator pair and turns the graph's weakest-link contracts into
the concrete knobs ``mixle.inference.optimize`` accepts:

- **compute band** -> ``precision``: a ``FLOAT32_ELIGIBLE`` tree plans ``precision="minimal"`` (the
  runtime planner still applies its data-side checks -- the band answers "may the validated reduced
  kernel ever be used here", the data decides "should it be this time"); a ``FLOAT64`` tree plans
  full precision and says which node dropped the band.
- **convergence certificate** -> ``monotone``: ``MONOTONE_CERTIFIED`` plans the strict generalized-EM
  gate; ``BEST_VISITED`` / ``ROBBINS_MONRO_SCHEDULE`` plan best-visited selection (``monotone=False``);
  ``UNKNOWN`` defers to ``optimize``'s own structural resolution (``monotone=None``) rather than
  guessing.

Compilation failures keep raising exactly as before -- that IS the fail-before-fitting boundary.
Limits of the typed *adapters* (narrower than compilation, per this package's README) surface as
``adapter_notes``: they do not block ``optimize`` (which has its own execution paths), they tell you
what ``run_typed_mixture_em`` would refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mixle.experimental.typed_runtime.compiler import compile_update_graph
from mixle.experimental.typed_runtime.contracts import ComputeBand, ConvergenceCertificate
from mixle.experimental.typed_runtime.graph import UpdateGraph

__all__ = ["ExecutionPlan", "plan_execution"]


@dataclass(frozen=True)
class ExecutionPlan:
    """Contract-derived execution knobs plus the receipts that justify them."""

    precision: str | None  # "minimal" when the tree's compute band permits the reduced kernel
    monotone: bool | None  # True = strict GEM gate; False = best-visited; None = defer to optimize
    notes: tuple[str, ...] = ()
    adapter_notes: tuple[str, ...] = ()  # what the (narrower) typed execution adapters would refuse
    blockers: tuple[str, ...] = ()
    graph: UpdateGraph | None = field(default=None, repr=False, compare=False)

    @property
    def optimize_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``mixle.inference.optimize``; raises if the plan carries blockers."""
        if self.blockers:
            raise RuntimeError("execution plan carries blockers:\n" + "\n".join(f"- {b}" for b in self.blockers))
        kwargs: dict[str, Any] = {"monotone": self.monotone}
        if self.precision is not None:
            kwargs["precision"] = self.precision
        return kwargs

    def explain(self) -> str:
        lines = [
            f"precision: {self.precision or 'float64 (default)'}",
            f"monotone: {self.monotone if self.monotone is not None else 'defer to optimize'}",
        ]
        lines += [f"note: {n}" for n in self.notes]
        lines += [f"adapter-note: {n}" for n in self.adapter_notes]
        lines += [f"BLOCKER: {b}" for b in self.blockers]
        if self.graph is not None:
            lines += ["", self.graph.explain()]
        return "\n".join(lines)


def _adapter_notes(model: Any) -> tuple[str, ...]:
    """What ``run_typed_mixture_em`` (execution narrower than compilation) would refuse for ``model``."""
    notes: list[str] = []
    components = getattr(model, "components", None)
    if components is not None:
        seen: dict[int, int] = {}
        for index, component in enumerate(components):
            first = seen.setdefault(id(component), index)
            if first != index:
                notes.append(
                    "typed local mixture EM would refuse this model: components %d and %d are the same "
                    "object, and shared components are not yet split into a joint proposal (optimize() "
                    "handles them through the full-tree path)." % (first, index)
                )
                break
    return tuple(notes)


def plan_execution(model: Any, estimator: Any, *, nobs: int) -> ExecutionPlan:
    """Compile ``model``/``estimator`` and derive optimize-ready execution knobs from the contracts.

    Raises whatever :func:`compile_update_graph` raises for an uncompilable pair -- unsupported
    semantics fail here, before any fitting.
    """
    graph = compile_update_graph(model, estimator, nobs=nobs)
    notes: list[str] = []

    band = graph.compute_band
    if band is ComputeBand.FLOAT32_ELIGIBLE:
        precision: str | None = "minimal"
        notes.append(
            "every leaf family is in the validated float32 set and the tree is fusible; the runtime "
            "precision planner still applies its data-side checks before committing."
        )
    else:
        precision = None
        dropped = [n.node_id for n in graph.nodes if n.contract.compute_band is not ComputeBand.FLOAT32_ELIGIBLE]
        notes.append("compute band is float64 (weakest link: %s)." % ", ".join(dropped[:4]))

    certificate = graph.convergence_certificate
    if certificate is ConvergenceCertificate.MONOTONE_CERTIFIED:
        monotone: bool | None = True
        notes.append("every update is monotone-certified: planning the strict generalized-EM gate.")
    elif certificate in (ConvergenceCertificate.BEST_VISITED, ConvergenceCertificate.ROBBINS_MONRO_SCHEDULE):
        monotone = False
        notes.append(
            "weakest certificate is %s: planning best-visited selection (non-monotone trajectory, "
            "best outer-objective state retained)." % certificate.value
        )
    else:
        monotone = None
        notes.append("weakest certificate is unknown: deferring the gate to optimize()'s own resolution.")

    return ExecutionPlan(
        precision=precision,
        monotone=monotone,
        notes=tuple(notes),
        adapter_notes=_adapter_notes(model),
        graph=graph,
    )
