"""Side-effect-free compiler from a model tree to a typed update graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import (
    ArtifactKind,
    ComputeBand,
    ConsistencyRequirement,
    ContractEvidenceKind,
    ConvergenceCertificate,
    CostEstimate,
    CurvatureKind,
    MergeLaw,
    ObjectiveKind,
    StateSemantics,
    UpdateContract,
    UpdateKind,
)
from mixle.experimental.typed_runtime.graph import DependencyEdge, UpdateGraph, UpdateNode
from mixle.experimental.typed_runtime.measurement import MeasurementCatalog
from mixle.stats.compute.pdist import ParameterEstimator, ProbabilityDistribution


@dataclass(frozen=True)
class _Child:
    attr: str
    label: str
    value: Any
    index: Any = None


@dataclass(frozen=True)
class _RegisteredContract:
    model_type: type[Any]
    estimator_type: type[Any] | None
    contract: UpdateContract


class ContractRegistry:
    """Explicit contract adapters for models that cannot declare a hook.

    Registries are caller-owned rather than process-global. Compiling a test or
    plugin therefore cannot silently change another run's semantics.
    """

    def __init__(self) -> None:
        self._entries: list[_RegisteredContract] = []

    def register(
        self,
        model_type: type[Any],
        contract: UpdateContract,
        *,
        estimator_type: type[Any] | None = None,
    ) -> None:
        """Register one immutable, already constructed contract for a model/estimator pair."""

        if not isinstance(model_type, type):
            raise TypeError("model_type must be a type.")
        if estimator_type is not None and not isinstance(estimator_type, type):
            raise TypeError("estimator_type must be a type or None.")
        _validate_contract(contract)
        self._entries.append(_RegisteredContract(model_type, estimator_type, contract))

    def resolve(self, model: Any, estimator: Any | None) -> UpdateContract | None:
        """Resolve the most recently registered matching adapter."""

        for entry in reversed(self._entries):
            if not _nominal_instance(model, entry.model_type):
                continue
            if entry.estimator_type is not None and not _nominal_instance(estimator, entry.estimator_type):
                continue
            return entry.contract
        return None


def _nominal_instance(value: Any, expected_type: type[Any]) -> bool:
    """Test ordinary inheritance without invoking a metaclass ``__instancecheck__`` hook."""
    if value is None:
        return False
    try:
        return any(cls is expected_type for cls in type.__getattribute__(type(value), "__mro__"))
    except (AttributeError, TypeError):
        return False


def _instance_state(value: Any) -> dict[str, Any]:
    """Return a real instance dictionary without dispatching to an overridden ``__getattribute__``."""
    try:
        state = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return {}
    return state if isinstance(state, dict) else {}


def _type_attribute(value: Any, attr: str) -> str:
    """Read a type's built-in identity metadata without custom metaclass dispatch."""
    return type.__getattribute__(type(value), attr)


def _mapping_child(attr: str, ordinal: int, key: Any, child: Any, *, owner: str | None) -> _Child:
    """Describe a mapping child without invoking arbitrary key ``repr``/``str`` methods."""
    if type(key) in (str, int):
        index: Any = key
        key_label = repr(key)
    else:
        # Never infer a semantic match for arbitrary keys: equality itself can execute user code,
        # and matching model/estimator mappings by insertion position can silently swap children.
        index = object()
        key_label = f"<key-{ordinal}>"
    prefix = f"{owner}.{attr}" if owner is not None else attr
    return _Child(attr, f"{prefix}[{key_label}]", child, index)


def _distribution_children(model: Any) -> list[_Child]:
    children: list[_Child] = []
    for attr, value in sorted(_instance_state(model).items()):
        if attr.startswith("_"):
            continue
        if isinstance(value, ProbabilityDistribution):
            children.append(_Child(attr, "%s.%s" % (_type_attribute(model, "__name__"), attr), value))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, ProbabilityDistribution):
                    children.append(
                        _Child(attr, "%s.%s[%d]" % (_type_attribute(model, "__name__"), attr, index), child, index)
                    )
        elif isinstance(value, dict):
            for ordinal, (key, child) in enumerate(value.items()):
                if isinstance(child, ProbabilityDistribution):
                    children.append(_mapping_child(attr, ordinal, key, child, owner=_type_attribute(model, "__name__")))
    return children


def _estimator_children(estimator: Any | None) -> list[_Child]:
    if estimator is None:
        return []
    children: list[_Child] = []
    for attr, value in sorted(_instance_state(estimator).items()):
        if attr.startswith("_"):
            continue
        if isinstance(value, ParameterEstimator):
            children.append(_Child(attr, attr, value))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, ParameterEstimator):
                    children.append(_Child(attr, "%s[%d]" % (attr, index), child, index))
        elif isinstance(value, dict):
            for ordinal, (key, child) in enumerate(value.items()):
                if isinstance(child, ParameterEstimator):
                    children.append(_mapping_child(attr, ordinal, key, child, owner=None))
    return children


def _canonical_attr(attr: str) -> str:
    value = attr.lower()
    for suffix in ("_distribution", "_dist", "_estimator", "_model"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    if value in ("components", "dists", "estimators", "children"):
        return "children"
    if value.endswith("s"):
        value = value[:-1]
    return value


def _bind_child_estimators(model_children: list[_Child], estimator: Any | None) -> dict[int, Any]:
    estimator_children = _estimator_children(estimator)
    bound: dict[int, Any] = {}
    used: set[int] = set()

    for model_index, model_child in enumerate(model_children):
        model_key = (_canonical_attr(model_child.attr), model_child.index)
        for estimator_index, estimator_child in enumerate(estimator_children):
            estimator_key = (_canonical_attr(estimator_child.attr), estimator_child.index)
            if estimator_index not in used and model_key == estimator_key:
                bound[model_index] = estimator_child.value
                used.add(estimator_index)
                break

    unbound_models = [index for index in range(len(model_children)) if index not in bound]
    unbound_estimators = [index for index in range(len(estimator_children)) if index not in used]
    # Positional fallback only when there is exactly ONE straggler on each side: pairing the sole
    # remaining model child with the sole remaining estimator child is the only inference possible
    # regardless of what either is named. With two or more unbound on each side, matching by
    # `zip()`'s enumeration order (both lists are built from `sorted(vars(...).items())`, i.e.
    # alphabetical by attribute name) is a silent coin flip whenever the model's and estimator's
    # naming conventions don't happen to sort into the same relative order -- it can swap which
    # estimator a child is bound to with no error raised. Leaving 2+-way mismatches unbound (falling
    # through to whatever infer_update_contract's own no-estimator branches report) is the safe
    # default; a wrong-but-confident contract is worse than an honestly conservative one here.
    if len(unbound_models) == len(unbound_estimators) == 1:
        bound[unbound_models[0]] = estimator_children[unbound_estimators[0]].value
    return bound


_MISSING = object()
_PARTIAL_DECLARATION_FIELDS = (
    "objective_kind",
    "update_kind",
    "merge_law",
    "state_semantics",
    "consistency_requirement",
    "curvature_kind",
    "decomposition_axes",
    "convergence_certificate",
    "compute_band",
    "outer_objective_compatible",
    "update_exact",
)


def _static_attribute(owner: Any, attr: str) -> Any:
    """Read an instance/class dictionary entry without invoking descriptors."""
    if owner is None:
        return _MISSING
    instance_state = _instance_state(owner)
    if attr in instance_state:
        return instance_state[attr]
    for cls in type.__getattribute__(type(owner), "__mro__"):
        namespace = type.__getattribute__(cls, "__dict__")
        if attr in namespace:
            return namespace[attr]
    return _MISSING


def _validate_contract(contract: Any, *, require_explicit: bool = True) -> UpdateContract:
    """Validate every runtime contract field before the compiler trusts it."""
    if type(contract) is not UpdateContract:
        raise TypeError("contract must be an UpdateContract.")
    enum_fields = (
        ("objective_kind", ObjectiveKind),
        ("update_kind", UpdateKind),
        ("merge_law", MergeLaw),
        ("consistency", ConsistencyRequirement),
        ("curvature_kind", CurvatureKind),
        ("convergence_certificate", ConvergenceCertificate),
        ("compute_band", ComputeBand),
        ("evidence_kind", ContractEvidenceKind),
    )
    for field_name, enum_type in enum_fields:
        if not isinstance(getattr(contract, field_name), enum_type):
            raise TypeError(f"contract {field_name} must be {enum_type.__name__}.")
    for field_name, enum_type in (
        ("state_semantics", StateSemantics),
        ("reads", ArtifactKind),
        ("writes", ArtifactKind),
    ):
        values = getattr(contract, field_name)
        if not isinstance(values, frozenset) or any(not isinstance(value, enum_type) for value in values):
            raise TypeError(f"contract {field_name} must be a frozenset of {enum_type.__name__}.")
    if (
        not isinstance(contract.decomposition_axes, tuple)
        or any(not isinstance(axis, str) or not axis for axis in contract.decomposition_axes)
        or len(set(contract.decomposition_axes)) != len(contract.decomposition_axes)
    ):
        raise TypeError("contract decomposition_axes must be unique non-empty strings.")
    if not isinstance(contract.outer_objective_compatible, bool) or not isinstance(contract.exact, bool):
        raise TypeError("contract compatibility and exactness flags must be bool.")
    if contract.exact and contract.objective_kind is ObjectiveKind.UNKNOWN:
        raise ValueError("an exact contract must identify the objective it solves.")
    if not isinstance(contract.declared_by, str) or not contract.declared_by:
        raise TypeError("contract declared_by must be a non-empty provenance string.")
    if contract.declared_by in {"compiler_default", "structural_inference"}:
        raise ValueError("explicit contracts cannot claim compiler-default or structural-inference provenance.")
    if require_explicit:
        if contract.evidence_kind is not ContractEvidenceKind.EXPLICIT_DECLARATION:
            raise ValueError("caller-supplied contracts require explicit_declaration evidence.")
        if not isinstance(contract.evidence_id, str) or not contract.evidence_id.strip():
            raise ValueError("caller-supplied contracts require a non-empty evidence_id.")
    if not isinstance(contract.notes, tuple) or any(not isinstance(note, str) for note in contract.notes):
        raise TypeError("contract notes must be a tuple of strings.")
    proof_prefix = "acceptance-proof:"
    if contract.convergence_certificate is ConvergenceCertificate.MONOTONE_CERTIFIED and not any(
        note.startswith(proof_prefix) and note[len(proof_prefix) :].strip() for note in contract.notes
    ):
        raise ValueError("monotone_certified contracts require an acceptance-proof note.")
    return contract


def _declared_contract(model: Any, estimator: Any | None) -> UpdateContract | None:
    for owner in (estimator, model):
        value = _static_attribute(owner, "update_contract")
        if value is _MISSING:
            continue
        if not isinstance(value, UpdateContract):
            raise TypeError(
                "update_contract must be a static UpdateContract value; callable hooks and descriptors "
                "are not executed during compilation."
            )
        return _validate_contract(value)
    return None


def _validate_partial_declarations(model: Any, estimator: Any | None) -> tuple[str, ...]:
    """Reject malformed fragments but never promote valid fragments into a full contract."""
    validators = {
        "objective_kind": lambda value: isinstance(value, ObjectiveKind),
        "update_kind": lambda value: isinstance(value, UpdateKind),
        "merge_law": lambda value: isinstance(value, MergeLaw),
        "state_semantics": lambda value: (
            isinstance(value, frozenset) and all(isinstance(item, StateSemantics) for item in value)
        ),
        "consistency_requirement": lambda value: isinstance(value, ConsistencyRequirement),
        "curvature_kind": lambda value: isinstance(value, CurvatureKind),
        "decomposition_axes": lambda value: (
            isinstance(value, tuple) and all(isinstance(item, str) and item for item in value)
        ),
        "convergence_certificate": lambda value: isinstance(value, ConvergenceCertificate),
        "compute_band": lambda value: isinstance(value, ComputeBand),
        "outer_objective_compatible": lambda value: isinstance(value, bool),
        "update_exact": lambda value: isinstance(value, bool),
    }
    declared: list[str] = []
    for owner in (estimator, model):
        if owner is None:
            continue
        for field in _PARTIAL_DECLARATION_FIELDS:
            value = _static_attribute(owner, field)
            if value is _MISSING:
                continue
            label = f"{_type_attribute(owner, '__name__')}.{field}"
            if not validators[field](value):
                raise TypeError(
                    f"invalid partial declaration {label}; supply the documented enum/value type inside "
                    "one complete UpdateContract."
                )
            declared.append(label)
    return tuple(declared)


def _type_id(value: Any) -> str:
    return f"{_type_attribute(value, '__module__')}.{_type_attribute(value, '__qualname__')}"


def _audited_builtin_contract(model: Any, estimator: Any | None) -> UpdateContract | None:
    """Resolve narrowly audited built-ins by exact type and inert instance state only."""
    pair = (_type_id(model), _type_id(estimator) if estimator is not None else None)
    gaussian_pair = (
        "mixle.stats.univariate.continuous.gaussian.GaussianDistribution",
        "mixle.stats.univariate.continuous.gaussian.GaussianEstimator",
    )
    if pair == gaussian_pair:
        model_prior = _instance_state(model).get("prior")
        estimator_prior = _instance_state(estimator).get("prior")
        if (model_prior is None) != (estimator_prior is None):
            return None
        objective = ObjectiveKind.MAP if estimator_prior is not None else ObjectiveKind.MLE
        return UpdateContract(
            objective_kind=objective,
            update_kind=UpdateKind.EXACT_CLOSED_FORM,
            merge_law=MergeLaw.ADDITIVE,
            state_semantics=frozenset({StateSemantics.IMMUTABLE_RESULT}),
            consistency=ConsistencyRequirement.STRICT_SYNCHRONOUS,
            curvature_kind=CurvatureKind.FISHER,
            reads=frozenset({ArtifactKind.OBSERVATIONS, ArtifactKind.PARAMETERS}),
            writes=frozenset({ArtifactKind.SUFFICIENT_STATISTICS, ArtifactKind.PARAMETERS}),
            outer_objective_compatible=True,
            exact=True,
            convergence_certificate=ConvergenceCertificate.UNKNOWN,
            compute_band=ComputeBand.FLOAT64,
            declared_by="audited_builtin_catalog:gaussian-v1",
            evidence_kind=ContractEvidenceKind.AUDITED_CATALOG,
            evidence_id="catalog:gaussian-v1",
            notes=("Exactness covers the one-step parameter map, not objective monotonicity.",),
        )

    mixture_pair = (
        "mixle.stats.latent.mixture.MixtureDistribution",
        "mixle.stats.latent.mixture.MixtureEstimator",
    )
    if pair == mixture_pair:
        return UpdateContract(
            objective_kind=ObjectiveKind.MLE,
            update_kind=UpdateKind.GENERALIZED_EM,
            merge_law=MergeLaw.ASSOCIATIVE_MONOID,
            state_semantics=frozenset({StateSemantics.IMMUTABLE_RESULT}),
            consistency=ConsistencyRequirement.STRICT_SYNCHRONOUS,
            curvature_kind=CurvatureKind.UNAVAILABLE,
            decomposition_axes=("component",),
            reads=frozenset(
                {
                    ArtifactKind.OBSERVATIONS,
                    ArtifactKind.PARAMETERS,
                    ArtifactKind.SCORES,
                    ArtifactKind.POSTERIORS,
                }
            ),
            writes=frozenset(
                {
                    ArtifactKind.SUFFICIENT_STATISTICS,
                    ArtifactKind.PARAMETERS,
                    ArtifactKind.POSTERIORS,
                }
            ),
            outer_objective_compatible=True,
            exact=False,
            convergence_certificate=ConvergenceCertificate.UNKNOWN,
            compute_band=ComputeBand.FLOAT64,
            declared_by="audited_builtin_catalog:finite-mixture-em-v1",
            evidence_kind=ContractEvidenceKind.AUDITED_CATALOG,
            evidence_id="catalog:finite-mixture-em-v1",
            notes=("No monotonicity certificate is inferred; acceptance evidence is required separately.",),
        )
    return None


def _unknown_contract(*, no_estimator: bool) -> UpdateContract:
    if no_estimator:
        return UpdateContract(
            objective_kind=ObjectiveKind.UNKNOWN,
            update_kind=UpdateKind.FROZEN,
            merge_law=MergeLaw.REPLICATED,
            state_semantics=frozenset({StateSemantics.IMMUTABLE_RESULT}),
            consistency=ConsistencyRequirement.LOCAL_ONLY,
            reads=frozenset({ArtifactKind.PARAMETERS}),
            writes=frozenset(),
            outer_objective_compatible=False,
            exact=False,
            convergence_certificate=ConvergenceCertificate.UNKNOWN,
            compute_band=ComputeBand.FLOAT64,
            declared_by="compiler:no-estimator",
            evidence_kind=ContractEvidenceKind.CONSERVATIVE_FALLBACK,
            evidence_id="compiler:no-estimator-v1",
            notes=("No estimator was supplied; the node is preserved but its objective is unknown.",),
        )
    return UpdateContract(
        objective_kind=ObjectiveKind.UNKNOWN,
        update_kind=UpdateKind.UNKNOWN,
        merge_law=MergeLaw.NON_MERGEABLE,
        state_semantics=frozenset({StateSemantics.EXTERNAL_STATE}),
        consistency=ConsistencyRequirement.LOCAL_ONLY,
        reads=frozenset({ArtifactKind.OBSERVATIONS, ArtifactKind.PARAMETERS, ArtifactKind.EXTERNAL_STATE}),
        writes=frozenset({ArtifactKind.EXTERNAL_STATE}),
        outer_objective_compatible=False,
        exact=False,
        convergence_certificate=ConvergenceCertificate.UNKNOWN,
        compute_band=ComputeBand.FLOAT64,
        declared_by="compiler:unknown",
        evidence_kind=ContractEvidenceKind.CONSERVATIVE_FALLBACK,
        evidence_id="compiler:unknown-v1",
        notes=("No explicit or audited contract evidence was available.",),
    )


def infer_update_contract(model: Any, estimator: Any | None) -> UpdateContract:
    """Return a static/audited contract or an explicit ``UNKNOWN`` contract.

    No factories, descriptors, methods, capability predicates, scoring functions, or estimator hooks are
    called. Names and method presence are not semantic evidence.
    """
    declared = _declared_contract(model, estimator)
    if declared is not None:
        return declared
    partial = _validate_partial_declarations(model, estimator)
    builtin = _audited_builtin_contract(model, estimator)
    if builtin is not None:
        return builtin
    unknown = _unknown_contract(no_estimator=estimator is None)
    if partial:
        return UpdateContract(
            **{
                **vars(unknown),
                "notes": unknown.notes + ("Ignored incomplete declaration fragments: " + ", ".join(partial) + ".",),
            }
        )
    return unknown


def _parameter_count(model: Any) -> int:
    """Count inert numeric instance state without calling declarations or module hooks."""
    values_seen: set[int] = set()
    total = 0
    stack = [model]
    while stack:
        value = stack.pop()
        if value is None or isinstance(value, ProbabilityDistribution) and value is not model:
            continue
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number) and not np.issubdtype(value.dtype, np.bool_):
                total += int(value.size)
            continue
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            total += 1
            continue
        ident = id(value)
        if ident in values_seen:
            continue
        values_seen.add(ident)
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        else:
            state = _instance_state(value)
            stack.extend(child for name, child in state.items() if not name.startswith("_"))
    return max(total, 1)


def _proxy_cost(contract: UpdateContract, parameter_count: int, nobs: float) -> CostEstimate:
    if contract.update_kind is UpdateKind.FROZEN:
        return CostEstimate(source="structural_proxy")
    multiplier = {
        UpdateKind.EXACT_CLOSED_FORM: 1.0,
        UpdateKind.GENERALIZED_EM: 10.0,
        UpdateKind.FIRST_ORDER: 50.0,
        UpdateKind.PRECONDITIONED: 65.0,
        UpdateKind.MONTE_CARLO: 100.0,
    }.get(contract.update_kind, 5.0)
    compute = float(parameter_count) * max(float(nobs), 1.0) + float(parameter_count) * multiplier
    return CostEstimate(
        compute_units=compute,
        communication_bytes=0,
        peak_memory_bytes=parameter_count * 8,
        source="structural_proxy",
    )


def compile_update_graph(
    model: ProbabilityDistribution,
    estimator: ParameterEstimator | None = None,
    *,
    nobs: float = 1.0,
    backend: str = "local",
    registry: ContractRegistry | None = None,
    bindings: Mapping[str, ParameterEstimator] | None = None,
    contract_overrides: Mapping[str, UpdateContract] | None = None,
    measurements: MeasurementCatalog | None = None,
) -> UpdateGraph:
    """Compile a model into an immutable typed update and invalidation graph.

    Compilation reads inert instance/class dictionaries only. It never calls factories, descriptors,
    capability predicates, contract hooks, ``parameters()``, scoring/sampling methods, estimators, or
    accumulators. Child estimators are aligned from the supplied estimator's stored child objects where the
    binding is unambiguous; otherwise the child contract is ``UNKNOWN``/frozen unless an explicit path binding
    is supplied.
    """

    if not isinstance(model, ProbabilityDistribution):
        raise TypeError("compile_update_graph requires a ProbabilityDistribution model.")
    if estimator is not None and not isinstance(estimator, ParameterEstimator):
        raise TypeError("estimator must be a ParameterEstimator or None.")
    if isinstance(nobs, bool) or not isinstance(nobs, (int, float)) or not np.isfinite(nobs) or nobs < 0.0:
        raise ValueError("nobs must be finite and non-negative.")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string.")
    if registry is not None and not isinstance(registry, ContractRegistry):
        raise TypeError("registry must be a ContractRegistry or None.")
    registry = ContractRegistry() if registry is None else registry
    bindings = dict(bindings or {})
    overrides = dict(contract_overrides or {})
    if any(not isinstance(path, str) or not path for path in bindings):
        raise ValueError("binding paths must be non-empty strings.")
    if any(not isinstance(value, ParameterEstimator) for value in bindings.values()):
        raise TypeError("binding values must be ParameterEstimator instances.")
    if any(not isinstance(path, str) or not path for path in overrides):
        raise ValueError("contract override paths must be non-empty strings.")
    for contract in overrides.values():
        _validate_contract(contract)
    root_estimator = estimator

    nodes: list[UpdateNode] = []
    edges: list[DependencyEdge] = []
    edge_keys: set[tuple[str, str, ArtifactKind]] = set()
    by_identity: dict[int, str] = {}
    used_bindings: set[str] = set()
    used_overrides: set[str] = set()

    def add_edge(edge: DependencyEdge) -> None:
        key = (edge.source_node, edge.target_node, edge.artifact)
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append(edge)

    def visit(current: Any, current_estimator: Any | None, path: str, parent_id: str | None) -> str:
        ident = id(current)
        if ident in by_identity:
            node_id = by_identity[ident]
            if parent_id is not None:
                add_edge(
                    DependencyEdge(
                        node_id,
                        parent_id,
                        ArtifactKind.PARAMETERS,
                        "shared child update invalidates every consuming parent",
                    )
                )
            return node_id

        node_id = "n%04d" % len(nodes)
        by_identity[ident] = node_id
        if path in bindings:
            current_estimator = bindings[path]
            used_bindings.add(path)
        explicit = None
        if path in overrides:
            explicit = overrides[path]
            used_overrides.add(path)
        explicit = explicit or _declared_contract(current, current_estimator)
        contract = (
            explicit
            or registry.resolve(current, current_estimator)
            or infer_update_contract(current, current_estimator)
        )
        _validate_contract(contract, require_explicit=False)
        parameter_count = _parameter_count(current)
        cost = None
        if measurements is not None:
            cost = measurements.estimate(_type_attribute(current, "__name__"), contract.update_kind, backend)
        cost = cost or _proxy_cost(contract, parameter_count, nobs)
        nodes.append(
            UpdateNode(
                node_id=node_id,
                path=path,
                model_type=_type_attribute(current, "__name__"),
                estimator_type=_type_attribute(current_estimator, "__name__")
                if current_estimator is not None
                else None,
                contract=contract,
                cost=cost,
                parameter_count=parameter_count,
                model=current,
                estimator=current_estimator,
            )
        )
        if parent_id is not None:
            add_edge(DependencyEdge(node_id, parent_id))

        children = _distribution_children(current)
        child_estimators = _bind_child_estimators(children, current_estimator)
        for index, child in enumerate(children):
            child_path = "%s -> %s" % (path, child.label)
            child_estimator = child_estimators.get(index)
            visit(child.value, child_estimator, child_path, node_id)
        return node_id

    root_id = visit(model, root_estimator, "root", None)
    unused_bindings = sorted(set(bindings) - used_bindings)
    unused_overrides = sorted(set(overrides) - used_overrides)
    if unused_bindings or unused_overrides:
        details = []
        if unused_bindings:
            details.append("unused bindings: " + ", ".join(unused_bindings))
        if unused_overrides:
            details.append("unused contract overrides: " + ", ".join(unused_overrides))
        raise ValueError("; ".join(details))
    return UpdateGraph.from_parts(nodes, edges, root_node=root_id)


__all__ = ["ContractRegistry", "compile_update_graph", "infer_update_contract"]
