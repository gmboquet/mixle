"""Side-effect-free compiler from a model tree to a typed update graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import (
    ArtifactKind,
    ConsistencyRequirement,
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

ContractFactory = Callable[[Any, Any | None], UpdateContract]


@dataclass(frozen=True)
class _Child:
    attr: str
    label: str
    value: Any
    index: int | str | None = None


@dataclass(frozen=True)
class _RegisteredContract:
    model_type: type[Any]
    estimator_type: type[Any] | None
    factory: ContractFactory


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
        contract: UpdateContract | ContractFactory,
        *,
        estimator_type: type[Any] | None = None,
    ) -> None:
        """Register a constant contract or factory for a model/estimator pair."""

        if isinstance(contract, UpdateContract):
            factory = lambda _model, _estimator, value=contract: value
        elif callable(contract):
            factory = contract
        else:
            raise TypeError("contract must be an UpdateContract or callable factory.")
        self._entries.append(_RegisteredContract(model_type, estimator_type, factory))

    def resolve(self, model: Any, estimator: Any | None) -> UpdateContract | None:
        """Resolve the most recently registered matching adapter."""

        for entry in reversed(self._entries):
            if not isinstance(model, entry.model_type):
                continue
            if entry.estimator_type is not None and not isinstance(estimator, entry.estimator_type):
                continue
            return entry.factory(model, estimator)
        return None


def _distribution_children(model: Any) -> list[_Child]:
    children: list[_Child] = []
    for attr, value in sorted(vars(model).items()):
        if attr.startswith("_"):
            continue
        if isinstance(value, ProbabilityDistribution):
            children.append(_Child(attr, "%s.%s" % (type(model).__name__, attr), value))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, ProbabilityDistribution):
                    children.append(_Child(attr, "%s.%s[%d]" % (type(model).__name__, attr, index), child, index))
        elif isinstance(value, dict):
            for key, child in sorted(value.items(), key=lambda item: repr(item[0])):
                if isinstance(child, ProbabilityDistribution):
                    children.append(_Child(attr, "%s.%s[%r]" % (type(model).__name__, attr, key), child, str(key)))
    return children


def _estimator_children(estimator: Any | None) -> list[_Child]:
    if estimator is None:
        return []
    children: list[_Child] = []
    for attr, value in sorted(vars(estimator).items()):
        if attr.startswith("_"):
            continue
        if isinstance(value, ParameterEstimator):
            children.append(_Child(attr, attr, value))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, ParameterEstimator):
                    children.append(_Child(attr, "%s[%d]" % (attr, index), child, index))
        elif isinstance(value, dict):
            for key, child in sorted(value.items(), key=lambda item: repr(item[0])):
                if isinstance(child, ParameterEstimator):
                    children.append(_Child(attr, "%s[%r]" % (attr, key), child, str(key)))
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
    if len(unbound_models) == len(unbound_estimators):
        for model_index, estimator_index in zip(unbound_models, unbound_estimators):
            bound[model_index] = estimator_children[estimator_index].value
    return bound


def _declared_enum(owner: Any, attr: str, enum_type: type[Any]) -> Any | None:
    if owner is None or not hasattr(owner, attr):
        return None
    value = getattr(owner, attr)
    if callable(value):
        value = value()
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError:
        return None


def _declared_contract(model: Any, estimator: Any | None) -> UpdateContract | None:
    for owner in (estimator, model):
        if owner is None:
            continue
        hook = getattr(owner, "update_contract", None)
        if isinstance(hook, UpdateContract):
            return hook
        if callable(hook):
            contract = hook()
            if not isinstance(contract, UpdateContract):
                raise TypeError("update_contract() must return UpdateContract.")
            return contract
    return None


def _has_torch_like_module(*roots: Any) -> bool:
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        obj = stack.pop()
        if obj is None or isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, np.ndarray)):
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        if (
            callable(getattr(obj, "state_dict", None))
            and callable(getattr(obj, "load_state_dict", None))
            and callable(getattr(obj, "parameters", None))
        ):
            return True
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
    return False


def _surrogate_objective(estimator: Any | None) -> ObjectiveKind | None:
    if estimator is None:
        return None
    module = getattr(estimator, "module", None)
    if getattr(estimator, "loss", None) is not None:
        return ObjectiveKind.USER_SURROGATE
    if all(hasattr(estimator, attr) for attr in ("policy", "ref", "beta")):
        return ObjectiveKind.PREFERENCE
    if (
        module is not None
        and callable(getattr(module, "energy", None))
        and hasattr(module, "log_norm")
        and hasattr(estimator, "noise_ratio")
    ):
        return ObjectiveKind.CONTRASTIVE
    if callable(getattr(estimator, "residual_fn", None)) and hasattr(estimator, "residual_weight"):
        return ObjectiveKind.CONSTRAINT
    return None


def _objective_kind(model: Any, estimator: Any | None) -> tuple[ObjectiveKind, bool, tuple[str, ...]]:
    declared = _declared_enum(estimator, "objective_kind", ObjectiveKind) or _declared_enum(
        model, "objective_kind", ObjectiveKind
    )
    if declared is not None:
        compatible = bool(getattr(estimator, "outer_objective_compatible", True))
        return declared, compatible, ("objective declared by model/estimator",)

    surrogate = _surrogate_objective(estimator)
    if surrogate is not None:
        return surrogate, False, ("surrogate objective inferred from structural estimator protocol",)
    if callable(getattr(model, "seq_local_elbo", None)):
        return ObjectiveKind.ELBO, True, ("model exposes seq_local_elbo",)
    get_prior = getattr(estimator, "get_prior", None)
    prior = get_prior() if callable(get_prior) else getattr(estimator, "prior", None)
    if prior is not None:
        return ObjectiveKind.MAP, True, ("estimator carries a parameter prior",)
    return ObjectiveKind.MLE, True, ()


def _update_kind(model: Any, estimator: Any | None) -> tuple[UpdateKind, bool, tuple[str, ...]]:
    declared = _declared_enum(estimator, "update_kind", UpdateKind) or _declared_enum(model, "update_kind", UpdateKind)
    if declared is not None:
        exact = bool(getattr(estimator, "update_exact", declared is UpdateKind.EXACT_CLOSED_FORM))
        return declared, exact, ("update kind declared by model/estimator",)

    if estimator is None:
        return UpdateKind.FROZEN, True, ("no usable estimator; preserved as a frozen dependency",)

    try:
        from mixle.capability import ConjugateUpdatable, LatentStructured, Neutral, supports

        if supports(model, Neutral):
            return UpdateKind.FROZEN, True, ()
        if callable(getattr(model, "seq_posterior", None)) or supports(model, LatentStructured):
            return UpdateKind.GENERALIZED_EM, True, ()
        if _has_torch_like_module(model, estimator):
            return UpdateKind.FIRST_ORDER, False, ()
        if supports(model, ConjugateUpdatable):
            return UpdateKind.EXACT_CLOSED_FORM, True, ()
    except (ImportError, TypeError):
        pass
    return UpdateKind.EXACT_CLOSED_FORM, True, ("conservative accumulator-backed default",)


def _merge_law(model: Any, estimator: Any | None, update_kind: UpdateKind) -> MergeLaw:
    declared = _declared_enum(estimator, "merge_law", MergeLaw) or _declared_enum(model, "merge_law", MergeLaw)
    if declared is not None:
        return declared
    if update_kind is UpdateKind.FROZEN:
        return MergeLaw.REPLICATED
    try:
        from mixle.capability import ExponentialFamily, supports

        if supports(model, ExponentialFamily):
            return MergeLaw.ADDITIVE
    except (ImportError, TypeError):
        pass
    if estimator is not None and callable(getattr(estimator, "accumulator_factory", None)):
        return MergeLaw.ASSOCIATIVE_MONOID
    return MergeLaw.NON_MERGEABLE


def _state_semantics(model: Any, estimator: Any | None, update_kind: UpdateKind) -> frozenset[StateSemantics]:
    if update_kind is UpdateKind.FROZEN:
        return frozenset({StateSemantics.IMMUTABLE_RESULT})
    states: set[StateSemantics] = set()
    if _has_torch_like_module(model, estimator):
        states.add(StateSemantics.MUTABLE_PARAMETERS)
    if hasattr(estimator, "_rng") or update_kind in (UpdateKind.MONTE_CARLO, UpdateKind.FIRST_ORDER):
        states.add(StateSemantics.STOCHASTIC_RNG)
    if getattr(estimator, "persistent_optimizer", False):
        states.add(StateSemantics.MUTABLE_OPTIMIZER)
    return frozenset(states or {StateSemantics.IMMUTABLE_RESULT})


def _curvature_kind(model: Any, estimator: Any | None) -> CurvatureKind:
    declared = _declared_enum(estimator, "curvature_kind", CurvatureKind) or _declared_enum(
        model, "curvature_kind", CurvatureKind
    )
    if declared is not None:
        return declared
    try:
        from mixle.capability import ExponentialFamily, supports

        if supports(model, ExponentialFamily):
            return CurvatureKind.FISHER
    except (ImportError, TypeError):
        pass
    return CurvatureKind.UNAVAILABLE


def _decomposition(model: Any) -> tuple[tuple[str, ...], bool]:
    try:
        from mixle.stats.compute.decomposition import decomposition_for

        descriptor = decomposition_for(model)
        axes = (descriptor.axis.value,) if descriptor.is_shardable else ()
        return axes, bool(descriptor.exact)
    except (AttributeError, ImportError, TypeError):
        return (), True


def infer_update_contract(model: Any, estimator: Any | None) -> UpdateContract:
    """Infer a conservative contract without scoring, sampling, or mutating the model."""

    objective, compatible, objective_notes = _objective_kind(model, estimator)
    update, exact_update, update_notes = _update_kind(model, estimator)
    merge = _merge_law(model, estimator, update)
    states = _state_semantics(model, estimator, update)
    axes, exact_decomposition = _decomposition(model)

    reads = {ArtifactKind.OBSERVATIONS, ArtifactKind.PARAMETERS}
    writes = {ArtifactKind.SUFFICIENT_STATISTICS, ArtifactKind.PARAMETERS}
    if update is UpdateKind.GENERALIZED_EM:
        reads.update((ArtifactKind.SCORES, ArtifactKind.POSTERIORS))
        writes.add(ArtifactKind.POSTERIORS)
    if StateSemantics.MUTABLE_OPTIMIZER in states:
        reads.add(ArtifactKind.OPTIMIZER_STATE)
        writes.add(ArtifactKind.OPTIMIZER_STATE)
    if StateSemantics.STOCHASTIC_RNG in states:
        reads.add(ArtifactKind.RNG_STATE)
        writes.add(ArtifactKind.RNG_STATE)
    if update is UpdateKind.FROZEN:
        writes.clear()

    consistency = ConsistencyRequirement.STRICT_SYNCHRONOUS
    declared_consistency = _declared_enum(estimator, "consistency_requirement", ConsistencyRequirement)
    if declared_consistency is not None:
        consistency = declared_consistency

    return UpdateContract(
        objective_kind=objective,
        update_kind=update,
        merge_law=merge,
        state_semantics=states,
        consistency=consistency,
        curvature_kind=_curvature_kind(model, estimator),
        decomposition_axes=axes,
        reads=frozenset(reads),
        writes=frozenset(writes),
        outer_objective_compatible=compatible,
        exact=exact_update and exact_decomposition and compatible,
        declared_by="structural_inference",
        notes=objective_notes + update_notes,
    )


def _parameter_count(model: Any) -> int:
    try:
        from mixle.stats.compute.declarations import declaration_for

        declaration = declaration_for(model)
        if declaration is not None and declaration.parameters:
            total = 0
            for spec in declaration.parameters:
                value = getattr(model, spec.name, None)
                if isinstance(value, np.ndarray):
                    total += int(value.size)
                elif value is not None:
                    total += 1
            if total:
                return total
    except (AttributeError, ImportError, TypeError):
        pass

    modules_seen: set[int] = set()
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
        if ident in modules_seen:
            continue
        modules_seen.add(ident)
        parameters = getattr(value, "parameters", None)
        if callable(parameters) and callable(getattr(value, "state_dict", None)):
            total += sum(int(parameter.numel()) for parameter in parameters())
            continue
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif hasattr(value, "__dict__"):
            stack.extend(child for name, child in vars(value).items() if not name.startswith("_"))
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


def _fallback_estimator(model: Any) -> Any | None:
    factory = getattr(model, "estimator", None)
    if not callable(factory):
        return None
    try:
        return factory()
    except (NotImplementedError, TypeError, ValueError):
        return None


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

    Compilation is introspective only. It never calls ``sampler``,
    ``log_density``, ``estimate``, or an accumulator update. Child estimators are
    aligned structurally where possible and otherwise constructed from the child
    model's ordinary ``estimator()`` factory. Explicit path bindings always win.
    """

    if not isinstance(model, ProbabilityDistribution):
        raise TypeError("compile_update_graph requires a ProbabilityDistribution model.")
    if nobs < 0.0:
        raise ValueError("nobs must be non-negative.")
    registry = ContractRegistry() if registry is None else registry
    bindings = dict(bindings or {})
    overrides = dict(contract_overrides or {})
    root_estimator = estimator if estimator is not None else _fallback_estimator(model)

    nodes: list[UpdateNode] = []
    edges: list[DependencyEdge] = []
    by_identity: dict[int, str] = {}

    def visit(current: Any, current_estimator: Any | None, path: str, parent_id: str | None) -> str:
        ident = id(current)
        if ident in by_identity:
            node_id = by_identity[ident]
            if parent_id is not None:
                edges.append(
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
        current_estimator = bindings.get(path, current_estimator)
        explicit = overrides.get(path) or _declared_contract(current, current_estimator)
        contract = (
            explicit
            or registry.resolve(current, current_estimator)
            or infer_update_contract(current, current_estimator)
        )
        parameter_count = _parameter_count(current)
        cost = None
        if measurements is not None:
            cost = measurements.estimate(type(current).__name__, contract.update_kind, backend)
        cost = cost or _proxy_cost(contract, parameter_count, nobs)
        nodes.append(
            UpdateNode(
                node_id=node_id,
                path=path,
                model_type=type(current).__name__,
                estimator_type=type(current_estimator).__name__ if current_estimator is not None else None,
                contract=contract,
                cost=cost,
                parameter_count=parameter_count,
                model=current,
                estimator=current_estimator,
            )
        )
        if parent_id is not None:
            edges.append(DependencyEdge(node_id, parent_id))

        children = _distribution_children(current)
        child_estimators = _bind_child_estimators(children, current_estimator)
        for index, child in enumerate(children):
            child_path = "%s -> %s" % (path, child.label)
            child_estimator = (
                child_estimators.get(index) or bindings.get(child_path) or _fallback_estimator(child.value)
            )
            visit(child.value, child_estimator, child_path, node_id)
        return node_id

    root_id = visit(model, root_estimator, "root", None)
    return UpdateGraph.from_parts(nodes, edges, root_node=root_id)


__all__ = ["ContractRegistry", "compile_update_graph", "infer_update_contract"]
