"""Declarative metadata for generated distribution scoring and fitting kernels.

Declarations describe parameters, constraints, sufficient statistics, and
exponential-family structure so generic backend code can generate safe scalar,
stacked, and optional Numba scoring paths.
"""

from __future__ import annotations

import dis
import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np

from mixle.stats.compute.capability_decline import KernelCapabilityDeclinedError
from mixle.utils.optional_deps import numba


class GeneratedKernelCompilationError(RuntimeError):
    """A declared generated kernel failed to trace or compile unexpectedly."""


@dataclass(frozen=True)
class ParameterSpec:
    """A fitted distribution parameter used by scoring.

    Constraints are interpreted by generic generated/scoring utilities. In
    addition to scalar constraints such as ``positive`` and vector/matrix
    constraints such as ``simplex_vector``, ``row_simplex_matrix``, and
    ``column_simplex_matrix``, ``greater_than:<parameter>`` marks a coupled
    ordered bound.
    """

    name: str
    constraint: str = "real"
    differentiable: bool = True


@dataclass(frozen=True)
class StatisticSpec:
    """A sufficient-statistic entry produced by accumulation."""

    name: str
    kind: str = "moment"
    additive: bool = True
    scales: bool = True


@dataclass(frozen=True)
class ExponentialFamilySpec:
    """Conditional exponential-family pieces for generated scalar scoring."""

    sufficient_statistics: Callable[[Any, Any], tuple[Any, ...]]
    natural_parameters: Callable[[dict[str, Any], Any], tuple[Any, ...]]
    log_partition: Callable[[dict[str, Any], Any], Any]
    base_measure: Callable[[Any, Any], Any] | None = None
    sufficient_statistics_from_params: Callable[[Any, dict[str, Any], Any], tuple[Any, ...]] | None = None
    base_measure_from_params: Callable[[Any, dict[str, Any], Any], Any] | None = None
    legacy_sufficient_statistics: Callable[[Any, dict[str, Any], Any], tuple[Any, ...]] | None = None
    analytic_mean_parameters: Callable[[dict[str, Any], Any], Any] | None = None
    log_partition_from_natural: Callable[[Any, Any], Any] | None = None
    fixed_base: bool = True
    runtime_scoring: bool = True
    """Whether the generated exp-family form may drive *runtime* scoring (scalar + stacked + numba).

    Set ``False`` when the canonical ``<eta, T(x)>`` dot form is numerically unsafe as a runtime
    scorer even though the family is a valid exponential family. The motivating case is the
    categorical: ``eta = log(p)`` has ``-inf`` entries for zero-probability categories, so the
    generic dot product hits ``0 * -inf = NaN`` for observations of *other* categories (its own
    ``seq_log_density`` avoids this by indexing). With ``runtime_scoring=False`` the family keeps its
    ``backend_*`` scoring path while ``to_exponential_family`` still exposes the canonical map (valid
    where ``p > 0``)."""
    """Whether the base measure ``h(x)`` is independent of the per-component parameters.

    The generated *stacked* exp-family loops broadcast a single per-row base ``(n,)`` across all
    ``k`` components, so they are only valid when ``h(x)`` does not vary by component. Families
    whose base depends on a per-component parameter (e.g. NegativeBinomial's ``lgamma(x+r)`` for a
    varying shape ``r``) set this ``False``: they keep the scalar canonical map / ``to_exponential_family``
    view but route stacked scoring through their ``backend_*`` hooks instead of the fixed-base loop.
    """


@dataclass(frozen=True)
class DistributionDeclaration:
    """Metadata needed by generated kernels and future autograd paths."""

    name: str
    distribution_type: type[Any]
    parameters: tuple[ParameterSpec, ...]
    statistics: tuple[StatisticSpec, ...]
    support: str
    children: tuple[DistributionDeclaration, ...] = ()
    child_roles: tuple[str, ...] = ()
    differentiable: bool = True
    exponential_family: ExponentialFamilySpec | None = None
    legacy_sufficient_statistics: Callable[[Any, dict[str, Any], Any], tuple[Any, ...]] | None = None

    def parameter_values(self, dist: Any) -> dict[str, Any]:
        """Extract declared parameter values from a distribution instance."""
        if not isinstance(dist, self.distribution_type):
            raise TypeError("expected %s, got %s" % (self.distribution_type.__name__, type(dist).__name__))
        return {spec.name: getattr(dist, spec.name) for spec in self.parameters}

    def statistic_values(self, suff_stat: Any) -> dict[str, Any]:
        """Map a legacy accumulator value into declared statistic names."""
        if not self.statistics:
            return {}
        if len(self.statistics) == 1:
            return {self.statistics[0].name: suff_stat}
        if not isinstance(suff_stat, (tuple, list)) or len(suff_stat) != len(self.statistics):
            raise ValueError(
                "%s expected %d statistic entries, got %s."
                % (self.name, len(self.statistics), type(suff_stat).__name__)
            )
        return {spec.name: value for spec, value in zip(self.statistics, suff_stat)}

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return declared parameter names in storage order."""
        return tuple(spec.name for spec in self.parameters)

    @property
    def statistic_names(self) -> tuple[str, ...]:
        """Return declared sufficient-statistic names in accumulator order."""
        return tuple(spec.name for spec in self.statistics)

    @property
    def has_exponential_family(self) -> bool:
        """Return whether generated exponential-family scoring is available."""
        return self.exponential_family is not None


_DECLARATIONS: dict[type[Any], DistributionDeclaration] = {}
_DECLARATION_LOCK = RLock()


def register_declaration(declaration: DistributionDeclaration) -> None:
    """Validate and atomically register one declaration.

    Re-registering the same declaration is idempotent. A different declaration
    for an already registered type is rejected instead of silently changing
    generated execution semantics process-wide.
    """
    validated = validate_declaration(declaration)
    with _DECLARATION_LOCK:
        existing = _DECLARATIONS.get(validated.distribution_type)
        if existing is not None and existing != validated:
            raise ValueError(
                "Conflicting distribution declaration for %s is already registered."
                % validated.distribution_type.__name__
            )
        _DECLARATIONS[validated.distribution_type] = validated


def declaration_for(x: Any) -> DistributionDeclaration | None:
    """Return a declaration for a distribution instance or class, if present."""
    cls = x if isinstance(x, type) else type(x)
    hook = (
        getattr(x, "compute_declaration", None)
        if not isinstance(x, type)
        else getattr(cls, "compute_declaration", None)
    )
    if callable(hook):
        try:
            return hook()
        except TypeError:
            if not isinstance(x, type):
                raise

    for base in cls.mro():
        declaration = _DECLARATIONS.get(base)
        if declaration is not None:
            return declaration
    return None


def declared_distribution_types() -> Iterable[type[Any]]:
    """Return classes that currently have declarations."""
    return tuple(_DECLARATIONS.keys())


def declaration_issues(x: Any) -> tuple[str, ...]:
    """Return structural issues in a distribution declaration.

    This is intentionally schema-level validation: it checks names,
    constraints, child roles, and callable exponential-family pieces without
    importing or special-casing concrete distribution implementations.
    """
    declaration = x if isinstance(x, DistributionDeclaration) else declaration_for(x)
    if declaration is None:
        cls = x if isinstance(x, type) else type(x)
        return ("%s has no declaration." % cls.__name__,)
    return tuple(
        _declaration_issues(
            declaration,
            path=declaration.name or "<unnamed>",
            ancestors=frozenset(),
        )
    )


def validate_declaration(x: Any) -> DistributionDeclaration:
    """Return a declaration or raise ``ValueError`` with schema issues."""
    declaration = x if isinstance(x, DistributionDeclaration) else declaration_for(x)
    issues = declaration_issues(declaration if declaration is not None else x)
    if issues:
        raise ValueError("Invalid distribution declaration: %s" % "; ".join(issues))
    return declaration


def statistic_layout_issues(x: Any, suff_stat: Any) -> tuple[str, ...]:
    """Return issues mapping a legacy sufficient-statistic payload to a declaration.

    This validates the top-level ``statistic_values(...)`` arity and, when a
    declaration exposes child roles, recursively validates child statistic
    payloads for tuple/list/map-shaped entries. It stays schema-driven and does
    not import concrete distribution implementations.
    """
    declaration = x if isinstance(x, DistributionDeclaration) else declaration_for(x)
    if declaration is None:
        cls = x if isinstance(x, type) else type(x)
        return ("%s has no declaration." % cls.__name__,)
    return tuple(_statistic_layout_issues(declaration, suff_stat, declaration.name or "<unnamed>"))


def validate_statistic_layout(x: Any, suff_stat: Any) -> DistributionDeclaration:
    """Return a declaration or raise ``ValueError`` with statistic-layout issues."""
    declaration = x if isinstance(x, DistributionDeclaration) else declaration_for(x)
    issues = statistic_layout_issues(declaration if declaration is not None else x, suff_stat)
    if issues:
        raise ValueError("Invalid statistic layout: %s" % "; ".join(issues))
    return declaration


def _exp_family_runtime_scoring(declaration: DistributionDeclaration) -> bool:
    """Whether the exp-family form may drive scalar/numba runtime scoring for this family."""
    ef = declaration.exponential_family
    return ef is not None and ef.runtime_scoring


def _exp_family_stacked_scoring(declaration: DistributionDeclaration) -> bool:
    """Whether the exp-family form may drive the fixed-base *stacked* runtime loop."""
    ef = declaration.exponential_family
    return ef is not None and ef.fixed_base and ef.runtime_scoring


def generated_stacked_available(dist_type: type[Any]) -> bool:
    """Return true when declarations can generate stacked leaf scoring."""
    declaration = declaration_for(dist_type)
    if declaration is None:
        return False
    if _exp_family_stacked_scoring(declaration):
        return True
    return _generated_backend_hook_supported(dist_type, declaration)


def generated_stacked_preferred(dist_type: type[Any]) -> bool:
    """Return true when a family explicitly opts into declaration-generated scoring."""
    declaration = declaration_for(dist_type)
    return declaration is not None and _exp_family_stacked_scoring(declaration)


def generated_stacked_strategy(dist_type: type[Any]) -> str:
    """Describe the declaration-generated stacked scoring route for a family."""
    declaration = declaration_for(dist_type)
    if declaration is None:
        return "none"
    if _exp_family_stacked_scoring(declaration):
        return "exp_family"
    if _generated_backend_hook_supported(dist_type, declaration):
        return "backend_log_density_from_params"
    return "none"


def generated_log_density_diagnostics(x: Any, encoded_symbols: Sequence[str] | None = None) -> dict[str, Any]:
    """Trace a generated scalar log-density formula with the symbolic engine.

    The returned dictionary contains the symbolic expression string, referenced
    data/parameter symbols, operation counts, and expression depth. This is an
    author-facing inspection tool for declaration-generated kernels; it does
    not select or execute a runtime backend.
    """
    from mixle.engines import SymbolicEngine

    declaration = declaration_for(x)
    if declaration is None:
        cls = x if isinstance(x, type) else type(x)
        raise ValueError("%s has no compute declaration." % cls.__name__)
    dist_type = declaration.distribution_type
    engine = SymbolicEngine()
    encoded_names = _diagnostic_encoded_symbols(dist_type, declaration, encoded_symbols)
    encoded_values = _diagnostic_encoded_values(encoded_names, engine, declaration)
    enc = encoded_values[0] if len(encoded_values) == 1 else tuple(encoded_values)
    params = _diagnostic_param_symbols(declaration, engine)
    fallback_reason = None

    if declaration.exponential_family is not None:
        try:
            _ensure_diagnostic_exp_family_trace_supported(declaration)
            expr = _generated_exp_family_scalar_expression(enc, params, declaration.exponential_family, engine)
            strategy = "exp_family"
        except KernelCapabilityDeclinedError as exc:
            fn = getattr(dist_type, "backend_log_density_from_params", None)
            if not callable(fn):
                raise
            fallback_reason = "%s: %s" % (type(exc).__name__, exc)
            _ensure_diagnostic_backend_trace_supported(declaration)
            try:
                expr = _diagnostic_backend_log_density_expression(dist_type, params, encoded_values, engine)
            except KernelCapabilityDeclinedError:
                raise
            except Exception as backend_exc:
                raise GeneratedKernelCompilationError(
                    "%s backend diagnostic trace failed: %s"
                    % (dist_type.__name__, backend_exc)
                ) from backend_exc
            strategy = "backend_log_density_from_params"
        except Exception as exc:
            raise GeneratedKernelCompilationError(
                "%s exponential-family diagnostic trace failed: %s"
                % (dist_type.__name__, exc)
            ) from exc
    else:
        _ensure_diagnostic_backend_trace_supported(declaration)
        try:
            expr = _diagnostic_backend_log_density_expression(dist_type, params, encoded_values, engine)
        except KernelCapabilityDeclinedError:
            raise
        except Exception as exc:
            raise GeneratedKernelCompilationError(
                "%s backend diagnostic trace failed: %s" % (dist_type.__name__, exc)
            ) from exc
        strategy = "backend_log_density_from_params"

    rv = engine.diagnostics(expr)
    rv.update(
        {
            "expression": str(expr),
            "strategy": strategy,
            "encoded_symbols": tuple(encoded_names),
            "parameter_symbols": declaration.parameter_names,
        }
    )
    if fallback_reason is not None:
        rv["fallback_reason"] = fallback_reason
    return rv


def _ensure_diagnostic_exp_family_trace_supported(declaration: DistributionDeclaration) -> None:
    spec = declaration.exponential_family
    if spec is None:
        return
    if spec.sufficient_statistics_from_params is not None:
        raise KernelCapabilityDeclinedError(
            "symbolic diagnostics cannot infer runtime-dependent sufficient-statistic geometry"
        )
    if declaration.support.endswith("_vector"):
        raise KernelCapabilityDeclinedError(
            "symbolic diagnostics do not infer vector observation geometry"
        )


def _ensure_diagnostic_backend_trace_supported(declaration: DistributionDeclaration) -> None:
    if not _generated_backend_hook_supported(declaration.distribution_type, declaration):
        raise KernelCapabilityDeclinedError(
            "backend diagnostic hook is not declaration-conformant"
        )
    non_symbolic = {
        "simplex_map",
        "row_simplex_map",
        "log_probability_tables",
    }
    if any(spec.constraint in non_symbolic for spec in declaration.parameters):
        raise KernelCapabilityDeclinedError(
            "symbolic backend diagnostics do not synthesize mapping-valued parameters"
        )


def _numeric_parameter_array(value: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("declared parameter %s must be numeric." % name) from exc
    if np.any(np.isnan(arr)) or np.any(np.isposinf(arr)):
        raise ValueError("declared parameter %s cannot contain NaN or positive infinity." % name)
    return arr


def _parameter_value_issues(spec: ParameterSpec, value: Any) -> tuple[str, ...]:
    constraint = spec.constraint
    name = spec.name
    if constraint in ("fixed", "metadata", "log_probability_tables"):
        return ()
    if constraint in ("optional_integer", "optional_log_unit_interval_vector") and value is None:
        return ()
    if constraint == "simplex_map":
        if not isinstance(value, Mapping):
            return ("%s must be a mapping." % name,)
        try:
            arr = _numeric_parameter_array(tuple(value.values()), name)
        except ValueError as exc:
            return (str(exc),)
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            return ("%s mapping probabilities must lie in [0, 1]." % name,)
        return ()
    if constraint == "row_simplex_map":
        if not isinstance(value, Mapping):
            return ("%s must be a mapping of probability rows." % name,)
        issues = []
        for key, row in value.items():
            if not isinstance(row, Mapping):
                issues.append("%s[%r] must be a probability mapping." % (name, key))
                continue
            try:
                arr = _numeric_parameter_array(tuple(row.values()), "%s[%r]" % (name, key))
            except ValueError as exc:
                issues.append(str(exc))
                continue
            if np.any(arr < 0.0) or np.any(arr > 1.0) or (
                arr.size and not np.isclose(arr.sum(), 1.0, rtol=1.0e-7, atol=1.0e-10)
            ):
                issues.append("%s[%r] must be a probability simplex." % (name, key))
        return tuple(issues)

    try:
        arr = _numeric_parameter_array(value, name)
    except ValueError as exc:
        return (str(exc),)

    scalar_constraints = {
        "real",
        "positive",
        "unit_interval",
        "integer",
        "positive_integer",
        "non_negative_integer",
        "optional_integer",
    }
    if constraint in scalar_constraints and arr.ndim != 0:
        return ("%s must be scalar for constraint %s." % (name, constraint),)
    if constraint in ("real_vector", "positive_vector", "simplex", "simplex_vector") and arr.ndim != 1:
        return ("%s must be one-dimensional for constraint %s." % (name, constraint),)
    if constraint in (
        "positive_matrix",
        "row_simplex_matrix",
        "column_simplex_matrix",
        "integer_matrix",
    ) and arr.ndim != 2:
        return ("%s must be two-dimensional for constraint %s." % (name, constraint),)
    if constraint == "integer_vector" and arr.ndim != 1:
        return ("%s must be one-dimensional for constraint integer_vector." % name,)
    if _is_ordered_constraint(constraint) and (arr.ndim != 0 or not np.isfinite(arr)):
        return ("%s must be a finite scalar for ordered constraint %s." % (name, constraint),)

    finite_constraints = {
        "real",
        "real_vector",
        "positive",
        "positive_vector",
        "positive_matrix",
        "unit_interval",
        "simplex",
        "simplex_vector",
        "row_simplex_matrix",
        "column_simplex_matrix",
        "integer",
        "integer_vector",
        "integer_matrix",
        "positive_integer",
        "non_negative_integer",
        "optional_integer",
    }
    if constraint in finite_constraints and np.any(~np.isfinite(arr)):
        return ("%s must contain only finite values." % name,)
    if constraint in ("positive", "positive_vector") and np.any(arr <= 0.0):
        return ("%s must be strictly positive." % name,)
    if constraint == "positive_matrix":
        if arr.shape[0] == arr.shape[1] and np.allclose(arr, arr.T):
            if np.any(np.linalg.eigvalsh(arr) <= 0.0):
                return ("%s must be positive definite." % name,)
        elif np.any(arr <= 0.0):
            return ("%s must contain strictly positive entries." % name,)
    if constraint == "unit_interval" and (float(arr) < 0.0 or float(arr) > 1.0):
        return ("%s must lie in [0, 1]." % name,)
    if constraint in ("simplex", "simplex_vector"):
        if np.any(arr < 0.0) or not np.isclose(arr.sum(), 1.0, rtol=1.0e-7, atol=1.0e-10):
            return ("%s must be a probability simplex." % name,)
    if constraint in ("row_simplex_matrix", "column_simplex_matrix"):
        axis = 1 if constraint == "row_simplex_matrix" else 0
        if np.any(arr < 0.0) or np.any(
            ~np.isclose(arr.sum(axis=axis), 1.0, rtol=1.0e-7, atol=1.0e-10)
        ):
            return ("%s must contain probability-simplex %s." % (name, "rows" if axis == 1 else "columns"),)
    if constraint in (
        "integer",
        "integer_vector",
        "integer_matrix",
        "positive_integer",
        "non_negative_integer",
        "optional_integer",
    ) and np.any(arr != np.floor(arr)):
        return ("%s must contain exact integers." % name,)
    if constraint == "positive_integer" and np.any(arr <= 0):
        return ("%s must be a positive integer." % name,)
    if constraint == "non_negative_integer" and np.any(arr < 0):
        return ("%s must be a non-negative integer." % name,)
    if constraint in ("log_unit_interval_vector", "optional_log_unit_interval_vector"):
        if constraint.startswith("optional") and value is None:
            return ()
        if arr.ndim != 1 or np.any(arr > 0.0):
            return ("%s must be a one-dimensional log-probability vector." % name,)
    return ()


def _validated_parameter_values(dist: Any, declaration: DistributionDeclaration) -> dict[str, Any]:
    if not isinstance(dist, declaration.distribution_type):
        raise TypeError(
            "expected %s, got %s" % (declaration.distribution_type.__name__, type(dist).__name__)
        )
    values = {}
    issues = []
    for spec in declaration.parameters:
        if not hasattr(dist, spec.name):
            issues.append("%s has no declared parameter %s." % (type(dist).__name__, spec.name))
            continue
        value = getattr(dist, spec.name)
        values[spec.name] = value
        issues.extend(_parameter_value_issues(spec, value))
    for spec in declaration.parameters:
        if not _is_ordered_constraint(spec.constraint) or spec.name not in values:
            continue
        anchor_name = _ordered_constraint_anchor(spec.constraint)
        if anchor_name not in values:
            continue
        try:
            value = float(values[spec.name])
            anchor = float(values[anchor_name])
        except (TypeError, ValueError):
            issues.append("%s and ordered anchor %s must be scalar numeric values." % (spec.name, anchor_name))
            continue
        if spec.constraint.startswith("greater_than:") and not value > anchor:
            issues.append("%s must be greater than %s." % (spec.name, anchor_name))
        if spec.constraint.startswith("less_than:") and not value < anchor:
            issues.append("%s must be less than %s." % (spec.name, anchor_name))
    if issues:
        raise ValueError(
            "Invalid declared parameters for %s: %s"
            % (type(dist).__name__, "; ".join(issues))
        )
    return values


def generated_stacked_params(dists: Sequence[Any], engine: Any) -> dict[str, Any]:
    """Stack declared distribution parameters for generated homogeneous-mixture scoring.

    The generated path is intentionally conservative: it supports scalar,
    vector, or matrix per-component parameters that can be broadcast over
    per-row encoded fields. Non-differentiable support metadata such as integer
    bounds must still be shared across components unless a family keeps an
    explicit ``backend_stacked_*`` route.
    """
    if not dists:
        raise ValueError("generated_stacked_params requires at least one component.")
    dist_type = type(dists[0])
    if any(type(dist) is not dist_type for dist in dists):
        raise ValueError("generated stacked scoring requires homogeneous component types.")
    declaration = declaration_for(dists[0])
    if declaration is None:
        raise ValueError("%s has no declaration." % dist_type.__name__)
    if declaration.exponential_family is None and not _generated_backend_hook_supported(dist_type, declaration):
        raise ValueError("%s has no generated stacked scoring hook." % dist_type.__name__)

    params: dict[str, Any] = {
        "__pysp_dist_type__": dist_type,
        "__pysp_param_names__": tuple(spec.name for spec in declaration.parameters),
        "__pysp_component_count__": len(dists),
    }
    declared_values = [_validated_parameter_values(dist, declaration) for dist in dists]
    for spec in declaration.parameters:
        values = [values[spec.name] for values in declared_values]
        if not spec.differentiable and _generated_stacked_requires_shared_param(spec):
            if _all_same(values):
                params[spec.name] = values[0]
                continue
            raise KernelCapabilityDeclinedError(
                "generated stacked fixed parameter %s must match across components." % spec.name
            )
        if _all_same(values) and spec.constraint in ("fixed", "optional_integer"):
            params[spec.name] = values[0]
            continue
        arr = np.asarray(values)
        if arr.dtype.kind in ("O", "U", "S"):
            if _all_same(values):
                params[spec.name] = values[0]
                continue
            raise KernelCapabilityDeclinedError("generated stacked parameter %s is not numeric." % spec.name)
        max_ndim = 3 if declaration.exponential_family is not None else 2
        if arr.ndim > max_ndim:
            raise KernelCapabilityDeclinedError(
                "generated stacked parameter %s has unsupported rank %d." % (spec.name, arr.ndim)
            )
        params[spec.name] = engine.asarray(arr)
    return params


def generated_stacked_log_density(enc: Any, params: dict[str, Any], engine: Any) -> Any:
    """Return an ``(n, k)`` log-density matrix from declaration-stacked params."""
    dist_type = params["__pysp_dist_type__"]
    declaration = declaration_for(dist_type)
    if declaration is not None and _exp_family_stacked_scoring(declaration):
        result = _generated_exp_family_log_density(enc, params, declaration.exponential_family, engine)
    else:
        fn = dist_type.backend_log_density_from_params
        sig_names = tuple(inspect.signature(fn).parameters.keys())
        if not sig_names or sig_names[-1] != "engine":
            raise ValueError("%s backend_log_density_from_params must end with engine." % dist_type.__name__)
        call_names = sig_names[:-1]
        param_names = set(params.get("__pysp_param_names__", ()))
        data_count = 0
        for name in call_names:
            if name in param_names:
                break
            data_count += 1
        if data_count <= 0:
            raise ValueError("%s generated scorer could not infer encoded arguments." % dist_type.__name__)

        args = list(_generated_data_args(enc, data_count, engine))
        for name in call_names[data_count:]:
            if name not in param_names:
                raise ValueError("%s generated scorer requires undeclared parameter %s." % (dist_type.__name__, name))
            args.append(_generated_param_arg(params[name], engine))
        args.append(engine)
        result = fn(*args)
    return _validated_generated_score(
        result,
        "%s generated stacked score" % dist_type.__name__,
        component_count=params.get("__pysp_component_count__"),
    )


def generated_log_density(dist: Any, enc: Any, engine: Any) -> Any:
    """Return per-row log densities from declaration-owned scoring metadata.

    This is the single-distribution counterpart of
    ``generated_stacked_log_density``.  It gives generic kernels an engine-aware
    fallback before they drop to legacy NumPy ``seq_log_density`` methods, and
    it stays fully metadata-driven: no caller imports or switches on concrete
    distribution implementations.
    """
    declaration = declaration_for(dist)
    if declaration is None:
        raise ValueError("%s has no declaration." % type(dist).__name__)
    params = _generated_scalar_params(dist, declaration, engine)
    if _exp_family_runtime_scoring(declaration):
        result = _generated_exp_family_scalar_expression(enc, params, declaration.exponential_family, engine)
    else:
        result = _generated_backend_log_density(dist, enc, params, declaration, engine)
    return _validated_generated_score(result, "%s generated score" % type(dist).__name__)


def generated_log_density_available(x: Any) -> bool:
    """Return whether declaration metadata exposes a scalar runtime scorer."""
    declaration = declaration_for(x)
    if declaration is None:
        return False
    return _exp_family_runtime_scoring(declaration) or _generated_backend_hook_supported(
        declaration.distribution_type, declaration
    )


def generated_stacked_sufficient_statistics_available(x: Any) -> bool:
    """Return true when a declaration can generate resident stacked stats."""
    dist_type = x.get("__pysp_dist_type__") if isinstance(x, dict) else x
    declaration = declaration_for(dist_type)
    return declaration is not None and callable(_legacy_sufficient_statistics_fn(declaration))


def generated_sufficient_statistics_available(x: Any) -> bool:
    """Return true when a declaration can generate single-distribution stats."""
    declaration = declaration_for(x)
    return declaration is not None and callable(_legacy_sufficient_statistics_fn(declaration))


def generated_sufficient_statistics(dist: Any, enc: Any, weights: Any, engine: Any) -> tuple[Any, ...]:
    """Return legacy sufficient statistics from declaration-owned row stats.

    This is the single-distribution analogue of
    ``generated_stacked_sufficient_statistics``.  It performs row-wise
    sufficient-statistic algebra on the active engine, then converts only the
    small legacy payload back across the boundary for the existing estimator
    M-step.
    """
    declaration = declaration_for(dist)
    if declaration is None:
        raise ValueError("%s has no declaration." % type(dist).__name__)
    stats_fn = _legacy_sufficient_statistics_fn(declaration)
    if not callable(stats_fn):
        raise ValueError("%s has no generated legacy sufficient-statistic hook." % type(dist).__name__)
    params = _generated_scalar_params(dist, declaration, engine)
    row_stats = tuple(stats_fn(enc, params, engine))
    if len(row_stats) != len(declaration.statistics):
        raise ValueError(
            "%s generated %d legacy statistics for %d declared statistics."
            % (type(dist).__name__, len(row_stats), len(declaration.statistics))
        )
    ww = engine.asarray(weights)
    _validate_generated_statistic_rows(row_stats, declaration.statistics, ww)
    return tuple(
        _weighted_histogram(stat, ww, engine)
        if spec.kind == "histogram"
        else _host_legacy_value(_weighted_row_sum(stat, spec, ww, engine), engine)
        for spec, stat in zip(declaration.statistics, row_stats)
    )


def generated_numba_log_density_available(x: Any) -> bool:
    """Return true when declarations can emit a generated numba leaf scorer.

    Exponential-family leaves use the stacked exp-family loop; other leaves with a
    ``backend_log_density_from_params`` hook whose per-row formula lowers cleanly to a numba
    scalar loop (single encoded array, supported ops, scalar parameters) use the generic
    symbolic-to-numba compiler in :func:`_build_generic_numba_kernel`.
    """
    declaration = declaration_for(x)
    if declaration is None:
        return False
    if _exp_family_runtime_scoring(declaration):
        return True
    dist_type = x if isinstance(x, type) else type(x)
    return _build_generic_numba_kernel(dist_type, declaration) is not None


def generated_numba_stacked_available(x: Any) -> bool:
    """Return true when declarations can emit a generated stacked numba scorer."""
    declaration = declaration_for(x)
    return declaration is not None and _exp_family_stacked_scoring(declaration)


def generated_numba_log_density(dist: Any, enc: Any) -> np.ndarray:
    """Return per-row log densities from a declaration-generated numba loop.

    The generated loop evaluates the exponential-family form
    ``base(x) + T(x) dot eta(theta) - A(theta)``.  Distribution declarations
    still own the statistical metadata; this helper only lowers the row fold to
    a nopython scalar loop.
    """
    from mixle.engines import NUMPY_ENGINE

    declaration = declaration_for(dist)
    if declaration is None:
        raise ValueError("%s has no declaration." % type(dist).__name__)
    if not _exp_family_runtime_scoring(declaration):
        return _generated_generic_numba_log_density(dist, enc, declaration)
    params = _generated_scalar_params(dist, declaration, NUMPY_ENGINE)
    row_stats, base = _generated_numba_row_pieces(enc, params, declaration.exponential_family)
    eta = _generated_numba_eta_vector(params, declaration.exponential_family)
    log_partition = _generated_numba_scalar(
        declaration.exponential_family.log_partition(params, NUMPY_ENGINE), "log_partition"
    )
    if eta.shape[0] != row_stats.shape[1]:
        raise ValueError("generated numba statistic/natural-parameter widths differ.")
    out = np.empty(row_stats.shape[0], dtype=np.float64)
    _numba_exp_family_log_density(row_stats, base, eta, float(log_partition), out)
    return _validated_generated_score(out, "%s generated numba score" % type(dist).__name__)


# ---------------------------------------------------------------------------
# Generic symbolic -> numba lowering for non-exponential-family leaves.
#
# Many leaves (Laplace, Logistic, StudentT, Weibull, Pareto, ...) are not fixed-base exponential
# families, so they cannot use the stacked exp-family loop. They do own an engine-neutral
# ``backend_log_density_from_params(data, *params, engine)`` whose per-row formula traces to a
# SymbolicExpression. This compiler lowers that expression to a numba nopython scalar loop, giving
# those families a real generated kernel without re-implementing their math.
# ---------------------------------------------------------------------------


class _UnsupportedNumbaLowering(Exception):
    """Raised when a symbolic expression cannot be lowered to the numba scalar loop."""


_NUMBA_INFIX_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "**",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
}

_NUMBA_FUNC_OPS = {
    "log": "math.log",
    "exp": "math.exp",
    "sqrt": "math.sqrt",
    "abs": "abs",
    "floor": "math.floor",
    "gammaln": "math.lgamma",
    "erf": "math.erf",
    "isnan": "math.isnan",
    "isinf": "math.isinf",
}

_GENERIC_NUMBA_KERNEL_CACHE: dict[type[Any], tuple[Any, int, tuple[str, ...]] | None] = {}


_NUMBA_NAMED_CONSTANTS = {
    "pi": "math.pi",
    "e": "math.e",
    "euler_gamma": repr(0.5772156649015328606),  # no math.euler_gamma in the stdlib
    "inf": "_INF",
}


def _lower_symbolic_to_numba(expr: Any) -> str:
    """Lower a SymbolicExpression to a numba-compatible Python expression string."""
    op = expr.op
    if op == "symbol":
        return str(expr.args[0])
    if op == "const":
        value = expr.args[0]
        if isinstance(value, bool):
            return "True" if value else "False"
        if value is None:
            raise _UnsupportedNumbaLowering("None constant")
        fvalue = float(value)
        if math.isinf(fvalue):
            return "_INF" if fvalue > 0 else "(-_INF)"
        if math.isnan(fvalue):
            return "_NAN"
        return repr(fvalue)
    if op in _NUMBA_NAMED_CONSTANTS and not expr.args:
        # nullary named constants (SymbolicEngine.pi/e/euler_gamma/inf) -- distinct from "const",
        # which wraps an actual numeric value. These entered traced expressions for the first time
        # once engine-parameterized backend formulas started reading engine.pi instead of a frozen
        # Python float baked in at trace time (see gaussian.py/student_t.py/etc.'s pi fix).
        return _NUMBA_NAMED_CONSTANTS[op]

    sub = [_lower_symbolic_to_numba(arg) for arg in expr.args]

    if op in _NUMBA_INFIX_OPS and len(sub) == 2:
        return "(%s %s %s)" % (sub[0], _NUMBA_INFIX_OPS[op], sub[1])
    if op == "neg" and len(sub) == 1:
        return "(-%s)" % sub[0]
    if op == "invert" and len(sub) == 1:
        return "(not %s)" % sub[0]
    if op == "and" and len(sub) == 2:
        return "(%s and %s)" % (sub[0], sub[1])
    if op == "or" and len(sub) == 2:
        return "(%s or %s)" % (sub[0], sub[1])
    if op == "where" and len(sub) == 3:
        return "(%s if %s else %s)" % (sub[1], sub[0], sub[2])
    if op == "max":
        return "(max(%s))" % ", ".join(sub)
    if op == "betaln" and len(sub) == 2:
        return "(math.lgamma(%s) + math.lgamma(%s) - math.lgamma((%s) + (%s)))" % (sub[0], sub[1], sub[0], sub[1])
    if op == "clip" and len(sub) == 3:
        result = sub[0]
        lo, hi = expr.args[1], expr.args[2]
        if not (lo.op == "const" and lo.args[0] is None):
            result = "max(%s, %s)" % (result, sub[1])
        if not (hi.op == "const" and hi.args[0] is None):
            result = "min(%s, %s)" % (result, sub[2])
        return "(%s)" % result
    if op in _NUMBA_FUNC_OPS:
        return "%s(%s)" % (_NUMBA_FUNC_OPS[op], ", ".join(sub))

    raise _UnsupportedNumbaLowering(op)


def _build_generic_numba_kernel(
    dist_type: type[Any], declaration: DistributionDeclaration
) -> tuple[Any, int, tuple[str, ...]] | None:
    """Compile (and cache) a numba scalar-loop kernel for a non-exp-family leaf, or None.

    Returns ``(kernel, n_data, ordered_param_names)`` where the kernel has signature
    ``kernel(*data_arrays, *param_scalars, out)``; returns ``None`` when the family cannot be
    lowered (vector encoded data, a non-parameter call argument, or an unsupported operation).
    """
    if dist_type in _GENERIC_NUMBA_KERNEL_CACHE:
        return _GENERIC_NUMBA_KERNEL_CACHE[dist_type]

    try:
        from mixle.engines.symbolic_engine import SymbolicEngine

        fn = getattr(dist_type, "backend_log_density_from_params", None)
        if not callable(fn):
            raise _UnsupportedNumbaLowering("no backend_log_density_from_params")

        encoded_names = _diagnostic_encoded_symbols(dist_type, declaration, None)
        if not encoded_names or declaration.support.endswith("_vector"):
            raise _UnsupportedNumbaLowering("non-scalar encoded data")
        data_names = tuple(encoded_names)
        n_data = len(data_names)

        sig_names = tuple(inspect.signature(fn).parameters.keys())
        if not sig_names or sig_names[-1] != "engine":
            raise _UnsupportedNumbaLowering("backend signature must end with engine")
        call_names = sig_names[:-1]
        param_names = set(declaration.parameter_names)
        # The leading data arguments, then the parameters in call order.
        ordered_params = tuple(call_names[n_data:])
        for name in ordered_params:
            if name not in param_names:
                raise _UnsupportedNumbaLowering("non-parameter call argument %r" % name)

        engine = SymbolicEngine()
        param_symbols = _diagnostic_param_symbols(declaration, engine)
        for name in ordered_params:
            sym = param_symbols.get(name)
            # scalar symbols only; vector/matrix params become numpy arrays and are unsupported here
            if not hasattr(sym, "op"):
                raise _UnsupportedNumbaLowering("non-scalar parameter %r" % name)
        encoded_values = tuple(engine.symbol(name) for name in data_names)
        expr = _diagnostic_backend_log_density_expression(dist_type, param_symbols, encoded_values, engine)
        body = _lower_symbolic_to_numba(expr)

        data_args = tuple("_data%d" % k for k in range(n_data))
        arg_list = ", ".join(data_args + ordered_params + ("out",))
        bind_lines = "".join("        %s = _data%d[_i]\n" % (name, k) for k, name in enumerate(data_names))
        source = (
            "def _generic_numba_kernel(%s):\n    _n = out.shape[0]\n    for _i in range(_n):\n%s        out[_i] = %s\n"
        ) % (arg_list, bind_lines, body)

        namespace: dict[str, Any] = {"math": math, "np": np, "_INF": np.inf, "_NAN": np.nan}
        exec(compile(source, "<generated_numba:%s>" % dist_type.__name__, "exec"), namespace)
        kernel = numba.njit(cache=False)(namespace["_generic_numba_kernel"])
        result = (kernel, n_data, ordered_params)
    except _UnsupportedNumbaLowering:
        _GENERIC_NUMBA_KERNEL_CACHE[dist_type] = None
        return None
    except Exception as exc:
        raise GeneratedKernelCompilationError(
            "%s generated numba compilation failed: %s" % (dist_type.__name__, exc)
        ) from exc

    _GENERIC_NUMBA_KERNEL_CACHE[dist_type] = result
    return result


def _generic_numba_data_arrays(enc: Any, n_data: int) -> tuple[np.ndarray, ...]:
    """Return the ``n_data`` per-row encoded arrays as contiguous 1-D float64 vectors."""
    raw = (enc,) if n_data == 1 else tuple(enc[:n_data])
    if len(raw) != n_data:
        raise ValueError("encoded payload does not contain %d generated arrays." % n_data)
    arrays = []
    row_count = None
    for arg in raw:
        arr = np.asarray(arg)
        if arr.ndim != 1:
            raise ValueError("generated generic numba encoded arrays must be one-dimensional.")
        try:
            arr = np.ascontiguousarray(arr, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("generated generic numba encoded arrays must be numeric.") from exc
        if row_count is None:
            row_count = arr.shape[0]
        elif arr.shape[0] != row_count:
            raise ValueError("generated generic numba encoded arrays have inconsistent row counts.")
        if np.any(np.isnan(arr)) or np.any(np.isposinf(arr)):
            raise ValueError("generated generic numba encoded arrays cannot contain NaN or positive infinity.")
        arrays.append(arr)
    return tuple(arrays)


def _generated_generic_numba_log_density(dist: Any, enc: Any, declaration: DistributionDeclaration) -> np.ndarray:
    built = _build_generic_numba_kernel(type(dist), declaration)
    if built is None:
        raise ValueError("%s has no generated numba kernel." % type(dist).__name__)
    kernel, n_data, ordered_params = built
    declared_values = _validated_parameter_values(dist, declaration)
    param_values = []
    for name in ordered_params:
        value = declared_values.get(name)
        if value is None or not isinstance(value, (bool, int, float, np.number)):
            raise ValueError(
                "%s parameter %r is not a scalar usable in the generated numba kernel." % (type(dist).__name__, name)
            )
        param_values.append(float(value))
    data_arrays = _generic_numba_data_arrays(enc, n_data)
    out = np.empty(data_arrays[0].shape[0], dtype=np.float64)
    kernel(*data_arrays, *param_values, out)
    return _validated_generated_score(out, "%s generated numba score" % type(dist).__name__)


def generated_numba_stacked_log_density(enc: Any, params: dict[str, Any]) -> np.ndarray:
    """Return an ``(n, k)`` score matrix from a declaration-generated numba loop."""
    dist_type = params["__pysp_dist_type__"]
    declaration = declaration_for(dist_type)
    if declaration is None or declaration.exponential_family is None:
        raise ValueError("%s has no exponential-family declaration." % dist_type.__name__)
    row_stats, base = _generated_numba_row_pieces(enc, params, declaration.exponential_family)
    eta = _generated_numba_eta_matrix(params, declaration.exponential_family)
    log_partition = _generated_numba_vector(
        declaration.exponential_family.log_partition(params, _numpy_engine()),
        "log_partition",
    )
    if eta.shape[0] != log_partition.shape[0]:
        raise ValueError("generated numba eta/log-partition component counts differ.")
    if eta.shape[1] != row_stats.shape[1]:
        raise ValueError("generated numba statistic/natural-parameter widths differ.")
    k = eta.shape[0]
    if base.ndim == 1:
        base = np.ascontiguousarray(np.broadcast_to(base[:, None], (base.shape[0], k)))
    elif base.shape[1] != k:
        raise ValueError("generated numba base_measure component count differs from eta.")
    out = np.empty((row_stats.shape[0], k), dtype=np.float64)
    _numba_stacked_exp_family_log_density(row_stats, base, eta, log_partition, out)
    return _validated_generated_score(
        out,
        "%s generated stacked numba score" % dist_type.__name__,
        component_count=params.get("__pysp_component_count__", k),
    )


def generated_stacked_sufficient_statistics(
    enc: Any, weights: Any, params: dict[str, Any], engine: Any
) -> tuple[Any, ...]:
    """Return component-stacked legacy sufficient statistics from declarations."""
    dist_type = params["__pysp_dist_type__"]
    declaration = declaration_for(dist_type)
    if declaration is None:
        raise ValueError("%s has no declaration." % dist_type.__name__)
    stats_fn = _legacy_sufficient_statistics_fn(declaration)
    if not callable(stats_fn):
        raise ValueError("%s has no generated legacy sufficient-statistic hook." % dist_type.__name__)
    row_stats = tuple(stats_fn(enc, params, engine))
    if len(row_stats) != len(declaration.statistics):
        raise ValueError(
            "%s generated %d legacy statistics for %d declared statistics."
            % (dist_type.__name__, len(row_stats), len(declaration.statistics))
        )
    ww = engine.asarray(weights)
    _validate_generated_statistic_rows(
        row_stats,
        declaration.statistics,
        ww,
        component_count=params.get("__pysp_component_count__"),
    )
    return tuple(
        _weighted_component_sum(stat, spec, ww, engine) for spec, stat in zip(declaration.statistics, row_stats)
    )


def _generated_exp_family_log_density(
    enc: Any, params: dict[str, Any], spec: ExponentialFamilySpec, engine: Any
) -> Any:
    if spec.sufficient_statistics_from_params is not None:
        statistics = tuple(spec.sufficient_statistics_from_params(enc, params, engine))
    else:
        statistics = tuple(spec.sufficient_statistics(enc, engine))
    natural = tuple(spec.natural_parameters(params, engine))
    if len(statistics) != len(natural):
        raise ValueError("exponential-family statistic/natural-parameter arity mismatch.")
    if not statistics:
        raise ValueError("exponential-family declarations require at least one statistic.")
    if spec.base_measure_from_params is not None:
        base = spec.base_measure_from_params(enc, params, engine)
    else:
        base = (
            spec.base_measure(enc, engine)
            if spec.base_measure is not None
            else _generated_exp_family_zero_base(statistics[0], engine)
        )
    # A component-dependent base measure already carries an ``(n, k)`` component axis
    # (e.g. a degenerate boundary that the smooth natural parameter cannot represent);
    # a shared ``(n,)`` base is broadcast across components via the trailing axis.
    rv = base if len(tuple(getattr(base, "shape", ()))) >= 2 else base[:, None]
    for stat, eta in zip(statistics, natural):
        rv = rv + _generated_exp_family_pair_term(stat, eta, engine, stacked=True)
    return rv - _generated_param_arg(spec.log_partition(params, engine), engine)


def _generated_exp_family_scalar_expression(
    enc: Any, params: dict[str, Any], spec: ExponentialFamilySpec, engine: Any
) -> Any:
    if spec.sufficient_statistics_from_params is not None:
        statistics = tuple(spec.sufficient_statistics_from_params(enc, params, engine))
    else:
        statistics = tuple(spec.sufficient_statistics(enc, engine))
    natural = tuple(spec.natural_parameters(params, engine))
    if len(statistics) != len(natural):
        raise ValueError("exponential-family statistic/natural-parameter arity mismatch.")
    if not statistics:
        raise ValueError("exponential-family declarations require at least one statistic.")
    if spec.base_measure_from_params is not None:
        base = spec.base_measure_from_params(enc, params, engine)
    else:
        base = (
            spec.base_measure(enc, engine)
            if spec.base_measure is not None
            else _generated_exp_family_zero_base(statistics[0], engine)
        )
    rv = base
    for stat, eta in zip(statistics, natural):
        rv = rv + _generated_exp_family_pair_term(stat, eta, engine, stacked=False)
    return rv - spec.log_partition(params, engine)


def _generated_exp_family_pair_term(stat: Any, eta: Any, engine: Any, stacked: bool) -> Any:
    stat_arr = engine.asarray(stat)
    stat_shape = tuple(getattr(stat_arr, "shape", ()))
    if stacked and len(stat_shape) >= 2 and not isinstance(eta, (str, bytes, bool, int, float, np.number)):
        # Stacked Frobenius inner product <stat[n], eta[k]> summed over the feature
        # axes. stat is (n, *feat), the raw natural parameter eta is (k, *feat), and
        # the result is (n, k). Computed as a single matmul over the flattened feature
        # axes so the (n, k, *feat) broadcast tensor is never materialized — for a
        # matrix (second-moment) statistic that intermediate is N*K*dim*dim (~20 GB at
        # n=2e4, k=8, dim=128, which OOMs a GPU); the gemm result is only (n, k).
        eta_arr = engine.asarray(eta)
        eta_shape = tuple(getattr(eta_arr, "shape", ()))
        if len(eta_shape) >= 1 and eta_shape[1:] == stat_shape[1:]:
            n_rows, k = stat_shape[0], eta_shape[0]
            return engine.matmul(stat_arr.reshape(n_rows, -1), eta_arr.reshape(k, -1).T)
    eta_arr = _generated_param_arg(eta, engine) if stacked else eta
    product = stat_arr[:, None] * eta_arr if stacked else stat_arr * eta_arr
    shape = tuple(getattr(product, "shape", ()))
    if stacked:
        while len(shape) > 2:
            product = engine.sum(product, axis=-1)
            shape = tuple(getattr(product, "shape", ()))
    else:
        while len(shape) > 1:
            product = engine.sum(product, axis=-1)
            shape = tuple(getattr(product, "shape", ()))
    return product


def _generated_exp_family_zero_base(stat: Any, engine: Any) -> Any:
    base = engine.asarray(stat) * engine.asarray(0.0)
    shape = tuple(getattr(base, "shape", ()))
    while len(shape) > 1:
        base = engine.sum(base, axis=-1)
        shape = tuple(getattr(base, "shape", ()))
    return base


def _diagnostic_encoded_symbols(
    dist_type: type[Any], declaration: DistributionDeclaration, encoded_symbols: Sequence[str] | None
) -> tuple[str, ...]:
    if encoded_symbols is not None:
        if isinstance(encoded_symbols, str):
            return (encoded_symbols,)
        return tuple(str(name) for name in encoded_symbols)
    fn = getattr(dist_type, "backend_log_density_from_params", None)
    if not callable(fn):
        return ("x",)
    sig_names = tuple(inspect.signature(fn).parameters.keys())
    if not sig_names or sig_names[-1] != "engine":
        return ("x",)
    param_names = set(declaration.parameter_names)
    data_names = []
    for name in sig_names[:-1]:
        if name in param_names:
            break
        data_names.append(name)
    return tuple(data_names) if data_names else ("x",)


def _diagnostic_encoded_values(
    encoded_names: Sequence[str], engine: Any, declaration: DistributionDeclaration
) -> tuple[Any, ...]:
    if len(encoded_names) == 1 and declaration.support.endswith("_vector"):
        return (
            np.asarray(
                [
                    engine.symbol("%s_0" % encoded_names[0]),
                    engine.symbol("%s_1" % encoded_names[0]),
                ],
                dtype=object,
            ),
        )
    return tuple(engine.symbol(name) for name in encoded_names)


def _diagnostic_param_symbols(declaration: DistributionDeclaration, engine: Any) -> dict[str, Any]:
    params = {}
    for spec in declaration.parameters:
        if spec.constraint in (
            "real_vector",
            "positive_vector",
            "simplex",
            "simplex_vector",
            "integer_vector",
            "log_unit_interval_vector",
            "optional_log_unit_interval_vector",
        ):
            params[spec.name] = np.asarray(
                [
                    engine.symbol("%s_0" % spec.name),
                    engine.symbol("%s_1" % spec.name),
                ],
                dtype=object,
            )
        elif spec.constraint in (
            "positive_matrix",
            "row_simplex_matrix",
            "column_simplex_matrix",
            "integer_matrix",
        ):
            params[spec.name] = np.asarray(
                [
                    [engine.symbol("%s_00" % spec.name), engine.symbol("%s_01" % spec.name)],
                    [engine.symbol("%s_10" % spec.name), engine.symbol("%s_11" % spec.name)],
                ],
                dtype=object,
            )
        else:
            params[spec.name] = engine.symbol(spec.name)
    return params


def _diagnostic_backend_log_density_expression(
    dist_type: type[Any], params: dict[str, Any], encoded_values: Sequence[Any], engine: Any
) -> Any:
    fn = getattr(dist_type, "backend_log_density_from_params", None)
    if not callable(fn):
        raise ValueError("%s has no generated log-density hook." % dist_type.__name__)
    sig_names = tuple(inspect.signature(fn).parameters.keys())
    if not sig_names or sig_names[-1] != "engine":
        raise ValueError("%s backend_log_density_from_params must end with engine." % dist_type.__name__)
    args = []
    encoded_iter = iter(encoded_values)
    for name in sig_names[:-1]:
        if name in params:
            args.append(params[name])
        else:
            try:
                args.append(next(encoded_iter))
            except StopIteration:
                args.append(engine.symbol(name))
    args.append(engine)
    return fn(*args)


@numba.njit(cache=True)
def _numba_exp_family_log_density(
    row_stats: np.ndarray, base: np.ndarray, eta: np.ndarray, log_partition: float, out: np.ndarray
) -> None:
    n, m = row_stats.shape
    for i in range(n):
        value = base[i]
        for j in range(m):
            value += row_stats[i, j] * eta[j]
        out[i] = value - log_partition


@numba.njit(cache=True)
def _numba_stacked_exp_family_log_density(
    row_stats: np.ndarray, base: np.ndarray, eta: np.ndarray, log_partition: np.ndarray, out: np.ndarray
) -> None:
    n, m = row_stats.shape
    k_count = eta.shape[0]
    for i in range(n):
        for k in range(k_count):
            value = base[i, k]
            for j in range(m):
                value += row_stats[i, j] * eta[k, j]
            out[i, k] = value - log_partition[k]


def _numpy_engine() -> Any:
    from mixle.engines import NUMPY_ENGINE

    return NUMPY_ENGINE


def _generated_numba_row_pieces(
    enc: Any, params: dict[str, Any], spec: ExponentialFamilySpec
) -> tuple[np.ndarray, np.ndarray]:
    engine = _numpy_engine()
    if spec.sufficient_statistics_from_params is not None:
        statistics = tuple(spec.sufficient_statistics_from_params(enc, params, engine))
    else:
        statistics = tuple(spec.sufficient_statistics(enc, engine))
    row_stats = _generated_numba_row_matrix(statistics)
    if spec.base_measure_from_params is not None:
        base_value = spec.base_measure_from_params(enc, params, engine)
    elif spec.base_measure is not None:
        base_value = spec.base_measure(enc, engine)
    else:
        base_value = np.zeros(row_stats.shape[0], dtype=np.float64)
    base = _generated_numba_base(base_value, row_stats.shape[0])
    return row_stats, base


def _generated_numba_row_matrix(statistics: Sequence[Any]) -> np.ndarray:
    if not statistics:
        raise ValueError("generated numba scoring requires at least one row statistic.")
    columns = []
    row_count = None
    for idx, stat in enumerate(statistics):
        arr = np.asarray(_numpy_engine().to_numpy(stat), dtype=np.float64)
        if arr.ndim < 1:
            raise ValueError("generated numba statistic %d must have a row axis." % idx)
        if row_count is None:
            row_count = arr.shape[0]
        elif arr.shape[0] != row_count:
            raise ValueError("generated numba row statistics have inconsistent lengths.")
        columns.append(arr.reshape((arr.shape[0], -1)))
    return np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.float64)


def _generated_numba_base(value: Any, row_count: int) -> np.ndarray:
    """Return a base measure as a scalar-filled ``(n,)`` vector or a component ``(n, k)`` matrix.

    A component-dependent base measure (one a single shared natural parameter cannot
    encode, such as a degenerate exponential-family boundary) keeps its ``(n, k)``
    component axis so the stacked numba loop can index it per component.
    """
    arr = np.asarray(_numpy_engine().to_numpy(value), dtype=np.float64)
    if arr.ndim == 0:
        return np.full(row_count, float(arr), dtype=np.float64)
    if arr.shape[0] != row_count:
        raise ValueError("generated numba base_measure must have a matching row axis.")
    if arr.ndim in (1, 2):
        return np.ascontiguousarray(arr, dtype=np.float64)
    raise ValueError("generated numba base_measure must be scalar, a row vector, or an (n, k) matrix.")


def _generated_numba_eta_vector(params: dict[str, Any], spec: ExponentialFamilySpec) -> np.ndarray:
    natural = tuple(spec.natural_parameters(params, _numpy_engine()))
    if not natural:
        raise ValueError("generated numba scoring requires at least one natural parameter.")
    parts = [_generated_numba_flat_vector(value, "natural_parameter_%d" % idx) for idx, value in enumerate(natural)]
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)


def _generated_numba_eta_matrix(params: dict[str, Any], spec: ExponentialFamilySpec) -> np.ndarray:
    natural = tuple(spec.natural_parameters(params, _numpy_engine()))
    if not natural:
        raise ValueError("generated numba scoring requires at least one natural parameter.")
    columns = []
    component_count = None
    for idx, value in enumerate(natural):
        arr = _generated_numba_component_matrix(value, "natural_parameter_%d" % idx)
        if component_count is None:
            component_count = arr.shape[0]
        elif arr.shape[0] != component_count:
            raise ValueError("generated numba natural-parameter component counts differ.")
        columns.append(arr)
    return np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.float64)


def _generated_numba_scalar(value: Any, name: str) -> float:
    arr = np.asarray(_numpy_engine().to_numpy(value), dtype=np.float64)
    if arr.ndim != 0:
        raise ValueError("generated numba %s must be scalar." % name)
    return float(arr)


def _generated_numba_vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(_numpy_engine().to_numpy(value), dtype=np.float64)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("generated numba %s must be scalar or one-dimensional." % name)
    return np.ascontiguousarray(arr, dtype=np.float64)


def _generated_numba_flat_vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(_numpy_engine().to_numpy(value), dtype=np.float64)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(-1), dtype=np.float64)


def _generated_numba_component_matrix(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(_numpy_engine().to_numpy(value), dtype=np.float64)
    if arr.ndim == 0:
        return np.asarray([[float(arr)]], dtype=np.float64)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr.reshape((-1, 1)), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape((arr.shape[0], -1)), dtype=np.float64)


def _generated_scalar_params(dist: Any, declaration: DistributionDeclaration, engine: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    values = _validated_parameter_values(dist, declaration)
    for spec in declaration.parameters:
        value = values[spec.name]
        if value is None or isinstance(value, (str, bytes, bool, int, float, np.number, type)):
            params[spec.name] = value
        else:
            params[spec.name] = engine.asarray(value)
    return params


def _generated_backend_log_density(
    dist: Any, enc: Any, params: dict[str, Any], declaration: DistributionDeclaration, engine: Any
) -> Any:
    dist_type = type(dist)
    fn = getattr(dist_type, "backend_log_density_from_params", None)
    if not callable(fn):
        raise ValueError("%s has no generated log-density hook." % dist_type.__name__)
    sig_names = tuple(inspect.signature(fn).parameters.keys())
    if not sig_names or sig_names[-1] != "engine":
        raise ValueError("%s backend_log_density_from_params must end with engine." % dist_type.__name__)
    call_names = sig_names[:-1]
    param_names = set(declaration.parameter_names)
    data_count = 0
    for name in call_names:
        if name in param_names:
            break
        data_count += 1
    if data_count <= 0:
        raise ValueError("%s generated scorer could not infer encoded arguments." % dist_type.__name__)

    args = list(_generated_scalar_data_args(enc, data_count, engine))
    for name in call_names[data_count:]:
        if name not in param_names:
            raise ValueError("%s generated scorer requires undeclared parameter %s." % (dist_type.__name__, name))
        args.append(params[name])
    args.append(engine)
    return fn(*args)


def _generated_scalar_data_args(enc: Any, count: int, engine: Any) -> tuple[Any, ...]:
    raw_args = (enc,) if count == 1 else tuple(enc[:count])
    if len(raw_args) != count:
        raise ValueError("encoded payload does not contain %d generated arguments." % count)
    arrays = tuple(engine.asarray(arg) for arg in raw_args)
    shapes = [tuple(getattr(arr, "shape", ())) for arr in arrays]
    if any(not shape for shape in shapes):
        raise ValueError("generated scalar scoring requires per-row encoded arrays.")
    if any(shape[0] != shapes[0][0] for shape in shapes[1:]):
        raise ValueError("generated scalar encoded arrays have inconsistent row counts.")
    return arrays


_KNOWN_PARAMETER_CONSTRAINTS = {
    "real",
    "real_vector",
    "positive",
    "positive_vector",
    "positive_matrix",
    "unit_interval",
    "simplex",
    "simplex_vector",
    "simplex_map",
    "row_simplex_matrix",
    "row_simplex_map",
    "column_simplex_matrix",
    "integer",
    "integer_vector",
    "integer_matrix",
    "positive_integer",
    "non_negative_integer",
    "optional_integer",
    "fixed",
    "metadata",
    "log_probability_tables",
    "log_unit_interval_vector",
    "optional_log_unit_interval_vector",
}


_SHARED_STACKED_PARAMETER_CONSTRAINTS = {
    "fixed",
    "metadata",
    "integer",
    "integer_vector",
    "integer_matrix",
    "positive_integer",
    "non_negative_integer",
    "optional_integer",
}

_KNOWN_STATISTIC_KINDS = {
    "child_stat",
    "child_stats",
    "choice_child_stats",
    "count_map",
    "count_maps",
    "count_tensor",
    "count_vector",
    "histogram",
    "mapping",
    "matrix_moment",
    "metadata",
    "moment",
    "none",
    "pairwise_count_tensor",
    "raw_observations",
    "support_bound",
    "tuple",
    "vector_moment",
    "weights",
}


def _generated_stacked_requires_shared_param(spec: ParameterSpec) -> bool:
    return spec.constraint in _SHARED_STACKED_PARAMETER_CONSTRAINTS


def _declared_parameter_exists(dist_type: type[Any], name: str) -> bool:
    """Conservatively find constructor, descriptor, or initialized attributes."""
    try:
        signature = inspect.signature(dist_type.__init__)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and name in signature.parameters:
        return True
    for base in dist_type.mro():
        if name in getattr(base, "__dict__", {}):
            return True
        for member in getattr(base, "__dict__", {}).values():
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            elif isinstance(member, property):
                member = member.fget
            if not inspect.isfunction(member):
                continue
            try:
                if any(
                    instruction.opname == "STORE_ATTR" and instruction.argval == name
                    for instruction in dis.get_instructions(member)
                ):
                    return True
            except TypeError:
                continue
    return False


def _declaration_issues(
    declaration: DistributionDeclaration,
    path: str,
    ancestors: frozenset[int],
) -> tuple[str, ...]:
    if id(declaration) in ancestors:
        return ("%s creates a child declaration cycle." % path,)
    ancestors = ancestors | {id(declaration)}
    issues = []
    if not declaration.name:
        issues.append("%s has an empty name." % path)
    if not isinstance(declaration.distribution_type, type):
        issues.append("%s has a non-type distribution_type." % path)
    if not isinstance(declaration.support, str) or not declaration.support:
        issues.append("%s has an empty or non-string support." % path)
    if not isinstance(declaration.differentiable, bool):
        issues.append("%s differentiable must be bool." % path)

    param_names = []
    seen_params = set()
    for spec in declaration.parameters:
        if not isinstance(spec, ParameterSpec):
            issues.append("%s has a parameter that is not a ParameterSpec." % path)
            continue
        if not spec.name:
            issues.append("%s has an empty parameter name." % path)
            continue
        if spec.name in seen_params:
            issues.append("%s has duplicate parameter %s." % (path, spec.name))
        constraint = str(spec.constraint)
        if _is_ordered_constraint(constraint):
            anchor = _ordered_constraint_anchor(constraint)
            if anchor not in seen_params:
                issues.append("%s parameter %s references missing or later anchor %s." % (path, spec.name, anchor))
        elif constraint not in _KNOWN_PARAMETER_CONSTRAINTS:
            issues.append("%s parameter %s has unknown constraint %s." % (path, spec.name, constraint))
        seen_params.add(spec.name)
        param_names.append(spec.name)
        if isinstance(declaration.distribution_type, type) and not _declared_parameter_exists(
            declaration.distribution_type, spec.name
        ):
            issues.append(
                "%s parameter %s is not exposed by %s."
                % (path, spec.name, declaration.distribution_type.__name__)
            )

    stat_names = set()
    for spec in declaration.statistics:
        if not isinstance(spec, StatisticSpec):
            issues.append("%s has a statistic that is not a StatisticSpec." % path)
            continue
        if not spec.name:
            issues.append("%s has an empty statistic name." % path)
            continue
        if spec.name in stat_names:
            issues.append("%s has duplicate statistic %s." % (path, spec.name))
        if spec.kind not in _KNOWN_STATISTIC_KINDS:
            issues.append("%s statistic %s has unknown kind %s." % (path, spec.name, spec.kind))
        stat_names.add(spec.name)

    if declaration.child_roles and len(declaration.child_roles) != len(declaration.children):
        issues.append(
            "%s has %d child roles for %d children." % (path, len(declaration.child_roles), len(declaration.children))
        )
    if len(set(declaration.child_roles)) != len(declaration.child_roles):
        issues.append("%s has duplicate child roles." % path)
    for i, child in enumerate(declaration.children):
        role = declaration.child_roles[i] if i < len(declaration.child_roles) else str(i)
        child_path = "%s.%s" % (path, role)
        if child is None:
            issues.append("%s child declaration is missing." % child_path)
        elif not isinstance(child, DistributionDeclaration):
            issues.append("%s child is not a DistributionDeclaration." % child_path)
        else:
            issues.extend(_declaration_issues(child, path=child_path, ancestors=ancestors))

    if declaration.exponential_family is not None:
        spec = declaration.exponential_family
        if not isinstance(spec, ExponentialFamilySpec):
            issues.append("%s exponential_family is not an ExponentialFamilySpec." % path)
            return tuple(issues)
        if not callable(spec.sufficient_statistics):
            issues.append("%s exponential-family sufficient_statistics is not callable." % path)
        if not callable(spec.natural_parameters):
            issues.append("%s exponential-family natural_parameters is not callable." % path)
        if not callable(spec.log_partition):
            issues.append("%s exponential-family log_partition is not callable." % path)
        if spec.base_measure is not None and not callable(spec.base_measure):
            issues.append("%s exponential-family base_measure is not callable." % path)
        if spec.sufficient_statistics_from_params is not None and not callable(spec.sufficient_statistics_from_params):
            issues.append("%s exponential-family sufficient_statistics_from_params is not callable." % path)
        if spec.base_measure_from_params is not None and not callable(spec.base_measure_from_params):
            issues.append("%s exponential-family base_measure_from_params is not callable." % path)
        if spec.legacy_sufficient_statistics is not None and not callable(spec.legacy_sufficient_statistics):
            issues.append("%s exponential-family legacy_sufficient_statistics is not callable." % path)
        if not isinstance(spec.fixed_base, bool):
            issues.append("%s exponential-family fixed_base must be bool." % path)
        if not isinstance(spec.runtime_scoring, bool):
            issues.append("%s exponential-family runtime_scoring must be bool." % path)
        if spec.fixed_base is False and spec.base_measure_from_params is None:
            issues.append(
                "%s exponential-family fixed_base=False requires base_measure_from_params." % path
            )
    if declaration.legacy_sufficient_statistics is not None and not callable(declaration.legacy_sufficient_statistics):
        issues.append("%s legacy_sufficient_statistics is not callable." % path)
    return tuple(issues)


def _statistic_layout_issues(declaration: DistributionDeclaration, suff_stat: Any, path: str) -> tuple[str, ...]:
    issues = []
    try:
        values = declaration.statistic_values(suff_stat)
    except Exception as exc:  # noqa: BLE001
        return ("%s statistics do not match declaration: %s" % (path, exc),)

    names = tuple(values.keys())
    if names != declaration.statistic_names:
        issues.append(
            "%s statistic names %r do not match declaration names %r." % (path, names, declaration.statistic_names)
        )

    for spec in declaration.statistics:
        if spec.name not in values:
            continue
        issues.extend(_statistic_value_issues(declaration, spec, values[spec.name], "%s.%s" % (path, spec.name)))
    return tuple(issues)


def _statistic_value_issues(
    declaration: DistributionDeclaration, spec: StatisticSpec, value: Any, path: str
) -> tuple[str, ...]:
    child_indices = _child_indices_for_stat(declaration, spec)
    if not child_indices:
        if spec.kind == "count_map" and not isinstance(value, Mapping):
            return ("%s expected a mapping." % path,)
        if spec.kind == "count_maps" and not (
            isinstance(value, Mapping)
            or (
                isinstance(value, (tuple, list))
                and all(isinstance(item, Mapping) for item in value)
            )
        ):
            return ("%s expected a mapping or sequence of mappings." % path,)
        if spec.kind in ("count_vector", "vector_moment"):
            try:
                shape = np.asarray(value).shape
            except (TypeError, ValueError):
                return ("%s expected a one-dimensional statistic." % path,)
            if len(shape) != 1:
                return ("%s expected a one-dimensional statistic." % path,)
        if spec.kind == "matrix_moment":
            try:
                shape = np.asarray(value).shape
            except (TypeError, ValueError):
                return ("%s expected a two-dimensional statistic." % path,)
            if len(shape) != 2:
                return ("%s expected a two-dimensional statistic." % path,)
        if spec.kind in ("count_tensor", "pairwise_count_tensor"):
            try:
                shape = np.asarray(value).shape
            except (TypeError, ValueError):
                return ("%s expected a tensor statistic." % path,)
            if len(shape) < 2:
                return ("%s expected a tensor statistic with rank at least two." % path,)
        return ()

    if spec.kind == "child_stat":
        if len(child_indices) != 1:
            return ("%s maps to %d child declarations; expected one." % (path, len(child_indices)),)
        return _child_statistic_issues(declaration, child_indices[0], value, path)

    if spec.kind == "choice_child_stats":
        if not isinstance(value, (tuple, list)):
            return ("%s expected a sequence of per-choice child statistics." % path,)
        issues = []
        if len(value) != len(child_indices):
            issues.append("%s expected %d child statistics, got %d." % (path, len(child_indices), len(value)))
        for i, child_idx in enumerate(child_indices[: len(value)]):
            child_value = value[i]
            if isinstance(child_value, (tuple, list)) and len(child_value) == 2:
                child_value = child_value[1]
            issues.extend(_child_statistic_issues(declaration, child_idx, child_value, "%s[%d]" % (path, i)))
        return tuple(issues)

    if spec.kind == "mapping" and isinstance(value, dict):
        return _mapping_statistic_issues(declaration, spec, value, path)

    if spec.kind in ("tuple", "mapping", "choice_child_stats"):
        if not isinstance(value, (tuple, list)):
            return ("%s expected a sequence of child statistics." % path,)
        issues = []
        if len(value) != len(child_indices):
            issues.append("%s expected %d child statistics, got %d." % (path, len(child_indices), len(value)))
        for i, child_idx in enumerate(child_indices[: len(value)]):
            issues.extend(_child_statistic_issues(declaration, child_idx, value[i], "%s[%d]" % (path, i)))
        return tuple(issues)

    return ()


def _mapping_statistic_issues(
    declaration: DistributionDeclaration, spec: StatisticSpec, value: dict[Any, Any], path: str
) -> tuple[str, ...]:
    issues = []
    role_to_idx = {role: i for i, role in enumerate(declaration.child_roles)}
    if spec.name == "conditions":
        for key, child_value in value.items():
            role = "condition_%r" % key
            child_idx = role_to_idx.get(role)
            if child_idx is None:
                issues.append("%s has no child declaration for condition key %r." % (path, key))
            else:
                issues.extend(_child_statistic_issues(declaration, child_idx, child_value, "%s[%r]" % (path, key)))
        return tuple(issues)

    child_indices = _child_indices_for_stat(declaration, spec)
    if value and len(value) != len(child_indices):
        issues.append("%s expected %d mapped child statistics, got %d." % (path, len(child_indices), len(value)))
    for role, child_idx in zip(declaration.child_roles, child_indices):
        if role in value:
            issues.extend(_child_statistic_issues(declaration, child_idx, value[role], "%s[%r]" % (path, role)))
    return tuple(issues)


def _child_statistic_issues(
    declaration: DistributionDeclaration, child_idx: int, value: Any, path: str
) -> tuple[str, ...]:
    child = declaration.children[child_idx]
    role = declaration.child_roles[child_idx] if child_idx < len(declaration.child_roles) else str(child_idx)
    return _statistic_layout_issues(child, value, "%s->%s" % (path, role))


def _child_indices_for_stat(declaration: DistributionDeclaration, spec: StatisticSpec) -> tuple[int, ...]:
    if not declaration.children:
        return ()
    roles = tuple(declaration.child_roles)
    normalized = _normalized_statistic_role(spec.name)
    exact = tuple(i for i, role in enumerate(roles) if role == spec.name or role == normalized)
    if exact:
        return exact

    component_digit = _trailing_digit(spec.name)
    if component_digit is not None:
        stem = _normalized_statistic_role(spec.name[:-1])
        prefix = "x%d_%s" % (component_digit, stem)
        digit_matches = tuple(
            i for i, role in enumerate(roles) if role.startswith(prefix) or role.endswith("_%s" % stem)
        )
        if digit_matches:
            return digit_matches

    if spec.kind in ("tuple", "choice_child_stats") or spec.name == "fields":
        stem = normalized
        matches = tuple(
            i
            for i, role in enumerate(roles)
            if role == stem or role.startswith("%s_" % stem) or role.endswith("_%s" % stem) or ("_%s_" % stem) in role
        )
        if matches:
            return matches
        return tuple(range(len(declaration.children)))

    return ()


def _normalized_statistic_role(name: str) -> str:
    aliases = {
        "children": "choice",
        "components": "component",
        "emissions": "emission",
        "elements": "element",
        "fields": "field",
        "lengths": "length",
        "topics": "topic",
        "values": "value",
    }
    if name in aliases:
        return aliases[name]
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("s") and len(name) > 1:
        return name[:-1]
    return name


def _trailing_digit(name: str) -> int | None:
    if name and name[-1].isdigit():
        return int(name[-1])
    return None


def _is_ordered_constraint(constraint: str) -> bool:
    return constraint.startswith("greater_than:") or constraint.startswith("less_than:")


def _ordered_constraint_anchor(constraint: str) -> str:
    return constraint.split(":", 1)[1] if ":" in constraint else ""


def _generated_backend_hook_supported(dist_type: type[Any], declaration: DistributionDeclaration) -> bool:
    fn = getattr(dist_type, "backend_log_density_from_params", None)
    if not callable(fn):
        return False
    sig_names = tuple(inspect.signature(fn).parameters.keys())
    if not sig_names or sig_names[-1] != "engine":
        return False
    param_names = set(declaration.parameter_names)
    data_count = 0
    for name in sig_names[:-1]:
        if name in param_names:
            break
        data_count += 1
    if data_count <= 0:
        return False
    return all(name in param_names for name in sig_names[data_count:-1])


def _validated_generated_score(
    value: Any,
    name: str,
    component_count: int | None = None,
) -> Any:
    shape = tuple(getattr(value, "shape", ()))
    expected_rank = 2 if component_count is not None else 1
    if len(shape) != expected_rank:
        raise ValueError("%s must have rank %d, got shape %r." % (name, expected_rank, shape))
    if component_count is not None and shape[1] != component_count:
        raise ValueError(
            "%s must have %d component columns, got shape %r."
            % (name, component_count, shape)
        )
    if isinstance(value, np.ndarray):
        try:
            numeric = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be numeric." % name) from exc
        if np.any(np.isnan(numeric)) or np.any(np.isposinf(numeric)):
            raise ValueError("%s cannot contain NaN or positive infinity." % name)
    return value


def _generated_data_args(enc: Any, count: int, engine: Any) -> tuple[Any, ...]:
    raw_args = (enc,) if count == 1 else tuple(enc[:count])
    if len(raw_args) != count:
        raise ValueError("encoded payload does not contain %d generated arguments." % count)
    rv = []
    row_count = None
    for arg in raw_args:
        arr = engine.asarray(arg)
        shape = tuple(getattr(arr, "shape", ()))
        if len(shape) == 0:
            raise ValueError("generated stacked scoring requires per-row encoded arrays.")
        if row_count is None:
            row_count = shape[0]
        elif shape[0] != row_count:
            raise ValueError("generated stacked encoded arrays have inconsistent row counts.")
        rv.append(arr[:, None] if len(shape) == 1 else arr[:, None, ...])
    return tuple(rv)


def _generated_param_arg(value: Any, engine: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bool, int, float, np.number)):
        return value
    arr = engine.asarray(value)
    shape = tuple(getattr(arr, "shape", ()))
    if len(shape) == 0:
        return arr
    if len(shape) == 1:
        return arr[None, :]
    if len(shape) == 2:
        return arr[None, :, :]
    if len(shape) == 3:
        return arr[None, :, :, :]
    raise ValueError("generated stacked scoring currently supports scalar/vector/matrix per-component parameters only.")


def _astype(x: Any, dtype: Any) -> Any:
    """Cast an engine array to ``dtype`` (float64 accumulator) across numpy/torch/jax; no-op if None."""
    if dtype is None:
        return x
    if hasattr(x, "astype"):  # numpy / jax
        return x.astype(dtype)
    return x.to(dtype)  # torch


def _validate_generated_statistic_rows(
    row_stats: Sequence[Any],
    specs: Sequence[StatisticSpec],
    weights: Any,
    component_count: int | None = None,
) -> None:
    weight_shape = tuple(getattr(weights, "shape", ()))
    expected_weight_rank = 2 if component_count is not None else 1
    if len(weight_shape) != expected_weight_rank:
        raise ValueError(
            "generated statistic weights must have rank %d, got shape %r."
            % (expected_weight_rank, weight_shape)
        )
    if component_count is not None and weight_shape[1] != component_count:
        raise ValueError(
            "generated statistic weights must have %d component columns."
            % component_count
        )
    if isinstance(weights, np.ndarray):
        numeric_weights = np.asarray(weights, dtype=np.float64)
        if np.any(~np.isfinite(numeric_weights)) or np.any(numeric_weights < 0.0):
            raise ValueError("generated statistic weights must be finite and non-negative.")

    row_count = weight_shape[0]
    for index, (stat, spec) in enumerate(zip(row_stats, specs)):
        shape = tuple(getattr(stat, "shape", ()))
        if not shape or shape[0] != row_count:
            raise ValueError(
                "generated %s statistic %d must have a matching row axis."
                % (spec.kind, index)
            )
        if spec.kind == "vector_moment" and len(shape) < 2:
            raise ValueError("generated vector_moment statistics require a feature axis.")
        if spec.kind == "matrix_moment" and len(shape) < 3:
            raise ValueError("generated matrix_moment statistics require two feature axes.")
        if isinstance(stat, np.ndarray):
            try:
                numeric_stat = np.asarray(stat, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("generated statistic %d must be numeric." % index) from exc
            if np.any(np.isnan(numeric_stat)) or np.any(np.isposinf(numeric_stat)):
                raise ValueError(
                    "generated statistic %d cannot contain NaN or positive infinity." % index
                )


def _weighted_component_sum(stat: Any, spec: StatisticSpec, weights: Any, engine: Any) -> Any:
    # Accumulate the over-observations reduction in float64 (no-op for full-precision engines) so a
    # reduced-precision component fit does not drift on large N.
    acc = getattr(engine, "accumulator_dtype", None)
    arr = engine.asarray(stat)
    shape = tuple(getattr(arr, "shape", ()))
    weights_shape = tuple(getattr(weights, "shape", ()))
    if len(shape) == 1:
        arr = arr[:, None]
    elif spec.kind in ("vector_moment", "matrix_moment"):
        if not weights_shape or shape[0] != weights_shape[0]:
            raise ValueError("generated %s statistics must have a row axis." % spec.kind)
        # The weighted component sum sum_n w[n,k] * arr[n, ...] -> (K, *trailing) computed as a MATMUL
        # over the flattened trailing axes (w.T @ arr_flat), NOT a (K,)-fold broadcast. The broadcast
        # form materialized an (N, K, *trailing) intermediate that OOMs the GPU for matrix moments
        # (N*K*dim*dim -- e.g. 21 GB at N=2e4, K=8, dim=128); the matmul only ever holds (K, prod). It
        # is bit-identical, and the float64 accumulation is preserved by casting to the accumulator dtype.
        n_rows, k = shape[0], weights_shape[1]
        w = _astype(engine.asarray(weights), acc)
        a = _astype(engine.asarray(arr), acc)
        out = engine.matmul(w.T, a.reshape(n_rows, -1))
        return out.reshape((k, *shape[1:]))
    elif len(shape) != 2 or shape != weights_shape:
        raise ValueError(
            "generated sufficient statistics must be row, row-component, or declared vector/matrix arrays."
        )
    return engine.sum(weights * arr, axis=0, dtype=acc)


def _weighted_row_sum(stat: Any, spec: StatisticSpec, weights: Any, engine: Any) -> Any:
    arr = engine.asarray(stat)
    shape = tuple(getattr(arr, "shape", ()))
    weights_shape = tuple(getattr(weights, "shape", ()))
    if not shape or not weights_shape or shape[0] != weights_shape[0]:
        raise ValueError("generated %s statistics must have a row axis." % spec.kind)
    # Accumulate the over-observations reduction in the engine's high-precision dtype (float64) so a
    # reduced-precision (float32) fit does not drift on large N. No-op for full-precision engines.
    acc = getattr(engine, "accumulator_dtype", None)
    if len(shape) == 1:
        return engine.sum(weights * arr, axis=0, dtype=acc)
    extra_axes = (None,) * (len(shape) - 1)
    return engine.sum(weights[(slice(None),) + extra_axes] * arr, axis=0, dtype=acc)


def _weighted_histogram(stat: Any, weights: Any, engine: Any) -> dict[int, float]:
    """Fold per-row integer counts into a weighted ``{value: weight}`` histogram.

    This mirrors :meth:`NegativeBinomialAccumulator.seq_update` so the generated
    (torch/stacked) accumulate path produces the exact dict the numpy host
    accumulator returns for histogram-kind statistics.
    """
    vals = np.asarray(engine.to_numpy(stat))
    wts = np.asarray(engine.to_numpy(weights), dtype=np.float64)
    hist: dict[int, float] = {}
    if vals.size == 0:
        return hist
    ints = np.rint(vals.reshape(-1)).astype(np.int64)
    uniq, inv = np.unique(ints, return_inverse=True)
    wsum = np.zeros(uniq.shape[0], dtype=np.float64)
    np.add.at(wsum, inv, wts.reshape(-1))
    for k, w in zip(uniq.tolist(), wsum.tolist()):
        hist[int(k)] = w
    return hist


def _host_legacy_value(value: Any, engine: Any) -> Any:
    try:
        arr = np.asarray(engine.to_numpy(value))
    except Exception:  # noqa: BLE001
        return value
    if arr.ndim == 0:
        return float(arr)
    return arr


def _legacy_sufficient_statistics_fn(declaration: DistributionDeclaration) -> Callable | None:
    if callable(declaration.legacy_sufficient_statistics):
        return declaration.legacy_sufficient_statistics
    if declaration.exponential_family is not None and callable(
        declaration.exponential_family.legacy_sufficient_statistics
    ):
        return declaration.exponential_family.legacy_sufficient_statistics
    return None


def _all_same(values: Sequence[Any]) -> bool:
    if not values:
        return True
    first = values[0]
    for value in values[1:]:
        try:
            equal = np.array_equal(first, value)
        except Exception:  # noqa: BLE001
            equal = first == value
        if not bool(equal):
            return False
    return True
