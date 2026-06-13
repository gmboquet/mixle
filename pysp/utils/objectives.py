"""Generic objective optimization and variational projection utilities.

This module is intentionally model-facing rather than engine-facing: callers
provide objective functions, distributions provide their own scoring math, and
compute engines provide only tensor arithmetic/autograd.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.random import RandomState

from pysp.stats import seq_encode
from pysp.stats.backend import backend_seq_log_density
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution
from pysp.utils.estimation import (
    GradientFitError,
    _gradient_build_state,
    _gradient_raw_state,
    _gradient_shadow_state,
    _torch_for_gradient_fit,
)


ObjectiveCallable = Callable[[Any, Any, Any], Any]
ParameterObjectiveCallable = Callable[[Mapping[str, Any], Any, Any], Any]


@dataclass
class ObjectiveFitResult:
    """Optimization result for arbitrary differentiable objectives.

    By default objective helpers restore the best trainable state seen during
    optimization.  In that mode ``value`` and ``model`` refer to
    ``best_value`` / ``best_iteration`` while ``history`` still records every
    attempted iterate and ``final_delta`` describes the last attempted step.
    """

    model: Any
    value: float
    iterations: int
    history: Tuple[float, ...] = ()
    converged: bool = False
    initial_value: Optional[float] = None
    final_delta: Optional[float] = None
    maximize: bool = True
    best_value: Optional[float] = None
    best_iteration: Optional[int] = None
    final_gradient_norm: Optional[float] = None

    def as_tuple(self) -> Tuple[Any, float]:
        """Return the historical ``(model, value)`` shape used by fit helpers."""
        return self.model, self.value

    @property
    def objective_change(self) -> Optional[float]:
        """Return the signed objective change from the start of optimization."""
        if self.initial_value is None:
            return None
        return self.value - self.initial_value

    @property
    def improvement(self) -> Optional[float]:
        """Return positive improvement in the requested optimization direction."""
        change = self.objective_change
        if change is None:
            return None
        return change if self.maximize else -change

    @property
    def best_improvement(self) -> Optional[float]:
        """Return best positive improvement seen during optimization."""
        if self.initial_value is None or self.best_value is None:
            return None
        change = self.best_value - self.initial_value
        return change if self.maximize else -change


@dataclass(frozen=True)
class ObjectiveParameter:
    """Named trainable parameter for arbitrary differentiable objectives.

    Supported constraints are ``real``, ``positive`` / ``positive_vector`` /
    ``positive_matrix``, ``unit_interval``, ``simplex`` /
    ``simplex_vector``, ``row_simplex_matrix``,
    ``column_simplex_matrix``, ``greater_than:<name>``, and
    ``less_than:<name>``.  Coupled bound constraints refer to an
    earlier parameter in the same set.  The optimizer
    stores unconstrained raw tensors and presents constrained values to the
    objective callable.
    """

    name: str
    value: Any
    constraint: str = 'real'


class ObjectiveParameterSet(object):
    """Engine-backed named parameters for user-supplied objectives."""

    def __init__(self, parameters: Any, engine: Optional[Any] = None,
                 precision: Optional[Any] = None, torch: Optional[Any] = None) -> None:
        if torch is None:
            torch, engine = _torch_for_gradient_fit(engine, precision=precision)
        self.torch = torch
        self.engine = engine
        self.specs = tuple(_normalize_objective_parameters(parameters))
        if not self.specs:
            raise ValueError('ObjectiveParameterSet requires at least one parameter.')
        self.raw = {}
        initial_values = {}
        for spec in self.specs:
            constraint = str(spec.constraint)
            if _objective_is_coupled_bound_constraint(constraint):
                anchor = _objective_bound_anchor(constraint)
                if anchor not in initial_values:
                    raise ValueError('Objective parameter %s requires earlier anchor parameter %s.' %
                                     (spec.name, anchor))
                delta = _objective_bound_delta(spec.value, initial_values[anchor], constraint)
                self.raw[spec.name] = _objective_raw_tensor(delta, 'positive', self.engine, self.torch)
            else:
                self.raw[spec.name] = _objective_raw_tensor(spec.value, constraint, self.engine, self.torch)
            initial_values[spec.name] = spec.value

    def trainable_tensors(self) -> Sequence[Any]:
        """Return raw tensors passed to the optimizer."""
        return [self.raw[spec.name] for spec in self.specs]

    def values(self) -> Mapping[str, Any]:
        """Return constrained tensors keyed by parameter name."""
        values = {}
        for spec in self.specs:
            values[spec.name] = _objective_constrained_value(
                self.raw[spec.name], spec.constraint, self.torch, values)
        return values

    def detached_values(self) -> Mapping[str, Any]:
        """Return plain Python/NumPy constrained values keyed by parameter name."""
        return {key: _detach_objective_value(value) for key, value in self.values().items()}


class ExpectedLogDensity(object):
    """Objective ``sum_i w_i log q_model(x_i)`` for encoded observations."""

    def __init__(self, weights: Optional[Sequence[float]] = None, normalize: bool = False) -> None:
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.normalize = bool(normalize)

    def __call__(self, model: SequenceEncodableProbabilityDistribution, enc: Any, engine: Any) -> Any:
        scores = backend_seq_log_density(model, enc, engine)
        if self.weights is None:
            obj = engine.sum(scores)
            if self.normalize:
                obj = obj / engine.asarray(float(scores.shape[0]))
            return obj
        weights = engine.asarray(self.weights)
        obj = engine.sum(scores * weights)
        if self.normalize:
            obj = obj / engine.sum(weights)
        return obj


class ObjectiveSum(object):
    """Add several model objectives into one scalar objective."""

    def __init__(self, *objectives: ObjectiveCallable) -> None:
        if not objectives:
            raise ValueError('ObjectiveSum requires at least one objective.')
        self.objectives = objectives

    def __call__(self, model: Any, enc: Any, engine: Any) -> Any:
        rv = self.objectives[0](model, enc, engine)
        for objective in self.objectives[1:]:
            rv = rv + objective(model, enc, engine)
        return rv


class UnnormalizedLogLikelihood(object):
    """Objective for models specified by unnormalized log likelihoods.

    The objective is

        sum_i w_i * log f_theta(x_i) - sum_i w_i * log Z(theta)

    where ``log f_theta`` is supplied by the caller.  ``log Z(theta)`` can be
    supplied exactly with ``log_partition`` or estimated from reference samples
    using self-normalized importance form
    ``logmeanexp(log f_theta(y_j) - log q(y_j))``.
    """

    def __init__(self,
                 log_unnormalized: ObjectiveCallable,
                 log_partition: Optional[Callable[[Any, Any], Any]] = None,
                 partition_enc: Optional[Any] = None,
                 reference_log_density: Optional[Callable[[Any, Any], Any]] = None,
                 weights: Optional[Sequence[float]] = None,
                 normalize: bool = False) -> None:
        if log_partition is None and partition_enc is None:
            raise ValueError('UnnormalizedLogLikelihood requires log_partition or partition_enc.')
        self.log_unnormalized = log_unnormalized
        self.log_partition = log_partition
        self.partition_enc = partition_enc.payload if hasattr(partition_enc, 'payload') else partition_enc
        self.reference_log_density = reference_log_density
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.normalize = bool(normalize)

    def __call__(self, model: Any, enc: Any, engine: Any) -> Any:
        raw = self.log_unnormalized(model, enc, engine)
        if self.weights is None:
            data_term = engine.sum(raw)
            nobs = engine.asarray(float(raw.shape[0]))
        else:
            weights = engine.asarray(self.weights)
            data_term = engine.sum(raw * weights)
            nobs = engine.sum(weights)
        log_z = self._log_partition(model, engine)
        obj = data_term - nobs * log_z
        if self.normalize:
            obj = obj / nobs
        return obj

    def _log_partition(self, model: Any, engine: Any) -> Any:
        if self.log_partition is not None:
            return self.log_partition(model, engine)
        scores = self.log_unnormalized(model, self.partition_enc, engine)
        if self.reference_log_density is not None:
            scores = scores - self.reference_log_density(self.partition_enc, engine)
        return engine.logsumexp(scores, axis=0) - engine.log(engine.asarray(float(scores.shape[0])))


class CallableObjective(object):
    """Small adapter naming arbitrary objective callables."""

    def __init__(self, fn: ObjectiveCallable, name: str = 'callable_objective') -> None:
        self.fn = fn
        self.name = name

    def __call__(self, model: Any, enc: Any, engine: Any) -> Any:
        return self.fn(model, enc, engine)


def fit_objective(enc: Any,
                  model: SequenceEncodableProbabilityDistribution,
                  objective: ObjectiveCallable,
                  engine: Optional[Any] = None,
                  max_its: int = 500,
                  lr: float = 0.05,
                  optimizer: str = 'adam',
                  tol: float = 1.0e-7,
                  maximize: bool = True,
                  out: Optional[Any] = None,
                  print_iter: int = 100,
                  return_result: bool = False,
                  precision: Optional[Any] = None,
                  restore_best: bool = True) -> Any:
    """Optimize a user-supplied differentiable objective over a distribution tree.

    The objective is called as ``objective(shadow_model, enc, engine)`` and must
    return a scalar engine tensor. Distribution parameters are obtained from the
    same declaration-driven raw state used by ``fit_mle`` / ``fit_map``.
    When ``restore_best`` is true, the returned model/value correspond to the
    best objective value seen, not necessarily the last attempted optimizer
    step.
    """
    torch, engine = _torch_for_gradient_fit(engine, precision=precision)
    if hasattr(enc, 'payload'):
        enc = enc.payload

    leaves = []
    state = _gradient_raw_state(model, engine, torch, leaves)
    if not leaves:
        raise GradientFitError('%s has no differentiable parameters.' % type(model).__name__)

    opt = _make_optimizer(torch, optimizer, leaves, lr)
    sign = 1.0 if maximize else -1.0
    iterations = max(1, int(max_its))
    converged = False

    def objective_value():
        shadow = _gradient_shadow_state(state, torch)
        return objective(shadow, enc, engine)

    history = [_objective_scalar(objective_value())]
    best_value = history[0]
    best_iteration = 0
    best_state = _clone_parameter_state(leaves)
    for i in range(iterations):
        if optimizer == 'lbfgs':
            def closure():
                opt.zero_grad()
                loss = -sign * objective_value()
                loss.backward()
                return loss
            loss = opt.step(closure)
        else:
            opt.zero_grad()
            loss = -sign * objective_value()
            loss.backward()
            opt.step()

        cur = _objective_scalar(objective_value())
        history.append(cur)
        if out is not None and (i + 1) % max(1, int(print_iter)) == 0:
            out.write('objective iteration %d: value=%e\n' % (i + 1, cur))
        if _objective_is_better(cur, best_value, maximize=maximize):
            best_value = cur
            best_iteration = len(history) - 1
            best_state = _clone_parameter_state(leaves)
        if len(history) > 2 and abs(cur - history[-2]) < tol * max(1.0, abs(cur)):
            iterations = i + 1
            converged = True
            break

    final_delta = history[-1] - history[-2] if len(history) > 1 else None
    if restore_best:
        _restore_parameter_state(torch, leaves, best_state)
        final_value = best_value
    else:
        final_value = history[-1]
        best_value, best_iteration = _objective_best_entry(history, maximize=maximize)
    final_gradient_norm = _objective_gradient_norm(torch, leaves, objective_value)
    result = ObjectiveFitResult(
        _gradient_build_state(state, torch),
        final_value,
        iterations,
        history=tuple(history),
        converged=converged,
        initial_value=history[0],
        final_delta=final_delta,
        maximize=bool(maximize),
        best_value=best_value,
        best_iteration=best_iteration,
        final_gradient_norm=final_gradient_norm,
    )
    return result if return_result else result.as_tuple()


def variational_projection(source: SequenceEncodableProbabilityDistribution,
                           target: SequenceEncodableProbabilityDistribution,
                           data: Optional[Sequence[Any]] = None,
                           enc: Optional[Any] = None,
                           sample_size: int = 1000,
                           seed: Optional[int] = None,
                           engine: Optional[Any] = None,
                           max_its: int = 500,
                           lr: float = 0.05,
                           optimizer: str = 'adam',
                           tol: float = 1.0e-7,
                           out: Optional[Any] = None,
                           print_iter: int = 100,
                           return_result: bool = False,
                           precision: Optional[Any] = None,
                           restore_best: bool = True) -> Any:
    """Project ``source`` onto the family represented by ``target``.

    This minimizes a Monte-Carlo estimate of the forward KL
    ``KL(source || target)`` by maximizing ``E_source[log target(X)]``.
    Pass ``data``/``enc`` to reuse common random numbers or a curated design set;
    otherwise samples are drawn from ``source``.
    """
    if enc is None:
        if data is None:
            if sample_size <= 0:
                raise ValueError('sample_size must be positive when data/enc is not supplied.')
            data = source.sampler(seed=seed).sample(size=int(sample_size))
        enc = target.dist_to_encoder().seq_encode(data)
    objective = ExpectedLogDensity(normalize=True)
    return fit_objective(enc, target, objective, engine=engine, max_its=max_its, lr=lr,
                         optimizer=optimizer, tol=tol, maximize=True, out=out,
                         print_iter=print_iter, precision=precision,
                         return_result=return_result, restore_best=restore_best)


def optimize_torch_objective(parameters: Iterable[Any],
                             objective: Callable[[], Any],
                             engine: Optional[Any] = None,
                             max_its: int = 500,
                             lr: float = 0.05,
                             optimizer: str = 'adam',
                             tol: float = 1.0e-7,
                             maximize: bool = True,
                             out: Optional[Any] = None,
                             print_iter: int = 100,
                             precision: Optional[Any] = None,
                             return_result: bool = False,
                             restore_best: bool = True) -> Any:
    """Optimize an arbitrary Torch objective over supplied tensor parameters.

    This is the escape hatch for models whose likelihood is not an iid
    distribution score, such as Gaussian-process marginal likelihoods or
    supervised neural-network losses.  When ``restore_best`` is true, supplied
    tensors are copied back to their best seen values before returning.
    """
    torch, _ = _torch_for_gradient_fit(engine, precision=precision)
    params = [p for p in parameters if getattr(p, 'requires_grad', False)]
    if not params:
        raise ValueError('optimize_torch_objective requires at least one trainable parameter.')
    opt = _make_optimizer(torch, optimizer, params, lr)
    sign = 1.0 if maximize else -1.0
    iterations = max(1, int(max_its))
    converged = False
    history = [_objective_scalar(objective())]
    best_value = history[0]
    best_iteration = 0
    best_state = _clone_parameter_state(params)

    for i in range(iterations):
        if optimizer == 'lbfgs':
            def closure():
                opt.zero_grad()
                loss = -sign * objective()
                loss.backward()
                return loss
            loss = opt.step(closure)
        else:
            opt.zero_grad()
            loss = -sign * objective()
            loss.backward()
            opt.step()

        cur = _objective_scalar(objective())
        history.append(cur)
        if out is not None and (i + 1) % max(1, int(print_iter)) == 0:
            out.write('torch objective iteration %d: value=%e\n' % (i + 1, cur))
        if _objective_is_better(cur, best_value, maximize=maximize):
            best_value = cur
            best_iteration = len(history) - 1
            best_state = _clone_parameter_state(params)
        if len(history) > 2 and abs(cur - history[-2]) < tol * max(1.0, abs(cur)):
            iterations = i + 1
            converged = True
            break

    if restore_best:
        _restore_parameter_state(torch, params, best_state)
        final_value = best_value
    else:
        final_value = history[-1]
        best_value, best_iteration = _objective_best_entry(history, maximize=maximize)
    if return_result:
        final_delta = history[-1] - history[-2] if len(history) > 1 else None
        result = ObjectiveFitResult(
            tuple(_detach_objective_value(param) for param in params),
            final_value,
            iterations,
            history=tuple(history),
            converged=converged,
            initial_value=history[0],
            final_delta=final_delta,
            maximize=bool(maximize),
            best_value=best_value,
            best_iteration=best_iteration,
            final_gradient_norm=_objective_gradient_norm(torch, params, objective),
        )
        return result
    return final_value, iterations


def fit_parameter_objective(parameters: Any,
                            objective: ParameterObjectiveCallable,
                            enc: Optional[Any] = None,
                            engine: Optional[Any] = None,
                            max_its: int = 500,
                            lr: float = 0.05,
                            optimizer: str = 'adam',
                            tol: float = 1.0e-7,
                            maximize: bool = True,
                            out: Optional[Any] = None,
                            print_iter: int = 100,
                            precision: Optional[Any] = None,
                            return_result: bool = False,
                            restore_best: bool = True) -> Any:
    """Optimize an arbitrary objective over named constrained parameters.

    ``parameters`` may be a mapping of ``name -> initial_value`` for real
    parameters, a sequence of ``ObjectiveParameter`` objects, or a sequence of
    ``(name, initial_value, constraint)`` tuples.  ``objective`` is called as
    ``objective(params, enc, engine)``, where ``params`` is a mapping of
    constrained engine tensors.  When ``restore_best`` is true, returned
    detached values are the best seen named parameters.
    """
    if isinstance(parameters, ObjectiveParameterSet):
        param_set = parameters
        torch = param_set.torch
        engine = param_set.engine
        if precision is not None:
            raise ValueError('precision cannot be changed for an existing ObjectiveParameterSet.')
    else:
        torch, engine = _torch_for_gradient_fit(engine, precision=precision)
        param_set = ObjectiveParameterSet(parameters, engine=engine, torch=torch)
    params = list(param_set.trainable_tensors())
    if not params:
        raise ValueError('fit_parameter_objective requires at least one trainable parameter.')
    if hasattr(enc, 'payload'):
        enc = enc.payload

    opt = _make_optimizer(torch, optimizer, params, lr)
    sign = 1.0 if maximize else -1.0
    iterations = max(1, int(max_its))
    converged = False

    def objective_value():
        return objective(param_set.values(), enc, engine)

    history = [_objective_scalar(objective_value())]
    best_value = history[0]
    best_iteration = 0
    best_state = _clone_parameter_state(params)
    for i in range(iterations):
        if optimizer == 'lbfgs':
            def closure():
                opt.zero_grad()
                loss = -sign * objective_value()
                loss.backward()
                return loss
            loss = opt.step(closure)
        else:
            opt.zero_grad()
            loss = -sign * objective_value()
            loss.backward()
            opt.step()

        cur = _objective_scalar(objective_value())
        history.append(cur)
        if out is not None and (i + 1) % max(1, int(print_iter)) == 0:
            out.write('parameter objective iteration %d: value=%e\n' % (i + 1, cur))
        if _objective_is_better(cur, best_value, maximize=maximize):
            best_value = cur
            best_iteration = len(history) - 1
            best_state = _clone_parameter_state(params)
        if len(history) > 2 and abs(cur - history[-2]) < tol * max(1.0, abs(cur)):
            iterations = i + 1
            converged = True
            break

    final_delta = history[-1] - history[-2] if len(history) > 1 else None
    if restore_best:
        _restore_parameter_state(torch, params, best_state)
        final_value = best_value
    else:
        final_value = history[-1]
        best_value, best_iteration = _objective_best_entry(history, maximize=maximize)
    final_gradient_norm = _objective_gradient_norm(torch, params, objective_value)
    result = ObjectiveFitResult(
        param_set.detached_values(),
        final_value,
        iterations,
        history=tuple(history),
        converged=converged,
        initial_value=history[0],
        final_delta=final_delta,
        maximize=bool(maximize),
        best_value=best_value,
        best_iteration=best_iteration,
        final_gradient_norm=final_gradient_norm,
    )
    return result if return_result else result.as_tuple()


def _make_optimizer(torch: Any, optimizer: str, parameters: Sequence[Any], lr: float) -> Any:
    opt_classes = {'adam': torch.optim.Adam, 'lbfgs': torch.optim.LBFGS}
    if optimizer not in opt_classes:
        raise ValueError('Unknown optimizer %s. Expected one of %s.' %
                         (optimizer, ', '.join(sorted(opt_classes))))
    return opt_classes[optimizer](parameters, lr=lr)


def _objective_scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _objective_best_entry(history: Sequence[float], maximize: bool = True) -> Tuple[float, int]:
    values = np.asarray(history, dtype=np.float64)
    idx = int(np.nanargmax(values) if maximize else np.nanargmin(values))
    return float(values[idx]), idx


def _objective_is_better(value: float, best_value: float, maximize: bool = True) -> bool:
    return value > best_value if maximize else value < best_value


def _clone_parameter_state(parameters: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(param.detach().clone() for param in parameters)


def _restore_parameter_state(torch: Any, parameters: Sequence[Any], values: Sequence[Any]) -> None:
    with torch.no_grad():
        for param, value in zip(parameters, values):
            param.copy_(value)


def _objective_gradient_norm(torch: Any,
                             parameters: Sequence[Any],
                             objective: Callable[[], Any]) -> float:
    for param in parameters:
        if getattr(param, 'grad', None) is not None:
            param.grad = None
    value = objective()
    value.backward()
    total = None
    for param in parameters:
        grad = getattr(param, 'grad', None)
        if grad is None:
            continue
        term = torch.sum(grad.detach() * grad.detach())
        total = term if total is None else total + term
    for param in parameters:
        if getattr(param, 'grad', None) is not None:
            param.grad = None
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu().item())


def _normalize_objective_parameters(parameters: Any) -> Sequence[ObjectiveParameter]:
    if isinstance(parameters, ObjectiveParameterSet):
        return parameters.specs
    if isinstance(parameters, Mapping):
        return [
            value if isinstance(value, ObjectiveParameter) else ObjectiveParameter(str(name), value)
            for name, value in parameters.items()
        ]
    rv = []
    for item in parameters:
        if isinstance(item, ObjectiveParameter):
            rv.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            rv.append(ObjectiveParameter(str(item[0]), item[1]))
        elif isinstance(item, tuple) and len(item) == 3:
            rv.append(ObjectiveParameter(str(item[0]), item[1], str(item[2])))
        else:
            raise TypeError('Objective parameters must be ObjectiveParameter objects, '
                            '(name, value), or (name, value, constraint) tuples.')
    return rv


def _objective_raw_tensor(value: Any, constraint: str, engine: Any, torch: Any) -> Any:
    tensor = engine.asarray(value, dtype=getattr(engine, 'dtype', None)).clone().detach()
    eps = 1.0e-8
    if constraint in ('positive', 'positive_vector', 'positive_matrix'):
        tensor = torch.log(torch.clamp(tensor, min=eps))
    elif constraint == 'unit_interval':
        tensor = torch.logit(torch.clamp(tensor, min=eps, max=1.0 - eps))
    elif constraint in ('simplex', 'simplex_vector'):
        tensor = torch.clamp(tensor, min=eps)
        tensor = tensor / torch.sum(tensor)
        tensor = torch.log(tensor)
    elif constraint == 'row_simplex_matrix':
        tensor = torch.clamp(tensor, min=eps)
        tensor = tensor / torch.sum(tensor, dim=1, keepdim=True)
        tensor = torch.log(tensor)
    elif constraint == 'column_simplex_matrix':
        tensor = torch.clamp(tensor, min=eps)
        tensor = tensor / torch.sum(tensor, dim=0, keepdim=True)
        tensor = torch.log(tensor)
    elif _objective_is_coupled_bound_constraint(constraint):
        raise ValueError('Coupled objective constraints are initialized by ObjectiveParameterSet.')
    elif constraint != 'real':
        raise ValueError('Unknown objective parameter constraint %s.' % constraint)
    tensor.requires_grad_(True)
    return tensor


def _objective_constrained_value(raw: Any, constraint: str, torch: Any, values: Optional[Mapping[str, Any]] = None) -> Any:
    if constraint in ('positive', 'positive_vector', 'positive_matrix'):
        return torch.exp(raw)
    if constraint == 'unit_interval':
        return torch.sigmoid(raw)
    if constraint in ('simplex', 'simplex_vector'):
        return torch.softmax(raw, dim=0)
    if constraint == 'row_simplex_matrix':
        return torch.softmax(raw, dim=1)
    if constraint == 'column_simplex_matrix':
        return torch.softmax(raw, dim=0)
    if _objective_is_greater_than_constraint(constraint):
        anchor = _objective_bound_anchor(constraint)
        if values is None or anchor not in values:
            raise ValueError('Objective constraint %s requires resolved anchor %s.' % (constraint, anchor))
        return values[anchor] + torch.exp(raw)
    if _objective_is_less_than_constraint(constraint):
        anchor = _objective_bound_anchor(constraint)
        if values is None or anchor not in values:
            raise ValueError('Objective constraint %s requires resolved anchor %s.' % (constraint, anchor))
        return values[anchor] - torch.exp(raw)
    return raw


def _objective_is_greater_than_constraint(constraint: str) -> bool:
    return str(constraint).startswith('greater_than:')


def _objective_is_less_than_constraint(constraint: str) -> bool:
    return str(constraint).startswith('less_than:')


def _objective_is_coupled_bound_constraint(constraint: str) -> bool:
    return _objective_is_greater_than_constraint(constraint) or _objective_is_less_than_constraint(constraint)


def _objective_bound_anchor(constraint: str) -> str:
    anchor = str(constraint).split(':', 1)[1] if ':' in str(constraint) else ''
    if not anchor:
        raise ValueError('%s constraint requires an anchor parameter.' % constraint)
    return anchor


def _objective_bound_delta(value: Any, anchor_value: Any, constraint: str) -> Any:
    value_arr = np.asarray(value, dtype=np.float64)
    anchor_arr = np.asarray(anchor_value, dtype=np.float64)
    if _objective_is_greater_than_constraint(constraint):
        delta = value_arr - anchor_arr
    else:
        delta = anchor_arr - value_arr
    if np.any(delta <= 0.0) or not np.all(np.isfinite(delta)):
        raise ValueError('Initial value for %s must satisfy its coupled bound.' % constraint)
    return delta


def _detach_objective_value(value: Any) -> Any:
    if hasattr(value, 'detach'):
        arr = value.detach().cpu().numpy()
        return float(arr) if np.ndim(arr) == 0 else arr
    return value


def projection_samples(source: SequenceEncodableProbabilityDistribution,
                       sample_size: int,
                       seed: Optional[int] = None) -> Sequence[Any]:
    """Draw reusable samples for Monte-Carlo projection experiments."""
    if sample_size <= 0:
        raise ValueError('sample_size must be positive.')
    rng = RandomState(seed)
    return source.sampler(seed=int(rng.randint(2 ** 31 - 1))).sample(size=int(sample_size))
