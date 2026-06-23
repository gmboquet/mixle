"""Typed contracts for the PDE/ODE forward-model machinery.

The PDE-inverse stack does not model a forward operator as a per-operator *object* with
``apply``/``adjoint``/``assemble`` methods (the way :class:`pysp.ppl.dynamics.DynamicsOperator` is a
formal ABC for the method-of-lines spatial operator). Instead the forward map is expressed as a pair of
user **callbacks** handed two namespaces:

* a ``p`` namespace -- the latent drivers exposed by name (``p.k``, ``p.field``); see
  :class:`pysp.ppl.ops._Params`;
* an ``ops`` namespace -- a backend-agnostic facade of curated math, structured-grid assembly, the
  adjoint sparse solve, and the ODE integrators; see :class:`pysp.ppl.ops._Ops`.

This module formalizes those two de-facto interfaces as ``@runtime_checkable`` Protocols so they can be
named in type annotations and documented in one place:

* :class:`ForwardOperator` -- the ``ops`` namespace surface that assembly/solve callbacks rely on. This is
  the "operator" of the stack: it both *assembles* the differentiable sparse system
  (``divergence_form`` / ``helmholtz_operator`` / ``laplacian``) and *applies its inverse* (``sparse_solve``)
  or the operator itself (``matvec``), with adjoint gradients computed inside ``sparse_solve``.
* :class:`ForwardModel` / :class:`ObserveFn` / :class:`RhsFn` -- the callable signatures of the user-supplied
  ``forward(p, ops)`` / ``observe(solution, p, ops)`` / ``rhs(u, t, p, ops)`` callbacks.

These are typing/documentation-level: they capture the existing contract without restructuring the free
functions in :mod:`pysp.ppl.pde_solve` or changing any numerics.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ForwardOperator(Protocol):
    """The ``ops`` namespace handed to a PDE/ODE forward model: assemble the differentiable system and
    apply its (adjoint-capable) solve.

    A "forward operator" in this stack is not an object carrying ``A`` -- it is this facade. The assembly
    methods return a ``(rows, cols, vals, n)`` sparsity-pattern + value tuple for a structured grid (with
    ``vals`` a backend tensor that carries gradients to the coefficient field); :meth:`sparse_solve` then
    solves ``A u = b`` for that assembled ``A`` with gradients supplied by the adjoint method (one extra
    solve), and :meth:`matvec` applies an assembled operator without solving. The elementwise math and the
    ODE integrators let the callback compose physics without importing a tensor backend.

    Only the structurally load-bearing methods are declared here (the ones the PDE/wave/flow/shape forward
    models actually call); the concrete :class:`pysp.ppl.ops._Ops` provides additional convenience helpers
    (``log``, ``sin``, ``stack``, ``grad``, ...). The protocol is ``@runtime_checkable`` so an ``ops``
    argument can be validated with ``isinstance(ops, ForwardOperator)``.
    """

    # --- differentiable structured-grid assembly (returns (rows, cols, vals, n)) -------------------
    def divergence_form(self, kappa: Any, shape: Any, *, spacing: Any = 1.0) -> Any:
        """Assemble ``-div(kappa grad u)`` (Dirichlet) as ``(rows, cols, vals, n)``, differentiable in
        ``kappa``."""

    def helmholtz_operator(self, slowness2: Any, shape: Any, *, omega: Any, spacing: Any = 1.0) -> Any:
        """Assemble ``-lap u - omega^2 * slowness2(x) * u`` (Dirichlet) as ``(rows, cols, vals, n)``,
        differentiable in ``slowness2``."""

    def laplacian(self, shape: Any, *, spacing: Any = 1.0) -> Any:
        """Assemble the constant-coefficient negative Laplacian ``-lap`` (Dirichlet) as
        ``(rows, cols, vals, n)``."""

    # --- apply the assembled operator (solve its inverse, or the operator itself) -------------------
    def sparse_solve(self, rows: Any, cols: Any, vals: Any, n: Any, b: Any) -> Any:
        """Solve ``A u = b`` for ``A = sparse(rows, cols, vals)`` (n x n) with adjoint gradients (one extra
        solve). Gradients flow to both ``vals`` (the coefficient field) and ``b``."""

    def matvec(self, rows: Any, cols: Any, vals: Any, n: Any, x: Any) -> Any:
        """Differentiable sparse matrix-vector product ``A x`` -- apply an assembled operator without a
        solve (e.g. an explicit Laplacian in a time-stepping forward)."""

    def solve(self, A: Any, b: Any) -> Any:
        """Dense linear solve ``A u = b`` for small systems (use :meth:`sparse_solve` for large/sparse)."""

    # --- ODE / time integration (forward-model convenience) -----------------------------------------
    def integrate(self, rhs: Any, y0: Any, t_grid: Any, *, method: str = "rk4") -> Any:
        """Integrate ``du/dt = rhs(u, t)`` from ``y0`` over ``t_grid`` (``'rk4'`` or ``'euler'``)."""

    def integrate_record(self, step: Any, y0: Any, n_steps: Any, record: Any, *, checkpoint: Any = None) -> Any:
        """Step a time-dependent system, recording ``record(y, i)`` per step; ``checkpoint=K`` runs the
        adjoint-state (checkpointed) scheme for O(sqrt(steps)) memory."""

    # --- elementwise math (the structurally needed subset) ------------------------------------------
    def exp(self, x: Any) -> Any:
        """Elementwise exponential (e.g. to map a real GP field to a positive coefficient)."""

    def tensor(self, x: Any) -> Any:
        """Lift a Python/numpy value into a backend float64 tensor."""


class ObserveFn(Protocol):
    """Observation operator: map a forward solution to the predicted observables.

    ``observe(solution, p, ops) -> predicted`` selects/transforms the part of the solution that is actually
    measured (e.g. the field at receiver nodes). Optional; the default is the whole solution.
    """

    def __call__(self, solution: Any, p: Any, ops: ForwardOperator) -> Any: ...


class ForwardModel(Protocol):
    """The forward map of a PDE/ODE inverse problem.

    ``forward(p, ops) -> solution`` solves the physics from the latent drivers ``p`` using the ``ops``
    operator namespace (e.g. ``ops.sparse_solve(*ops.divergence_form(p.field, shape), b)``). Its output is
    passed to an :class:`ObserveFn` (or scored directly) by the differential proxy.
    """

    def __call__(self, p: Any, ops: ForwardOperator) -> Any: ...


class RhsFn(Protocol):
    """Initial-value right-hand side: ``rhs(u, t, p, ops) -> du/dt``.

    The framework wraps this with :meth:`ForwardOperator.integrate` to build a forward model when an
    explicit ``forward`` is not supplied.
    """

    def __call__(self, u: Any, t: Any, p: Any, ops: ForwardOperator) -> Any: ...
