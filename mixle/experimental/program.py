"""Declarative optimization programs (differentiable games) over parameter groups and objectives.

The whole zoo of "fit this model" -- supervised, LoRA fine-tuning, multi-objective, GANs, constrained
optimization, policy-gradient RL, continual learning, and classical EM -- is one idea at the optimization
level: a **program** built from MOVES and COMBINATORS.

- A **move** is *minimize / maximize an objective over scoped parameters* (:func:`minimize`, :func:`maximize`).
  An ``em(...)`` step on a mixle estimator is also a move, so probabilistic and neural models compose in one
  program.
- **Scoped parameters** decide *which* tensors move: :func:`trainable`, :func:`freeze`, :func:`subset`, and
  :func:`lora` (low-rank adapters -- the base stays frozen).
- **Combinators** schedule the moves: :func:`weighted` (cooperative / multi-objective), :func:`alternate`
  (adversarial / coordinate / EM). :func:`constrain` adds a Lagrange-multiplier *dual player* (constrained
  optimization is the same min-max game as a GAN). :func:`reinforce` turns a sampled reward into a
  score-function objective (RL).
- :func:`fit` runs the program.

Examples::

    fit(minimize(nll, over=trainable(net)))                                  # supervised
    fit(minimize(lm_loss, over=lora(model, rank=8)))                         # LoRA fine-tune
    fit(weighted([(recon, 1.0), (kl, beta)], over=trainable([enc, dec])))    # multi-objective (VAE)
    fit(alternate(minimize(d_loss, over=D), minimize(g_loss, over=G)))       # GAN
    fit(minimize(f, over=th), constraints=[constrain(g, 0.0, "<=")])         # constrained (primal-dual)
    fit(maximize(reinforce(sample_reward), over=policy))                     # policy-gradient RL
    fit(weighted([(new_loss, 1.0), (replay_loss, 1.0)], over=net))           # continual learning (replay)

Torch is imported lazily, so this module imports without it; the gradient moves require torch, the ``em``
move requires only mixle estimators.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover - environment without torch
        raise ImportError("mixle.program gradient moves require torch (`pip install mixle-learn[torch]`).") from e
    return torch


# ---------------------------------------------------------------------------------------------------------
# Scoped parameter handles: which tensors a move is allowed to change.
# ---------------------------------------------------------------------------------------------------------
def trainable(module: Any) -> list:
    """All ``requires_grad`` parameters of a module (or a list of modules)."""
    mods = module if isinstance(module, (list, tuple)) else [module]
    return [p for m in mods for p in m.parameters() if p.requires_grad]


def freeze(module: Any) -> Any:
    """Freeze a module in place (``requires_grad = False``) and return it -- e.g. a teacher / old checkpoint."""
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def subset(module: Any, *name_substrings: str) -> list:
    """Trainable parameters whose name contains any of ``name_substrings`` (partial fine-tuning)."""
    return [p for n, p in module.named_parameters() if p.requires_grad and any(s in n for s in name_substrings)]


class LoRALinear:
    """A ``torch.nn.Module`` wrapping a frozen ``Linear`` with a trainable low-rank adapter ``B @ A``."""

    def __new__(cls, base: Any, rank: int, alpha: float) -> Any:
        torch = _torch()
        if isinstance(rank, bool) or not isinstance(rank, Integral) or int(rank) <= 0:
            raise ValueError("rank must be a positive exact integer")
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(float(alpha)):
            raise ValueError("alpha must be a finite real number")
        rank = int(rank)

        class _LoRALinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                for p in base.parameters():
                    p.requires_grad_(False)
                self.A = torch.nn.Parameter(base.weight.new_empty(base.in_features, rank))
                torch.nn.init.normal_(self.A, std=1.0 / rank**0.5)
                self.B = torch.nn.Parameter(base.weight.new_zeros(rank, base.out_features))
                self.scaling = float(alpha) / float(rank)

            def forward(self, x: Any) -> Any:
                return self.base(x) + (x @ self.A @ self.B) * self.scaling

        return _LoRALinear()


def lora(module: Any, rank: int = 8, alpha: float = 16.0) -> list:
    """Replace every ``Linear`` under ``module`` with a LoRA adapter (base frozen) and return the adapter params.

    The module's forward now routes through low-rank adapters; only the ``A``/``B`` matrices are trainable, so
    ``minimize(loss, over=lora(model, rank=8))`` fine-tunes a large model cheaply.
    """
    torch = _torch()
    adapters: list = []

    def replace(m: Any) -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, torch.nn.Linear):
                w = LoRALinear(child, rank, alpha)
                setattr(m, name, w)
                adapters.append(w)
            else:
                replace(child)

    replace(module)
    if not adapters:
        raise ValueError("lora requires a module containing at least one torch.nn.Linear layer")
    return [p for a in adapters for p in (a.A, a.B)]


# ---------------------------------------------------------------------------------------------------------
# Moves.
# ---------------------------------------------------------------------------------------------------------
class Move:
    """Minimize (``sign=+1``) or maximize (``sign=-1``) ``objective()`` over ``params`` (a list of tensors)."""

    def __init__(self, objective: Callable[[], Any], params: Iterable, sign: float, lr: float | None = None) -> None:
        if not callable(objective):
            raise TypeError("objective must be callable")
        self.objective = objective
        self.params = list(params)
        if not self.params:
            raise ValueError("gradient moves require at least one parameter")
        self.sign = float(sign)
        self.lr = lr

    def _step(self, optimizer: Any) -> float:
        optimizer.zero_grad()
        loss = self.sign * self.objective()
        if not hasattr(loss, "backward"):
            raise TypeError("a gradient objective must return a differentiable scalar tensor")
        if loss.numel() != 1:
            raise ValueError("a gradient objective must return a scalar tensor")
        if not bool(_torch().isfinite(loss.detach()).item()):
            raise ValueError("a gradient objective must return a finite scalar")
        if not loss.requires_grad:
            raise ValueError("a gradient objective must depend on at least one trainable parameter")
        loss.backward()
        optimizer.step()
        return float(loss.detach())


def minimize(objective: Callable[[], Any], over: Iterable, lr: float | None = None) -> Move:
    """Create a gradient move that minimizes ``objective`` over the provided parameters."""
    return Move(objective, over, +1.0, lr)


def maximize(objective: Callable[[], Any], over: Iterable, lr: float | None = None) -> Move:
    """Create a gradient move that maximizes ``objective`` over the provided parameters."""
    return Move(objective, over, -1.0, lr)


class _ModelState:
    def __init__(self, model: Any) -> None:
        self.model = model


class EMMove:
    """An EM step on a mixle estimator -- a first-class move, so stats models join the program.

    ``move.model`` is the current fitted distribution; other moves' objectives may read it (e.g. a gating
    network reading the mixture's responsibilities), making neural<->stats coupling one program.
    """

    def __init__(self, estimator: Any, data: Sequence, init: Any) -> None:
        self.estimator = estimator
        self.data = data
        self._state = _ModelState(init)

    @property
    def model(self) -> Any:
        """Current fitted model carried by the EM move."""
        return self._state.model

    def _step(self) -> None:
        from mixle.inference import estimate

        self._state.model = estimate(self.data, self.estimator, self._state.model)


def em(estimator: Any, data: Sequence, init: Any) -> EMMove:
    """A mixle EM step as a move. ``init`` is the starting distribution (the E-step needs a current model)."""
    return EMMove(estimator, data, init)


# ---------------------------------------------------------------------------------------------------------
# Combinators: a Program is a schedule of moves run once each per round.
# ---------------------------------------------------------------------------------------------------------
class Program:
    """Ordered schedule of optimization or EM moves."""

    def __init__(self, moves: Sequence) -> None:
        self.moves = list(moves)
        self.constraint_state: ConstraintState | None = None


def _as_program(p: Any) -> Program:
    if isinstance(p, Program):
        return p
    if isinstance(p, (list, tuple)):
        return Program(list(p))
    if hasattr(p, "_step"):  # any move: Move / EMMove / StreamingEMMove / ParetoMove
        return Program([p])
    raise TypeError("expected a Move / Program / list, got %r" % type(p))


def alternate(*items: Any) -> Program:
    """Run each move (or sub-program) once per round, in order -- GANs, EM, coordinate ascent."""
    moves: list = []
    for it in items:
        moves.extend(_as_program(it).moves)
    return Program(moves)


def weighted(terms: Sequence[tuple], over: Iterable) -> Program:
    """A single move minimizing ``sum(w * objective() for objective, w in terms)`` -- cooperative multi-objective."""
    term_list = list(terms)
    if not term_list:
        raise ValueError("weighted requires at least one (objective, weight) term")
    for objective, weight in term_list:
        if not callable(objective):
            raise TypeError("every weighted objective must be callable")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(float(weight)):
            raise ValueError("every weighted objective weight must be a finite real number")

    def combined() -> Any:
        total = None
        for objective, w in term_list:
            v = float(w) * objective()
            total = v if total is None else total + v
        return total

    return Program([Move(combined, over, +1.0)])


# ---------------------------------------------------------------------------------------------------------
# Constraints: a Lagrange multiplier is the "dual player" -- constrained optimization is a min-max game.
# ---------------------------------------------------------------------------------------------------------
class Constraint:
    """Scalar inequality constraint represented for primal-dual optimization."""

    def __init__(self, g: Callable[[], Any], bound: float = 0.0, kind: str = "<=") -> None:
        if not callable(g):
            raise TypeError("constraint function must be callable")
        if kind not in ("<=", ">="):
            raise ValueError("constraint kind must be '<=' or '>='")
        if isinstance(bound, bool) or not isinstance(bound, Real) or not math.isfinite(float(bound)):
            raise ValueError("constraint bound must be a finite real number")
        self.g = g
        self.bound = float(bound)
        self.kind = kind

    def violation(self) -> Any:
        """Return the signed constraint violation optimized by the dual player."""
        return (self.g() - self.bound) if self.kind == "<=" else (self.bound - self.g())


def constrain(g: Callable[[], Any], bound: float = 0.0, kind: str = "<=") -> Constraint:
    """``g() <= bound`` (or ``>=``). Passed to :func:`fit` as ``constraints=[...]``; enforced by dual ascent."""
    return Constraint(g, bound, kind)


@dataclass(frozen=True)
class ConstraintState:
    """Observable state for one constrained :func:`fit` invocation."""

    constraints: tuple[Constraint, ...]
    multipliers: tuple[Any, ...]
    primal_move_index: int

    def multiplier_values(self) -> tuple[float, ...]:
        """Return a detached snapshot of the non-negative Lagrange multipliers."""
        return tuple(float(multiplier.detach().item()) for multiplier in self.multipliers)


def _constraint_value(constraint: Constraint, reference: Any, torch: Any) -> Any:
    value = constraint.violation()
    if not torch.is_tensor(value):
        value = reference.new_tensor(value)
    else:
        value = value.to(device=reference.device, dtype=reference.dtype)
    if value.numel() != 1:
        raise ValueError("constraint functions must return scalar values")
    if not bool(torch.isfinite(value.detach()).item()):
        raise ValueError("constraint functions must return finite values")
    return value.reshape(())


class _DualConstraintMove(Move):
    def __init__(self, multiplier: Any, constraint: Constraint, reference: Any) -> None:
        self.constraint = constraint
        self.reference = reference
        super().__init__(lambda: multiplier, [multiplier], -1.0)

    def _step(self, optimizer: Any) -> float:
        torch = _torch()
        optimizer.zero_grad()
        violation = _constraint_value(self.constraint, self.reference, torch).detach()
        loss = -self.params[0] * violation
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.params[0].clamp_(min=0.0)
        return float(loss.detach())


def _augment_with_constraints(
    moves: list, constraints: Sequence[Constraint], torch: Any
) -> tuple[list, ConstraintState]:
    """Build an ephemeral sign-correct primal/dual game without mutating caller moves."""
    constraint_list = tuple(constraints)
    if not constraint_list:
        raise ValueError("constraints must contain at least one Constraint")
    if any(not isinstance(constraint, Constraint) for constraint in constraint_list):
        raise TypeError("constraints must contain only Constraint instances")
    try:
        primal_index, primal = next(
            (index, move) for index, move in enumerate(moves) if type(move) is Move and move.params
        )
    except StopIteration as exc:
        raise ValueError("constraints require at least one scalar gradient Move with parameters") from exc
    reference = primal.params[0]
    if not torch.is_tensor(reference) or not (reference.is_floating_point() or reference.is_complex()):
        raise TypeError("constrained primal parameters must be floating-point or complex tensors")
    if reference.is_complex():
        raise TypeError("constrained optimization does not support complex parameters")
    multipliers = tuple(reference.new_zeros((), requires_grad=True) for _ in constraint_list)
    original_objective = primal.objective

    def augmented() -> Any:
        objective = original_objective()
        penalty = reference.new_zeros(())
        for multiplier, constraint in zip(multipliers, constraint_list):
            penalty = penalty + multiplier.detach() * _constraint_value(constraint, reference, torch)
        # Move._step minimizes sign*objective. This makes that optimizer loss
        # sign*original + penalty for both minimization and maximization.
        return objective + primal.sign * penalty

    constrained_primal = Move(augmented, primal.params, primal.sign, primal.lr)
    augmented_moves = list(moves)
    augmented_moves[primal_index] = constrained_primal
    augmented_moves.extend(
        _DualConstraintMove(multiplier, constraint, reference)
        for multiplier, constraint in zip(multipliers, constraint_list)
    )
    state = ConstraintState(constraint_list, multipliers, primal_index)
    return augmented_moves, state


# ---------------------------------------------------------------------------------------------------------
# RL: a sampled reward becomes a score-function (REINFORCE) objective.
# ---------------------------------------------------------------------------------------------------------
def reinforce(sample_and_reward: Callable[[], tuple]) -> Callable[[], Any]:
    """Wrap ``sample_and_reward() -> (log_probs, rewards)`` into the score-function surrogate ``E[r·logπ]``.

    Maximizing it gives the policy gradient ``E[r·∇logπ]``. ``log_probs`` are the log-probabilities of the
    sampled actions (carry grad); ``rewards`` are detached returns.
    """

    def objective() -> Any:
        logp, rewards = sample_and_reward()
        return (rewards.detach() * logp).mean()

    return objective


# ---------------------------------------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------------------------------------
def fit(
    program: Any,
    steps: int = 1000,
    lr: float = 1e-3,
    constraints: Sequence[Constraint] | None = None,
    callback: Callable[[int, Program], None] | None = None,
    data: Stream | None = None,
    steps_per_chunk: int = 1,
) -> Program:
    """Run an optimization program; each round runs every move once, in order.

    Gradient moves take one optimizer step (Adam, per-move learning rate ``move.lr`` or the global ``lr``);
    ``em`` moves take one EM step. ``constraints`` add dual-ascent multiplier moves (primal-dual).

    Fixed mode (``data`` is None): run ``steps`` rounds. **Streaming mode** (``data`` is a :class:`Stream`):
    advance through the data chunks -- the parameters/optimizers persist across chunks (warm-started), running
    ``steps_per_chunk`` rounds per chunk; objectives read ``stream.current``. This is the continuous-pretraining
    loop (combine the task loss with an anti-forget term via :func:`weighted`). Returns the program updated in
    place.
    """
    prog = _as_program(program)
    moves = list(prog.moves)
    if not moves:
        raise ValueError("fit requires a program containing at least one move")
    prog.constraint_state = None
    if constraints is not None:
        torch = _torch()
        moves, prog.constraint_state = _augment_with_constraints(moves, constraints, torch)
    grad_moves = [m for m in moves if isinstance(m, Move) and m.params]
    optimizers = {}
    if grad_moves:
        torch = _torch()
        optimizers = {id(m): torch.optim.Adam(m.params, lr=(m.lr or lr)) for m in grad_moves}

    def run_round() -> None:
        for m in moves:
            if isinstance(m, Move):
                if m.params:
                    m._step(optimizers[id(m)])
            else:  # EMMove / StreamingEMMove and any other stats move
                m._step()

    if data is None:
        for step in range(int(steps)):
            run_round()
            if callback is not None:
                callback(step, prog)
    else:
        chunk = 0
        while steps is None or chunk < int(steps):  # consume the stream (params persist across chunks)
            data.advance()
            if data.done:
                break
            for _ in range(int(steps_per_chunk)):
                run_round()
            if callback is not None:
                callback(chunk, prog)
            chunk += 1
    return prog


# ---------------------------------------------------------------------------------------------------------
# Continuous pre-training (CPT): a streaming program with anti-forgetting terms.
# ---------------------------------------------------------------------------------------------------------
class Stream:
    """A holder over an iterable of data chunks. ``fit(data=stream)`` advances it each round; objectives read
    ``stream.current`` (the active chunk). The model's parameters persist across chunks (warm-started)."""

    def __init__(self, chunks: Iterable) -> None:
        self._it = iter(chunks)
        self.current: Any = None
        self.done = False
        self.index = -1

    def advance(self) -> None:
        """Advance to the next chunk and mark the stream done at exhaustion."""
        try:
            self.current = next(self._it)
            self.index += 1
        except StopIteration:
            self.done = True


class ReplayBuffer:
    """Fixed-capacity FIFO of past chunks, for replay-based anti-forgetting."""

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = int(capacity)
        self.items: list = []

    def add(self, item: Any) -> ReplayBuffer:
        """Append a chunk and evict the oldest item when capacity is exceeded."""
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)
        return self

    def all(self) -> list:
        """Return a snapshot list of buffered replay chunks."""
        return list(self.items)


def snapshot(params: Iterable) -> list:
    """Detached clones of ``params`` -- the anchor for :func:`ewc` / L2-SP regularization."""
    return [p.detach().clone() for p in params]


def replay(loss_fn: Callable[[Any], Any], buffer: ReplayBuffer) -> Callable[[], Any]:
    """An objective averaging ``loss_fn(chunk)`` over the replay buffer -- a term for :func:`weighted`."""

    def obj() -> Any:
        chunks = buffer.all()
        if not chunks:
            raise ValueError("cannot evaluate a replay objective with an empty buffer")
        total = None
        for c in chunks:
            v = loss_fn(c)
            total = v if total is None else total + v
        return total / len(chunks)

    return obj


def distill(student_out: Callable[[], Any], teacher_out: Callable[[], Any]) -> Callable[[], Any]:
    """MSE distillation: keep student outputs near a frozen teacher's (logit/feature matching anti-forget)."""

    def obj() -> Any:
        return ((student_out() - teacher_out().detach()) ** 2).mean()

    return obj


def ewc(params: Iterable, fisher: Sequence, anchor: Sequence, weight: float = 1.0) -> Callable[[], Any]:
    """Elastic Weight Consolidation penalty ``weight · Σ Fᵢ (θᵢ - anchorᵢ)²`` -- anchors params important to the
    old task. Pair with :func:`fisher_diagonal` (a torch net) or a mixle leaf's ``to_fisher``."""
    plist = list(params)
    fisher_list = list(fisher)
    anchor_list = list(anchor)
    if not plist:
        raise ValueError("ewc requires at least one parameter")
    if len(fisher_list) != len(plist) or len(anchor_list) != len(plist):
        raise ValueError("params, fisher, and anchor must have equal non-zero lengths")

    def obj() -> Any:
        total = plist[0].new_zeros(())
        for f, p, a in zip(fisher_list, plist, anchor_list):
            total = total + (f * (p - a) ** 2).sum()
        return float(weight) * total

    return obj


def fisher_diagonal(net: Any, batches: Iterable, kind: str = "classification") -> list:
    """Diagonal Fisher information for EWC, using MODEL-sampled labels so it does NOT vanish at convergence.

    ``kind='classification'`` (``net`` returns logits) or ``'regression'`` (``net`` returns a Gaussian mean).
    Returns one tensor per trainable parameter. (The naive data-label Fisher is ~0 at a converged optimum --
    sampling the label from the model is what makes EWC actually anchor.)
    """
    torch = _torch()
    if kind not in {"classification", "regression"}:
        raise ValueError("kind must be 'classification' or 'regression'")
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        raise ValueError("fisher_diagonal requires at least one trainable parameter")
    fisher = [torch.zeros_like(p) for p in params]
    n = 0
    for x in batches:
        out = net(x)
        if kind == "classification":
            y = torch.distributions.Categorical(logits=out).sample()
            ll = -torch.nn.functional.cross_entropy(out, y, reduction="sum")
        else:
            y = out.detach() + torch.randn_like(out)
            ll = -0.5 * ((y - out) ** 2).sum()
        net.zero_grad()
        ll.backward()
        for f, p in zip(fisher, params):
            if p.grad is not None:
                f += p.grad.detach() ** 2
        n += int(out.shape[0]) if hasattr(out, "shape") and out.dim() > 0 else 1
    if n == 0:
        raise ValueError("fisher_diagonal requires at least one non-empty batch")
    return [f / n for f in fisher]


# ---------------------------------------------------------------------------------------------------------
# Meta-learning: bilevel (the inner adaptation is differentiated through -- MAML).
# ---------------------------------------------------------------------------------------------------------
def bilevel(
    model: Any,
    inner_loss: Callable[[Callable, Any], Any],
    outer_loss: Callable[[Callable, Any], Any],
    sample_tasks: Callable[[], Iterable[tuple]],
    inner_steps: int = 1,
    inner_lr: float = 0.01,
) -> Move:
    """Meta-learning (MAML): meta-learn ``model``'s params so a few inner gradient steps adapt to a task.

    ``sample_tasks()`` yields ``(support, query)`` batches; ``inner_loss(forward, support)`` and
    ``outer_loss(forward, query)`` return scalars, where ``forward(x)`` runs the model with the *current*
    (adapted) parameters. The returned move's objective differentiates **through** the inner adaptation
    (second-order), so ``fit(bilevel(...), steps=N)`` is MAML; ``fit`` minimizes the query loss over the
    meta-parameters.
    """
    torch = _torch()
    from torch.func import functional_call

    names = [n for n, p in model.named_parameters() if p.requires_grad]

    def outer_objective() -> Any:
        meta = {n: p for n, p in model.named_parameters() if p.requires_grad}
        total = None
        count = 0
        for support, query in sample_tasks():
            adapted = dict(meta)
            for _ in range(int(inner_steps)):
                loss = inner_loss(lambda x, a=adapted: functional_call(model, a, x), support)
                grads = torch.autograd.grad(loss, list(adapted.values()), create_graph=True)
                adapted = {n: adapted[n] - inner_lr * g for n, g in zip(names, grads)}
            q = outer_loss(lambda x, a=adapted: functional_call(model, a, x), query)
            total = q if total is None else total + q
            count += 1
        if count == 0:
            raise ValueError("sample_tasks must yield at least one (support, query) task")
        return total / count

    return minimize(outer_objective, over=trainable(model))


# ---------------------------------------------------------------------------------------------------------
# True multi-objective: MGDA -- step along the minimum-norm common-descent direction (Pareto).
# ---------------------------------------------------------------------------------------------------------
def _mgda_weights(grads: list, torch: Any) -> Any:
    """Frank-Wolfe for the min-norm point in the convex hull of the per-objective gradients (MGDA)."""
    n = len(grads)
    if n == 0:
        raise ValueError("MGDA requires at least one objective")
    if n == 1:
        return grads[0][0].new_ones(1)
    flat = [torch.cat([g.flatten() for g in gi]) for gi in grads]
    gram = torch.stack([torch.stack([(a * b).sum() for b in flat]) for a in flat])  # (n, n)
    alpha = gram.new_full((n,), 1.0 / n)
    for _ in range(50):
        t = int(torch.argmin(gram @ alpha).item())  # vertex with steepest descent of alpha^T M alpha
        e = torch.zeros_like(alpha)
        e[t] = 1.0
        d = e - alpha
        denom = float((d @ gram @ d).item())
        gamma = float(torch.clamp(-(alpha @ gram @ d) / (denom + 1e-12), 0.0, 1.0).item()) if denom > 1e-12 else 0.0
        if gamma <= 1e-9:
            break
        alpha = alpha + gamma * d
    return alpha


class ParetoMove(Move):
    """A move that steps along the MGDA common-descent direction -- decreases every objective at once."""

    def __init__(self, objectives: Sequence[Callable[[], Any]], params: Iterable, lr: float | None = None) -> None:
        super().__init__(objective=lambda: None, params=params, sign=+1.0, lr=lr)
        self.objectives = list(objectives)
        if not self.objectives:
            raise ValueError("pareto requires at least one objective")
        if any(not callable(objective) for objective in self.objectives):
            raise TypeError("every Pareto objective must be callable")

    def _step(self, optimizer: Any) -> float:
        torch = _torch()
        grads = []
        for obj in self.objectives:
            optimizer.zero_grad()
            obj().backward()
            grads.append(
                [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in self.params]
            )
        alpha = _mgda_weights(grads, torch)
        optimizer.zero_grad()
        for j, p in enumerate(self.params):
            p.grad = torch.stack([alpha[i] * grads[i][j] for i in range(len(grads))]).sum(dim=0)
        optimizer.step()
        return 0.0


def pareto(objectives: Sequence[Callable[[], Any]], over: Iterable, lr: float | None = None) -> Program:
    """Multi-objective optimization with NO fixed weights: each step descends all objectives at once (MGDA)."""
    return Program([ParetoMove(objectives, over, lr)])


# ---------------------------------------------------------------------------------------------------------
# Streaming EM: a stats model that continually adapts over a chunk stream (alongside neural moves).
# ---------------------------------------------------------------------------------------------------------
class StreamingEMMove:
    """Online EM over a :class:`Stream`: each round, warm-started EM iterations on the current chunk.

    Composes in ``fit(data=stream)`` next to gradient moves, so a stats model and a neural net adapt to the
    same stream together (the LLM<->stats continual-coupling case). ``move.model`` is the current fit.
    """

    def __init__(self, estimator: Any, stream: Stream, init: Any, iters_per_chunk: int = 1) -> None:
        self.estimator = estimator
        self.stream = stream
        self.iters_per_chunk = int(iters_per_chunk)
        self._state = _ModelState(init)

    @property
    def model(self) -> Any:
        """Current model carried across streaming EM chunks."""
        return self._state.model

    def _step(self) -> None:
        from mixle.inference import estimate

        chunk = self.stream.current
        if chunk is None:
            return
        m = self._state.model
        for _ in range(self.iters_per_chunk):
            m = estimate(chunk, self.estimator, m)
        self._state.model = m


def streaming_em(estimator: Any, stream: Stream, init: Any, iters_per_chunk: int = 1) -> StreamingEMMove:
    """A stats EM move that continually adapts over a chunk stream (use inside ``fit(data=stream)``)."""
    return StreamingEMMove(estimator, stream, init, iters_per_chunk)


# ---------------------------------------------------------------------------------------------------------
# Inverse RL: learn the OBJECTIVE (reward) from demonstrations -- compositions of the combinators above.
# ---------------------------------------------------------------------------------------------------------
def gail(
    discriminator: Callable[[Any], Any],
    sample_expert: Callable[[], Any],
    sample_policy: Callable[[], tuple],
    disc_params: Iterable,
    policy_params: Iterable,
) -> Program:
    """GAIL / adversarial inverse RL = ``alternate(minimize(disc_loss), maximize(reinforce(policy)))``.

    Recover an expert's behavior (and a reward) from demonstrations alone. ``discriminator(features) -> logits``
    (high = expert; **this logit is the recovered reward**). ``sample_expert() -> features`` is a batch of
    expert transition features; ``sample_policy() -> (features, action_logprobs)`` is a policy rollout. The
    discriminator separates expert from policy transitions while the policy is reinforced to fool it.
    """
    torch = _torch()
    f = torch.nn.functional

    def disc_loss() -> Any:
        d_e = discriminator(sample_expert())
        d_p = discriminator(sample_policy()[0].detach())
        return -(f.logsigmoid(d_e).mean() + f.logsigmoid(-d_p).mean())

    def policy_return() -> tuple:
        feats, logp = sample_policy()
        return logp, discriminator(feats).detach()

    return alternate(minimize(disc_loss, disc_params), maximize(reinforce(policy_return), policy_params))


def maxent_irl(
    reward: Callable[[Any], Any],
    reward_params: Iterable,
    expert_features: Any,
    policy_features: Callable[[], Any],
) -> Move:
    """Maximum-entropy inverse RL by feature matching (Ziebart et al.).

    ``reward(features) -> scalar`` (e.g. ``w·φ``). ``expert_features`` is the expert's expected feature vector
    (from demonstrations). ``policy_features() -> features`` returns the expected features under the
    **maxent-optimal policy for the current reward** -- the inner forward / soft-value solve, recomputed each
    step. For *structured* dynamics that inner solve is exactly mixle's forward (soft-value) pass, so the
    partition function is computed without sampling. The move's gradient matches expert to policy features.
    """

    def objective() -> Any:
        return -(reward(expert_features) - reward(policy_features().detach()))

    return minimize(objective, over=reward_params)
