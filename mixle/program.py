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

from collections.abc import Callable, Iterable, Sequence
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
    return [
        p
        for n, p in module.named_parameters()
        if p.requires_grad and any(s in n for s in name_substrings)
    ]


class LoRALinear:
    """A ``torch.nn.Module`` wrapping a frozen ``Linear`` with a trainable low-rank adapter ``B @ A``."""

    def __new__(cls, base: Any, rank: int, alpha: float) -> Any:
        torch = _torch()

        class _LoRALinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                for p in base.parameters():
                    p.requires_grad_(False)
                self.A = torch.nn.Parameter(torch.randn(base.in_features, rank) * (1.0 / rank**0.5))
                self.B = torch.nn.Parameter(torch.zeros(rank, base.out_features))
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
    return [p for a in adapters for p in (a.A, a.B)]


# ---------------------------------------------------------------------------------------------------------
# Moves.
# ---------------------------------------------------------------------------------------------------------
class Move:
    """Minimize (``sign=+1``) or maximize (``sign=-1``) ``objective()`` over ``params`` (a list of tensors)."""

    def __init__(self, objective: Callable[[], Any], params: Iterable, sign: float, lr: float | None = None) -> None:
        self.objective = objective
        self.params = list(params)
        self.sign = float(sign)
        self.lr = lr

    def _step(self, optimizer: Any) -> float:
        optimizer.zero_grad()
        loss = self.sign * self.objective()
        loss.backward()
        optimizer.step()
        return float(loss.detach())


def minimize(objective: Callable[[], Any], over: Iterable, lr: float | None = None) -> Move:
    return Move(objective, over, +1.0, lr)


def maximize(objective: Callable[[], Any], over: Iterable, lr: float | None = None) -> Move:
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
    def __init__(self, moves: Sequence) -> None:
        self.moves = list(moves)


def _as_program(p: Any) -> Program:
    if isinstance(p, Program):
        return p
    if isinstance(p, (Move, EMMove)):
        return Program([p])
    if isinstance(p, (list, tuple)):
        return Program(list(p))
    raise TypeError("expected a Move / EMMove / Program / list, got %r" % type(p))


def alternate(*items: Any) -> Program:
    """Run each move (or sub-program) once per round, in order -- GANs, EM, coordinate ascent."""
    moves: list = []
    for it in items:
        moves.extend(_as_program(it).moves)
    return Program(moves)


def weighted(terms: Sequence[tuple], over: Iterable) -> Program:
    """A single move minimizing ``sum(w * objective() for objective, w in terms)`` -- cooperative multi-objective."""
    term_list = list(terms)

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
    def __init__(self, g: Callable[[], Any], bound: float = 0.0, kind: str = "<=") -> None:
        if kind not in ("<=", ">="):
            raise ValueError("constraint kind must be '<=' or '>='")
        self.g = g
        self.bound = float(bound)
        self.kind = kind

    def violation(self) -> Any:
        return (self.g() - self.bound) if self.kind == "<=" else (self.bound - self.g())


def constrain(g: Callable[[], Any], bound: float = 0.0, kind: str = "<=") -> Constraint:
    """``g() <= bound`` (or ``>=``). Passed to :func:`fit` as ``constraints=[...]``; enforced by dual ascent."""
    return Constraint(g, bound, kind)


def _augment_with_constraints(moves: list, constraints: Sequence[Constraint], torch: Any) -> list:
    """Augment the primal (first) move with ``Σ λ·violation`` and return one maximizing dual move per λ."""
    softplus = torch.nn.functional.softplus
    primal = moves[0]
    lams = [(torch.zeros((), requires_grad=True), c) for c in constraints]
    orig = primal.objective

    def augmented() -> Any:
        v = orig()
        for raw, c in lams:
            v = v + softplus(raw).detach() * c.violation()
        return v

    primal.objective = augmented
    duals = []
    for raw, c in lams:
        duals.append(Move(lambda raw=raw, c=c: softplus(raw) * c.violation().detach(), [raw], -1.0))
    return duals


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
    if constraints:
        torch = _torch()
        moves = moves + _augment_with_constraints(moves, constraints, torch)
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
            elif isinstance(m, EMMove):
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
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)
        return self

    def all(self) -> list:
        return list(self.items)


def snapshot(params: Iterable) -> list:
    """Detached clones of ``params`` -- the anchor for :func:`ewc` / L2-SP regularization."""
    return [p.detach().clone() for p in params]


def replay(loss_fn: Callable[[Any], Any], buffer: ReplayBuffer) -> Callable[[], Any]:
    """An objective averaging ``loss_fn(chunk)`` over the replay buffer -- a term for :func:`weighted`."""

    def obj() -> Any:
        chunks = buffer.all()
        if not chunks:
            return _torch().zeros(())
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

    def obj() -> Any:
        return float(weight) * sum((f * (p - a) ** 2).sum() for f, p, a in zip(fisher, plist, anchor))

    return obj


def fisher_diagonal(net: Any, batches: Iterable, kind: str = "classification") -> list:
    """Diagonal Fisher information for EWC, using MODEL-sampled labels so it does NOT vanish at convergence.

    ``kind='classification'`` (``net`` returns logits) or ``'regression'`` (``net`` returns a Gaussian mean).
    Returns one tensor per trainable parameter. (The naive data-label Fisher is ~0 at a converged optimum --
    sampling the label from the model is what makes EWC actually anchor.)
    """
    torch = _torch()
    params = [p for p in net.parameters() if p.requires_grad]
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
            f += p.grad.detach() ** 2
        n += int(out.shape[0]) if hasattr(out, "shape") and out.dim() > 0 else 1
    return [f / max(n, 1) for f in fisher]
