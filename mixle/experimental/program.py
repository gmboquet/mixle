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
    return [p for n, p in module.named_parameters() if p.requires_grad and any(s in n for s in name_substrings)]


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
        if kind not in ("<=", ">="):
            raise ValueError("constraint kind must be '<=' or '>='")
        self.g = g
        self.bound = float(bound)
        self.kind = kind

    def violation(self) -> Any:
        """Return the signed constraint violation optimized by the dual player."""
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
        return total / max(count, 1)

    return minimize(outer_objective, over=trainable(model))


# ---------------------------------------------------------------------------------------------------------
# True multi-objective: MGDA -- step along the minimum-norm common-descent direction (Pareto).
# ---------------------------------------------------------------------------------------------------------
def _mgda_weights(grads: list, torch: Any) -> list:
    """Frank-Wolfe for the min-norm point in the convex hull of the per-objective gradients (MGDA)."""
    n = len(grads)
    if n == 1:
        return [1.0]
    flat = [torch.cat([g.flatten() for g in gi]) for gi in grads]
    gram = torch.stack([torch.stack([(a * b).sum() for b in flat]) for a in flat])  # (n, n)
    alpha = torch.ones(n) / n
    for _ in range(50):
        t = int(torch.argmin(gram @ alpha))  # vertex with steepest descent of alpha^T M alpha
        e = torch.zeros(n)
        e[t] = 1.0
        d = e - alpha
        denom = float(d @ gram @ d)
        gamma = float(torch.clamp(-(alpha @ gram @ d) / (denom + 1e-12), 0.0, 1.0)) if denom > 1e-12 else 0.0
        if gamma <= 1e-9:
            break
        alpha = alpha + gamma * d
    return [float(a) for a in alpha]


class ParetoMove(Move):
    """A move that steps along the MGDA common-descent direction -- decreases every objective at once."""

    def __init__(self, objectives: Sequence[Callable[[], Any]], params: Iterable, lr: float | None = None) -> None:
        super().__init__(objective=lambda: None, params=params, sign=+1.0, lr=lr)
        self.objectives = list(objectives)

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
            p.grad = sum(alpha[i] * grads[i][j] for i in range(len(grads)))
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
