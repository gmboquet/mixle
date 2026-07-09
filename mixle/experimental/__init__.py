"""``mixle.experimental`` -- exploratory surfaces that are not (yet) part of mixle's mature API.

Code here is kept for exploration and may change or be removed without the usual stability guarantees.

Current contents:

- :mod:`mixle.experimental.program` -- the optimization-*program* approach (moves + combinators: ``minimize`` /
  ``maximize`` / ``em`` / ``alternate`` / ``weighted`` / ``constrain`` / ``reinforce`` / ``pareto`` / ``bilevel``
  / ``gail`` / ``maxent_irl``) to fitting heterogeneous neural + stats models. A reasonable idea that wasn't
  mature: its closure-taking surface (``minimize(lambda: loss, over=params)``) is exactly the PyTorch-style jank
  it set out to avoid. For the common cases it is **superseded by the declarative neural surface** --
  ``Categorical(logits=Net(...)).fit(y, given=...)``, ``Normal(Net(...), free).fit(...)``, and mixtures of
  ``SoftmaxNeuralLeaf`` experts -- which compose into the PPL with no loss closures. It is kept here for the
  genuinely game-shaped cases the declarative surface does not reach (GANs, on-policy RL).
- :mod:`mixle.experimental.graduation` -- the bookkeeping ledger (:class:`~mixle.experimental.graduation.ExperimentalMechanism`,
  ``REGISTRY``) that later long-context mechanisms register against to track graduation eligibility. See
  ``mixle/experimental/README.md`` for the graduation contract itself.
- :mod:`mixle.experimental.context_spine` -- E1, the chunked-recurrent training spine (TBPTT):
  the :class:`~mixle.experimental.context_spine.ContextMechanism` protocol (``init_state``/``step``/``detach``),
  the ``train_tbptt`` driver, and :class:`~mixle.experimental.context_spine.SlidingWindowSpine` -- the baseline
  mechanism (RoPE + sliding-window attention with a stop-gradient carried KV cache, Transformer-XL style) every
  later Track-E mechanism (E2-E6) is compared against. See ``notes/designs/E1.md`` for the design.

Tests for code under here are tagged ``@pytest.mark.experimental`` (see ``pyproject.toml``) so they can be
run and reported on distinctly from the stable-package suite.
"""
