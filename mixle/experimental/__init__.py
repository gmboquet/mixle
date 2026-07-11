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
- :mod:`mixle.experimental.retrieval_memory_spine` -- E6, retrieval memory over frozen past:
  :class:`~mixle.experimental.retrieval_memory_spine.RetrievalMemorySpine` pairs E1's local sliding window
  with a brute-force kNN index of detached past chunks, retrieving the top-k per query each step. Gradients
  flow exactly through the retrieval softmax over the selected top-k; the archived index contents themselves
  are stop-gradient -- that non-differentiable boundary is a receipt field on the returned state, not just a
  docstring claim.
- :mod:`mixle.experimental.selective_scan` -- E5 part 1, the S6/Mamba selective-scan module:
  :class:`~mixle.experimental.selective_scan.SelectiveScan`, its ``_scan_layer`` recurrence (shared with
  :mod:`mixle.experimental.ssm_hybrid`, not duplicated), and the S4D-real / dt-bias inits verified against
  ``mamba-ssm`` source.
- :mod:`mixle.experimental.ssm_hybrid` -- E5 part 2, the hybrid block:
  :class:`~mixle.experimental.ssm_hybrid.HybridBlock` composes E1's local windowed attention, E5 part 1's
  selective-scan SSM branch, and E2's moment-closure far field into one ``ContextMechanism``, with a real
  per-mechanism contribution receipt exposed via ``report()``. See ``notes/designs/E5.md`` for the design.
- :mod:`mixle.experimental.long_context_eval` -- E7, the long-context referee suite (needle / copy /
  multi-hop / multi-scale-perplexity probes, a length curriculum, matched-FLOPs / matched-state-bytes
  bookkeeping) every Track-E mechanism is measured against on the same terms.
- :mod:`mixle.experimental.summary_tree` -- E4, the hierarchical summary tree: E1's exact near field
  plus a persistent, bounded far-field tree of learned summaries built via mixed-radix carry
  propagation over evicted tokens (the fast-multipole-method structure), a tree-path positional
  encoding replacing RoPE for the far field, a predict-the-summary auxiliary loss, and a receipted
  stop-gradient horizon. See ``notes/designs/E4.md`` for the design.
- :mod:`mixle.experimental.tensor_network` -- P4, matrix-product-state (tensor-train) leaves:
  :class:`~mixle.experimental.tensor_network.MPS` is a Born-machine density over discrete sequences with
  exact normalization / marginals / conditionals by contraction,
  :func:`~mixle.experimental.tensor_network.entanglement_entropy` as the long-range-structure receipt
  (bounded by ``log(bond)``), and :func:`~mixle.experimental.tensor_network.truncate_error` whose discarded
  Schmidt weight tracks the truncation's distribution error.

Tests for code under here are tagged ``@pytest.mark.experimental`` (see ``pyproject.toml``) so they can be
run and reported on distinctly from the stable-package suite.
"""
