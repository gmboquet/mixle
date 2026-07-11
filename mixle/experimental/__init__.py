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
- :mod:`mixle.experimental.wake_sleep` -- P12, wake-sleep library learning over the model-structure
  grammar: :func:`~mixle.experimental.wake_sleep.wake_sleep` solves a corpus by greedy MDL structure
  search (WAKE), anti-unifies recurring subtrees into a reusable library fragment
  (:func:`~mixle.experimental.wake_sleep.abstract_fragment`, SLEEP-ABSTRACTION), and measures that the
  fragment cuts held-out median search cost >= 2x -- while returning no fragment when the tasks share no
  motif (it does not invent structure).
- :mod:`mixle.experimental.tensor_network` -- P4, matrix-product-state (tensor-train) leaves:
  :class:`~mixle.experimental.tensor_network.MPS` is a Born-machine density over discrete sequences with
  exact normalization / marginals / conditionals by contraction,
  :func:`~mixle.experimental.tensor_network.entanglement_entropy` as the long-range-structure receipt
  (bounded by ``log(bond)``), and :func:`~mixle.experimental.tensor_network.truncate_error` whose discarded
  Schmidt weight tracks the truncation's distribution error.
- :mod:`mixle.experimental.active_causal` -- P15, active causal discovery:
  :func:`~mixle.experimental.active_causal.active_discovery` chooses ``do(.)`` interventions by expected
  information gain over a posterior on candidate causal structures (chain/reverse/fork), identifying the
  true structure in far fewer experiments than random or observation-only selection. Exact linear-Gaussian
  so the ground truth is known and the design can be graded exactly.
- :mod:`mixle.experimental.v_information` -- P13, usable-information receipts: V-information
  (:func:`~mixle.experimental.v_information.v_information`) is the family's realized reduction in held-out
  predictive log-loss from conditioning on ``X``, and the usability gap
  ``I(X;Y) - I_V`` (against the closed-form Gaussian ``I(X;Y)``) is a receipt on the *library's* ceiling --
  large when the generative law sits outside the current grammar, closing when the missing feature is added.
- :mod:`mixle.experimental.pac_bayes` -- P10, compositional PAC-Bayes generalization certificates:
  closed-form :func:`~mixle.experimental.pac_bayes.gaussian_kl`, an additively-composing
  :func:`~mixle.experimental.pac_bayes.total_kl` with per-node blame, the McAllester
  :func:`~mixle.experimental.pac_bayes.mcallester_bound`, and
  :func:`~mixle.experimental.pac_bayes.certify_generalization` turning a Gaussian-mixture fit into a
  non-vacuous, ``1 - delta``-valid held-out-risk certificate with per-subtree blame.
- :mod:`mixle.experimental.ot_geometry` -- P6, optimal-transport geometry of model space:
  :func:`~mixle.experimental.ot_geometry.bures_wasserstein` (closed-form ``W2`` between Gaussians),
  :func:`~mixle.experimental.ot_geometry.gaussian_barycenter` (Bures barycenter fixed-point), and
  :func:`~mixle.experimental.ot_geometry.mixture_barycenter` (Wasserstein barycenter of Gaussian mixtures
  via Hungarian component alignment) -- merging models in distribution space instead of parameter space.
- :mod:`mixle.experimental.unlearning` -- P5, exact machine unlearning for closed-form leaves:
  :func:`~mixle.experimental.unlearning.certify_unlearning` re-reduces the retained shards' stored
  sufficient statistics (via each accumulator's ``combine``) in canonical order and certifies the result
  equals the never-saw-it fit bit-for-bit. NOT by subtraction -- ``T_all - T_j`` is not bitwise and can
  catastrophically cancel to a negative variance (both shown in the test). Exact for closed-form leaves;
  iterative-EM latent models are out of scope for the exact certificate.
- :mod:`mixle.experimental.spectral_health` -- P16, data-free spectral-health receipts:
  :func:`~mixle.experimental.spectral_health.spectral_health` fits the power-law tail exponent of a weight
  matrix's eigenvalue spectrum (plus stable/effective rank) and classifies the layer under-trained /
  well-trained / memorizing from the weights alone -- the heavy-tailed self-regularization lens, complementing
  G1 (moments) and R1/G4 (quantile profiles).
- :mod:`mixle.experimental.e_process` -- P9, anytime-valid receipts: :class:`~mixle.experimental.e_process.EProcess`
  (the generic running-product e-process from per-step density ratios) and the closed-form Robbins
  :func:`~mixle.experimental.e_process.normal_mixture_eprocess` / :class:`~mixle.experimental.e_process.MeanShiftDetector`
  for drift. Because a mixle density ratio is an e-value, monitoring ``E_t >= 1/alpha`` gives type-I control
  under continuous peeking and optional stopping (Ville's inequality) -- verified empirically in the test.

Tests for code under here are tagged ``@pytest.mark.experimental`` (see ``pyproject.toml``) so they can be
run and reported on distinctly from the stable-package suite.
"""
