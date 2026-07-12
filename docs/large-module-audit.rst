Large-Module Audit
==================

Fifteen modules in ``mixle`` exceed 1,500 lines. Size alone is not a defect, and this audit is **not** a
mandate to split them. Its purpose (worklist A1.7) is to record, for each large module, what it owns, the
state it carries, the optional dependencies it touches, its hot paths, its serialization hooks, and where a
*safe* extraction boundary lies — so that a future change can be scoped without a blind refactor.

The governing rule, restated from :doc:`support-policy`: **refactor a large module only where it removes a
demonstrated defect or enables isolated testing.** Do not split stable numerical code merely to satisfy a
line-count target — the EM kernels and forward/backward recursions below are load-bearing and bit-checked,
and re-slicing them risks a correctness regression for no user-visible gain.

The set of modules audited here is pinned by ``mixle/tests/large_module_audit_test.py``: a new module that
crosses 1,500 lines, or an audited entry that is deleted, fails that test until this page is updated.

How to read an entry
--------------------

* **Responsibilities** — what the module owns.
* **Stateful globals** — module-level mutable/registry state (the first thing to check before extraction).
* **Optional imports** — heavy dependencies reached lazily; base import must not require them.
* **Hot paths** — the numerically load-bearing routines; treat as bit-checked, change with parity tests.
* **Serialization** — round-trip hooks (``to_json`` / ``from_json`` / encoders); a schema-visible surface.
* **Extraction boundary** — the safe seam, or "leave as-is" when no clean seam removes a defect.

Latent sequence and mixture families
------------------------------------

These follow the canonical six-type contract (Distribution / Sampler / Accumulator / AccumulatorFactory /
Estimator / DataEncoder). Their bulk is the vectorized ``seq_update`` / ``seq_log_density`` EM machinery and,
for several, compiled Numba kernels. They share one extraction principle: the compiled kernels and the data
encoder are the only clean seams; the distribution/estimator pair is a single numerical unit.

``mixle/stats/latent/hidden_markov.py`` (4,395)
    * **Responsibilities:** the principal HMM path — distribution, scaled forward/backward, Baum-Welch EM,
      terminal/terminal-value likelihoods, Dirichlet chain prior.
    * **Stateful globals:** only ``TypeVar``\s (``T``, ``E1``, ``E2`` …); no mutable registry.
    * **Optional imports:** none at this layer (the compiled kernels live in
      ``_hidden_markov_numba_kernels.py``, already a separate module).
    * **Hot paths:** ``_hmm_forward_ll``, ``seq_log_density``, ``seq_update``, ``kernel``, the Numba route.
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** the Numba kernels are already extracted. The terminal-state and
      terminal-value likelihood blocks are the next candidate seam **if** a defect ever localizes there;
      absent that, leave as-is — this is the most parity-tested numerical code in the tree.

``mixle/stats/latent/tree_hidden_markov_model.py`` (2,571)
    * **Responsibilities:** tree-structured HMM; Numba forward/backward, Baum-Welch, posteriors, Viterbi.
    * **Stateful globals:** ``TypeVar``\s and shape aliases only.
    * **Optional imports:** none (Numba kernels inline as module functions ``numba_*``).
    * **Hot paths:** ``numba_seq_log_density``, ``numba_baum_welch``, ``numba_posteriors``, ``numba_viterbi``.
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** the ``numba_*`` kernel group is the clean seam (mirror ``hidden_markov``'s
      split into a ``_..._numba_kernels`` module) — worth doing only alongside a kernel change.

``mixle/stats/latent/structured_hmm.py`` (1,822)
    * **Responsibilities:** HMM with pluggable transition operators (dense / low-rank / sparse /
      block-diagonal / banded), 24 classes.
    * **Stateful globals:** ``_DENSE_FB_NUMBA`` (compiled-kernel handle).
    * **Optional imports:** ``jax``, ``numba``.
    * **Hot paths:** forward/backward, ``seq_update``, ``seq_log_density``, ``logsumexp``.
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** the ``TransitionOperator`` family (already an internal class hierarchy) is
      the natural module split — a genuine isolated-testing win, low numerical risk.

``mixle/stats/sequences/markov_chain.py`` (2,009)
    * **Responsibilities:** Markov chain distribution/estimator, Dirichlet prior, stationary distribution,
      gradient-fit state.
    * **Stateful globals:** ``TypeVar``\s only.
    * **Optional imports:** none.
    * **Hot paths:** forward, ``seq_update``, ``seq_log_density``, ``kernel``.
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** leave as-is; the gradient-fit state (``_MarkovChainGradientFitState``) is the
      only separable piece and is small.

``mixle/stats/latent/mixture.py`` (1,829)
    * **Responsibilities:** the general finite mixture — responsibilities, weight prior, EM.
    * **Stateful globals:** ``TypeVar``\s only.
    * **Optional imports:** none (Numba referenced through the compute layer).
    * **Hot paths:** responsibility ``logsumexp``, ``seq_update``, ``seq_log_density``, ``kernel``.
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** leave as-is; the prior-construction helpers (``mixture_prior`` and friends)
      could move to a ``_priors`` helper but that is cosmetic, not defect-driven.

``mixle/stats/latent/lda.py`` (1,721) and ``mixle/stats/latent/labeled_lda.py`` (1,608)
    * **Responsibilities:** LDA and label-supervised LDA — variational E-step, alpha updates.
    * **Stateful globals:** shape aliases only; no registry.
    * **Optional imports:** none.
    * **Hot paths:** variational fixed point, ``seq_update``, ``seq_log_density``.
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** the shared alpha-update math (``update_alpha`` exists in both) is a real
      duplication seam — a small shared helper would remove copy drift; safe, defect-adjacent.

``mixle/stats/combinator/conditional.py`` (1,570)
    * **Responsibilities:** conditional-distribution combinator (feature → response) and its accumulation.
    * **Stateful globals:** shape aliases only.
    * **Optional imports:** none.
    * **Hot paths:** ``seq_update``, ``seq_log_density``, ``kernel``.
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** leave as-is; the two module-level stat helpers are already factored.

``mixle/stats/graphs/temporal_graph_grammar.py`` (2,509)
    * **Responsibilities:** temporal graph-grammar family (37 classes: motifs, labeled/unlabeled variants).
    * **Stateful globals:** ``_EPS`` only.
    * **Optional imports:** none at import (graph backends reached lazily).
    * **Hot paths:** emission log-likelihood, ``seq_update``, ``seq_log_density``.
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** the labeled vs unlabeled variant families are separable, but they share the
      motif and alignment helpers; split only if a variant grows an independent defect surface.

Probabilistic-programming surface
---------------------------------

``mixle/ppl/inference.py`` (2,843)
    * **Responsibilities:** posterior inference for the PPL — conjugate, Laplace, VI, MCMC routes.
    * **Stateful globals:** several **registries** — ``_CONJUGATE``, ``_CONJ_LOGM``, ``_HIERARCHICAL``,
      ``_PRIOR_DICT_BUILDERS``, ``_EXPECTED_PRIOR_FAMILY`` (dispatch tables; the first thing to preserve).
    * **Optional imports:** ``torch``.
    * **Hot paths:** ``seq_log_density``, ``kernel`` (through the fitted posteriors).
    * **Serialization:** ``dist_to_encoder`` / ``seq_encode``.
    * **Extraction boundary:** the conjugate-posterior registry (tables + their builders) is a clean module
      of its own, separable from the Laplace/VI/MCMC numeric routes — a real isolated-testing win. Move the
      registries as a unit so dispatch stays in one place.

``mixle/ppl/core.py`` (2,235)
    * **Responsibilities:** the PPL authoring DSL (``Field`` / ``Group`` / ``Net`` / ``Conv`` /
      ``Transformer``) and the fitter registry.
    * **Stateful globals:** ``_ROUTE_CAVEATS``, ``_CMP``, ``_RESIDUAL`` plus the fitter registry populated by
      ``register_fitter``.
    * **Optional imports:** ``torch``.
    * **Hot paths:** ``forward``, ``seq_log_density``.
    * **Extraction boundary:** the neural predictors (``_NeuralPredictor`` / ``Net`` / ``Conv`` /
      ``Transformer``) are the torch-touching subset and a natural split from the pure-DSL field/group core.

Compiled-kernel code generation
-------------------------------

``mixle/stats/compute/fused_codegen.py`` (1,703)
    * **Responsibilities:** the fused-kernel source generator — per-family ``LeafTemplate``\s (scalar,
      vector, matrix, tabulated, categorical, chain), plan analysis (``analyze``/``fusible``), source
      emission for the one-pass scorer and E-step (sequential and chunk-parallel prange variants), the
      secure disk-cached ``_njit`` loader, and the data/parameter marshalling that feeds generated
      signatures.
    * **Stateful globals:** the ``_TEMPLATES`` registry (append-ordered; matching is first-hit, so
      registration order is part of dispatch semantics) and the ``_COMPILED`` / ``_ESTEP_COMPILED``
      kernel caches keyed by ``(plan.signature, parallel)`` — plus the on-disk module cache under
      ``MIXLE_FUSED_CACHE_DIR`` with its ownership/symlink checks.
    * **Optional imports:** ``numba`` — reached lazily inside ``_njit``; importing this module must stay
      numba-free (the freeze-rollup resolver depends on that, guarded by ``HAS_NUMBA`` at its call site).
    * **Hot paths:** every generated ``_fused`` / ``_estep`` kernel; parity is pinned by
      ``generated_kernel_parity_test.py``, ``fused_codegen_test.py``, ``fused_parallel_test.py`` (ULP /
      bit-stability receipts), and ``fused_chain_test.py`` — treat template ``row``/``acc`` fragments as
      bit-checked and change them only with those suites green.
    * **Serialization:** none — kernels are runtime artifacts; the disk cache is a per-user performance
      cache keyed by source digest, never a schema surface.
    * **Extraction boundary:** the leaf templates (registrations plus their param/table builders) are the
      clean seam — they are data, not machinery, and moving them to a ``_templates`` module would halve the
      file without touching emission. The emitter + marshalling + cache trio is a single numerical unit;
      split only against a demonstrated defect.

Infrastructure and facade
-------------------------

``mixle/stats/__init__.py`` (2,133)
    * **Responsibilities:** the ``mixle.stats`` facade — lazy ``__getattr__`` re-exports, capability
      registration, and the ``load_models`` / ``dump_models`` model-collection serialization entry points.
    * **Stateful globals:** ``_INTERNAL_SUFFIXES`` plus the lazy-export and capability tables.
    * **Optional imports:** none (that is the point — the facade must import in a bare environment).
    * **Serialization:** ``to_json`` / ``from_json`` (``load_models`` / ``dump_models``).
    * **Extraction boundary:** the builtin-compute-metadata and lazy-capability registration blocks
      (``_register_builtin_compute_metadata`` / ``_register_lazy_module_capabilities``) can move to a
      ``_registration`` helper, leaving ``__init__`` as thin re-export glue. Low risk, improves import
      readability; do it when the registration tables next change.

``mixle/utils/automatic/profiling.py`` (2,180)
    * **Responsibilities:** automatic-modeling data profiling — marginal/pairwise structure detection.
    * **Stateful globals:** tuning constants ``_MIXTURE_EM_CAP``, ``_NUMERIC_MODEL_MARGIN_BITS``,
      ``GOF_ABSTAIN_PVALUE`` (thresholds, not mutable state).
    * **Optional imports:** none.
    * **Extraction boundary:** the goodness-of-fit / entropy scoring helpers are separable from the
      profile dataclasses; a defect-neutral tidy, not urgent.

``mixle/utils/parallel/planner.py`` (1,884)
    * **Responsibilities:** device/placement/sharding planning and the encoded-data backend registry.
    * **Stateful globals:** the encoded-data backend registry (via ``register_encoded_data_backend``).
    * **Optional imports:** ``dask``, ``torch``.
    * **Serialization:** ``to_json`` / ``from_json`` (placement/plan round-trip — schema-visible).
    * **Extraction boundary:** the encoded-data backend registry and its backends are separable from the
      placement/calibration solver; a real seam, but touch it only with a distributed-path change.

``mixle/stats/compute/declarations.py`` (1,528)
    * **Responsibilities:** distribution declarations (parameter/statistic specs, exponential-family specs)
      and the Numba-lowering validation.
    * **Stateful globals:** ``_NUMBA_INFIX_OPS``, ``_NUMBA_FUNC_OPS``, ``_KNOWN_PARAMETER_CONSTRAINTS``,
      ``_SHARED_STACKED_PARAMETER_CONSTRAINTS`` (lowering tables).
    * **Optional imports:** none.
    * **Extraction boundary:** the Numba-lowering tables and their validation are separable from the spec
      dataclasses; split when the lowering rules next grow.
