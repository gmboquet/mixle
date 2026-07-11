# Statistically typed runtime

This experimental package turns a mixle model into an inspectable update graph before work is executed. It keeps
statistical meaning, update mechanics, merge algebra, mutable state, consistency, curvature, decomposition, cost,
and cache invalidation separate rather than treating every parameter as an undifferentiated gradient tensor.

```python
from mixle.experimental.typed_runtime import compile_update_graph

graph = compile_update_graph(model, estimator, nobs=len(data))
print(graph.explain())
```

## Implemented

- Side-effect-free compilation of the public distribution fixture catalog.
- Explicit MLE, MAP, ELBO, contrastive, preference, constraint, and surrogate objective types.
- Exact, generalized-EM, first-order, proximal, message, Monte Carlo, search, and frozen update types.
- Dependency closures and versioned local artifact invalidation.
- Measured/proxy costs, effective-context measurements, time-to-target traces, and failure ledgers.
- Deterministic lower-confidence-bound gain-per-cost scheduling with objective compatibility and bounded starvation.
- An objective-gated local mixture-EM adapter whose component blocks are selected by typed node IDs.
- Immutable proposal packets with deterministic non-pickle payload fingerprints and exact additive shard merging.
- Versioned transactional commit with dependency-conflict checks, canaries, verified mutable-state rollback, and
  poisoned-coordinator handling when rollback cannot be proven.
- Independent step/token/observation update clocks with hard staleness bounds.
- Bitwise replay and explicit-tolerance replay with semantic receipt and optional numeric-state comparisons.
- Measured host/island/provider topology, link calibration, and exact structured-shard placement inside fast islands.
- Versioned, ordered, duplicate-safe shard-boundary messages with approximation and error disclosure.
- Contract-gated low-rank/top-k communication compression with checkpointable error-feedback residuals.
- Strict, bounded-stale, and corrected-eventual proposal admission plus deterministic fault injection.
- A hierarchical island coordinator and a real typed adapter over mixle's exact model-parallel statistic fold.
- Per-parameter geometry routing for exact statistics, AdamW, Muon-style orthogonalization, Kronecker
  preconditioning, natural gradients, proximal blocks, low-rank adapters, and sparse-expert clocks.
- An executable routed PyTorch optimizer with native AdamW passthrough for plans whose geometry does not justify
  custom kernels, plus fixed-token/fixed-update batch semantics, curvature caching, and measured fallback based on
  time-to-target evidence.
- A provenance-preserving context-action IR for retrieval, source expansion, hypothesis/query generation,
  summarization, verification, tools, linking, pruning, materialization, and explicit stopping.
- Closed-loop value-of-information context construction with transactional graph mutation and actual-cost budgets.
- Bounded graph partition/prefetch/LRU memory and exact-near plus retrieved-far attention whose active tokens do not
  grow with source horizon.
- Provenance-complete materialization that excludes contradicted evidence and unverified generated claims.
- An integrated graph-memory mixture-of-experts pilot with local-window and AdamW ablations, real stochastic
  microbatches and gradient accumulation, bounded active context, explicit negative controls, and bitwise
  model/optimizer/sampler-RNG restart parity.
- Machine-readable claim gates that remain closed until externally measured 8-GPU transport, multi-host recovery,
  1B-parameter quality/efficiency, trillion-token source-horizon, bounded-active-context, and provenance receipts exist.

## Not implemented yet

- Real multi-host transport and an 8-GPU validation run; current boundary/fault tests are deterministic in-process
  simulations, and the structured executor uses local worker threads.
- General typed execution for every compiled estimator. Compilation does not imply an execution adapter exists.
- Fused production kernels for routed Muon/Kronecker updates and real multi-host optimizer-state sharding. The
  current torch adapter uses fixed-step Newton-Schulz Muon updates (with exact SVD available as an audit backend),
  while Kronecker inverse roots still use eigendecomposition and may be slower than AdamW.
- Persistent external graph/vector-store adapters and production retrieval indexes. Trillion-scale **source horizons**
  are represented and tested; trillion-token dense attention is neither implemented nor claimed.
- A real frontier-scale training result. The integrated pilot is a deterministic synthetic falsification fixture, not
  evidence that the reference runtime can train a competitive frontier model.

The local mixture adapter deliberately rejects shared component objects and conjugate weight priors until their joint
proposal semantics are implemented. Unsupported semantics fail before execution rather than silently changing the
model family or objective.

## First executable path

```python
from mixle.experimental.typed_runtime import run_typed_mixture_em

run = run_typed_mixture_em(encoded_data, estimator, initial_model, max_its=50)
print(run.objective_trace)
print(run.total_model_evaluations)
```

Each round records the scheduler decision, actual observed-data objective before and after, acceptance or rejection,
cache invalidation closure, component evaluations, operations, and elapsed time. Internal gain evidence is the
committed global-objective gain and is labelled `joint_with_coordinator`; it is not presented as an isolated causal
contribution from one component.
