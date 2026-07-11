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
- An executable routed PyTorch optimizer plus fixed-token/fixed-update batch semantics, curvature caching, and
  measured AdamW fallback based on time-to-target evidence.

## Not implemented yet

- Real multi-host transport and an 8-GPU validation run; current boundary/fault tests are deterministic in-process
  simulations, and the structured executor uses local worker threads.
- General typed execution for every compiled estimator. Compilation does not imply an execution adapter exists.
- Fused production kernels for routed Muon/Kronecker updates and real multi-host optimizer-state sharding. The
  current torch adapter uses exact SVD/eigendecomposition as a correctness reference and may be slower than AdamW.
- Context action planning, graph memory, or claims of trillion-token dense attention.

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
