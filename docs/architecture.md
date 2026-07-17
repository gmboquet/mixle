# Mixle Core architecture

Document ID: CORE-DOC-ARCHITECTURE-001
Version scope: 0.8.x development
Owner: PRJ-CORE

## Structure

Core is organized as composable layers:

1. mixle.stats defines probability distributions, estimators, samplers,
   encoders, combinators, and latent structures.
2. mixle.inference, mixle.ppl, and mixle.enumeration fit, condition, compare,
   sample, calibrate, and traverse those models.
3. mixle.capability, mixle.contracts, and mixle.semantics expose typed
   capability and quantitative-meaning boundaries.
4. mixle.engines, mixle.data, and mixle.utils provide compute, storage,
   serialization, and parallel support without moving domain logic into the
   runtime.
5. mixle.models, mixle.task, mixle.reason, mixle.doe, mixle.evolve, and
   mixle.experimental assemble newer workflows over the shared contracts.
6. mixle.telemetry, mixle.inference.receipt, and related artifact surfaces
   carry execution and evidence metadata.

The human package map is in package-map.rst. Generated module reference is in
docs/api/.

## Data and control flow

Caller data enters an explicit encoder or data source. An estimator or model
structure selects an inference route. The route uses a compute engine and
returns a fitted object plus applicable diagnostics, uncertainty, or receipts.
Downstream operations inspect capabilities before assuming support.

Operational metadata may change placement, job identity, or artifact location;
it must not silently change the semantic identity of an unchanged scientific
problem.

## Dependency direction

Probability contracts and semantics are lower-level than applied workflows.
Optional backends are loaded at their boundary. Shared modules must not import
domain projects. Cross-project behavior uses versioned contracts rather than
copying domain implementations into Core.

## Failure modes

- Unsupported capability: reject or return an explicit unsupported result.
- Missing optional backend: preserve the original dependency error at the
  boundary.
- Invalid or non-finite scientific input: fail before fitting, ranking, or
  declaring success.
- Numerical non-convergence: retain diagnostics and the last valid state when
  the specific algorithm supports recovery.
- Partial distributed failure: preserve artifact identity and checkpoint
  metadata; never describe multi-worker work as atomic without evidence.
- Experimental behavior: require local validation and do not promote it to a
  stable contract by import location alone.

## Extension rule

Add behavior through the smallest owning contract. New engines implement
engine protocols; new distributions implement probability contracts; new
workflows compose those pieces. A cross-cutting special case in a central
dispatcher requires evidence that a normal extension point is insufficient.
