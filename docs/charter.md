# Mixle Core charter

Document ID: CORE-DOC-CHARTER-001
Version scope: 0.8.x development
Owner: PRJ-CORE
Status: active

## Mission

Mixle Core supplies small, composable primitives for probabilistic modeling,
provider-neutral execution, uncertainty-aware inference, evaluation, and
evidence-carrying results. It is the shared software substrate used by the
other Mixle projects, not the owner of their domain decisions.

## Users

- Library users building heterogeneous probability and inference systems.
- Mixle projects consuming shared artifacts, protocols, execution hooks, and
  quantitative semantics.
- Maintainers validating numerical behavior, compatibility, and release
  evidence.

## Scope

Core owns distribution and estimator contracts, inference and calibration
primitives, capability inspection, common execution and backend abstractions,
portable quantitative semantics, task/model utilities that remain
domain-neutral, and the documentation needed to use those surfaces honestly.

## Non-goals

Core does not own domain-specific physics, PDE formulation, data acquisition,
fleet deployment, portfolio orchestration, or claims that every optional
backend has equal maturity. Those responsibilities stay in their project
repositories and integrate through versioned contracts.

## Ownership and decisions

PRJ-CORE is accountable for public Core behavior. Cross-project interfaces
must have one named owner and an explicit compatibility range. Material
architectural changes require a recorded decision and migration path in the
Mixle status repository.

## Current status

Version 0.8.0 is an active development target, not a published-release claim.
The maturity guide, changelog, source tests, and dated status evidence define
what is observed or verified. Forward-looking plans never override those
sources.
