Capability Lifecycle Contract
=============================

``mixle.capability_lifecycle`` is the shared, provider-neutral record used when
a capability moves between research, evaluation, supported use, operation, and
retirement.  It complements the model convenience facade in
:doc:`lifecycle` and the public-API stability registry in :doc:`maturity`; it
does not replace either one.

Why the state is factored
-------------------------

One status is not enough for a scientific AI system.  Mixle records five facts
independently:

* ``CapabilityMaturity`` records product support, from concept through retired;
* ``OperationalState`` records whether an implementation can accept work now;
* ``EvaluationState`` records the current evaluation and whether it is stale;
* ``EpistemicStanding`` records how the supporting claim is regarded; and
* ``AuthorizationDecision`` records who allowed which immutable capability
  version, for which scope, until when, and whether that permission was revoked.

A passed evaluation does not promote maturity.  Promotion is a separate,
explicit transition, and promotion to ``validated`` or ``supported`` requires
a passed evaluation at that boundary.  A later evaluation may become stale or
fail without rewriting history.  Operational availability likewise does not
create authorization, and authorization does not claim scientific validity.

Portable snapshots
------------------

``CapabilityIdentity``, ``AuthorizationDecision``, and
``CapabilityLifecycle`` are frozen value objects.  Their ``as_dict`` and
``from_dict`` methods use JSON-compatible data and preserve timezone-aware
timestamps.  Every lifecycle transition creates a new revision; illegal
backward transitions, cross-capability authorization, time reversal, and
partially retired state fail deterministically.

The existing substrate scope-governance API remains supported.  Completed
``propose`` / ``approve`` / ``reject`` records now carry decision timestamps,
and ``mixle.substrate.authorization_decision`` adapts them to the shared
authorization contract for cross-project exchange.
