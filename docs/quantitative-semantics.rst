Quantitative semantics
======================

``mixle.semantics`` defines the small domain-neutral contract shared by
probabilistic consumers. A ``ValueSpec`` states identity, role, unit, shape,
transform, constraints, prior and derivation without naming a solver or job.
``ObservationSpec`` binds immutable data content to a likelihood and optional
measurement uncertainty. Posterior, predictive, calibration, decision and
uncertainty artifacts retain those links.

Semantic identity uses canonical JSON and excludes fields explicitly marked as
operational, currently sample location, backend identifier and job identifier.
Moving identical content between NumPy, JAX or a remote job therefore preserves
identity; changing a prior, unit, transform, constraint, observation, seed,
likelihood, method, summary, diagnostic or uncertainty component does not.

The packaged ``fixtures/quantitative-semantics-v1.json`` is the cross-project
contract for Inquiry and inverse-physics readers. It declares one positive
latent source rate, log transform, bounded constraint, lognormal prior,
content-addressed observation, discrepancy-aware likelihood, fixed random seed
and required uncertainty classes. Consumer repositories should validate the
fixture without importing their orchestration into Core.

Transforms expose forward, inverse and log-absolute-Jacobian operations and
fail outside their mathematical domains. Invalid role/prior/value combinations,
unit omissions, dangling posterior references and incomplete uncertainty
payloads fail at construction.

Boundary
--------

Core owns these meanings and quantitative algorithms. It does not decide which
question to ask, store claims, acquire data, choose physical laws, discretize a
domain, schedule a job, or deploy a model. ``CapabilityExtension`` and
``TraceSink`` are deliberately structural protocols so consumers need no
private imports or reverse dependencies.
