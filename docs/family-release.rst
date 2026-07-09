Family Release Coordination
===========================

Mixle is a package family, not a single wheel. Public documentation should
make that clear: a release claim is only trustworthy when the core package,
sister packages, clients, notebooks, demos, and release notes agree on what is
being shipped and what evidence proves it.

Use this page as the public-facing summary of the coordinated release process.
The detailed release records should live in the repository's release evidence
area and must be completed before any final publication claim.

Package Roles
-------------

Every repository in the family needs one explicit release role:

``published Python package``
    A package distributed through a Python package index. It needs final version
    metadata, build artifacts, metadata validation, fresh-install evidence,
    tests, dependency-resolution evidence, and tag/artifact hashes.

``private Node workspace``
    A source or private-registry workspace such as ``mixle-agent``. It needs
    Node/npm version evidence, install/build/typecheck/test output, and a clear
    distribution target.

``app artifact``
    A native client such as ``mixle-ios``. It needs app version/build evidence,
    SDK/simulator details, bundled-data checks, and distribution-target notes.

``notebook bundle``
    A worked-example or educational bundle. It needs a notebook inventory,
    execution status for every linked notebook, skip/block reasons, and data
    provenance review.

``specification package``
    A repository that publishes design or readiness documentation but no
    runtime artifact. It must not advertise package-index install commands or
    runtime behavior.

``excluded``
    A repository intentionally out of scope for the release. Public pages and
    manifests should not imply it participates.

Version Policy
--------------

Before final publication, the release owner should choose and record one
version policy:

* lockstep versions, where all runtime packages publish the same release
  version;
* independent package versions validated as one family release;
* mixed public/private artifacts; or
* a deferred sibling release where some repositories are excluded.

The current documentation describes pre-publication release work, not
publication evidence. Final package metadata, changelogs, tags, package-index
links, and website labels should not claim more than the manifest proves.

Support and Reproducibility
---------------------------

The support policy should record the effective matrix actually tested, not only
the broad metadata floor. At minimum, public release evidence should identify:

* Python versions and operating systems for each Python package;
* Node and package-manager versions for TypeScript workspaces;
* Xcode, iOS SDK, and simulator/device availability for mobile artifacts;
* CPU/GPU status for optional Torch or accelerator workflows;
* resolver output or lockfiles for supported Python versions; and
* old-version smoke checks where dependency drift could break reproducibility.

If an optional dependency narrows the effective matrix, document that in the
release notes instead of inferring support from ``requires-python`` alone.
See :doc:`support-policy` for the public compatibility, dependency-bound, and
reproducibility policy.

Family Co-Install Gate
----------------------

Passing package tests in isolation is necessary but not sufficient. For every
package classified as a published Python package, the final release should
install the chosen versions together in a fresh environment and then run small
cross-package smoke checks.

The co-install evidence should prove:

* ``mixle-mlops`` imports and serves against the final core artifact;
* ``mixle-pde`` imports and runs as a plugin against the final core artifact;
* ``mixle-demos`` validates without relying on developer-local sibling checkout
  state unless that checkout is explicitly documented;
* ``mixle-knowledge`` validates contracts consumed by sibling packages; and
* no package depends on an unpublished sibling version.

Publication Order
-----------------

Publication should happen only after validation gates pass, and it should
happen in dependency order:

1. Freeze roles, versions, branches, uncommitted-worktree disposition, and manifest
   state.
2. Build and validate artifacts from clean checkouts.
3. Dry-run publication to the chosen test index and verify installability.
4. Publish to the final package index or distribution target in dependency
   order.
5. Tag the exact commits that produced the published artifacts.
6. Post-publish, install from the public target and repeat smoke checks.
7. Publish final documentation and record website URLs in the manifest.

Rollback and Incident Notes
---------------------------

Published package versions are immutable. If a faulty release reaches a package
index, fix it with a new patch version and record the incident. Do not move
tags, reuse versions, or leave the family half-published without a compensating
plan.

Documentation Responsibility
----------------------------

The package docs should remain precise at every stage:

* call in-progress work pre-publication release work, not a published release;
* separate tracked release evidence from developer-local evidence;
* state when a sibling is specification-only, runtime-bearing, or excluded;
* link public claims to manifest evidence once available; and
* keep changelogs synchronized with final artifact roles and versions.
