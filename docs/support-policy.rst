Support Policy
==============

This page states what Mixle documentation may claim about supported runtimes,
dependencies, compatibility, and reproducibility. It is intentionally
evidence-oriented: a version or platform is supported only when the final
release manifest records validation for it.

Current Status
--------------

The current release docs describe pre-publication work. They are not proof that
a public artifact has been published. In particular, the core package metadata
must be aligned with the release candidate before publication.

Declared Runtime Floors
-----------------------

The package family currently declares these runtime floors:

.. list-table::
   :header-rows: 1

   * - Surface
     - Runtime floor
     - Release evidence required
   * - Core ``mixle``
     - Python 3.10 and newer
     - Wheel install, import sweep, tests, examples, and docs builds on the
       effective Python/OS matrix.
   * - Python sister packages
     - Usually Python 3.10 and newer; ``mixle-demos`` currently declares
       Python 3.11 and newer
     - Package-specific build, clean install, tests, docs, and family
       co-install evidence.
   * - ``mixle-agent``
     - Node 20 and newer
     - ``npm install``, build, typecheck, test, and runtime smoke evidence.
   * - ``mixle-ios``
     - Xcode and iOS SDK, with simulator or device availability recorded
     - App build, bundle, asset, decoding, and manual smoke evidence.
   * - ``mixle-notebooks``
     - Notebook environment rather than importable package metadata
     - Kernel, Python, sibling package versions, notebook execution status,
       and data provenance.

Effective Matrix
----------------

The effective matrix is the intersection of package metadata, dependency
support, and the surfaces being claimed:

* base ``mixle`` should import without optional heavy dependencies;
* optional Torch, JAX, Spark, Dask, MPI, symbolic, data, and GPU paths should
  be claimed only when their dependency stacks are installed and tested;
* CPU fallback status should be stated for GPU-oriented workflows;
* Windows should not be listed as supported unless it is actually tested; and
* the final manifest should record exact Python, OS, Node, Xcode, SDK, and
  package-manager versions used during release validation.

Dependency Bound Policy
-----------------------

A public release should be reproducible after publication. Dependency ranges
therefore need enough bounds and evidence to keep old versions installable:

``bounded``
    Has lower bounds and an upper bound where future major releases could break
    behavior. Record the supported range and resolver evidence.

``lower-bound-only``
    Acceptable only when the release owner records why future drift is safe, or
    when resolver evidence is captured for the release and the risk is tracked.

``bare``
    A release-review finding. Add bounds or record an explicit exception.

``git-url`` or local source
    Avoid in public release dependencies unless pinned and documented. Prefer
    published package versions for final release evidence.

``optional``
    Must be truly optional: base imports work without it, and feature use gives
    a clear install hint when the dependency is absent.

Compatibility and Deprecation
-----------------------------

Mixle should prefer additive compatibility over breaking moves. When a public
surface changes:

* keep compatibility shims or re-exports when practical;
* document migration notes for removed, renamed, or behavior-changing APIs;
* include tests for old payloads or old import paths when compatibility is
  retained;
* bump the appropriate package version before publication; and
* make changelog wording match the actual package role and version policy.

Schema-owning packages such as ``mixle-knowledge`` need extra care: removing or
renaming fields requires migration notes, validation fixtures, and consuming
manifest checks.

Reproducibility Evidence
------------------------

The final release manifest should record:

* exact package versions and commit SHAs;
* built artifact filenames and hashes;
* resolver output or lockfiles for each supported Python where the graph
  differs;
* clean-install commands and results;
* import-sweep and test commands;
* notebook and example execution status;
* family co-install evidence; and
* publication, tag, and documentation website URLs.

Old-Version Policy
------------------

Published versions and tags are immutable. Do not reuse package versions, move
tags, or re-upload artifacts. If an old exact pin stops installing because an
unbounded dependency drifted, treat that as a reproducibility defect and fix
the current release bounds or document the incident.

Documentation Rules
-------------------

Public documentation should use release-branch language until final evidence
exists. Avoid implying that:

* a package is published when it is only documented;
* a notebook is healthy when it has not executed;
* a platform is supported because metadata allows it but tests did not run; or
* a family release is coherent before co-install and manifest evidence exists.

See :doc:`family-release` for the cross-package release process and
:doc:`release-readiness` for the core package checklist.

Evidence Over Metadata
----------------------

Package metadata can declare compatibility, but release documentation should
claim support only from evidence. If ``requires-python`` allows an interpreter
that was not tested with the effective dependency graph, describe it as allowed
by metadata, not validated by the release. The same rule applies to optional
extras, GPU paths, distributed backends, and notebook environments.

When evidence is missing, use explicit release states such as ``skipped``,
``blocked``, or ``needs rerun``. Ambiguous phrases make support hard to audit
after publication.
