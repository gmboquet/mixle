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

Core 0.8.0 declares this runtime floor. Related projects are independently
versioned and excluded from this release's support claim:

.. list-table::
   :header-rows: 1

   * - Surface
     - Runtime floor
     - Release evidence required
   * - Core ``mixle``
     - Python 3.11 and 3.12
     - Wheel install, import sweep, tests, examples, and docs builds on the
       effective Python/OS matrix.
   * - Related projects
     - Independently declared
     - Explicitly excluded from Core 0.8.0 compatibility and co-install claims.

Effective Matrix
----------------

The effective matrix is the intersection of package metadata, dependency
support, and the surfaces being claimed:

* base ``mixle`` stable-core behavior is validated on Ubuntu and macOS arm64
  under Python 3.11 and 3.12;
* the exhaustive non-optional suite and clean wheel/sdist artifact lanes use
  Ubuntu/Python 3.12 as the representative full cell; platform cells add
  stable-core and platform-specific contracts rather than duplicating every
  platform-independent stochastic test;
* optional Torch, JAX, Spark, Dask, MPI, symbolic, data, and GPU paths should
  be claimed only when their dependency stacks are installed and tested. For
  0.8.0 the extras resolver and optional behavior baseline is Linux
  x86_64/Python 3.12 unless a backend row records broader evidence;
* CPU fallback status should be stated for GPU-oriented workflows;
* Windows should not be listed as supported unless it is actually tested; and
* the final manifest should record exact Python, OS, Node, Xcode, SDK, and
  package-manager versions used during release validation.

An optional extra resolving on another platform or Python version is allowed
by metadata, not a validated support claim. This distinction preserves the
base Python/macOS support statement without pretending every accelerator,
cluster, and native stack was exercised in every cell.

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

API maturity tiers
~~~~~~~~~~~~~~~~~~~

Every public name sits in one of three tiers (see :doc:`maturity`), which set how
much stability it promises:

* **stable** -- covered by this policy; changes follow the deprecation lifecycle
  below.
* **provisional** -- usable and tested, but the signature or defaults may still
  change within a minor release; changes are called out in the changelog.
* **experimental** -- everything under ``mixle.experimental``; no compatibility
  guarantee, and it must not be imported from stable modules.

Deprecation lifecycle
~~~~~~~~~~~~~~~~~~~~~~~

When a **stable** name is renamed or retired, the old spelling is *deprecated*
rather than deleted:

* it keeps working and forwards to the replacement, unchanged in behavior;
* it emits a ``DeprecationWarning`` (never a bare ``UserWarning`` or a print),
  attributed to the caller's line, in one message format::

      <old> is deprecated since mixle <since>; use <new> instead. It will be removed in mixle <removed_in>.

* it is kept for at least **2 minor releases** after the release that announces
  the deprecation (announced in ``0.8.0`` → removable no earlier than ``0.10.0``);
  and
* its removal ships with a migration note and, for a renamed API, a runnable
  before/after example.

Deprecations are wired through one helper, :func:`mixle.utils.deprecation.deprecated_alias`
(or :func:`~mixle.utils.deprecation.warn_deprecated` for finer-grained cases), so
the category and message format are uniform. A test
(``mixle/tests/deprecation_test.py``) statically enforces that every "Deprecated
alias" callable actually carries the decorator -- a deprecated name that forgets
to warn is itself a defect.

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
* family co-install evidence only for a future coordinated family release; and
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
claim support only from evidence. For 0.8.0, ``requires-python`` is capped below
3.13 so installer compatibility and the tested 3.11/3.12 matrix agree. If a later
``requires-python`` range allows an interpreter
that was not tested with the effective dependency graph, describe it as allowed
by metadata, not validated by the release. The same rule applies to optional
extras, GPU paths, distributed backends, and notebook environments.

When evidence is missing, use explicit release states such as ``skipped``,
``blocked``, or ``needs rerun``. Ambiguous phrases make support hard to audit
after publication.
