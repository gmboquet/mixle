Changelog
=========

This changelog records documentation-visible release changes for the core
``mixle`` package. Distribution metadata, tags, and the coordinated family
manifest must be updated before a public release is cut.

The changelog is intentionally narrower than the full repository history. It
summarizes user-visible documentation, public API coverage, validation
expectations, and migration guidance. Use :doc:`release-notes` for the detailed
current-release narrative and the git history for exact implementation commits.

Unreleased
----------

See :doc:`release-notes` for the full release-branch summary.

Added
~~~~~

* Task and agentic distillation documentation for teacher/student workflows,
  tool-call traces, planning traces, and task-serving economics.
* DOE documentation for distillation and cross-modal training selectors.
* PPL, mixture, missing-data, stability, and automatic-inference release
  notes tied to the current hardening work.
* Release-readiness checklist for supported environments, package artifacts,
  behavior gates, documentation gates, and coordinated family evidence.
* Family release coordination guide covering package roles, version policy,
  support matrix evidence, co-install checks, publication order, and rollback
  expectations.
* Support policy for runtime floors, effective support matrix, dependency
  bounds, compatibility/deprecation, and reproducibility evidence.
* Security and data-boundary guidance for examples, artifacts, credentials,
  missing-data semantics, and decision-support claims.
* Validation guidance for wheel installs, import sweeps, numerical stress
  checks, PPL lowering routes, strict Sphinx builds, and partial evidence
  labels.
* Architecture notes covering core contracts, capabilities, workflow layers,
  extension guidance, and compatibility expectations.

Changed
~~~~~~~

* The docs tree is Sphinx/reStructuredText only; Markdown sources and Markdown
  parser configuration are no longer part of the package docs.
* Generated API pages are supported by narrative guides for core models,
  inference, PPL, DOE, data, reasoning, and production surfaces.
* The generated API reference is treated as a broad public module map rather
  than a substitute for workflow documentation.
* Public-facing docstrings have been edited to remove internal planning labels
  and to state caller-visible behavior, failure modes, and validation
  expectations directly.

Documentation Quality Bar
~~~~~~~~~~~~~~~~~~~~~~~~~

Release documentation should satisfy the following standard before publication:

* every top-level guide in the navigation explains when to use the surface and
  what validation evidence matters;
* generated API pages exist for public modules added by the release branch;
* high-level menu items use stable names such as "Release Notes" rather than a
  version-specific title;
* release-specific details live in release notes, changelog entries, or
  validation records rather than scattered across conceptual pages;
* examples use public imports and avoid private data, local paths, credentials,
  and undocumented optional services; and
* docstrings exposed through autodoc are accurate, professional, and free of
  notes-to-self or internal planning shorthand.

Release Gate
~~~~~~~~~~~~

A public release is not complete until the package version, built artifacts,
strict Sphinx build, wheel install/import sweep, tests, examples, notebooks,
and coordinated family manifest all refer to the same commit.

For documentation-only release work, the minimum local gate is a strict Sphinx
build with warnings treated as errors. That gate does not replace wheel,
example, notebook, or full-test evidence for the release itself; it only proves
that the documentation source is internally consistent.

Review Guidance
~~~~~~~~~~~~~~~

Documentation changes should stay tied to verifiable behavior. When a guide
mentions a public model, estimator, PPL construct, DOE selector, or production
artifact, the generated API reference, examples, and validation record should
name the same surface. Release notes should distinguish stable APIs from
experimental workflow guidance so users can choose an appropriate validation
standard.

Reviewers should also scan for claims that imply publication has already
happened when the branch is still a release candidate. Until tags, package
metadata, built artifacts, and family manifests are aligned, use
"release candidate", "release branch", or "pre-publication evidence" rather than
phrases that imply a published package.
