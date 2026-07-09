Validation
==========

Core Mixle validation has three layers: package importability, numerical
behavior, and documentation evidence. A release review should record each layer
from the checkout or wheel being proposed, not from an unrelated developer
environment.

The goal is not just to show that a local tree is convenient to import. The
goal is to prove that the package users will install has the same documented
surface, the same numerical behavior, and the same optional-dependency guards
as the source tree being reviewed.

Local Package Checks
--------------------

Run the package tests from a clean environment whenever practical:

.. code-block:: console

   python -m pytest

For faster branch review, record the exact subset that ran and the reason the
full suite was deferred. Focused subsets should still cover any touched model,
estimator, latent model, probabilistic-programming route, DOE selector, and
capability contract.

When a change touches generated API documentation only, the test obligation is
usually lower, but importability still matters: autodoc imports the public API
surface, so importability failures there catch invalid imports, stale
re-exports, and missing optional-dependency guards. When a docstring change
describes runtime behavior, prefer at least a focused test or example run that
exercises the behavior being documented.

For documentation-only patches, the minimum useful evidence is a strict Sphinx
build plus a content scan for stale release labels, private paths, notes to
authors, and overstated claims. If the documentation names a command, example,
or guarantee, either run the command or mark the evidence as not rerun in the
validation record.

Numerical Checks
----------------

Changed statistical code should be exercised with ordinary inputs, degenerate
inputs, and missing-data inputs. In particular, release evidence should show:

* mixture and HMM responsibilities remain finite when observations are
  impossible, singular, high-dimensional, or heavily weighted;
* ``NaN`` observations preserve the package's missing-data contract rather
  than being silently rewritten;
* ``-inf`` likelihoods represent impossible observations without poisoning
  unrelated batches;
* estimator initialization reports clear errors for empty or unsupported data;
* stochastic tests use fixed seeds or tolerance bands that make failures
  actionable; and
* optional backend paths stay guarded when their dependency is absent.

For mixture models and latent models, validation should include both scoring
and fitting paths. A model that scores impossible observations as ``-inf`` can
still fail during EM if responsibilities are normalized naively. Conversely, a
fit loop can remain finite while the resulting model exposes invalid
posterior, sampler, or prediction behavior. Release evidence should therefore
exercise the route users actually call: estimator construction, optimization,
scoring, sampling when supported, and any posterior or diagnostic helper named
by the docs.

For probabilistic-programming routes, record the lowering path as well as the
fitter. A PPL example should identify whether it lowers to a distribution,
estimator, state-space model, custom potential, or sampler route, and whether
missing data are rejected, marginalized, or handled by the selected inference
method.

Import and Artifact Checks
--------------------------

Before publication, build and inspect the package artifact:

.. code-block:: console

   python -m build
   twine check dist/*

Then install the built wheel in a fresh environment and run an import sweep over
public modules with optional-dependency guards. This catches missing package
data, accidental source-tree imports, and unguarded heavy dependencies.

The import sweep should run from outside the repository checkout so Python
cannot accidentally resolve modules from the source tree. If optional
dependencies are intentionally absent, record which imports are expected to be
mocked, skipped, or feature-gated. A clean import failure is acceptable for a
documented optional feature; an import-time failure from a core namespace is a
release blocker.

Documentation Checks
--------------------

The Sphinx build is a release gate:

.. code-block:: console

   make -C docs html SPHINXOPTS="-W --keep-going"

Generated API ``.rst`` sources are part of the documentation source, not local
build output. When modules are added, removed, or renamed, regenerate and commit
the affected API pages, then repeat the strict build from a clean archive or
clean checkout.

Documentation evidence should also include a quick content review. Confirm that
top-level menu items are stable and generic, release-specific details live under
release notes or changelog pages, and narrative guides explain the concepts
that generated API pages cannot. Generated reference pages should not be used as
the only explanation for workflows such as automatic modeling, PPL fitting,
task distillation, DOE, missing data, or production artifacts.

Docstring Review
----------------

Autodoc makes public docstrings part of the release documentation. Review
docstrings for:

* accurate observation shapes, parameter semantics, and return contracts;
* optional dependency and backend requirements;
* explicit behavior for missing data and non-finite observations;
* clear failure modes for unsupported routes;
* professional language without notes to authors, local paths, roadmap labels,
  or private project shorthand.

Comments may explain implementation choices, but public docstrings should read
as user-facing API documentation. If a docstring describes behavior outside the
current tests or examples, treat that as a validation gap.

Evidence Record
---------------

Keep a short validation record with the branch, commit, Python version,
dependency source, commands, pass/fail status, skipped gates, and artifact
paths. If a gate is skipped because it is too expensive, requires private data,
or needs unavailable hardware, mark it as skipped or blocked rather than
passing it by implication.

Use explicit language for partial evidence:

``passed``
    The command ran against the release candidate artifact or checkout and the
    output is available.

``skipped``
    The command was intentionally not run, with a reason and an owner for
    follow-up.

``blocked``
    The command could not run because of missing credentials, unavailable
    hardware, external service failures, or another condition outside the
    package tree.

``not applicable``
    The gate does not apply to the touched surface, for example GPU-only checks
    on a documentation-only patch.
