Release Notes
=============

The 0.7.0 release is a capability and hardening release. It expands Mixle's
task, DOE, neural, latent-model, and reasoning surfaces while tightening
numerical behavior in mixture, automatic-inference, and registry paths.

Highlights
----------

This release focuses on three themes:

* stronger numerical behavior for mixture, EM, HMM, automatic-inference, and
  missing-data paths;
* broader task, distillation, DOE, neural, and reasoning APIs; and
* clearer documentation for maturity, validation, production use, and family
  release expectations.

Added
-----

Task and distillation capabilities
    The task layer now includes richer teacher/student workflows: soft-label
    distillation, structured task distillation, active labeling, cascade
    economics, local harvest/retrain loops, and agentic task distillation for
    tool calls and plans.

DOE for distillation and cross-modal training
    ``mixle.doe.distillation`` adds pool-based experiment design for task
    distillation and cross-modal training. It helps choose informative teacher
    calls, balance task coverage, and treat label acquisition as an expensive
    experimental design problem. See :doc:`doe` for the Sphinx examples and
    selector contract.

Neural and energy models
    The branch adds reusable neural-model builders including Deep Sets,
    monotonic MLPs, input-convex networks, Hamiltonian networks, and
    energy-based product-of-experts helpers.

Latent and dependence models
    New or expanded latent surfaces include gated mixtures, copulas, structured
    mixture/reduction utilities, and additional mixture-of-experts style
    building blocks.

Reasoning and evidence surfaces
    The reasoning package adds cross-modal transport checks, task-sufficient
    projections, cycle-consistency signals, anchor harnesses, answer receipts,
    and provenance-aware explanation helpers.

Changed
-------

Automatic modeling is more defensive
    Empty data, all-empty nested sequences, detector failure paths, marginal
    field validation, and model-recommendation fallbacks now fail more
    explicitly.

Mixture and EM paths are harder to destabilize
    Mixture code has additional stress coverage for high-dimensional Gaussian
    mixtures, singular or near-singular covariance paths, weighted
    responsibilities, and impossible-observation updates. See
    :doc:`stability-and-missing-data` for the branch-level contract around
    ``NaN`` inputs, ``-inf`` impossible observations, robust mixture
    initialization, and DOE score validation.

PPL route behavior is more explicit
    The PPL guide now documents ``explain_fit``, explicit missing-data
    marginalization, composite custom potentials, state-space fitted
    distributions, and indexed latent sampler routes. Unsupported
    route/feature combinations are expected to raise clear errors instead of
    returning partially applied models.

Task documentation is broader
    The task guides now cover one-call replacement patterns, calibrated
    structured outputs, density gates, serving cascades, economics, extraction,
    and agentic/task-planning variants.

Fixed
-----

The branch includes hardening commits for foundational EM, precision,
registry, automatic-inference, oracle timeout, stochastic test, safetensors,
and mixture-stability issues. See the repository history for the exact patch
commits used for the final tag.

Compatibility And Migration
---------------------------

No broad removal is documented for this release. Users upgrading from the
previous release should pay closest attention to expanded task APIs, DOE
distillation helpers, reasoning receipts, and the clarified missing-data and
impossible-observation contracts.

Validation Focus
----------------

The release should be validated through the gates in :doc:`release-readiness`.
At minimum, release evidence should cover:

* ``python -m build`` and ``twine check dist/*``;
* install from the built wheel in a fresh virtual environment;
* import sweep over public ``mixle`` modules with optional-dependency guards;
* full test suite, not only the fast marker subset;
* examples and notebooks that are shipped or linked by the docs;
* strict Sphinx build with warnings as errors; and
* the coordinated family resolver/integration check across sibling packages.
