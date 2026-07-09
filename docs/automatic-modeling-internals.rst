Automatic Modeling Internals
============================

Automatic modeling in Mixle is not a single opaque estimator. It is a set of
profilers, factory functions, scoring heuristics, validation checks, and
recommendation reports that turn heterogeneous Python data into an explicit
estimator tree.

The public workflow is documented in :doc:`automatic-inference`. This page
documents the machinery behind that workflow so users can understand what was
chosen and extension authors can improve it deliberately.

Entry Points
------------

The low-level automatic-modeling functions live in ``mixle.utils.automatic``:

``get_estimator(data, ...)``
    Infer a first estimator from a sequence of observations.

``get_prototype(data, ...)``
    Build a prototype distribution when the downstream route wants a model
    shape rather than an estimator.

``analyze_structure(data, ...)``
    Return a ``StructureProfile`` with field profiles, pairwise dependency
    hints, warnings, and an assembled estimator.

``get_dpm_mixture(data, ...)``
    Build a Dirichlet-process mixture path over automatically typed data.

The task-layer wrapper lives in ``mixle.task``:

``recommend_model(data, ...)``
    Turn the structure profile into a user-facing recommendation object with
    field choices, confidence gaps, dependencies, warnings, and fit helpers.

Use ``analyze_structure`` when you want to inspect the automatic choice. Use
``get_estimator`` when you want a quick baseline. Use ``recommend_model`` when
the result must be explained to a human or stored in a report.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.task import recommend_model
   from mixle.utils.automatic import analyze_structure, get_estimator

   profile = analyze_structure(rows, pairwise=True, validate_marginals=True)
   estimator = profile.recommend()
   model = optimize(rows, estimator, max_its=50, out=None)

   baseline = get_estimator(rows)
   recommendation = recommend_model(rows, pairwise=True)
   low_confidence = recommendation.low_confidence_fields()
   explanation = recommendation.explain()

Factory Functions
-----------------

``mixle.utils.automatic.factories`` contains explicit builders for each
automatically chosen shape:

.. list-table::
   :header-rows: 1

   * - Builder
     - Purpose
   * - ``get_optional_estimator``
     - Wrap a child estimator with missing-value behavior.
   * - ``get_length_estimator``
     - Choose an integer categorical or Poisson length model.
   * - ``get_sequence_estimator``
     - Build a sequence estimator with optional length model.
   * - ``get_set_estimator``
     - Build a Bernoulli set model.
   * - ``get_ignored_estimator``
     - Ignore identifier-like or unsupported fields.
   * - ``get_composite_estimator``
     - Build a positional composite.
   * - ``get_dict_record_estimator``
     - Build a named-record estimator.
   * - ``get_categorical_estimator``
     - Build a categorical estimator for strings and discrete values.
   * - ``get_integer_categorical_estimator``
     - Build a bounded integer categorical estimator.
   * - ``get_poisson_estimator``
     - Build a count estimator.
   * - ``get_gaussian_estimator``
     - Build a Gaussian estimator.
   * - ``get_lognormal_estimator``
     - Build a log-normal estimator for positive skewed values.
   * - ``get_gamma_estimator``
     - Build a Gamma estimator for positive continuous values.
   * - ``get_student_t_estimator``
     - Build a Student-t estimator for heavy-tailed continuous values.
   * - ``get_gaussian_mixture_estimator``
     - Build a small Gaussian mixture candidate.
   * - ``get_multivariate_gaussian_estimator``
     - Build a multivariate Gaussian estimator for vector-like fields.

The factory layer is intentionally plain. When automatic modeling chooses a
family, it calls the same builder a user could call directly.

Structure Profiles
------------------

``mixle.utils.automatic.profiling`` exposes report objects:

``MarginalFieldProfile``
    Per-field evidence: path, role, missingness, observed kind, recommendation,
    entropy, cardinality, numeric summaries, BIC-style model scores, model
    weights, validation scores, goodness-of-fit statistics, and notes.

``PairwiseDependencyHint``
    Unconditional pairwise dependency evidence: mutual information, adjusted
    mutual information, BIC gain, normalized mutual information, sample count,
    method, optional p-value, and notes.

``StructureProfile``
    Full result: estimator, field profiles, pairwise hints, dependency tree,
    residual dependency edges, warnings, sampled-row counts, and explanation
    helpers.

These objects are part of the audit trail. They let automatic modeling explain
where it was confident, where it was ambiguous, and which dependencies look
worth modeling jointly.

Persist the profile when automatic modeling influences a production model. The
profile is the evidence behind the estimator tree, including low-confidence
fields, ignored fields, warnings, and dependency hints that may be hidden by
the fitted parameters alone.

Audit Fields
------------

For each automatic run, keep enough profile metadata to reconstruct why a model
shape was chosen:

* field path and observed kind;
* missing-value rate and unsupported-value notes;
* recommended family and runner-up family;
* score gap between the best candidate and the runner-up;
* validation score or goodness-of-fit warning when available;
* dependency hints that changed the recommended structure; and
* warnings for ignored, identifier-like, sparse, or low-sample fields.

The estimator tree is the executable artifact. The profile is the audit record
that explains the estimator tree.

Scoring Logic
-------------

For scalar fields, automatic profiling compares candidate families with
penalized likelihood and validation checks. Numeric candidates include:

* categorical and integer categorical models for small or dense discrete
  supports;
* Poisson for count-like nonnegative integers;
* Gaussian for ordinary continuous values;
* log-normal and Gamma for positive skewed values;
* Student-t for heavy-tailed continuous values;
* small Gaussian mixtures when multimodality is plausible;
* additional detector families such as Beta, Weibull, Gumbel, Laplace,
  logistic, Pareto, generalized Pareto, generalized extreme value, skew normal,
  inverse Gaussian, Tweedie, ex-Gaussian, negative binomial, and generalized
  Gaussian where the detector modules are available.

The profile records bit-scale scores and a gap between the winner and runner-up.
Small gaps are important: they mean the data do not strongly distinguish the
families. In that case, a user should either collect more data, use domain
knowledge, or keep the choice explicit in a model card.

Structured Data
---------------

Automatic modeling recursively handles heterogeneous shapes:

.. list-table::
   :header-rows: 1

   * - Data shape
     - Typical automatic model
   * - Missing values
     - ``OptionalEstimator`` around the inferred child.
   * - Tuples/lists with fixed roles
     - ``CompositeEstimator``.
   * - Variable-length sequences
     - ``SequenceEstimator`` plus a length model.
   * - Sets
     - Bernoulli set estimator.
   * - Dictionaries
     - Named record estimator over discovered keys.
   * - Identifier-like fields
     - Ignored estimator, with a warning or note.
   * - Numeric vectors
     - Multivariate Gaussian candidate when shape and sample size support it.

For production data, treat ignored fields and dependency hints as review items.
They are often where identifiers, leakage, or meaningful structure enter the
system.

Bayesian Mode
-------------

Most factory functions accept ``use_bstats=True``. This keeps the same
automatically inferred shape but attaches default conjugate priors where the
family supports them. The result follows the Bayesian path through the same
estimator contracts, using closed-form conjugate or MAP updates where
available.

Default priors are deliberately conservative and generic. They are useful for
small samples and smoothing, but domain priors should be specified explicitly
when they matter.

Missing and Non-Finite Values
-----------------------------

Automatic modeling should not silently convert data quality issues into model
assumptions. Missingness appears in field profiles and, where supported,
factory functions choose optional or marginalizing wrappers. Non-finite numeric
values are not ordinary observations; they should either be rejected by the
chosen family or handled through an explicit missing-data contract.

When ``NaN`` carries semantic meaning in the upstream data, preserve that
meaning outside automatic modeling or define a visible field transformation.
Do not rely on the automatic path to impute, coerce, or erase it.

The automatic path owns the estimator choice, not the caller's data buffer. If
the caller passes a list, array, or record object containing ``NaN`` or
``inf``, automatic modeling should either route the value through an explicit
missing/non-finite contract or surface a warning or rejection. It should not
rewrite the original object as a side effect of type detection.

Dependency Hints
----------------

Pairwise dependency hints are modeling evidence, not causal claims. A high BIC
gain or mutual-information estimate says that two observed fields may be better
modeled jointly than independently. It does not say which field causes the
other, whether a hidden confounder is present, or whether the relationship will
remain stable under intervention.

Use dependency hints to decide whether to:

* replace independent leaves with a joint family;
* add a latent factor or mixture;
* move from a record model to a graphical or conditional model;
* collect targeted data for ambiguous fields.

Recommended Workflow
--------------------

For exploratory modeling:

1. Run ``analyze_structure`` or ``recommend_model``.
2. Read the field explanations and warnings.
3. Inspect low-confidence fields and pairwise hints.
4. Fit the recommended estimator and an independence baseline.
5. Compare held-out log-density or task-specific utility.
6. Freeze the chosen estimator tree when the model becomes production-facing.

For production modeling, do not leave the important choice hidden in automatic
typing. Persist the estimator or a model specification, store profile summaries,
and record why ambiguous fields were accepted or overridden.

What It Does Not Decide
-----------------------

Automatic modeling does not decide whether the dataset is exchangeable, whether
an identifier is safe to use, whether a dependency is causal, or whether a
metric is acceptable for the application. Those decisions belong in the model
card or artifact review. The automatic report supplies evidence; the caller
owns the final modeling judgment.

Failure Modes
-------------

.. list-table::
   :header-rows: 1

   * - Symptom
     - Response
   * - Identifier field is modeled as categorical
     - Mark it ignored or remove it before fitting.
   * - Winner and runner-up have a small score gap
     - Use domain knowledge or collect more data.
   * - Positive skewed data is split between Gamma and log-normal
     - Compare held-out likelihood and tail behavior.
   * - Count data is overdispersed
     - Consider negative binomial, mixture, or latent structure.
   * - Pairwise hints are dense
     - Prefer a latent model or graphical structure over many one-off joints.
   * - LLM-designed model disagrees with profile
     - Fit-validate both and keep the frontier report.

API Reference
-------------

* :doc:`api/mixle.utils.automatic`
* :doc:`api/mixle.utils.automatic.factories`
* :doc:`api/mixle.utils.automatic.profiling`
* :doc:`api/mixle.utils.automatic.detectors`
* :doc:`api/mixle.task.recommend`
