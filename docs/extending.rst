Extending mixle
===============

Extend mixle by adding behavior to the existing contracts rather than by
special-casing fit loops. A new family, engine, backend, or PPL constructor
should make itself visible through the same estimator, encoder, capability, and
inference surfaces as the built-in objects.

Add a Distribution Family
-------------------------

A full distribution family usually has five pieces:

``Distribution``
    Stores fitted parameters. Implements ``log_density(x)``, ``sampler()``,
    ``estimator()``, and, for vectorized paths, ``seq_log_density(encoded)`` and
    ``dist_to_encoder()``.

``Sampler``
    Produces seeded samples. Seed handling should be deterministic and local to
    the sampler.

``Estimator``
    Declares the family to fit. Implements ``accumulator_factory()`` and
    ``estimate(nobs, suff_stat)``.

``Accumulator``
    Collects sufficient statistics or training telemetry. Implements scalar and
    sequence update paths, ``combine``, ``value``, and ``from_value``.

``DataEncoder``
    Converts raw Python observations into encoded arrays or structured payloads.

Use a nearby family as the template. For example, start from another continuous
univariate family when adding a scalar density, or from an existing combinator
when the new object delegates to child distributions.

Keep the public docstring and guide text in the same pass as the code. A new
family is not complete until its observation shape, fitted parameters,
capabilities, missing-data behavior, and optional dependencies are documented.

Minimum Family Checklist
------------------------

* scalar ``log_density`` agrees with vectorized ``seq_log_density``;
* sampler is reproducible under a fixed seed;
* estimator recovers parameters on synthetic data;
* accumulator values combine correctly across chunks;
* encoder accepts the public observation shape;
* ``str``/serialization behavior is tested when the family supports it;
* optional capability methods are present only when they are correct.
* non-finite observations reject, marginalize, or model missingness explicitly;
* impossible observations score as ``-inf`` or raise a documented error, not
  ``NaN``.

Capabilities
------------

Capabilities are behavior contracts. Add them when the object truly supports
the behavior:

.. list-table::
   :header-rows: 1

   * - Capability behavior
     - Implement when
   * - enumeration
     - the support can be traversed in descending probability
   * - finite support
     - support size is known and finite
   * - rank/seek
     - the family supports structural unranking or count-budget indexes
   * - conjugate update
     - priors and posterior updates are mathematically valid
   * - conditioning/marginalization
     - the family can return exact conditional or marginal distributions
   * - latent posterior
     - hidden assignments or paths can be queried
   * - backend scoring
     - scoring can run safely on a compute engine

Use ``mixle.describe`` during development to confirm the object advertises the
expected behavior.

Capability claims should be exact. If a method is approximate, conditional on
optional dependencies, or valid only for a subset of parameters, document that
restriction in the docstring and guide page.

For the full list of behavior contracts and predicates, see
:doc:`capabilities-contracts`.

Add a Combinator
----------------

Combinators should preserve child capabilities when possible. For example, a
composite can enumerate when its children can enumerate; a transform can expose
density only when the change of variables is valid; a latent wrapper can expose
posterior queries only when it can compute them.

Keep the observation shape inspectable. A user should be able to look at one raw
record and understand which child handles each part.

Add a Neural Leaf
-----------------

Neural leaves still participate through estimators and distributions. The
parent model should not need to know whether a child M-step is closed-form or
gradient-based.

Practical rules:

* keep the public observation shape explicit, such as ``(context, target)``;
* keep module ownership and optimizer lifetime clear;
* avoid buffering entire corpora in accumulators unless the design requires it;
* expose telemetry separately from sufficient statistics when training is
  streamed;
* test both scalar and encoded scoring paths.

Add a PPL Constructor
---------------------

The PPL lowers symbolic ``RandomVariable`` expressions to concrete
distributions, estimators, or inference targets. Add or register a lowering rule
instead of branching inside the fit loop.

The PPL constructor should document:

* fixed parameter slots;
* ``free`` parameter slots;
* prior-bearing slots;
* constraints or named parameters;
* the lowered distribution/estimator family.

Add an Engine or Backend
------------------------

Engines own array math. Backends own where encoded data are folded.

For a new engine:

* implement the ``ComputeEngine`` surface;
* register array types with ``register_array_type``;
* make host/device boundaries explicit;
* provide ``to_numpy`` behavior;
* add precision and dtype normalization where needed.

For a new encoded-data backend:

* preserve the ``[(count, payload)]`` contract;
* support estimator encoders without changing model code;
* keep partitioning compatible with data sample structure;
* expose failures clearly when optional dependencies are missing.

The lower-level compute contracts, declaration metadata, encoded payloads, and
kernel-selection machinery are documented in :doc:`compute-layer`.

Public Surface Checklist
------------------------

Public extensions should update:

* the relevant guide page;
* :doc:`api-overview` if a new public namespace or common import is added;
* examples or tutorials when the behavior is user-facing;
* generated API reference pages via ``make -C docs apidoc``.
* :doc:`stability-and-missing-data` when the extension changes non-finite or
  missing-data behavior.

Testing Requirements
--------------------

Add tests at the same level as the extension:

* unit tests for the family or helper;
* estimator recovery tests on synthetic data;
* vectorized/scalar parity tests;
* capability-specific tests;
* optional-dependency skip markers where needed;
* integration tests when the extension participates in ``optimize``.
* artifact reload tests when the extension can be saved or registered.

Design Rule
-----------

Prefer adding one capability-aware object over adding one new branch to a
central algorithm. The rest of mixle should discover the behavior through the
contract.
