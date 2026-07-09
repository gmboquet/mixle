Fitting Heterogeneous Records
=============================

This tutorial fits a mixture over records shaped like:

.. code-block:: text

   (category, real_value, variable_length_count_sequence)

That one observation shape contains three different supports. ``mixle`` handles
it by making the model a composition of three estimators.

The point of the example is not the particular families. The point is the
shape rule: one observation is one Python value, and the estimator should have
the same structure as that value.

1. Import the pieces
--------------------

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import (
       CategoricalEstimator,
       CompositeEstimator,
       GaussianEstimator,
       MixtureEstimator,
       PoissonEstimator,
       SequenceEstimator,
   )

2. Prepare data
---------------

The records below are intentionally small, but they show the shape:

.. code-block:: python

   data = [
       ("a", -0.4, [5, 7]),
       ("b", 4.9, [11, 9]),
       ("a", 0.2, [6, 5, 4]),
       ("b", 5.3, [10, 12, 11]),
       ("a", -1.1, [4, 6]),
       ("b", 4.5, [9, 10]),
       ("a", 0.7, [5, 5]),
       ("b", 5.1, [12, 8]),
   ]

Before fitting, verify that this shape is stable across the dataset. Mixed
record workflows usually fail because a field is sometimes missing, a scalar
is sometimes wrapped in a list, or a category value appears with a different
type. Normalize those cases before the estimator is built; the estimator should
describe the intended data contract, not repair arbitrary input rows.

3. Mirror the data shape with estimators
----------------------------------------

``CompositeEstimator`` means "one observation is a tuple." Its children are
matched position by position.

.. code-block:: python

   component = CompositeEstimator(
       (
           CategoricalEstimator(),
           GaussianEstimator(),
           SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
       )
   )

The third field is a sequence of counts. ``SequenceEstimator`` fits the element
distribution and, when supplied, a distribution over sequence length.

4. Add latent structure
-----------------------

Wrap two copies of the component in ``MixtureEstimator``:

.. code-block:: python

   estimator = MixtureEstimator([component, component])
   model = optimize(data, estimator, max_its=20, out=None)

The fitted object is a ``MixtureDistribution``. Each component is a
``CompositeDistribution`` with the same three-child structure.

Mixtures can have local optima. For a real analysis, run several random starts
with :func:`mixle.inference.best_of` or pass a validation set to the fitting
workflow before interpreting the components.

Use a held-out score or a repeatable initialization strategy when the result
will be compared across runs. A mixture component should not be named or
reported because it appeared in one fit; it should be stable enough to survive
the validation protocol for the application.

5. Query the fitted model
-------------------------

.. code-block:: python

   score = model.log_density(("a", 0.0, [5, 6]))
   samples = model.sampler(seed=0).sample(3)

``log_density`` returns one joint score for the whole record. Low probability
can come from the category, the real value, the count sequence, the sequence
length, or the mixture assignment implied by the record.

When a record scores poorly, inspect each field under the fitted component
structure before treating it as a global anomaly. In heterogeneous data, a
single malformed field can dominate the joint score.

6. Inspect posterior responsibility
-----------------------------------

For latent models, inspect responsibilities before naming clusters.

.. code-block:: python

   responsibilities = model.posterior(data)
   print(responsibilities[:3])

High responsibility for one component means the row is strongly associated
with that latent type under the fitted model. Ambiguous rows are often more
useful than the obvious ones when deciding whether the component structure is
scientifically meaningful.

Responsibilities are model-relative. They should be used to inspect the fitted
latent structure, not as externally validated labels. If the clusters will
drive decisions, check them against domain labels, downstream outcomes, or a
separate stability analysis.

7. Use dictionaries when fields are named
-----------------------------------------

Tuple position is compact, but production records usually have names. Use
``RecordEstimator`` for dictionary-shaped observations:

.. code-block:: python

   from mixle.stats import RecordEstimator, field

   named = RecordEstimator(
       (
           field("category", CategoricalEstimator()),
           field("value", GaussianEstimator()),
           field(
               "counts",
               SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
           ),
       )
   )

The same fitting route applies; only the observation shape changes.

Production Checks
-----------------

* Replace ``CompositeEstimator`` with ``RecordEstimator`` if your observations
  are dictionaries.
* Replace ``GaussianEstimator`` with another scalar family if the real-valued
  field has skew, tails, or bounded support.
* Use :func:`mixle.inference.best_of` for more robust mixture fitting.
* Pass ``backend="mp"`` or an engine when the data is large enough to justify
  parallel work.
* Use :doc:`/capabilities-contracts` before relying on enumeration,
  conditioning, or exact posterior behavior.
* Save the fitted schema, component count, initialization policy, validation
  score, and any field-level preprocessing assumptions with the model artifact.
