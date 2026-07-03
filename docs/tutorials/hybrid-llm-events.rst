Hybrid LLM Event Streams
========================

This optional tutorial uses the incubating ``mixle.models`` neural-leaf
surface. A Transformer predicts the next event type from history while a Gamma
distribution models the wait time. The fitted object is one joint distribution,
so scoring an event gives one anomaly signal across type and timing.

1. Define The Observation Shape
-------------------------------

One event is:

.. code-block:: text

   ((history_window, next_event_type), wait_seconds)

The first field is a language-model-style training pair. The second is a
positive continuous value. Keeping them in one composite model lets the fitted
distribution score the whole event.

2. Build The Estimator
----------------------

Use the ready Transformer estimator when you do not need to bring your own
Torch module.

.. code-block:: python

   from mixle.models import TransformerLMEstimator
   from mixle.stats import CompositeEstimator, GammaEstimator

   estimator = CompositeEstimator(
       (
           TransformerLMEstimator(
               vocab=16,
               d_model=96,
               n_layer=3,
               n_head=4,
               block=16,
           ),
           GammaEstimator(),
       )
   )

The ``CompositeEstimator`` is the important move. It does not care that one
child is neural and the other is closed-form; both participate in the same
outer fitting loop.

3. Fit And Score
----------------

.. code-block:: python

   from mixle.inference import optimize

   model = optimize(data, estimator, max_its=20, out=None)
   joint_log_density = model.log_density(((history, next_type), wait))

Because both channels are in the same distribution, an event can be unusual in
type, timing, or the combination of the two.

4. Bring Your Own Module
------------------------

If you already have a Torch module with the expected language-model interface,
wrap it with ``StreamingTransformer`` instead.

.. code-block:: python

   from mixle.models import LM, StreamingTransformer

   leaf = StreamingTransformer(
       LM(vocab=16, d_model=96, n_layer=3, n_head=4, block=16).module
   )
   neural_estimator = leaf.estimator()

Use this route when you need custom module construction, device placement, or
shared embeddings across several leaves.

5. Validate The Neural Leaf
---------------------------

Neural leaves need the ordinary neural-model discipline in addition to Mixle's
distribution checks:

* fix seeds for articles, notebooks, and regression tests;
* hold out event streams by time or source, not by random row when leakage is
  possible;
* monitor the Transformer loss and the joint log-density separately;
* compare against a non-neural baseline before adding the Transformer;
* record the optional dependencies and device configuration in provenance.

Run The Script
--------------

The repository includes a self-contained synthetic example:

.. code-block:: sh

   python examples/hybrid_llm_example.py

Read :doc:`/neural-llm` for the full neural-leaf guide and :doc:`/models` for
the broader applied-model namespace.
