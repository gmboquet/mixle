Temporal And Stochastic Processes
=================================

``mixle.process`` collects temporal and stochastic-process families that are
otherwise easy to miss in the broader distribution tree. These models are for
event-time data, self-excitation, renewal structure, birth-death trajectories,
and random partition processes.

The public namespace re-exports:

* ``HawkesProcessDistribution``;
* ``PowerLawHawkesDistribution``;
* ``MultivariateHawkesProcessDistribution``;
* ``InhomogeneousPoissonProcessDistribution``;
* ``RenewalProcessDistribution``;
* ``BirthDeathSamplingDistribution``;
* ``ChineseRestaurantProcessDistribution``.

When To Use A Process Model
---------------------------

Use a process model when the observation is not just a row, but an event
history or trajectory. The timing is part of the likelihood.

.. list-table::
   :header-rows: 1

   * - Process
     - Use when
     - Key idea
   * - Hawkes
     - Events trigger more events
     - Intensity rises after recent arrivals and decays over time.
   * - Power-law Hawkes
     - Excitation has long memory
     - Triggering decays with a heavy-tailed kernel.
   * - Multivariate Hawkes
     - Event types excite each other
     - A matrix controls cross-type excitation.
   * - Inhomogeneous Poisson
     - Rate changes over time but events do not self-excite
     - Intensity is time-varying and exogenous.
   * - Renewal
     - Waiting times are iid or family-modeled
     - Interarrival distribution drives the process.
   * - Birth-death sampling
     - Counts evolve through arrivals and removals
     - Trajectory likelihood depends on birth and death rates.
   * - Chinese restaurant process
     - Partitions grow sequentially
     - New clusters appear with concentration-controlled probability.

Point Processes
---------------

Hawkes and Poisson process models are appropriate for timestamped event
sequences. Depending on the fitted family, the model may expose intensity,
expected count, sampling, and sequence log-density behavior.

Use Hawkes processes for:

* incident cascades;
* user activity bursts;
* aftershock-like phenomena;
* alert streams where one event increases near-future risk.

Use inhomogeneous Poisson processes for:

* seasonality or time-of-day rates;
* scheduled operational load;
* background arrivals without self-excitation.

Use multivariate Hawkes processes when event types influence each other, such
as support categories, market event classes, operational alerts, or social
interaction types.

Renewal Processes
-----------------

``RenewalProcessDistribution`` models sequences through the distribution of
interarrival times. It is useful when the waiting-time law is the main
scientific question and there is no self-exciting feedback.

Renewal models compose naturally with scalar interarrival distributions. For
example, a Gamma or log-normal interarrival family can be fit as part of a
larger event model.

Birth-Death Processes
---------------------

``BirthDeathSamplingDistribution`` models trajectories where a population,
queue, active set, or count evolves by births and deaths. Use it when both
increments and decrements are meaningful, and the path itself is observed.

Examples include:

* queue size traces;
* active session counts;
* population dynamics;
* open/closed case counts.

Chinese Restaurant Processes
----------------------------

``ChineseRestaurantProcessDistribution`` models a sequence of cluster
assignments where new clusters can appear over time. It is useful as a prior or
standalone distribution for partition-valued data.

For fitted finite approximations and variational mixtures, see the
Dirichlet-process material in :doc:`models`.

Composing Process Models
------------------------

Process models are often only one field in a heterogeneous observation. A
single application might combine:

* a Hawkes process over event times;
* a Transformer over event text;
* a categorical distribution over event type;
* a Gaussian or Gamma distribution over magnitude;
* a calibrated task model that decides whether to escalate.

That is the intended Mixle shape. Timing, content, labels, and decisions can
remain separate components while sharing one scoring and inference story.

Diagnostics
-----------

For process models, inspect more than aggregate likelihood:

* compare observed and simulated event counts;
* check residual waiting times;
* inspect intensity around bursts;
* hold out contiguous time ranges;
* compare a self-exciting model against an inhomogeneous Poisson baseline;
* verify calibration of predicted counts or intervals.

Use :doc:`analysis` for extreme-value and spatial diagnostics, :doc:`inference`
for proper scoring and model comparison, and :doc:`production` for drift
monitoring.

