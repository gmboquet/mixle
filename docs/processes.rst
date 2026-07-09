Temporal and Stochastic Processes
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
* ``ContinuousTimeMarkovChainDistribution``;
* ``ChineseRestaurantProcessDistribution``.

When to Use a Process Model
---------------------------

Use a process model when the observation is not just a row, but an event
history or trajectory. The timing is part of the likelihood.

Keep the observation window with the data. Event-history likelihoods depend on
start time, stop time, censoring, and exposure; dropping that context can make
the same timestamp sequence mean something different.

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
   * - Continuous-time Markov chain
     - State trajectories are fully observed with dwell times
     - Transition rates are estimated from jump counts and state exposure.
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

Compare self-exciting models against a non-self-exciting baseline before
interpreting triggering. Apparent excitation can come from seasonality,
batching, missing exposure, or changing observation windows.

Renewal Processes
-----------------

``RenewalProcessDistribution`` models sequences through the distribution of
interarrival times. It is useful when the waiting-time law is the main
scientific question and there is no self-exciting feedback.

Renewal models compose naturally with scalar interarrival distributions. For
example, a Gamma or log-normal interarrival family can be fit as part of a
larger event model.

For renewal workflows, inspect residual waiting times and censoring behavior.
A fitted interarrival distribution can look acceptable in aggregate while
systematically missing early or late arrivals.

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

Check whether zero, absorbing, or capacity states need explicit support. A
birth-death fit should not hide impossible negative counts or unmodeled
capacity limits in preprocessing.

Continuous-Time Markov Chains
-----------------------------

``ContinuousTimeMarkovChainDistribution`` models fully observed trajectories
with an initial state and a sequence of ``(dwell_time, next_state)`` jumps.
The generator matrix has off-diagonal rates ``q_ij`` and diagonal entries
derived from total exit rates.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats.processes.ctmc import ContinuousTimeMarkovChainEstimator

   trajectories = [
       (0, [(0.8, 1), (1.2, 0), (0.5, 2)]),
       (1, [(1.0, 0), (0.7, 2)]),
   ]

   est = ContinuousTimeMarkovChainEstimator(num_states=3)
   ctmc = optimize(trajectories, est, max_its=1, out=None)
   print(ctmc.generator)

For fully observed trajectories, the MLE is closed form:
``q_ij = n_ij / T_i``, where ``n_ij`` is the observed transition count and
``T_i`` is total dwell time in state ``i``. ``mixle.inference.certify`` reports
this family as ``GLOBAL_UNIQUE``.

That certificate depends on the fully observed trajectory assumption. If jumps,
dwell times, or state labels are censored or inferred upstream, record that
source of uncertainty separately instead of treating the CTMC fit as a fully
observed closed-form result.

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
* a categorical distribution over event type;
* a positive continuous family over severity, amount, or duration;
* a calibrated rule that escalates low-confidence records for review.

That is the Mixle modeling shape: timing, labels, magnitudes, and decisions can
remain separate components while sharing one scoring and inference story.

Diagnostics
-----------

For process models, inspect more than aggregate likelihood:

* compare observed and simulated event counts;
* check residual waiting times;
* inspect intensity around bursts;
* hold out contiguous time ranges;
* compare a self-exciting model against an inhomogeneous Poisson baseline;
* verify calibration of predicted counts or intervals;
* for CTMCs, compare simulated dwell times and transition counts against
  held-out trajectories.

Certification
-------------

Process families now participate in estimation certificates:

* inhomogeneous Poisson, birth-death, and CTMC fits are classified as
  ``GLOBAL_UNIQUE`` when their closed-form count/exposure MLE applies;
* Hawkes variants are classified as ``STATIONARY`` because branching EM or
  ML over self-excitation is non-convex;
* renewal-process certificates inherit the guarantee of the interarrival
  family used in the M-step.

Use :doc:`analysis` for extreme-value and spatial diagnostics, :doc:`inference`
for proper scoring and model comparison, and :doc:`production` for drift
monitoring.

Release Evidence
----------------

For process models, preserve:

* observation-window definitions and exposure assumptions;
* timestamp units, time zones, and censoring policy;
* baseline comparisons such as renewal or inhomogeneous Poisson alternatives;
* residual, simulation, or count-calibration diagnostics;
* certificate assumptions for closed-form fits; and
* blocked or missing optional dependencies for accelerated process examples.

This evidence keeps temporal structure from being mistaken for ordinary row
modeling.
