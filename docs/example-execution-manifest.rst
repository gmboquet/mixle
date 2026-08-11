Example Execution Manifest
==========================

This page is the release-facing execution manifest for scripts shipped under
``examples/``. The examples guide describes what each script teaches; this
manifest records what must be executed, skipped, or marked blocked before a
public release.

Authoritative Release Evidence
------------------------------

The dated narrative below is historical engineering context, **not** 0.8.0
release evidence. The authoritative record is the generated
``example-execution-manifest.json`` attached to the GitHub release. Publication
builds it only after all entries in
``release-checklists/0.8.0-repro-bundle.json`` have passing receipts. It binds
every required example's exact command, dependency tier, duration, output
validation contract, and output/receipt digests to the final 40-character
candidate commit and exact wheel SHA-256. Missing, failed, duplicated, stale,
over-budget, or wrong-contract receipts abort publication. The manifest also
binds the versioned dependency-profile digest.

The local required set covers univariate, structured, production/provenance,
and scaling workflows; every entry is self-contained and replays offline.
Other examples remain instructional or optional and are not silently counted
as release evidence.

Current Inventory
-----------------

The core package currently ships 57 Python example scripts (the count was 64 before the
2026-08-04 removal of the direct-dataset flagships, and this line had not caught up -- the
execution status in "Release Execution Status" below and the tree both say 57):

.. list-table::
   :header-rows: 1

   * - Group
     - Examples
     - Release tier
   * - Base distribution galleries
     - ``gallery_*``, ``auto_example.py``, ``ppl_example.py``, ``quickstart_example.py``,
       ``capability_layer_example.py``
     - smoke/validation
   * - Latent and structured models
     - HMM, mixture, association, and structure-learning examples
     - validation
   * - DOE, enumeration, production, and scaling
     - DOE, enumeration (including ``autoregressive_enumeration_example.py``), registry/provenance,
       and backend examples
     - validation/manual
   * - Model comparison, dependence structure, symbolic export, and combinatorial scheduling
     - ``model_comparison_example.py``, ``copula_vine_example.py``, ``symbolic_export_example.py``,
       ``precedence_scheduling_example.py``
     - validation (``symbolic_export_example.py`` needs the ``sympy`` extra)
   * - Task and distillation workflows
     - task distillation, active labeling, extraction, cascade economics
     - validation/manual
   * - Reasoning, cross-modal, and scientist workflows
     - frontier ecosystem, KG agent, scientist, physics inverse, receipts
     - manual unless dependencies are provisioned

Release Execution Status
-------------------------

**2026-08-10 re-verification.** All 57 examples were re-executed against the current
``release/0.8.0`` tip (``7f0c7bd1``), because the 2026-08-07 pass below predates an adversarial
statistical-review campaign (passes 2-23, 2026-08-08..10) that changed public behavior on several
``mixle.inference``, ``mixle.ppl``, ``mixle.reason``, and ``mixle.task`` surfaces -- including one
this pass found the campaign itself had not touched: ``cross_modal_fit_receipt.py`` exercises the
CrossModal transactional-fit repair (D-0171/D-0172) directly. Same method as 2026-08-07: source
tree, Python 3.12.12, one script at a time in its own process under a resident-memory and
free-disk watchdog (debounced against transient readings from other concurrent work on a shared
machine). **57 passed, 0 failed** -- 355.5s total, slowest script (``skeptic_challenge_example.py``)
16.2s, both faster than the 2026-08-07 figures (439s / 24.3s) and consistent with normal run-to-run
variance rather than a real change. One environment gap, unrelated to ``mixle``: ``peft`` was
missing from the execution venv despite being declared in the ``examples`` extras group
(``pyproject.toml``); installed at the declared bound (``peft>=0.11,<1``), after which
``peft_lora_grad_leaf.py`` passes. Zero regressions from the review campaign reached this corpus.

**2026-08-07 pass (superseded environment identity; narrative below still current).** All 57
examples were executed against ``release/0.8.0``, from the source tree on
Python 3.12.12, one at a time in its own process session under a resident-memory watchdog. All 57
completed successfully: 439s in total, peak 2.3 GB, and the slowest single script
``project_neural_to_structured.py`` at 24.3s.

* **57 passed, 0 failed.**
* **Environment: every optional extra provisioned** -- scikit-learn, torch, transformers, datasets,
  pyspark, mpi4py and dask were all importable. That is what the number means and what it does not.
  It says every script EXECUTES TO COMPLETION with exit code 0 against this candidate: no import
  breaks, no API drift, no crash. It is not proof the printed numbers are scientifically correct
  -- an example can run clean and still demonstrate the wrong thing, which only the per-example
  assertions and the review gates can speak to -- and it says nothing about a base install. The
  per-example ``blocked`` classifications in the Inventory below are a separate axis and remain
  accurate for a minimal environment -- ``skeptic_challenge_example.py`` passes here only because
  scikit-learn is present, and still fails without it.

This supersedes the 2026-08-04 pass, which measured only the 23 base-install scripts and predates
the library repairs below. Two of its statements no longer hold: the remaining 34 examples were
recorded as "not executed in this pass" and all of them execute when their prerequisites exist, and
the slowest script was recorded at 46s against 24.3s now.

This run also stands as regression evidence for eight library defects fixed on 2026-08-05/06 --
parameter tying (``keys=``) unusable for the LDA/pLSI/PCFG families, ``BernoulliSetDistribution``
and ``MarkovChainDistribution`` unpicklable at every protocol, and the automatic type-detection
path building models the library then refused, among others. Those fixes touch the key validator
that every estimator walks, sequence length handling, Dirichlet-process component priors, and
pickling; 57 examples and the 131-notebook corpus both execute clean on top of them.

This supersedes the 2026-07-17 pass, which is no longer accurate in three respects.

* ``hierarchical_mixture_example.py`` was recorded as not completing even under a 15-minute
  allowance, and was flagged as the more likely of two candidates to be a real behaviour change
  from the EM-monotonicity work in #435. It now completes in **50 seconds**. The earlier
  measurement was taken while the execution host was under load averages of roughly 140-175
  against 10 cores.
* ``lookback_hmm_example.py`` was recorded as exceeding a 90s budget and then completing at 86s on
  re-run. It now completes in **39 seconds**, consistent with the same host-load explanation. It
  still calls ``optimize(..., max_its=1000, delta=None)``, so it always runs the full thousand EM
  iterations by design; that is a fixed amount of work, not a convergence problem.
* ``win_demo_example.py`` was reclassified to blocked for needing ``torch``. It **passes** when
  torch is present, so it is blocked only on the base install, not unconditionally.

``semi_supervised_mixture_example.py`` was listed among the 19 that passed in July and did not.
It built its estimator from ``SequenceEstimator(CategoricalEstimator(...))``, whose default
``len_estimator=NullEstimator()`` yields a sequence law with no distribution over length -- a law
conditional on the length it is handed, which mixle reports as a ``LIKELIHOOD_FACTOR`` and a
generative mixture refuses as a component. The ground truth it samples from draws lengths from
``CategoricalDistribution({seq_samp: 1.0})``, so the estimator was also fitting a different family
than the one that produced the data. With a length estimator supplied it recovers the generating
parameters: weights 0.598/0.296/0.106 against a true 0.6/0.3/0.1.

Real-data flagship demonstrations (Banking77, UCI Adult, sunspots) were removed
from this repository on 2026-08-04: the repository carries no direct dataset
usage, and real-data demonstrations live in notebooks outside it. Their
historical execution evidence remains in this page's git history.

**``hierarchical_mixture_example.py`` follow-up (2026-07-17).** A later
re-verification pass flagged this example as exceeding its 90s budget against
current ``0.8.0`` source, and -- after it still had not finished with a
15-minute allowance -- raised the EM-monotonicity fix in #435 (2026-07-13,
``mixle/stats/latent/hierarchical_mixture.py``) as the plausible cause, since
that commit changed how the outer-mixture weights are derived. Investigated
directly with a per-iteration log-likelihood/delta trace
(``optimize(..., on_step=...)``) of the example's exact model/data/seed,
captured on three points in history: ``v0.7.0``, the commit immediately
before #435 (``2ed006ca``), and current ``release/0.8.0``.

* **Not a #435 regression.** ``mixle/stats/latent/hierarchical_mixture.py`` is
  byte-identical between ``v0.7.0`` and ``2ed006ca`` (``git diff`` confirms
  zero changes), and the iteration-indexed trajectories for all three points
  agree closely throughout: e.g. log-likelihood at iteration 3 is
  -22681.620149 on both ``v0.7.0`` and ``2ed006ca``, -22681.620138 after
  #435; at iteration 10000, delta is 3.07e-6 before #435 vs. 8.63e-6 after --
  same order of magnitude, no widening gap. The slow-convergence shape below
  predates #435 by at least back to ``v0.7.0``; #435's outer-weight fix did
  not change it.
* **Root cause: this configuration has always converged very slowly against
  the estimator's exact ``delta=1.0e-9`` default.** It is weakly identified --
  4 outer mixture components (3 near-degenerate single-topic, one blended)
  over only 3 mildly-skewed categorical symbols, 8-10 tokens per document. A
  direct trace to the example's ``max_its=10000`` shows log-likelihood
  plateaus near -22681.6 for ~1000 iterations, jumps to ~-22510 between
  iterations 1000-1500 (EM escaping a saddle), then creeps from -22510 to
  -22508.1 over the remaining 8500 iterations without ever satisfying
  ``delta<1e-9`` -- delta is still ~8.6e-6 at iteration 10000, and does not
  drop below ``1e-4`` for good until iteration ~7095. It was always going to
  run the full ``max_its=10000`` rather than stop early on genuine
  convergence; that run took ~262s under this pass's load (``uptime`` load
  averages ~20-40 against 10 cores), plausibly several times that under the
  ~140-175 load of the prior pass. No non-monotonicity, oscillation, or
  hang was observed at any point (``mixle/tests/latent_readout_correctness_test.py``
  gained a new ``HierarchicalMixtureBoundedConvergenceTest`` covering this gap
  on a small, fast, well-separated corpus).
* **Resolution.** The example's ``max_its`` was reduced from 10000 to 2000 --
  comfortably past the iteration-1000-1500 escape -- capturing log-likelihood
  -22509.97 vs. the 10000-iteration run's -22508.13 (over 99.99% of the total
  achievable improvement). The re-budgeted example now completes in
  approximately 20s. The estimator's actual default convergence tolerance
  (``delta=1.0e-9``) is unchanged for real callers; only this example's
  requested ``max_its`` moved.

**2026-07-21 re-verification.** All 19 previously-passed examples plus both
previously-resolved timed-out cases (``hierarchical_mixture_example.py``,
``lookback_hmm_example.py``) were re-run against current ``release-prep/0.8.0``
source (roughly 40 additional bug-fix commits landed since the 2026-07-17 pass,
none in these examples' own code paths) and still pass. This pass also:

* Confirmed ``auto_example.py`` and ``enumeration_showcase_example.py`` --
  both inventoried "Execute." but absent from the 2026-07-17 evidence above --
  pass cleanly and quickly (0.55s and 0.68s respectively). Recorded here as
  the evidence that was previously missing; not a live concern.
* Corrected the ``lookback_hmm_example.py`` mechanism claim above: an
  instrumented per-iteration trace shows only about 480 of the requested 1000
  iterations actually execute, not the full 1000 as previously stated.
  ``delta=None`` does disable the ``delta``-gated convergence check, but a
  separate, ``delta``-independent monotonicity guard in ``_fused_em_loop``
  (``mixle/inference/estimation.py:738-741``, added by a commit predating even
  the 2026-07-17 baseline) breaks the loop when a per-iteration log-likelihood
  delta goes slightly negative -- which happens near convergence here. The
  bottom-line ``passed`` status and the ~86s runtime are unaffected; only the
  stated reason was wrong.
* Documented, in the Inventory table below, a previously-unrecorded ``torch``
  hard dependency (no classical fallback) for ``doe_example.py`` (via
  ``GaussianProcessRegressor``) and for ``task_cascade_economics_example.py``,
  ``task_distill_example.py``, ``task_extraction_example.py``, and
  ``task_llm_active_example.py`` (all via ``mixle.task.distill._fit_mlp`` or
  ``mixle.task.extract``'s distillation path) -- the same shape of blocker
  already documented for ``win_demo_example.py``. This is why these five,
  despite being policy-required by the Minimum Release Run below, had no
  recorded execution evidence: a base install cannot run them at all, and the
  Inventory entries did not say so.
* Confirmed ``geoscience_inversion_report.py`` -- also absent from the
  2026-07-17 evidence despite exercising ``mixle.task.inverse`` -- passes its
  execution contract. The script's calibration layer detected a poorly
  calibrated candidate and abstained. Exact measurements belong in the
  content-addressed receipt, not in this mutable narrative.

**2026-07-23 addition.** ``quickstart_example.py`` was added to close a
coverage gap: neither ``mixle.describe()`` (the package docstring's own
"start here") nor ``mixle.propose()`` (headlined in the README's Package
highlights, paired with ``optimize()``) had a runnable example calling them
directly. Passes in a base install, no optional dependencies, 3.4s.

**2026-07-23 addition (batch 2).** Six more scripts closed the remaining
README-headlined coverage gaps identified in the same audit. All were run
directly (not just imported) and their printed output checked against an
independent hand or brute-force calculation, not just "exited zero":

* ``precedence_scheduling_example.py`` -- ``mixle.precedence_scheduling``
  (maximum-weight closure, time-phased MILP scheduling), on a software-release
  dependency DAG rather than the test suite's mine-planning framing, to show
  the module's generality is real. Base install, 2.3s. The script's own
  in-line assertions verify the returned schedule against the raw capacity
  and precedence arrays.
* ``capability_layer_example.py`` -- the rest of ``mixle.capability``
  (``capabilities``, ``supports``, ``require``, ``catalog``, ``what_supports``,
  ``summarize``) beyond ``describe()``. Base install, 2.2s.
* ``symbolic_export_example.py`` -- ``mixle.engines.symbolic_engine`` /
  ``symbolic_export`` (LaTeX / SymPy / optional Sage). Needs ``mixle[sympy]``;
  3.5s. Differentiates each of two families' closed-form log-density to its
  score function and confirms Gaussian's score grows without bound as an
  observation moves further from the mean, while Student-t's saturates back
  toward zero -- checked against the textbook closed forms by
  hand, not just printed. Separately found (not fixed; see below): constants
  like ``pi`` render as decimals rather than staying symbolic under
  ``SYMBOLIC_ENGINE`` -- correct value, not textbook-pretty LaTeX.
* ``autoregressive_enumeration_example.py`` -- ``mixle.enumeration.AutoregressiveEnumerable``
  over a small synthetic prefix-dependent logit table (base install, no torch,
  no network -- a real LM plugs into the same ``next_logprobs`` interface).
  2.2s. Instruments the model callable itself to show the number of distinct
  prefixes actually queried stays far below the full sequence-space size.
  Separately found and since fixed (see below): ``nucleus_size()`` was wrong
  for this class.
* ``model_comparison_example.py`` -- ``mixle.ppl``'s ``waic``/``loo``/``compare()``,
  ranking a deliberately-wrong unimodal fit against a 2-component mixture and
  a second wrong (Student-t) fit on genuinely bimodal data. The script reports
  the measured ranking and diagnostics; this narrative does not preserve a
  result from an unbound historical run.
* ``copula_vine_example.py`` -- direct ``CopulaDistribution`` /
  ``RVineCopulaDistribution`` fitting on Clayton-generated
  lower-tail-dependent heterogeneous-marginal data, contrasted against a
  Gaussian-copula core fit to the same data. The script checks its measured
  tail diagnostics; exact results belong in the candidate-bound receipt.

Two real, non-blocking issues were found incidentally while building these
(not introduced by them), both since fixed -- see the 2026-07-23 follow-ups
below: ``SYMBOLIC_ENGINE``'s constants not staying symbolic, and
``AutoregressiveEnumerable.nucleus_size()`` returning an incorrect size
(``size_lower=size_upper=0`` when the correct answer is 5, confirmed against
brute force) while its ``covered_mass`` field stayed correct.

**2026-07-23 follow-up: pi fixed.** Root cause: ``GaussianDistribution`` /
``LogGaussianDistribution`` / ``MultivariateGaussianDistribution`` got ``pi``
via ``from mixle.engines.arithmetic import *`` (``StudentTDistribution`` used
``math.pi`` directly); both are ordinary Python import/reference semantics --
the value is bound once, at each family module's own import time, against
whichever engine was the default THEN. Passing an explicit
``engine=SYMBOLIC_ENGINE`` to a ``backend_log_density_from_params`` /
``backend_seq_log_density`` / ``exp_family_log_partition`` call never
revisited that binding, so the LaTeX output always showed a decimal literal
regardless of which engine was requested. Fixed by reading ``engine.pi`` off
the actual passed-in engine at each of the 8 confirmed call sites across the
4 affected families (every ``ComputeEngine`` already exposes ``.pi``); no
numeric change on the ordinary numpy path (``engine.pi`` is the same
``math.pi`` value), confirmed by the full ``backend_scoring_test.py`` suite
(264 subtests) plus a dedicated regression test. Checked every other family
using the same import pattern for the same bug -- ``diagonal_gaussian.py``
and ``von_mises_fisher.py``'s ``pi`` hits were false positives (a docstring
formula and plain-float ``__init__`` setup respectively), and
``markov_chain.py`` / ``hidden_markov.py`` / ``lookback_hidden_markov_model.py``'s
were the unrelated HMM initial-state-distribution variable of the same name.
``MultivariateGaussianDistribution``'s fix is verified correct but its full
symbolic export still hits a separate, pre-existing array-conversion
limitation in the symbolic engine, unrelated to this bug and not fixed here.

**2026-07-23 follow-up: nucleus_size() fixed.** Root cause: ``nucleus_size()``
delegated to the generic ``density_rank.count_dp_top_p``, which assumes its
exact per-item mass histogram (bucketed by the floor of the EXACT total
log-density) and the count index share one bucket numbering -- true for
Composite/Record/Sequence, false for ``AutoregressiveEnumerable``, whose
``quantized_count_index`` buckets a sequence by the SUM of its per-step
floor-quantized buckets (``floor(a) + floor(b) <= floor(a + b)``, so the
structural bucket is systematically <= the exact one, confirmed both
theoretically and empirically -- a 0-1 bucket gap per real sequence at
``oversample=64, L=3``). Fixed by giving ``nucleus_size()`` its own
implementation (mirroring ``mass_above``'s existing fix for the identical
discrepancy) that derives every quantity from the structural count index
alone, rather than delegating to ``count_dp_top_p``; also fixed
``nucleus_size()`` silently ignoring the instance's own ``oversample`` /
``bin_width_bits``. ``mixture_cross_rank`` and the other
``quantized_count_index`` call sites in ``density_rank.py`` were checked for
the same coupling assumption and found unaffected -- each builds and reads
one self-consistent histogram rather than mixing two independently-bucketed
ones. Verified against the exact repro (previously 0, now matches
brute-force truth of 5) across a full sweep of targets on both a
fixed-length and a terminating (``eos``) model; a naive fix that always
searched for a fully-certified bracket was correct but required exponential
search depth on the terminating model (steps_bound scales with the
model's ``max_depth`` safety cap, not the depth of sequences actually
relevant to a given target) -- the shipped fix instead uses a fast,
bounded search and reports ``truncated=True`` (a documented, pre-existing
part of ``CountDPTopPResult``'s contract: ``size_upper`` is a floor, not a
cover, when truncated) rather than searching indefinitely for a certificate.

Execution status should be recorded as evidence, not inferred from import
success or from an earlier notebook run. If an example writes an artifact, the
artifact path and any cleanup policy should be captured with the status.

Execution state and claim state are separate. Every attempted example receives
one execution state:

``passed``
    The script completed from a clean install with the documented command. This
    proves runnability only; it does not verify any advertised result.

``failed``
    The script exited non-zero. Record the first meaningful exception.

``timed_out``
    The script exceeded the declared runtime budget.

``blocked``
    A required dataset, credential, GPU, model weight, external service, or
    optional dependency was unavailable.

``skipped``
    The release configuration intentionally omitted the script with a recorded
    reason.

Do not merge ``blocked`` and ``skipped``. A blocked example needs an external
prerequisite; a skipped example was deliberately left out of the release gate.
That distinction matters when deciding whether documentation can claim the
workflow is healthy.

An evidence-class example additionally receives claim state ``verified`` only
when a machine-readable acceptance contract checks the advertised behavior and
the process fails if that contract is absent or false. Publication's
``mixle.example_execution_manifest/v2`` therefore records
``execution_status``, ``claim_status``, and ``acceptance_contract`` separately.
The exact-output and JSON assertions in
``release-checklists/0.8.0-repro-bundle.json`` are the release claim oracles.
Inventory entries below that merely say “Execute” are runnability targets, not
claim evidence.

Minimum Release Run
-------------------

The minimum release run should include:

* all base-install examples listed in :doc:`examples`;
* every example referenced by README files or Sphinx pages;
* every example touching public APIs changed in the current release scope;
* task-distillation examples because the current release scope includes task
  and DOE-distillation surfaces;
* DOE examples because the current release scope includes pool-based DOE for
  distillation and cross-modal training; and
* vision or reasoning examples only when their optional dependencies are
  installed, otherwise mark them blocked with the missing prerequisite.

Example command shape:

.. code-block:: console

   python examples/gallery_univariate_example.py
   python examples/task_distill_example.py
   python examples/doe_example.py

When release validation uses a timeout wrapper, record the timeout and whether
the script is expected to be short, long, or manual.

Timeouts should be chosen before execution. A script that times out under the
declared budget should be recorded as ``timed_out`` even if it might finish
eventually on a warmer machine.

Inventory
---------

.. list-table::
   :header-rows: 1

   * - Path
     - Expected status before release
   * - ``examples/auto_example.py``
     - Execute or record failure.
   * - ``examples/autoregressive_enumeration_example.py``
     - Execute. Base install (numpy only; synthetic toy model, no torch/network).
   * - ``examples/calibrated_report_demo.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/capability_layer_example.py``
     - Execute. Base install.
   * - ``examples/copula_vine_example.py``
     - Execute. Base install.
   * - ``examples/cross_modal_fit_receipt.py``
     - Execute as a synthetic multi-vector feature tutorial. It is not real
       cross-modal evidence because it has no raw modality inputs or encoders.
   * - ``examples/doe_example.py``
     - Blocked on ``torch`` in a base install (``minimize()``'s Bayesian
       optimization routes unconditionally through
       ``mixle.models.gaussian_process.GaussianProcessRegressor``, which has no
       classical fallback); execute with ``torch`` installed for DOE coverage.
   * - ``examples/engine_benchmark_example.py``
     - Manual/benchmark or bounded smoke run.
   * - ``examples/enumeration_example.py``
     - Execute.
   * - ``examples/enumeration_showcase_example.py``
     - Execute or classify as long-running.
   * - ``examples/extensibility_seams_example.py``
     - Execute.
   * - ``examples/flagship_kg_agent.py``
     - Manual unless KG/RAG prerequisites are provisioned.
   * - ``examples/flagship_physics_inverse.py``
     - Execute or mark blocked on PDE/scientific dependencies.
   * - ``examples/flagship_triage_app.py``
     - Manual unless local reasoning prerequisites are provisioned.
   * - ``examples/frontier_family_showcase.py``
     - Manual/integration.
   * - ``examples/geoscience_inversion_report.py``
     - Execute or mark blocked on scientific dependencies.
   * - ``examples/frontier_ecosystem_demo.py``
     - Manual/integration.
   * - ``examples/gallery_combinators_example.py``
     - Execute.
   * - ``examples/gallery_directional_example.py``
     - Execute.
   * - ``examples/gallery_graphs_example.py``
     - Execute.
   * - ``examples/gallery_multivariate_example.py``
     - Execute.
   * - ``examples/gallery_processes_example.py``
     - Execute.
   * - ``examples/gallery_rankings_example.py``
     - Execute.
   * - ``examples/gallery_structured_example.py``
     - Execute.
   * - ``examples/gallery_univariate_example.py``
     - Execute.
   * - ``examples/heterogeneous_correctness_example.py``
     - Execute.
   * - ``examples/heterogeneous_representation_example.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/hidden_association_example.py``
     - Execute.
   * - ``examples/hierarchical_mixture_example.py``
     - Execute.
   * - ``examples/joint_mixture_example.py``
     - Execute.
   * - ``examples/label_economics_demo.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/latent_variable_models_example.py``
     - Execute.
   * - ``examples/lookback_hmm_example.py``
     - Execute.
   * - ``examples/mixture_reduction_benchmark.py``
     - Manual/benchmark or bounded smoke run.
   * - ``examples/model_comparison_example.py``
     - Execute. Base install.
   * - ``examples/multimodal_stage1_demo.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/peft_lora_grad_leaf.py``
     - Execute with optional-dependency status recorded (needs ``peft``,
       not a mixle dependency).
   * - ``examples/ppl_example.py``
     - Execute.
   * - ``examples/precedence_scheduling_example.py``
     - Execute. Base install.
   * - ``examples/production_example.py``
     - Execute with artifact-output path recorded.
   * - ``examples/project_neural_to_structured.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/quickstart_example.py``
     - Execute.
   * - ``examples/reasoner_investigation_demo.py``
     - Manual/integration.
   * - ``examples/scaling_example.py``
     - Execute or classify by backend availability.
   * - ``examples/semi_supervised_mixture_example.py``
     - Execute.
   * - ``examples/shared_embedding_example.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/skeptic_challenge_example.py``
     - Blocked on ``scikit-learn`` (Act 1's sklearn-baseline comparison
       imports it directly; not a mixle dependency or extra, install
       separately to run).
   * - ``examples/structure_learning_example.py``
     - Execute.
   * - ``examples/structured_hmm_example.py``
     - Execute.
   * - ``examples/structured_leaves_example.py``
     - Execute.
   * - ``examples/symbolic_export_example.py``
     - Blocked on ``sympy`` in a base install (optional ``mixle[sympy]`` extra,
       not a core dependency); execute with ``sympy`` installed. Section 4
       (Sage export) is separately gated and skips gracefully without
       ``mixle[sage]``.
   * - ``examples/task_cascade_economics_example.py``
     - Blocked on ``torch`` in a base install (``mixle.task.distill._fit_mlp``,
       same chokepoint as ``win_demo_example.py``); execute with ``torch``
       installed for task coverage.
   * - ``examples/task_distill_example.py``
     - Blocked on ``torch`` in a base install (``mixle.task.distill._fit_mlp``);
       execute with ``torch`` installed for task coverage.
   * - ``examples/task_extraction_example.py``
     - Blocked on ``torch`` in a base install (``mixle.task.extract``'s
       distillation path); execute with ``torch`` installed for task coverage.
   * - ``examples/task_llm_active_example.py``
     - Blocked on ``torch`` in a base install (the local teacher call itself
       does not need it, but the local-student distillation step routes
       through ``mixle.task.distill._fit_mlp``); execute with ``torch``
       installed, not just teacher/provider requirements.
   * - ``examples/vlm_trust_receipts_demo.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/win_demo_example.py``
     - Blocked on ``torch`` (``solve()``'s MLP distillation path
       (``mixle.task.distill._fit_mlp``) always needs torch, no classical
       fallback; ``pip install mixle[torch]``).

Evidence to Record
------------------

For each executed example, record:

* command;
* package version or wheel filename installed;
* optional extras installed;
* timeout;
* status;
* claim status and its machine acceptance contract, if used as evidence;
* first error or output artifact path; and
* whether network or external data was used.

For blocked or skipped examples, record:

* the missing prerequisite or release-scope reason;
* whether the example is referenced by a public guide;
* the owner of the follow-up decision; and
* the condition that would move it back into the release gate.

The manifest should make it clear which examples are evidence for the release
and which examples remain illustrative but unexecuted.
