Training At Scale
==================

A frontier-scale training run executes for weeks unattended across hundreds or thousands of
devices. Nobody is staring at a loss curve the whole time, so the only way to know a run is healthy
-- or to decide whether to commit more hardware to the next size up -- is receipts computed *from
the loop itself*. This page covers the two pieces that produce those receipts:
:class:`~mixle.utils.parallel.training_health.TrainingHealthMonitor` (MFU, loss/grad-norm anomaly
detection, per-restart continuity) and :mod:`mixle.task.pilot_ladder` (a rung-by-rung GO/NO-GO
staging walker that gates progression to the next scale on those exact receipts).

.. code-block:: python

   import torch
   from mixle.models.transformer import build_causal_lm
   from mixle.utils.parallel.training_health import TrainingHealthMonitor, flop_config_from_causal_lm

   model = build_causal_lm(vocab=32, d_model=16, n_layer=2, n_head=2, block=8)
   cfg = flop_config_from_causal_lm(model, seq_len=8)   # theoretical FLOPs/step from model shape
   opt = torch.optim.SGD(model.parameters(), lr=1e-3)

   monitor = TrainingHealthMonitor(flop_config=cfg, peak_flops_per_sec=1e12)
   for step in range(5):
       x, y = ...  # your batch
       opt.zero_grad()
       loss = torch.nn.functional.cross_entropy(model(x), y)
       loss.backward()
       grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
       opt.step()
       monitor.observe_step(step, float(loss.item()), grad_norm=float(grad_norm.item()),
                             step_time_s=..., batch_size=4)

   monitor.report()   # {"mfu": ..., "anomalies_by_kind": ..., "restarts": {"continuity_ok": ...}}

Running exactly that loop against a real (tiny) transformer for five steps on this machine
produced ``n_params = 7104`` (position-embedding params excluded, the same convention nanoGPT's
``get_num_params(non_embedding=True)`` uses), ``flops_per_iter(batch=4) = 1.462e+06``, and an MFU
report of ``mean=1.125e-02, min=5.909e-03, max=1.903e-02`` over 5 samples against a made-up
``peak_flops_per_sec=1e12``. The *absolute* MFU number a laptop produces is not comparable to a
real cluster's -- that comparison is deferred until the real distributed trainer (roadmap F1)
runs the same accounting on real hardware. What is real here is the FLOPs formula and the
achieved/peak ratio math, both pinned by ``mixle/tests/training_health_test.py``.

The Compute Box
----------------

The roadmap's F5/F7 cards frame a headline training run as a choice made *under a fixed compute
box*: a declared budget (devices, wall-clock, FLOPs) that the run's final size, context length,
and dense-vs-MoE architecture are chosen against, rather than a run that simply keeps scaling
until someone notices the bill. F5 (the scaling-law fitter that would make that choice
model-based rather than a guess) lives on ``origin/scaling-law-fits`` and is not merged into this
branch's base -- so this page cannot show a real scaling-law fit choosing a headline
configuration.

What this page *can* show, and what actually exists on this branch, is the discipline a compute
box is supposed to enforce even before a real fit is available: :mod:`mixle.task.pilot_ladder`
never silently fabricates a compute-allocation decision it cannot make. A rung that would depend
on F5 to pick its size records that dependency by name in ``skipped_pieces`` with the real reason
it could not be exercised, and a rung that explicitly opts into ``exercise_scaling_law_fit=True``
raises ``NotImplementedError`` rather than inventing a number. That is the honest version of "the
compute box governs the next decision" available from this worktree today: name the missing piece,
do not fake the decision it would have made.

The Pilot Ladder: Rung-By-Rung GO/NO-GO
------------------------------------------

The real roadmap rungs are unmeasurable in this environment -- (i) 1B params / 8k context / 8
GPUs, (ii) 8B / 128k / 256 GPUs, (iii) 8B / 10M context / 1000 GPUs, (iv) a headline run sized by
F5 -- no such hardware exists here. ``run_pilot_ladder`` instead builds the orchestration
machinery those rungs would actually run through, exercised at a tiny simulated scale standing in
for the real progression: train each :class:`~mixle.task.pilot_ladder.Rung`'s model, collect its
MFU / loss-curve / forgetting-curve artifacts through the exact
:class:`~mixle.utils.parallel.training_health.TrainingHealthMonitor` machinery above, append one
Bayesian decision-journal entry, and gate progression to the next rung on a real GO/NO-GO check of
those measured receipts -- not a human eyeballing a dashboard.

.. code-block:: python

   from mixle.task.pilot_ladder import Rung, run_pilot_ladder

   rungs = [
       Rung(name="rung_i_shakeout", real_target="1B params / 8k context / 8 GPUs",
            decision_pieces=("F1", "F4"), vocab=24, d_model=8, n_layer=2, n_head=2, block=6,
            steps=30, batch_size=8, lr=5e-2, seed=0, max_final_loss=4.0, max_forgetting_gap=4.0),
       Rung(name="rung_ii_bakeoff", real_target="8B params / 128k context / 256 GPUs",
            decision_pieces=("E7", "H2", "F9"), vocab=24, d_model=8, n_layer=2, n_head=2, block=6,
            steps=30, batch_size=8, lr=5e-2, seed=0, max_final_loss=4.0, max_forgetting_gap=4.0,
            exercise_mup_transfer=True, mup_base_width=8,
            exercise_moe_decision=True, moe_experts=4),
       Rung(name="rung_iii_context", real_target="8B params / 10M context / 1000 GPUs",
            decision_pieces=("E8", "F5"), vocab=24, d_model=8, n_layer=2, n_head=2, block=6,
            steps=30, batch_size=8, lr=5e-2, seed=0, max_final_loss=4.0, max_forgetting_gap=4.0),
   ]
   result = run_pilot_ladder(rungs, peak_flops_per_sec=1.0e9)

   result.passed_rungs()   # names of every rung that cleared its own GO/NO-GO bar
   result.halted_at        # the rung name where the ladder stopped, or None

Running that exact ladder produced ``halted_at=None`` and
``passed_rungs=['rung_i_shakeout', 'rung_ii_bakeoff', 'rung_iii_context']``. Rung i finished at
``final_loss=3.2497``, ``forgetting_gap=0.2642``, ``mfu_mean=1.519e-01`` over 30 MFU samples.
Rung ii, which opted into the F9 (muP width transfer) and H2 (MoE-vs-dense) receipts, actually
exercised both: ``F9_mup_transfer`` recorded a transferred learning rate
(``base_lr=0.05, base_width=8, target_width=8, transferred_lr=0.05`` -- a no-op transfer here
because the rung's base and target widths are equal by construction, but the same
:func:`mixle.models.mup.transfer_lr` call a real width change would use), and
``H2_moe_vs_dense`` recorded a measured ``relative_output_diff=0.1643`` against the rung's
``moe_max_relative_diff=1.0`` threshold, yielding ``decision="moe"``. The whole run's decision
journal has 3 entries and ``journal.verify() is True`` -- every belief update is a real, tamper-
evident record, not prose.

Honest About What Is Not Reachable Here
--------------------------------------------

Every rung names the roadmap sub-pieces it depends on in ``decision_pieces``. When a piece is
merged into this branch's base, the rung wires it in for real (F1, F4, F9, H2 above). When a piece
lives on a branch that has not merged, or does not exist yet anywhere in the repository, the rung
records that honestly in ``artifacts.skipped_pieces`` instead of silently no-op-ing. Rung ii's
run above recorded, verbatim:

.. code-block:: text

   skipped_pieces["E7"] = "the E7 referee evaluation suite does not exist yet anywhere in this
   repository (see mixle/experimental/README.md's graduation rule); this rung ran without an E7
   bake-off."

and rung iii's run recorded both of its named-but-unreachable pieces:

.. code-block:: text

   skipped_pieces["E8"] = "E8 is a later long-context roadmap item that has not been built yet;
   this rung ran without it. (F1's TP/PP/CP, PR #171, already covers context parallelism as a
   separate roadmap item ... it is not the same thing as E8.)"

   skipped_pieces["F5"] = "scaling-law fits (roadmap F5) live on origin/scaling-law-fits, not
   reachable from this worktree's base; this rung did not fit a scaling law and used a
   manually-chosen stand-in configuration instead of one F5 would have chosen."

If a caller explicitly opts a rung into one of these via ``exercise_fault_tolerance=True``,
``exercise_eval_suite=True``, ``exercise_context_parallel=True``, or
``exercise_scaling_law_fit=True``, ``run_pilot_ladder`` raises ``NotImplementedError`` rather than
quietly skipping -- an explicit request for something unavailable is a bug to surface, not a
silent downgrade.

Detecting An Anomaly
-----------------------

``TrainingHealthMonitor`` scores every step's loss and grad-norm against a robust rolling
(median/MAD) baseline, plus unconditional NaN/Inf checks that fire even during the baseline's
warmup window. Injecting a loss spike into an otherwise stable run --

.. code-block:: python

   from mixle.utils.parallel.training_health import TrainingHealthMonitor

   monitor = TrainingHealthMonitor(loss_window=10, loss_min_periods=5, loss_z_thresh=6.0)
   for step, loss in enumerate([3.0, 3.05, 2.95, 3.02, 2.98, 3.01]):
       monitor.observe_step(step, loss)

   anomalies = monitor.observe_step(6, 50.0)   # a deliberate spike

-- flags it the same step it happens: this exact run produced ``detected_step=6`` for
``injected_step=6`` (latency 0) with ``kind="loss_spike"`` and ``z_score=1584.885`` against the
stable baseline. The rolling baseline is causal -- each step is scored against the window as it
stood *before* that step, so a checkpoint restart that silently drops optimizer/RNG state and
produces a real loss jump is caught as ``restart_discontinuity`` on the very next step (see
``RestartContinuityTest`` in the receipts table below), while a well-behaved resume that continues
the same trend is not flagged at all.

When A Rung Fails: The Halt Case
-------------------------------------

The gate is real, not decorative. Giving a rung an unachievable target (``max_final_loss=1e-6``
in 10 steps on a laptop-sized toy model) halts the ladder at that rung and never attempts the
ones after it:

.. code-block:: python

   rungs = [
       Rung(name="rung_i_shakeout", real_target="1B params / 8k context / 8 GPUs",
            decision_pieces=("F1", "F4"), max_final_loss=5.0),
       Rung(name="rung_ii_impossible", real_target="8B params / 128k context / 256 GPUs",
            decision_pieces=("H2",), max_final_loss=1.0e-6, steps=10),
       Rung(name="rung_iii_unreached", real_target="8B params / 10M context / 1000 GPUs",
            decision_pieces=("E8", "F5")),
   ]
   result = run_pilot_ladder(rungs, peak_flops_per_sec=1.0e9)

Running this produced ``halted_at="rung_ii_impossible"`` with only 2 outcomes recorded (rung iii
was never attempted). The second outcome's ``reason`` is the literal measured margin:
``"final loss 5.8563 > target 1e-06"``, and its journal entry's ``action_chosen`` is
``"halt_ladder"`` -- the same decision-journal machinery that records an advance also records a
halt, with the number that caused it.

Where Each Receipt Lives
----------------------------

.. list-table::
   :header-rows: 1

   * - Claim
     - Test file
   * - Theoretical FLOPs formula matches a hand-computed reference; position-embedding params
       excluded from the count
     - ``mixle/tests/training_health_test.py`` (``TheoreticalFlopsTest``)
   * - MFU is exactly ``achieved_flops_per_sec / peak_flops_per_sec``; the monitor tracks it from
       real wall-clock timing of a real model
     - ``mixle/tests/training_health_test.py`` (``MFURatioTest``)
   * - The rolling baseline is causal (no leakage of the current point into its own score) and
       round-trips through ``state()``/``from_state()`` for checkpoint continuity
     - ``mixle/tests/training_health_test.py`` (``RollingBaselineTest``)
   * - Injected loss spikes, grad-norm spikes, NaN loss, and Inf grad-norm are each flagged within
       0-1 steps, including inside a real training loop
     - ``mixle/tests/training_health_test.py`` (``InjectedAnomalyDetectionTest``)
   * - A well-behaved checkpoint restart passes; a restart that silently drops
       optimizer/RNG state and produces a real loss jump is flagged as
       ``restart_discontinuity``
     - ``mixle/tests/training_health_test.py`` (``RestartContinuityTest``)
   * - ``report()`` is complete and JSON-serializable: step count, MFU stats, anomalies by kind,
       restart continuity verdict
     - ``mixle/tests/training_health_test.py`` (``ReportSmokeTest``)
   * - The ladder collects MFU/loss-curve/forgetting-curve artifacts per rung, exercises F9/H2 for
       real when a rung opts in, and journals one tamper-evident entry per rung
     - ``mixle/tests/pilot_ladder_test.py`` (``PilotLadderOrchestrationTest``)
   * - A rung that fails its own GO/NO-GO criteria halts the ladder before the next rung runs
     - ``mixle/tests/pilot_ladder_test.py``
       (``test_gate_halts_the_ladder_at_a_rung_that_fails_its_own_criteria``)
   * - Opting a rung into an unreachable piece (F2, F5, E7, E8) raises ``NotImplementedError``
       rather than silently skipping
     - ``mixle/tests/pilot_ladder_test.py``
       (``test_unavailable_piece_raises_notimplementederror_not_a_silent_skip``,
       ``test_unavailable_fault_tolerance_opt_in_raises``)
   * - The decision journal replays the whole ladder run and is tamper-evident (mutating a stored
       snapshot breaks ``verify()``)
     - ``mixle/tests/pilot_ladder_test.py`` (``PilotLadderJournalIntegrityTest``)

See Also
------------

* :doc:`utilities-and-parallelism` -- the wider ``mixle.utils.parallel`` surface (checkpointing,
  torchrun helpers, model decomposition) that ``TrainingHealthMonitor`` and the pilot ladder build
  on top of.
* :doc:`neural-llm` -- ``GradLeaf``/``NeuralDensity`` and the rest of the torch-native surfaces
  the pilot ladder's tiny models are built through.
* :doc:`bring_your_own_model` -- the interop pattern for an externally-trained checkpoint, as
  distinct from a model trained in-process the way every pilot-ladder rung is here.
