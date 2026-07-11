"""Pytest collection policy for the legacy unittest suite.

The tests are still ordinary ``unittest.TestCase`` tests.  Pytest is used as
the collection and CI harness so we can attach stable markers without rewriting
hundreds of existing tests at once.
"""

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

# Force reproducible CPU math for torch-training tests. Multi-threaded matmuls are not bit-reproducible, so tests
# with tight parameter-recovery thresholds pass in isolation but flake under the parallel runner (the flake hops
# to whichever threshold is tightest). torch.set_num_threads(1) alone doesn't cover the MKL/OpenMP BLAS pool --
# those honor these env vars, which must be set before the first torch/numpy import (this runs at collection
# start, before any test body). The per-test fixture below adds torch's own determinism knobs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

MarkerTuple = tuple[str, ...]


FILE_MARKERS: dict[str, MarkerTuple] = {
    # Heavy integration / PDE-inversion / stochastic-recovery / exhaustive-precision tests: each has a
    # call >~5 s and together they floor the fast gate at ~100 s. Tagged `slow` so they leave the fast
    # gate (`pytest -m fast` -> <30 s) while still running in the full CI gate (`-m "not optional ..."`).
    # Torch-trained conditional-transport + belief-composition tests: each fits a neural conditional
    # density and/or runs a Monte-Carlo calibration check (100s of samples per held-out point across
    # multiple hop counts) -- legitimately needed for a real coverage/calibration claim, not padding.
    # Unmarked, these alone floor the fast gate at 60+ s; tagged `slow` so they still run in full CI.
    "belief_walk_test.py": ("torch", "stochastic", "slow"),
    "cycle_consistency_test.py": ("torch", "stochastic", "slow"),
    "cross_modal_model_test.py": ("torch", "stochastic", "slow"),
    "transport_edge_test.py": ("torch", "stochastic", "slow"),
    "transport_proof_test.py": ("torch", "stochastic", "slow"),
    "task_extract_test.py": ("torch", "slow"),
    "data_mixture_test.py": ("torch", "slow"),
    "task_constrained_test.py": ("torch", "slow"),
    "task_sft_plan_test.py": ("torch", "slow"),
    "task_plan_refine_test.py": ("torch", "slow"),
    "bingham_test.py": ("distribution", "stochastic", "slow"),
    "conformal_test.py": ("ppl", "integration", "slow"),
    "fused_codegen_test.py": ("numba", "optional"),
    "jax_engine_test.py": ("jax", "optional"),
    "fused_em_hmm_family_test.py": ("hmm", "integration", "slow"),
    "fused_em_variational_test.py": ("latent", "integration", "slow"),
    "hmm_sampler_batching_test.py": ("hmm", "stochastic", "slow"),
    "infer_parallel_chains_test.py": ("parallel", "stochastic", "slow"),
    "lookback_terminal_states_test.py": ("hmm", "integration", "slow"),
    "ppl_engine_test.py": ("ppl", "integration", "slow"),
    "graph_distribution_test.py": ("graph", "integration", "slow"),
    "hyperedge_replacement_grammar_test.py": ("graph", "pcfg", "slow"),
    "hawkes_process_test.py": ("distribution", "stochastic", "slow"),
    "hmm_terminal_states_test.py": ("hmm", "integration", "slow"),
    "scheduled_hmm_test.py": ("hmm", "integration", "slow"),
    "program_test.py": ("torch", "integration", "slow"),
    "neural_leaf_test.py": ("torch", "integration", "slow"),
    "multimodal_stage1_demo_smoke_test.py": ("torch", "integration", "slow"),
    "vlm_trust_receipts_demo_smoke_test.py": ("torch", "integration", "slow"),
    "neural_ppl_test.py": ("torch", "integration", "slow"),
    "language_model_sft_test.py": ("torch", "integration", "slow"),
    "project_neural_test.py": ("torch", "integration", "slow"),
    "reason_adapter_test.py": ("torch", "integration", "slow"),
    "reason_fusion_test.py": ("torch", "integration", "slow"),
    "structure_embedded_test.py": ("torch", "integration", "slow"),
    "estimation_structure_default_test.py": ("integration", "slow"),
    "scientist_test.py": ("torch", "integration", "optional", "slow"),
    # Downloads a real tiny HF checkpoint (peft/transformers, network-gated) and fits it -- skips cleanly
    # offline, but is neither fast nor a core-path test.
    "peft_lora_grad_leaf_smoke_test.py": ("torch", "integration", "optional", "slow"),
    "planning_test.py": ("planner",),
    "uq_test.py": ("torch", "integration", "slow"),
    "infer_backends_test.py": ("numba", "integration", "slow"),
    "kent_test.py": ("distribution", "stochastic", "slow"),
    "knowledge_graph_test.py": ("graph", "stochastic", "slow"),
    "lkj_test.py": ("distribution", "stochastic", "slow"),
    "matching_test.py": ("graph", "stochastic", "slow"),
    "max_stable_test.py": ("distribution", "stochastic", "slow"),
    "nuts_mass_adaptation_test.py": ("stochastic", "slow"),
    "nuts_torch_test.py": ("torch", "optional", "stochastic", "slow"),
    "ppl_composite_sampling_test.py": ("ppl", "stochastic", "slow"),
    "ppl_relabel_test.py": ("ppl", "stochastic", "slow"),
    "ppl_hierarchical_test.py": ("ppl", "stochastic", "slow"),
    "ppl_plate_test.py": ("ppl", "stochastic", "slow"),
    "ppl_convergence_diagnostics_test.py": ("ppl", "stochastic", "slow"),
    "ppl_inference_test.py": ("ppl", "stochastic", "slow"),
    "ppl_new_distributions_test.py": ("ppl", "stochastic"),
    "ppl_predictive_test.py": ("ppl", "stochastic"),
    "ppl_survival_test.py": ("ppl", "stochastic"),
    "ppl_summarize_test.py": ("ppl", "stochastic"),
    "ppl_leaf_families_test.py": ("ppl", "stochastic", "slow"),
    "ppl_vector_params_test.py": ("ppl", "stochastic", "slow"),
    "reflective_hmc_test.py": ("stochastic", "slow"),
    "segmental_terminal_states_test.py": ("hmm", "integration", "slow"),
    "semi_supervised_terminal_states_test.py": ("hmm", "integration", "slow"),
    "spanning_tree_test.py": ("graph", "stochastic", "slow"),
    "survival_test.py": ("distribution", "integration", "slow"),
    "thompson_acquisition_test.py": ("doe", "stochastic", "slow"),
    "doe_factorial_test.py": ("doe",),
    "doe_batch_test.py": ("doe",),
    "doe_entropy_test.py": ("doe",),
    "doe_turbo_test.py": ("doe", "slow"),
    "doe_active_test.py": ("doe", "slow"),
    "doe_multifidelity_test.py": ("doe", "slow"),
    "doe_mixture_test.py": ("doe",),
    "doe_analysis_test.py": ("doe",),
    "doe_criteria_sensitivity_test.py": ("doe",),
    "automatic_scientific_test.py": ("automatic", "integration", "slow"),
    "automatic_test.py": ("automatic", "distribution"),
    "base_dist_test.py": ("distribution", "integration", "slow"),
    # Volume-scale performance receipts (roadmap A7): GradLeaf/mixture-E-step/A3-patch-streamed
    # timing receipts, each printing a measured wall-clock number against a generous pinned floor.
    "bench_receipts_test.py": ("benchmark", "slow"),
    "bayes_test.py": ("bayes", "distribution", "integration"),
    "bayes_streaming_test.py": ("bayes",),
    "wave_bayes1_test.py": ("bayes",),
    "wave_bayes2_test.py": ("bayes",),
    "wave_bayes3_test.py": ("bayes",),
    "wave_bayes4_test.py": ("bayes",),
    "stats_bayes_gaussian_test.py": ("bayes", "distribution"),
    "stats_bayes_gamma_group_test.py": ("bayes", "distribution"),
    "stats_bayes_beta_group_test.py": ("bayes", "distribution"),
    "stats_bayes_dirichlet_group_test.py": ("bayes", "distribution"),
    "stats_bayes_mvgaussian_group_test.py": ("bayes", "distribution"),
    "stats_bayes_mixture_test.py": ("bayes", "latent"),
    "stats_bayes_markov_test.py": ("bayes", "hmm"),
    "stats_bayes_setdist_test.py": ("bayes", "distribution"),
    "stats_bayes_wrappers_test.py": ("bayes",),
    "stats_bayes_dpm_test.py": ("bayes", "latent"),
    "objective_resolution_test.py": ("bayes",),
    "categorical_test.py": ("distribution",),
    "chow_liu_tree_test.py": ("distribution", "latent"),
    "distribution_additions_test.py": ("distribution",),
    "dask_encoded_data_test.py": ("dask", "optional", "parallel", "slow"),
    "density_rank_test.py": ("enumeration",),
    "em_strategies_test.py": ("em",),
    "enumeration_test.py": ("enumeration", "distribution"),
    "enumerator_coverage_test.py": ("enumeration",),
    "estimator_stability_test.py": ("distribution",),
    "fisher_view_test.py": ("fisher", "integration", "slow"),
    "gradient_fit_test.py": ("torch", "optional"),
    "heterogeneous_pcfg_test.py": ("pcfg", "integration", "slow"),
    "hidden_association_keys_test.py": ("latent",),
    "hmm_keys_test.py": ("hmm",),
    "hmm_zero_prob_test.py": ("hmm", "numba"),
    "tree_hmm_zero_prob_test.py": ("hmm", "numba"),
    "hvis_test.py": ("hvis", "integration", "slow"),
    "ibp_test.py": ("distribution", "latent"),
    "int_hidden_association_test.py": ("latent",),
    "kernels_ext_test.py": ("kernel", "integration", "slow"),
    "kernels_test.py": ("kernel", "integration"),
    "lda_len_test.py": ("latent",),
    "llda_alpha_test.py": ("latent", "integration"),
    "lookback_lag0_test.py": ("hmm",),
    "marginal_seek_test.py": ("enumeration",),
    "model_helpers_test.py": ("latent", "graph", "pomdp", "knowledge_graph", "causal", "grammar"),
    "mcmc_test.py": ("stochastic",),
    "responsibility_attention_test.py": ("distribution", "latent", "stochastic"),
    "chained_attention_test.py": ("distribution", "latent", "stochastic", "slow"),
    "variational_multihop_attention_test.py": ("distribution", "latent", "stochastic", "slow"),
    "variational_embedding_attention_test.py": ("distribution", "latent", "stochastic", "slow"),
    "glm_test.py": ("distribution",),
    "model_comparison_test.py": ("distribution",),
    "conformal_array_test.py": ("distribution", "stochastic"),
    "measurement_error_test.py": ("distribution", "stochastic"),
    "kriging_test.py": ("distribution", "stochastic"),
    "kde_test.py": ("distribution", "stochastic"),
    "extreme_value_test.py": ("distribution", "stochastic"),
    "ordinal_test.py": ("distribution",),
    "decomposition_test.py": ("distribution",),
    "nonparametric_test.py": ("distribution",),
    "rank_aggregation_test.py": ("distribution",),
    "permutation_kernels_test.py": ("distribution", "numba"),
    "bradley_terry_test.py": ("distribution", "numba", "stochastic"),
    "low_rank_permutation_test.py": ("distribution", "numba", "stochastic"),
    "thurstone_test.py": ("distribution", "numba", "stochastic"),
    "paired_comparison_test.py": ("distribution", "stochastic"),
    "ewens_test.py": ("distribution", "numba", "stochastic"),
    "generalized_mallows_test.py": ("distribution", "numba", "stochastic"),
    "generalized_mallows_model_test.py": ("distribution", "numba", "stochastic"),
    "survival_regression_test.py": ("distribution", "stochastic", "slow"),
    "scoring_rules_test.py": ("distribution",),
    "calibration_diagnostics_test.py": ("distribution",),
    "multiple_testing_test.py": ("distribution",),
    "resampling_test.py": ("distribution", "stochastic"),
    "robust_covariance_test.py": ("distribution",),
    "cross_validation_test.py": ("distribution",),
    "coverage_estimation_test.py": ("distribution",),
    "mixture_heterogeneous_test.py": ("distribution", "latent"),
    "numerics_test.py": ("distribution",),
    "numerical_guards_test.py": ("distribution", "bayes"),
    "objectives_test.py": ("torch", "optional"),
    "parallel_test.py": ("parallel", "integration", "slow"),
    "placement_test.py": ("parallel", "planner"),
    "model_decomposition_test.py": ("parallel", "planner"),
    "model_parallel_test.py": ("parallel", "planner"),
    "ppl_separation_test.py": ("ppl",),
    "random_graph_models_test.py": ("graph",),
    "quantized_hmm_test.py": ("hmm", "integration", "slow"),
    "quantized_triangular_hmm_test.py": ("hmm", "enumeration", "integration", "slow"),
    "hmm_determinize_test.py": ("hmm", "enumeration", "integration"),
    "missing_data_test.py": ("distribution", "hmm", "ppl", "integration", "slow"),
    "provenance_test.py": ("distribution", "serialization"),
    "drift_test.py": ("distribution", "doe", "stochastic"),
    "serving_test.py": ("distribution", "serialization"),
    "checkpoint_test.py": ("distribution", "latent", "serialization"),
    "lineage_test.py": ("distribution", "latent", "serialization"),
    "ppl_provenance_test.py": ("ppl", "serialization"),
    "quantized_index_test.py": ("enumeration",),
    "sampler_accuracy_test.py": ("distribution", "stochastic", "slow"),
    "sampler_seed_test.py": ("distribution", "stochastic"),
    "segmental_hmm_test.py": ("hmm", "integration"),
    "serialization_test.py": ("serialization",),
    "sparse_markov_transform_test.py": ("latent",),
    "spearman_rho_test.py": ("distribution",),
    "spark_encoded_data_test.py": ("spark", "optional", "parallel", "slow"),
    "spark_executor_test.py": ("spark", "optional", "parallel", "slow"),
    "ray_encoded_data_test.py": ("ray", "optional", "parallel", "slow"),
    "torchrun_encoded_data_test.py": ("torchrun", "torch", "optional", "parallel", "slow"),
    "torch_neural_test.py": ("torchrun", "torch", "optional", "parallel", "slow"),
    "torch_engine_ext_test.py": ("torch", "optional", "integration", "slow"),
    "torch_engine_test.py": ("torch", "optional", "integration", "slow"),
    "tree_hmm_len_test.py": ("hmm", "numba"),
    "utils_test.py": ("distribution",),
    "vmf_test.py": ("distribution", "stochastic"),
    "wave_core_test.py": ("distribution", "enumeration"),
    "wave_hmmlegacy_test.py": ("hmm",),
    "wave_latent_test.py": ("latent", "enumeration"),
    "wave_lookback_test.py": ("hmm",),
    "wave_markov_test.py": ("pcfg", "latent"),
    "wave_multinomial_enum_test.py": ("distribution", "enumeration"),
    "wave_mvn_test.py": ("distribution",),
    "wave_select_test.py": ("distribution", "enumeration", "latent"),
    "wave_setdist_test.py": ("distribution", "enumeration"),
    "zero_count_estimate_test.py": ("distribution",),
    # Newer files never triaged into this registry: each floors well above the ~1s/test the fast gate
    # is built around (profiled via `pytest -m fast --durations=40`, individual calls 8-105s), so the
    # fast gate silently regressed by ~9 minutes of wall time across all three of them as these landed.
    # Tagged the same way their nearest existing sibling already is.
    "doe_robust_test.py": ("doe", "stochastic", "slow"),  # 10-seed BO loop averaged for a noise claim
    "quotient_leaf_test.py": ("torch", "integration", "slow"),  # conv+pool leaf, fits/compares real nets
    "ppl_guide_test.py": ("ppl", "stochastic", "slow"),  # structured VI, admixture/LDA recovery
    "structure_learning_test.py": ("integration", "slow"),  # multi-restart EM structure search
    "task_traces_test.py": ("integration", "slow"),
    "temporal_graph_grammar_test.py": ("graph", "stochastic", "slow"),
    "anchor_harness_test.py": ("torch", "stochastic", "slow"),  # neural conditional-transport + calibration
    "edge_distill_test.py": ("torch", "integration", "slow"),
    "structured_hmm_test.py": ("hmm", "integration", "slow"),
    "task_realteacher_smoke_test.py": ("integration", "slow"),
    "task_model_test.py": ("integration", "slow"),  # fresh-process save/load round trip
    "symbolic_export_test.py": ("integration", "slow"),
    "task_tune_test.py": ("doe", "stochastic", "slow"),  # BO over student recipes via mixle.doe
    "task_plan_test.py": ("integration", "slow"),
    "task_distill_structured_test.py": ("integration", "slow"),
    # E1 chunked-recurrent spine (mixle/experimental/context_spine.py): several small TBPTT training
    # loops plus a repeated-timing receipt -- ~4s total, tagged slow so it leaves the fast gate while
    # still running in full CI under the `experimental` marker's own tests.
    "context_spine_test.py": ("torch", "experimental", "slow"),
    # E7 long-context referee (mixle/experimental/long_context_eval.py): needle/copy/multi-hop suites x
    # small stand-in ranges x a length-curriculum bandit round -- several TBPTT training loops, tagged
    # slow for the same reason as context_spine_test.py.
    "long_context_eval_test.py": ("torch", "experimental", "slow"),
    # E6 retrieval memory over frozen past (mixle/experimental/retrieval_memory_spine.py): several TBPTT
    # training loops (including a needle-suite baseline comparison), tagged slow for the same reason as
    # context_spine_test.py / long_context_eval_test.py.
    "retrieval_memory_spine_test.py": ("torch", "experimental", "slow"),
    # E3 sketch-state attention (mixle/experimental/sketch_state_attention.py): FD's deterministic bound
    # over several seeded streams, chunked-scan equivalence, ContextMechanism conformance via train_tbptt,
    # a tensor-sketch concentration check, misfit receipts, and an E7 bake-off across four mechanisms --
    # tagged slow for the same reason as context_spine_test.py/long_context_eval_test.py.
    "sketch_state_attention_test.py": ("torch", "experimental", "slow"),
    # E2 moment-closure (mixture-state) attention (mixle/experimental/moment_closure_attention.py):
    # gradcheck, Welford-vs-batch, birth/merge, TBPTT protocol conformance, an E7 referee smoke test, and
    # a real (multi-model-training) Spearman correlation measurement -- tagged slow for the same reason as
    # context_spine_test.py / long_context_eval_test.py.
    "moment_closure_attention_test.py": ("torch", "experimental", "slow"),
    # E5 part 1: S6/Mamba selective scan (mixle/experimental/selective_scan.py) -- protocol conformance,
    # detach, log_density, and a real 3000-step TBPTT Selective Copying training receipt -- tagged slow for
    # the same reason as context_spine_test.py / moment_closure_attention_test.py.
    "selective_scan_test.py": ("torch", "experimental", "slow"),
    # E5 part 2: the hybrid block (mixle/experimental/ssm_hybrid.py) -- local attention + SSM + E2 far
    # field, protocol conformance, contribution-receipt bookkeeping, and a real matched-parameter E7
    # referee-suite comparison against local-only and SSM-only ablations -- several TBPTT training loops,
    # tagged slow for the same reason as the other Track-E mechanism tests.
    "ssm_hybrid_test.py": ("torch", "experimental", "slow"),
    # E4 hierarchical summary tree (mixle/experimental/summary_tree.py): needle-suite training runs
    # over several seeds (needle receipt + auxiliary-loss ablation) plus a re-chunking topology check
    # -- several TBPTT training loops, tagged slow for the same reason as context_spine_test.py.
    "summary_tree_test.py": ("torch", "experimental", "slow"),
    # E8 context parallelism (mixle/utils/parallel/context_parallel_spine.py): parametrized exact-match
    # correctness sweeps plus a real torch.multiprocessing.spawn/gloo 4-process test -- tagged slow so it
    # leaves the fast gate while still running in full CI under the `parallel` marker's own tests.
    "context_parallel_spine_test.py": ("torch", "experimental", "parallel", "slow"),
    # GP-surrogate active-learning + multi-fidelity placement (roadmap M4): each test fits several torch
    # GPs across a sequential design loop, mirroring doe_active_test.py / doe_multifidelity_test.py.
    "task_emulate_test.py": ("doe", "torch", "slow"),
    # P11 certified bounds (mixle/experimental/certified_bounds.py): interval-propagation soundness +
    # monotonicity certificates validated against dense grids -- pure numpy, no torch.
    "certified_bounds_test.py": ("experimental",),
}


NODEID_MARKERS: tuple[tuple[str, MarkerTuple], ...] = (
    ("Benchmark", ("benchmark", "slow")),
    ("Torch", ("torch", "optional")),
    ("MPIBackend", ("mpi", "optional", "parallel")),
    ("MPS", ("torch", "optional")),
    ("numba", ("numba",)),
    ("umap", ("optional", "hvis")),
    # PeakRssPatchStreamingTest writes+reads a ~476 MiB synthetic zarr volume to exercise the A3
    # patch-streaming peak-RSS receipt; the rest of array_data_sources_test.py stays in the fast gate.
    ("PeakRssPatchStreamingTest", ("optional", "slow")),
    # G2's real-perplexity acceptance test trains several real small LMs end to end (not a layer-local
    # proxy) to get an honest, independent perplexity number -- multiple real training runs are the point,
    # so it's slow by construction; the rest of sigma_weighted_projection_test.py stays in the fast gate.
    ("DataFreeSigmaBeatsPlainSvdTest", ("slow",)),
)


def _add_markers(item: pytest.Item, names: Iterable[str], assigned: set[str]) -> None:
    for name in names:
        item.add_marker(getattr(pytest.mark, name))
        assigned.add(name)


def pytest_collection_modifyitems(items) -> None:
    """Apply subsystem and tier markers during collection.

    Any test that is not slow, optional, or benchmark-oriented is marked
    ``fast``.  That keeps the fast CI command stable as new tests are added:
    either the file/test is explicitly marked as heavier, or it joins the fast
    gate automatically.
    """
    for item in items:
        assigned: set[str] = set()
        filename = Path(str(item.fspath)).name
        _add_markers(item, FILE_MARKERS.get(filename, ()), assigned)

        for token, marker_names in NODEID_MARKERS:
            if token in item.nodeid:
                _add_markers(item, marker_names, assigned)

        if not {"slow", "optional", "benchmark"} & assigned:
            _add_markers(item, ("fast",), assigned)


@pytest.fixture(autouse=True)
def _isolate_global_process_state():
    """Snapshot and restore process-global state around every test.

    Some tests set the default compute engine, consume the global numpy RNG, or change the numpy error
    mode. Without isolation those leaks make *other* tests order-dependent under the parallel runner --
    e.g. a ``fit()``-based parameter-recovery test that is deterministic in isolation but flakes only in
    the full suite because an earlier test left a non-numpy default engine in place. Restoring the
    default engine, the numpy RNG state, and the numpy error mode after each test closes that hole.

    Torch globals get the same treatment. The default dtype is restored (a test that sets float64 and
    leaks it makes every float32-module test scheduled after it on that worker die with Float/Double
    matmul errors -- a different victim set each run, depending on how xdist distributed the files).
    The global torch RNG is re-seeded before each test and restored after, so ambient torch draws
    depend neither on the worker's entropy-derived base seed nor on which tests ran earlier in the
    process: a test passes or fails the same way on every worker, in every order, on every run.
    """
    import numpy as np

    from mixle.engines.arithmetic import get_default_engine, set_default_engine

    engine = get_default_engine()
    rng_state = np.random.get_state()
    err_mode = np.geterr()
    torch_globals = None
    try:
        import torch  # force single-threaded + deterministic before this test runs (a prior test may have changed it)

        torch.set_num_threads(1)  # (multi-threaded CPU matmuls aren't bit-reproducible -> training-threshold flakes)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch_globals = (torch.get_default_dtype(), torch.random.get_rng_state())
        torch.manual_seed(0)  # fixed ambient stream: unseeded torch draws are identical on every worker, every run
    except Exception:  # noqa: BLE001
        pass
    try:
        yield
    finally:
        set_default_engine(engine)
        np.random.set_state(rng_state)
        np.seterr(**err_mode)
        if torch_globals is not None:
            import torch

            torch.set_default_dtype(torch_globals[0])
            torch.random.set_rng_state(torch_globals[1])
