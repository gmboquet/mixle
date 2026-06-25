"""Pytest collection policy for the legacy unittest suite.

The tests are still ordinary ``unittest.TestCase`` tests.  Pytest is used as
the collection and CI harness so we can attach stable markers without rewriting
hundreds of existing tests at once.
"""

from collections.abc import Iterable
from pathlib import Path

import pytest

MarkerTuple = tuple[str, ...]


FILE_MARKERS: dict[str, MarkerTuple] = {
    # Heavy integration / PDE-inversion / stochastic-recovery / exhaustive-precision tests: each has a
    # call >~5 s and together they floor the fast gate at ~100 s. Tagged `slow` so they leave the fast
    # gate (`pytest -m fast` -> <30 s) while still running in the full CI gate (`-m "not optional ..."`).
    "bingham_test.py": ("distribution", "stochastic", "slow"),
    "conformal_test.py": ("ppl", "integration", "slow"),
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
    "mixture_heterogeneous_test.py": ("distribution", "latent"),
    "numerics_test.py": ("distribution",),
    "numerical_guards_test.py": ("distribution", "bayes"),
    "objectives_test.py": ("torch", "optional"),
    "parallel_test.py": ("parallel", "integration", "slow"),
    "placement_test.py": ("parallel", "planner"),
    "ppl_separation_test.py": ("ppl",),
    "random_graph_models_test.py": ("graph",),
    "quantized_hmm_test.py": ("hmm", "integration", "slow"),
    "quantized_index_test.py": ("enumeration",),
    "sampler_accuracy_test.py": ("distribution", "stochastic", "slow"),
    "sampler_seed_test.py": ("distribution", "stochastic"),
    "segmental_hmm_test.py": ("hmm", "integration"),
    "serialization_test.py": ("serialization",),
    "sparse_markov_transform_test.py": ("latent",),
    "spearman_rho_test.py": ("distribution",),
    "spark_encoded_data_test.py": ("spark", "optional", "parallel", "slow"),
    "torchrun_encoded_data_test.py": ("torchrun", "torch", "optional", "parallel", "slow"),
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
}


NODEID_MARKERS: tuple[tuple[str, MarkerTuple], ...] = (
    ("Benchmark", ("benchmark", "slow")),
    ("Torch", ("torch", "optional")),
    ("MPIBackend", ("mpi", "optional", "parallel")),
    ("MPS", ("torch", "optional")),
    ("numba", ("numba",)),
    ("umap", ("optional", "hvis")),
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
    """
    import numpy as np

    from pysp.engines.arithmetic import get_default_engine, set_default_engine

    engine = get_default_engine()
    rng_state = np.random.get_state()
    err_mode = np.geterr()
    try:
        yield
    finally:
        set_default_engine(engine)
        np.random.set_state(rng_state)
        np.seterr(**err_mode)
