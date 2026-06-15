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
    "htsne_test.py": ("htsne", "integration", "slow"),
    "ibp_test.py": ("distribution", "latent"),
    "int_hidden_association_test.py": ("latent",),
    "kernels_ext_test.py": ("kernel", "integration"),
    "kernels_test.py": ("kernel", "integration"),
    "lda_len_test.py": ("latent",),
    "llda_alpha_test.py": ("latent", "integration"),
    "lookback_lag0_test.py": ("hmm",),
    "model_helpers_test.py": ("latent", "graph", "pomdp", "knowledge_graph", "causal", "grammar"),
    "mcmc_test.py": ("stochastic",),
    "numerics_test.py": ("distribution",),
    "objectives_test.py": ("torch", "optional"),
    "parallel_test.py": ("parallel", "integration", "slow"),
    "placement_test.py": ("parallel", "planner"),
    "random_graph_models_test.py": ("graph",),
    "quantized_hmm_test.py": ("hmm", "integration"),
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
    ("umap", ("optional", "htsne")),
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
