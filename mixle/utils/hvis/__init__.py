"""Model-based (hierarchical) t-SNE and UMAP for heterogeneous data.

Pairwise affinities are derived from a fitted mixture model rather than from
Euclidean distances, so anything mixle can model (tuples, sequences,
sets, variable-length data, ...) can be embedded. Six affinity definitions
are supported (the `affinity` argument):

- 'local' (the 'auto' default whenever raw data is available): the model is
  flattened into leaf fields and each field contributes a local statistical
  affinity combining the per-field posterior (between-cluster structure)
  with a component-local Mahalanobis metric (within-cluster structure).
  Continuous/count fields use their native coordinates; every other leaf --
  HMMs, Markov chains, categoricals, sequence-of-discrete element fields --
  uses typicality coordinates (per-component log-density; per-token rate
  plus a log-length axis for sequence-valued leaves), so no field type
  degrades to posterior-only geometry. Thus the same component is never a
  zero-distance quotient: sharp posteriors stop collapsing clusters into
  tiny structureless points, variable-length fields keep length as one
  honest axis instead of the dominant one, and mixed continuous/discrete
  fields are made commensurate by the per-component whitening. See
  affinity_health() for measurable receipts of these degeneracies.

- 'balanced': the model
  is flattened into its leaf fields (nested composites, sequence
  element/length models, and optional wrappers all decompose), a
  field-restricted posterior z^f is computed from each field's likelihoods
  alone, and the pair distance is the sum over fields of per-field
  Bhattacharyya distances -log sum_k sqrt(z^f_ik z^f_jk), each Winsorized at
  `evidence_cap` nats. The per-field posteriors keep every field's structure
  visible regardless of its likelihood scale (by default, a 15-token sequence
  field contributes summed sequence evidence while length is a separate field;
  if the sequence model was explicitly fit with len_normalized=True, the
  sequence field instead contributes a per-token composition quotient),
  and the cap bounds each field's influence so one spuriously sharp field
  cannot veto a pair's similarity that every other field supports.

- 'fisher': each observation is mapped through the model's to_fisher() view to
  posterior-expected sufficient statistics and, by default, whitened by the
  empirical observed Fisher covariance of those score vectors. Pair affinities
  are Gaussian in that Fisher-vector space, so htsne can use the same
  sufficient-statistic geometry exposed to downstream tools.

- 'bhattacharyya': the Bhattacharyya coefficient between joint posteriors,
  s_ij = sum_k sqrt(z_ik z_jk); -log s_ij is the Bhattacharyya distance on
  the posterior simplex. The square root amplifies shared low-probability
  components, so affinities stay *graded* even when hard assignments
  coincide - which is what gives the embedding within-cluster geometry. Like
  'coassign', it depends on the data only through posteriors, so
  variable-length observations need no adjustments.

- 'coassign': the co-assignment probability

      s_ij = P(z_i = z_j | x_i, x_j) = sum_k z_ik z_jk,

  the posterior similarity matrix of Bayesian clustering - an exact
  probability under the fitted model. The principled choice when the
  affinity itself must be a probability, but near-deterministic posteriors
  make it almost binary: every same-component pair ties at ~1, and t-SNE
  renders tied groups as rings/blobs with no internal structure.

- 'likelihood': the predictive affinity s_ij = sum_k p(x_i | theta_k) z_jk
  (likelihood of x_i under the posterior mixture of x_j). Retains within-
  component likelihood detail, but for variable-length data the evidence in
  x_i grows with its length, so long observations reduce to their single best
  component while short ones stay blended.

For t-SNE the affinities are converted to input probabilities by
row-conditional normalization p_{j|i} = softmax_j(log s_ij), optionally
calibrated to a target perplexity per row, and symmetrized
P = (P + P^T) / (2n).

Two t-SNE engines are provided:

- 'exact': a full-matrix gradient descent supporting a heavy-tailed student-t
  kernel q_ij ~ (1 + d_ij^2 / alpha)^{-(alpha+1)/2} whose tail parameter alpha
  can be optimized along with the embedding. O(n^2) per iteration.
- 'barnes_hut': scalable O(n log n) t-SNE run by an internal Barnes-Hut
  optimizer on a sparse model-neighbor probability matrix. The dense affinity
  matrix is never materialized; neighbor search can be exact blockwise or
  approximate via a random-projection candidate forest.

humap embeds the same model-based kNN graph with UMAP (umap-learn).

This package preserves the public API of the former single-module
``mixle.utils.hvis``: every name below remains importable from
``mixle.utils.hvis``. The implementation is split into:

- ``affinity`` - factor/affinity computation and probability calibration
- ``neighbors`` - sparse model-distance graphs, RP-trees, and kNN
- ``tsne`` - the t-SNE embedding cores (exact and Barnes-Hut)
- ``embed`` - the htsne/humap/dpmsne entry points
"""

# The submodules below carry the implementation; this package re-exports every
# name the former single-module htsne exposed (the documented public surface in
# __all__ plus the private helpers that mixle.tests.htsne_test imports directly).
# The `name as name` redundant-alias form marks these as deliberate re-exports.
from mixle.utils.hvis.affinity import (
    _affinity_factors as _affinity_factors,
)
from mixle.utils.hvis.affinity import (
    _calibrate_row as _calibrate_row,
)
from mixle.utils.hvis.affinity import (
    _component_inv_covariances as _component_inv_covariances,
)
from mixle.utils.hvis.affinity import (
    _factor_n as _factor_n,
)
from mixle.utils.hvis.affinity import (
    _factor_parts as _factor_parts,
)
from mixle.utils.hvis.affinity import (
    _factor_similarity_block as _factor_similarity_block,
)
from mixle.utils.hvis.affinity import (
    _factor_similarity_candidates as _factor_similarity_candidates,
)
from mixle.utils.hvis.affinity import (
    _factor_weight as _factor_weight,
)
from mixle.utils.hvis.affinity import (
    _field_log_densities as _field_log_densities,
)
from mixle.utils.hvis.affinity import (
    _field_log_density_features as _field_log_density_features,
)
from mixle.utils.hvis.affinity import (
    _fisher_similarity_block as _fisher_similarity_block,
)
from mixle.utils.hvis.affinity import (
    _hbeta as _hbeta,
)
from mixle.utils.hvis.affinity import (
    _is_fisher_factor as _is_fisher_factor,
)
from mixle.utils.hvis.affinity import (
    _is_local_factor as _is_local_factor,
)
from mixle.utils.hvis.affinity import (
    _is_prebuilt_affinity as _is_prebuilt_affinity,
)
from mixle.utils.hvis.affinity import (
    _leaf_feature_matrix as _leaf_feature_matrix,
)
from mixle.utils.hvis.affinity import (
    _local_similarity_block as _local_similarity_block,
)
from mixle.utils.hvis.affinity import (
    _observed_fisher_vectors as _observed_fisher_vectors,
)
from mixle.utils.hvis.affinity import (
    _posteriors_and_loglikes as _posteriors_and_loglikes,
)
from mixle.utils.hvis.affinity import (
    _resolve_affinity as _resolve_affinity,
)
from mixle.utils.hvis.affinity import (
    affinity_health as affinity_health,
)
from mixle.utils.hvis.affinity import (
    balanced_factors as balanced_factors,
)
from mixle.utils.hvis.affinity import (
    barycentric_init as barycentric_init,
)
from mixle.utils.hvis.affinity import (
    component_map as component_map,
)
from mixle.utils.hvis.affinity import (
    conditional_pmat as conditional_pmat,
)
from mixle.utils.hvis.affinity import (
    fisher_factors as fisher_factors,
)
from mixle.utils.hvis.affinity import (
    get_pmat as get_pmat,
)
from mixle.utils.hvis.affinity import (
    local_factors as local_factors,
)
from mixle.utils.hvis.affinity import (
    log_affinity_block as log_affinity_block,
)
from mixle.utils.hvis.affinity import (
    mixture_coordinates as mixture_coordinates,
)
from mixle.utils.hvis.affinity import (
    model_log_affinity as model_log_affinity,
)
from mixle.utils.hvis.direct import (
    ModelMap as ModelMap,
)
from mixle.utils.hvis.direct import (
    model_map as model_map,
)
from mixle.utils.hvis.distributed import (
    FiberStats as FiberStats,
)
from mixle.utils.hvis.distributed import (
    distributed_model_map as distributed_model_map,
)
from mixle.utils.hvis.distributed import (
    fiber_stats as fiber_stats,
)
from mixle.utils.hvis.distributed import (
    fuzzy_nerve_from_stats as fuzzy_nerve_from_stats,
)
from mixle.utils.hvis.embed import (
    dpmsne as dpmsne,
)
from mixle.utils.hvis.embed import (
    htsne as htsne,
)
from mixle.utils.hvis.embed import (
    humap as humap,
)
from mixle.utils.hvis.front import (
    Map as Map,
)
from mixle.utils.hvis.front import (
    hvis_map as hvis_map,
)
from mixle.utils.hvis.goals import (
    Anchor as Anchor,
)
from mixle.utils.hvis.goals import (
    AxisAlign as AxisAlign,
)
from mixle.utils.hvis.goals import (
    LabelCohesion as LabelCohesion,
)
from mixle.utils.hvis.neighbors import (
    _augment_candidates as _augment_candidates,
)
from mixle.utils.hvis.neighbors import (
    _build_rp_tree as _build_rp_tree,
)
from mixle.utils.hvis.neighbors import (
    _candidate_features as _candidate_features,
)
from mixle.utils.hvis.neighbors import (
    _candidate_log_affinity as _candidate_log_affinity,
)
from mixle.utils.hvis.neighbors import (
    _query_rp_tree as _query_rp_tree,
)
from mixle.utils.hvis.neighbors import (
    _RPTreeNode as _RPTreeNode,
)
from mixle.utils.hvis.neighbors import (
    approx_sparse_model_distances as approx_sparse_model_distances,
)
from mixle.utils.hvis.neighbors import (
    model_knn as model_knn,
)
from mixle.utils.hvis.neighbors import (
    sparse_model_distances as sparse_model_distances,
)
from mixle.utils.hvis.stream import (
    StreamingHvis as StreamingHvis,
)
from mixle.utils.hvis.stream import (
    place_in_atlas as place_in_atlas,
)
from mixle.utils.hvis.topology import (
    component_tree as component_tree,
)
from mixle.utils.hvis.topology import (
    embedding_health as embedding_health,
)
from mixle.utils.hvis.topology import (
    fuzzy_nerve as fuzzy_nerve,
)
from mixle.utils.hvis.topology import (
    model_fit_health as model_fit_health,
)
from mixle.utils.hvis.topology import (
    nerve_report as nerve_report,
)
from mixle.utils.hvis.tsne import (
    _barnes_hut_negative_forces as _barnes_hut_negative_forces,
)
from mixle.utils.hvis.tsne import (
    _BHNode as _BHNode,
)
from mixle.utils.hvis.tsne import (
    _build_bh_tree as _build_bh_tree,
)
from mixle.utils.hvis.tsne import (
    _csr_without_diagonal as _csr_without_diagonal,
)
from mixle.utils.hvis.tsne import (
    _exact_negative_forces as _exact_negative_forces,
)
from mixle.utils.hvis.tsne import (
    _exact_tsne_gradient as _exact_tsne_gradient,
)
from mixle.utils.hvis.tsne import (
    _flatten_bh_tree as _flatten_bh_tree,
)
from mixle.utils.hvis.tsne import (
    _kl as _kl,
)
from mixle.utils.hvis.tsne import (
    _negative_forces as _negative_forces,
)
from mixle.utils.hvis.tsne import (
    _numba_barnes_hut_negative_forces as _numba_barnes_hut_negative_forces,
)
from mixle.utils.hvis.tsne import (
    _python_barnes_hut_negative_forces as _python_barnes_hut_negative_forces,
)
from mixle.utils.hvis.tsne import (
    _sparse_conditional_pmat as _sparse_conditional_pmat,
)
from mixle.utils.hvis.tsne import (
    _sparse_joint_pmat as _sparse_joint_pmat,
)
from mixle.utils.hvis.tsne import (
    _sparse_positive_forces as _sparse_positive_forces,
)
from mixle.utils.hvis.tsne import (
    _sparse_positive_forces_from_edges as _sparse_positive_forces_from_edges,
)
from mixle.utils.hvis.tsne import (
    _sparse_positive_forces_symmetric_from_edges as _sparse_positive_forces_symmetric_from_edges,
)
from mixle.utils.hvis.tsne import (
    _sparse_tsne_kl as _sparse_tsne_kl,
)
from mixle.utils.hvis.tsne import (
    _sparse_tsne_kl_from_edges as _sparse_tsne_kl_from_edges,
)
from mixle.utils.hvis.tsne import (
    _tsne_barnes_hut as _tsne_barnes_hut,
)
from mixle.utils.hvis.tsne import (
    _tsne_barnes_hut_from_p as _tsne_barnes_hut_from_p,
)
from mixle.utils.hvis.tsne import (
    t_kernel as t_kernel,
)
from mixle.utils.hvis.tsne import (
    tsne_barnes_hut as tsne_barnes_hut,
)
from mixle.utils.hvis.tsne import (
    tsne_exact as tsne_exact,
)
from mixle.utils.hvis.tsne import (
    update_alpha as update_alpha,
)
from mixle.utils.hvis.tsne import (
    update_embed as update_embed,
)

__all__ = [
    "htsne",
    "StreamingHvis",
    "place_in_atlas",
    "Anchor",
    "LabelCohesion",
    "AxisAlign",
    "humap",
    "dpmsne",
    "model_log_affinity",
    "affinity_health",
    "log_affinity_block",
    "mixture_coordinates",
    "component_map",
    "barycentric_init",
    "model_map",
    "ModelMap",
    "fuzzy_nerve",
    "nerve_report",
    "embedding_health",
    "model_fit_health",
    "component_tree",
    "hvis_map",
    "Map",
    "map",
    "sparse_model_distances",
    "approx_sparse_model_distances",
    "model_knn",
    "get_pmat",
    "balanced_factors",
    "local_factors",
    "fisher_factors",
    "tsne_barnes_hut",
]

# the design review promised hvis.map(); hvis_map is the shadow-safe import name.
map = hvis_map  # noqa: A001
