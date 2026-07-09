"""Top-level model-based embedding entry points: htsne, humap, dpmsne.

These orchestrate the pieces in this package: fit/accept a mixture model,
resolve the affinity, build the (sparse or dense) probability matrix from the
model posteriors, and run the chosen embedding engine.
"""

import sys

import numpy as np

from mixle.utils.hvis.affinity import (
    _affinity_factors,
    _factor_n,
    _is_prebuilt_affinity,
    _posteriors_and_loglikes,
    _resolve_affinity,
    get_pmat,
)
from mixle.utils.hvis.neighbors import (
    approx_sparse_model_distances,
    model_knn,
    sparse_model_distances,
)
from mixle.utils.hvis.tsne import _tsne_barnes_hut, tsne_exact


def htsne(
    data,
    emb_dim: int = 2,
    alpha: float = 1.0,
    max_components: int = 50,
    Y: np.ndarray | None = None,
    perplexity: float | None = 30.0,
    max_its: int = 1000,
    print_iter: int = 100,
    eta: float | None = None,
    momentum: float = 0.8,
    min_gain: float = 0.01,
    min_value: float = 1.0e-128,
    optimize_alpha: bool = False,
    min_alpha: float = 1.0e-6,
    max_alpha_its: int = 3,
    seed: int | None = None,
    mix_model=None,
    enc_data=None,
    method: str = "auto",
    early_exaggeration: float | None = None,
    tol: float = 1.0e-7,
    dpm_max_its: int = 200,
    affinity="auto",
    field_weights=None,
    evidence_cap: float | None = 1.0,
    fisher_metric: str = "diagonal",
    fisher_ridge: float = 1.0e-8,
    fisher_information: str = "observed",
    out=None,
    variable_length: bool = False,
    barnes_hut_theta: float = 0.5,
    barnes_hut_leaf_size: int = 16,
    neighbor_method: str = "auto",
    neighbor_threshold: int = 5000,
    neighbor_trees: int = 8,
    neighbor_leaf_size: int | None = None,
    candidate_multiplier: int = 8,
    repulsion_method: str = "auto",
    exact_repulsion_threshold: int = 5000,
    goals=None,
):
    """Embed heterogeneous data with model-based t-SNE.

    goals: optional sequence of embedding goals (mixle.utils.hvis.goals) -- Anchor pins for
    anchoring, LabelCohesion for partial labeling, AxisAlign for layout objectives. Goal gradients
    join the data gradient every iteration on BOTH engines; hard anchors are re-projected exactly
    after every step.

    Y='barycentric' initializes every observation at its posterior-weighted combination of
    component vertices laid out by overlap geometry (see affinity.barycentric_init): the layout's
    global arrangement comes from the model instead of the random seed, so runs are globally
    consistent and mixed-membership points start (and tend to stay) between their clusters.

    early_exaggeration=None (the default) resolves to 12.0 for a random init and 1.0 for an
    informative one (Y='barycentric' or a supplied array): exaggeration exists to FORM global
    structure from randomness, and given a meaningful init it does the opposite -- crushes
    confusable clusters together before the refine phase can save them (measured: 0.71-0.89
    purity and seed-dependent arrangements at 12.0, a seed-stable 0.96 at 1.0). Pass a number to
    override.

    A mixture model is fit to the data (a Dirichlet process mixture with
    automatically typed components by default, or pass mix_model), pairwise
    affinities are computed from the model, and the affinities are embedded
    with t-SNE. Passing affinity='fisher' with any model that exposes
    to_fisher(), or passing a pre-built affinity factor list, bypasses the
    mixture-posterior affinity path and does not require a DPM/mixture model.

    method:
        'exact'      - full-matrix gradient descent (supports optimize_alpha)
        'barnes_hut' - sparse model probabilities + internal Barnes-Hut t-SNE
        'auto'       - barnes_hut for n > 10 unless optimize_alpha is set

    affinity:
        'auto' (default) - 'local' whenever raw data is available and the
            model decomposes into leaf fields, else 'bhattacharyya'
        'local'      - per-field posterior overlap plus component-local
            Mahalanobis geometry for continuous/count fields, estimated from
            the realized data; discrete fields fall back to posterior overlap
        'balanced'   - per-field posteriors (the model's flattened leaves:
            nested composites, sequence element/length models, and optional
            wrappers all decompose) combined by per-field Bhattacharyya, so a
            sharp discrete field cannot drown an overlapping continuous one
            (or vice versa); optional field_weights sets exponents on whole
            field-level Bhattacharyya coefficients
        'fisher'     - posterior-expected sufficient statistics from
            mix_model.to_fisher(), whitened by an observed Fisher metric;
            fisher_information='observed' uses the empirical covariance of
            observed score vectors, while 'model' uses the view's model metric;
            fisher_metric is 'diagonal' by default, with 'identity' and 'full'
            also accepted
        'bhattacharyya' - Bhattacharyya coefficient between joint posteriors;
            graded even under hard assignments, so embeddings retain
            within-cluster geometry
        'coassign'   - co-assignment probability P(z_i = z_j | x); exact but
            near-binary when posteriors are sharp
        'likelihood' - predictive affinity sum_k p(x_i|theta_k) z_jk

    variable_length is retained for backward compatibility and does not
    rescale densities. Variable-length behavior is determined by the fitted
    sequence model: ordinary SequenceDistribution leaves use summed element
    log-likelihood with length as a separate field, while
    SequenceDistribution(len_normalized=True) intentionally uses a per-token
    composition quotient for the element field.

    evidence_cap (default 1.0 nats) bounds the dissimilarity evidence any
    single field may contribute to a pair's distance under multi-field
    affinities: without it, one spuriously sharp field (a serial-number-like
    categorical the model micro-clustered) drives its per-field affinity to
    zero and vetoes the pair's similarity no matter what every other field
    says. None disables the cap; single-field affinities ignore it.

    barnes_hut_theta controls the Barnes-Hut opening angle for method='barnes_hut';
    0.0 gives exact repulsive forces and larger values are faster/coarser.

    repulsion_method controls repulsive forces for method='barnes_hut':
    'exact' uses a vectorized all-pairs calculation, 'barnes_hut' uses the
    tree approximation, and 'auto' uses exact repulsion when n is at most
    exact_repulsion_threshold.

    neighbor_method controls graph construction for method='barnes_hut':
    'exact' uses blockwise all-pairs top-k, 'approx' uses a random-projection
    candidate forest, and 'auto' switches to 'approx' when n >= neighbor_threshold.

    Returns the n x emb_dim embedding.
    """
    if out is None:
        out = sys.stdout

    if mix_model is None and not _is_prebuilt_affinity(affinity):
        from mixle.utils.automatic import get_dpm_mixture

        mix_model = get_dpm_mixture(
            data,
            rng=np.random.RandomState(seed),
            max_components=max_components,
            max_its=dpm_max_its,
            print_iter=print_iter,
            out=out,
        )

    if mix_model is not None:
        affinity = _resolve_affinity(
            affinity,
            mix_model,
            data,
            field_weights,
            enc_data=enc_data,
            fisher_metric=fisher_metric,
            fisher_ridge=fisher_ridge,
            fisher_information=fisher_information,
        )

    if _is_prebuilt_affinity(affinity):
        z_ij, l_ij = None, None
        n = _factor_n(_affinity_factors(None, None, affinity)[0])
        if data is not None and len(data) != n:
            raise ValueError("pre-built affinity row count does not match data length.")
    else:
        z_ij, l_ij = _posteriors_and_loglikes(mix_model, data=data, enc_data=enc_data)
        n = z_ij.shape[0]

    informative_init = Y is not None
    if isinstance(Y, str):
        if Y != "barycentric":
            raise ValueError("Y accepts an (n, emb_dim) array, None, or the string 'barycentric'.")
        if z_ij is not None:
            z_bary = z_ij
        elif mix_model is not None:  # 'auto'/'local' resolve to factor lists before posteriors exist
            z_bary, _ = _posteriors_and_loglikes(mix_model, data=data, enc_data=enc_data)
        else:
            raise ValueError(
                "Y='barycentric' needs mixture posteriors (a mix_model, or data to fit one); "
                "a pre-built affinity factor list without a model carries no posterior to take "
                "barycentric coordinates of."
            )
        from mixle.utils.hvis.affinity import barycentric_init

        Y = barycentric_init(z_bary, emb_dim=emb_dim, seed=seed)

    if early_exaggeration is None:
        early_exaggeration = 1.0 if informative_init else 12.0

    if method == "auto":
        method = "exact" if (optimize_alpha or n <= 10) else "barnes_hut"

    if method == "barnes_hut":
        px = 30.0 if perplexity is None else float(perplexity)
        px = min(px, max(1.0, n - 1.0))
        k = min(n - 1, int(3.0 * px) + 5)
        graph_method = neighbor_method
        if graph_method == "auto":
            graph_method = "approx" if n >= neighbor_threshold else "exact"
        if graph_method == "exact":
            dist_csr = sparse_model_distances(z_ij, l_ij, k=k, affinity=affinity, evidence_cap=evidence_cap)
        elif graph_method == "approx":
            dist_csr = approx_sparse_model_distances(
                z_ij,
                l_ij,
                k=k,
                affinity=affinity,
                evidence_cap=evidence_cap,
                n_trees=neighbor_trees,
                leaf_size=neighbor_leaf_size,
                candidate_multiplier=candidate_multiplier,
                seed=seed,
            )
        else:
            raise ValueError("neighbor_method must be 'auto', 'exact', or 'approx'.")
        return _tsne_barnes_hut(
            dist_csr,
            emb_dim,
            px,
            max_its,
            eta,
            momentum,
            early_exaggeration,
            min_gain,
            tol,
            print_iter,
            seed,
            Y,
            out=out,
            theta=barnes_hut_theta,
            leaf_size=barnes_hut_leaf_size,
            repulsion_method=repulsion_method,
            exact_repulsion_threshold=exact_repulsion_threshold,
            goals=goals,
        )

    P = get_pmat(z_ij, l_ij, targ_perplexity=perplexity, affinity=affinity, evidence_cap=evidence_cap)
    return tsne_exact(
        P,
        emb_dim=emb_dim,
        alpha=alpha,
        Y=Y,
        max_its=max_its,
        eta=eta,
        momentum=momentum,
        early_exaggeration=early_exaggeration,
        min_gain=min_gain,
        min_value=min_value,
        optimize_alpha=optimize_alpha,
        min_alpha=min_alpha,
        max_alpha_its=max_alpha_its,
        tol=tol,
        print_iter=print_iter,
        seed=seed,
        out=out,
        goals=goals,
    )


def humap(
    data,
    emb_dim: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_components: int = 50,
    seed: int | None = None,
    mix_model=None,
    enc_data=None,
    dpm_max_its: int = 200,
    print_iter: int = 100,
    affinity="auto",
    field_weights=None,
    evidence_cap: float | None = 1.0,
    fisher_metric: str = "diagonal",
    fisher_ridge: float = 1.0e-8,
    fisher_information: str = "observed",
    n_epochs: int | None = None,
    out=None,
    engine: str = "auto",
    goals=None,
    **umap_kwargs,
):
    """Embed heterogeneous data with model-based UMAP.

    The same mixture-model affinities as htsne (see the affinity and
    evidence_cap arguments there), but the k-nearest-neighbor graph of model
    distances -log s_ij is handed to UMAP's fuzzy simplicial set construction
    and layout instead of t-SNE. Scales like UMAP: the dense affinity matrix
    is never built.

    engine selects the layout backend:
        'umap-learn' - the optional umap-learn package (extra keyword
            arguments are passed to umap.UMAP). Cannot honor goals: its numba
            SGD loop takes no external gradients, so goals raise rather than
            being silently dropped.
        'internal'   - mixle.utils.hvis.umap_np, a dependency-free UMAP core
            (same construction: smoothed-kNN fuzzy graph, fitted a/b curve,
            epochs-per-sample SGD with negative sampling). Slower than the
            numba path but always available, and the only engine that can
            steer the layout with goals.
        'auto'       - umap-learn when it is installed AND no goals were
            given; the internal engine otherwise.

    goals: optional sequence of embedding goals (mixle.utils.hvis.goals) --
    Anchor / LabelCohesion / AxisAlign, as in htsne. Requires the internal
    engine (auto selects it when goals are present).
    """
    if engine not in ("auto", "umap-learn", "internal"):
        raise ValueError("engine must be 'auto', 'umap-learn', or 'internal'.")

    umap = None
    if engine in ("auto", "umap-learn"):
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Tensorflow not installed; ParametricUMAP will be unavailable",
                    category=ImportWarning,
                )
                import umap
        except ImportError:
            if engine == "umap-learn":
                from mixle.utils.optional_deps import require

                require("umap-learn", "umap")

    if engine == "umap-learn" and goals:
        raise ValueError(
            "goals require engine='internal' (or 'auto'): umap-learn's layout loop cannot honor them, "
            "and silently dropping a stated goal would be worse than refusing."
        )
    if engine == "auto":
        engine = "umap-learn" if (umap is not None and not goals) else "internal"

    if out is None:
        out = sys.stdout

    if mix_model is None and not _is_prebuilt_affinity(affinity):
        from mixle.utils.automatic import get_dpm_mixture

        mix_model = get_dpm_mixture(
            data,
            rng=np.random.RandomState(seed),
            max_components=max_components,
            max_its=dpm_max_its,
            print_iter=print_iter,
            out=out,
        )

    if mix_model is not None:
        affinity = _resolve_affinity(
            affinity,
            mix_model,
            data,
            field_weights,
            enc_data=enc_data,
            fisher_metric=fisher_metric,
            fisher_ridge=fisher_ridge,
            fisher_information=fisher_information,
        )

    if _is_prebuilt_affinity(affinity):
        z_ij, l_ij = None, None
        n = _factor_n(_affinity_factors(None, None, affinity)[0])
        if data is not None and len(data) != n:
            raise ValueError("pre-built affinity row count does not match data length.")
    else:
        z_ij, l_ij = _posteriors_and_loglikes(mix_model, data=data, enc_data=enc_data)
        n = z_ij.shape[0]
    k = min(n_neighbors, n - 1)

    knn_idx, knn_dist = model_knn(z_ij, l_ij, k=k, affinity=affinity, evidence_cap=evidence_cap)

    if engine == "internal":
        from mixle.utils.hvis.umap_np import internal_umap

        return internal_umap(
            knn_idx,
            knn_dist,
            emb_dim=emb_dim,
            min_dist=min_dist,
            n_epochs=n_epochs,
            seed=seed,
            goals=goals,
            **umap_kwargs,
        )

    reducer = umap.UMAP(
        n_components=emb_dim,
        n_neighbors=k,
        min_dist=min_dist,
        precomputed_knn=(knn_idx, knn_dist),
        random_state=seed,
        n_epochs=n_epochs,
        **umap_kwargs,
    )

    with warnings.catch_warnings():
        # expected with precomputed knn / fixed seed; not actionable here
        warnings.filterwarnings("ignore", message=".*knn_search_index.*")
        warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
        return reducer.fit_transform(np.zeros((n, 1), dtype=np.float32))


def dpmsne(
    P=None,
    emb_dim: int = 2,
    alpha: float = 1.0,
    Y: np.ndarray | None = None,
    max_its: int = 1000,
    print_iter: int = 100,
    eta: float | None = None,
    momentum: float = 0.8,
    min_gain: float = 0.01,
    min_value: float = 1.0e-128,
    optimize_alpha: bool = False,
    min_alpha: float = 1.0e-6,
    max_alpha_its: int = 3,
    seed: int | None = None,
    early_exaggeration: float = 12.0,
    tol: float = 1.0e-7,
    out=None,
    **_compat_kwargs,
):
    """Embed a precomputed (symmetric, non-negative) affinity matrix P with exact t-SNE."""
    return tsne_exact(
        np.asarray(P, dtype=np.float64),
        emb_dim=emb_dim,
        alpha=alpha,
        Y=Y,
        max_its=max_its,
        eta=eta,
        momentum=momentum,
        early_exaggeration=early_exaggeration,
        min_gain=min_gain,
        min_value=min_value,
        optimize_alpha=optimize_alpha,
        min_alpha=min_alpha,
        max_alpha_its=max_alpha_its,
        tol=tol,
        print_iter=print_iter,
        seed=seed,
        out=out,
    )
