"""Factor/affinity computation for model-based (hierarchical) t-SNE/UMAP.

This module turns a fitted mixture model (or pre-built factors) into the
per-pair log affinities that drive the embedding: it builds the affinity
factor list for each affinity mode, computes the dense log-affinity matrix,
and converts a log-affinity matrix into row-conditional t-SNE probabilities.
See the package docstring for the affinity definitions.
"""

import numpy as np


def _affinity_factors(posterior_mat, ll_mat, affinity):
    """Factor the log-affinity as log S = sum_f log(G_f H_f^T), returned as a
    list of (G_f, H_f) pairs (up to immaterial per-row scale).

    The single-factor modes return one pair; 'balanced' (built by
    balanced_factors) supplies one pair per composite field.
    """
    if isinstance(affinity, (list, tuple)):
        factors = list(affinity)
        if not factors:
            raise ValueError("affinity factor list must not be empty.")
        return factors  # pre-built factor list

    if affinity == "fisher":
        raise ValueError("affinity='fisher' requires fisher_factors(model, data=...) or htsne/humap with a model.")

    z = np.asarray(posterior_mat, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("posterior_mat must be a two-dimensional array.")

    if affinity == "coassign":
        return [(z, z)]
    if affinity == "bhattacharyya":
        zs = np.sqrt(z)
        return [(zs, zs)]
    if affinity == "likelihood":
        if ll_mat is None:
            raise ValueError("affinity='likelihood' requires the component log-likelihood matrix.")
        l = np.asarray(ll_mat, dtype=np.float64)
        if l.shape != z.shape:
            raise ValueError("ll_mat must have the same shape as posterior_mat.")
        return [(np.exp(l - l.max(axis=1, keepdims=True)), z)]

    raise ValueError(
        "affinity must be 'coassign', 'bhattacharyya', 'likelihood', "
        "'local', 'balanced', 'fisher', or a pre-built factor list."
    )


def _is_prebuilt_affinity(affinity) -> bool:
    return isinstance(affinity, (list, tuple))


def _leaf_feature_matrix(dists, items):
    """Local coordinates for supported scalar/vector leaves, or None.

    These coordinates are not a global feature embedding. They are only used
    inside component-local covariance estimates, so unsupported/discrete leaves
    correctly fall back to posterior geometry.
    """
    tname = type(dists[0]).__name__
    try:
        if tname == "GaussianDistribution":
            return np.asarray(items, dtype=np.float64).reshape(-1, 1)
        if tname == "DiagonalGaussianDistribution":
            return np.asarray(items, dtype=np.float64)
        if tname == "LogGaussianDistribution":
            x = np.asarray(items, dtype=np.float64)
            if np.any(x <= 0):
                return None
            return np.log(x).reshape(-1, 1)
        if tname == "GammaDistribution":
            x = np.asarray(items, dtype=np.float64)
            if np.any(x <= 0):
                return None
            return np.column_stack((x, np.log(x)))
        if tname in ("PoissonDistribution", "ExponentialDistribution", "GeometricDistribution", "BinomialDistribution"):
            return np.asarray(items, dtype=np.float64).reshape(-1, 1)
    except (TypeError, ValueError):
        return None
    return None


def _typicality_coordinates(l, items):
    """Universal within-component coordinates for leaves with NO native vector coordinates.

    Every leaf -- an HMM, a Markov chain, a categorical, anything with seq_log_density -- already
    yields the per-component log-density matrix ``l`` (n x K). Its columns are a real, continuous
    "how typical is this observation under regime k" coordinate: exactly the within-component
    structure a posterior-overlap factor throws away (sharp posteriors make same-component pairs
    indistinguishable, which is what renders clusters as tiny structureless points). The component-
    local Mahalanobis machinery downstream whitens per component, so these coordinates are made
    dimensionless there -- no manual scale tuning, and continuous/discrete/sequence fields become
    commensurate for free.

    Sequence-valued leaves (list/tuple/ndarray items) use the PER-TOKEN log-density rate plus one
    explicit log-length column: total sequence evidence grows with length, so unnormalized columns
    would all reduce to a single length axis -- length stays visible, but as one honest coordinate
    among K+1 rather than as all of them. Non-finite entries (impossible categories score -inf) are
    floored a little below the finite range so they read as "maximally atypical", never as NaN.
    """
    l = np.asarray(l, dtype=np.float64)
    first = items[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        lens = np.asarray([len(x) for x in items], dtype=np.float64)
        coords = np.column_stack([l / np.maximum(lens, 1.0)[:, None], np.log1p(lens)])
    else:
        coords = l.copy()
    return _floor_nonfinite(coords)


def _floor_nonfinite(coords: np.ndarray) -> np.ndarray:
    """Replace non-finite coordinate entries with a value a little below the column's finite range
    -- "maximally atypical", never NaN (a -inf score would otherwise poison the Mahalanobis block)."""
    finite = np.isfinite(coords)
    if not finite.all():
        for j in range(coords.shape[1]):
            col_finite = coords[finite[:, j], j]
            if len(col_finite):
                lo = float(col_finite.min())
                spread = float(col_finite.std()) or 1.0
                coords[~finite[:, j], j] = lo - 3.0 * spread
            else:
                coords[:, j] = 0.0
    return coords


def _field_log_density_features(dists, items):
    """Yield (log_density_matrix, feature_matrix_or_None, native_coords) for leaf fields.

    ``native_coords`` records whether the feature matrix came from the leaf's own value coordinates
    (True) or from universal typicality coordinates (False) -- the latter are scored per degree of
    freedom downstream, and guessing the origin from array shapes would be fragile.

    - composite records recurse into their child fields (nested composites
      flatten all the way down),
    - sequences score each child field by summed element log-likelihood by
      default, or by mean element log-likelihood only when the fitted
      SequenceDistribution has len_normalized=True; the length model
      contributes its own field,
    - optional wrappers contribute a missing-ness field, with the inner
      distribution's fields scored only on rows where the value is present,
    - ignored/null distributions contribute nothing,
    - everything else (Gaussian, categorical, Markov chains, ...) is a leaf
      scored with its own seq_log_density.
    """
    tname = type(dists[0]).__name__
    n, K = len(items), len(dists)

    if "Ignored" in tname or "Null" in tname:
        return

    if "Composite" in tname and hasattr(dists[0], "dists"):
        for f in range(len(dists[0].dists)):
            yield from _field_log_density_features([d.dists[f] for d in dists], [x[f] for x in items])
        return

    if "Sequence" in tname and hasattr(dists[0], "dist") and hasattr(dists[0], "len_dist"):
        lens = [len(x) for x in items]
        elems = [e for x in items for e in x]
        if elems:
            seg = np.repeat(np.arange(n), lens)
            for l_e, x_e, native_e in _field_log_density_features([d.dist for d in dists], elems):
                l_f = np.zeros((n, K))
                np.add.at(l_f, seg, l_e)
                if getattr(dists[0], "len_normalized", False):
                    denom = np.asarray(lens, dtype=np.float64)
                    mask = denom > 0
                    l_f[mask] /= denom[mask, None]
                if x_e is not None:
                    x_f = np.zeros((n, x_e.shape[1]), dtype=np.float64)
                    np.add.at(x_f, seg, x_e)
                    denom = np.asarray(lens, dtype=np.float64)
                    mask = denom > 0
                    x_f[mask] /= denom[mask, None]
                    native_f = native_e
                else:
                    # element leaves without native coordinates (categorical/Markov tokens): the
                    # per-token typicality rate is still a real within-component coordinate. The
                    # separate length field carries length, so no log-length column here.
                    x_f = np.zeros((n, l_e.shape[1]), dtype=np.float64)
                    np.add.at(x_f, seg, l_e)
                    denom = np.asarray(lens, dtype=np.float64)
                    mask = denom > 0
                    x_f[mask] /= denom[mask, None]
                    x_f = _floor_nonfinite(x_f)
                    native_f = False
                yield l_f, x_f, native_f
        len_dists = [d.len_dist for d in dists]
        if len_dists[0] is not None:
            yield from _field_log_density_features(len_dists, lens)
        return

    if "Optional" in tname and hasattr(dists[0], "dist"):
        mv = getattr(dists[0], "missing_value", None)
        mv_is_nan = isinstance(mv, float) and np.isnan(mv)
        miss = np.asarray([x is None or x is mv or (mv_is_nan and isinstance(x, float) and np.isnan(x)) for x in items])
        has_gate = [getattr(d, "has_p", True) for d in dists]
        if any(has_gate):
            lp0 = np.asarray(
                [getattr(d, "log_p0", getattr(d, "log_p", 0.0)) if has_p else 0.0 for d, has_p in zip(dists, has_gate)],
                dtype=np.float64,
            )  # log P(missing)
            lp1 = np.asarray(
                [
                    getattr(d, "log_p1", getattr(d, "log_pn", 0.0)) if has_p else 0.0
                    for d, has_p in zip(dists, has_gate)
                ],
                dtype=np.float64,
            )  # log P(present)
            # the presence pattern is real within-cluster structure: a 0/1 coordinate lets the
            # local machinery separate present-vs-missing neighbors instead of posterior-only ties.
            yield np.where(miss[:, None], lp0[None, :], lp1[None, :]), (~miss).astype(np.float64)[:, None], False
        if (~miss).any():
            fill = items[int(np.argmax(~miss))]
            sub = [fill if m else x for x, m in zip(items, miss)]
            keep = (~miss).astype(np.float64)[:, None]
            for l_in, _x, _nat in _field_log_density_features([d.dist for d in dists], sub):
                yield np.where(keep > 0.0, l_in, 0.0), None, True
        return

    if hasattr(dists[0], "dist_to_encoder"):
        enc = dists[0].dist_to_encoder().seq_encode(items)
    elif hasattr(dists[0], "seq_encode"):
        enc = dists[0].seq_encode(items)
    else:
        enc = None

    l = np.empty((n, K))
    for k, d in enumerate(dists):
        if enc is not None:
            l[:, k] = np.asarray(d.seq_log_density(enc), dtype=np.float64)
        else:
            l[:, k] = [d.log_density(x) for x in items]
    x_native = _leaf_feature_matrix(dists, items)
    if x_native is not None:
        yield l, x_native, True
    else:
        yield l, _typicality_coordinates(l, items), False


def _field_log_densities(dists, items):
    for l_f, _x, _native in _field_log_density_features(dists, items):
        yield l_f


def balanced_factors(mix_model, data, field_weights=None):
    """Per-field Bhattacharyya affinity factors for heterogeneous models.

    The joint posterior is dominated by whichever field has the largest
    log-likelihood contrast across components - sharp categorical fields,
    long token-sequence fields, or collapsed continuous components can
    contribute many nats of contrast while overlapping continuous fields
    contribute fractions of one. The drowned fields' relationships then become
    invisible to any affinity computed from the joint posterior.

    'balanced' fixes the scale problem at the affinity level: a *field-
    restricted* posterior z^f is computed from each field's likelihoods alone
    (fields are the model's flattened leaves - nested composites, sequence
    element/length models, and optional wrappers all decompose; see
    _field_log_densities), and the affinity combines per-field Bhattacharyya
    coefficients, so every field contributes comparably regardless of its
    likelihood scale. field_weights apply as exponents on whole field
    coefficients, i.e. weights on log field-affinities. Combined with an
    evidence cap (see model_log_affinity) no single field can veto a pair's
    similarity either.
    """
    comps = list(mix_model.components)
    log_w = np.asarray(mix_model.log_w, dtype=np.float64).reshape(1, -1)

    l_fields = list(_field_log_densities(comps, list(data)))
    if not l_fields:
        raise ValueError("affinity='balanced' found no scorable fields in the mixture components.")

    if field_weights is None:
        field_weights = [1.0] * len(l_fields)
    elif len(field_weights) != len(l_fields):
        raise ValueError(
            "field_weights has %d entries but the model flattens to %d "
            "leaf fields." % (len(field_weights), len(l_fields))
        )

    factors = []
    for l_f, w_f in zip(l_fields, field_weights):
        if w_f < 0:
            raise ValueError("field_weights must be non-negative.")
        z_f = np.asarray(l_f, dtype=np.float64) + log_w
        finite_rows = np.isfinite(z_f).any(axis=1)
        if not np.all(finite_rows):
            z_f[~finite_rows] = log_w
        z_f -= z_f.max(axis=1, keepdims=True)
        np.exp(z_f, out=z_f)
        z_f /= z_f.sum(axis=1, keepdims=True)

        sq = np.sqrt(z_f)
        factors.append((sq, sq) if w_f == 1.0 else (sq, sq, float(w_f)))

    return factors


def _component_inv_covariances(x: np.ndarray, z: np.ndarray, ridge: float = 1.0e-4) -> np.ndarray:
    """Component-local inverse covariances for local feature coordinates."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n, dim = x.shape
    K = z.shape[1]

    finite = np.isfinite(x).all(axis=1)
    if not np.all(finite):
        fill = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        x = np.where(np.isfinite(x), x, fill)

    xc = x - x.mean(axis=0, keepdims=True)
    denom = max(n - 1, 1)
    global_cov = np.dot(xc.T, xc) / denom
    scale = float(np.trace(global_cov) / max(dim, 1))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    global_cov = global_cov + np.eye(dim) * (ridge * scale + 1.0e-8)

    inv_covs = np.empty((K, dim, dim), dtype=np.float64)
    for k in range(K):
        wk = z[:, k]
        sw = float(wk.sum())
        if sw <= dim + 1.0e-8:
            cov = global_cov
        else:
            mu = np.dot(wk, x) / sw
            dx = x - mu
            cov = np.dot((wk[:, None] * dx).T, dx) / sw
            local_scale = float(np.trace(cov) / max(dim, 1))
            if not np.isfinite(local_scale) or local_scale <= 0.0:
                local_scale = scale
            cov = cov + np.eye(dim) * (ridge * local_scale + 1.0e-8)
        inv_covs[k] = np.linalg.pinv(cov)

    return inv_covs


def local_factors(mix_model, data, field_weights=None):
    """Per-field local statistical affinity factors.

    Each leaf field is first represented by its field-restricted component
    posterior, and EVERY field also carries within-component local geometry:
    continuous/count leaves (and averages of such leaves inside sequences) use
    their native coordinates; every other leaf -- HMMs, Markov chains,
    categoricals, sequence-of-discrete element fields -- uses typicality
    coordinates (per-component log-density; per-token rate plus a log-length
    axis for sequence-valued leaves, see _typicality_coordinates). Without
    that universal fallback, sharp posteriors make all same-component pairs
    exact ties and clusters render as tiny structureless points -- the
    collapse this affinity exists to prevent. The factor carries
    component-local inverse covariances estimated from the realized data
    (which also makes heterogeneous fields dimensionless, so continuous,
    discrete, and sequence evidence are commensurate). Pair affinities then
    use

        sum_k sqrt(z_ik z_jk) exp(-delta_ijk / 8),

    where delta_ijk is the component-local Mahalanobis distance in that
    field's coordinates. This is the local Fisher quadratic in the plug-in
    model, with posterior overlap handling component uncertainty.
    """
    comps = list(mix_model.components)
    log_w = np.asarray(mix_model.log_w, dtype=np.float64).reshape(1, -1)

    terms = list(_field_log_density_features(comps, list(data)))
    if not terms:
        raise ValueError("affinity='local' found no scorable fields in the mixture components.")

    if field_weights is None:
        field_weights = [1.0] * len(terms)
    elif len(field_weights) != len(terms):
        raise ValueError(
            "field_weights has %d entries but the model flattens to %d leaf fields." % (len(field_weights), len(terms))
        )

    factors = []
    for (l_f, x_f, native_f), w_f in zip(terms, field_weights):
        if w_f < 0:
            raise ValueError("field_weights must be non-negative.")
        z_f = np.asarray(l_f, dtype=np.float64) + log_w
        finite_rows = np.isfinite(z_f).any(axis=1)
        if not np.all(finite_rows):
            z_f[~finite_rows] = log_w
        z_f -= z_f.max(axis=1, keepdims=True)
        np.exp(z_f, out=z_f)
        z_f /= z_f.sum(axis=1, keepdims=True)

        sq = np.sqrt(z_f)
        if x_f is None or np.asarray(x_f).ndim != 2 or np.asarray(x_f).shape[1] == 0:
            factors.append((sq, sq) if w_f == 1.0 else (sq, sq, float(w_f)))
        else:
            x_f = np.asarray(x_f, dtype=np.float64)
            # typicality coordinates span one column per component (+ length), so a whitened
            # Mahalanobis delta averages ~dim and would saturate the evidence cap as K grows,
            # making far-within-cluster pairs indistinguishable from cross-cluster pairs.
            # Score them per degree of freedom; native low-dim coordinates keep the exact
            # behavior the /8 exponent was tuned on.
            factors.append(
                {
                    "kind": "local",
                    "sqrt_z": sq,
                    "x": x_f,
                    "inv_cov": _component_inv_covariances(x_f, z_f),
                    "weight": float(w_f),
                    "delta_scale": 1.0 if native_f else float(x_f.shape[1]),
                }
            )

    return factors


def _observed_fisher_vectors(view, stats: np.ndarray, metric: str, ridge: float) -> np.ndarray:
    return view.observed_fisher_vectors(stats=np.asarray(stats, dtype=np.float64), metric=metric, ridge=ridge)


def fisher_factors(
    model,
    data=None,
    enc_data=None,
    metric: str = "diagonal",
    ridge: float = 1.0e-8,
    weight: float = 1.0,
    information: str = "observed",
):
    """Fisher-vector affinity factor for a model and observations.

    The model supplies posterior-expected sufficient statistics through
    to_fisher().  By default those statistics are treated as observed score
    vectors and whitened by their empirical observed Fisher covariance.  Set
    information='model' to use the view's model Fisher metric directly.  Pair
    affinities are s_ij = exp(-0.5 ||v_i - v_j||^2).
    """
    if data is None and enc_data is None:
        raise ValueError("affinity='fisher' requires raw data or encoded data.")
    if data is not None and enc_data is not None:
        raise ValueError("pass only one of data or enc_data for affinity='fisher'.")
    if weight < 0:
        raise ValueError("fisher affinity weight must be non-negative.")
    if information not in ("observed", "model"):
        raise ValueError("information must be 'observed' or 'model'.")

    view = model.to_fisher()
    if data is not None:
        stats = view.expected_statistics_matrix(data=list(data))
    else:
        stats = view.seq_expected_statistics(enc_data)
    if information == "observed":
        x = _observed_fisher_vectors(view, stats=stats, metric=metric, ridge=ridge)
    else:
        x = view.fisher_vectors(stats=stats, metric=metric, ridge=ridge)
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    return [
        {
            "kind": "fisher",
            "x": x,
            "weight": float(weight),
            "metric": metric,
            "ridge": float(ridge),
            "information": information,
        }
    ]


def _factor_parts(factor):
    if isinstance(factor, dict):
        raise ValueError("local/fisher affinity factors are not dot-product factors.")
    if len(factor) == 2:
        g, h = factor
        weight = 1.0
    elif len(factor) == 3:
        g, h, weight = factor
    else:
        raise ValueError("affinity factors must be (G, H) or (G, H, weight) tuples.")

    if weight < 0:
        raise ValueError("affinity factor weights must be non-negative.")
    return np.asarray(g, dtype=np.float64), np.asarray(h, dtype=np.float64), float(weight)


def _is_local_factor(factor) -> bool:
    return isinstance(factor, dict) and factor.get("kind") == "local"


def _is_fisher_factor(factor) -> bool:
    return isinstance(factor, dict) and factor.get("kind") == "fisher"


def _factor_n(factor) -> int:
    if _is_local_factor(factor):
        return int(factor["sqrt_z"].shape[0])
    if _is_fisher_factor(factor):
        return int(factor["x"].shape[0])
    return int(_factor_parts(factor)[0].shape[0])


def _factor_weight(factor) -> float:
    if _is_local_factor(factor) or _is_fisher_factor(factor):
        return float(factor.get("weight", 1.0))
    return _factor_parts(factor)[2]


def _local_similarity_block(factor, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    sq = np.asarray(factor["sqrt_z"], dtype=np.float64)
    x = np.asarray(factor["x"], dtype=np.float64)
    inv_cov = np.asarray(factor["inv_cov"], dtype=np.float64)
    delta_scale = float(factor.get("delta_scale", 1.0))

    zr, zc = sq[row_idx], sq[col_idx]
    xr, xc = x[row_idx], x[col_idx]
    sim = np.zeros((len(row_idx), len(col_idx)), dtype=np.float64)
    diff = xr[:, None, :] - xc[None, :, :]
    sqdiff1 = diff[..., 0] * diff[..., 0] if diff.shape[2] == 1 else None

    for k in range(sq.shape[1]):
        gate = np.outer(zr[:, k], zc[:, k])
        if not np.any(gate > 0):
            continue
        if sqdiff1 is None:
            delta = np.einsum("...d,de,...e->...", diff, inv_cov[k], diff)
        else:
            delta = sqdiff1 * inv_cov[k, 0, 0]
        delta = np.maximum(delta, 0.0) / delta_scale
        sim += gate * np.exp(-0.125 * np.minimum(delta, 80.0))

    return sim


def _fisher_similarity_block(factor, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    x = np.asarray(factor["x"], dtype=np.float64)
    xr = x[row_idx]
    xc = x[col_idx]

    rr = np.sum(xr * xr, axis=1, keepdims=True)
    cc = np.sum(xc * xc, axis=1, keepdims=True).T
    d2 = rr + cc - 2.0 * np.dot(xr, xc.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-0.5 * np.minimum(d2, 1400.0))


def _factor_similarity_block(factor, row_idx: np.ndarray, col_idx: np.ndarray | None = None) -> np.ndarray:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if col_idx is None:
        col_idx = np.arange(_factor_n(factor), dtype=np.int64)
    else:
        col_idx = np.asarray(col_idx, dtype=np.int64)

    if _is_local_factor(factor):
        return _local_similarity_block(factor, row_idx, col_idx)
    if _is_fisher_factor(factor):
        return _fisher_similarity_block(factor, row_idx, col_idx)

    g, h, _ = _factor_parts(factor)
    return np.dot(g[row_idx], h[col_idx].T)


def _factor_similarity_candidates(factor, i: int, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if _is_local_factor(factor):
        return _local_similarity_block(factor, np.asarray([i], dtype=np.int64), candidates)[0]
    if _is_fisher_factor(factor):
        return _fisher_similarity_block(factor, np.asarray([i], dtype=np.int64), candidates)[0]

    g, h, _ = _factor_parts(factor)
    return np.dot(h[candidates], g[i])


def _resolve_affinity(
    affinity,
    mix_model,
    data,
    field_weights,
    enc_data=None,
    fisher_metric: str = "diagonal",
    fisher_ridge: float = 1.0e-8,
    fisher_information: str = "observed",
):
    """Resolve named affinity modes to a concrete affinity for the model.

    'auto' uses local statistical factors whenever raw data is available to
    split into the model's leaf fields; if the data cannot be decomposed (or
    no raw data was given) it falls back to 'bhattacharyya' on the joint
    posterior.
    """
    if affinity == "auto":
        if getattr(mix_model, "components", None) is not None and data is not None:
            try:
                return local_factors(mix_model, data, field_weights=field_weights)
            except Exception:
                return "bhattacharyya"
        return "bhattacharyya"

    if affinity == "local":
        if data is None:
            raise ValueError("affinity='local' requires the raw data to extract local field coordinates.")
        return local_factors(mix_model, data, field_weights=field_weights)

    if affinity == "balanced":
        if data is None:
            raise ValueError("affinity='balanced' requires the raw data to extract per-field values.")
        return balanced_factors(mix_model, data, field_weights=field_weights)

    if affinity == "fisher":
        f_data = None if enc_data is not None else data
        return fisher_factors(
            mix_model,
            data=f_data,
            enc_data=enc_data,
            metric=fisher_metric,
            ridge=fisher_ridge,
            information=fisher_information,
        )

    return affinity


def _posteriors_and_loglikes(mix_model, data=None, enc_data=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (posterior_mat, component_log_like_mat), each n x K, for a mixture-like model.

    Uses the model's seq_posterior/seq_component_log_density when available and
    otherwise computes both from the component distributions and log weights,
    which covers mixle.stats finite mixtures and Dirichlet process mixtures alike.
    """
    if enc_data is None:
        if hasattr(mix_model, "dist_to_encoder"):
            enc_data = mix_model.dist_to_encoder().seq_encode(data)
        else:
            enc_data = mix_model.seq_encode(data)

    if hasattr(mix_model, "seq_component_log_density"):
        ll_mat = np.asarray(mix_model.seq_component_log_density(enc_data), dtype=np.float64)
        log_w = getattr(mix_model, "log_w", None)
        if log_w is not None:
            # The component posterior is softmax(ll + log_w); deriving it here avoids a second
            # full model evaluation. seq_posterior re-scores every component, which doubles the
            # work for expensive components (HMM forward-backward, PCFG inside-outside).
            z_mat = ll_mat + np.asarray(log_w, dtype=np.float64).reshape(1, -1)
            z_mat -= z_mat.max(axis=1, keepdims=True)
            np.exp(z_mat, out=z_mat)
            z_mat /= z_mat.sum(axis=1, keepdims=True)
            return z_mat, ll_mat
        if hasattr(mix_model, "seq_posterior"):
            z_mat = np.asarray(mix_model.seq_posterior(enc_data), dtype=np.float64)
            return z_mat, ll_mat

    ll_mat = np.asarray([u.seq_log_density(enc_data) for u in mix_model.components], dtype=np.float64).T
    log_w = np.asarray(mix_model.log_w, dtype=np.float64).reshape(1, -1)

    z_mat = ll_mat + log_w
    z_mat -= z_mat.max(axis=1, keepdims=True)
    np.exp(z_mat, out=z_mat)
    z_mat /= z_mat.sum(axis=1, keepdims=True)

    return z_mat, ll_mat


def model_log_affinity(
    posterior_mat: np.ndarray,
    ll_mat: np.ndarray | None = None,
    affinity: str = "bhattacharyya",
    evidence_cap: float | None = None,
) -> np.ndarray:
    """Dense n x n matrix of log affinities (see module docstring) with -inf diagonal.

    Rows are comparable up to a per-row shift, which both the row-conditional
    normalization and per-row perplexity calibration are invariant to.

    evidence_cap bounds the dissimilarity evidence any single factor (field)
    may contribute: each factor's log affinity is floored at -evidence_cap
    nats before the factors are summed. Without the cap a single sharp field
    with (near-)disjoint per-field posteriors drives its log affinity to -inf
    and vetoes the pair no matter what every other field says; with it, a
    field can at most testify "these differ by evidence_cap nats". The cap is
    only applied to multi-factor (per-field) affinities - for a single factor
    it could only create ties.
    """
    factors = _affinity_factors(posterior_mat, ll_mat, affinity)
    n = _factor_n(factors[0])
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    log_s = np.zeros((n, n))
    with np.errstate(divide="ignore"):
        idx = np.arange(n, dtype=np.int64)
        for factor in factors:
            weight = _factor_weight(factor)
            if weight == 0.0:
                continue
            if _factor_n(factor) != n:
                raise ValueError("affinity factor arrays must have compatible row counts.")
            term = np.log(np.maximum(_factor_similarity_block(factor, idx), 1.0e-300))
            if cap is not None:
                np.maximum(term, -cap, out=term)
            log_s += weight * term
    log_s[np.arange(n), np.arange(n)] = -np.inf

    return log_s


def mixture_coordinates(mix_model, data, field_weights=None) -> dict:
    """The observation decomposition made first-class: ``x -> (posterior, remainder per field)``.

    The mixture describes every observation at two levels, and this returns both explicitly:
    ``"posterior"`` -- the (n, K) component posterior, literally barycentric coordinates on the
    simplex whose vertices are the components (the between-cluster geometry); ``"fields"`` -- one
    entry per flattened leaf field with its per-component log-densities, its within-component
    coordinates (``"coords"``: native value coordinates where the leaf has them, universal
    typicality coordinates otherwise), and ``"native"`` recording which. This is exactly the
    decomposition the 'local' affinity is built from; exposing it lets a caller inspect or plot the
    two levels directly (e.g. a ternary plot of the posterior for K=3) instead of trusting the
    embedding blindly.
    """
    data = list(data)
    z, _ = _posteriors_and_loglikes(mix_model, data=data)
    fields = []
    for l_f, x_f, native_f in _field_log_density_features(list(mix_model.components), data):
        fields.append({"log_density": np.asarray(l_f, dtype=np.float64), "coords": x_f, "native": bool(native_f)})
    if field_weights is not None and len(field_weights) != len(fields):
        raise ValueError(f"field_weights has {len(field_weights)} entries but the model flattens to {len(fields)}.")
    return {"posterior": z, "fields": fields}


def _classical_mds(d2: np.ndarray, emb_dim: int) -> np.ndarray:
    k = d2.shape[0]
    j = np.eye(k) - np.ones((k, k)) / k
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:emb_dim]
    coords = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0.0))
    flips = np.sign(coords[np.abs(coords).argmax(axis=0), np.arange(coords.shape[1])])
    flips[flips == 0] = 1.0
    coords = coords * flips  # canonical eigenvector signs: determinism must not depend on LAPACK
    if coords.shape[1] < emb_dim:
        coords = np.column_stack([coords, np.zeros((k, emb_dim - coords.shape[1]))])
    return coords


def _smacof(d: np.ndarray, emb_dim: int, n_iter: int = 300) -> np.ndarray:
    """Deterministic stress majorization (uniform weights, classical-MDS init) toward target
    distances ``d`` -- the Guttman transform, no randomness anywhere."""
    k = d.shape[0]
    if k == 1:
        return np.zeros((1, emb_dim))
    x = _classical_mds(np.square(d), emb_dim)
    for _ in range(int(n_iter)):
        cur = np.sqrt(np.maximum(np.square(x[:, None, :] - x[None, :, :]).sum(axis=2), 1.0e-12))
        np.fill_diagonal(cur, 1.0)
        ratio = d / cur
        np.fill_diagonal(ratio, 0.0)
        b = -ratio
        b[np.arange(k), np.arange(k)] = ratio.sum(axis=1)
        x_new = (b @ x) / k
        if float(np.abs(x_new - x).max()) < 1.0e-10:
            x = x_new
            break
        x = x_new
    return x - x.mean(axis=0, keepdims=True)


def component_map(
    z: np.ndarray, emb_dim: int = 2, *, method: str = "nerve", edge_threshold: float = 0.02
) -> np.ndarray:
    """Lay out the K components as vertices by their overlap geometry on the data.

    ``method='nerve'`` (default): geodesic layout of the cover's nerve -- edge lengths are
    ``-log BC`` on STRONG edges only (see :func:`mixle.utils.hvis.topology.fuzzy_nerve`), all-pairs
    shortest paths give the target metric, and deterministic stress majorization embeds it. This is
    Isomap on the nerve: a ring of components renders as a ring and a chain as a line, where bare
    MDS on the clipped dense ``-log BC`` matrix (every non-overlapping pair saturating at the same
    huge distance) distorts both -- the classic horseshoe failure. Disconnected pieces of the nerve
    are laid out separately and placed side by side with an explicit gap; their on-screen separation
    is a RENDERING choice, which :func:`mixle.utils.hvis.topology.nerve_report` also says outright.

    ``method='mds'``: the previous behavior -- classical MDS on the dense clipped ``-log BC``
    matrix. Kept as the fallback and for comparison.

    Component confusability itself is unchanged: the Bhattacharyya coefficient between the
    components' responsibility profiles. These vertices anchor :func:`barycentric_init` and
    :func:`mixle.utils.hvis.direct.model_map`.
    """
    z = np.asarray(z, dtype=np.float64)
    k = z.shape[1]
    if k == 1:
        return np.zeros((1, emb_dim))
    sq = np.sqrt(z)
    if method not in ("nerve", "mds"):
        raise ValueError("method must be 'nerve' or 'mds'.")
    strong = None
    if method == "nerve":
        from mixle.utils.hvis.topology import fuzzy_nerve

        nerve = fuzzy_nerve(z, edge_threshold=edge_threshold)
        strong = {e for e, w in nerve["edges"].items() if w >= edge_threshold}
    return _component_map_core(sq.T @ sq, z.sum(axis=0), strong, emb_dim, method)


def _component_map_core(
    sqrt_overlap: np.ndarray, masses: np.ndarray, strong: set | None, emb_dim: int, method: str
) -> np.ndarray:
    """:func:`component_map` finalization from additive statistics (``sqrt(z)^T sqrt(z)``, ``sum z``,
    and the strong-edge set). Shared with :mod:`mixle.utils.hvis.distributed`, whose shards compute
    the same statistics chunk-wise."""
    k = sqrt_overlap.shape[0]
    mass = np.sqrt(np.maximum(masses, 1.0e-12))
    bc = sqrt_overlap / np.outer(mass, mass)
    np.clip(bc, 1.0e-12, 1.0, out=bc)
    neg_log_bc = -np.log(bc)
    np.fill_diagonal(neg_log_bc, 0.0)

    if method == "mds":
        return _classical_mds(np.square(neg_log_bc), emb_dim)
    if not strong:  # no measured overlaps at all: fall back to the dense view
        return _classical_mds(np.square(neg_log_bc), emb_dim)

    import scipy.sparse
    import scipy.sparse.csgraph

    rows = [a for a, _b in strong] + [b for _a, b in strong]
    cols = [b for _a, b in strong] + [a for a, _b in strong]
    lengths = [max(float(neg_log_bc[a, b]), 1.0e-6) for a, b in strong] * 2
    graph = scipy.sparse.csr_matrix((lengths, (rows, cols)), shape=(k, k))
    geo = scipy.sparse.csgraph.shortest_path(graph, directed=False)

    n_pieces, piece_of = scipy.sparse.csgraph.connected_components(graph, directed=False)
    coords = np.zeros((k, emb_dim))
    offset = 0.0
    typical = float(np.median([length for length in lengths])) if lengths else 1.0
    for piece in range(n_pieces):
        members = np.flatnonzero(piece_of == piece)
        sub = _smacof(geo[np.ix_(members, members)], emb_dim)
        half_width = float(sub[:, 0].max() - sub[:, 0].min()) / 2.0 if len(members) > 1 else 0.0
        offset += half_width
        sub = sub + np.array([offset] + [0.0] * (emb_dim - 1))
        coords[members] = sub
        offset += half_width + 1.5 * max(typical, 1.0)  # explicit rendering gap between pieces
    return coords - coords.mean(axis=0, keepdims=True)


def barycentric_init(z: np.ndarray, emb_dim: int = 2, *, jitter: float = 0.15, seed: int | None = None) -> np.ndarray:
    """Initial embedding coordinates from the barycentric reading of the posterior.

    Each observation starts at ``z @ vertices`` -- its posterior-weighted combination of the
    component vertices from :func:`component_map` -- so the layout's GLOBAL arrangement (which
    clusters sit near which, where mixed-membership points fall) is decided by the model's own
    geometry rather than by the random seed, and t-SNE's optimization refines locally from there.

    ``jitter`` is a fraction of the smallest nonzero inter-vertex distance and matters more than it
    looks: sharp posteriors put every same-regime point EXACTLY on its vertex, and t-SNE from
    near-coincident starts is chaotic (microscopic noise decides the layout) and slow to develop
    local structure. The decomposition needs both levels even at init time -- the barycentric base
    supplies the between geometry, the jitter stands in for the within spread the optimization then
    makes real. Rescaled to the conventional 1e-4 standard deviation so optimizer dynamics (early
    exaggeration, learning rates) match the random-init path.
    """
    z = np.asarray(z, dtype=np.float64)
    vertices = component_map(z, emb_dim=emb_dim)
    y = z @ vertices
    gaps = [
        float(np.linalg.norm(vertices[a] - vertices[b]))
        for a in range(len(vertices))
        for b in range(a + 1, len(vertices))
    ]
    nonzero = [g for g in gaps if g > 0]
    scale = min(nonzero) if nonzero else 1.0
    rng = np.random.RandomState(seed)
    y = y + rng.randn(*y.shape) * (float(jitter) * scale)
    spread = float(y.std())
    if spread > 0:
        y = y * (1.0e-4 / spread)
    return y


def log_affinity_block(
    factors, row_idx: np.ndarray, col_idx: np.ndarray, evidence_cap: float | None = None
) -> np.ndarray:
    """Rectangular (rows x cols) log-affinity block -- :func:`model_log_affinity` for a sub-block.

    Mirrors the square path exactly: per-factor similarity blocks, log, per-factor evidence cap
    (multi-factor affinities only), weighted sum. Used by streaming placement (new points x
    landmarks) and by :func:`affinity_health` (subsampled diagnostics).
    """
    row_idx = np.asarray(row_idx, dtype=np.int64)
    col_idx = np.asarray(col_idx, dtype=np.int64)
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None
    log_s = np.zeros((len(row_idx), len(col_idx)))
    with np.errstate(divide="ignore"):
        for factor in factors:
            weight = _factor_weight(factor)
            if weight == 0.0:
                continue
            term = np.log(np.maximum(_factor_similarity_block(factor, row_idx, col_idx), 1.0e-300))
            if cap is not None:
                np.maximum(term, -cap, out=term)
            log_s += weight * term
    return log_s


def affinity_health(
    mix_model,
    data,
    *,
    affinity="auto",
    perplexity: float | None = 30.0,
    field_weights=None,
    evidence_cap: float | None = 1.0,
    max_rows: int = 400,
    seed: int = 0,
) -> dict:
    """Receipts for "why does my embedding look like this": measure the affinity's degeneracies
    BEFORE spending an optimization on them.

    The classic failure this catches is posterior collapse: sharp posteriors make every
    same-component pair an exact tie, rows cannot reach the requested perplexity, and t-SNE renders
    each cluster as a tiny structureless point. That is a property of the AFFINITY, measurable in
    milliseconds -- not a property of the optimizer, discoverable after a thousand iterations.

    Returns a dict with per-field entries (``geometry``: ``'local'``/``'fisher'``/
    ``'posterior-only'``; ``posterior_sharpness``: mean max field-posterior, 1.0 = fully hard) and
    overall numbers on a row subsample of at most ``max_rows``:

    * ``top_tie_fraction`` -- mean fraction of each row's neighbors tied (within 1e-9) with its
      best neighbor. Near 0 is healthy; large means nearest-neighbor structure is degenerate.
    * ``row_entropy_deficit_nats`` -- mean shortfall between the requested ``log(perplexity)`` and
      the entropy each row can actually reach (ties saturate the calibration). 0 is healthy.
    * ``diagnosis`` -- plain-language findings, empty when healthy.
    """
    data = list(data)
    resolved = _resolve_affinity(affinity, mix_model, data, field_weights)
    if isinstance(resolved, str):
        z, ll = _posteriors_and_loglikes(mix_model, data=data)
        factors = _affinity_factors(z, ll, resolved)
        mode = resolved
    else:
        factors = _affinity_factors(None, None, resolved)
        mode = "local" if any(_is_local_factor(f) for f in resolved) else "prebuilt"

    fields = []
    for factor in factors:
        if _is_local_factor(factor):
            geometry, sq = "local", np.asarray(factor["sqrt_z"], dtype=np.float64)
        elif _is_fisher_factor(factor):
            geometry, sq = "fisher", None
        else:
            geometry, sq = "posterior-only", np.asarray(_factor_parts(factor)[0], dtype=np.float64)
        sharpness = float(np.mean(np.max(sq**2, axis=1))) if sq is not None else None
        fields.append({"geometry": geometry, "posterior_sharpness": sharpness})

    n = _factor_n(factors[0])
    rng = np.random.RandomState(seed)
    idx = np.arange(n) if n <= max_rows else np.sort(rng.choice(n, size=max_rows, replace=False))
    log_s = log_affinity_block(factors, idx, idx, evidence_cap)
    np.fill_diagonal(log_s, -np.inf)

    tie_fracs = np.empty(len(idx))
    deficits = np.empty(len(idx))
    target = None if perplexity is None else np.log(min(float(perplexity), max(1.0, len(idx) - 1.0)))
    for i in range(len(idx)):
        row = log_s[i]
        finite = np.isfinite(row)
        if not np.any(finite):
            tie_fracs[i], deficits[i] = 1.0, 0.0
            continue
        vals = row[finite]
        tie_fracs[i] = float(np.mean(vals >= vals.max() - 1.0e-9))
        if target is None:
            deficits[i] = 0.0
        else:
            p = _calibrate_row(vals.copy(), target)
            h = -float(np.dot(p, np.log(np.maximum(p, 1.0e-300))))
            deficits[i] = max(0.0, target - h)

    top_tie = float(tie_fracs.mean())
    deficit = float(deficits.mean())

    diagnosis = []
    if top_tie > 0.05:
        diagnosis.append(
            f"degenerate neighborhoods: on average {top_tie:.0%} of a row's neighbors are exact ties with its "
            "nearest -- pairs the affinity cannot rank. Expect collapsed blobs; prefer affinity='local' "
            "(every field carries within-component geometry) over posterior-only modes."
        )
    if deficit > 0.25:
        diagnosis.append(
            f"rows fall {deficit:.2f} nats short of the requested perplexity (tie groups saturate the "
            "calibration) -- clusters will render smaller and denser than the perplexity asks for."
        )
    for f_pos, field in enumerate(fields):
        if field["geometry"] == "posterior-only" and (field["posterior_sharpness"] or 0.0) > 0.9:
            diagnosis.append(
                f"field {f_pos} is posterior-only with near-hard posteriors (sharpness "
                f"{field['posterior_sharpness']:.3f}): it contributes cluster identity but no within-cluster "
                "geometry."
            )

    return {
        "mode": mode,
        "n": n,
        "n_sampled": len(idx),
        "fields": fields,
        "top_tie_fraction": top_tie,
        "row_entropy_deficit_nats": deficit,
        "diagnosis": diagnosis,
    }


def _hbeta(neg_d: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """Entropy (nats) and probabilities of p_j ~ exp(neg_d_j * beta) for one row."""
    p = neg_d * beta
    p -= p.max()
    np.exp(p, out=p)
    p /= p.sum()
    h = -np.dot(p, np.log(np.maximum(p, 1.0e-300)))
    return h, p


def _calibrate_row(
    neg_d: np.ndarray, target_entropy: float, tol: float = 1.0e-5, max_iter: int = 64, beta_cap: float = 1.0e12
) -> np.ndarray:
    """Binary-search a precision beta so the row entropy hits target_entropy.

    Model-based affinities can contain large groups of (near-)ties, in which
    case entropies below log(tie-group size) are unreachable; beta is capped so
    the search saturates gracefully at the sharpest achievable distribution.
    """
    beta, beta_min, beta_max = 1.0, 0.0, np.inf
    h, p = _hbeta(neg_d, beta)

    for _ in range(max_iter):
        if abs(h - target_entropy) < tol or beta >= beta_cap:
            break
        if h > target_entropy:  # too flat -> increase precision
            beta_min = beta
            beta = beta * 2.0 if np.isinf(beta_max) else (beta + beta_max) / 2.0
            beta = min(beta, beta_cap)
        else:
            beta_max = beta
            beta = beta / 2.0 if beta_min == 0.0 else (beta + beta_min) / 2.0
        h_new, p = _hbeta(neg_d, beta)
        if h_new == h and beta_max - beta_min < 1.0e-12 * max(beta, 1.0):
            break
        h = h_new

    return p


def conditional_pmat(log_aff: np.ndarray, perplexity: float | None = None) -> np.ndarray:
    """Row-conditional probabilities p_{j|i} from log affinities (diagonal -inf).

    With perplexity set, each row is calibrated so its entropy equals
    log(perplexity); otherwise the raw row softmax is used.
    """
    log_aff = np.asarray(log_aff, dtype=np.float64)
    n = log_aff.shape[0]
    if log_aff.ndim != 2 or log_aff.shape[1] != n:
        raise ValueError("log_aff must be a square matrix.")
    if n < 2:
        raise ValueError("at least two observations are required.")

    finite = np.isfinite(log_aff)

    if perplexity is None:
        p = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            cols = finite[i]
            if np.any(cols):
                row = log_aff[i, cols] - log_aff[i, cols].max()
                p[i, cols] = np.exp(row)
                p[i, cols] /= p[i, cols].sum()
            else:
                p[i, :] = 1.0 / (n - 1)
                p[i, i] = 0.0
        return p

    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")

    target_entropy = np.log(perplexity)
    p = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n)
    for i in range(n):
        cols = idx[finite[i]]
        if len(cols) == 0:
            # the model gives this row no information (e.g. a posterior with
            # support disjoint from every other point): fall back to uniform
            p[i, :] = 1.0 / (n - 1)
            p[i, i] = 0.0
        else:
            p[i, cols] = _calibrate_row(log_aff[i, cols].copy(), target_entropy)
    return p


def get_pmat(
    posterior_mat,
    ll_mat=None,
    targ_perplexity=None,
    vlen=False,
    affinity: str = "bhattacharyya",
    evidence_cap: float | None = None,
):
    """Symmetrized t-SNE input probabilities from model posteriors (and optionally
    component log-likelihoods, for affinity='likelihood').

    The vlen flag is kept for backward compatibility and ignored.
    """
    log_s = model_log_affinity(posterior_mat, ll_mat, affinity=affinity, evidence_cap=evidence_cap)
    p = conditional_pmat(log_s, perplexity=targ_perplexity)
    p = (p + p.T) / (2.0 * p.shape[0])
    return p
