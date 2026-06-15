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


def _field_log_density_features(dists, items):
    """Yield (log_density_matrix, feature_matrix_or_None) for leaf fields.

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
            for l_e, x_e in _field_log_density_features([d.dist for d in dists], elems):
                l_f = np.zeros((n, K))
                np.add.at(l_f, seg, l_e)
                if getattr(dists[0], "len_normalized", False):
                    denom = np.asarray(lens, dtype=np.float64)
                    mask = denom > 0
                    l_f[mask] /= denom[mask, None]
                x_f = None
                if x_e is not None:
                    x_f = np.zeros((n, x_e.shape[1]), dtype=np.float64)
                    np.add.at(x_f, seg, x_e)
                    denom = np.asarray(lens, dtype=np.float64)
                    mask = denom > 0
                    x_f[mask] /= denom[mask, None]
                yield l_f, x_f
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
            yield np.where(miss[:, None], lp0[None, :], lp1[None, :]), None
        if (~miss).any():
            fill = items[int(np.argmax(~miss))]
            sub = [fill if m else x for x, m in zip(items, miss)]
            keep = (~miss).astype(np.float64)[:, None]
            for l_in, _ in _field_log_density_features([d.dist for d in dists], sub):
                yield np.where(keep > 0.0, l_in, 0.0), None
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
    yield l, _leaf_feature_matrix(dists, items)


def _field_log_densities(dists, items):
    for l_f, _ in _field_log_density_features(dists, items):
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
    posterior. If the leaf has meaningful local coordinates (continuous/count
    leaves, and averages of such leaves inside sequences), the factor also
    carries component-local inverse covariances estimated from the realized
    data. Pair affinities then use

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
    for (l_f, x_f), w_f in zip(terms, field_weights):
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
            factors.append(
                {
                    "kind": "local",
                    "sqrt_z": sq,
                    "x": x_f,
                    "inv_cov": _component_inv_covariances(x_f, z_f),
                    "weight": float(w_f),
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
        delta = np.maximum(delta, 0.0)
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
    which covers pysp.stats finite mixtures and Dirichlet process mixtures alike.
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
