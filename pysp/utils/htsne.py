"""Model-based (hierarchical) t-SNE and UMAP for heterogeneous data.

Pairwise affinities are derived from a fitted mixture model rather than from
Euclidean distances, so anything pysparkplug can model (tuples, sequences,
sets, variable-length data, ...) can be embedded. Five affinity definitions
are supported (the `affinity` argument):

- 'local' (the 'auto' default whenever raw data is available): the model is
  flattened into leaf fields and each field contributes a local statistical
  affinity. Discrete fields use the per-field posterior Bhattacharyya
  geometry; continuous/count fields additionally use a component-local
  Mahalanobis metric in sufficient-statistic-like coordinates learned from
  the realized data. Thus the same component is no longer a zero-distance
  quotient: within-component neighborhoods are resolved when the field has
  actual local structure.

- 'balanced': the model
  is flattened into its leaf fields (nested composites, sequence
  element/length models, and optional wrappers all decompose), a
  field-restricted posterior z^f is computed from each field's likelihoods
  alone, and the pair distance is the sum over fields of per-field
  Bhattacharyya distances -log sum_k sqrt(z^f_ik z^f_jk), each Winsorized at
  `evidence_cap` nats. The per-field posteriors keep every field's structure
  visible regardless of its likelihood scale (a 15-token sequence field
  contributes ~17 nats of contrast per observation, an overlapping Gaussian
  fractions of one - the joint posterior only ever sees the loudest field),
  and the cap bounds each field's influence so one spuriously sharp field
  cannot veto a pair's similarity that every other field supports.

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
"""
import sys
from typing import Optional, Tuple

import numpy as np
import scipy.sparse

__all__ = ['htsne', 'humap', 'dpmsne', 'model_log_affinity', 'sparse_model_distances',
           'approx_sparse_model_distances', 'model_knn', 'get_pmat', 'balanced_factors',
           'local_factors', 'tsne_barnes_hut']


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
        return factors                       # pre-built factor list

    z = np.asarray(posterior_mat, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("posterior_mat must be a two-dimensional array.")

    if affinity == 'coassign':
        return [(z, z)]
    if affinity == 'bhattacharyya':
        zs = np.sqrt(z)
        return [(zs, zs)]
    if affinity == 'likelihood':
        if ll_mat is None:
            raise ValueError("affinity='likelihood' requires the component log-likelihood matrix.")
        l = np.asarray(ll_mat, dtype=np.float64)
        if l.shape != z.shape:
            raise ValueError("ll_mat must have the same shape as posterior_mat.")
        return [(np.exp(l - l.max(axis=1, keepdims=True)), z)]

    raise ValueError("affinity must be 'coassign', 'bhattacharyya', 'likelihood', "
                     "'local', 'balanced', or a pre-built factor list.")


def _leaf_feature_matrix(dists, items):
    """Local coordinates for supported scalar/vector leaves, or None.

    These coordinates are not a global feature embedding. They are only used
    inside component-local covariance estimates, so unsupported/discrete leaves
    correctly fall back to posterior geometry.
    """
    tname = type(dists[0]).__name__
    try:
        if tname == 'GaussianDistribution':
            return np.asarray(items, dtype=np.float64).reshape(-1, 1)
        if tname == 'DiagonalGaussianDistribution':
            return np.asarray(items, dtype=np.float64)
        if tname == 'LogGaussianDistribution':
            x = np.asarray(items, dtype=np.float64)
            if np.any(x <= 0):
                return None
            return np.log(x).reshape(-1, 1)
        if tname == 'GammaDistribution':
            x = np.asarray(items, dtype=np.float64)
            if np.any(x <= 0):
                return None
            return np.column_stack((x, np.log(x)))
        if tname in ('PoissonDistribution', 'ExponentialDistribution',
                     'GeometricDistribution', 'BinomialDistribution'):
            return np.asarray(items, dtype=np.float64).reshape(-1, 1)
    except (TypeError, ValueError):
        return None
    return None


def _field_log_density_features(dists, items):
    """Yield (log_density_matrix, feature_matrix_or_None) for leaf fields.

    - composite records recurse into their child fields (nested composites
      flatten all the way down),
    - sequences score each child field summed over the sequence's elements,
      with the length model contributing its own field,
    - optional wrappers contribute a missing-ness field, with the inner
      distribution's fields scored only on rows where the value is present,
    - ignored/null distributions contribute nothing,
    - everything else (Gaussian, categorical, Markov chains, ...) is a leaf
      scored with its own seq_log_density.
    """
    tname = type(dists[0]).__name__
    n, K = len(items), len(dists)

    if 'Ignored' in tname or 'Null' in tname:
        return

    if 'Composite' in tname and hasattr(dists[0], 'dists'):
        for f in range(len(dists[0].dists)):
            yield from _field_log_density_features([d.dists[f] for d in dists],
                                                   [x[f] for x in items])
        return

    if 'Sequence' in tname and hasattr(dists[0], 'dist') and hasattr(dists[0], 'len_dist'):
        lens = [len(x) for x in items]
        elems = [e for x in items for e in x]
        if elems:
            seg = np.repeat(np.arange(n), lens)
            for l_e, x_e in _field_log_density_features([d.dist for d in dists], elems):
                l_f = np.zeros((n, K))
                np.add.at(l_f, seg, l_e)
                if getattr(dists[0], 'len_normalized', False):
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

    if 'Optional' in tname and hasattr(dists[0], 'dist'):
        mv = getattr(dists[0], 'missing_value', None)
        mv_is_nan = isinstance(mv, float) and np.isnan(mv)
        miss = np.asarray([x is None or x is mv or
                           (mv_is_nan and isinstance(x, float) and np.isnan(x))
                           for x in items])
        has_gate = [getattr(d, 'has_p', True) for d in dists]
        if any(has_gate):
            lp0 = np.asarray([
                getattr(d, 'log_p0', getattr(d, 'log_p', 0.0)) if has_p else 0.0
                for d, has_p in zip(dists, has_gate)
            ], dtype=np.float64)                  # log P(missing)
            lp1 = np.asarray([
                getattr(d, 'log_p1', getattr(d, 'log_pn', 0.0)) if has_p else 0.0
                for d, has_p in zip(dists, has_gate)
            ], dtype=np.float64)                  # log P(present)
            yield np.where(miss[:, None], lp0[None, :], lp1[None, :]), None
        if (~miss).any():
            fill = items[int(np.argmax(~miss))]
            sub = [fill if m else x for x, m in zip(items, miss)]
            keep = (~miss).astype(np.float64)[:, None]
            for l_in, _ in _field_log_density_features([d.dist for d in dists], sub):
                yield l_in * keep, None
        return

    if hasattr(dists[0], 'dist_to_encoder'):
        enc = dists[0].dist_to_encoder().seq_encode(items)
    elif hasattr(dists[0], 'seq_encode'):
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
    log-likelihood contrast across components - sharp categorical or
    token-sequence fields contribute many nats per observation while
    overlapping continuous fields contribute fractions of one, or a collapsed
    continuous component contributes thousands. The drowned fields'
    relationships then become invisible to any affinity computed from the
    joint posterior.

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
        raise ValueError('field_weights has %d entries but the model flattens to %d '
                         'leaf fields.' % (len(field_weights), len(l_fields)))

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
        raise ValueError('field_weights has %d entries but the model flattens to %d '
                         'leaf fields.' % (len(field_weights), len(terms)))

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
            factors.append({
                'kind': 'local',
                'sqrt_z': sq,
                'x': x_f,
                'inv_cov': _component_inv_covariances(x_f, z_f),
                'weight': float(w_f),
            })

    return factors


def _factor_parts(factor):
    if isinstance(factor, dict):
        raise ValueError("local affinity factors are not dot-product factors.")
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
    return isinstance(factor, dict) and factor.get('kind') == 'local'


def _factor_n(factor) -> int:
    if _is_local_factor(factor):
        return int(factor['sqrt_z'].shape[0])
    return int(_factor_parts(factor)[0].shape[0])


def _factor_weight(factor) -> float:
    if _is_local_factor(factor):
        return float(factor.get('weight', 1.0))
    return _factor_parts(factor)[2]


def _local_similarity_block(factor, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    sq = np.asarray(factor['sqrt_z'], dtype=np.float64)
    x = np.asarray(factor['x'], dtype=np.float64)
    inv_cov = np.asarray(factor['inv_cov'], dtype=np.float64)

    zr, zc = sq[row_idx], sq[col_idx]
    xr, xc = x[row_idx], x[col_idx]
    sim = np.zeros((len(row_idx), len(col_idx)), dtype=np.float64)

    for k in range(sq.shape[1]):
        gate = np.outer(zr[:, k], zc[:, k])
        if not np.any(gate > 0):
            continue
        diff = xr[:, None, :] - xc[None, :, :]
        delta = np.einsum('...d,de,...e->...', diff, inv_cov[k], diff)
        delta = np.maximum(delta, 0.0)
        sim += gate * np.exp(-0.125 * np.minimum(delta, 80.0))

    return sim


def _factor_similarity_block(factor, row_idx: np.ndarray, col_idx: Optional[np.ndarray] = None) -> np.ndarray:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if col_idx is None:
        col_idx = np.arange(_factor_n(factor), dtype=np.int64)
    else:
        col_idx = np.asarray(col_idx, dtype=np.int64)

    if _is_local_factor(factor):
        return _local_similarity_block(factor, row_idx, col_idx)

    g, h, _ = _factor_parts(factor)
    return np.dot(g[row_idx], h[col_idx].T)


def _factor_similarity_candidates(factor, i: int, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if _is_local_factor(factor):
        return _local_similarity_block(factor, np.asarray([i], dtype=np.int64), candidates)[0]

    g, h, _ = _factor_parts(factor)
    return np.dot(h[candidates], g[i])


def _resolve_affinity(affinity, mix_model, data, field_weights):
    """Resolve 'auto'/'local'/'balanced' to a concrete affinity for the model.

    'auto' uses local statistical factors whenever raw data is available to
    split into the model's leaf fields; if the data cannot be decomposed (or
    no raw data was given) it falls back to 'bhattacharyya' on the joint
    posterior.
    """
    if affinity == 'auto':
        if getattr(mix_model, 'components', None) is not None and data is not None:
            try:
                return local_factors(mix_model, data, field_weights=field_weights)
            except Exception:
                return 'bhattacharyya'
        return 'bhattacharyya'

    if affinity == 'local':
        if data is None:
            raise ValueError("affinity='local' requires the raw data to extract local field coordinates.")
        return local_factors(mix_model, data, field_weights=field_weights)

    if affinity == 'balanced':
        if data is None:
            raise ValueError("affinity='balanced' requires the raw data to extract per-field values.")
        return balanced_factors(mix_model, data, field_weights=field_weights)

    return affinity


def _posteriors_and_loglikes(mix_model, data=None, enc_data=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (posterior_mat, component_log_like_mat), each n x K, for a mixture-like model.

    Uses the model's seq_posterior/seq_component_log_density when available and
    otherwise computes both from the component distributions and log weights,
    which covers pysp.stats mixtures and pysp.bstats DPM models alike.
    """
    if enc_data is None:
        if hasattr(mix_model, 'dist_to_encoder'):
            enc_data = mix_model.dist_to_encoder().seq_encode(data)
        else:
            enc_data = mix_model.seq_encode(data)

    if hasattr(mix_model, 'seq_component_log_density') and hasattr(mix_model, 'seq_posterior'):
        ll_mat = np.asarray(mix_model.seq_component_log_density(enc_data), dtype=np.float64)
        z_mat = np.asarray(mix_model.seq_posterior(enc_data), dtype=np.float64)
        return z_mat, ll_mat

    ll_mat = np.asarray([u.seq_log_density(enc_data) for u in mix_model.components], dtype=np.float64).T
    log_w = np.asarray(mix_model.log_w, dtype=np.float64).reshape(1, -1)

    z_mat = ll_mat + log_w
    z_mat -= z_mat.max(axis=1, keepdims=True)
    np.exp(z_mat, out=z_mat)
    z_mat /= z_mat.sum(axis=1, keepdims=True)

    return z_mat, ll_mat


def model_log_affinity(posterior_mat: np.ndarray, ll_mat: Optional[np.ndarray] = None,
                       affinity: str = 'bhattacharyya',
                       evidence_cap: Optional[float] = None) -> np.ndarray:
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
    with np.errstate(divide='ignore'):
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


def _hbeta(neg_d: np.ndarray, beta: float) -> Tuple[float, np.ndarray]:
    """Entropy (nats) and probabilities of p_j ~ exp(neg_d_j * beta) for one row."""
    p = neg_d * beta
    p -= p.max()
    np.exp(p, out=p)
    p /= p.sum()
    h = -np.dot(p, np.log(np.maximum(p, 1.0e-300)))
    return h, p


def _calibrate_row(neg_d: np.ndarray, target_entropy: float, tol: float = 1.0e-5,
                   max_iter: int = 64, beta_cap: float = 1.0e12) -> np.ndarray:
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


def conditional_pmat(log_aff: np.ndarray, perplexity: Optional[float] = None) -> np.ndarray:
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


def get_pmat(posterior_mat, ll_mat=None, targ_perplexity=None, vlen=False, affinity: str = 'bhattacharyya',
             evidence_cap: Optional[float] = None):
    """Symmetrized t-SNE input probabilities from model posteriors (and optionally
    component log-likelihoods, for affinity='likelihood').

    The vlen flag is kept for backward compatibility and ignored.
    """
    log_s = model_log_affinity(posterior_mat, ll_mat, affinity=affinity, evidence_cap=evidence_cap)
    p = conditional_pmat(log_s, perplexity=targ_perplexity)
    p = (p + p.T) / (2.0 * p.shape[0])
    return p


def sparse_model_distances(posterior_mat: np.ndarray, ll_mat: Optional[np.ndarray] = None, k: int = 90,
                           block_size: int = 1024, affinity: str = 'bhattacharyya',
                           evidence_cap: Optional[float] = None) -> scipy.sparse.csr_matrix:
    """Sparse n x n matrix of model distances d_ij = -log s_ij.

    Keeps the k nearest neighbors (largest affinity) per row. Built blockwise so
    the dense n x n affinity matrix is never materialized. Distances are
    non-negative. evidence_cap as in model_log_affinity.
    """
    factors = _affinity_factors(posterior_mat, ll_mat, affinity)
    n = _factor_n(factors[0])
    k = min(k, n - 1)
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    if k <= 0:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=np.int64)
    vals = np.empty(n * k, dtype=np.float64)

    for s0 in range(0, n, block_size):
        s1 = min(s0 + block_size, n)
        row_idx = np.arange(s0, s1, dtype=np.int64)
        log_s = np.zeros((s1 - s0, n))
        with np.errstate(divide='ignore'):
            for factor in factors:
                weight = _factor_weight(factor)
                if weight == 0.0:
                    continue
                term = np.log(np.maximum(_factor_similarity_block(factor, row_idx), 1.0e-300))
                if cap is not None:
                    np.maximum(term, -cap, out=term)
                log_s += weight * term
        log_s[np.arange(s1 - s0), np.arange(s0, s1)] = -np.inf
        s_blk = log_s

        nbr = np.argpartition(-s_blk, k - 1, axis=1)[:, :k]
        log_s = s_blk

        for bi, i in enumerate(range(s0, s1)):
            c = nbr[bi]
            d = -log_s[bi, c]
            np.maximum(d, 0.0, out=d)
            cols[i * k:(i + 1) * k] = c
            vals[i * k:(i + 1) * k] = d

    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


class _RPTreeNode:
    __slots__ = ('idx', 'direction', 'threshold', 'left', 'right')

    def __init__(self, idx=None, direction=None, threshold=None, left=None, right=None):
        self.idx = idx
        self.direction = direction
        self.threshold = threshold
        self.left = left
        self.right = right


def _candidate_features(factors) -> Tuple[np.ndarray, np.ndarray]:
    """Feature coordinates used only for approximate neighbor proposals."""
    row_blocks = []
    col_blocks = []
    n = _factor_n(factors[0])

    for factor in factors:
        weight = _factor_weight(factor)
        if weight == 0.0:
            continue

        if _is_local_factor(factor):
            sq = np.asarray(factor['sqrt_z'], dtype=np.float64)
            x = np.asarray(factor['x'], dtype=np.float64)
            mu = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            x = np.where(np.isfinite(x), x, mu)
            sd = np.std(x, axis=0, keepdims=True)
            x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(sd, 1.0e-8)
            rg = np.hstack((sq, 0.25 * x))
            ch = rg
        else:
            g, h, _ = _factor_parts(factor)
            rg = np.nan_to_num(np.asarray(g, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            ch = np.nan_to_num(np.asarray(h, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        rg_norm = np.linalg.norm(rg, axis=1, keepdims=True)
        ch_norm = np.linalg.norm(ch, axis=1, keepdims=True)
        rg = rg / np.maximum(rg_norm, 1.0e-300)
        ch = ch / np.maximum(ch_norm, 1.0e-300)

        scale = np.sqrt(weight)
        row_blocks.append(scale * rg)
        col_blocks.append(scale * ch)

    if not row_blocks:
        z = np.zeros((n, 1), dtype=np.float64)
        return z, z.copy()

    return np.hstack(row_blocks), np.hstack(col_blocks)


def _build_rp_tree(X: np.ndarray, idx: np.ndarray, leaf_size: int,
                   rng: np.random.RandomState, max_depth: int) -> _RPTreeNode:
    if len(idx) <= leaf_size or max_depth <= 0:
        return _RPTreeNode(idx=idx)

    pts = X[idx]
    for _ in range(12):
        direction = rng.randn(X.shape[1])
        dn = np.linalg.norm(direction)
        if dn <= 0:
            continue
        direction /= dn
        proj = np.dot(pts, direction)
        if not np.all(np.isfinite(proj)) or proj.max() - proj.min() <= 1.0e-12:
            continue
        threshold = float(np.median(proj))
        left_mask = proj <= threshold
        if np.any(left_mask) and np.any(~left_mask):
            left = _build_rp_tree(X, idx[left_mask], leaf_size, rng, max_depth - 1)
            right = _build_rp_tree(X, idx[~left_mask], leaf_size, rng, max_depth - 1)
            return _RPTreeNode(direction=direction, threshold=threshold, left=left, right=right)

    return _RPTreeNode(idx=idx)


def _query_rp_tree(node: _RPTreeNode, x: np.ndarray) -> np.ndarray:
    while node.idx is None:
        node = node.left if float(np.dot(x, node.direction)) <= node.threshold else node.right
    return node.idx


def _augment_candidates(candidates: np.ndarray, i: int, n: int, target: int,
                        rng: np.random.RandomState) -> np.ndarray:
    target = min(max(target, 0), n - 1)
    candidates = np.unique(candidates)
    candidates = candidates[candidates != i]

    if len(candidates) > target:
        candidates = rng.choice(candidates, size=target, replace=False)
        return np.asarray(sorted(candidates), dtype=np.int64)
    if len(candidates) >= target:
        return candidates
    if target >= n - 1:
        all_idx = np.arange(n, dtype=np.int64)
        return all_idx[all_idx != i]

    seen = set(int(j) for j in candidates)
    attempts = 0
    while len(seen) < target and attempts < 20:
        needed = target - len(seen)
        extra = rng.randint(0, n, size=max(16, needed * 3))
        for j in extra:
            jj = int(j)
            if jj != i:
                seen.add(jj)
                if len(seen) >= target:
                    break
        attempts += 1

    if len(seen) < target:
        for jj in range(n):
            if jj != i:
                seen.add(jj)
                if len(seen) >= target:
                    break

    return np.asarray(sorted(seen), dtype=np.int64)


def _candidate_log_affinity(factors, i: int, candidates: np.ndarray,
                            cap: Optional[float]) -> np.ndarray:
    log_s = np.zeros(len(candidates), dtype=np.float64)
    with np.errstate(divide='ignore'):
        for factor in factors:
            weight = _factor_weight(factor)
            if weight == 0.0:
                continue
            term = np.log(np.maximum(_factor_similarity_candidates(factor, i, candidates), 1.0e-300))
            if cap is not None:
                np.maximum(term, -cap, out=term)
            log_s += weight * term
    return log_s


def approx_sparse_model_distances(posterior_mat: np.ndarray, ll_mat: Optional[np.ndarray] = None,
                                  k: int = 90, affinity: str = 'bhattacharyya',
                                  evidence_cap: Optional[float] = None, n_trees: int = 8,
                                  leaf_size: Optional[int] = None, candidate_multiplier: int = 8,
                                  seed: Optional[int] = None) -> scipy.sparse.csr_matrix:
    """Approximate sparse model distances without all-pairs graph construction.

    A random-projection forest proposes candidate neighbors in normalized
    model-factor coordinates. Candidate pairs are then rescored with the exact
    model affinity used by sparse_model_distances, so approximation only enters
    through candidate recall. This is local/non-distributed today, but the
    proposal/evaluation split is the intended boundary for future distributed
    graph construction.
    """
    factors = _affinity_factors(posterior_mat, ll_mat, affinity)
    n = _factor_n(factors[0])
    k = min(k, n - 1)
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    if k <= 0:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    if leaf_size is None:
        leaf_size = max(64, 2 * k)
    leaf_size = min(max(int(leaf_size), k), n)
    n_trees = max(int(n_trees), 1)
    target_candidates = min(n - 1, max(k, int(candidate_multiplier) * k, leaf_size * n_trees))

    row_feat, col_feat = _candidate_features(factors)
    rng = np.random.RandomState(seed)
    idx = np.arange(n, dtype=np.int64)
    max_depth = max(1, int(np.ceil(np.log2(max(n / float(leaf_size), 1.0)))) + 2)
    trees = [_build_rp_tree(col_feat, idx, leaf_size, rng, max_depth) for _ in range(n_trees)]

    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=np.int64)
    vals = np.empty(n * k, dtype=np.float64)

    for i in range(n):
        cand_parts = [_query_rp_tree(tree, row_feat[i]) for tree in trees]
        candidates = np.concatenate(cand_parts) if cand_parts else np.empty(0, dtype=np.int64)
        candidates = _augment_candidates(candidates, i, n, target_candidates, rng)

        log_s = _candidate_log_affinity(factors, i, candidates, cap)
        nbr = np.argpartition(-log_s, k - 1)[:k]
        c = candidates[nbr]
        d = -log_s[nbr]
        np.maximum(d, 0.0, out=d)

        s = i * k
        cols[s:s + k] = c
        vals[s:s + k] = d

    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def model_knn(posterior_mat: np.ndarray, ll_mat: Optional[np.ndarray] = None, k: int = 15,
              block_size: int = 1024, affinity: str = 'bhattacharyya',
              evidence_cap: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """k-nearest-neighbor arrays under the model distance d_ij = -log s_ij.

    Returns (indices, distances), each n x k, sorted ascending per row with
    each point as its own first neighbor at distance 0 (the convention
    expected by umap-learn, where self counts toward n_neighbors). Built
    blockwise; the dense affinity matrix is never materialized.
    evidence_cap as in model_log_affinity.
    """
    factors = _affinity_factors(posterior_mat, ll_mat, affinity)
    n = _factor_n(factors[0])
    k = min(k, n)
    m = k - 1  # non-self neighbors
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    knn_idx = np.empty((n, k), dtype=np.int64)
    knn_dist = np.empty((n, k), dtype=np.float64)
    knn_idx[:, 0] = np.arange(n)
    knn_dist[:, 0] = 0.0

    if m == 0:
        return knn_idx, knn_dist

    for s0 in range(0, n, block_size):
        s1 = min(s0 + block_size, n)
        row_idx = np.arange(s0, s1, dtype=np.int64)
        log_s = np.zeros((s1 - s0, n))
        with np.errstate(divide='ignore'):
            for factor in factors:
                weight = _factor_weight(factor)
                if weight == 0.0:
                    continue
                term = np.log(np.maximum(_factor_similarity_block(factor, row_idx), 1.0e-300))
                if cap is not None:
                    np.maximum(term, -cap, out=term)
                log_s += weight * term
        log_s[np.arange(s1 - s0), np.arange(s0, s1)] = -np.inf
        s_blk = log_s

        nbr = np.argpartition(-s_blk, m - 1, axis=1)[:, :m]

        for bi, i in enumerate(range(s0, s1)):
            c = nbr[bi]
            d = -log_s[bi, c]
            np.maximum(d, 0.0, out=d)
            order = np.argsort(d)
            knn_idx[i, 1:] = c[order]
            knn_dist[i, 1:] = d[order]

    return knn_idx, knn_dist


def t_kernel(tx: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heavy-tailed student-t kernel on embedding tx.

    Returns (Q, num, d2): normalized probabilities Q, the gradient weights
    num_ij = 1 / (1 + d_ij^2 / alpha), and squared distances d2. At alpha = 1
    this is the standard t-SNE kernel.
    """
    n = tx.shape[0]
    rsum = np.sum(np.square(tx), axis=1, keepdims=True)
    d2 = np.dot(-2.0 * tx, tx.T)
    d2 += rsum
    d2 += rsum.T
    np.maximum(d2, 0.0, out=d2)

    num = 1.0 / (1.0 + d2 / alpha)
    qt = num ** ((alpha + 1.0) / 2.0)
    qt[np.arange(n), np.arange(n)] = 0.0
    q = qt / qt.sum()

    return q, num, d2


def update_embed(P: np.ndarray, Y: np.ndarray, iY: np.ndarray, gains: np.ndarray,
                 momentum: float, eta: float, alpha: float, min_gain: float,
                 min_value: float = 1.0e-128) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One delta-bar-delta gradient step of KL(P || Q) on the embedding Y.

    Gradient of the heavy-tailed kernel:
        dC/dy_i = (2(alpha+1)/alpha) * sum_j (p_ij - q_ij) num_ij (y_i - y_j),
    computed in matrix form (no per-row Python loop).
    """
    Q, num, _ = t_kernel(Y, alpha)
    np.maximum(Q, min_value, out=Q)

    W = (P - Q) * num
    dC = (np.sum(W, axis=1, keepdims=True) * Y - np.dot(W, Y)) * (2.0 * (alpha + 1.0) / alpha)

    inc = (dC > 0) != (iY > 0)
    gains = np.where(inc, gains + 0.2, gains * 0.8)
    np.maximum(gains, min_gain, out=gains)

    iY = momentum * iY - eta * (gains * dC)
    Y = Y + iY
    Y -= np.mean(Y, axis=0, keepdims=True)

    return Y, iY, gains, Q


def _kl(P: np.ndarray, Q: np.ndarray) -> float:
    m = (P > 0) & (Q > 0)
    return float(np.dot(P[m], np.log(P[m]) - np.log(Q[m])))


def update_alpha(P: np.ndarray, Y: np.ndarray, alpha: float, min_alpha: float,
                 min_value: float, max_its: int = 30, step: float = 0.1,
                 eps: float = 1.0e-6, max_alpha: float = 1.0e6) -> float:
    """Optimize the kernel tail parameter alpha by guarded Newton steps.

    d(-log q~_ij)/d(alpha) = 0.5 log(1 + d2/alpha) - (alpha+1) d2 / (2 alpha^2 (1 + d2/alpha)),
    and dKL/d(alpha) = sum_ij (p_ij - q_ij) d(-log q~_ij)/d(alpha) (the partition
    function term cancels because sum P = sum Q = 1). Steps are clipped to a
    +-step trust region; alpha is kept in [min_alpha, max_alpha].
    """
    Q, num, d2 = t_kernel(Y, alpha)
    np.maximum(Q, min_value, out=Q)
    kl = _kl(P, Q)

    for _ in range(max_its):
        e_a = 1.0 + d2 / alpha
        dlq = 0.5 * np.log(e_a) - ((alpha + 1.0) / (2.0 * alpha * alpha)) * (d2 / e_a)
        g = float(np.sum((P - Q) * dlq))

        if not np.isfinite(g) or g == 0.0:
            break

        prop = alpha - kl / g if g != 0 else alpha
        prop = min(max(prop, alpha * (1.0 - step)), alpha * (1.0 + step))
        new_alpha = min(max(prop, min_alpha), max_alpha)

        if new_alpha == alpha:
            break

        Q, num, d2 = t_kernel(Y, new_alpha)
        np.maximum(Q, min_value, out=Q)
        new_kl = _kl(P, Q)

        if new_kl > kl - eps:
            break
        alpha, kl = new_alpha, new_kl

    return alpha


def tsne_exact(P: np.ndarray, emb_dim: int = 2, alpha: float = 1.0, Y: Optional[np.ndarray] = None,
               max_its: int = 1000, eta: Optional[float] = None, momentum: float = 0.8,
               early_exaggeration: float = 12.0, early_its: int = 250, min_gain: float = 0.01,
               min_value: float = 1.0e-128, optimize_alpha: bool = False, min_alpha: float = 1.0e-6,
               max_alpha_its: int = 3, tol: float = 1.0e-7, check_every: int = 50,
               print_iter: int = 100, seed: Optional[int] = None, out=None) -> np.ndarray:
    """Full-matrix t-SNE on symmetrized probabilities P with convergence stopping."""
    if out is None:
        out = sys.stdout

    P = np.asarray(P, dtype=np.float64).copy()
    P /= P.sum()
    n = P.shape[0]

    if eta is None:
        eta = max(n / early_exaggeration, 50.0)

    if Y is None:
        rng = np.random.RandomState(seed)
        Y = rng.randn(n, emb_dim) * 1.0e-4
    else:
        Y = np.array(Y, dtype=np.float64)
        emb_dim = Y.shape[1]

    iY = np.zeros((n, emb_dim))
    gains = np.ones((n, emb_dim))

    P *= early_exaggeration
    np.maximum(P, min_value, out=P)

    last_kl = np.inf
    for i in range(1, max_its + 1):
        if i == early_its + 1:
            P /= early_exaggeration
            np.maximum(P, min_value, out=P)

        mom = 0.5 if i <= early_its else momentum
        Y, iY, gains, Q = update_embed(P, Y, iY, gains, mom, eta, alpha, min_gain, min_value)

        if optimize_alpha and i > early_its:
            alpha = update_alpha(P, Y, alpha, min_alpha, min_value, max_alpha_its)

        if (i % print_iter) == 0:
            out.write('Iteration %d: alpha = %f, KL(P||Q)=%f\n' % (i, alpha, _kl(P, Q)))

        if i > early_its and (i % check_every) == 0:
            kl = _kl(P, Q)
            if last_kl - kl < tol * max(1.0, abs(last_kl)):
                break
            last_kl = kl

    return Y


def _sparse_conditional_pmat(dist_csr: scipy.sparse.csr_matrix,
                             perplexity: float) -> scipy.sparse.csr_matrix:
    """Row-conditional probabilities on a sparse model-neighbor graph.

    dist_csr stores D_ij = -log s_ij, so calibration is performed on -D_ij,
    exactly matching the dense model-affinity path. No metric-distance
    squaring is applied.
    """
    dist_csr = dist_csr.tocsr().astype(np.float64, copy=True)
    n = dist_csr.shape[0]
    if dist_csr.shape[1] != n:
        raise ValueError("dist_csr must be square.")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")

    data = np.empty_like(dist_csr.data, dtype=np.float64)
    indptr, indices = dist_csr.indptr, dist_csr.indices
    target_entropy = np.log(perplexity)

    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        neg_d = -np.asarray(dist_csr.data[start:end], dtype=np.float64)
        finite = np.isfinite(neg_d)
        if not np.all(finite):
            neg_d = neg_d.copy()
            neg_d[~finite] = -1.0e300
        data[start:end] = _calibrate_row(neg_d, target_entropy)

    return scipy.sparse.csr_matrix((data, indices.copy(), indptr.copy()), shape=dist_csr.shape)


def _sparse_joint_pmat(dist_csr: scipy.sparse.csr_matrix,
                       perplexity: float) -> scipy.sparse.csr_matrix:
    """Symmetrized sparse t-SNE input probabilities from model distances."""
    p_cond = _sparse_conditional_pmat(dist_csr, perplexity)
    p = (p_cond + p_cond.T).tocsr()
    p *= 1.0 / (2.0 * p.shape[0])
    p.setdiag(0.0)
    p.eliminate_zeros()
    total = p.sum()
    if total > 0:
        p *= 1.0 / total
    return p


class _BHNode:
    __slots__ = ('idx', 'bmin', 'bmax', 'center', 'width', 'mass', 'children')

    def __init__(self, idx, bmin, bmax, center, width, children):
        self.idx = idx
        self.bmin = bmin
        self.bmax = bmax
        self.center = center
        self.width = width
        self.mass = len(idx)
        self.children = children


def _build_bh_tree(Y: np.ndarray, idx: np.ndarray, leaf_size: int) -> _BHNode:
    pts = Y[idx]
    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    center = pts.mean(axis=0)
    width = float(np.max(bmax - bmin))

    if len(idx) <= leaf_size or width <= 1.0e-12:
        return _BHNode(idx, bmin, bmax, center, width, None)

    mid = 0.5 * (bmin + bmax)
    codes = np.zeros(len(idx), dtype=np.int64)
    for d in range(Y.shape[1]):
        codes |= ((pts[:, d] > mid[d]).astype(np.int64) << d)

    children = []
    for code in np.unique(codes):
        child_idx = idx[codes == code]
        if len(child_idx) > 0:
            children.append(_build_bh_tree(Y, child_idx, leaf_size))

    if len(children) <= 1:
        return _BHNode(idx, bmin, bmax, center, width, None)

    return _BHNode(idx, bmin, bmax, center, width, children)


def _barnes_hut_negative_forces(Y: np.ndarray, theta: float = 0.5,
                                leaf_size: int = 16) -> Tuple[np.ndarray, float]:
    """Approximate t-SNE repulsive forces and normalization with Barnes-Hut.

    Returns (F, Z) where

        F_i ~= sum_j (1 + ||y_i-y_j||^2)^-2 (y_i - y_j)
        Z   ~= sum_ij (1 + ||y_i-y_j||^2)^-1.

    With theta <= 0 the traversal descends to leaves, giving the exact sums
    up to floating-point order.
    """
    Y = np.asarray(Y, dtype=np.float64)
    n, emb_dim = Y.shape
    if n <= 1:
        return np.zeros_like(Y), 1.0

    theta = max(float(theta), 0.0)
    leaf_size = max(int(leaf_size), 1)
    root = _build_bh_tree(Y, np.arange(n, dtype=np.int64), leaf_size)
    forces = np.zeros_like(Y)
    z_sum = 0.0
    eps = 1.0e-12

    for i in range(n):
        yi = Y[i]
        fi = np.zeros(emb_dim, dtype=np.float64)
        zi = 0.0
        stack = [root]

        while stack:
            node = stack.pop()
            if node.mass == 0:
                continue

            if node.children is None:
                for j in node.idx:
                    if j == i:
                        continue
                    diff = yi - Y[j]
                    d2 = float(np.dot(diff, diff))
                    q = 1.0 / (1.0 + d2)
                    fi += (q * q) * diff
                    zi += q
                continue

            diff = yi - node.center
            d2 = float(np.dot(diff, diff))
            inside = bool(np.all(yi >= node.bmin - eps) and np.all(yi <= node.bmax + eps))

            if (not inside) and d2 > 0.0 and theta > 0.0 and node.width / np.sqrt(d2) < theta:
                q = 1.0 / (1.0 + d2)
                fi += node.mass * (q * q) * diff
                zi += node.mass * q
            else:
                stack.extend(node.children)

        forces[i] = fi
        z_sum += zi

    return forces, max(float(z_sum), 1.0e-300)


def _sparse_positive_forces_from_edges(rows: np.ndarray, cols: np.ndarray, data: np.ndarray,
                                       Y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Exact attractive t-SNE forces over sparse probability edges."""
    if len(data) == 0:
        return np.zeros_like(Y)

    diff = Y[rows] - Y[cols]
    d2 = np.sum(diff * diff, axis=1)
    num = 1.0 / (1.0 + d2)
    weighted = (scale * data * num)[:, None] * diff
    forces = np.zeros_like(Y)
    np.add.at(forces, rows, weighted)
    return forces


def _sparse_positive_forces(P: scipy.sparse.csr_matrix,
                            Y: np.ndarray) -> np.ndarray:
    """Exact attractive t-SNE forces over nonzero sparse probabilities."""
    p = P.tocoo()
    return _sparse_positive_forces_from_edges(p.row, p.col, p.data, Y)


def _sparse_tsne_kl(P: scipy.sparse.csr_matrix, Y: np.ndarray,
                    z_sum: Optional[float] = None) -> float:
    p = P.tocoo()
    if p.nnz == 0:
        return 0.0
    if z_sum is None:
        _, z_sum = _barnes_hut_negative_forces(Y, theta=0.0, leaf_size=1)
    return _sparse_tsne_kl_from_edges(p.row, p.col, p.data, Y, z_sum)


def _sparse_tsne_kl_from_edges(rows: np.ndarray, cols: np.ndarray, data: np.ndarray,
                               Y: np.ndarray, z_sum: float) -> float:
    if len(data) == 0:
        return 0.0
    diff = Y[rows] - Y[cols]
    d2 = np.sum(diff * diff, axis=1)
    q = np.maximum((1.0 / (1.0 + d2)) / z_sum, 1.0e-300)
    pdata = np.maximum(data, 1.0e-300)
    return float(np.dot(pdata, np.log(pdata) - np.log(q)))


def _tsne_barnes_hut_from_p(P: scipy.sparse.csr_matrix, emb_dim: int = 2,
                            max_its: int = 1000, eta: Optional[float] = None,
                            momentum: float = 0.8, early_exaggeration: float = 12.0,
                            early_its: int = 250, min_gain: float = 0.01,
                            tol: float = 1.0e-7, check_every: int = 50,
                            print_iter: int = 100, theta: float = 0.5,
                            leaf_size: int = 16, seed: Optional[int] = None,
                            Y: Optional[np.ndarray] = None, out=None) -> np.ndarray:
    """Barnes-Hut t-SNE on a sparse symmetric probability matrix."""
    if out is None:
        out = sys.stdout

    P = P.tocsr().astype(np.float64, copy=True)
    P.setdiag(0.0)
    P.eliminate_zeros()
    total = P.sum()
    if total <= 0:
        raise ValueError("P must contain positive off-diagonal probabilities.")
    P *= 1.0 / total

    n = P.shape[0]
    if eta is None:
        eta = max(n / early_exaggeration, 50.0)

    if Y is None:
        rng = np.random.RandomState(seed)
        Y = rng.randn(n, emb_dim) * 1.0e-4
    else:
        Y = np.asarray(Y, dtype=np.float64).copy()
        if Y.shape[0] != n:
            raise ValueError("initial embedding Y has the wrong number of rows.")
        emb_dim = Y.shape[1]
    Y -= np.mean(Y, axis=0, keepdims=True)

    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    n_iter = max(int(max_its), int(early_its) + 50)
    last_kl = np.inf
    p_edges = P.tocoo()
    p_rows, p_cols, p_data = p_edges.row, p_edges.col, p_edges.data

    for it in range(1, n_iter + 1):
        exaggeration = early_exaggeration if it <= early_its else 1.0

        pos = _sparse_positive_forces_from_edges(p_rows, p_cols, p_data, Y, exaggeration)
        neg, z_sum = _barnes_hut_negative_forces(Y, theta=theta, leaf_size=leaf_size)
        dC = 4.0 * (pos - neg / z_sum)

        inc = (dC > 0) != (iY > 0)
        gains = np.where(inc, gains + 0.2, gains * 0.8)
        np.maximum(gains, min_gain, out=gains)

        mom = 0.5 if it <= early_its else momentum
        iY = mom * iY - eta * gains * dC
        Y = Y + iY
        Y -= np.mean(Y, axis=0, keepdims=True)

        if (it % print_iter) == 0:
            _, kl_z_sum = _barnes_hut_negative_forces(Y, theta=theta, leaf_size=leaf_size)
            kl = _sparse_tsne_kl_from_edges(p_rows, p_cols, p_data, Y, kl_z_sum)
            out.write('Iteration %d: KL(P||Q)=%f\n' % (it, kl))

        if it > early_its and (it % check_every) == 0:
            _, kl_z_sum = _barnes_hut_negative_forces(Y, theta=theta, leaf_size=leaf_size)
            kl = _sparse_tsne_kl_from_edges(p_rows, p_cols, p_data, Y, kl_z_sum)
            if last_kl - kl < tol * max(1.0, abs(last_kl)):
                break
            last_kl = kl

    return Y


def _tsne_barnes_hut(dist_csr: scipy.sparse.csr_matrix, emb_dim: int, perplexity: float,
                     max_its: int, eta, momentum: float, early_exaggeration: float,
                     min_gain: float, tol: float, print_iter: int, seed: Optional[int],
                     Y: Optional[np.ndarray], out=None, theta: float = 0.5,
                     leaf_size: int = 16) -> np.ndarray:
    P = _sparse_joint_pmat(dist_csr, perplexity)
    return _tsne_barnes_hut_from_p(P, emb_dim=emb_dim, max_its=max_its, eta=eta,
                                  momentum=momentum, early_exaggeration=early_exaggeration,
                                  min_gain=min_gain, tol=tol, print_iter=print_iter,
                                  theta=theta, leaf_size=leaf_size, seed=seed,
                                  Y=Y, out=out)


def tsne_barnes_hut(P: scipy.sparse.csr_matrix, emb_dim: int = 2,
                    max_its: int = 1000, eta: Optional[float] = None,
                    momentum: float = 0.8, early_exaggeration: float = 12.0,
                    min_gain: float = 0.01, tol: float = 1.0e-7,
                    print_iter: int = 100, theta: float = 0.5,
                    leaf_size: int = 16, seed: Optional[int] = None,
                    Y: Optional[np.ndarray] = None, out=None) -> np.ndarray:
    """Embed a precomputed sparse t-SNE probability matrix with Barnes-Hut.

    P must be a symmetric, non-negative affinity/probability matrix. It is
    normalized internally. This function is self-contained and does not call
    sklearn.
    """
    return _tsne_barnes_hut_from_p(P, emb_dim=emb_dim, max_its=max_its, eta=eta,
                                  momentum=momentum, early_exaggeration=early_exaggeration,
                                  min_gain=min_gain, tol=tol, print_iter=print_iter,
                                  theta=theta, leaf_size=leaf_size, seed=seed,
                                  Y=Y, out=out)


def htsne(data, emb_dim: int = 2, alpha: float = 1.0, max_components: int = 50,
          Y: Optional[np.ndarray] = None, perplexity: Optional[float] = 30.0,
          max_its: int = 1000, print_iter: int = 100, eta: Optional[float] = None,
          momentum: float = 0.8, min_gain: float = 0.01, min_value: float = 1.0e-128,
          optimize_alpha: bool = False, min_alpha: float = 1.0e-6, max_alpha_its: int = 3,
          seed: Optional[int] = None, mix_model=None, enc_data=None, method: str = 'auto',
          early_exaggeration: float = 12.0, tol: float = 1.0e-7, dpm_max_its: int = 200,
          affinity='auto', field_weights=None, evidence_cap: Optional[float] = 1.0,
          out=None, variable_length: bool = False, barnes_hut_theta: float = 0.5,
          barnes_hut_leaf_size: int = 16, neighbor_method: str = 'auto',
          neighbor_threshold: int = 5000, neighbor_trees: int = 8,
          neighbor_leaf_size: Optional[int] = None, candidate_multiplier: int = 8):
    """Embed heterogeneous data with model-based t-SNE.

    A mixture model is fit to the data (a Dirichlet process mixture with
    automatically typed components by default, or pass mix_model), pairwise
    affinities are computed from the model, and the affinities are embedded
    with t-SNE.

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
        'bhattacharyya' - Bhattacharyya coefficient between joint posteriors;
            graded even under hard assignments, so embeddings retain
            within-cluster geometry
        'coassign'   - co-assignment probability P(z_i = z_j | x); exact but
            near-binary when posteriors are sharp
        'likelihood' - predictive affinity sum_k p(x_i|theta_k) z_jk

    evidence_cap (default 1.0 nats) bounds the dissimilarity evidence any
    single field may contribute to a pair's distance under multi-field
    affinities: without it, one spuriously sharp field (a serial-number-like
    categorical the model micro-clustered) drives its per-field affinity to
    zero and vetoes the pair's similarity no matter what every other field
    says. None disables the cap; single-field affinities ignore it.

    barnes_hut_theta controls the Barnes-Hut opening angle for method='barnes_hut';
    0.0 gives exact repulsive forces and larger values are faster/coarser.

    neighbor_method controls graph construction for method='barnes_hut':
    'exact' uses blockwise all-pairs top-k, 'approx' uses a random-projection
    candidate forest, and 'auto' switches to 'approx' when n >= neighbor_threshold.

    Returns the n x emb_dim embedding.
    """
    if out is None:
        out = sys.stdout

    if mix_model is None:
        from pysp.utils.automatic import get_dpm_mixture
        mix_model = get_dpm_mixture(data, rng=np.random.RandomState(seed),
                                    max_components=max_components, max_its=dpm_max_its,
                                    print_iter=print_iter, out=out)

    affinity = _resolve_affinity(affinity, mix_model, data, field_weights)

    z_ij, l_ij = _posteriors_and_loglikes(mix_model, data=data, enc_data=enc_data)
    n = z_ij.shape[0]

    if method == 'auto':
        method = 'exact' if (optimize_alpha or n <= 10) else 'barnes_hut'

    if method == 'barnes_hut':
        px = 30.0 if perplexity is None else float(perplexity)
        px = min(px, max(1.0, n - 1.0))
        k = min(n - 1, int(3.0 * px) + 5)
        graph_method = neighbor_method
        if graph_method == 'auto':
            graph_method = 'approx' if n >= neighbor_threshold else 'exact'
        if graph_method == 'exact':
            dist_csr = sparse_model_distances(z_ij, l_ij, k=k, affinity=affinity,
                                              evidence_cap=evidence_cap)
        elif graph_method == 'approx':
            dist_csr = approx_sparse_model_distances(
                z_ij, l_ij, k=k, affinity=affinity, evidence_cap=evidence_cap,
                n_trees=neighbor_trees, leaf_size=neighbor_leaf_size,
                candidate_multiplier=candidate_multiplier, seed=seed)
        else:
            raise ValueError("neighbor_method must be 'auto', 'exact', or 'approx'.")
        return _tsne_barnes_hut(dist_csr, emb_dim, px, max_its, eta, momentum,
                                early_exaggeration, min_gain, tol, print_iter, seed, Y,
                                out=out, theta=barnes_hut_theta,
                                leaf_size=barnes_hut_leaf_size)

    P = get_pmat(z_ij, l_ij, targ_perplexity=perplexity, affinity=affinity,
                 evidence_cap=evidence_cap)
    return tsne_exact(P, emb_dim=emb_dim, alpha=alpha, Y=Y, max_its=max_its, eta=eta,
                      momentum=momentum, early_exaggeration=early_exaggeration,
                      min_gain=min_gain, min_value=min_value, optimize_alpha=optimize_alpha,
                      min_alpha=min_alpha, max_alpha_its=max_alpha_its, tol=tol,
                      print_iter=print_iter, seed=seed, out=out)


def humap(data, emb_dim: int = 2, n_neighbors: int = 15, min_dist: float = 0.1,
          max_components: int = 50, seed: Optional[int] = None, mix_model=None,
          enc_data=None, dpm_max_its: int = 200, print_iter: int = 100,
          affinity='auto', field_weights=None, evidence_cap: Optional[float] = 1.0,
          n_epochs: Optional[int] = None, out=None, **umap_kwargs):
    """Embed heterogeneous data with model-based UMAP.

    The same mixture-model affinities as htsne (see the affinity and
    evidence_cap arguments there), but the k-nearest-neighbor graph of model
    distances -log s_ij is handed to UMAP's fuzzy simplicial set construction
    and layout (umap-learn) instead of t-SNE. Scales like UMAP: the dense
    affinity matrix is never built.

    Extra keyword arguments are passed to umap.UMAP. Returns the n x emb_dim
    embedding.
    """
    try:
        import umap
    except ImportError:
        from pysp.utils.optional_deps import require
        require('umap-learn', 'umap')

    if out is None:
        out = sys.stdout

    if mix_model is None:
        from pysp.utils.automatic import get_dpm_mixture
        mix_model = get_dpm_mixture(data, rng=np.random.RandomState(seed),
                                    max_components=max_components, max_its=dpm_max_its,
                                    print_iter=print_iter, out=out)

    affinity = _resolve_affinity(affinity, mix_model, data, field_weights)

    z_ij, l_ij = _posteriors_and_loglikes(mix_model, data=data, enc_data=enc_data)
    n = z_ij.shape[0]
    k = min(n_neighbors, n - 1)

    knn_idx, knn_dist = model_knn(z_ij, l_ij, k=k, affinity=affinity,
                                  evidence_cap=evidence_cap)

    reducer = umap.UMAP(n_components=emb_dim, n_neighbors=k, min_dist=min_dist,
                        precomputed_knn=(knn_idx, knn_dist), random_state=seed,
                        n_epochs=n_epochs, **umap_kwargs)

    import warnings
    with warnings.catch_warnings():
        # expected with precomputed knn / fixed seed; not actionable here
        warnings.filterwarnings('ignore', message='.*knn_search_index.*')
        warnings.filterwarnings('ignore', message='.*n_jobs value.*overridden.*')
        return reducer.fit_transform(np.zeros((n, 1), dtype=np.float32))


def dpmsne(P=None, emb_dim: int = 2, alpha: float = 1.0, Y: Optional[np.ndarray] = None,
           max_its: int = 1000, print_iter: int = 100, eta: Optional[float] = None,
           momentum: float = 0.8, min_gain: float = 0.01, min_value: float = 1.0e-128,
           optimize_alpha: bool = False, min_alpha: float = 1.0e-6, max_alpha_its: int = 3,
           seed: Optional[int] = None, early_exaggeration: float = 12.0, tol: float = 1.0e-7,
           out=None, **_compat_kwargs):
    """Embed a precomputed (symmetric, non-negative) affinity matrix P with exact t-SNE."""
    return tsne_exact(np.asarray(P, dtype=np.float64), emb_dim=emb_dim, alpha=alpha, Y=Y,
                      max_its=max_its, eta=eta, momentum=momentum,
                      early_exaggeration=early_exaggeration, min_gain=min_gain,
                      min_value=min_value, optimize_alpha=optimize_alpha, min_alpha=min_alpha,
                      max_alpha_its=max_alpha_its, tol=tol, print_iter=print_iter,
                      seed=seed, out=out)
