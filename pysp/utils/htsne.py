"""Model-based (hierarchical) t-SNE and UMAP for heterogeneous data.

Pairwise affinities are derived from a fitted mixture model rather than from
Euclidean distances, so anything pysparkplug can model (tuples, sequences,
sets, variable-length data, ...) can be embedded. Four affinity definitions
are supported (the `affinity` argument):

- 'balanced' (the 'auto' default whenever raw data is available): the model
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
- 'barnes_hut': scalable O(n log n) t-SNE (scikit-learn) run on a sparse
  k-nearest-neighbor matrix of model distances -log s_ij. The dense affinity
  matrix is never materialized; neighbor search is done blockwise.

humap embeds the same model-based kNN graph with UMAP (umap-learn).
"""
import sys
from typing import Optional, Tuple

import numpy as np
import scipy.sparse

__all__ = ['htsne', 'humap', 'dpmsne', 'model_log_affinity', 'sparse_model_distances',
           'model_knn', 'get_pmat', 'balanced_factors']


def _affinity_factors(posterior_mat, ll_mat, affinity):
    """Factor the log-affinity as log S = sum_f log(G_f H_f^T), returned as a
    list of (G_f, H_f) pairs (up to immaterial per-row scale).

    The single-factor modes return one pair; 'balanced' (built by
    balanced_factors) supplies one pair per composite field.
    """
    if isinstance(affinity, list):
        return affinity                      # pre-built factor list

    z = np.asarray(posterior_mat, dtype=np.float64)

    if affinity == 'coassign':
        return [(z, z)]
    if affinity == 'bhattacharyya':
        zs = np.sqrt(z)
        return [(zs, zs)]
    if affinity == 'likelihood':
        if ll_mat is None:
            raise ValueError("affinity='likelihood' requires the component log-likelihood matrix.")
        l = np.asarray(ll_mat, dtype=np.float64)
        return [(np.exp(l - l.max(axis=1, keepdims=True)), z)]

    raise ValueError("affinity must be 'coassign', 'bhattacharyya', 'likelihood', "
                     "'balanced', or a pre-built factor list.")


def _field_log_densities(dists, items):
    """Yield per-field (n, K) component log-density matrices for one model
    subtree, flattening structure into its leaf fields:

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
            yield from _field_log_densities([d.dists[f] for d in dists],
                                            [x[f] for x in items])
        return

    if 'Sequence' in tname and hasattr(dists[0], 'dist') and hasattr(dists[0], 'len_dist'):
        lens = [len(x) for x in items]
        elems = [e for x in items for e in x]
        if elems:
            seg = np.repeat(np.arange(n), lens)
            for l_e in _field_log_densities([d.dist for d in dists], elems):
                l_f = np.zeros((n, K))
                np.add.at(l_f, seg, l_e)
                yield l_f
        len_dists = [d.len_dist for d in dists]
        if len_dists[0] is not None:
            yield from _field_log_densities(len_dists, lens)
        return

    if 'Optional' in tname and hasattr(dists[0], 'dist'):
        mv = getattr(dists[0], 'missing_value', None)
        mv_is_nan = isinstance(mv, float) and np.isnan(mv)
        miss = np.asarray([x is None or x is mv or
                           (mv_is_nan and isinstance(x, float) and np.isnan(x))
                           for x in items])
        lp0 = np.asarray([getattr(d, 'log_p0', getattr(d, 'log_p', None)) for d in dists],
                         dtype=np.float64)        # log P(missing)
        lp1 = np.asarray([getattr(d, 'log_p1', getattr(d, 'log_pn', None)) for d in dists],
                         dtype=np.float64)        # log P(present)
        yield np.where(miss[:, None], lp0[None, :], lp1[None, :])
        if (~miss).any():
            fill = items[int(np.argmax(~miss))]
            sub = [fill if m else x for x, m in zip(items, miss)]
            keep = (~miss).astype(np.float64)[:, None]
            for l_in in _field_log_densities([d.dist for d in dists], sub):
                yield l_in * keep
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
    yield l


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
    likelihood scale. Combined with an evidence cap (see model_log_affinity)
    no single field can veto a pair's similarity either.
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
        z_f = l_f + log_w
        z_f -= z_f.max(axis=1, keepdims=True)
        np.exp(z_f, out=z_f)
        z_f /= z_f.sum(axis=1, keepdims=True)

        sq = np.sqrt(z_f) ** w_f
        factors.append((sq, sq))

    return factors


def _resolve_affinity(affinity, mix_model, data, field_weights):
    """Resolve 'auto'/'balanced' to a concrete affinity for the given model.

    'auto' uses per-field balancing whenever raw data is available to split
    into the model's leaf fields, since a single sharp field otherwise
    dominates the joint posterior; if the data cannot be decomposed (or no
    raw data was given) it falls back to 'bhattacharyya' on the joint
    posterior.
    """
    if affinity == 'auto':
        if getattr(mix_model, 'components', None) is not None and data is not None:
            try:
                return balanced_factors(mix_model, data, field_weights=field_weights)
            except Exception:
                return 'bhattacharyya'
        return 'bhattacharyya'

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
    n = factors[0][0].shape[0]
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    log_s = np.zeros((n, n))
    with np.errstate(divide='ignore'):
        for g, h in factors:
            term = np.log(np.dot(g, h.T))
            if cap is not None:
                np.maximum(term, -cap, out=term)
            log_s += term
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
    n = log_aff.shape[0]
    finite = np.isfinite(log_aff)

    if perplexity is None:
        p = log_aff - np.where(finite, log_aff, -np.inf).max(axis=1, keepdims=True)
        np.exp(p, out=p)
        p[~finite] = 0.0
        p /= p.sum(axis=1, keepdims=True)
        return p

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
    """Sparse n x n matrix of model distances d_ij = max_j' log s_ij' - log s_ij.

    Keeps the k nearest neighbors (largest affinity) per row. Built blockwise so
    the dense n x n affinity matrix is never materialized. Distances are
    non-negative; any per-row shift is immaterial because t-SNE's perplexity
    calibration is per row. evidence_cap as in model_log_affinity.
    """
    factors = _affinity_factors(posterior_mat, ll_mat, affinity)
    n = factors[0][0].shape[0]
    k = min(k, n - 1)
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=np.int64)
    vals = np.empty(n * k, dtype=np.float64)

    hts = [h.T for _, h in factors]
    for s0 in range(0, n, block_size):
        s1 = min(s0 + block_size, n)
        log_s = np.zeros((s1 - s0, n))
        with np.errstate(divide='ignore'):
            for (g, _), ht in zip(factors, hts):
                term = np.log(np.maximum(np.dot(g[s0:s1], ht), 1.0e-300))
                if cap is not None:
                    np.maximum(term, -cap, out=term)
                log_s += term
        log_s[np.arange(s1 - s0), np.arange(s0, s1)] = -np.inf
        s_blk = log_s

        nbr = np.argpartition(-s_blk, k - 1, axis=1)[:, :k]
        log_s = s_blk

        for bi, i in enumerate(range(s0, s1)):
            c = nbr[bi]
            d = log_s[bi, c].max() - log_s[bi, c]
            cols[i * k:(i + 1) * k] = c
            vals[i * k:(i + 1) * k] = d

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
    n = factors[0][0].shape[0]
    k = min(k, n)
    m = k - 1  # non-self neighbors
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    knn_idx = np.empty((n, k), dtype=np.int64)
    knn_dist = np.empty((n, k), dtype=np.float64)
    knn_idx[:, 0] = np.arange(n)
    knn_dist[:, 0] = 0.0

    hts = [h.T for _, h in factors]
    for s0 in range(0, n, block_size):
        s1 = min(s0 + block_size, n)
        log_s = np.zeros((s1 - s0, n))
        with np.errstate(divide='ignore'):
            for (g, _), ht in zip(factors, hts):
                term = np.log(np.maximum(np.dot(g[s0:s1], ht), 1.0e-300))
                if cap is not None:
                    np.maximum(term, -cap, out=term)
                log_s += term
        log_s[np.arange(s1 - s0), np.arange(s0, s1)] = -np.inf
        s_blk = log_s

        nbr = np.argpartition(-s_blk, m - 1, axis=1)[:, :m]

        for bi, i in enumerate(range(s0, s1)):
            c = nbr[bi]
            d = log_s[bi, c].max() - log_s[bi, c]
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


def _tsne_barnes_hut(dist_csr: scipy.sparse.csr_matrix, emb_dim: int, perplexity: float,
                     max_its: int, eta, early_exaggeration: float,
                     seed: Optional[int], Y: Optional[np.ndarray]) -> np.ndarray:
    from sklearn.manifold import TSNE
    from sklearn.neighbors import sort_graph_by_row_values

    dist_csr = sort_graph_by_row_values(dist_csr, warn_when_not_sorted=False)
    init = Y if Y is not None else 'random'
    learning_rate = 'auto' if eta is None else float(eta)

    ts = TSNE(n_components=emb_dim, perplexity=perplexity, metric='precomputed',
              method='barnes_hut', init=init, learning_rate=learning_rate,
              early_exaggeration=early_exaggeration, max_iter=max(max_its, 250),
              random_state=seed)
    return ts.fit_transform(dist_csr)


def htsne(data, emb_dim: int = 2, alpha: float = 1.0, max_components: int = 50,
          Y: Optional[np.ndarray] = None, perplexity: Optional[float] = 30.0,
          max_its: int = 1000, print_iter: int = 100, eta: Optional[float] = None,
          momentum: float = 0.8, min_gain: float = 0.01, min_value: float = 1.0e-128,
          optimize_alpha: bool = False, min_alpha: float = 1.0e-6, max_alpha_its: int = 3,
          seed: Optional[int] = None, mix_model=None, enc_data=None, method: str = 'auto',
          early_exaggeration: float = 12.0, tol: float = 1.0e-7, dpm_max_its: int = 200,
          affinity='auto', field_weights=None, evidence_cap: Optional[float] = 1.0,
          out=None, variable_length: bool = False):
    """Embed heterogeneous data with model-based t-SNE.

    A mixture model is fit to the data (a Dirichlet process mixture with
    automatically typed components by default, or pass mix_model), pairwise
    affinities are computed from the model, and the affinities are embedded
    with t-SNE.

    method:
        'exact'      - full-matrix gradient descent (supports optimize_alpha)
        'barnes_hut' - sparse kNN distances + scikit-learn Barnes-Hut t-SNE
        'auto'       - barnes_hut for n > 10 unless optimize_alpha is set

    affinity:
        'auto' (default) - 'balanced' whenever raw data is available and the
            model decomposes into leaf fields, else 'bhattacharyya'
        'balanced'   - per-field posteriors (the model's flattened leaves:
            nested composites, sequence element/length models, and optional
            wrappers all decompose) combined by per-field Bhattacharyya, so a
            sharp discrete field cannot drown an overlapping continuous one
            (or vice versa); optional field_weights sets per-field exponents
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
        # sklearn requires int(3*perplexity + 1) + 1 neighbors per row of the
        # precomputed graph (self excluded), so cap perplexity accordingly
        px = 30.0 if perplexity is None else float(perplexity)
        px = min(px, (n - 4) / 3.0)
        k = min(n - 1, int(3.0 * px) + 5)
        dist_csr = sparse_model_distances(z_ij, l_ij, k=k, affinity=affinity,
                                          evidence_cap=evidence_cap)
        return _tsne_barnes_hut(dist_csr, emb_dim, px, max_its, eta,
                                early_exaggeration, seed, Y)

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
    import umap

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
