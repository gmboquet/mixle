"""Shared benchmark infrastructure: fair, reproducible, correctness-checked.

Fairness contract (enforced, not assumed):
  * every package fits the SAME data array,
  * from the SAME initialization (means / covariances / weights / transitions fed in),
  * for the SAME fixed number of EM iterations (no early stop),
  * and the final mean log-likelihood is asserted equal across packages to a tight
    tolerance -- a timing number is only reported if the fit actually landed at the
    same optimum. A faster-but-different fit is not a win, it is a different computation.

Timing is the median of ``reps`` wall-clock runs after one discarded warm-up (numba /
torch JIT, allocator warm-up). Only the ``.fit`` call is timed, never data prep or init.
"""

import os

# Pin the thread budget BEFORE numpy/torch import so every package gets the same compute.
_THREADS = "4"
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, _THREADS)

import time  # noqa: E402
from math import lgamma, log  # noqa: E402

import numpy as np  # noqa: E402


def pin_torch():
    try:
        import torch
    except ImportError:  # torch only matters for the pomegranate arm; a clean base clone has neither
        return
    torch.set_num_threads(int(_THREADS))
    torch.set_default_dtype(torch.float64)  # match numpy float64 for LL parity


def timed(make_and_fit, reps=5):
    """Median wall-clock of ``make_and_fit()`` over ``reps`` runs after one warm-up.

    ``make_and_fit`` must build a fresh (unfitted) model from the shared init and fit
    it, returning ``(mean_log_likelihood, iters)``. Rebuilding each rep keeps every run
    identical (fitting mutates the model in place). A package that raises (e.g. a
    non-PD covariance it cannot recover from) is recorded as a failure, not fabricated
    and not allowed to abort the sweep.
    """
    try:
        ll, iters = make_and_fit()  # warm-up (discarded)
    except Exception as e:  # noqa: BLE001 - a crash is an honest, reportable outcome
        return {"sec": None, "mean_ll": None, "iters": None, "failed": type(e).__name__}
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        make_and_fit()
        ts.append(time.perf_counter() - t0)
    return {"sec": float(np.median(ts)), "sec_min": float(np.min(ts)), "mean_ll": float(ll), "iters": int(iters)}


# --------------------------------------------------------------------------------------
# Workload generators (real problems, shared across every package)
# --------------------------------------------------------------------------------------


def _lloyd_kmeans(X, k, *, max_iter, seed):
    """Seeded plain Lloyd's iteration -- the sklearn-free stand-in for the shared k-means init."""
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].astype(np.float64)
    for _ in range(max_iter):
        d = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
        assign = d.argmin(1)
        for j in range(k):
            members = X[assign == j]
            if len(members):
                centers[j] = members.mean(0)
    return centers


def make_full_cov_gmm(n, dim, k, seed=42):
    """A full-covariance Gaussian mixture with genuinely correlated components.

    This is the density-estimation / soft-clustering workload: correlated multivariate
    features (think MFCC frames, tabular embeddings), separated but overlapping clusters.
    Returns ``(X, init)`` where init = (means0, cov0, weights0) shared by every package.
    """
    rng = np.random.RandomState(seed)
    true_means = rng.randn(k, dim) * 3.0
    X = np.empty((n, dim), dtype=np.float64)
    z = rng.randint(0, k, size=n)
    for j in range(k):
        a = rng.randn(dim, dim) * 0.4
        cov = a @ a.T + np.eye(dim)  # PD, correlated
        m = z == j
        X[m] = rng.multivariate_normal(true_means[j], cov, size=int(m.sum()))
    # shared init: k-means centres (the standard, non-degenerate GMM starting point),
    # global covariance, uniform weights -- computed once, fed identically to every
    # package, and NOT counted in any fit time. sklearn's KMeans when available (the
    # configuration every published results.json used); otherwise a seeded numpy Lloyd's
    # with the same role -- still deterministic and still SHARED, so comparisons stay
    # fair, and with sklearn absent there is no sklearn arm to compare against anyway.
    try:
        from sklearn.cluster import KMeans

        means0 = KMeans(n_clusters=k, n_init=1, max_iter=25, random_state=0).fit(X).cluster_centers_.astype(np.float64)
    except ImportError:
        means0 = _lloyd_kmeans(X, k, max_iter=25, seed=0)
    cov0 = np.cov(X.T) + 1e-6 * np.eye(dim)
    weights0 = np.full(k, 1.0 / k)
    return X, (means0, cov0, weights0)


def make_gaussian_hmm(n_seq, length, states, seed=7):
    """A Gaussian-emission HMM: many sequences from a latent regime process.

    The sequence-modeling / regime-detection workload (financial states, sensor modes,
    genomics). Returns ``(seqs, Xcat, lengths, init)`` with init shared across packages.
    """
    rng = np.random.RandomState(seed)
    means = (rng.randn(states) * 5).reshape(states, 1)
    trans = rng.dirichlet(np.ones(states) * 2, size=states)
    start = rng.dirichlet(np.ones(states))
    seqs = []
    for _ in range(n_seq):
        s = rng.choice(states, p=start)
        obs = np.empty(length)
        for t in range(length):
            obs[t] = rng.randn() + means[s, 0]
            s = rng.choice(states, p=trans[s])
        seqs.append(obs.reshape(length, 1))
    xcat = np.vstack(seqs)
    lengths = [length] * n_seq
    means0 = (rng.randn(states) * 2).reshape(states, 1)
    init = (start.copy(), trans.copy(), means0, np.ones((states, 1)))
    return seqs, xcat, lengths, init


# --------------------------------------------------------------------------------------
# GMM adapters -- each returns (mean_ll, iters); each rebuilds from the shared init
# --------------------------------------------------------------------------------------


def gmm_sklearn(X, init, max_its):
    means0, cov0, w0 = init
    k, dim = means0.shape
    prec0 = np.stack([np.linalg.inv(cov0)] * k)

    def run():
        # imported here, inside the timed thunk's failure contract: a missing package is
        # recorded by `timed` as an honest per-package failure instead of aborting the sweep
        # (the first call is the discarded warm-up, so import cost never lands in a timing)
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            max_iter=max_its,
            tol=1e-12,
            reg_covar=1e-6,
            means_init=means0,
            weights_init=w0,
            precisions_init=prec0,
            random_state=0,
        )
        gm.fit(X)
        return gm.score(X), gm.n_iter_

    return run


def gmm_pomegranate(X, init, max_its):
    means0, cov0, w0 = init
    k = means0.shape[0]

    def run():
        # inside the timed thunk's failure contract (see gmm_sklearn); the tensor conversion
        # repeats per rep, but it is microseconds against multi-second fits
        import torch
        from pomegranate.distributions import Normal
        from pomegranate.gmm import GeneralMixtureModel

        Xt = torch.tensor(X, dtype=torch.float64)
        dists = [
            Normal(means=torch.tensor(means0[j]), covs=torch.tensor(cov0), covariance_type="full") for j in range(k)
        ]
        model = GeneralMixtureModel(dists, priors=torch.tensor(w0), max_iter=max_its, tol=1e-12, verbose=False)
        model.fit(Xt)
        n = Xt.shape[0]
        return float(model.log_probability(Xt).sum()) / n, max_its

    return run


def gmm_mixle(X, init, max_its):
    import mixle.stats as st
    from mixle.inference import optimize

    means0, cov0, w0 = init
    k, dim = means0.shape
    data = list(X)

    def run():
        comps = [st.MultivariateGaussianDistribution(means0[j].copy(), cov0.copy()) for j in range(k)]
        m0 = st.MixtureDistribution(comps, list(w0))
        est = st.MixtureEstimator([st.MultivariateGaussianEstimator(dim=dim) for _ in range(k)])
        m = optimize(data, est, prev_estimate=m0, max_its=max_its, delta=None, out=None)
        enc = m.dist_to_encoder().seq_encode(data)
        return float(np.sum(np.asarray(m.seq_log_density(enc)))) / len(data), max_its

    return run


# --------------------------------------------------------------------------------------
# HMM adapters -- mean log-likelihood PER SEQUENCE, HMM-only (no length term)
# --------------------------------------------------------------------------------------


def hmm_hmmlearn(seqs, xcat, lengths, init, max_its):
    start0, trans0, means0, covar0 = init
    states = start0.shape[0]
    n_seq = len(seqs)

    def run():
        # inside the timed thunk's failure contract (see gmm_sklearn)
        from hmmlearn.hmm import GaussianHMM

        hm = GaussianHMM(
            n_components=states, covariance_type="diag", n_iter=max_its, tol=-1e9, init_params="", params="stmc"
        )
        hm.startprob_, hm.transmat_, hm.means_, hm.covars_ = start0.copy(), trans0.copy(), means0.copy(), covar0.copy()
        hm.fit(xcat, lengths)
        return hm.score(xcat, lengths) / n_seq, hm.monitor_.iter

    return run


def hmm_mixle(seqs, init, length, max_its):
    import mixle.stats as st
    from mixle.inference import optimize

    start0, trans0, means0, _ = init
    states = start0.shape[0]
    data = [list(s.ravel()) for s in seqs]
    # mixle models sequence length explicitly; subtract the Poisson log-pmf at ``length``
    # so the reported LL is the emission+transition term hmmlearn also computes.
    lam = float(length)
    len_term = -lam + length * log(lam) - lgamma(length + 1)

    def run():
        comps = [st.GaussianDistribution(float(means0[j, 0]), 1.0) for j in range(states)]
        m0 = st.HiddenMarkovModelDistribution(
            comps, list(start0), trans0.tolist(), len_dist=st.PoissonDistribution(lam)
        )
        est = st.HiddenMarkovEstimator(
            [st.GaussianEstimator() for _ in range(states)], len_estimator=st.PoissonEstimator()
        )
        m = optimize(data, est, prev_estimate=m0, max_its=max_its, delta=None, out=None)
        enc = m.dist_to_encoder().seq_encode(data)
        full = float(np.sum(np.asarray(m.seq_log_density(enc)))) / len(data)
        return full - len_term, max_its

    return run
