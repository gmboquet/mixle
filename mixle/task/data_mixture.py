"""DoReMi-style data-mixture optimization: domain weights as a bandit/DOE problem (roadmap F8).

A pretraining corpus is normally split into named domains (web text, code, books, ...) mixed by a
hand-picked weight vector. DoReMi (Xie et al.) instead treats the weight vector itself as something to
OPTIMIZE: run many small, cheap proxy trainings at different mixtures, score each on held-out loss, and
search the simplex for the mixture that generalizes best -- then apply that learned mixture to the real,
much larger run. This module is the small, honest version of that loop, reusing mixle's own optimization
machinery rather than inventing new search code:

  * :class:`SyntheticDomain` -- a synthetic "domain": a name plus a stand-in data-generating distribution
    (a fixed periodic token pattern with configurable noise, or pure noise for an unlearnable domain).
  * :func:`proxy_run_score` -- one proxy run: build a token stream from a mixture of domains, train a real
    (tiny) :class:`mixle.models.language_model.LM` for a handful of steps, and return the mean held-out
    NLL across domains (lower is better). This is the "small-run proxy" DoReMi scores mixtures with.
  * :func:`optimize_mixture` -- the DoReMi search loop: repeated proxy runs scored via
    :func:`proxy_run_score`, with candidate mixtures proposed by ``mixle.task.bandit``'s
    :class:`~mixle.task.bandit.ThompsonGaussian` (discrete arms on a simplex-lattice design from
    ``mixle.doe.mixture``) or ``mixle.doe.optimizer``'s :class:`~mixle.doe.optimizer.BayesianOptimizer`
    (continuous search over a softmax-reparameterized simplex). No new optimizer machinery -- both paths
    are the same modules F5/I1/D5 already reuse this session.
  * :func:`estimate_near_duplicate_rate` -- a minimal, honest corpus dedup/quality receipt: a MinHash
    estimate of the fraction of documents with a near-duplicate elsewhere in the corpus.

F5 (scaling-law fits) integration point: F5's fitted scaling laws could extrapolate a proxy run's
held-out loss at *this* scale to a prediction at the real target scale, letting the search compare
mixtures by extrapolated real-scale loss instead of raw proxy-scale loss. F5's branch was not reachable
from this worktree at the time F8 was built, so that extrapolation is not wired in here -- the natural
integration point is inside :func:`proxy_run_score`, mapping its returned proxy-scale loss through a
fitted ``mixle.task.<f5-module>`` law before it reaches the optimizer. The search loop itself
(:func:`optimize_mixture`) does not need to change: it only requires a scalar score per mixture, however
that score is produced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "SyntheticDomain",
    "estimate_near_duplicate_rate",
    "optimize_mixture",
    "proxy_run_score",
]


@dataclass(frozen=True)
class SyntheticDomain:
    """One synthetic "domain": a fixed periodic token pattern, optionally corrupted by noise.

    ``pattern_seed`` fixes a length-``period`` sequence of token ids (drawn once, from ``0..vocab)``)
    that repeats forever -- the domain's learnable structure. Each sampled token then has independent
    probability ``noise_p`` of being replaced by a uniform-random token, so ``noise_p=0`` is a perfectly
    learnable domain and ``noise_p=1`` (or ``period=None``) is pure, irreducible noise: no amount of
    training data lowers a model's achievable loss on it below ``log(vocab)``. Distinct
    ``(period, pattern_seed, noise_p)`` triples give genuinely different data-generating distributions,
    standing in for e.g. "web text" vs "code" vs "books" without needing real corpora.
    """

    name: str
    vocab: int
    period: int | None = 8
    noise_p: float = 0.0
    pattern_seed: int = 0
    _pattern: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.vocab < 2:
            raise ValueError("vocab must be >= 2.")
        if not 0.0 <= self.noise_p <= 1.0:
            raise ValueError("noise_p must lie in [0, 1].")
        if self.period is not None and self.period < 1:
            raise ValueError("period must be positive (or None for pure noise).")
        if self.period is None:
            pattern = np.zeros(0, dtype=np.int64)
        else:
            pattern = np.random.RandomState(self.pattern_seed).randint(0, self.vocab, size=int(self.period))
        object.__setattr__(self, "_pattern", pattern)

    def sample(self, n_tokens: int, *, seed: int = 0) -> np.ndarray:
        """Draw ``n_tokens`` ids (int64 array) from this domain's distribution."""
        n_tokens = int(n_tokens)
        if n_tokens <= 0:
            return np.zeros(0, dtype=np.int64)
        rng = np.random.RandomState(seed)
        if self.period is None:
            return rng.randint(0, self.vocab, size=n_tokens).astype(np.int64)
        idx = np.arange(n_tokens) % int(self.period)
        ids = self._pattern[idx].copy()
        if self.noise_p > 0.0:
            corrupt = rng.random_sample(n_tokens) < self.noise_p
            if np.any(corrupt):
                ids[corrupt] = rng.randint(0, self.vocab, size=int(corrupt.sum()))
        return ids.astype(np.int64)


def _normalize_weights(mixture_weights: Sequence[float], n_domains: int) -> np.ndarray:
    w = np.asarray(list(mixture_weights), dtype=np.float64)
    if w.shape != (n_domains,):
        raise ValueError(f"mixture_weights must have {n_domains} entries, got shape {w.shape}.")
    if np.any(w < 0.0):
        raise ValueError("mixture_weights must be non-negative.")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("mixture_weights must sum to a positive total.")
    return w / total


def proxy_run_score(
    mixture_weights: Sequence[float],
    domains: Sequence[SyntheticDomain],
    proxy_steps: int,
    *,
    batch_size: int = 16,
    d_model: int = 16,
    n_layer: int = 1,
    n_head: int = 2,
    block: int = 8,
    lr: float = 3.0e-3,
    eval_tokens: int = 512,
    seed: int = 0,
    eval_seed: int = 999_000,
    return_detail: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Run one short proxy training and return the mean held-out NLL across ``domains`` (lower is better).

    Builds a training token stream by drawing ``mixture_weights[i]``-proportional tokens from each domain
    (concatenated; the number of tokens is chosen so training runs roughly ``proxy_steps`` gradient
    steps at ``batch_size``), trains a real (tiny) :class:`mixle.models.language_model.LM` on it for one
    epoch, then scores held-out NLL on ``eval_tokens`` fresh tokens from EACH domain (independent of the
    mixture) and returns the unweighted mean across domains -- the DoReMi objective is generalizing to
    every domain, not just the ones the mixture over-samples. ``return_detail=True`` also returns the
    per-domain NLL dict, keyed by domain name.

    ``seed`` controls the training-data draw (and so varies across repeated proxy runs, e.g. inside
    :func:`optimize_mixture`'s search loop); ``eval_seed`` controls the held-out draw and is fixed by
    default so different mixtures proposed during a search are scored against the SAME held-out set --
    comparing candidate mixtures on a moving eval target would swamp the (often small) between-mixture
    signal in eval-sampling noise.
    """
    import torch

    from mixle.models.language_model import LM

    domains = list(domains)
    if len(domains) < 2:
        raise ValueError("need at least two domains to mix.")
    vocabs = {d.vocab for d in domains}
    if len(vocabs) != 1:
        raise ValueError("all domains must share the same vocab.")
    vocab = domains[0].vocab
    proxy_steps = max(1, int(proxy_steps))
    weights = _normalize_weights(mixture_weights, len(domains))
    torch.manual_seed(int(seed))  # deterministic model init + fit-loop RNG (dropout/etc.) given `seed`

    total_train_tokens = proxy_steps * batch_size + block + 1
    chunks = []
    for i, (domain, w) in enumerate(zip(domains, weights)):
        n_i = int(round(w * total_train_tokens))
        n_i = max(n_i, 0)
        if n_i:
            chunks.append(domain.sample(n_i, seed=seed * 10_000 + i))
    if not chunks:
        raise ValueError("mixture assigns zero tokens to every domain.")
    train_tokens = np.concatenate(chunks)
    if len(train_tokens) <= block:
        # a degenerate (near-empty) mixture request; pad with more of whatever domain had weight
        train_tokens = np.tile(train_tokens, block // max(len(train_tokens), 1) + 2)

    lm = LM(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block, device="cpu")
    lm.fit(train_tokens, epochs=1, batch_size=batch_size, lr=lr, shuffle=True)

    detail: dict[str, float] = {}
    for i, domain in enumerate(domains):
        held_out = domain.sample(eval_tokens + block + 1, seed=eval_seed + i)
        detail[domain.name] = lm.nll(held_out)
    aggregate = float(np.mean(list(detail.values())))
    if return_detail:
        return aggregate, detail
    return aggregate


def _simplex_arms(n_domains: int, budget: int) -> np.ndarray:
    """Candidate mixture-weight vectors on the ``(n_domains - 1)``-simplex, capped to ``budget`` arms."""
    from mixle.doe.mixture import simplex_lattice

    arms = simplex_lattice(n_domains, m=2)
    if len(arms) > budget:
        idx = np.unique(np.round(np.linspace(0, len(arms) - 1, budget)).astype(int))
        arms = arms[idx]
    return arms


def _bandit_search(
    domains: Sequence[SyntheticDomain], proxy_steps: int, budget: int, proxy_kwargs: dict[str, Any], seed: int
) -> np.ndarray:
    from mixle.task.bandit import ThompsonGaussian

    arms = _simplex_arms(len(domains), budget)
    n_arms = len(arms)
    if n_arms < 2:
        raise ValueError("budget too small to form at least two mixture-weight arms.")
    bandit = ThompsonGaussian(n_arms, seed=seed)
    for t in range(int(budget)):
        arm = bandit.select()
        loss = proxy_run_score(arms[arm], domains, proxy_steps, seed=seed * 1000 + t, **proxy_kwargs)
        bandit.update(arm, reward=-loss)  # higher reward = lower held-out loss
    best_arm = int(np.argmax(bandit.means))
    return arms[best_arm]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def _doe_search(
    domains: Sequence[SyntheticDomain], proxy_steps: int, budget: int, proxy_kwargs: dict[str, Any], seed: int
) -> np.ndarray:
    from mixle.doe.optimizer import BayesianOptimizer

    n = len(domains)
    bounds = [(-3.0, 3.0)] * n
    n_init = min(max(2 * n + 1, 2), max(budget - 1, 2))
    opt = BayesianOptimizer(bounds, acq="ei", maximize=False, n_init=n_init, seed=seed)
    for t in range(int(budget)):
        x = opt.ask()
        w = _softmax(np.asarray(x, dtype=np.float64))
        loss = proxy_run_score(w, domains, proxy_steps, seed=seed * 1000 + t, **proxy_kwargs)
        opt.tell(x, loss)
    return _softmax(np.asarray(opt.best.best_x, dtype=np.float64))


def optimize_mixture(
    domains: Sequence[SyntheticDomain],
    proxy_steps: int,
    budget: int,
    *,
    method: str = "bandit",
    proxy_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Learn domain mixture weights via repeated short proxy runs (DoReMi-style search).

    ``budget`` proxy runs (each :func:`proxy_run_score` at ``proxy_steps`` gradient steps) are used to
    search the mixture-weight simplex. ``method="bandit"`` (default) discretizes the simplex into a
    lattice of candidate mixtures (``mixle.doe.mixture.simplex_lattice``) and searches them with
    ``mixle.task.bandit.ThompsonGaussian`` (reward = negative held-out loss); ``method="doe"`` searches
    continuously via ``mixle.doe.optimizer.BayesianOptimizer`` over a softmax-reparameterized simplex.
    Returns the learned weight vector (one entry per domain, summing to 1).
    """
    domains = list(domains)
    if len(domains) < 2:
        raise ValueError("need at least two domains to mix.")
    if budget < 2:
        raise ValueError("budget must allow at least two proxy runs.")
    kwargs = dict(proxy_kwargs or {})
    if kwargs.get("return_detail"):
        # the search loop needs a bare scalar loss (bandit.update(reward=-loss), opt.tell(x, loss));
        # return_detail=True makes proxy_run_score return a (loss, per_domain_dict) tuple instead,
        # which crashes the loop far from this call site with an opaque TypeError.
        raise ValueError("proxy_kwargs['return_detail']=True is incompatible with optimize_mixture's search loop.")
    if method == "bandit":
        return _bandit_search(domains, proxy_steps, budget, kwargs, seed)
    if method == "doe":
        return _doe_search(domains, proxy_steps, budget, kwargs, seed)
    raise ValueError(f"unknown method {method!r}; expected 'bandit' or 'doe'.")


# --- corpus dedup / quality receipt --------------------------------------------------------------------


def _stable_token_hash(tokens: Sequence[str]) -> int:
    payload = "\x1f".join(tokens).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _shingles(text: str, k: int) -> frozenset[int]:
    toks = text.lower().split()
    if len(toks) < k:
        return frozenset({_stable_token_hash(toks)})
    return frozenset(_stable_token_hash(toks[i : i + k]) for i in range(len(toks) - k + 1))


def _minhash_signature(shingle_set: frozenset[int], a: np.ndarray, b: np.ndarray, prime: int) -> np.ndarray:
    if not shingle_set:
        return np.zeros(len(a), dtype=np.int64)
    hashes = np.array([s % prime for s in shingle_set], dtype=np.int64)
    sig = (np.outer(a, hashes) + b[:, None]) % prime  # (num_hashes, n_shingles)
    return sig.min(axis=1)


def estimate_near_duplicate_rate(
    corpus: Sequence[str],
    *,
    shingle_size: int = 5,
    num_hashes: int = 64,
    threshold: float = 0.8,
    seed: int = 0,
) -> float:
    """Estimate the fraction of documents in ``corpus`` that have a near-duplicate elsewhere in it.

    A minimal, honest MinHash quality/dedup receipt: each document is reduced to its set of
    word-``shingle_size`` shingles, each shingle set to a ``num_hashes``-entry MinHash signature (an
    unbiased estimator of Jaccard similarity), and two documents are called near-duplicates when their
    signatures agree on at least ``threshold`` of their entries. Returns
    ``|{documents with >= 1 near-duplicate partner}| / |corpus|``. ``O(n^2)`` in the corpus size --
    fine for the receipt-sized corpora this is meant for, not a production LSH dedup pipeline.
    """
    docs = list(corpus)
    if isinstance(shingle_size, bool) or not isinstance(shingle_size, int) or shingle_size <= 0:
        raise ValueError("shingle_size must be a positive integer")
    if isinstance(num_hashes, bool) or not isinstance(num_hashes, int) or num_hashes <= 0:
        raise ValueError("num_hashes must be a positive integer")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    n = len(docs)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    prime = (1 << 61) - 1
    rng = np.random.RandomState(seed)
    a = rng.randint(1, prime - 1, size=int(num_hashes))
    b = rng.randint(0, prime - 1, size=int(num_hashes))
    sigs = [_minhash_signature(_shingles(doc, shingle_size), a, b, prime) for doc in docs]
    has_dup = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.mean(sigs[i] == sigs[j]))
            if sim >= threshold:
                has_dup[i] = True
                has_dup[j] = True
    return float(has_dup.sum()) / n
