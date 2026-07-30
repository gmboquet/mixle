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
    "MixtureOptimizationReceipt",
    "SyntheticDomain",
    "estimate_near_duplicate_rate",
    "optimize_mixture",
    "proxy_run_score",
]


def _exact_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _random_state_seed(value: Any, name: str = "seed") -> int:
    result = _exact_int(value, name, minimum=0)
    if result >= 2**32:
        raise ValueError(f"{name} must be less than 2**32.")
    return result


def _derived_seed(seed: int, purpose: str) -> int:
    payload = f"{seed}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


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
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        vocab = _exact_int(self.vocab, "vocab")
        if vocab < 2:
            raise ValueError("vocab must be >= 2.")
        noise_p = float(self.noise_p)
        if not np.isfinite(noise_p) or not 0.0 <= noise_p <= 1.0:
            raise ValueError("noise_p must lie in [0, 1].")
        pattern_seed = _random_state_seed(self.pattern_seed, "pattern_seed")
        period = None if self.period is None else _exact_int(self.period, "period", minimum=1)
        if self.period is None:
            pattern = np.zeros(0, dtype=np.int64)
        else:
            pattern = np.random.RandomState(pattern_seed).randint(0, vocab, size=period)
        object.__setattr__(self, "_pattern", pattern)

    def sample(self, n_tokens: int, *, seed: int = 0) -> np.ndarray:
        """Draw ``n_tokens`` ids (int64 array) from this domain's distribution."""
        n_tokens = _exact_int(n_tokens, "n_tokens", minimum=0)
        if n_tokens == 0:
            return np.zeros(0, dtype=np.int64)
        rng = np.random.RandomState(_random_state_seed(seed))
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
    try:
        w = np.asarray(list(mixture_weights), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("mixture_weights must contain real numbers.") from exc
    if w.shape != (n_domains,):
        raise ValueError(f"mixture_weights must have {n_domains} entries, got shape {w.shape}.")
    if not np.all(np.isfinite(w)):
        raise ValueError("mixture_weights must be finite.")
    if np.any(w < 0.0):
        raise ValueError("mixture_weights must be non-negative.")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("mixture_weights must sum to a positive total.")
    return w / total


def _validate_domains(domains: Sequence[SyntheticDomain]) -> list[SyntheticDomain]:
    result = list(domains)
    if len(result) < 2:
        raise ValueError("need at least two domains to mix.")
    if any(not isinstance(domain, SyntheticDomain) for domain in result):
        raise TypeError("domains must contain only SyntheticDomain instances.")
    names = [domain.name for domain in result]
    if len(set(names)) != len(names):
        raise ValueError("domain names must be unique.")
    if len({domain.vocab for domain in result}) != 1:
        raise ValueError("all domains must share the same vocab.")
    return result


def _sample_interleaved_stream(
    domains: Sequence[SyntheticDomain],
    weights: np.ndarray,
    n_tokens: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample domain identity per token and return the stream plus those identities.

    Tokens belonging to one domain retain that domain's sequential generator order,
    but are scattered into positions drawn independently from the mixture.
    """
    n_tokens = _exact_int(n_tokens, "n_tokens", minimum=1)
    seed = _random_state_seed(seed)
    assignment = np.random.RandomState(_derived_seed(seed, "domain-assignment")).choice(
        len(domains), size=n_tokens, p=weights
    )
    stream = np.empty(n_tokens, dtype=np.int64)
    for index, domain in enumerate(domains):
        positions = np.flatnonzero(assignment == index)
        if positions.size:
            stream[positions] = domain.sample(
                int(positions.size),
                seed=_derived_seed(seed, f"domain-{index}"),
            )
    return stream, assignment


@dataclass(frozen=True)
class MixtureOptimizationReceipt:
    """Independent evaluation evidence for a selected mixture."""

    method: str
    weights: tuple[float, ...]
    search_runs: int
    selection_eval_seed: int
    audit_training_seed: int
    audit_eval_seed: int
    audit_loss: float
    audit_per_domain: tuple[tuple[str, float], ...]


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

    Builds a training token stream by independently drawing each token's domain from
    ``mixture_weights`` (the number of tokens is chosen so training runs roughly ``proxy_steps`` gradient
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

    domains = _validate_domains(domains)
    vocab = domains[0].vocab
    proxy_steps = _exact_int(proxy_steps, "proxy_steps", minimum=1)
    batch_size = _exact_int(batch_size, "batch_size", minimum=1)
    block = _exact_int(block, "block", minimum=1)
    eval_tokens = _exact_int(eval_tokens, "eval_tokens", minimum=1)
    seed = _random_state_seed(seed)
    eval_seed = _random_state_seed(eval_seed, "eval_seed")
    weights = _normalize_weights(mixture_weights, len(domains))
    torch.manual_seed(seed)  # deterministic model init + fit-loop RNG (dropout/etc.) given `seed`

    total_train_tokens = proxy_steps * batch_size + block + 1
    train_tokens, _ = _sample_interleaved_stream(domains, weights, total_train_tokens, seed=seed)

    lm = LM(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block, device="cpu")
    lm.fit(train_tokens, epochs=1, batch_size=batch_size, lr=lr, shuffle=True)

    detail: dict[str, float] = {}
    for i, domain in enumerate(domains):
        held_out = domain.sample(
            eval_tokens + block + 1,
            seed=_derived_seed(eval_seed, f"held-out-domain-{i}"),
        )
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
        loss = proxy_run_score(
            arms[arm],
            domains,
            proxy_steps,
            seed=_derived_seed(seed, f"bandit-search-{t}"),
            **proxy_kwargs,
        )
        bandit.update(arm, reward=-loss)  # higher reward = lower held-out loss
    best_arm = int(np.argmax(bandit.means))
    return arms[best_arm]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


# Finalist re-scoring controls for _doe_search's confirmation round; see the note at its return.
_DOE_FINALISTS = 3
_DOE_CONFIRM_RUNS = 2


def _logits_from_free(free_logits: Any, n: int) -> np.ndarray:
    """Expand ``n - 1`` free logits to ``n`` by pinning the last at 0 -- the identifiable softmax chart."""
    free = np.asarray(free_logits, dtype=np.float64).reshape(-1)
    if free.shape[0] != max(n - 1, 1):
        raise ValueError(f"expected {max(n - 1, 1)} free logits for {n} domains, got {free.shape[0]}")
    return free if n == 1 else np.append(free, 0.0)


def _doe_search(
    domains: Sequence[SyntheticDomain], proxy_steps: int, budget: int, proxy_kwargs: dict[str, Any], seed: int
) -> np.ndarray:
    from mixle.doe.optimizer import BayesianOptimizer

    n = len(domains)
    # Search n-1 free logits with the last pinned at 0, not all n. ``softmax(x + c * 1) == softmax(x)``,
    # so searching n logits gives every mixture an entire line of duplicate representations: the
    # objective is exactly flat along ``1``, the box's corners collapse onto each other (``[-2.6, -2.8,
    # -3.0]`` and ``[2.6, 3.0, 2.5]`` are both essentially uniform), and the surrogate is fitting a
    # non-identifiable function. Measured, the acquisition then spent every post-initialization ask on
    # box corners and never refined near the incumbent, so the search returned byte-identical weights at
    # budget 10, 16, 24 and 40 -- extra evaluations bought no information at all. Pinning one logit makes
    # the parameterization a bijection onto the simplex's own n-1 dimensions and the budget effective
    # again.
    free = max(n - 1, 1)
    bounds = [(-3.0, 3.0)] * free
    n_init = min(max(2 * free + 1, 2), max(budget - 1, 2))
    opt = BayesianOptimizer(bounds, acq="ei", maximize=False, n_init=n_init, seed=seed)
    observed: list[tuple[float, np.ndarray]] = []
    for t in range(int(budget)):
        x = opt.ask()
        w = _softmax(_logits_from_free(x, n))
        loss = proxy_run_score(
            w,
            domains,
            proxy_steps,
            seed=_derived_seed(seed, f"doe-search-{t}"),
            **proxy_kwargs,
        )
        opt.tell(x, loss)
        observed.append((float(loss), np.asarray(x, dtype=np.float64)))

    # Select by CONFIRMED score, not by the single best search observation. proxy_run_score is noisy --
    # re-running one mixture at fresh seeds moves it by several tenths of a nat -- and ``opt.best`` is
    # documented as the best OBSERVED point, so taking it makes the winner whichever candidate drew the
    # most favourable noise. Measured on the four-domain difficulty ladder: the argmin-of-observed scored
    # 2.9193 during the search and 3.4280 when the same weights were re-scored at five fresh seeds, an
    # optimism of +0.51, against uniform's 3.2245 -- i.e. the search returned a mixture genuinely WORSE
    # than uniform, and one that starved a domain to 0.002, because extreme corners are exactly where a
    # lucky draw is most likely to look best. It also explains why more budget did not help: additional
    # evaluations give more chances at a low outlier, so argmin-of-observed does not converge on the
    # optimum. Re-scoring the finalists at seeds the search never used breaks that selection bias.
    finalists = [x for _loss, x in sorted(observed, key=lambda row: row[0])[:_DOE_FINALISTS]]
    scored: list[tuple[float, np.ndarray]] = []
    for rank, candidate in enumerate(finalists):
        weights = _softmax(_logits_from_free(candidate, n))
        confirmations = [
            proxy_run_score(
                weights,
                domains,
                proxy_steps,
                seed=_derived_seed(seed, f"doe-confirm-{rank}-{repeat}"),
                **proxy_kwargs,
            )
            for repeat in range(_DOE_CONFIRM_RUNS)
        ]
        scored.append((float(np.mean(confirmations)), candidate))
    return _softmax(_logits_from_free(min(scored, key=lambda row: row[0])[1], n))


def optimize_mixture(
    domains: Sequence[SyntheticDomain],
    proxy_steps: int,
    budget: int,
    *,
    method: str = "bandit",
    proxy_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
    return_receipt: bool = False,
) -> np.ndarray | tuple[np.ndarray, MixtureOptimizationReceipt]:
    """Learn domain mixture weights via repeated short proxy runs (DoReMi-style search).

    ``budget - 1`` proxy runs (each :func:`proxy_run_score` at ``proxy_steps`` gradient steps) search
    the mixture-weight simplex; the final reserved run audits the selected mixture with independent
    training and evaluation seeds. ``method="bandit"`` (default) discretizes the simplex into a
    lattice of candidate mixtures (``mixle.doe.mixture.simplex_lattice``) and searches them with
    ``mixle.task.bandit.ThompsonGaussian`` (reward = negative held-out loss); ``method="doe"`` searches
    continuously via ``mixle.doe.optimizer.BayesianOptimizer`` over a softmax-reparameterized simplex.
    Returns the learned weight vector (one entry per domain, summing to 1). With
    ``return_receipt=True``, also returns the immutable independent-audit receipt.
    """
    domains = _validate_domains(domains)
    proxy_steps = _exact_int(proxy_steps, "proxy_steps", minimum=1)
    budget = _exact_int(budget, "budget", minimum=3)
    seed = _random_state_seed(seed)
    if not isinstance(method, str):
        raise ValueError("method must be 'bandit' or 'doe'.")
    if not isinstance(return_receipt, bool):
        raise ValueError("return_receipt must be a boolean.")
    kwargs = dict(proxy_kwargs or {})
    if "seed" in kwargs:
        raise ValueError("proxy_kwargs must not override optimize_mixture's training seed.")
    if "return_detail" in kwargs:
        # the search loop needs a bare scalar loss (bandit.update(reward=-loss), opt.tell(x, loss));
        # return_detail=True makes proxy_run_score return a (loss, per_domain_dict) tuple instead,
        # which crashes the loop far from this call site with an opaque TypeError.
        raise ValueError("proxy_kwargs must not override optimize_mixture's return_detail setting.")
    selection_eval_seed = _random_state_seed(kwargs.get("eval_seed", 999_000), "proxy_kwargs['eval_seed']")
    kwargs["eval_seed"] = selection_eval_seed
    search_runs = budget - 1
    if method == "bandit":
        weights = _bandit_search(domains, proxy_steps, search_runs, kwargs, seed)
    elif method == "doe":
        weights = _doe_search(domains, proxy_steps, search_runs, kwargs, seed)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'bandit' or 'doe'.")

    audit_training_seed = _derived_seed(seed, "independent-audit-training")
    audit_eval_seed = _derived_seed(selection_eval_seed, "independent-audit-evaluation")
    if audit_eval_seed == selection_eval_seed:
        audit_eval_seed = (audit_eval_seed + 1) % (2**32)
    audit_kwargs = dict(kwargs)
    audit_kwargs.pop("eval_seed", None)
    audit_loss, audit_detail = proxy_run_score(
        weights,
        domains,
        proxy_steps,
        seed=audit_training_seed,
        eval_seed=audit_eval_seed,
        return_detail=True,
        **audit_kwargs,
    )
    receipt = MixtureOptimizationReceipt(
        method=method,
        weights=tuple(float(weight) for weight in weights),
        search_runs=search_runs,
        selection_eval_seed=selection_eval_seed,
        audit_training_seed=audit_training_seed,
        audit_eval_seed=audit_eval_seed,
        audit_loss=audit_loss,
        audit_per_domain=tuple(audit_detail.items()),
    )
    if return_receipt:
        return weights, receipt
    return weights


# --- corpus dedup / quality receipt --------------------------------------------------------------------


def _stable_token_hash(tokens: Sequence[str]) -> int:
    payload = "\x1f".join(tokens).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _shingles(text: str, k: int) -> frozenset[int]:
    toks = text.lower().split()
    if not toks:
        return frozenset()
    if len(toks) < k:
        # A short document has one shingle per token rather than one synthetic
        # whole-document shingle. An empty document has the empty shingle set.
        return frozenset(_stable_token_hash((token,)) for token in toks)
    return frozenset(_stable_token_hash(toks[i : i + k]) for i in range(len(toks) - k + 1))


def _minhash_signatures(
    shingle_sets: Sequence[frozenset[int]],
    num_hashes: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return exact random-permutation MinHash signatures over this corpus.

    Drawing a uniform permutation of the observed shingle universe avoids the
    signed-integer overflow and limited affine-hash family of the former
    implementation. For every non-empty pair, each equality indicator is an
    unbiased Jaccard-similarity estimate over the random permutation.
    Empty sets retain the ``-1`` sentinel and are handled explicitly by the
    caller, never compared to non-empty signatures.
    """
    universe = sorted(set().union(*shingle_sets))
    signatures = np.full((len(shingle_sets), num_hashes), -1, dtype=np.int64)
    if not universe:
        return signatures

    locations = {shingle: index for index, shingle in enumerate(universe)}
    members = [np.fromiter((locations[shingle] for shingle in shingles), dtype=np.int64) for shingles in shingle_sets]
    rng = np.random.RandomState(_random_state_seed(seed))
    ranks = np.empty(len(universe), dtype=np.int64)
    for column in range(num_hashes):
        permutation = rng.permutation(len(universe))
        ranks[permutation] = np.arange(len(universe), dtype=np.int64)
        for row, indices in enumerate(members):
            if indices.size:
                signatures[row, column] = int(ranks[indices].min())
    return signatures


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
    word-``shingle_size`` shingles, each non-empty shingle set to a ``num_hashes``-entry exact
    random-permutation MinHash signature (an unbiased estimator of Jaccard similarity over the observed
    shingle universe), and two documents are called near-duplicates when their signatures agree on at
    least ``threshold`` of their entries. Two empty documents have similarity 1; an empty and non-empty
    document have similarity 0. Returns
    ``|{documents with >= 1 near-duplicate partner}| / |corpus|``. ``O(n^2)`` in the corpus size --
    fine for the receipt-sized corpora this is meant for, not a production LSH dedup pipeline.
    """
    docs = list(corpus)
    if any(not isinstance(doc, str) for doc in docs):
        raise TypeError("corpus must contain only strings")
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
    seed = _random_state_seed(seed)
    shingle_sets = [_shingles(doc, shingle_size) for doc in docs]
    sigs = _minhash_signatures(shingle_sets, num_hashes, seed=seed)
    has_dup = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if not shingle_sets[i] and not shingle_sets[j]:
                sim = 1.0
            elif not shingle_sets[i] or not shingle_sets[j]:
                sim = 0.0
            else:
                sim = float(np.mean(sigs[i] == sigs[j]))
            if sim >= threshold:
                has_dup[i] = True
                has_dup[j] = True
    return float(has_dup.sum()) / n
