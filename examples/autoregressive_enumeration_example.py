"""Exact count / rank / unrank over an autoregressive model's sequence space, without enumerating it.

:class:`mixle.enumeration.AutoregressiveEnumerable` wraps a plain ``next_logprobs(prefix) -> [(token,
log_prob), ...]`` callable -- the next-token log-probabilities given a prefix, e.g. the log_softmax of
a transformer's logits -- and answers count-/rank-/quantile-style queries over the exponentially large
space of complete sequences it defines: "how many sequences are at least this probable" (``count``),
"what is the log-probability of the k-th most probable sequence" (``threshold``), "what is the i-th
most probable sequence" (``unrank``), all *without* ever materializing or sorting the full sequence
space. The trick is that the number of model queries is bounded by the number of *distinct prefixes*
the recursion actually visits, not by the sequence count or the requested rank.

This example uses a small synthetic prefix-dependent logit table -- a fixed random tensor indexed by
(depth, previous token) -- to keep it fast and dependency-free (no torch, no network, no download). A
real language model plugs into the exact same ``next_logprobs`` interface; this demonstrates the
mechanism, not a specific model integration (see the README's flagship examples for a real pretrained
model wired into mixle).

Run: python examples/autoregressive_enumeration_example.py
"""

from __future__ import annotations

import numpy as np

from mixle.enumeration import AutoregressiveEnumerable


class CountingModel:
    """Wraps a ``next_logprobs`` callable and records every distinct prefix it is asked to score.

    ``AutoregressiveEnumerable`` memoizes by prefix internally, so this wrapper's ``queried`` set ends
    up identical to its internal cache -- it is here purely to make the "how many forward passes did
    this actually take" count visible from the outside, as a black box.
    """

    def __init__(self, next_logprobs):
        self._next_logprobs = next_logprobs
        self.queried: set[tuple] = set()

    def __call__(self, prefix: tuple):
        self.queried.add(prefix)
        return self._next_logprobs(prefix)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    return logits - (m + np.log(np.sum(np.exp(logits - m))))


def make_synthetic_model(vocab_size: int, length: int, seed: int = 0, scale: float = 1.5):
    """A small genuinely prefix-dependent model: ``table[depth, last_token]`` -> next-token logits.

    Every position's distribution depends on the previous token through a per-depth transition table,
    so consecutive tokens are actually coupled -- "autoregressive" is a demonstrated property here, not
    just a label. In a real model this table is a transformer's next-token logits computed on the fly;
    here it is a fixed random tensor, so the whole example needs nothing beyond numpy.
    """
    rng = np.random.RandomState(seed)
    table = rng.randn(length, vocab_size, vocab_size) * scale

    def next_logprobs(prefix: tuple):
        depth = len(prefix)
        last = prefix[-1] if prefix else 0
        return list(enumerate(_log_softmax(table[depth, last])))

    return next_logprobs


def make_terminating_model():
    """A 2-symbol-body + EOS model: transition probabilities that let the sequence keep extending.

    With ``eos`` (rather than ``max_len``) the model's support is every EOS-terminated sequence of ANY
    length, bounded by probability budget rather than a length cap -- this is what "terminating" means.
    """
    eos = 2
    # p(next token | last token); None is the start-of-sequence state. Token 2 is EOS everywhere.
    transition = {
        None: [0.35, 0.35, 0.30],
        0: [0.30, 0.30, 0.40],
        1: [0.40, 0.20, 0.40],
    }

    def next_logprobs(prefix: tuple):
        last = prefix[-1] if prefix else None
        return list(enumerate(np.log(np.array(transition[last]))))

    return next_logprobs, eos


def main():
    vocab_size, length = 5, 3
    full_space = vocab_size**length
    prefix_bound = sum(vocab_size**d for d in range(length))  # 1 + V + V^2 (all depths < length)
    print(f"== fixed-length model: vocab_size={vocab_size}, length={length}, full sequence space={full_space} ==\n")
    print(
        f"Tracking every distinct prefix next_logprobs is actually called on (max possible: {prefix_bound}, "
        f"i.e. 1 + V + ... + V^(L-1); brute-force scoring all sequences would need {full_space} model calls).\n"
    )

    counted = CountingModel(make_synthetic_model(vocab_size, length, seed=0))
    ar = AutoregressiveEnumerable(counted, max_len=length, oversample=64)

    print("1) top-5 most probable length-3 sequences (exact, descending, best-first):")
    for seq, log_p in ar.top_k(5):
        print(f"     {seq}   p={np.exp(log_p):.5f}  log_p={log_p:.4f}")
    print(
        f"     -> prefixes queried so far: {len(counted.queried)} of {prefix_bound}  (best-first only expands the head)"
    )

    print("\n2) random-access unrank + rank round-trip (no listing up to the index):")
    query_index = 42
    seq, log_p = ar.unrank(query_index)
    back = ar.rank(seq)
    print(f"     unrank({query_index}) -> {seq}   log_p={log_p:.4f}")
    print(f"     rank({seq}) -> {back.rank}   (round-trips to {query_index}: {back.rank == query_index})")
    print(
        f"     -> prefixes queried so far: {len(counted.queried)} of {prefix_bound}  (unrank built the full seek index once)"
    )

    print("\n3) threshold <-> count consistency (count is computed from histograms, never listed):")
    k = 20
    tau = ar.threshold(k)
    c = ar.count(tau)
    print(f"     threshold({k}) -> log_p={tau:.4f}  (the {k}-th most probable sequence's log-prob)")
    print(f"     count(min_log_prob={tau:.4f}) -> {c}   (matches k={k}: {c == k})")
    n_queried = len(counted.queried)
    print(
        f"     -> prefixes queried so far: {n_queried} of {prefix_bound}  (unchanged: threshold/count reused unrank's cached index)"
    )

    print(
        f"\n4) the actual point: {n_queried} total model queries -- across top-5, an unrank/rank round-trip at "
        f"index {query_index}, and a threshold+count pair -- versus {full_space} sequences a brute-force sort "
        f"would have to score and hold in memory. The index is built once (by whichever query needs it first) "
        f"and every later count-/rank-/unrank-style query reuses it for free."
    )

    print("\n== terminating model: EOS-bounded, so support is every-length sequence, not a fixed length ==\n")
    term_next_logprobs, eos = make_terminating_model()
    ar_term = AutoregressiveEnumerable(term_next_logprobs, eos=eos, oversample=64, max_depth=40)

    print("5) top-5 most probable complete (EOS-terminated) sequences, of varying length:")
    top5_term = ar_term.top_k(5)
    for seq, log_p in top5_term:
        print(f"     {seq}   p={np.exp(log_p):.5f}  log_p={log_p:.4f}")

    print("\n6) rank + cumulative probability: locate a sequence in the descending-probability order,")
    print("   without listing up to it (the quantile-style query for a terminating, variable-length model):")
    target, _target_log_p = top5_term[-1]
    r = ar_term.rank(target)
    cum = ar_term.cumulative(target)
    print(f"     rank({target}) -> {r.rank}  (0-based count of strictly-more-probable sequences; exact={r.exact})")
    print(
        f"     cumulative({target}) -> {cum:.4f}  (mass of every sequence at least as probable as this one; "
        f"agrees with rank()'s own cumulative_probability: {abs(cum - r.cumulative_probability) < 1e-9})"
    )


if __name__ == "__main__":
    main()
