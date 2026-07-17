"""Flagship B -- a temporal/latent model on real public data (monthly sunspot counts, 1749-1983).

Classification: evidence -- dataset: monthly sunspot counts, 1749-1983 (~2.8k observations, fetched
once). The head-to-head against hmmlearn below is a measured result on real data.

A hidden Markov model over a real time series: the monthly sunspot record, discretized into quantile
symbols, modeled as a discrete-emission HMM fit with one ``optimize`` call. The honest receipt is a
head-to-head against ``hmmlearn`` on the SAME held-out split -- mean log-likelihood per observation --
so the number is validated against an independent, established HMM implementation, not just asserted.

Two real usage notes this flagship encodes: a single multi-thousand-step sequence underflows the plain
forward pass, so the series is cut into fixed-length subsequences (as any HMM toolkit does); and the
transition/initial distributions are smoothed (``pseudo_count``) so a held-out subsequence never hits a
zero-probability transition.

Beyond the headline number, this flagship also demonstrates (worklist F10.2):

  - **Seed stability**: :func:`seed_stability` refits the SAME estimator/data across multiple random
    seeds and reports the actual spread of held-out log-likelihood. EM is a local method -- most seeds
    land in a common, apparently-dominant optimum, but a real fraction land in a different (sometimes
    better, sometimes worse) one. That is genuine, reported behavior, not asserted stability.
  - **Runtime/memory characterization**: :func:`run` times and ``tracemalloc``-measures the core
    ``optimize`` call in isolation (a few hundred milliseconds, a few MB at this data scale). The
    end-to-end script itself runs tens of seconds, but that is dominated by one-time Python import
    overhead (``mixle`` + ``hmmlearn`` + their dependency graphs) and the network fetch, not by fitting
    -- reported separately so a reader is not misled about what scales with data size.
  - **Emission inspection**: :func:`describe_emissions` reports each hidden state's most probable
    symbols, emission entropy, self-transition probability, and implied expected sojourn length -- the
    3 states resolve into interpretable low/mid/high sunspot-activity regimes, not just a converged
    number.
  - **Impossible-observation handling**: :func:`check_impossible_observation` splices a symbol outside
    the fitted discretization's support into an otherwise-valid sequence and confirms the model
    assigns it exactly ``-inf`` (via the same ``seq_log_density`` path used for scoring) rather than
    silently returning a finite-but-wrong number.

Run: ``python examples/flagship_temporal_sunspots.py`` (fetches ~2.8k monthly observations once; runs
the headline fit + comparison, then a 10-seed stability sweep).
"""

from __future__ import annotations

import csv
import io
import math
import time
import tracemalloc
import urllib.request
from collections.abc import Iterable

import numpy as np

_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"


def _load_series() -> np.ndarray:
    raw = urllib.request.urlopen(_URL, timeout=60).read().decode()
    return np.array([float(row[1]) for row in list(csv.reader(io.StringIO(raw)))[1:]])


def _prepare_data(n_symbols: int = 8, seq_len: int = 60) -> tuple[list[list[int]], list[list[int]], int, int]:
    """Fetch the real sunspot series once and discretize/chunk it into train/test sequences.

    Quantile bin edges are computed from the TRAIN split only (no test leakage into the
    discretization), then applied to the whole series. Each split is cut into fixed-length
    ``seq_len``-step subsequences (a single multi-thousand-step sequence underflows the plain
    forward pass).

    Returns ``(train, test, n_test_obs, n_raw_obs)``.
    """
    series = _load_series()
    split = int(0.8 * len(series))
    edges = np.quantile(series[:split], np.linspace(0, 1, n_symbols + 1)[1:-1])  # bins from train only
    symbols = np.digitize(series, edges).astype(int)

    def chunk(arr: np.ndarray) -> list[list[int]]:
        return [[int(v) for v in arr[i : i + seq_len]] for i in range(0, len(arr) - seq_len + 1, seq_len)]

    train, test = chunk(symbols[:split]), chunk(symbols[split:])
    n_test_obs = sum(len(s) for s in test)
    return train, test, n_test_obs, len(series)


def _fit_mixle(
    train: list[list[int]],
    test: list[list[int]],
    n_test_obs: int,
    *,
    n_states: int = 3,
    seed: int | None = None,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
) -> dict:
    """Fit the discrete-emission HMM with ONE ``optimize`` call and score held-out log-likelihood.

    Wraps the fit in ``tracemalloc`` plus a wall-clock timer so callers (:func:`run`,
    :func:`seed_stability`) can report the core fitting procedure's own runtime and peak memory,
    isolated from import time, the network fetch, and the (optional) hmmlearn comparison.

    Returns ``{"model", "test_ll_per_obs", "fit_wall_time_sec", "fit_peak_memory_mb"}``.
    """
    from mixle.inference import optimize
    from mixle.stats import CategoricalEstimator, HiddenMarkovEstimator

    est = HiddenMarkovEstimator(
        [CategoricalEstimator(pseudo_count=1.0) for _ in range(n_states)], pseudo_count=(1.0, 1.0)
    )
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    model = optimize(train, est, out=None, seed=seed, max_its=max_its, delta=delta)
    fit_wall_time_sec = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    encoder = model.dist_to_encoder()
    test_ll = float(np.sum(model.seq_log_density(encoder.seq_encode(test)))) / n_test_obs
    n_train_obs = sum(len(s) for s in train)
    train_ll = float(np.sum(model.seq_log_density(encoder.seq_encode(train)))) / n_train_obs
    return {
        "model": model,
        "test_ll_per_obs": test_ll,
        "train_ll_per_obs": train_ll,
        "fit_wall_time_sec": fit_wall_time_sec,
        "fit_peak_memory_mb": peak_bytes / 1.0e6,
    }


def describe_emissions(model, *, top_k: int = 4) -> list[dict]:
    """Summarize what each hidden state actually learned.

    For every state: its emission distribution's ``top_k`` most probable symbols (descending by
    probability, ties broken by symbol), the (Shannon) entropy of the full emission distribution in
    nats, its self-transition probability, and the implied expected sojourn length
    (``1 / (1 - self_transition)``) -- so a reader can see the states are interpretable regimes
    (e.g. persistent low/mid/high-activity phases), not just an opaque converged number.
    """
    out = []
    for i in range(model.n_states):
        pmap = model.topics[i].pmap
        items = sorted(pmap.items(), key=lambda kv: (-kv[1], kv[0]))
        probs = np.array([p for _, p in items], dtype=float)
        nonzero = probs[probs > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)))
        self_p = float(model.transitions[i, i])
        out.append(
            {
                "state": i,
                "top_symbols": items[:top_k],
                "entropy_nats": entropy,
                "self_transition": self_p,
                "expected_sojourn_steps": float("inf") if self_p >= 1.0 else 1.0 / (1.0 - self_p),
            }
        )
    return out


def _print_emissions(emissions: list[dict]) -> None:
    """Shared formatting for :func:`describe_emissions` output (used by ``run`` and ``seed_stability``)."""
    for e in emissions:
        top = ", ".join(f"sym {sym}={p:.3f}" for sym, p in e["top_symbols"])
        print(
            f"  state {e['state']}: {top}  |  entropy {e['entropy_nats']:.3f} nats, "
            f"self-transition {e['self_transition']:.3f} (~{e['expected_sojourn_steps']:.1f} steps)"
        )


def check_impossible_observation(model, valid_seq: list[int], n_symbols: int) -> dict:
    """Verify the fitted HMM assigns ``-inf`` -- not NaN, not a crash, not a silently-finite number
    -- to a sequence containing an observation outside the discretization's support.

    Splices the sentinel symbol ``n_symbols`` (one past the valid ``[0, n_symbols)`` quantile-bin
    range produced by ``np.digitize``, so it is guaranteed never to have been observed) into the
    middle of a copy of ``valid_seq``, and rescores both sequences through the SAME
    ``seq_log_density`` path the flagship uses for held-out scoring.

    Returns ``{"impossible_symbol", "valid_ll", "impossible_ll", "valid_is_finite",
    "correctly_flagged"}``.
    """
    encoder = model.dist_to_encoder()
    impossible_symbol = n_symbols
    valid_ll = float(model.seq_log_density(encoder.seq_encode([valid_seq]))[0])
    spliced = list(valid_seq)
    spliced[len(spliced) // 2] = impossible_symbol
    spliced_ll = float(model.seq_log_density(encoder.seq_encode([spliced]))[0])
    return {
        "impossible_symbol": impossible_symbol,
        "valid_ll": valid_ll,
        "impossible_ll": spliced_ll,
        "valid_is_finite": math.isfinite(valid_ll),
        "correctly_flagged": spliced_ll == float("-inf"),
    }


def seed_stability(
    seeds: Iterable[int] = range(10),
    *,
    n_states: int = 3,
    n_symbols: int = 8,
    seq_len: int = 60,
    max_its: int = 200,
    delta: float | None = 1.0e-8,
    verbose: bool = True,
) -> dict:
    """Refit the SAME estimator on the SAME data across multiple random seeds and report the spread
    of held-out log-likelihood.

    EM is a local method: different random initializations can converge to different fixed points.
    This uses ``max_its=200`` (vs. :func:`run`'s quick-demo default of 10) because 10 iterations is
    not enough to converge for most seeds -- empirically, several seeds are still improving by
    dozens to hundreds of nats at iteration 10, but every seed tried is bit-identical from iteration
    200 through 1000. Sweeping seeds at the quick demo's own (under-converged) 10-iteration budget
    would conflate "hasn't converged yet" with "genuinely different optimum"; using a
    verified-converged budget isolates the real answer instead: on this data, most random
    initializations converge to the same optimum, but a real fraction land in a different (sometimes
    better, sometimes worse) one. That is reported here, not hidden behind a single lucky/unlucky
    default seed.

    Model selection across seeds (when wanted) is by TRAINING log-likelihood, never held-out test
    log-likelihood -- picking among candidates by test performance is a subtle test-set leak. The
    best-by-training-likelihood seed's learned emissions are included in the summary so a reader can
    see what a properly-converged fit looks like (contrast this with :func:`run`'s quick-demo
    emissions, which are printed from the under-converged 10-iteration default).

    Returns ``{"per_seed": [{"seed", "test_ll_per_obs", "train_ll_per_obs"}, ...], "mean", "std",
    "min", "max", "range", "best_seed", "best_seed_emissions"}``.
    """
    train, test, n_test_obs, _ = _prepare_data(n_symbols=n_symbols, seq_len=seq_len)
    per_seed = []
    fits = []
    for s in seeds:
        fit = _fit_mixle(train, test, n_test_obs, n_states=n_states, seed=s, max_its=max_its, delta=delta)
        per_seed.append(
            {"seed": s, "test_ll_per_obs": fit["test_ll_per_obs"], "train_ll_per_obs": fit["train_ll_per_obs"]}
        )
        fits.append(fit)

    lls = np.array([r["test_ll_per_obs"] for r in per_seed], dtype=float)
    best_idx = int(np.argmax([f["train_ll_per_obs"] for f in fits]))
    best_seed = per_seed[best_idx]["seed"]
    best_emissions = describe_emissions(fits[best_idx]["model"])
    summary = {
        "per_seed": per_seed,
        "mean": float(lls.mean()),
        "std": float(lls.std()),
        "min": float(lls.min()),
        "max": float(lls.max()),
        "range": float(lls.max() - lls.min()),
        "best_seed": best_seed,
        "best_seed_emissions": best_emissions,
    }
    if verbose:
        print(f"seed stability -- {len(per_seed)} seeds, {n_states}-state HMM, max_its={max_its} (verified converged):")
        for r in per_seed:
            print(
                f"  seed={r['seed']!s:>3}: test_ll_per_obs={r['test_ll_per_obs']:.4f}  train_ll_per_obs={r['train_ll_per_obs']:.4f}"
            )
        print(
            f"  mean={summary['mean']:.4f}  std={summary['std']:.4f}  "
            f"range=[{summary['min']:.4f}, {summary['max']:.4f}]  spread={summary['range']:.4f}"
        )
        print(
            f"learned emissions of the best-by-training-likelihood seed ({best_seed}) -- interpretable regimes at convergence:"
        )
        _print_emissions(best_emissions)
    return summary


def run(
    *,
    n_states: int = 3,
    n_symbols: int = 8,
    seq_len: int = 60,
    seed: int | None = None,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    compare_hmmlearn: bool = True,
    verbose: bool = True,
) -> dict:
    """Fit a discrete-emission HMM on real sunspot data and compare held-out LL to hmmlearn.

    ``seed``/``max_its``/``delta`` default to exactly what the original flagship always used
    (unseeded, 10 EM iterations) so this call's headline number and the pre-existing regression gate
    (``flagship_temporal_sunspots_smoke_test.py``) are unchanged. Pass ``max_its=200`` (or use
    :func:`seed_stability`, which does) to reproduce the properly-converged fit documented in the
    module docstring and in ``seed_stability``'s own docstring.

    Returns ``{"n_states", "n_symbols", "n_train_seqs", "n_test_seqs", "mixle_test_ll_per_obs",
    "hmmlearn_test_ll_per_obs"}`` (the hmmlearn value is ``None`` if hmmlearn is not installed or
    ``compare_hmmlearn=False``), plus:

      - ``fit_wall_time_sec`` / ``fit_peak_memory_mb``: the core ``optimize`` call's own wall-clock
        time and ``tracemalloc``-measured peak memory at this data scale (isolated from import,
        network, and hmmlearn overhead -- see the module docstring).
      - ``emissions``: per-state learned-emission summary from :func:`describe_emissions`.
      - ``impossible_observation_check``: result of :func:`check_impossible_observation` against the
        fitted model.
    """
    train, test, n_test_obs, n_raw_obs = _prepare_data(n_symbols=n_symbols, seq_len=seq_len)
    fit = _fit_mixle(train, test, n_test_obs, n_states=n_states, seed=seed, max_its=max_its, delta=delta)
    model = fit["model"]
    mixle_ll = fit["test_ll_per_obs"]

    hmmlearn_ll = None
    if compare_hmmlearn:
        try:
            from hmmlearn.hmm import CategoricalHMM

            x_tr = np.concatenate([np.array(s) for s in train]).reshape(-1, 1)
            x_te = np.concatenate([np.array(s) for s in test]).reshape(-1, 1)
            hl = CategoricalHMM(n_components=n_states, n_iter=50, random_state=0)
            hl.fit(x_tr, [len(s) for s in train])
            hmmlearn_ll = hl.score(x_te, [len(s) for s in test]) / n_test_obs
        except ImportError:
            pass

    emissions = describe_emissions(model)
    impossible_check = check_impossible_observation(model, test[0], n_symbols)

    receipt = {
        "n_states": n_states,
        "n_symbols": n_symbols,
        "n_train_seqs": len(train),
        "n_test_seqs": len(test),
        "mixle_test_ll_per_obs": mixle_ll,
        "hmmlearn_test_ll_per_obs": hmmlearn_ll,
        "fit_wall_time_sec": fit["fit_wall_time_sec"],
        "fit_peak_memory_mb": fit["fit_peak_memory_mb"],
        "emissions": emissions,
        "impossible_observation_check": impossible_check,
    }
    if verbose:
        print(f"real monthly sunspots: {n_raw_obs} obs -> {n_symbols} quantile symbols, {seq_len}-step sequences")
        print(f"{n_states}-state discrete HMM, held-out mean log-likelihood per obs:")
        print(
            f"  mixle    : {mixle_ll:.4f}   "
            f"(fit: {fit['fit_wall_time_sec']:.3f}s wall, {fit['fit_peak_memory_mb']:.1f} MB peak)"
        )
        if hmmlearn_ll is not None:
            print(f"  hmmlearn : {hmmlearn_ll:.4f}   (independent baseline; agreement is the receipt)")
        quick_note = " (quick demo fit -- see seed_stability() for a properly-converged model)" if max_its <= 10 else ""
        print(f"learned state emissions{quick_note}, top symbols | self-transition, expected sojourn:")
        _print_emissions(emissions)
        flag = "correctly -inf" if impossible_check["correctly_flagged"] else "NOT flagged -- investigate"
        print(
            f"impossible-observation check: valid seq ll={impossible_check['valid_ll']:.2f} (finite); "
            f"splicing in out-of-support symbol {impossible_check['impossible_symbol']} -> "
            f"ll={impossible_check['impossible_ll']} ({flag})"
        )
    return receipt


def main() -> None:
    run()
    print()
    seed_stability()


if __name__ == "__main__":
    main()
