"""Flagship B -- a temporal/latent model on real public data (monthly sunspot counts, 1749-1983).

A hidden Markov model over a real time series: the monthly sunspot record, discretized into quantile
symbols, modeled as a discrete-emission HMM fit with one ``optimize`` call. The honest receipt is a
head-to-head against ``hmmlearn`` on the SAME held-out split -- mean log-likelihood per observation --
so the number is validated against an independent, established HMM implementation, not just asserted.

Two real usage notes this flagship encodes: a single multi-thousand-step sequence underflows the plain
forward pass, so the series is cut into fixed-length subsequences (as any HMM toolkit does); and the
transition/initial distributions are smoothed (``pseudo_count``) so a held-out subsequence never hits a
zero-probability transition.

Run: ``python examples/flagship_temporal_sunspots.py`` (fetches ~2.8k monthly observations once).
"""

from __future__ import annotations

import csv
import io
import urllib.request

import numpy as np

_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"


def _load_series() -> np.ndarray:
    raw = urllib.request.urlopen(_URL, timeout=60).read().decode()
    return np.array([float(row[1]) for row in list(csv.reader(io.StringIO(raw)))[1:]])


def run(*, n_states: int = 3, n_symbols: int = 8, seq_len: int = 60, verbose: bool = True) -> dict:
    """Fit a discrete-emission HMM on real sunspot data and compare held-out LL to hmmlearn.

    Returns ``{"n_states", "n_symbols", "n_train_seqs", "n_test_seqs", "mixle_test_ll_per_obs",
    "hmmlearn_test_ll_per_obs"}`` (the hmmlearn value is ``None`` if hmmlearn is not installed).
    """
    from mixle.inference import optimize
    from mixle.stats import CategoricalEstimator, HiddenMarkovEstimator

    series = _load_series()
    split = int(0.8 * len(series))
    edges = np.quantile(series[:split], np.linspace(0, 1, n_symbols + 1)[1:-1])  # bins from train only
    symbols = np.digitize(series, edges).astype(int)

    def chunk(arr: np.ndarray) -> list[list[int]]:
        return [[int(v) for v in arr[i : i + seq_len]] for i in range(0, len(arr) - seq_len + 1, seq_len)]

    train, test = chunk(symbols[:split]), chunk(symbols[split:])
    n_test_obs = sum(len(s) for s in test)

    est = HiddenMarkovEstimator(
        [CategoricalEstimator(pseudo_count=1.0) for _ in range(n_states)], pseudo_count=(1.0, 1.0)
    )
    model = optimize(train, est, out=None)
    mixle_ll = float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(test)))) / n_test_obs

    hmmlearn_ll = None
    try:
        from hmmlearn.hmm import CategoricalHMM

        x_tr = np.concatenate([np.array(s) for s in train]).reshape(-1, 1)
        x_te = np.concatenate([np.array(s) for s in test]).reshape(-1, 1)
        hl = CategoricalHMM(n_components=n_states, n_iter=50, random_state=0)
        hl.fit(x_tr, [len(s) for s in train])
        hmmlearn_ll = hl.score(x_te, [len(s) for s in test]) / n_test_obs
    except ImportError:
        pass

    receipt = {
        "n_states": n_states,
        "n_symbols": n_symbols,
        "n_train_seqs": len(train),
        "n_test_seqs": len(test),
        "mixle_test_ll_per_obs": mixle_ll,
        "hmmlearn_test_ll_per_obs": hmmlearn_ll,
    }
    if verbose:
        print(f"real monthly sunspots: {len(series)} obs -> {n_symbols} quantile symbols, {seq_len}-step sequences")
        print(f"{n_states}-state discrete HMM, held-out mean log-likelihood per obs:")
        print(f"  mixle    : {mixle_ll:.4f}")
        if hmmlearn_ll is not None:
            print(f"  hmmlearn : {hmmlearn_ll:.4f}   (independent baseline; agreement is the receipt)")
    return receipt


def main() -> None:
    run()


if __name__ == "__main__":
    main()
